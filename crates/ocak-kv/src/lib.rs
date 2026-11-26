//! Fixed-page (16-token) KV cache allocator with Copy-on-Write Radix prefix tree.
use rustc_hash::FxHashMap;
use serde::Serialize;

pub const PAGE_TOKENS: usize = 16;
pub type PageId = u32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMatch {
    /// Always a multiple of `PAGE_TOKENS`.
    pub matched_tokens: usize,
    /// Physical pages (already incref'd for the caller).
    pub pages: Vec<PageId>,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct PagerStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub allocs: u64,
}

#[derive(Debug, Clone)]
struct PageMeta {
    refcount: u32,
    node_id: Option<usize>,
    last_access: u64,
}

#[derive(Debug, Clone)]
struct Node {
    parent: Option<usize>,
    key: [u32; PAGE_TOKENS],
    page: PageId,
    children: FxHashMap<[u32; PAGE_TOKENS], usize>,
    alive: bool,
}

#[derive(Debug)]
pub struct RadixPager {
    pages: Vec<PageMeta>,
    free_list: Vec<PageId>,
    nodes: Vec<Node>,
    free_nodes: Vec<usize>,
    roots: FxHashMap<[u32; PAGE_TOKENS], usize>,
    clock: u64,
    stats: PagerStats,
}

impl RadixPager {
    pub fn new(n_pages: usize) -> Self {
        let pages = (0..n_pages)
            .map(|_| PageMeta { refcount: 0, node_id: None, last_access: 0 })
            .collect();
        let free_list = (0..n_pages as PageId).rev().collect();
        Self {
            pages,
            free_list,
            nodes: Vec::new(),
            free_nodes: Vec::new(),
            roots: FxHashMap::default(),
            clock: 0,
            stats: PagerStats::default(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.pages.len()
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    pub fn match_prefix(&mut self, tokens: &[u32]) -> PrefixMatch {
        let mut matched = Vec::new();
        let mut cur: Option<usize> = None;
        for chunk in tokens.chunks_exact(PAGE_TOKENS) {
            let key: [u32; PAGE_TOKENS] = chunk.try_into().unwrap();
            let next = match cur {
                None => self.roots.get(&key).copied(),
                Some(nid) => self.nodes[nid].children.get(&key).copied(),
            };
            match next {
                Some(nid) => {
                    let pid = self.nodes[nid].page;
                    let t = self.tick();
                    let pm = &mut self.pages[pid as usize];
                    pm.refcount += 1;
                    pm.last_access = t;
                    matched.push(pid);
                    cur = Some(nid);
                }
                None => {
                    self.stats.misses += 1;
                    break;
                }
            }
        }
        self.stats.hits += matched.len() as u64;
        PrefixMatch { matched_tokens: matched.len() * PAGE_TOKENS, pages: matched }
    }

    pub fn alloc_page(&mut self) -> Option<PageId> {
        if let Some(pid) = self.free_list.pop() {
            let t = self.tick();
            let pm = &mut self.pages[pid as usize];
            debug_assert_eq!(pm.refcount, 0);
            debug_assert!(pm.node_id.is_none());
            pm.refcount = 1;
            pm.last_access = t;
            self.stats.allocs += 1;
            return Some(pid);
        }
        // Evict LRU zero-refcount leaf node.
        let mut victim: Option<(usize, PageId, u64)> = None;
        for (nid, node) in self.nodes.iter().enumerate() {
            if !node.alive || !node.children.is_empty() {
                continue;
            }
            let pm = &self.pages[node.page as usize];
            if pm.refcount == 0 {
                if victim.map_or(true, |(_, _, best_t)| pm.last_access < best_t) {
                    victim = Some((nid, node.page, pm.last_access));
                }
            }
        }
        let (nid, pid, _) = victim?;
        self.unlink_leaf(nid);
        let t = self.tick();
        let pm = &mut self.pages[pid as usize];
        pm.node_id = None;
        pm.refcount = 1;
        pm.last_access = t;
        self.stats.evictions += 1;
        self.stats.allocs += 1;
        Some(pid)
    }

    fn unlink_leaf(&mut self, nid: usize) {
        let (parent, key) = {
            let n = &mut self.nodes[nid];
            debug_assert!(n.children.is_empty());
            n.alive = false;
            n.children.clear();
            (n.parent, n.key)
        };
        match parent {
            None => {
                self.roots.remove(&key);
            }
            Some(p) => {
                self.nodes[p].children.remove(&key);
            }
        }
        self.free_nodes.push(nid);
    }

    pub fn insert_prefix(&mut self, tokens: &[u32], pages: &[PageId]) {
        let full = (tokens.len() / PAGE_TOKENS).min(pages.len());
        let mut cur: Option<usize> = None;
        for i in 0..full {
            let key: [u32; PAGE_TOKENS] = tokens[i * PAGE_TOKENS..(i + 1) * PAGE_TOKENS].try_into().unwrap();
            let existing = match cur {
                None => self.roots.get(&key).copied(),
                Some(nid) => self.nodes[nid].children.get(&key).copied(),
            };
            if let Some(nid) = existing {
                let t = self.tick();
                let pid = self.nodes[nid].page;
                self.pages[pid as usize].last_access = t;
                cur = Some(nid);
                continue;
            }
            let pid = pages[i];
            if self.pages[pid as usize].node_id.is_some() {
                // Already bound elsewhere in the tree; stop extending.
                break;
            }
            let node = Node {
                parent: cur,
                key,
                page: pid,
                children: FxHashMap::default(),
                alive: true,
            };
            let new_nid = if let Some(slot) = self.free_nodes.pop() {
                self.nodes[slot] = node;
                slot
            } else {
                let idx = self.nodes.len();
                self.nodes.push(node);
                idx
            };
            match cur {
                None => {
                    self.roots.insert(key, new_nid);
                }
                Some(p) => {
                    self.nodes[p].children.insert(key, new_nid);
                }
            }
            let t = self.tick();
            let pm = &mut self.pages[pid as usize];
            pm.node_id = Some(new_nid);
            pm.last_access = t;
            cur = Some(new_nid);
        }
    }

    pub fn release(&mut self, pages: &[PageId]) {
        for &pid in pages {
            let pm = &mut self.pages[pid as usize];
            if pm.refcount > 0 {
                pm.refcount -= 1;
                if pm.refcount == 0 && pm.node_id.is_none() {
                    self.free_list.push(pid);
                }
            }
        }
    }

    pub fn free_pages(&self) -> usize {
        self.free_list.len()
    }

    pub fn cached_pages(&self) -> usize {
        self.nodes.iter().filter(|n| n.alive).count()
    }

    pub fn node_slots(&self) -> usize {
        self.nodes.len()
    }

    pub fn stats(&self) -> PagerStats {
        self.stats
    }

    pub fn export_cached(&self) -> Vec<(Vec<u32>, PageId)> {
        let mut out = Vec::new();
        let mut path = Vec::new();
        for &nid in self.roots.values() {
            self.dfs_export(nid, &mut path, &mut out);
        }
        out
    }

    fn dfs_export(&self, nid: usize, path: &mut Vec<u32>, out: &mut Vec<(Vec<u32>, PageId)>) {
        let n = &self.nodes[nid];
        if !n.alive {
            return;
        }
        path.extend_from_slice(&n.key);
        out.push((path.clone(), n.page));
        for &child in n.children.values() {
            self.dfs_export(child, path, out);
        }
        path.truncate(path.len() - PAGE_TOKENS);
    }

    pub fn clear(&mut self) {
        let n = self.pages.len();
        *self = Self::new(n);
    }
}

/// Host-RAM Q8_0 compressed KV page store for `WarmSleep` teardown and long-context spill.
/// Compresses each 32-float block into `[f32 scale (4B) + 32 x i8 (32B)]` = 36B (3.55x reduction).
#[derive(Default)]
pub struct CompressedKvStore {
    pages: FxHashMap<Vec<u32>, Vec<u8>>,
    max_bytes: usize,
    used_bytes: usize,
}

impl CompressedKvStore {
    pub fn new(max_bytes: usize) -> Self {
        Self { pages: FxHashMap::default(), max_bytes, used_bytes: 0 }
    }

    pub fn compress_f32(src: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity((src.len() / 32) * 36);
        for blk in src.chunks_exact(32) {
            let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = amax / 127.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            out.extend_from_slice(&d.to_le_bytes());
            for &v in blk {
                out.push((v * inv).round().clamp(-127.0, 127.0) as i8 as u8);
            }
        }
        out
    }

    pub fn decompress_f32(src: &[u8], dst: &mut [f32]) {
        for (blk, out) in src.chunks_exact(36).zip(dst.chunks_exact_mut(32)) {
            let d = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
            for i in 0..32 {
                out[i] = (blk[4 + i] as i8) as f32 * d;
            }
        }
    }

    pub fn put(&mut self, prefix_key: Vec<u32>, kv_f32: &[f32]) -> bool {
        let comp = Self::compress_f32(kv_f32);
        if comp.len() > self.max_bytes {
            return false;
        }
        if let Some(old) = self.pages.remove(&prefix_key) {
            self.used_bytes -= old.len();
        }
        while self.used_bytes + comp.len() > self.max_bytes {
            if let Some(k) = self.pages.keys().next().cloned() {
                if let Some(v) = self.pages.remove(&k) {
                    self.used_bytes -= v.len();
                }
            } else {
                break;
            }
        }
        self.used_bytes += comp.len();
        self.pages.insert(prefix_key, comp);
        true
    }

    pub fn get(&self, prefix_key: &[u32], dst: &mut [f32]) -> bool {
        if let Some(comp) = self.pages.get(prefix_key) {
            Self::decompress_f32(comp, dst);
            true
        } else {
            false
        }
    }

    pub fn used_bytes(&self) -> usize { self.used_bytes }
    pub fn len(&self) -> usize { self.pages.len() }
    pub fn is_empty(&self) -> bool { self.pages.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(base: u32, len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| base + i).collect()
    }

    #[test]
    fn prefix_sharing_and_lru_eviction() {
        let mut p = RadixPager::new(4);
        let t1 = seq(100, 32); // 2 full pages
        let p0 = p.alloc_page().unwrap();
        let p1 = p.alloc_page().unwrap();
        p.insert_prefix(&t1, &[p0, p1]);
        p.release(&[p0, p1]);
        assert_eq!(p.cached_pages(), 2);
        assert_eq!(p.free_pages(), 2);

        // Second sequence shares first 16 tokens, diverges on second 16
        let mut t2 = t1[..16].to_vec();
        t2.extend(seq(500, 16));
        let m = p.match_prefix(&t2);
        assert_eq!(m.matched_tokens, 16);
        assert_eq!(m.pages, vec![p0]);
        let p2 = p.alloc_page().unwrap();
        p.insert_prefix(&t2, &[p0, p2]);
        p.release(&[p0, p2]);
        assert_eq!(p.cached_pages(), 3);

        // Allocate remaining free page + trigger LRU eviction of leaf p1
        let p3 = p.alloc_page().unwrap();
        let p4 = p.alloc_page().unwrap();
        assert_eq!(p4, p1);
        assert_eq!(p.stats().evictions, 1);
        p.release(&[p3, p4]);
    }

    #[test]
    fn exhaustion_returns_none_when_all_pinned() {
        let mut p = RadixPager::new(2);
        let a = p.alloc_page().unwrap();
        let b = p.alloc_page().unwrap();
        assert!(p.alloc_page().is_none());
        p.release(&[a]);
        assert_eq!(p.alloc_page(), Some(a));
        p.release(&[a, b]);
        assert_eq!(p.free_pages(), 2);
    }

    #[test]
    fn export_cached_paths() {
        let mut p = RadixPager::new(4);
        let t = seq(1, 32);
        let a = p.alloc_page().unwrap();
        let b = p.alloc_page().unwrap();
        p.insert_prefix(&t, &[a, b]);
        let ex = p.export_cached();
        assert_eq!(ex.len(), 2);
        assert!(ex.iter().any(|(path, pid)| path == &t[..16] && *pid == a));
        assert!(ex.iter().any(|(path, pid)| path == &t[..32] && *pid == b));
        p.release(&[a, b]);
    }

    #[test]
    fn compressed_kv_q8_roundtrip_and_budget() {
        let kv: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.13).sin() * 2.5).collect();
        let mut store = CompressedKvStore::new(4096);
        assert!(store.put(vec![1, 2, 3], &kv));
        assert_eq!(store.used_bytes(), (1024 / 32) * 36);
        let mut out = vec![0f32; 1024];
        assert!(store.get(&[1, 2, 3], &mut out));
        let max_err = kv.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max_err < 0.025, "max_err={max_err}");
    }

    #[test]
    fn radix_tree_reclaims_dead_nodes_bounded() {
        let mut p = RadixPager::new(8);
        for round in 0u32..500 {
            let toks = seq(round * 100 + 1, 32);
            let a = p.alloc_page().unwrap();
            let b = p.alloc_page().unwrap();
            p.insert_prefix(&toks, &[a, b]);
            p.release(&[a, b]);
        }
        assert!(p.node_slots() <= 8, "node_slots leaked: {}", p.node_slots());
        assert!(p.stats().evictions > 900);
    }
}
