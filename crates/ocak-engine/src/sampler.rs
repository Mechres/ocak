#[derive(Debug, Clone)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self { temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.0, repeat_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0, repeat_last_n: 64, seed: 0 }
    }
}

impl SamplerParams {
    pub fn greedy() -> Self {
        Self { temperature: 0.0, ..Default::default() }
    }
}

pub struct Sampler {
    p: SamplerParams,
    state: u64,
    recent: Vec<u32>,
}

impl Sampler {
    pub fn new(p: SamplerParams) -> Self {
        // seed 0 = random, unless OCAK_SEED fixes it (reproducible sampled runs for testing).
        let env_seed = std::env::var("OCAK_SEED").ok().and_then(|v| v.parse::<u64>().ok()).filter(|&v| v != 0);
        let seed = if let (0, Some(es)) = (p.seed, env_seed) {
            es
        } else if p.seed == 0 {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
        } else {
            p.seed
        };
        Self { p, state: seed, recent: Vec::new() }
    }

    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn accept(&mut self, tok: u32) {
        self.recent.push(tok);
        if self.recent.len() > self.p.repeat_last_n.max(1) * 4 {
            let drop = self.recent.len() - self.p.repeat_last_n.max(1);
            self.recent.drain(..drop);
        }
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        let p = self.p.clone();
        if p.repeat_last_n > 0 && (p.repeat_penalty != 1.0 || p.presence_penalty != 0.0 || p.frequency_penalty != 0.0) {
            let start = self.recent.len().saturating_sub(p.repeat_last_n);
            let mut counts: std::collections::HashMap<u32, u32> = Default::default();
            for &t in &self.recent[start..] {
                *counts.entry(t).or_default() += 1;
            }
            for (t, c) in counts {
                let l = &mut logits[t as usize];
                if l.is_finite() {
                    if *l > 0.0 { *l /= p.repeat_penalty } else { *l *= p.repeat_penalty }
                    *l -= c as f32 * p.frequency_penalty + p.presence_penalty;
                }
            }
        }
        if p.temperature <= 0.0 {
            return logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
        }
        let cmp = |a: &(u32, f32), b: &(u32, f32)| b.1.partial_cmp(&a.1).unwrap();
        if p.top_k > 0 && p.top_k <= 1024 {
            // Small top-k: one streaming pass keeping the k largest (no copy of the whole
            // vocabulary, no full partition).
            let mut cand = top_k_scan(logits, p.top_k);
            if cand.is_empty() {
                return 0;
            }
            cand.sort_by(cmp);
            return self.sample_sorted(&cand, &p);
        }
        let mut cand: Vec<(u32, f32)> = logits.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, &v)| (i as u32, v)).collect();
        if cand.is_empty() {
            return 0;
        }
        if p.top_k == 0 || p.top_k >= cand.len() {
            // No top-k truncation: avoid sorting the whole vocabulary (8 ms for 248k tokens).
            let mx = cand.iter().fold(f32::NEG_INFINITY, |a, c| a.max(c.1));
            let inv_t = 1.0 / p.temperature;
            if p.top_p >= 1.0 && p.min_p <= 0.0 {
                // Plain temperature sampling: no ordering needed; the weights are computed once
                // and reused for the draw (same values and summation order as before).
                let w: Vec<f32> = cand.iter().map(|c| ((c.1 - mx) * inv_t).exp()).collect();
                let sum_all: f32 = w.iter().sum();
                let mut r = self.next_f32() * sum_all;
                for (c, &wi) in cand.iter().zip(&w) {
                    r -= wi;
                    if r <= 0.0 {
                        return c.0;
                    }
                }
                return cand[cand.len() - 1].0;
            }
            let sum_all: f32 = cand.iter().map(|c| ((c.1 - mx) * inv_t).exp()).sum();
            // Nucleus / min-p: sort only a top-M prefix, growing M until it covers the cut.
            let mut m = 256usize.min(cand.len());
            loop {
                if m < cand.len() {
                    cand.select_nth_unstable_by(m - 1, cmp);
                }
                cand[..m].sort_by(cmp);
                let w0 = 1.0f32; // exp(0): the top candidate
                let thr = w0 * p.min_p;
                let mut cum = 0.0f32;
                let mut covered = m == cand.len();
                for c in &cand[..m] {
                    let w = ((c.1 - mx) * inv_t).exp();
                    if p.min_p > 0.0 && w < thr {
                        covered = true;
                        break;
                    }
                    cum += w / sum_all;
                    if p.top_p < 1.0 && cum >= p.top_p {
                        covered = true;
                        break;
                    }
                }
                if covered {
                    break;
                }
                m = (m * 4).min(cand.len());
            }
            cand.truncate(m);
        } else {
            cand.select_nth_unstable_by(p.top_k - 1, cmp);
            cand.truncate(p.top_k);
            cand.sort_by(cmp);
        }
        self.sample_sorted(&cand, &p)
    }

    /// Draw from candidates sorted by descending logit, applying temperature, min_p and top_p.
    fn sample_sorted(&mut self, cand: &[(u32, f32)], p: &SamplerParams) -> u32 {
        let mx = cand[0].1;
        let mut probs: Vec<f32> = cand.iter().map(|c| ((c.1 - mx) / p.temperature).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|v| *v /= sum);
        let mut keep = probs.len();
        if p.min_p > 0.0 {
            let thr = probs[0] * p.min_p;
            keep = probs.iter().position(|&v| v < thr).unwrap_or(keep).max(1);
        }
        if p.top_p < 1.0 {
            let mut cum = 0.0;
            let mut n = 0;
            for &v in &probs[..keep] {
                cum += v;
                n += 1;
                if cum >= p.top_p { break; }
            }
            keep = n.max(1);
        }
        let total: f32 = probs[..keep].iter().sum();
        let mut r = self.next_f32() * total;
        for i in 0..keep {
            r -= probs[i];
            if r <= 0.0 {
                return cand[i].0;
            }
        }
        cand[keep - 1].0
    }
}

/// The k largest finite logits as (id, logit), unordered, in one pass over the vocabulary.
fn top_k_scan(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
    let (mut min_i, mut min_v) = (0usize, f32::INFINITY);
    let refresh = |top: &[(u32, f32)]| top.iter().enumerate().fold((0usize, f32::INFINITY), |b, (i, c)| if c.1 < b.1 { (i, c.1) } else { b });
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        if top.len() < k {
            top.push((i as u32, v));
            if top.len() == k {
                (min_i, min_v) = refresh(&top);
            }
        } else if v > min_v {
            top[min_i] = (i as u32, v);
            (min_i, min_v) = refresh(&top);
        }
    }
    top
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn untruncated_sampling_matches_softmax() {
        // top_k = 0 paths (no full sort): frequencies follow the temperature softmax, and top_p
        // keeps exactly the smallest prefix reaching top_p.
        let logits = [2.0f32, 1.0, 0.5, 0.0, -1.0, -3.0];
        let t = 0.8f32;
        let w: Vec<f64> = logits.iter().map(|&l| ((l - 2.0) / t).exp() as f64).collect();
        let z: f64 = w.iter().sum();
        for top_p in [1.0f32, 0.8] {
            let mut sm = Sampler::new(SamplerParams { temperature: t, top_k: 0, top_p, seed: 42, ..Default::default() });
            let mut cnt = [0usize; 6];
            let n = 200_000;
            for _ in 0..n {
                let mut l = logits;
                cnt[sm.sample(&mut l) as usize] += 1;
            }
            // expected distribution after the nucleus cut
            let mut keep = 6;
            if top_p < 1.0 {
                let mut cum = 0.0;
                for (i, wi) in w.iter().enumerate() {
                    cum += wi / z;
                    if cum >= top_p as f64 { keep = i + 1; break; }
                }
            }
            let zk: f64 = w[..keep].iter().sum();
            for i in 0..6 {
                let exp = if i < keep { w[i] / zk } else { 0.0 };
                let got = cnt[i] as f64 / n as f64;
                assert!((got - exp).abs() < 0.01, "top_p={top_p} token {i}: got {got:.4} expected {exp:.4}");
            }
        }
    }

    #[test]
    fn greedy_picks_max() {
        let mut s = Sampler::new(SamplerParams::greedy());
        assert_eq!(s.sample(&mut [0.1, 3.0, -1.0, f32::NEG_INFINITY]), 1);
    }
    #[test]
    fn top_k_one_is_greedy_and_seed_reproducible() {
        let p = SamplerParams { top_k: 1, temperature: 1.0, seed: 7, ..Default::default() };
        let mut s = Sampler::new(p);
        assert_eq!(s.sample(&mut [0.1, 3.0, 2.9]), 1);
        let p = SamplerParams { seed: 99, ..Default::default() };
        let (mut a, mut b) = (Sampler::new(p.clone()), Sampler::new(p));
        let l: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin()).collect();
        for _ in 0..20 {
            assert_eq!(a.sample(&mut l.clone()), b.sample(&mut l.clone()));
        }
    }
    #[test]
    fn repeat_penalty_demotes_seen_token() {
        let mut s = Sampler::new(SamplerParams { repeat_penalty: 10.0, ..SamplerParams::greedy() });
        s.accept(1);
        assert_eq!(s.sample(&mut [1.0, 2.0, 1.5]), 2);
    }
}

#[cfg(test)]
mod speed {
    use super::*;
    #[test]
    #[ignore = "benchmark"]
    fn sampler_speed() {
        let n = 248_320usize;
        let mut s = 12345u64;
        let base: Vec<f32> = (0..n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); ((s >> 40) as f32 / 16777216.0) * 20.0 - 10.0 }).collect();
        for (k, p) in [(40usize, 0.95f32), (0, 0.95), (0, 1.0)] {
            let mut sm = Sampler::new(SamplerParams { temperature: 0.7, top_k: k, top_p: p, ..Default::default() });
            let t = std::time::Instant::now();
            for _ in 0..20 { let mut l = base.clone(); std::hint::black_box(sm.sample(&mut l)); }
            eprintln!("top_k={k} top_p={p}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / 20.0);
        }
    }
}
