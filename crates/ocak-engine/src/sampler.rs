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
        let seed = if p.seed == 0 {
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
        let mut cand: Vec<(u32, f32)> = logits.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, &v)| (i as u32, v)).collect();
        if cand.is_empty() {
            return 0;
        }
        let cmp = |a: &(u32, f32), b: &(u32, f32)| b.1.partial_cmp(&a.1).unwrap();
        if p.top_k > 0 && p.top_k < cand.len() {
            cand.select_nth_unstable_by(p.top_k - 1, cmp);
            cand.truncate(p.top_k);
        }
        cand.sort_by(cmp);
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

#[cfg(test)]
mod tests {
    use super::*;
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

