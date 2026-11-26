//! Parser for llama.cpp-style `models-preset.ini` (`[*]` defaults + per-model sections).
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
pub struct Presets {
    pub defaults: BTreeMap<String, String>,
    pub models: BTreeMap<String, BTreeMap<String, String>>,
}

impl Presets {
    pub fn parse(text: &str) -> Self {
        let mut p = Presets::default();
        let mut cur: Option<String> = None;
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
                continue;
            }
            if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                cur = Some(name.trim().to_string());
                if name.trim() != "*" {
                    p.models.entry(name.trim().to_string()).or_default();
                }
                continue;
            }
            if let (Some(sec), Some((k, v))) = (&cur, line.split_once('=')) {
                let (k, v) = (k.trim().to_string(), v.trim().to_string());
                if sec == "*" {
                    p.defaults.insert(k, v);
                } else {
                    p.models.get_mut(sec).unwrap().insert(k, v);
                }
            }
        }
        p
    }

    pub fn load(path: &std::path::Path) -> Self {
        std::fs::read_to_string(path).map(|t| Self::parse(&t)).unwrap_or_default()
    }

    /// Defaults overlaid with the model's own section.
    pub fn resolve(&self, model: &str) -> BTreeMap<String, String> {
        let mut m = self.defaults.clone();
        if let Some(o) = self.models.get(model) {
            m.extend(o.clone());
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn merges_defaults() {
        let p = Presets::parse("version = 1\n[*]\nc = 65536\nfa = 1\n; note\n[Xing]\nc = 131072\nn-cpu-moe = 10\n");
        let r = p.resolve("Xing");
        assert_eq!(r["c"], "131072");
        assert_eq!(r["fa"], "1");
        assert_eq!(r["n-cpu-moe"], "10");
        assert_eq!(p.resolve("other")["c"], "65536");
    }
    #[test]
    fn parses_real_preset() {
        let home = std::env::var("HOME").unwrap_or_default();
        let path = std::path::PathBuf::from(home).join("models/models-preset.ini");
        if !path.exists() {
            return;
        }
        let p = Presets::load(&path);
        if !p.models.is_empty() {
            assert_eq!(p.resolve("Xing4.0-29B-A4B-IQ3_M")["n-cpu-moe"], "10");
            assert_eq!(p.defaults["sleep-idle-seconds"], "120");
        }
    }
}
