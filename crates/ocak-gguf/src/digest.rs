//! `<model>.gguf.digest` files: `size mtime_sec mtime_nsec ino dev sha256`.
//! A cached digest is trusted only when the stat tuple still matches.
use sha2::{Digest, Sha256};
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DigestRecord {
    pub size: u64,
    pub mtime_sec: i64,
    pub mtime_nsec: i64,
    pub ino: u64,
    pub dev: u64,
    pub sha256: String,
}

pub fn digest_path(model: &Path) -> PathBuf {
    let mut s = model.as_os_str().to_owned();
    s.push(".digest");
    PathBuf::from(s)
}

pub fn parse_digest(text: &str) -> Option<DigestRecord> {
    let mut it = text.split_whitespace();
    let rec = DigestRecord {
        size: it.next()?.parse().ok()?,
        mtime_sec: it.next()?.parse().ok()?,
        mtime_nsec: it.next()?.parse().ok()?,
        ino: it.next()?.parse().ok()?,
        dev: it.next()?.parse().ok()?,
        sha256: it.next()?.to_string(),
    };
    (rec.sha256.len() == 64 && rec.sha256.bytes().all(|b| b.is_ascii_hexdigit())).then_some(rec)
}

/// Returns the cached SHA-256 when the `.digest` file matches the model's current stat.
pub fn cached_sha256(model: &Path) -> Option<String> {
    let rec = parse_digest(&std::fs::read_to_string(digest_path(model)).ok()?)?;
    let md = std::fs::metadata(model).ok()?;
    (rec.size == md.len()
        && rec.mtime_sec == md.mtime()
        && rec.mtime_nsec == md.mtime_nsec()
        && rec.ino == md.ino()
        && rec.dev == md.dev())
    .then_some(rec.sha256)
}

pub fn compute_sha256(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_real_digest_line() {
        let r = parse_digest("1587969920 1793668175 611652613 28531882 54 4a7d91c094b9a0f8ed094093f3a0eb3cfeeb46b9bd491e33115689fc5d658b22\n").unwrap();
        assert_eq!(r.size, 1587969920);
        assert_eq!(r.dev, 54);
        assert!(parse_digest("1 2 3 4 5 xyz").is_none());
    }
    #[test]
    fn sha256_of_known_bytes() {
        let p = std::env::temp_dir().join(format!("ocak-sha-{}", std::process::id()));
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(compute_sha256(&p).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        let _ = std::fs::remove_file(p);
    }
}
