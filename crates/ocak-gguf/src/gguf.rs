use crate::quant::GgmlType;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad GGUF: {0}")]
    Bad(String),
}

type R<T> = Result<T, GgufError>;

#[derive(Debug, Clone)]
pub enum ArrayValue {
    Strings(Vec<String>),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32), F32(f32), Bool(bool),
    Str(String), U64(u64), I64(i64), F64(f64),
    Array(ArrayValue),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            Value::U8(v) => *v as u64, Value::U16(v) => *v as u64, Value::U32(v) => *v as u64,
            Value::U64(v) => *v,
            Value::I8(v) if *v >= 0 => *v as u64, Value::I16(v) if *v >= 0 => *v as u64,
            Value::I32(v) if *v >= 0 => *v as u64, Value::I64(v) if *v >= 0 => *v as u64,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F32(v) => Some(*v as f64),
            Value::F64(v) => Some(*v),
            o => o.as_u64().map(|v| v as f64),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Value::Str(s) = self { Some(s) } else { None }
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Value::Bool(b) = self { Some(*b) } else { None }
    }
    pub fn display(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            Value::Array(ArrayValue::Strings(v)) => format!("[{} strings]", v.len()),
            Value::Array(ArrayValue::Ints(v)) => format!("[{} ints]", v.len()),
            Value::Array(ArrayValue::Floats(v)) => format!("[{} floats]", v.len()),
            Value::Bool(b) => b.to_string(),
            Value::F32(v) => v.to_string(),
            Value::F64(v) => v.to_string(),
            o => o.as_u64().map(|v| v.to_string()).unwrap_or_else(|| format!("{o:?}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub ne: [u64; 4],
    pub n_dims: u32,
    pub ty: GgmlType,
    /// Offset relative to the data section.
    pub offset: u64,
    pub n_bytes: u64,
}

impl TensorInfo {
    pub fn n_elems(&self) -> u64 {
        self.ne.iter().product()
    }
    /// ne[0] is the contiguous dimension (columns / K).
    pub fn n_cols(&self) -> usize { self.ne[0] as usize }
    pub fn n_rows(&self) -> usize { (self.ne[1] * self.ne[2] * self.ne[3]) as usize }
    pub fn row_bytes(&self) -> usize { self.ty.row_bytes(self.n_cols()).unwrap_or(0) }
}

pub struct GgufFile {
    pub path: PathBuf,
    pub version: u32,
    pub alignment: u64,
    pub data_offset: u64,
    pub kv: HashMap<String, Value>,
    pub kv_order: Vec<String>,
    pub tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
    mmap: Mmap,
}

struct Cur<'a> { b: &'a [u8], p: usize }

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> R<&'a [u8]> {
        if self.p.checked_add(n).map_or(true, |e| e > self.b.len()) {
            return Err(GgufError::Bad("truncated".into()));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> R<u8> { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> R<u16> { Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap())) }
    fn u32(&mut self) -> R<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn u64(&mut self) -> R<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn string(&mut self) -> R<String> {
        let n = self.u64()? as usize;
        if n > 256 << 20 { return Err(GgufError::Bad("string too long".into())); }
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn scalar(&mut self, ty: u32) -> R<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            t => return Err(GgufError::Bad(format!("unknown value type {t}"))),
        })
    }
    fn value(&mut self, ty: u32) -> R<Value> {
        if ty != 9 { return self.scalar(ty); }
        let et = self.u32()?;
        let n = self.u64()? as usize;
        if n > 64_000_000 { return Err(GgufError::Bad("array too large".into())); }
        Ok(Value::Array(match et {
            8 => ArrayValue::Strings((0..n).map(|_| self.string()).collect::<R<_>>()?),
            6 | 12 => ArrayValue::Floats(
                (0..n).map(|_| self.scalar(et).map(|v| v.as_f64().unwrap())).collect::<R<_>>()?),
            9 => return Err(GgufError::Bad("nested arrays".into())),
            _ => ArrayValue::Ints((0..n).map(|_| {
                self.scalar(et).map(|v| match v {
                    Value::Bool(b) => b as i64,
                    Value::I8(x) => x as i64, Value::I16(x) => x as i64,
                    Value::I32(x) => x as i64, Value::I64(x) => x,
                    o => o.as_u64().unwrap_or(0) as i64,
                })
            }).collect::<R<_>>()?),
        }))
    }
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> R<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut c = Cur { b: &mmap, p: 0 };
        if c.take(4)? != b"GGUF" { return Err(GgufError::Bad("magic".into())); }
        let version = c.u32()?;
        if version != 2 && version != 3 { return Err(GgufError::Bad(format!("version {version}"))); }
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;
        if n_tensors > 1_000_000 || n_kv > 1_000_000 { return Err(GgufError::Bad("counts".into())); }
        let mut kv = HashMap::new();
        let mut kv_order = Vec::new();
        for _ in 0..n_kv {
            let k = c.string()?;
            let t = c.u32()?;
            let v = c.value(t)?;
            kv_order.push(k.clone());
            kv.insert(k, v);
        }
        let alignment = kv.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32).max(1);
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()?;
            if n_dims == 0 || n_dims > 4 { return Err(GgufError::Bad(format!("n_dims {n_dims}"))); }
            let mut ne = [1u64; 4];
            for d in 0..n_dims as usize { ne[d] = c.u64()?; }
            let ty = GgmlType::from_u32(c.u32()?);
            let offset = c.u64()?;
            let n_elems: u64 = ne.iter().product();
            let n_bytes = match ty.block() {
                Some((be, bb)) => n_elems / be as u64 * bb as u64,
                None => 0,
            };
            tensors.push(TensorInfo { name, ne, n_dims, ty, offset, n_bytes });
        }
        let data_offset = (c.p as u64 + alignment - 1) / alignment * alignment;
        for t in &tensors {
            if data_offset + t.offset + t.n_bytes > mmap.len() as u64 {
                return Err(GgufError::Bad(format!("tensor {} out of bounds", t.name)));
            }
        }
        let index = tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        Ok(Self { path, version, alignment, data_offset, kv, kv_order, tensors, index, mmap })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn tensor_data(&self, t: &TensorInfo) -> &[u8] {
        let s = (self.data_offset + t.offset) as usize;
        &self.mmap[s..s + t.n_bytes as usize]
    }

    pub fn file_len(&self) -> u64 { self.mmap.len() as u64 }

    pub fn get_str(&self, k: &str) -> Option<&str> { self.kv.get(k).and_then(|v| v.as_str()) }
    pub fn get_u64(&self, k: &str) -> Option<u64> { self.kv.get(k).and_then(|v| v.as_u64()) }
    pub fn get_f64(&self, k: &str) -> Option<f64> { self.kv.get(k).and_then(|v| v.as_f64()) }
    pub fn get_bool(&self, k: &str) -> Option<bool> { self.kv.get(k).and_then(|v| v.as_bool()) }

    pub fn architecture(&self) -> Option<&str> { self.get_str("general.architecture") }

    /// Looks up `<arch>.<suffix>`.
    pub fn arch_u64(&self, suffix: &str) -> Option<u64> {
        let a = self.architecture()?;
        self.get_u64(&format!("{a}.{suffix}"))
    }
    pub fn arch_f64(&self, suffix: &str) -> Option<f64> {
        let a = self.architecture()?;
        self.get_f64(&format!("{a}.{suffix}"))
    }

    pub fn strings(&self, k: &str) -> Option<&Vec<String>> {
        match self.kv.get(k) { Some(Value::Array(ArrayValue::Strings(v))) => Some(v), _ => None }
    }
    pub fn ints(&self, k: &str) -> Option<&Vec<i64>> {
        match self.kv.get(k) { Some(Value::Array(ArrayValue::Ints(v))) => Some(v), _ => None }
    }
    pub fn floats(&self, k: &str) -> Option<&Vec<f64>> {
        match self.kv.get(k) { Some(Value::Array(ArrayValue::Floats(v))) => Some(v), _ => None }
    }

    /// Dequantizes a full tensor to f32 (reference path; used for embeddings, norms and tests).
    pub fn tensor_to_f32(&self, t: &TensorInfo) -> Vec<f32> {
        let n = t.n_elems() as usize;
        let mut out = vec![0f32; n];
        crate::quant::dequantize_row(t.ty, self.tensor_data(t), &mut out, n);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn push_str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn parses_minimal_synthetic_file() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // tensors
        b.extend_from_slice(&3u64.to_le_bytes()); // kv
        push_str(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes());
        push_str(&mut b, "llama");
        push_str(&mut b, "llama.block_count");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&22u32.to_le_bytes());
        push_str(&mut b, "tokenizer.ggml.tokens");
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        push_str(&mut b, "a");
        push_str(&mut b, "b");
        push_str(&mut b, "w");
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&4u64.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // F32
        b.extend_from_slice(&0u64.to_le_bytes());
        while b.len() % 32 != 0 { b.push(0); }
        for i in 0..8 { b.extend_from_slice(&(i as f32).to_le_bytes()); }
        let p = std::env::temp_dir().join(format!("ocak-gguf-test-{}.gguf", std::process::id()));
        std::fs::File::create(&p).unwrap().write_all(&b).unwrap();
        let g = GgufFile::open(&p).unwrap();
        assert_eq!(g.architecture(), Some("llama"));
        assert_eq!(g.arch_u64("block_count"), Some(22));
        assert_eq!(g.strings("tokenizer.ggml.tokens").unwrap().len(), 2);
        let t = g.tensor("w").unwrap();
        assert_eq!((t.n_cols(), t.n_rows(), t.row_bytes()), (4, 2, 16));
        assert_eq!(g.tensor_to_f32(t)[7], 7.0);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn rejects_bad_magic() {
        let p = std::env::temp_dir().join(format!("ocak-gguf-bad-{}.gguf", std::process::id()));
        std::fs::write(&p, b"NOPE0000000000000000").unwrap();
        assert!(GgufFile::open(&p).is_err());
        let _ = std::fs::remove_file(p);
    }
}
