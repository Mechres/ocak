//! Dynamic `dlopen` wrapper for `libhiprtc.so` with an on-disk HSACO cache.
//! Compiles HIP C++ kernels at runtime with zero ROCm SDK build-time dependency.
use crate::ffi::HipError;
use libloading::{Library, Symbol};
use sha2::{Digest, Sha256};
use std::ffi::{c_char, c_int, c_void, CString};
use std::fs;
use std::path::PathBuf;
use std::ptr;

type HiprtcVersionFn = unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int;
type HiprtcCreateProgramFn = unsafe extern "C" fn(
    *mut *mut c_void,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> c_int;
type HiprtcDestroyProgramFn = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
type HiprtcCompileProgramFn =
    unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> c_int;
type HiprtcGetProgramLogSizeFn = unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int;
type HiprtcGetProgramLogFn = unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int;
type HiprtcGetCodeSizeFn = unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int;
type HiprtcGetCodeFn = unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int;

pub struct Hiprtc {
    _lib: Library,
    pub version: (i32, i32),
    create_program: HiprtcCreateProgramFn,
    destroy_program: HiprtcDestroyProgramFn,
    compile_program: HiprtcCompileProgramFn,
    get_log_size: HiprtcGetProgramLogSizeFn,
    get_log: HiprtcGetProgramLogFn,
    get_code_size: HiprtcGetCodeSizeFn,
    get_code: HiprtcGetCodeFn,
}

unsafe impl Send for Hiprtc {}
unsafe impl Sync for Hiprtc {}

fn sym<T: Copy>(lib: &Library, name: &[u8]) -> Result<T, HipError> {
    unsafe {
        let s: Symbol<T> = lib
            .get(name)
            .map_err(|e| HipError::Symbol(format!("{}: {e}", String::from_utf8_lossy(name))))?;
        Ok(*s)
    }
}

impl Hiprtc {
    pub fn load() -> Result<Self, HipError> {
        let mut found = None;
        for cand in [
            "libhiprtc.so.7",
            "libhiprtc.so.6",
            "libhiprtc.so",
            "/opt/rocm/lib/libhiprtc.so.7",
            "/opt/rocm/lib/libhiprtc.so.6",
            "/opt/rocm/lib/libhiprtc.so",
        ] {
            if let Ok(lib) = unsafe { Library::new(cand) } {
                found = Some(lib);
                break;
            }
        }
        let lib = found.ok_or(HipError::RtcUnavailable)?;
        let version_fn: HiprtcVersionFn = sym(&lib, b"hiprtcVersion\0")?;
        let (mut maj, mut min): (c_int, c_int) = (0, 0);
        let _ = unsafe { version_fn(&mut maj, &mut min) };

        Ok(Self {
            version: (maj, min),
            create_program: sym(&lib, b"hiprtcCreateProgram\0")?,
            destroy_program: sym(&lib, b"hiprtcDestroyProgram\0")?,
            compile_program: sym(&lib, b"hiprtcCompileProgram\0")?,
            get_log_size: sym(&lib, b"hiprtcGetProgramLogSize\0")?,
            get_log: sym(&lib, b"hiprtcGetProgramLog\0")?,
            get_code_size: sym(&lib, b"hiprtcGetCodeSize\0")?,
            get_code: sym(&lib, b"hiprtcGetCode\0")?,
            _lib: lib,
        })
    }

    /// Compiles HIP C++ source to an HSACO binary for `gfx_arch` (e.g. `"gfx1031"`).
    pub fn compile_to_hsaco(
        &self,
        source: &str,
        unit_name: &str,
        gfx_arch: &str,
        extra_opts: &[&str],
    ) -> Result<Vec<u8>, HipError> {
        let c_src = CString::new(source).map_err(|e| HipError::InvalidArg(e.to_string()))?;
        let c_name = CString::new(unit_name).map_err(|e| HipError::InvalidArg(e.to_string()))?;

        let mut opts_c: Vec<CString> = Vec::with_capacity(extra_opts.len() + 3);
        if !gfx_arch.is_empty() {
            opts_c.push(
                CString::new(format!("--offload-arch={gfx_arch}"))
                    .map_err(|e| HipError::InvalidArg(e.to_string()))?,
            );
        }
        opts_c.push(CString::new("-O3").unwrap());
        opts_c.push(CString::new("-ffast-math").unwrap());
        for &o in extra_opts {
            opts_c.push(CString::new(o).map_err(|e| HipError::InvalidArg(e.to_string()))?);
        }
        let opt_ptrs: Vec<*const c_char> = opts_c.iter().map(|s| s.as_ptr()).collect();

        let mut prog: *mut c_void = ptr::null_mut();
        let rc = unsafe {
            (self.create_program)(
                &mut prog,
                c_src.as_ptr(),
                c_name.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            )
        };
        if rc != 0 {
            return Err(HipError::RtcCompile {
                code: rc,
                log: "hiprtcCreateProgram failed".into(),
            });
        }

        struct ProgGuard<'a> {
            rtc: &'a Hiprtc,
            prog: *mut c_void,
        }
        impl Drop for ProgGuard<'_> {
            fn drop(&mut self) {
                if !self.prog.is_null() {
                    let _ = unsafe { (self.rtc.destroy_program)(&mut self.prog) };
                }
            }
        }
        let guard = ProgGuard { rtc: self, prog };

        let rc = unsafe {
            (self.compile_program)(guard.prog, opt_ptrs.len() as c_int, opt_ptrs.as_ptr())
        };
        if rc != 0 {
            let mut log_sz: usize = 0;
            let _ = unsafe { (self.get_log_size)(guard.prog, &mut log_sz) };
            let mut log_buf = vec![0u8; log_sz.max(1)];
            if log_sz > 0 {
                let _ = unsafe {
                    (self.get_log)(guard.prog, log_buf.as_mut_ptr() as *mut c_char)
                };
            }
            while log_buf.last() == Some(&0) {
                log_buf.pop();
            }
            return Err(HipError::RtcCompile {
                code: rc,
                log: String::from_utf8_lossy(&log_buf).into_owned(),
            });
        }

        let mut code_sz: usize = 0;
        let rc = unsafe { (self.get_code_size)(guard.prog, &mut code_sz) };
        if rc != 0 || code_sz == 0 {
            return Err(HipError::RtcCompile {
                code: rc,
                log: "hiprtcGetCodeSize returned 0".into(),
            });
        }
        let mut code = vec![0u8; code_sz];
        let rc = unsafe { (self.get_code)(guard.prog, code.as_mut_ptr() as *mut c_char) };
        if rc != 0 {
            return Err(HipError::RtcCompile {
                code: rc,
                log: "hiprtcGetCode failed".into(),
            });
        }
        Ok(code)
    }

    /// Compiles to HSACO with an on-disk cache keyed by `(hiprtc_version, gfx_arch, opts, source)`.
    pub fn compile_cached(
        &self,
        source: &str,
        unit_name: &str,
        gfx_arch: &str,
        extra_opts: &[&str],
    ) -> Result<Vec<u8>, HipError> {
        let cache_disabled = std::env::var("OCAK_HIP_CACHE_DISABLE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let key = cache_key(self.version, gfx_arch, extra_opts, source);
        let path = cache_dir().join(format!("{gfx_arch}_{key}.hsaco"));

        if !cache_disabled {
            if let Ok(bytes) = fs::read(&path) {
                if !bytes.is_empty() {
                    return Ok(bytes);
                }
            }
        }

        let hsaco = self.compile_to_hsaco(source, unit_name, gfx_arch, extra_opts)?;
        if !cache_disabled {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
            if fs::write(&tmp, &hsaco).is_ok() {
                let _ = fs::rename(&tmp, &path);
            }
        }
        Ok(hsaco)
    }
}

fn cache_key(ver: (i32, i32), gfx_arch: &str, opts: &[&str], source: &str) -> String {
    let mut h = Sha256::new();
    h.update(b"ocak-hip-v1:");
    h.update(format!("{}.{}:{}:", ver.0, ver.1, gfx_arch).as_bytes());
    for o in opts {
        h.update(o.as_bytes());
        h.update(b"\0");
    }
    h.update(source.as_bytes());
    let digest = h.finalize();
    let mut hex = String::with_capacity(24);
    for b in &digest[..12] {
        use std::fmt::Write;
        let _ = write!(&mut hex, "{b:02x}");
    }
    hex
}

fn cache_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("ocak").join("hip");
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home).join(".cache").join("ocak").join("hip");
        }
    }
    std::env::temp_dir().join("ocak-hip-cache")
}
