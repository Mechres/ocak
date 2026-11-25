//! Dynamic `dlopen` AMD HIP runtime wrapper (`libamdhip64.so`) for `gfx1031` HSACO kernels.
//! Builds cleanly with zero ROCm SDK installed; lights up automatically when ROCm is present.
use libloading::{Library, Symbol};
use serde::Serialize;
use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::ptr;

#[derive(Debug, thiserror::Error)]
pub enum HipError {
    #[error("HIP runtime library not found (install rocm-hip-runtime to enable)")]
    Unavailable,
    #[error("symbol lookup failed: {0}")]
    Symbol(String),
    #[error("hipError_t = {0}")]
    Api(c_int),
}

#[derive(Debug, Clone, Serialize)]
pub struct HipStatus {
    pub available: bool,
    pub library: Option<String>,
    pub device_count: i32,
}

type HipGetDeviceCountFn = unsafe extern "C" fn(*mut c_int) -> c_int;
type HipMallocFn = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type HipFreeFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipMemcpyFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type HipDeviceSyncFn = unsafe extern "C" fn() -> c_int;
type HipModuleLoadDataFn = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> c_int;
type HipModuleGetFunctionFn = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int;
type HipModuleLaunchKernelFn = unsafe extern "C" fn(
    *mut c_void,
    c_uint, c_uint, c_uint,
    c_uint, c_uint, c_uint,
    c_uint, *mut c_void,
    *mut *mut c_void, *mut *mut c_void,
) -> c_int;

pub const HIP_MEMCPY_H2D: c_int = 1;
pub const HIP_MEMCPY_D2H: c_int = 2;

pub struct HipRuntime {
    lib: Library,
    pub library_name: String,
}

impl HipRuntime {
    pub fn load() -> Result<Self, HipError> {
        for cand in ["libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so", "/opt/rocm/lib/libamdhip64.so"] {
            if let Ok(lib) = unsafe { Library::new(cand) } {
                return Ok(Self { lib, library_name: cand.to_string() });
            }
        }
        Err(HipError::Unavailable)
    }

    pub fn probe() -> HipStatus {
        match Self::load() {
            Ok(rt) => HipStatus {
                available: true,
                library: Some(rt.library_name.clone()),
                device_count: rt.device_count().unwrap_or(0),
            },
            Err(_) => HipStatus { available: false, library: None, device_count: 0 },
        }
    }

    pub fn device_count(&self) -> Result<i32, HipError> {
        unsafe {
            let f: Symbol<HipGetDeviceCountFn> = self.lib.get(b"hipGetDeviceCount").map_err(|e| HipError::Symbol(e.to_string()))?;
            let mut n: c_int = 0;
            let rc = f(&mut n);
            if rc != 0 { return Err(HipError::Api(rc)); }
            Ok(n)
        }
    }

    pub fn malloc(&self, bytes: usize) -> Result<*mut c_void, HipError> {
        unsafe {
            let f: Symbol<HipMallocFn> = self.lib.get(b"hipMalloc").map_err(|e| HipError::Symbol(e.to_string()))?;
            let mut p: *mut c_void = ptr::null_mut();
            let rc = f(&mut p, bytes);
            if rc != 0 { return Err(HipError::Api(rc)); }
            Ok(p)
        }
    }

    pub fn free(&self, ptr: *mut c_void) -> Result<(), HipError> {
        unsafe {
            let f: Symbol<HipFreeFn> = self.lib.get(b"hipFree").map_err(|e| HipError::Symbol(e.to_string()))?;
            let rc = f(ptr);
            if rc != 0 { return Err(HipError::Api(rc)); }
            Ok(())
        }
    }

    pub fn memcpy(&self, dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int) -> Result<(), HipError> {
        unsafe {
            let f: Symbol<HipMemcpyFn> = self.lib.get(b"hipMemcpy").map_err(|e| HipError::Symbol(e.to_string()))?;
            let rc = f(dst, src, bytes, kind);
            if rc != 0 { return Err(HipError::Api(rc)); }
            Ok(())
        }
    }

    pub fn synchronize(&self) -> Result<(), HipError> {
        unsafe {
            let f: Symbol<HipDeviceSyncFn> = self.lib.get(b"hipDeviceSynchronize").map_err(|e| HipError::Symbol(e.to_string()))?;
            let rc = f();
            if rc != 0 { return Err(HipError::Api(rc)); }
            Ok(())
        }
    }

    /// Loads an offline-compiled `gfx1031` HSACO code object and resolves a kernel symbol.
    pub fn load_hsaco_kernel(&self, hsaco: &[u8], kernel_name: &str) -> Result<(*mut c_void, *mut c_void), HipError> {
        let cname = CString::new(kernel_name).map_err(|e| HipError::Symbol(e.to_string()))?;
        unsafe {
            let load: Symbol<HipModuleLoadDataFn> = self.lib.get(b"hipModuleLoadData").map_err(|e| HipError::Symbol(e.to_string()))?;
            let getfn: Symbol<HipModuleGetFunctionFn> = self.lib.get(b"hipModuleGetFunction").map_err(|e| HipError::Symbol(e.to_string()))?;
            let mut module: *mut c_void = ptr::null_mut();
            let rc = load(&mut module, hsaco.as_ptr() as *const c_void);
            if rc != 0 { return Err(HipError::Api(rc)); }
            let mut func: *mut c_void = ptr::null_mut();
            let rc = getfn(&mut func, module, cname.as_ptr());
            if rc != 0 { return Err(HipError::Api(rc)); }
            Ok((module, func))
        }
    }

    pub unsafe fn launch_kernel(
        &self,
        func: *mut c_void,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), HipError> {
        let launch: Symbol<HipModuleLaunchKernelFn> = self.lib.get(b"hipModuleLaunchKernel").map_err(|e| HipError::Symbol(e.to_string()))?;
        let rc = launch(
            func,
            grid.0, grid.1, grid.2,
            block.0, block.1, block.2,
            shared_bytes,
            ptr::null_mut(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        );
        if rc != 0 { return Err(HipError::Api(rc)); }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn probe_reports_cleanly_without_rocm() {
        let st = HipRuntime::probe();
        assert!(st.device_count >= 0);
    }
}
