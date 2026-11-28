//! Dynamic `dlopen` FFI bindings for AMD HIP (`libamdhip64.so`).
//! Zero build-time ROCm SDK dependency; symbols are resolved once when `HipApi::load()` succeeds.
use libloading::{Library, Symbol};
use serde::Serialize;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};

#[derive(Debug, thiserror::Error)]
pub enum HipError {
    #[error("HIP runtime library not found (install rocm-hip-runtime to enable)")]
    Unavailable,
    #[error("HIPRTC compiler library not found (install rocm-hip-runtime to enable)")]
    RtcUnavailable,
    #[error("symbol lookup failed: {0}")]
    Symbol(String),
    #[error("hipError_t({code}): {msg}")]
    Api { code: c_int, msg: String },
    #[error("hiprtc compile failed (code {code}): {log}")]
    RtcCompile { code: c_int, log: String },
    #[error("invalid argument: {0}")]
    InvalidArg(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Serialize)]
pub struct HipStatus {
    pub available: bool,
    pub library: Option<String>,
    pub device_count: i32,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    pub device_id: i32,
    pub name: String,
    pub gcn_arch_name: String,
    /// Base gfx target (e.g. `"gfx1031"` with `:xnack-` / `:sramecc+` stripped).
    pub gfx_arch: String,
    pub warp_size: i32,
    pub cu_count: i32,
    pub max_shared_mem_per_block: i32,
    pub clock_khz: i32,
    pub mem_clock_khz: i32,
    pub mem_bus_width_bits: i32,
    pub l2_cache_bytes: i32,
    pub total_vram_bytes: usize,
    pub free_vram_bytes: usize,
}

pub const HIP_MEMCPY_H2H: c_int = 0;
pub const HIP_MEMCPY_H2D: c_int = 1;
pub const HIP_MEMCPY_D2H: c_int = 2;
pub const HIP_MEMCPY_D2D: c_int = 3;

pub const HIP_HOST_MALLOC_DEFAULT: c_uint = 0;
pub const HIP_HOST_MALLOC_MAPPED: c_uint = 2;
pub const HIP_HOST_MALLOC_WRITE_COMBINED: c_uint = 4;

pub const HIP_STREAM_NON_BLOCKING: c_uint = 1;
pub const HIP_EVENT_DEFAULT: c_uint = 0;
pub const HIP_EVENT_DISABLE_TIMING: c_uint = 2;

pub const HIP_STREAM_CAPTURE_MODE_GLOBAL: c_int = 0;
pub const HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;
pub const HIP_STREAM_CAPTURE_MODE_RELAXED: c_int = 2;

// ROCm 6/7 hipDeviceAttribute_t values (verified against /opt/rocm/include/hip/hip_runtime_api.h).
pub const HIP_ATTR_CLOCK_RATE: c_int = 5;
pub const HIP_ATTR_L2_CACHE_SIZE: c_int = 19;
pub const HIP_ATTR_MEM_BUS_WIDTH: c_int = 59;
pub const HIP_ATTR_MEM_CLOCK_RATE: c_int = 60;
pub const HIP_ATTR_MULTIPROCESSOR_COUNT: c_int = 63;
pub const HIP_ATTR_MAX_SHARED_MEM_PER_BLOCK: c_int = 74;
pub const HIP_ATTR_WARP_SIZE: c_int = 87;

// ROCm 6/7 hipDeviceProp_tR0600 gcnArchName offset (1160 bytes, 256-byte C string).
pub const HIP_PROP_R0600_MIN_SIZE: usize = 4096;
pub const HIP_PROP_R0600_GCN_ARCH_OFFSET: usize = 1160;
pub const HIP_PROP_R0600_GCN_ARCH_LEN: usize = 256;

#[derive(Clone, Copy)]
pub enum MemcpyKind {
    H2H,
    H2D,
    D2H,
    D2D,
}

impl MemcpyKind {
    pub fn raw(self) -> c_int {
        match self {
            Self::H2H => HIP_MEMCPY_H2H,
            Self::H2D => HIP_MEMCPY_H2D,
            Self::D2H => HIP_MEMCPY_D2H,
            Self::D2D => HIP_MEMCPY_D2D,
        }
    }
}

type HipGetErrorStringFn = unsafe extern "C" fn(c_int) -> *const c_char;
type HipGetDeviceCountFn = unsafe extern "C" fn(*mut c_int) -> c_int;
type HipSetDeviceFn = unsafe extern "C" fn(c_int) -> c_int;
type HipDeviceResetFn = unsafe extern "C" fn() -> c_int;
type HipDeviceSyncFn = unsafe extern "C" fn() -> c_int;
type HipDeviceGetNameFn = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
type HipDeviceGetAttrFn = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
type HipGetDevicePropsFn = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type HipMemGetInfoFn = unsafe extern "C" fn(*mut usize, *mut usize) -> c_int;

type HipMallocFn = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type HipFreeFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipHostMallocFn = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> c_int;
type HipHostFreeFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipHostGetDevicePtrFn = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, c_uint) -> c_int;

type HipMemcpyFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type HipMemcpyAsyncFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, *mut c_void) -> c_int;
type HipMemsetFn = unsafe extern "C" fn(*mut c_void, c_int, usize) -> c_int;
type HipMemsetAsyncFn = unsafe extern "C" fn(*mut c_void, c_int, usize, *mut c_void) -> c_int;

type HipStreamCreateWithFlagsFn = unsafe extern "C" fn(*mut *mut c_void, c_uint) -> c_int;
type HipStreamDestroyFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipStreamSyncFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipStreamWaitEventFn = unsafe extern "C" fn(*mut c_void, *mut c_void, c_uint) -> c_int;

type HipEventCreateWithFlagsFn = unsafe extern "C" fn(*mut *mut c_void, c_uint) -> c_int;
type HipEventDestroyFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipEventRecordFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type HipEventSyncFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipEventElapsedTimeFn = unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> c_int;

type HipModuleLoadDataFn = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> c_int;
type HipModuleUnloadFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipModuleGetFunctionFn = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int;
type HipModuleLaunchKernelFn = unsafe extern "C" fn(
    *mut c_void,
    c_uint, c_uint, c_uint,
    c_uint, c_uint, c_uint,
    c_uint, *mut c_void,
    *mut *mut c_void, *mut *mut c_void,
) -> c_int;

type HipStreamBeginCaptureFn = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type HipStreamEndCaptureFn = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> c_int;
type HipGraphInstantiateFn = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *mut *mut c_void, *mut c_char, usize) -> c_int;
type HipGraphLaunchFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type HipGraphExecDestroyFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipGraphDestroyFn = unsafe extern "C" fn(*mut c_void) -> c_int;

/// Pre-resolved `libamdhip64` symbol table. Kept alive alongside the `Library` handle.
pub struct HipApi {
    _lib: Library,
    pub library_name: String,
    pub get_error_string: Option<HipGetErrorStringFn>,
    pub get_device_count: HipGetDeviceCountFn,
    pub set_device: HipSetDeviceFn,
    pub device_reset: HipDeviceResetFn,
    pub device_sync: HipDeviceSyncFn,
    pub device_get_name: HipDeviceGetNameFn,
    pub device_get_attr: HipDeviceGetAttrFn,
    pub get_device_props_r0600: Option<HipGetDevicePropsFn>,
    pub mem_get_info: HipMemGetInfoFn,

    pub malloc: HipMallocFn,
    pub free: HipFreeFn,
    pub host_malloc: HipHostMallocFn,
    pub host_free: HipHostFreeFn,
    pub host_get_device_ptr: HipHostGetDevicePtrFn,

    pub memcpy: HipMemcpyFn,
    pub memcpy_async: HipMemcpyAsyncFn,
    pub memset: HipMemsetFn,
    pub memset_async: HipMemsetAsyncFn,

    pub stream_create_with_flags: HipStreamCreateWithFlagsFn,
    pub stream_destroy: HipStreamDestroyFn,
    pub stream_sync: HipStreamSyncFn,
    pub stream_wait_event: HipStreamWaitEventFn,

    pub event_create_with_flags: HipEventCreateWithFlagsFn,
    pub event_destroy: HipEventDestroyFn,
    pub event_record: HipEventRecordFn,
    pub event_sync: HipEventSyncFn,
    pub event_elapsed_time: HipEventElapsedTimeFn,

    pub module_load_data: HipModuleLoadDataFn,
    pub module_unload: HipModuleUnloadFn,
    pub module_get_function: HipModuleGetFunctionFn,
    pub module_launch_kernel: HipModuleLaunchKernelFn,

    pub stream_begin_capture: Option<HipStreamBeginCaptureFn>,
    pub stream_end_capture: Option<HipStreamEndCaptureFn>,
    pub graph_instantiate: Option<HipGraphInstantiateFn>,
    pub graph_launch: Option<HipGraphLaunchFn>,
    pub graph_exec_destroy: Option<HipGraphExecDestroyFn>,
    pub graph_destroy: Option<HipGraphDestroyFn>,
}

// All fields are function pointers + a process-global shared library handle.
unsafe impl Send for HipApi {}
unsafe impl Sync for HipApi {}

fn sym<T: Copy>(lib: &Library, name: &[u8]) -> Result<T, HipError> {
    unsafe {
        let s: Symbol<T> = lib
            .get(name)
            .map_err(|e| HipError::Symbol(format!("{}: {e}", String::from_utf8_lossy(name))))?;
        Ok(*s)
    }
}

fn opt_sym<T: Copy>(lib: &Library, name: &[u8]) -> Option<T> {
    unsafe { lib.get::<T>(name).ok().map(|s| *s) }
}

impl HipApi {
    pub fn load() -> Result<Self, HipError> {
        let mut found = None;
        for cand in [
            "libamdhip64.so.7",
            "libamdhip64.so.6",
            "libamdhip64.so",
            "/opt/rocm/lib/libamdhip64.so.7",
            "/opt/rocm/lib/libamdhip64.so.6",
            "/opt/rocm/lib/libamdhip64.so",
        ] {
            if let Ok(lib) = unsafe { Library::new(cand) } {
                found = Some((lib, cand.to_string()));
                break;
            }
        }
        let (lib, library_name) = found.ok_or(HipError::Unavailable)?;

        Ok(Self {
            get_error_string: opt_sym(&lib, b"hipGetErrorString\0"),
            get_device_count: sym(&lib, b"hipGetDeviceCount\0")?,
            set_device: sym(&lib, b"hipSetDevice\0")?,
            device_reset: sym(&lib, b"hipDeviceReset\0")?,
            device_sync: sym(&lib, b"hipDeviceSynchronize\0")?,
            device_get_name: sym(&lib, b"hipDeviceGetName\0")?,
            device_get_attr: sym(&lib, b"hipDeviceGetAttribute\0")?,
            get_device_props_r0600: opt_sym(&lib, b"hipGetDevicePropertiesR0600\0"),
            mem_get_info: sym(&lib, b"hipMemGetInfo\0")?,

            malloc: sym(&lib, b"hipMalloc\0")?,
            free: sym(&lib, b"hipFree\0")?,
            host_malloc: sym(&lib, b"hipHostMalloc\0")?,
            host_free: sym(&lib, b"hipHostFree\0")?,
            host_get_device_ptr: sym(&lib, b"hipHostGetDevicePointer\0")?,

            memcpy: sym(&lib, b"hipMemcpy\0")?,
            memcpy_async: sym(&lib, b"hipMemcpyAsync\0")?,
            memset: sym(&lib, b"hipMemset\0")?,
            memset_async: sym(&lib, b"hipMemsetAsync\0")?,

            stream_create_with_flags: sym(&lib, b"hipStreamCreateWithFlags\0")?,
            stream_destroy: sym(&lib, b"hipStreamDestroy\0")?,
            stream_sync: sym(&lib, b"hipStreamSynchronize\0")?,
            stream_wait_event: sym(&lib, b"hipStreamWaitEvent\0")?,

            event_create_with_flags: sym(&lib, b"hipEventCreateWithFlags\0")?,
            event_destroy: sym(&lib, b"hipEventDestroy\0")?,
            event_record: sym(&lib, b"hipEventRecord\0")?,
            event_sync: sym(&lib, b"hipEventSynchronize\0")?,
            event_elapsed_time: sym(&lib, b"hipEventElapsedTime\0")?,

            module_load_data: sym(&lib, b"hipModuleLoadData\0")?,
            module_unload: sym(&lib, b"hipModuleUnload\0")?,
            module_get_function: sym(&lib, b"hipModuleGetFunction\0")?,
            module_launch_kernel: sym(&lib, b"hipModuleLaunchKernel\0")?,

            stream_begin_capture: opt_sym(&lib, b"hipStreamBeginCapture\0"),
            stream_end_capture: opt_sym(&lib, b"hipStreamEndCapture\0"),
            graph_instantiate: opt_sym(&lib, b"hipGraphInstantiate\0"),
            graph_launch: opt_sym(&lib, b"hipGraphLaunch\0"),
            graph_exec_destroy: opt_sym(&lib, b"hipGraphExecDestroy\0"),
            graph_destroy: opt_sym(&lib, b"hipGraphDestroy\0"),
            _lib: lib,
            library_name,
        })
    }

    pub fn check(&self, rc: c_int) -> Result<(), HipError> {
        if rc == 0 {
            return Ok(());
        }
        let msg = self
            .get_error_string
            .and_then(|f| {
                let p = unsafe { f(rc) };
                if p.is_null() {
                    None
                } else {
                    Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
                }
            })
            .unwrap_or_else(|| "unknown HIP error".to_string());
        Err(HipError::Api { code: rc, msg })
    }
}
