//! Safe RAII wrappers around the AMD HIP runtime (`HipContext`, `HipBuffer`, `HipPinnedBuffer`,
//! `HipStream`, `HipEvent`, `HipModule`, `HipFunction`, `HipGraphExec`).
use crate::ffi::{
    DeviceInfo, HipApi, HipError, MemcpyKind, HIP_ATTR_CLOCK_RATE, HIP_ATTR_L2_CACHE_SIZE,
    HIP_ATTR_MAX_SHARED_MEM_PER_BLOCK, HIP_ATTR_MEM_BUS_WIDTH, HIP_ATTR_MEM_CLOCK_RATE,
    HIP_ATTR_MULTIPROCESSOR_COUNT, HIP_ATTR_WARP_SIZE, HIP_EVENT_DEFAULT,
    HIP_EVENT_DISABLE_TIMING, HIP_HOST_MALLOC_MAPPED, HIP_PROP_R0600_GCN_ARCH_LEN,
    HIP_PROP_R0600_GCN_ARCH_OFFSET, HIP_PROP_R0600_MIN_SIZE, HIP_STREAM_CAPTURE_MODE_RELAXED,
    HIP_STREAM_NON_BLOCKING,
};
use crate::rtc::Hiprtc;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr;
use std::sync::{Arc, OnceLock};

/// Extra trailing bytes allocated on every `HipBuffer` so unaligned 32-bit funnel-shift loads
/// at the end of quantized rows never read past the allocation.
const BUFFER_TAIL_PAD: usize = 16;

pub struct HipContext {
    pub(crate) api: HipApi,
    pub info: DeviceInfo,
    rtc: OnceLock<Result<Hiprtc, String>>,
}

impl HipContext {
    /// Initializes HIP on `device_id` (typically `0`) and queries device properties.
    pub fn new(device_id: i32) -> Result<Arc<Self>, HipError> {
        let api = HipApi::load()?;
        let mut count: c_int = 0;
        api.check(unsafe { (api.get_device_count)(&mut count) })?;
        if device_id < 0 || device_id >= count {
            return Err(HipError::InvalidArg(format!(
                "device_id {device_id} out of range (found {count} HIP devices)"
            )));
        }
        api.check(unsafe { (api.set_device)(device_id) })?;
        let info = query_device_info(&api, device_id)?;
        Ok(Arc::new(Self {
            api,
            info,
            rtc: OnceLock::new(),
        }))
    }

    pub fn library_name(&self) -> &str {
        &self.api.library_name
    }

    /// Queries current `(free_bytes, total_bytes)` of VRAM.
    pub fn mem_info(&self) -> Result<(usize, usize), HipError> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        self.api.check(unsafe { (self.api.mem_get_info)(&mut free, &mut total) })?;
        Ok((free, total))
    }

    /// Synchronizes all work on the current device.
    pub fn synchronize(&self) -> Result<(), HipError> {
        self.api.check(unsafe { (self.api.device_sync)() })
    }

    /// Resets the current HIP device (releases KFD compute queues and all allocations).
    ///
    /// # Safety
    /// No live `HipBuffer`, `HipStream`, `HipEvent`, `HipModule`, or `HipGraphExec` may be used
    /// after calling `reset_device()`.
    pub unsafe fn reset_device(&self) -> Result<(), HipError> {
        self.api.check(unsafe { (self.api.device_reset)() })
    }

    /// Creates a non-blocking HIP stream.
    pub fn create_stream(self: &Arc<Self>) -> Result<HipStream, HipError> {
        let mut raw: *mut c_void = ptr::null_mut();
        self.api
            .check(unsafe { (self.api.stream_create_with_flags)(&mut raw, HIP_STREAM_NON_BLOCKING) })?;
        Ok(HipStream {
            ctx: Arc::clone(self),
            raw,
        })
    }

    /// Creates a HIP event. Set `timing = true` if you will call `elapsed_ms`.
    pub fn create_event(self: &Arc<Self>, timing: bool) -> Result<HipEvent, HipError> {
        let flags = if timing {
            HIP_EVENT_DEFAULT
        } else {
            HIP_EVENT_DISABLE_TIMING
        };
        let mut raw: *mut c_void = ptr::null_mut();
        self.api
            .check(unsafe { (self.api.event_create_with_flags)(&mut raw, flags) })?;
        Ok(HipEvent {
            ctx: Arc::clone(self),
            raw,
        })
    }

    /// Allocates `size` bytes of device VRAM (with small tail padding for safe funnel-shift reads).
    pub fn alloc_buffer(self: &Arc<Self>, size: usize) -> Result<HipBuffer, HipError> {
        let alloc_bytes = size.max(1).saturating_add(BUFFER_TAIL_PAD);
        let mut ptr: *mut c_void = ptr::null_mut();
        self.api.check(unsafe { (self.api.malloc)(&mut ptr, alloc_bytes) })?;
        Ok(HipBuffer {
            ctx: Arc::clone(self),
            ptr,
            size,
        })
    }

    /// Allocates a device buffer and synchronously uploads `data`.
    pub fn buffer_from_slice<T: bytemuck::Pod>(
        self: &Arc<Self>,
        data: &[T],
    ) -> Result<HipBuffer, HipError> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let buf = self.alloc_buffer(bytes.len())?;
        if !bytes.is_empty() {
            buf.upload(bytes)?;
        }
        Ok(buf)
    }

    /// Allocates `size` bytes of page-locked, device-mapped host memory (pinned GTT).
    pub fn alloc_pinned(self: &Arc<Self>, size: usize) -> Result<HipPinnedBuffer, HipError> {
        let alloc_bytes = size.max(1);
        let mut host_ptr: *mut c_void = ptr::null_mut();
        self.api.check(unsafe {
            (self.api.host_malloc)(&mut host_ptr, alloc_bytes, HIP_HOST_MALLOC_MAPPED)
        })?;
        let mut dev_ptr: *mut c_void = ptr::null_mut();
        if let Err(e) = self
            .api
            .check(unsafe { (self.api.host_get_device_ptr)(&mut dev_ptr, host_ptr, 0) })
        {
            let _ = unsafe { (self.api.host_free)(host_ptr) };
            return Err(e);
        }
        Ok(HipPinnedBuffer {
            ctx: Arc::clone(self),
            host_ptr,
            dev_ptr,
            size,
        })
    }

    /// Loads an HSACO binary into a `HipModule` (unloaded automatically on drop).
    pub fn load_module(self: &Arc<Self>, hsaco: &[u8]) -> Result<HipModule, HipError> {
        if hsaco.is_empty() {
            return Err(HipError::InvalidArg("empty HSACO binary".into()));
        }
        let mut module: *mut c_void = ptr::null_mut();
        self.api.check(unsafe {
            (self.api.module_load_data)(&mut module, hsaco.as_ptr() as *const c_void)
        })?;
        Ok(HipModule {
            ctx: Arc::clone(self),
            raw: module,
        })
    }

    /// Compiles HIP C++ source with `libhiprtc` (cached on disk by arch + options + source hash)
    /// and loads the resulting HSACO into a `HipModule`.
    pub fn compile_module(
        self: &Arc<Self>,
        source: &str,
        unit_name: &str,
        extra_opts: &[&str],
    ) -> Result<HipModule, HipError> {
        let rtc = self.rtc()?;
        let hsaco = rtc.compile_cached(source, unit_name, &self.info.gfx_arch, extra_opts)?;
        self.load_module(&hsaco)
    }

    fn rtc(&self) -> Result<&Hiprtc, HipError> {
        let res = self.rtc.get_or_init(|| Hiprtc::load().map_err(|e| e.to_string()));
        match res {
            Ok(r) => Ok(r),
            Err(_) => Err(HipError::RtcUnavailable),
        }
    }
}

fn query_device_info(api: &HipApi, device_id: i32) -> Result<DeviceInfo, HipError> {
    let mut name_buf = [0u8; 256];
    api.check(unsafe {
        (api.device_get_name)(
            name_buf.as_mut_ptr() as *mut c_char,
            name_buf.len() as c_int,
            device_id,
        )
    })?;
    let name = unsafe { CStr::from_ptr(name_buf.as_ptr() as *const c_char) }
        .to_string_lossy()
        .into_owned();

    let attr = |a: c_int| -> i32 {
        let mut v: c_int = 0;
        let rc = unsafe { (api.device_get_attr)(&mut v, a, device_id) };
        if rc == 0 { v } else { 0 }
    };

    let mut gcn_arch_name = String::new();
    if let Some(get_props) = api.get_device_props_r0600 {
        let mut raw = vec![0u8; HIP_PROP_R0600_MIN_SIZE];
        let rc = unsafe { get_props(raw.as_mut_ptr() as *mut c_void, device_id) };
        if rc == 0 {
            let slice = &raw[HIP_PROP_R0600_GCN_ARCH_OFFSET
                ..HIP_PROP_R0600_GCN_ARCH_OFFSET + HIP_PROP_R0600_GCN_ARCH_LEN];
            if let Some(nul) = slice.iter().position(|&b| b == 0) {
                gcn_arch_name = String::from_utf8_lossy(&slice[..nul]).into_owned();
            }
        }
    }
    let gfx_arch = gcn_arch_name
        .split(':')
        .next()
        .unwrap_or("")
        .to_string();

    let mut free_vram_bytes: usize = 0;
    let mut total_vram_bytes: usize = 0;
    let _ = unsafe { (api.mem_get_info)(&mut free_vram_bytes, &mut total_vram_bytes) };

    Ok(DeviceInfo {
        device_id,
        name,
        gcn_arch_name,
        gfx_arch,
        warp_size: attr(HIP_ATTR_WARP_SIZE),
        cu_count: attr(HIP_ATTR_MULTIPROCESSOR_COUNT),
        max_shared_mem_per_block: attr(HIP_ATTR_MAX_SHARED_MEM_PER_BLOCK),
        clock_khz: attr(HIP_ATTR_CLOCK_RATE),
        mem_clock_khz: attr(HIP_ATTR_MEM_CLOCK_RATE),
        mem_bus_width_bits: attr(HIP_ATTR_MEM_BUS_WIDTH),
        l2_cache_bytes: attr(HIP_ATTR_L2_CACHE_SIZE),
        total_vram_bytes,
        free_vram_bytes,
    })
}

/// Device-resident VRAM buffer (`hipMalloc` / `hipFree`).
pub struct HipBuffer {
    ctx: Arc<HipContext>,
    ptr: *mut c_void,
    size: usize,
}

unsafe impl Send for HipBuffer {}
unsafe impl Sync for HipBuffer {}

impl HipBuffer {
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_device_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Byte-offset device pointer (bounds-checked against `self.size`).
    pub fn device_ptr_at(&self, byte_offset: usize) -> Result<*mut c_void, HipError> {
        if byte_offset > self.size {
            return Err(HipError::InvalidArg(format!(
                "offset {byte_offset} > buffer size {}",
                self.size
            )));
        }
        Ok(unsafe { (self.ptr as *mut u8).add(byte_offset) as *mut c_void })
    }

    pub fn upload(&self, src: &[u8]) -> Result<(), HipError> {
        self.upload_at(0, src)
    }

    pub fn upload_at(&self, byte_offset: usize, src: &[u8]) -> Result<(), HipError> {
        let end = byte_offset
            .checked_add(src.len())
            .ok_or_else(|| HipError::InvalidArg("upload range overflow".into()))?;
        if end > self.size {
            return Err(HipError::InvalidArg(format!(
                "upload {byte_offset}..{end} exceeds buffer size {}",
                self.size
            )));
        }
        if src.is_empty() {
            return Ok(());
        }
        let dst = self.device_ptr_at(byte_offset)?;
        self.ctx.api.check(unsafe {
            (self.ctx.api.memcpy)(dst, src.as_ptr() as *const c_void, src.len(), MemcpyKind::H2D.raw())
        })
    }

    pub fn download(&self, dst: &mut [u8]) -> Result<(), HipError> {
        self.download_at(0, dst)
    }

    pub fn download_at(&self, byte_offset: usize, dst: &mut [u8]) -> Result<(), HipError> {
        let end = byte_offset
            .checked_add(dst.len())
            .ok_or_else(|| HipError::InvalidArg("download range overflow".into()))?;
        if end > self.size {
            return Err(HipError::InvalidArg(format!(
                "download {byte_offset}..{end} exceeds buffer size {}",
                self.size
            )));
        }
        if dst.is_empty() {
            return Ok(());
        }
        let src = self.device_ptr_at(byte_offset)?;
        self.ctx.api.check(unsafe {
            (self.ctx.api.memcpy)(
                dst.as_mut_ptr() as *mut c_void,
                src as *const c_void,
                dst.len(),
                MemcpyKind::D2H.raw(),
            )
        })
    }

    /// Asynchronously uploads from a pinned host buffer on `stream`.
    pub fn upload_pinned_async(
        &self,
        byte_offset: usize,
        src: &HipPinnedBuffer,
        bytes: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        let end = byte_offset
            .checked_add(bytes)
            .ok_or_else(|| HipError::InvalidArg("upload range overflow".into()))?;
        if end > self.size || bytes > src.size() {
            return Err(HipError::InvalidArg("pinned upload out of bounds".into()));
        }
        if bytes == 0 {
            return Ok(());
        }
        let dst = self.device_ptr_at(byte_offset)?;
        self.ctx.api.check(unsafe {
            (self.ctx.api.memcpy_async)(
                dst,
                src.host_ptr as *const c_void,
                bytes,
                MemcpyKind::H2D.raw(),
                stream.raw,
            )
        })
    }

    /// Asynchronously downloads into a pinned host buffer on `stream`.
    pub fn download_pinned_async(
        &self,
        byte_offset: usize,
        dst: &mut HipPinnedBuffer,
        bytes: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        let end = byte_offset
            .checked_add(bytes)
            .ok_or_else(|| HipError::InvalidArg("download range overflow".into()))?;
        if end > self.size || bytes > dst.size() {
            return Err(HipError::InvalidArg("pinned download out of bounds".into()));
        }
        if bytes == 0 {
            return Ok(());
        }
        let src = self.device_ptr_at(byte_offset)?;
        self.ctx.api.check(unsafe {
            (self.ctx.api.memcpy_async)(
                dst.host_ptr,
                src as *const c_void,
                bytes,
                MemcpyKind::D2H.raw(),
                stream.raw,
            )
        })
    }

    /// Synchronously zeros the entire buffer.
    pub fn zero(&self) -> Result<(), HipError> {
        if self.size == 0 {
            return Ok(());
        }
        self.ctx
            .api
            .check(unsafe { (self.ctx.api.memset)(self.ptr, 0, self.size) })
    }

    /// Asynchronously zeros the entire buffer on `stream`.
    pub fn zero_async(&self, stream: &HipStream) -> Result<(), HipError> {
        if self.size == 0 {
            return Ok(());
        }
        self.ctx
            .api
            .check(unsafe { (self.ctx.api.memset_async)(self.ptr, 0, self.size, stream.raw) })
    }

    /// Asynchronously copies `bytes` from `src[src_offset..]` to `self[dst_offset..]` on `stream`.
    pub fn copy_from_at_async(
        &self,
        dst_offset: usize,
        src: &HipBuffer,
        src_offset: usize,
        bytes: usize,
        stream: &HipStream,
    ) -> Result<(), HipError> {
        let d_end = dst_offset
            .checked_add(bytes)
            .ok_or_else(|| HipError::InvalidArg("copy dst overflow".into()))?;
        let s_end = src_offset
            .checked_add(bytes)
            .ok_or_else(|| HipError::InvalidArg("copy src overflow".into()))?;
        if d_end > self.size || s_end > src.size {
            return Err(HipError::InvalidArg("D2D copy out of bounds".into()));
        }
        if bytes == 0 {
            return Ok(());
        }
        let dst_p = self.device_ptr_at(dst_offset)?;
        let src_p = src.device_ptr_at(src_offset)?;
        self.ctx.api.check(unsafe {
            (self.ctx.api.memcpy_async)(
                dst_p,
                src_p as *const c_void,
                bytes,
                MemcpyKind::D2D.raw(),
                stream.raw,
            )
        })
    }
}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { (self.ctx.api.free)(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

/// Page-locked, device-mapped host buffer (`hipHostMalloc` with `hipHostMallocMapped`).
pub struct HipPinnedBuffer {
    ctx: Arc<HipContext>,
    host_ptr: *mut c_void,
    dev_ptr: *mut c_void,
    size: usize,
}

unsafe impl Send for HipPinnedBuffer {}
unsafe impl Sync for HipPinnedBuffer {}

impl HipPinnedBuffer {
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_device_ptr(&self) -> *mut c_void {
        self.dev_ptr
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.host_ptr as *const u8, self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.host_ptr as *mut u8, self.size) }
    }
}

impl Drop for HipPinnedBuffer {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            let _ = unsafe { (self.ctx.api.host_free)(self.host_ptr) };
            self.host_ptr = ptr::null_mut();
            self.dev_ptr = ptr::null_mut();
        }
    }
}

/// Non-blocking HIP execution stream.
pub struct HipStream {
    ctx: Arc<HipContext>,
    pub(crate) raw: *mut c_void,
}

unsafe impl Send for HipStream {}

impl HipStream {
    pub fn synchronize(&self) -> Result<(), HipError> {
        self.ctx.api.check(unsafe { (self.ctx.api.stream_sync)(self.raw) })
    }

    pub fn wait_event(&self, event: &HipEvent) -> Result<(), HipError> {
        self.ctx
            .api
            .check(unsafe { (self.ctx.api.stream_wait_event)(self.raw, event.raw, 0) })
    }

    /// Captures all launches recorded inside `f` into a reusable `HipGraphExec`.
    pub fn capture_graph<F>(&self, f: F) -> Result<HipGraphExec, HipError>
    where
        F: FnOnce(&HipStream) -> Result<(), HipError>,
    {
        let (Some(begin), Some(end), Some(inst), Some(gdestroy)) = (
            self.ctx.api.stream_begin_capture,
            self.ctx.api.stream_end_capture,
            self.ctx.api.graph_instantiate,
            self.ctx.api.graph_destroy,
        ) else {
            return Err(HipError::Symbol("HIP graph APIs unavailable".into()));
        };
        self.ctx
            .api
            .check(unsafe { begin(self.raw, HIP_STREAM_CAPTURE_MODE_RELAXED) })?;
        let body_res = f(self);
        let mut graph: *mut c_void = ptr::null_mut();
        let end_rc = unsafe { end(self.raw, &mut graph) };
        if let Err(e) = body_res {
            if !graph.is_null() {
                let _ = unsafe { gdestroy(graph) };
            }
            return Err(e);
        }
        self.ctx.api.check(end_rc)?;
        let mut exec: *mut c_void = ptr::null_mut();
        let inst_rc = unsafe { inst(&mut exec, graph, ptr::null_mut(), ptr::null_mut(), 0) };
        let _ = unsafe { gdestroy(graph) };
        self.ctx.api.check(inst_rc)?;
        Ok(HipGraphExec {
            ctx: Arc::clone(&self.ctx),
            raw: exec,
        })
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { (self.ctx.api.stream_destroy)(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

/// HIP event for GPU-side timing and cross-stream synchronization.
pub struct HipEvent {
    ctx: Arc<HipContext>,
    pub(crate) raw: *mut c_void,
}

unsafe impl Send for HipEvent {}

impl HipEvent {
    pub fn record(&self, stream: &HipStream) -> Result<(), HipError> {
        self.ctx
            .api
            .check(unsafe { (self.ctx.api.event_record)(self.raw, stream.raw) })
    }

    pub fn synchronize(&self) -> Result<(), HipError> {
        self.ctx
            .api
            .check(unsafe { (self.ctx.api.event_sync)(self.raw) })
    }

    /// Elapsed milliseconds between `start` and `self` (both must be created with `timing = true`
    /// and `self` must have completed).
    pub fn elapsed_ms(&self, start: &HipEvent) -> Result<f32, HipError> {
        let mut ms: f32 = 0.0;
        self.ctx
            .api
            .check(unsafe { (self.ctx.api.event_elapsed_time)(&mut ms, start.raw, self.raw) })?;
        Ok(ms)
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { (self.ctx.api.event_destroy)(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

/// Executable HIP graph captured from a stream.
pub struct HipGraphExec {
    ctx: Arc<HipContext>,
    raw: *mut c_void,
}

unsafe impl Send for HipGraphExec {}

impl HipGraphExec {
    pub fn launch(&self, stream: &HipStream) -> Result<(), HipError> {
        let launch = self
            .ctx
            .api
            .graph_launch
            .ok_or_else(|| HipError::Symbol("hipGraphLaunch unavailable".into()))?;
        self.ctx.api.check(unsafe { launch(self.raw, stream.raw) })
    }
}

impl Drop for HipGraphExec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            if let Some(destroy) = self.ctx.api.graph_exec_destroy {
                let _ = unsafe { destroy(self.raw) };
            }
            self.raw = ptr::null_mut();
        }
    }
}

/// Loaded HSACO module (`hipModuleUnload` on drop).
pub struct HipModule {
    ctx: Arc<HipContext>,
    raw: *mut c_void,
}

unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}

impl HipModule {
    /// Resolves a kernel entry point by `extern "C"` name.
    pub fn function(self: &Arc<Self>, name: &str) -> Result<HipFunction, HipError> {
        let cname = CString::new(name).map_err(|e| HipError::Symbol(e.to_string()))?;
        let mut func: *mut c_void = ptr::null_mut();
        self.ctx.api.check(unsafe {
            (self.ctx.api.module_get_function)(&mut func, self.raw, cname.as_ptr())
        })?;
        Ok(HipFunction {
            _module: Arc::clone(self),
            ctx: Arc::clone(&self.ctx),
            raw: func,
        })
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { (self.ctx.api.module_unload)(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

/// Resolved kernel handle that keeps its parent `HipModule` alive.
pub struct HipFunction {
    _module: Arc<HipModule>,
    ctx: Arc<HipContext>,
    raw: *mut c_void,
}

unsafe impl Send for HipFunction {}
unsafe impl Sync for HipFunction {}

impl HipFunction {
    /// Launches the kernel on `stream`.
    ///
    /// # Safety
    /// `params` must point to valid argument storage matching the kernel's parameter list, and
    /// any device pointers passed must remain alive until `stream` completes.
    pub unsafe fn launch(
        &self,
        stream: &HipStream,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), HipError> {
        let rc = unsafe {
            (self.ctx.api.module_launch_kernel)(
                self.raw,
                grid.0 as c_uint,
                grid.1 as c_uint,
                grid.2 as c_uint,
                block.0 as c_uint,
                block.1 as c_uint,
                block.2 as c_uint,
                shared_bytes as c_uint,
                stream.raw,
                params.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        self.ctx.api.check(rc)
    }
}
