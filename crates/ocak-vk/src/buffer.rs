//! Device memory: buffer allocation and `Buffer`.
use ash::{vk, Device};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use crate::*;

pub(crate) unsafe fn find_mem_type(
    mp: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..mp.memory_type_count).find(|&i| {
        (type_bits & (1 << i)) != 0 && mp.memory_types[i as usize].property_flags.contains(flags)
    })
}

pub(crate) unsafe fn alloc_raw_buffer(
    dev: &Device,
    mp: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
    flags: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), VkError> {
    // 16 bytes of tail padding: shaders read packed bytes as 32-bit words and may touch the word
    // after the last byte they use (e.g. `bytes4` in dequant_i8.glsl at the end of a tensor).
    // Descriptors bind WHOLE_SIZE, so the padding is in range.
    let buf = dev.create_buffer(
        &vk::BufferCreateInfo::default()
            .size(size + 16)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE),
        None,
    )?;
    let req = dev.get_buffer_memory_requirements(buf);
    let Some(mem_type) = find_mem_type(mp, req.memory_type_bits, flags) else {
        dev.destroy_buffer(buf, None);
        return Err(VkError::NoMemoryType);
    };
    // Out-of-memory here is routine (VRAM -> GTT fallback), so release the buffer on failure.
    let mem = match dev.allocate_memory(
        &vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type),
        None,
    ) {
        Ok(m) => m,
        Err(e) => {
            dev.destroy_buffer(buf, None);
            return Err(e.into());
        }
    };
    if let Err(e) = dev.bind_buffer_memory(buf, mem, 0) {
        dev.destroy_buffer(buf, None);
        dev.free_memory(mem, None);
        return Err(e.into());
    }
    Ok((buf, mem))
}

pub struct Buffer {
    pub(crate) ctx: Arc<VkContext>,
    pub(crate) buf: vk::Buffer,
    pub(crate) mem: vk::DeviceMemory,
    pub(crate) size: u64,
}

impl Buffer {
    pub fn size(&self) -> u64 { self.size }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // An `upload_async` copy into this buffer may still be in flight (e.g. a model load that
        // failed part-way): wait for it before the memory is freed.
        if self.ctx.async_pending.load(Ordering::Acquire) {
            let mut exec = self.ctx.exec.lock().unwrap_or_else(|e| e.into_inner());
            self.ctx.drain_async(&mut exec);
        }
        unsafe {
            self.ctx.device.destroy_buffer(self.buf, None);
            self.ctx.device.free_memory(self.mem, None);
        }
    }
}
