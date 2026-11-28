//! Ground-up Vulkan 1.3/1.4 compute backend (`ash`) targeting RDNA 2 (`gfx1031`).

mod buffer;
mod context;
mod pipelines;
mod profile;
mod recorder;
mod select;
mod wtype;
#[cfg(test)]
mod tests;

pub use buffer::*;
pub use context::*;
pub(crate) use pipelines::*;
pub use profile::*;
pub use recorder::*;
pub use select::*;
pub use wtype::*;

use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum VkError {
    #[error("vulkan load error: {0}")]
    Load(String),
    #[error("vulkan error: {0:?}")]
    Vk(vk::Result),
    #[error("no suitable Vulkan compute device found")]
    NoDevice,
    #[error("no suitable memory type")]
    NoMemoryType,
    #[error("unsupported weight type: {0}")]
    UnsupportedType(String),
}

impl From<vk::Result> for VkError {
    fn from(r: vk::Result) -> Self {
        VkError::Vk(r)
    }
}
