//! HTTP handlers, grouped by API surface.

mod admin;
mod anthropic;
mod audio;
mod openai;

pub(crate) use admin::*;
pub(crate) use anthropic::*;
pub(crate) use audio::*;
pub(crate) use openai::*;
