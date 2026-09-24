#![doc = include_str!("../README.md")]

mod cache;
mod error;

pub use crate::cache::Cache;
pub use crate::error::CacheError;
