use thiserror::Error;

#[derive(Error, Debug)]
#[error("Caching Error: {0}")]
/// An error encountered while locating, downloading, or clearing cached files.
pub struct CacheError(pub String);
