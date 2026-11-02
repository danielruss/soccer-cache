use std::path::PathBuf;

use hf_hub::{
    Cache as HFCache, Repo,
    api::sync::{Api, ApiBuilder},
};
use reqwest::blocking::Client;
use rustls::crypto::{CryptoProvider, ring};
use tempfile::NamedTempFile;

use crate::CacheError;

/// Downloads and locates files used by SOCcer applications.
///
/// A `Cache` owns the synchronous Hugging Face client used by the model helper
/// methods. Files downloaded from ordinary URLs are stored in the platform's
/// user cache directory under `soccernet`.
pub struct Cache {
    hf_api: Api,
}

impl Cache {
    /// Creates a cache client.
    ///
    /// This initializes the process-wide Rustls crypto provider when necessary
    /// and creates the synchronous Hugging Face API client.
    pub fn new() -> Result<Self, CacheError> {
        Self::ensure_crypto_provider();
        let hf_api = ApiBuilder::from_env()
            .build()
            .map_err(Self::make_cache_error)?;
        Ok(Self { hf_api })
    }

    fn ensure_crypto_provider() {
        if CryptoProvider::get_default().is_none() {
            let _ = ring::default_provider().install_default();
        }
    }

    /// Returns `true` when `value` begins with `http://` or `https://`.
    pub fn is_url<S: AsRef<str>>(value: S) -> bool {
        let url_str = value.as_ref();
        url_str.starts_with("https://") || url_str.starts_with("http://")
    }

    fn make_cache_error<E: ToString>(err: E) -> CacheError {
        CacheError(err.to_string())
    }

    // A private function for getting the cache dir.
    fn get_cache_dir() -> Result<PathBuf, CacheError> {
        let mut cache_dir =
            dirs::cache_dir().ok_or_else(|| CacheError("Cache Dir is not defined".to_string()))?;
        cache_dir.push("soccernet");
        std::fs::create_dir_all(&cache_dir).map_err(Cache::make_cache_error)?;

        Ok(cache_dir)
    }

    fn get_cached_filename(url: &str) -> Result<PathBuf, CacheError> {
        let filename = url
            .split("/")
            .last()
            .ok_or_else(|| Cache::make_cache_error("malformed URL"))?;
        Ok(Cache::get_cache_dir()?.join(filename))
    }

    /// Returns the cached file for `url`, downloading it when necessary.
    ///
    /// The final URL path segment is used as the filename in the platform's
    /// `soccernet` cache directory. Downloads are written to a temporary file
    /// and moved into place only after the HTTP response completes successfully,
    /// so an interrupted download is not treated as cached on the next call.
    pub fn get_from_url(&self, url: &str) -> Result<PathBuf, CacheError> {
        let cached_file = Cache::get_cached_filename(url)?;
        if !cached_file.exists() {
            let client = Client::new();
            let mut response = client
                .get(url)
                .send()
                .map_err(Cache::make_cache_error)?
                .error_for_status()
                .map_err(Cache::make_cache_error)?;

            let parent = cached_file
                .parent()
                .ok_or_else(|| CacheError("cache file has no parent".to_string()))?;

            let mut temp_file = NamedTempFile::new_in(parent).map_err(Cache::make_cache_error)?;

            std::io::copy(&mut response, &mut temp_file).map_err(Cache::make_cache_error)?;

            temp_file
                .persist(&cached_file)
                .map_err(Cache::make_cache_error)?;
        }

        Ok(cached_file)
    }

    /// Returns a file from a Hugging Face model repository.
    ///
    /// The file is downloaded through `hf-hub` when it is not already present
    /// in the Hugging Face cache.
    pub fn get_from_hf_hub(&self, model_id: &str, filename: &str) -> Result<PathBuf, CacheError> {
        self.hf_api
            .model(model_id.to_string())
            .get(filename)
            .map_err(Self::make_cache_error)
    }

    /// Returns the requested embedding model file from a Hugging Face model.
    ///
    /// When `filename` is `None`, this uses `onnx/model.onnx`.
    pub fn get_embedding_model_path(
        &self,
        model_id: &str,
        filename: Option<&str>,
    ) -> Result<PathBuf, CacheError> {
        // this could change in the future
        let model_path: &str = filename.unwrap_or("onnx/model.onnx");
        self.get_from_hf_hub(model_id, model_path)
    }
    /// Returns the requested tokenizer file from a Hugging Face model.
    ///
    /// When `filename` is `None`, this uses `tokenizer.json`.
    pub fn get_tokenizer_path(
        &self,
        model_id: &str,
        filename: Option<&str>,
    ) -> Result<PathBuf, CacheError> {
        // this could change in the future
        let tokenizer_path: &str = filename.unwrap_or("tokenizer.json");
        self.get_from_hf_hub(model_id, tokenizer_path)
    }

    /// Returns `config.json` from a Hugging Face model repository.
    pub fn get_config_path(&self, model_id: &str) -> Result<PathBuf, CacheError> {
        self.get_from_hf_hub(model_id, "config.json")
    }

    /// Caches the configuration, tokenizer, and default ONNX model.
    ///
    /// On success, this returns the Hugging Face snapshot directory containing
    /// `config.json`, `tokenizer.json`, and `onnx/model.onnx`.
    pub fn cache_hf_model(&self, model_id: &str) -> Result<PathBuf, CacheError> {
        let config = self.get_from_hf_hub(model_id, "config.json")?;
        self.get_from_hf_hub(model_id, "tokenizer.json")?;
        self.get_from_hf_hub(model_id, "onnx/model.onnx")?;

        config
            .parent()
            .map(|p| p.to_path_buf())
            .ok_or_else(|| CacheError(format!("config file has no parent {}", config.display())))
    }

    /// Checks whether the default ONNX model and tokenizer are present in the
    /// local Hugging Face cache. This does not perform a network request.
    pub fn is_cached(model_id: &str) -> bool {
        let repo = HFCache::from_env().model(model_id.to_string());

        repo.get("onnx/model.onnx").is_some()
            && repo.get("tokenizer.json").is_some()
            && repo.get("config.json").is_some()
    }

    /// Checks whether a named file exists in the `soccernet` cache directory.
    ///
    /// This does not create the cache directory or perform a network request.
    pub fn is_soccer_cached(model_name: &str) -> bool {
        dirs::cache_dir()
            .map(|p| p.join("soccernet").join(model_name).is_file())
            .unwrap_or(false)
    }

    /// Removes cached SOCcer files and one Hugging Face model repository.
    ///
    /// This removes the Hugging Face cache directory for `model_id` and the
    /// entire platform-specific `soccernet` cache directory. A missing directory
    /// is treated as success. If both removals fail, the returned error describes
    /// both failures.
    pub fn clear_cache(model_id: &str) -> Result<(), CacheError> {
        // only remove the GIST-Embedding
        let hf_cache = HFCache::from_env();
        let repo = Repo::model(model_id.to_string());
        let repo_dir = hf_cache.path().join(repo.folder_name());
        let del1_res = if repo_dir.exists() {
            std::fs::remove_dir_all(repo_dir).map_err(Self::make_cache_error)
        } else {
            Ok(())
        };

        // remove the soccernet/onnx models...
        let cache_dir = dirs::cache_dir()
            // if the cache dir does not exist
            .map(|p| p.join("soccernet"));

        let del2_res = cache_dir
            .map(|p| {
                if p.exists() {
                    std::fs::remove_dir_all(p).map_err(Self::make_cache_error)
                } else {
                    Ok(())
                }
            })
            .unwrap_or(Ok(()));

        match (del1_res, del2_res) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(hf_error), Err(soccer_error)) => Err(CacheError(format!(
                "Failed to clear Hugging Face cache: {}; \
                    failed to clear SOCcerNET cache: {}",
                hf_error.0, soccer_error.0,
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Cache, CacheError};

    #[test]
    fn test_is_cached() -> Result<(), CacheError> {
        let model_id = "Xenova/GIST-small-Embedding-v0";
        println!("is {} cached: {}", model_id, Cache::is_cached(model_id));

        let sn_cached = Cache::is_soccer_cached("SOCcer_v3.0.0.onnx");
        let c_cached = Cache::is_soccer_cached("clips_v0.0.2.onnx");
        println!(
            "is SoccerNET cached: {}\nis CLIPS cached: {}",
            sn_cached, c_cached
        );

        Ok(())
    }
}
