# soccer-cache

`soccer-cache` provides shared model-download and cache-management support for
the SOCcer ecosystem. It handles ordinary HTTP downloads and files hosted in
Hugging Face model repositories without depending on ONNX Runtime.

## Cache locations

Files downloaded from ordinary URLs are stored in a `soccernet` directory under
the platform's user cache directory, as selected by the [`dirs`](https://crates.io/crates/dirs)
crate. Hugging Face files use the standard cache managed by
[`hf-hub`](https://crates.io/crates/hf-hub).

HTTP downloads are first written to a temporary file. The final cache filename
is created only after a successful response and complete download, preventing
an interrupted transfer from leaving a partial file that appears cached.

## Usage

```rust,no_run
use soccer_cache::{Cache, CacheError};

fn main() -> Result<(), CacheError> {
    let cache = Cache::new()?;

    let classifier = cache.get_from_url(
        "https://danielruss.github.io/soccer-models/SOCcer_v3.0.0.onnx",
    )?;

    let embedding = cache.get_embedding_model_path(
        "Xenova/GIST-small-Embedding-v0",
        None,
    )?;

    println!("classifier: {}", classifier.display());
    println!("embedding: {}", embedding.display());
    Ok(())
}
```

To cache the default configuration, tokenizer, and ONNX embedding model in one
operation:

```rust,no_run
# use soccer_cache::{Cache, CacheError};
# fn example() -> Result<(), CacheError> {
let cache = Cache::new()?;
let snapshot = cache.cache_hf_model("Xenova/GIST-small-Embedding-v0")?;
println!("snapshot: {}", snapshot.display());
# Ok(())
# }
```

## Clearing cached files

```rust,no_run
use soccer_cache::{Cache, CacheError};

fn clear() -> Result<(), CacheError> {
    Cache::clear_cache("Xenova/GIST-small-Embedding-v0")
}
```

`clear_cache` removes both the selected Hugging Face model repository and the
entire platform-specific `soccernet` cache directory. Missing cache directories
are not errors.

## License

Licensed under the MIT License.
