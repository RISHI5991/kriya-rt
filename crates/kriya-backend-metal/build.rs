// build.rs — Embed Metal shader source into the binary at compile time.
// This allows the KriyaRT Metal backend to ship as a self-contained binary
// without requiring the kernels/metal/ directory on disk at runtime.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    // Locate the kernels/metal directory relative to this crate's manifest.
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernels")
        .join("metal");

    let shader_files = [
        "q4_gemm_and_moe.metal",
        "paged_gqa_and_deltanet.metal",
        "transformer_glue.metal",
    ];

    let mut shader_sources = Vec::new();
    for file in &shader_files {
        let path = kernels_dir.join(file);
        if !path.exists() {
            panic!("Missing Metal shader: {:?}", path);
        }
        let src = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", path, e));
        shader_sources.push((file.to_string(), src));
    }

    // Emit a Rust module with embedded shader sources.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out_path = out_dir.join("embedded_shaders.rs");

    let mut code = String::new();
    for (name, src) in &shader_sources {
        let var_name = name
            .replace('.', "_")
            .replace('-', "_")
            .to_uppercase();
        code.push_str(&format!(
            "pub const {}: &str = r#\"{}\"#;\n\n",
            var_name, src
        ));
    }

    fs::write(&out_path, code).expect("Failed to write embedded_shaders.rs");

    // Re-run build if any shader changes.
    for file in &shader_files {
        println!("cargo:rerun-if-changed={}", kernels_dir.join(file).display());
    }
    println!("cargo:rerun-if-changed=build.rs");
}
