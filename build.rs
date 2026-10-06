//! Compile le kernel CUDA en PTX au build time et lie l'API Driver CUDA.
//!
//! Si `nvcc` est absent, on build sans GPU (le mineur reste en CPU).

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(cuda)");
    println!("cargo:rerun-if-changed=cuda/miner.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-changed=build.rs");

    let nvcc = env::var("NVCC").unwrap_or_else(|_| {
        for cand in ["/opt/cuda/bin/nvcc", "/usr/local/cuda/bin/nvcc", "nvcc"] {
            if cand == "nvcc" || PathBuf::from(cand).exists() {
                return cand.to_string();
            }
        }
        "nvcc".to_string()
    });

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR");
    let ptx = format!("{out_dir}/miner.ptx");

    let status = Command::new(&nvcc)
        .args(["-O3", "-ptx", "cuda/miner.cu", "-o", &ptx])
        .status();

    match status {
        Ok(s) if s.success() => {
            // L'API Driver CUDA (libcuda.so) est livrée par le driver NVIDIA.
            println!("cargo:rustc-cfg=cuda");
            println!("cargo:rustc-link-lib=dylib=cuda");
        }
        Ok(s) => println!(
            "cargo:warning=nvcc ({nvcc}) a échoué (code {s}) : build CPU uniquement"
        ),
        Err(e) => println!("cargo:warning=nvcc introuvable ({e}) : build CPU uniquement"),
    }
}
