use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let bindings = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .exclude(&["moe_*.cu"]) // Exclude moe kernels for ptx build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;

    bindings.write(&ptx_path)?;

    // cookd fork: the fused-MoE kernels (src/moe/*.cu) are not built. Upstream
    // compiles them into a static libmoe.a and links it with `-lcudart` and
    // `-lstdc++`, which import-links the CUDA runtime. With cudarc's
    // dynamic-loading the binary must start on a driver-only machine and
    // cudarc adds no CUDA library search path, so that link fails
    // (CyberDiner #666). The `ffi` module keeps the MoE entry points as stubs.
    Ok(())
}
