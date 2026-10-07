//! Fused-MoE entry points.
//!
//! cookd fork: the fused-MoE CUDA kernels (`src/moe/*.cu`) are not built or
//! linked, so the binary does not import-link the CUDA runtime (`cudart`) and
//! starts on a driver-only machine (CyberDiner #666). These stubs keep the
//! upstream signatures so `candle-nn` and `candle-transformers` compile
//! unchanged; calling one panics.
#![allow(
    clippy::too_many_arguments,
    reason = "mirrors the upstream fused-MoE kernel signatures"
)]
#![allow(
    clippy::missing_safety_doc,
    reason = "the stubs never touch their pointer arguments"
)]
#![allow(
    unused_variables,
    reason = "the stubs keep the upstream parameter names and use none of them"
)]

use core::ffi::c_void;

const MOE_NOT_BUILT: &str =
    "fused MoE kernels are not built into this candle fork (cookd); see CyberDiner #666";

// for unquntized models
pub unsafe fn moe_gemm_wmma(
    input: *const c_void,         // device pointer [size_m, size_k]
    weights: *const c_void,       // device pointer [num_experts, size_n, size_k]
    sorted_token_ids: *const i32, // device pointer [size_m]
    expert_ids: *const i32,       // host array [size_m] (expert id per sorted token)
    topk_weights: *const f32,
    output: *mut c_void,      // device pointer [size_m, size_n]
    expert_counts: *mut i32,  // pre-allocated buffer [num_experts]
    expert_offsets: *mut i32, // pre-allocated buffer [num_experts + 1]
    num_experts: i32,
    topk: i32,
    size_m: i32,
    size_n: i32,
    size_k: i32,
    dtype: i32, // 0=float16, 1=bf16 (for input/output)
    is_prefill: bool,
    stream: i64,
) {
    panic!("{MOE_NOT_BUILT}")
}

pub unsafe fn moe_gemm_gguf(
    input: *const f32,      // input [size_m, size_k]
    weights: *const c_void, // weights [num_experts, size_n, size_k]
    sorted_token_ids: *const i32,
    expert_ids: *const i32,
    topk_weights: *const f32, // device ptr or nullptr
    output: *mut c_void,      // float output [size_m, size_n]
    num_experts: i32,
    topk: i32,
    size_m: i32,
    size_n: i32,
    size_k: i32,
    gguf_dtype: i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
    stream: i64,
) {
    panic!("{MOE_NOT_BUILT}")
}

pub unsafe fn moe_gemm_gguf_prefill(
    input: *const c_void, // input [size_m, size_k]
    weights: *const u8,   // weights [num_experts, size_n, size_k]
    sorted_token_ids: *const i32,
    expert_ids: *const i32,   //must be host ptr
    topk_weights: *const f32, // device ptr or nullptr
    output: *mut c_void,      // float output [size_m, size_n]
    num_experts: i32,
    topk: i32,
    size_m: i32,
    size_n: i32,
    size_k: i32,
    input_dtype: i32, // 0=f16, 1=bf16 (for inputs)
    gguf_dtype: i32,  //Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
    stream: i64,
) {
    panic!("{MOE_NOT_BUILT}")
}
