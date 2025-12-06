//! CUDA Code Generator
//!
//! Generates optimized CUDA code from MIR for GPU execution.
//! Supports tensor cores (WMMA), shared memory optimization, and
//! automatic kernel fusion.
//!
//! # Architecture
//!
//! ```text
//! MIR Function
//!      │
//!      ▼
//! ┌──────────────┐
//! │ Kernel       │  Analyze parallelism, determine launch config
//! │ Analysis     │
//! └──────────────┘
//!      │
//!      ▼
//! ┌──────────────┐
//! │ Memory       │  Plan shared memory, coalescing, bank conflicts
//! │ Planning     │
//! └──────────────┘
//!      │
//!      ▼
//! ┌──────────────┐
//! │ PTX/CUDA     │  Generate device code
//! │ Generation   │
//! └──────────────┘
//!      │
//!      ▼
//! ┌──────────────┐
//! │ Host Code    │  Generate kernel launch code
//! │ Generation   │
//! └──────────────┘
//! ```

pub mod codegen;
pub mod kernel;
pub mod memory;
pub mod tensor_core;
pub mod types;

pub use codegen::{CudaCodegen, CudaError, CudaModule, CudaResult};
pub use kernel::{KernelConfig, KernelInfo, LaunchBounds};
pub use memory::{MemoryPlan, SharedMemoryAlloc};
pub use tensor_core::{TensorCoreConfig, WmmaFragment, WmmaOp};
pub use types::{CudaType, GpuArch};
