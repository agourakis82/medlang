//! CUDA Kernel Configuration and Analysis
//!
//! Analyzes MIR functions to determine optimal kernel launch configurations,
//! including grid/block dimensions, shared memory usage, and register pressure.

use std::collections::HashMap;

use super::types::{CudaType, GpuArch};
use crate::mir::function::MirFunction;
use crate::mir::inst::Operation;
use crate::mir::types::MirType;
use crate::mir::value::ValueId;

/// Kernel launch bounds specification
#[derive(Clone, Debug, PartialEq)]
pub struct LaunchBounds {
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Minimum blocks per SM (optional hint)
    pub min_blocks_per_sm: Option<u32>,
    /// Maximum registers per thread (optional limit)
    pub max_registers_per_thread: Option<u32>,
}

impl LaunchBounds {
    pub fn new(max_threads: u32) -> Self {
        Self {
            max_threads_per_block: max_threads,
            min_blocks_per_sm: None,
            max_registers_per_thread: None,
        }
    }

    pub fn with_min_blocks(mut self, min_blocks: u32) -> Self {
        self.min_blocks_per_sm = Some(min_blocks);
        self
    }

    pub fn with_max_registers(mut self, max_regs: u32) -> Self {
        self.max_registers_per_thread = Some(max_regs);
        self
    }

    /// Generate __launch_bounds__ attribute
    pub fn to_cuda_attribute(&self) -> String {
        match self.min_blocks_per_sm {
            Some(min) => format!("__launch_bounds__({}, {})", self.max_threads_per_block, min),
            None => format!("__launch_bounds__({})", self.max_threads_per_block),
        }
    }
}

impl Default for LaunchBounds {
    fn default() -> Self {
        Self::new(256)
    }
}

/// Kernel execution configuration
#[derive(Clone, Debug)]
pub struct KernelConfig {
    /// Grid dimensions (blocks)
    pub grid_dim: Dim3,
    /// Block dimensions (threads per block)
    pub block_dim: Dim3,
    /// Dynamic shared memory size in bytes
    pub shared_mem_bytes: usize,
    /// CUDA stream (0 = default)
    pub stream: u64,
    /// Launch bounds
    pub launch_bounds: LaunchBounds,
}

impl KernelConfig {
    pub fn new(grid: Dim3, block: Dim3) -> Self {
        Self {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
            stream: 0,
            launch_bounds: LaunchBounds::default(),
        }
    }

    /// Total number of threads
    pub fn total_threads(&self) -> u64 {
        self.grid_dim.total() * self.block_dim.total()
    }

    /// Total number of blocks
    pub fn total_blocks(&self) -> u64 {
        self.grid_dim.total()
    }

    /// Threads per block
    pub fn threads_per_block(&self) -> u64 {
        self.block_dim.total()
    }

    /// Generate kernel launch syntax
    pub fn launch_syntax(&self) -> String {
        format!(
            "<<<dim3({}, {}, {}), dim3({}, {}, {}), {}, {}>>>",
            self.grid_dim.x,
            self.grid_dim.y,
            self.grid_dim.z,
            self.block_dim.x,
            self.block_dim.y,
            self.block_dim.z,
            self.shared_mem_bytes,
            self.stream
        )
    }
}

/// 3D dimension (for grid and block)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub fn d1(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    pub fn d2(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    pub fn total(&self) -> u64 {
        self.x as u64 * self.y as u64 * self.z as u64
    }
}

/// Kernel parallelism pattern
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParallelismPattern {
    /// Element-wise operation (map)
    ElementWise { size: usize },
    /// Reduction operation
    Reduction { size: usize, op: ReductionOp },
    /// Scan/prefix sum
    Scan { size: usize, inclusive: bool },
    /// Matrix multiplication
    MatMul { m: usize, n: usize, k: usize },
    /// Batched matrix multiplication
    BatchedMatMul {
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
    },
    /// Convolution
    Convolution {
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        spatial: Vec<usize>,
        kernel: Vec<usize>,
    },
    /// Stencil computation
    Stencil { shape: Vec<usize>, radius: usize },
    /// Custom/unanalyzed
    Custom,
}

/// Reduction operation type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionOp {
    Sum,
    Product,
    Min,
    Max,
    And,
    Or,
    Xor,
}

/// Information about a CUDA kernel
#[derive(Clone, Debug)]
pub struct KernelInfo {
    /// Kernel name
    pub name: String,
    /// Parameter types
    pub parameters: Vec<(String, CudaType)>,
    /// Return type (usually void for kernels)
    pub return_type: CudaType,
    /// Detected parallelism pattern
    pub pattern: ParallelismPattern,
    /// Launch bounds
    pub launch_bounds: LaunchBounds,
    /// Estimated register usage per thread
    pub estimated_registers: u32,
    /// Static shared memory usage
    pub static_shared_memory: usize,
    /// Uses tensor cores
    pub uses_tensor_cores: bool,
    /// Requires cooperative launch
    pub cooperative_launch: bool,
    /// Attributes (e.g., __noinline__, __forceinline__)
    pub attributes: Vec<String>,
}

impl KernelInfo {
    pub fn new(name: String) -> Self {
        Self {
            name,
            parameters: Vec::new(),
            return_type: CudaType::Void,
            pattern: ParallelismPattern::Custom,
            launch_bounds: LaunchBounds::default(),
            estimated_registers: 32,
            static_shared_memory: 0,
            uses_tensor_cores: false,
            cooperative_launch: false,
            attributes: Vec::new(),
        }
    }

    /// Generate kernel signature
    pub fn signature(&self) -> String {
        let params: Vec<String> = self
            .parameters
            .iter()
            .map(|(name, ty)| format!("{} {}", ty.cuda_name(), name))
            .collect();

        let attrs = if self.attributes.is_empty() {
            String::new()
        } else {
            format!("{} ", self.attributes.join(" "))
        };

        format!(
            "{}__global__ {} void {}({})",
            attrs,
            self.launch_bounds.to_cuda_attribute(),
            self.name,
            params.join(", ")
        )
    }
}

/// Kernel analyzer for MIR functions
pub struct KernelAnalyzer {
    arch: GpuArch,
}

impl KernelAnalyzer {
    pub fn new(arch: GpuArch) -> Self {
        Self { arch }
    }

    /// Analyze a MIR function for kernel generation
    pub fn analyze(&self, func: &MirFunction) -> KernelAnalysis {
        let mut analysis = KernelAnalysis::new(func.name.clone());

        // Analyze parameters
        for (i, param_ty) in func.signature.params.iter().enumerate() {
            let param_name = func
                .signature
                .param_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("arg{}", i));
            analysis
                .parameters
                .push((param_name, CudaType::from_mir(param_ty)));
        }

        // Analyze operations
        for block in &func.blocks {
            for inst in &block.instructions {
                self.analyze_instruction(&inst.op, &mut analysis);
            }
        }

        // Detect parallelism pattern
        analysis.pattern = self.detect_pattern(func, &analysis);

        // Compute optimal configuration
        analysis.suggested_config = self.compute_config(&analysis);

        analysis
    }

    /// Analyze a single instruction
    fn analyze_instruction(&self, op: &Operation, analysis: &mut KernelAnalysis) {
        match op {
            // Memory operations
            Operation::Load { .. } => {
                analysis.global_loads += 1;
            }
            Operation::Store { .. } => {
                analysis.global_stores += 1;
            }

            // Arithmetic operations
            Operation::FAdd { .. }
            | Operation::FSub { .. }
            | Operation::IAdd { .. }
            | Operation::ISub { .. } => {
                analysis.arithmetic_ops += 1;
            }
            Operation::FMul { .. } | Operation::IMul { .. } => {
                analysis.arithmetic_ops += 1;
            }
            Operation::FDiv { .. } | Operation::IDiv { .. } => {
                analysis.arithmetic_ops += 4; // Division is expensive
            }
            Operation::FMA { .. } => {
                analysis.fma_ops += 1;
            }

            // Transcendental
            Operation::Sqrt { .. }
            | Operation::Exp { .. }
            | Operation::Log { .. }
            | Operation::Sin { .. }
            | Operation::Cos { .. }
            | Operation::Tanh { .. } => {
                analysis.transcendental_ops += 1;
            }

            // Synchronization
            Operation::Fence { .. } => {
                analysis.barriers += 1;
            }

            _ => {}
        }
    }

    /// Detect parallelism pattern from function analysis
    fn detect_pattern(&self, func: &MirFunction, analysis: &KernelAnalysis) -> ParallelismPattern {
        // Check for matrix multiplication pattern
        if analysis.uses_tensor_cores || analysis.fma_ops > 10 {
            // Look for loop structure suggesting matmul
            return ParallelismPattern::MatMul {
                m: 128,
                n: 128,
                k: 128,
            };
        }

        // Check for reduction
        if analysis.has_reduction {
            return ParallelismPattern::Reduction {
                size: 1024,
                op: ReductionOp::Sum,
            };
        }

        // Default to element-wise
        ParallelismPattern::ElementWise { size: 1024 }
    }

    /// Compute optimal kernel configuration
    fn compute_config(&self, analysis: &KernelAnalysis) -> KernelConfig {
        match &analysis.pattern {
            ParallelismPattern::ElementWise { size } => {
                let threads_per_block = 256;
                let blocks = (*size + threads_per_block - 1) / threads_per_block;
                KernelConfig::new(Dim3::d1(blocks as u32), Dim3::d1(threads_per_block as u32))
            }

            ParallelismPattern::Reduction { size, .. } => {
                let threads_per_block = 256;
                let blocks = (*size + threads_per_block * 2 - 1) / (threads_per_block * 2);
                let mut config =
                    KernelConfig::new(Dim3::d1(blocks as u32), Dim3::d1(threads_per_block as u32));
                config.shared_mem_bytes = threads_per_block * 8; // For reduction
                config
            }

            ParallelismPattern::MatMul { m, n, k } => {
                // Use 16x16 thread blocks for tensor cores
                let block_m = 16;
                let block_n = 16;
                let blocks_x = (*n + block_n - 1) / block_n;
                let blocks_y = (*m + block_m - 1) / block_m;

                let mut config = KernelConfig::new(
                    Dim3::d2(blocks_x as u32, blocks_y as u32),
                    Dim3::d2(block_n as u32, block_m as u32),
                );
                // Shared memory for A and B tiles
                config.shared_mem_bytes = 2 * block_m * 16 * 4; // FP32
                config.launch_bounds = LaunchBounds::new(256).with_min_blocks(2);
                config
            }

            ParallelismPattern::BatchedMatMul { batch, m, n, k } => {
                let block_m = 16;
                let block_n = 16;
                let blocks_x = (*n + block_n - 1) / block_n;
                let blocks_y = (*m + block_m - 1) / block_m;

                KernelConfig::new(
                    Dim3::new(blocks_x as u32, blocks_y as u32, *batch as u32),
                    Dim3::d2(block_n as u32, block_m as u32),
                )
            }

            ParallelismPattern::Scan { size, .. } => {
                let threads_per_block = 256;
                let elements_per_block = threads_per_block * 2;
                let blocks = (*size + elements_per_block - 1) / elements_per_block;

                let mut config =
                    KernelConfig::new(Dim3::d1(blocks as u32), Dim3::d1(threads_per_block as u32));
                config.shared_mem_bytes = elements_per_block * 8;
                config
            }

            _ => {
                // Default configuration
                KernelConfig::new(Dim3::d1(256), Dim3::d1(256))
            }
        }
    }
}

/// Result of kernel analysis
#[derive(Clone, Debug)]
pub struct KernelAnalysis {
    /// Kernel name
    pub name: String,
    /// Parameters
    pub parameters: Vec<(String, CudaType)>,
    /// Detected pattern
    pub pattern: ParallelismPattern,
    /// Suggested configuration
    pub suggested_config: KernelConfig,
    /// Uses tensor cores
    pub uses_tensor_cores: bool,
    /// Has reduction operations
    pub has_reduction: bool,
    /// Number of global memory loads
    pub global_loads: usize,
    /// Number of global memory stores
    pub global_stores: usize,
    /// Number of arithmetic operations
    pub arithmetic_ops: usize,
    /// Number of FMA operations
    pub fma_ops: usize,
    /// Number of transcendental operations
    pub transcendental_ops: usize,
    /// Number of barriers
    pub barriers: usize,
}

impl KernelAnalysis {
    pub fn new(name: String) -> Self {
        Self {
            name,
            parameters: Vec::new(),
            pattern: ParallelismPattern::Custom,
            suggested_config: KernelConfig::new(Dim3::d1(1), Dim3::d1(256)),
            uses_tensor_cores: false,
            has_reduction: false,
            global_loads: 0,
            global_stores: 0,
            arithmetic_ops: 0,
            fma_ops: 0,
            transcendental_ops: 0,
            barriers: 0,
        }
    }

    /// Compute arithmetic intensity (ops / memory bytes)
    pub fn arithmetic_intensity(&self) -> f64 {
        let ops = (self.arithmetic_ops + self.fma_ops * 2 + self.transcendental_ops * 8) as f64;
        let bytes = ((self.global_loads + self.global_stores) * 8) as f64; // Assume 8 bytes avg
        if bytes > 0.0 {
            ops / bytes
        } else {
            0.0
        }
    }

    /// Estimate if kernel is compute or memory bound
    pub fn is_compute_bound(&self, arch: GpuArch) -> bool {
        // Rough estimate: A100 has ~2 TB/s bandwidth, ~20 TFLOPS FP64
        // Crossover at ~10 ops/byte
        self.arithmetic_intensity() > 10.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dim3() {
        let d = Dim3::new(16, 16, 4);
        assert_eq!(d.total(), 1024);

        let d1 = Dim3::d1(256);
        assert_eq!(d1.total(), 256);
    }

    #[test]
    fn test_launch_bounds() {
        let lb = LaunchBounds::new(512).with_min_blocks(2);
        assert_eq!(lb.to_cuda_attribute(), "__launch_bounds__(512, 2)");

        let lb2 = LaunchBounds::new(256);
        assert_eq!(lb2.to_cuda_attribute(), "__launch_bounds__(256)");
    }

    #[test]
    fn test_kernel_config() {
        let config = KernelConfig::new(Dim3::d2(64, 64), Dim3::d2(16, 16));

        assert_eq!(config.threads_per_block(), 256);
        assert_eq!(config.total_blocks(), 4096);
    }

    #[test]
    fn test_kernel_info_signature() {
        let mut info = KernelInfo::new("matmul_kernel".to_string());
        info.parameters = vec![
            (
                "A".to_string(),
                CudaType::Pointer {
                    pointee: Box::new(CudaType::Float),
                    address_space: super::super::types::AddressSpace::Global,
                },
            ),
            (
                "B".to_string(),
                CudaType::Pointer {
                    pointee: Box::new(CudaType::Float),
                    address_space: super::super::types::AddressSpace::Global,
                },
            ),
            (
                "C".to_string(),
                CudaType::Pointer {
                    pointee: Box::new(CudaType::Float),
                    address_space: super::super::types::AddressSpace::Global,
                },
            ),
            ("N".to_string(), CudaType::Int32),
        ];

        let sig = info.signature();
        assert!(sig.contains("__global__"));
        assert!(sig.contains("matmul_kernel"));
        assert!(sig.contains("float* A"));
    }

    #[test]
    fn test_kernel_analysis() {
        let analysis = KernelAnalysis::new("test_kernel".to_string());
        assert_eq!(analysis.arithmetic_intensity(), 0.0);
    }
}
