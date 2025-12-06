//! Tensor Core (WMMA) Operations
//!
//! Generates optimized tensor core code using CUDA's WMMA API.
//! Supports various matrix shapes and data types across GPU architectures.

use super::types::{CudaType, GpuArch, TensorCoreDataType, WmmaLayout, WmmaMatrix};

/// WMMA fragment configuration
#[derive(Clone, Debug)]
pub struct WmmaFragment {
    /// Fragment name
    pub name: String,
    /// Matrix type (A, B, C, or D)
    pub matrix: WmmaMatrix,
    /// M dimension
    pub m: u32,
    /// N dimension
    pub n: u32,
    /// K dimension
    pub k: u32,
    /// Element type
    pub element_type: CudaType,
    /// Layout (row or column major)
    pub layout: WmmaLayout,
}

impl WmmaFragment {
    /// Create fragment for matrix A
    pub fn matrix_a(m: u32, n: u32, k: u32, element_type: CudaType, layout: WmmaLayout) -> Self {
        Self {
            name: "frag_a".to_string(),
            matrix: WmmaMatrix::A,
            m,
            n,
            k,
            element_type,
            layout,
        }
    }

    /// Create fragment for matrix B
    pub fn matrix_b(m: u32, n: u32, k: u32, element_type: CudaType, layout: WmmaLayout) -> Self {
        Self {
            name: "frag_b".to_string(),
            matrix: WmmaMatrix::B,
            m,
            n,
            k,
            element_type,
            layout,
        }
    }

    /// Create accumulator fragment
    pub fn accumulator(m: u32, n: u32, k: u32, element_type: CudaType) -> Self {
        Self {
            name: "frag_c".to_string(),
            matrix: WmmaMatrix::C,
            m,
            n,
            k,
            element_type,
            layout: WmmaLayout::RowMajor, // Accumulators don't have layout
        }
    }

    /// Generate CUDA type declaration
    pub fn to_cuda_type(&self) -> String {
        let matrix_str = match self.matrix {
            WmmaMatrix::A => "matrix_a",
            WmmaMatrix::B => "matrix_b",
            WmmaMatrix::C | WmmaMatrix::D => "accumulator",
        };

        let type_str = self.element_type.cuda_name();

        match self.matrix {
            WmmaMatrix::A | WmmaMatrix::B => {
                let layout_str = match self.layout {
                    WmmaLayout::RowMajor => "row_major",
                    WmmaLayout::ColMajor => "col_major",
                };
                format!(
                    "nvcuda::wmma::fragment<nvcuda::wmma::{}, {}, {}, {}, {}, nvcuda::wmma::{}>",
                    matrix_str, self.m, self.n, self.k, type_str, layout_str
                )
            }
            WmmaMatrix::C | WmmaMatrix::D => {
                format!(
                    "nvcuda::wmma::fragment<nvcuda::wmma::{}, {}, {}, {}, {}>",
                    matrix_str, self.m, self.n, self.k, type_str
                )
            }
        }
    }

    /// Generate declaration statement
    pub fn to_cuda_decl(&self) -> String {
        format!("{} {};", self.to_cuda_type(), self.name)
    }
}

/// WMMA operation
#[derive(Clone, Debug)]
pub enum WmmaOp {
    /// Load fragment from memory
    Load {
        fragment: String,
        ptr: String,
        stride: String,
        layout: WmmaLayout,
    },
    /// Store fragment to memory
    Store {
        ptr: String,
        fragment: String,
        stride: String,
        layout: WmmaLayout,
    },
    /// Matrix multiply-accumulate: D = A * B + C
    Mma {
        d: String,
        a: String,
        b: String,
        c: String,
    },
    /// Fill fragment with value
    Fill { fragment: String, value: String },
}

impl WmmaOp {
    /// Generate CUDA code for operation
    pub fn to_cuda(&self) -> String {
        match self {
            WmmaOp::Load {
                fragment,
                ptr,
                stride,
                layout,
            } => {
                let layout_str = match layout {
                    WmmaLayout::RowMajor => "nvcuda::wmma::mem_row_major",
                    WmmaLayout::ColMajor => "nvcuda::wmma::mem_col_major",
                };
                format!(
                    "nvcuda::wmma::load_matrix_sync({}, {}, {}, {});",
                    fragment, ptr, stride, layout_str
                )
            }
            WmmaOp::Store {
                ptr,
                fragment,
                stride,
                layout,
            } => {
                let layout_str = match layout {
                    WmmaLayout::RowMajor => "nvcuda::wmma::mem_row_major",
                    WmmaLayout::ColMajor => "nvcuda::wmma::mem_col_major",
                };
                format!(
                    "nvcuda::wmma::store_matrix_sync({}, {}, {}, {});",
                    ptr, fragment, stride, layout_str
                )
            }
            WmmaOp::Mma { d, a, b, c } => {
                format!("nvcuda::wmma::mma_sync({}, {}, {}, {});", d, a, b, c)
            }
            WmmaOp::Fill { fragment, value } => {
                format!("nvcuda::wmma::fill_fragment({}, {});", fragment, value)
            }
        }
    }
}

/// Tensor core configuration
#[derive(Clone, Debug)]
pub struct TensorCoreConfig {
    /// Target architecture
    pub arch: GpuArch,
    /// Data type configuration
    pub data_type: TensorCoreDataType,
    /// Tile M dimension
    pub tile_m: u32,
    /// Tile N dimension
    pub tile_n: u32,
    /// Tile K dimension
    pub tile_k: u32,
    /// Number of warps in M dimension
    pub warps_m: u32,
    /// Number of warps in N dimension
    pub warps_n: u32,
}

impl TensorCoreConfig {
    /// Create configuration for FP16 tensor cores
    pub fn fp16(arch: GpuArch) -> Self {
        Self {
            arch,
            data_type: TensorCoreDataType::Fp16Fp32,
            tile_m: 16,
            tile_n: 16,
            tile_k: 16,
            warps_m: 2,
            warps_n: 2,
        }
    }

    /// Create configuration for TF32 tensor cores (Ampere+)
    pub fn tf32(arch: GpuArch) -> Self {
        assert!(arch.has_tf32_tensor(), "TF32 requires Ampere or later");
        Self {
            arch,
            data_type: TensorCoreDataType::Tf32Fp32,
            tile_m: 16,
            tile_n: 16,
            tile_k: 8,
            warps_m: 2,
            warps_n: 2,
        }
    }

    /// Create configuration for BF16 tensor cores (Ampere+)
    pub fn bf16(arch: GpuArch) -> Self {
        assert!(arch.has_bf16_tensor(), "BF16 requires Ampere or later");
        Self {
            arch,
            data_type: TensorCoreDataType::Bf16Fp32,
            tile_m: 16,
            tile_n: 16,
            tile_k: 16,
            warps_m: 2,
            warps_n: 2,
        }
    }

    /// Create configuration for INT8 tensor cores
    pub fn int8(arch: GpuArch) -> Self {
        assert!(arch.has_int8_tensor(), "INT8 requires Turing or later");
        Self {
            arch,
            data_type: TensorCoreDataType::Int8Int32,
            tile_m: 16,
            tile_n: 16,
            tile_k: 32,
            warps_m: 2,
            warps_n: 2,
        }
    }

    /// Get supported WMMA shapes for current configuration
    pub fn supported_shapes(&self) -> Vec<(u32, u32, u32)> {
        match self.data_type {
            TensorCoreDataType::Fp16Fp16 | TensorCoreDataType::Fp16Fp32 => {
                vec![(16, 16, 16), (32, 8, 16), (8, 32, 16)]
            }
            TensorCoreDataType::Tf32Fp32 => {
                vec![(16, 16, 8)]
            }
            TensorCoreDataType::Bf16Fp32 => {
                vec![(16, 16, 16), (32, 8, 16), (8, 32, 16)]
            }
            TensorCoreDataType::Int8Int32 => {
                vec![(16, 16, 32), (32, 8, 32), (8, 32, 32)]
            }
            _ => vec![(16, 16, 16)], // Default
        }
    }

    /// Get total threads needed
    pub fn total_threads(&self) -> u32 {
        self.warps_m * self.warps_n * 32
    }

    /// Get block dimensions
    pub fn block_dim(&self) -> (u32, u32) {
        (self.warps_n * 32, self.warps_m)
    }

    /// Generate required includes
    pub fn required_includes(&self) -> Vec<&'static str> {
        vec!["#include <mma.h>", "using namespace nvcuda;"]
    }
}

impl Default for TensorCoreConfig {
    fn default() -> Self {
        Self::fp16(GpuArch::default())
    }
}

/// Generate tensor core matrix multiplication kernel
pub struct TensorCoreMatmulGenerator {
    config: TensorCoreConfig,
}

impl TensorCoreMatmulGenerator {
    pub fn new(config: TensorCoreConfig) -> Self {
        Self { config }
    }

    /// Generate complete WMMA matrix multiplication kernel
    pub fn generate_kernel(&self, m: usize, n: usize, k: usize) -> String {
        let mut code = String::new();

        // Includes
        for inc in self.config.required_includes() {
            code.push_str(inc);
            code.push('\n');
        }
        code.push('\n');

        // Constants
        code.push_str(&format!("constexpr int WMMA_M = {};\n", self.config.tile_m));
        code.push_str(&format!("constexpr int WMMA_N = {};\n", self.config.tile_n));
        code.push_str(&format!("constexpr int WMMA_K = {};\n", self.config.tile_k));
        code.push('\n');

        // Kernel signature
        let input_type = self.config.data_type.input_type().cuda_name();
        let acc_type = self.config.data_type.accumulator_type().cuda_name();

        code.push_str(&format!(
            "__global__ void wmma_matmul(\n\
             \tconst {}* __restrict__ A,\n\
             \tconst {}* __restrict__ B,\n\
             \t{}* __restrict__ C,\n\
             \tint M, int N, int K) {{\n",
            input_type, input_type, acc_type
        ));

        // Thread indices
        code.push_str("\t// Warp and lane indices\n");
        code.push_str("\tint warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;\n");
        code.push_str("\tint warpN = (blockIdx.y * blockDim.y + threadIdx.y);\n");
        code.push('\n');

        // Fragment declarations
        code.push_str("\t// Declare fragments\n");
        code.push_str(&format!(
            "\twmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, {}, wmma::row_major> a_frag;\n",
            input_type
        ));
        code.push_str(&format!(
            "\twmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, {}, wmma::col_major> b_frag;\n",
            input_type
        ));
        code.push_str(&format!(
            "\twmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, {}> c_frag;\n",
            acc_type
        ));
        code.push('\n');

        // Initialize accumulator
        code.push_str("\t// Initialize accumulator to zero\n");
        code.push_str("\twmma::fill_fragment(c_frag, 0.0f);\n");
        code.push('\n');

        // Main loop over K
        code.push_str("\t// Loop over K dimension\n");
        code.push_str("\tfor (int i = 0; i < K; i += WMMA_K) {\n");
        code.push_str("\t\tint aRow = warpM * WMMA_M;\n");
        code.push_str("\t\tint aCol = i;\n");
        code.push_str("\t\tint bRow = i;\n");
        code.push_str("\t\tint bCol = warpN * WMMA_N;\n");
        code.push('\n');

        // Bounds check
        code.push_str("\t\tif (aRow < M && aCol < K && bRow < K && bCol < N) {\n");

        // Load fragments
        code.push_str("\t\t\t// Load matrix fragments\n");
        code.push_str("\t\t\twmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);\n");
        code.push_str("\t\t\twmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);\n");
        code.push('\n');

        // MMA
        code.push_str("\t\t\t// Perform matrix multiply-accumulate\n");
        code.push_str("\t\t\twmma::mma_sync(c_frag, a_frag, b_frag, c_frag);\n");
        code.push_str("\t\t}\n");
        code.push_str("\t}\n");
        code.push('\n');

        // Store result
        code.push_str("\t// Store result\n");
        code.push_str("\tint cRow = warpM * WMMA_M;\n");
        code.push_str("\tint cCol = warpN * WMMA_N;\n");
        code.push_str("\tif (cRow < M && cCol < N) {\n");
        code.push_str(
            "\t\twmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);\n",
        );
        code.push_str("\t}\n");

        code.push_str("}\n");

        code
    }

    /// Generate kernel launch code
    pub fn generate_launch(&self, m: usize, n: usize, _k: usize) -> String {
        let (block_x, block_y) = self.config.block_dim();
        let grid_x = (n + (self.config.tile_n as usize) - 1) / (self.config.tile_n as usize);
        let grid_y = (m + (self.config.tile_m as usize) - 1) / (self.config.tile_m as usize);

        format!(
            "dim3 grid({}, {});\n\
             dim3 block({}, {});\n\
             wmma_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);",
            grid_x, grid_y, block_x, block_y
        )
    }
}

/// Double-buffered tensor core implementation
pub struct DoubleBuferredTensorCore {
    config: TensorCoreConfig,
}

impl DoubleBuferredTensorCore {
    pub fn new(config: TensorCoreConfig) -> Self {
        Self { config }
    }

    /// Generate double-buffered pipeline code
    pub fn generate_pipeline_kernel(&self) -> String {
        let mut code = String::new();

        code.push_str("// Double-buffered tensor core pipeline\n");
        code.push_str("// Uses two sets of shared memory buffers to overlap\n");
        code.push_str("// memory loads with computation\n\n");

        let input_type = self.config.data_type.input_type().cuda_name();
        let tile_m = self.config.tile_m;
        let tile_n = self.config.tile_n;
        let tile_k = self.config.tile_k;

        // Shared memory declarations (double buffered)
        code.push_str(&format!(
            "__shared__ {} smem_A[2][{}][{}];\n",
            input_type, tile_m, tile_k
        ));
        code.push_str(&format!(
            "__shared__ {} smem_B[2][{}][{}];\n",
            input_type, tile_k, tile_n
        ));
        code.push('\n');

        // Pipeline loop structure
        code.push_str("int write_stage = 0;\n");
        code.push_str("int read_stage = 0;\n\n");

        code.push_str("// Prologue: load first tile\n");
        code.push_str("load_tile(smem_A[write_stage], smem_B[write_stage], ...);\n");
        code.push_str("__syncthreads();\n\n");

        code.push_str("for (int k_tile = 0; k_tile < K_tiles; k_tile++) {\n");
        code.push_str("\tread_stage = write_stage;\n");
        code.push_str("\twrite_stage = 1 - write_stage;\n\n");

        code.push_str("\t// Async load next tile while computing current\n");
        code.push_str("\tif (k_tile < K_tiles - 1) {\n");
        code.push_str("\t\tload_tile_async(smem_A[write_stage], smem_B[write_stage], ...);\n");
        code.push_str("\t}\n\n");

        code.push_str("\t// Compute with current tile\n");
        code.push_str("\twmma::load_matrix_sync(a_frag, &smem_A[read_stage][...], ...);\n");
        code.push_str("\twmma::load_matrix_sync(b_frag, &smem_B[read_stage][...], ...);\n");
        code.push_str("\twmma::mma_sync(c_frag, a_frag, b_frag, c_frag);\n\n");

        code.push_str("\t__syncthreads();\n");
        code.push_str("}\n");

        code
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wmma_fragment() {
        let frag = WmmaFragment::matrix_a(16, 16, 16, CudaType::Half, WmmaLayout::RowMajor);
        let decl = frag.to_cuda_decl();
        assert!(decl.contains("matrix_a"));
        assert!(decl.contains("half"));
        assert!(decl.contains("row_major"));
    }

    #[test]
    fn test_wmma_ops() {
        let load = WmmaOp::Load {
            fragment: "frag_a".to_string(),
            ptr: "ptr_a".to_string(),
            stride: "lda".to_string(),
            layout: WmmaLayout::RowMajor,
        };

        let cuda = load.to_cuda();
        assert!(cuda.contains("load_matrix_sync"));
        assert!(cuda.contains("mem_row_major"));
    }

    #[test]
    fn test_tensor_core_config() {
        let config = TensorCoreConfig::fp16(GpuArch::Sm80);
        assert_eq!(config.tile_m, 16);
        assert_eq!(config.total_threads(), 128);
    }

    #[test]
    fn test_matmul_generator() {
        let config = TensorCoreConfig::fp16(GpuArch::Sm80);
        let gen = TensorCoreMatmulGenerator::new(config);

        let kernel = gen.generate_kernel(1024, 1024, 1024);
        assert!(kernel.contains("wmma_matmul"));
        assert!(kernel.contains("mma_sync"));
        assert!(kernel.contains("load_matrix_sync"));
    }

    #[test]
    fn test_supported_shapes() {
        let config = TensorCoreConfig::fp16(GpuArch::Sm80);
        let shapes = config.supported_shapes();
        assert!(shapes.contains(&(16, 16, 16)));
    }

    #[test]
    fn test_tf32_requires_ampere() {
        // Should not panic for Ampere
        let _config = TensorCoreConfig::tf32(GpuArch::Sm80);
    }

    #[test]
    #[should_panic]
    fn test_tf32_not_available_on_volta() {
        // Should panic for Volta
        let _config = TensorCoreConfig::tf32(GpuArch::Sm70);
    }
}
