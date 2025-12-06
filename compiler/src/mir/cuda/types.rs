//! CUDA Type System and GPU Architecture Definitions
//!
//! Maps MIR types to CUDA types and provides GPU architecture specifications.

use crate::mir::types::MirType;

/// GPU architecture target
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuArch {
    /// Volta (SM 7.0) - First tensor core support
    Sm70,
    /// Turing (SM 7.5) - INT8 tensor cores
    Sm75,
    /// Ampere (SM 8.0) - TF32, BF16 tensor cores
    Sm80,
    /// Ampere (SM 8.6) - Consumer Ampere
    Sm86,
    /// Ada Lovelace (SM 8.9)
    Sm89,
    /// Hopper (SM 9.0) - FP8 tensor cores, TMA
    Sm90,
}

impl GpuArch {
    /// Check if architecture supports tensor cores
    pub fn has_tensor_cores(&self) -> bool {
        true // All supported architectures have tensor cores
    }

    /// Check if architecture supports FP16 tensor operations
    pub fn has_fp16_tensor(&self) -> bool {
        true
    }

    /// Check if architecture supports TF32 tensor operations
    pub fn has_tf32_tensor(&self) -> bool {
        matches!(
            self,
            GpuArch::Sm80 | GpuArch::Sm86 | GpuArch::Sm89 | GpuArch::Sm90
        )
    }

    /// Check if architecture supports BF16 tensor operations
    pub fn has_bf16_tensor(&self) -> bool {
        matches!(
            self,
            GpuArch::Sm80 | GpuArch::Sm86 | GpuArch::Sm89 | GpuArch::Sm90
        )
    }

    /// Check if architecture supports FP8 tensor operations
    pub fn has_fp8_tensor(&self) -> bool {
        matches!(self, GpuArch::Sm90)
    }

    /// Check if architecture supports INT8 tensor operations
    pub fn has_int8_tensor(&self) -> bool {
        !matches!(self, GpuArch::Sm70)
    }

    /// Get maximum threads per block
    pub fn max_threads_per_block(&self) -> u32 {
        1024
    }

    /// Get maximum shared memory per block (bytes)
    pub fn max_shared_memory_per_block(&self) -> u32 {
        match self {
            GpuArch::Sm70 => 96 * 1024,  // 96 KB
            GpuArch::Sm75 => 64 * 1024,  // 64 KB
            GpuArch::Sm80 => 163 * 1024, // 163 KB (configurable)
            GpuArch::Sm86 => 99 * 1024,  // 99 KB
            GpuArch::Sm89 => 99 * 1024,  // 99 KB
            GpuArch::Sm90 => 227 * 1024, // 227 KB
        }
    }

    /// Get maximum registers per thread
    pub fn max_registers_per_thread(&self) -> u32 {
        255
    }

    /// Get warp size
    pub fn warp_size(&self) -> u32 {
        32
    }

    /// Get number of SMs (approximate for target)
    pub fn typical_sm_count(&self) -> u32 {
        match self {
            GpuArch::Sm70 => 80,  // V100
            GpuArch::Sm75 => 68,  // RTX 2080 Ti
            GpuArch::Sm80 => 108, // A100
            GpuArch::Sm86 => 84,  // RTX 3090
            GpuArch::Sm89 => 128, // RTX 4090
            GpuArch::Sm90 => 132, // H100
        }
    }

    /// Get compute capability string
    pub fn compute_capability(&self) -> &'static str {
        match self {
            GpuArch::Sm70 => "70",
            GpuArch::Sm75 => "75",
            GpuArch::Sm80 => "80",
            GpuArch::Sm86 => "86",
            GpuArch::Sm89 => "89",
            GpuArch::Sm90 => "90",
        }
    }

    /// Get PTX architecture string
    pub fn ptx_arch(&self) -> &'static str {
        match self {
            GpuArch::Sm70 => "sm_70",
            GpuArch::Sm75 => "sm_75",
            GpuArch::Sm80 => "sm_80",
            GpuArch::Sm86 => "sm_86",
            GpuArch::Sm89 => "sm_89",
            GpuArch::Sm90 => "sm_90",
        }
    }
}

impl Default for GpuArch {
    fn default() -> Self {
        GpuArch::Sm80 // Default to Ampere
    }
}

/// CUDA type representation
#[derive(Clone, Debug, PartialEq)]
pub enum CudaType {
    /// Void type
    Void,
    /// Boolean
    Bool,
    /// 8-bit signed integer
    Int8,
    /// 16-bit signed integer
    Int16,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit unsigned integer
    UInt16,
    /// 32-bit unsigned integer
    UInt32,
    /// 64-bit unsigned integer
    UInt64,
    /// 16-bit float (half)
    Half,
    /// 16-bit brain float
    BFloat16,
    /// 32-bit float
    Float,
    /// 64-bit float (double)
    Double,
    /// CUDA half2 (packed two half values)
    Half2,
    /// Float2 vector
    Float2,
    /// Float4 vector
    Float4,
    /// Double2 vector
    Double2,
    /// Int2 vector
    Int2,
    /// Int4 vector
    Int4,
    /// Pointer to type
    Pointer {
        pointee: Box<CudaType>,
        address_space: AddressSpace,
    },
    /// Fixed-size array
    Array { element: Box<CudaType>, size: usize },
    /// Structure
    Struct {
        name: String,
        fields: Vec<(String, CudaType)>,
    },
    /// WMMA fragment type
    WmmaFragment {
        matrix: WmmaMatrix,
        m: u32,
        n: u32,
        k: u32,
        element_type: Box<CudaType>,
        layout: WmmaLayout,
    },
}

impl CudaType {
    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            CudaType::Void => 0,
            CudaType::Bool => 1,
            CudaType::Int8 | CudaType::UInt8 => 1,
            CudaType::Int16 | CudaType::UInt16 | CudaType::Half | CudaType::BFloat16 => 2,
            CudaType::Int32 | CudaType::UInt32 | CudaType::Float => 4,
            CudaType::Int64 | CudaType::UInt64 | CudaType::Double => 8,
            CudaType::Half2 => 4,
            CudaType::Float2 | CudaType::Int2 => 8,
            CudaType::Float4 | CudaType::Int4 | CudaType::Double2 => 16,
            CudaType::Pointer { .. } => 8, // 64-bit pointers
            CudaType::Array { element, size } => element.size_bytes() * size,
            CudaType::Struct { fields, .. } => {
                // Simplified - doesn't account for padding
                fields.iter().map(|(_, ty)| ty.size_bytes()).sum()
            }
            CudaType::WmmaFragment {
                m,
                n,
                k,
                element_type,
                matrix,
                ..
            } => {
                // Fragment size depends on matrix type
                let elements = match matrix {
                    WmmaMatrix::A => (*m * *k) as usize,
                    WmmaMatrix::B => (*k * *n) as usize,
                    WmmaMatrix::C | WmmaMatrix::D => (*m * *n) as usize,
                };
                elements * element_type.size_bytes() / 32 // Distributed across warp
            }
        }
    }

    /// Get alignment in bytes
    pub fn alignment(&self) -> usize {
        match self {
            CudaType::Void => 1,
            CudaType::Bool => 1,
            CudaType::Int8 | CudaType::UInt8 => 1,
            CudaType::Int16 | CudaType::UInt16 | CudaType::Half | CudaType::BFloat16 => 2,
            CudaType::Int32 | CudaType::UInt32 | CudaType::Float => 4,
            CudaType::Int64 | CudaType::UInt64 | CudaType::Double => 8,
            CudaType::Half2 => 4,
            CudaType::Float2 | CudaType::Int2 => 8,
            CudaType::Float4 | CudaType::Int4 | CudaType::Double2 => 16,
            CudaType::Pointer { .. } => 8,
            CudaType::Array { element, .. } => element.alignment(),
            CudaType::Struct { fields, .. } => fields
                .iter()
                .map(|(_, ty)| ty.alignment())
                .max()
                .unwrap_or(1),
            CudaType::WmmaFragment { element_type, .. } => element_type.alignment(),
        }
    }

    /// Get CUDA type name
    pub fn cuda_name(&self) -> String {
        match self {
            CudaType::Void => "void".to_string(),
            CudaType::Bool => "bool".to_string(),
            CudaType::Int8 => "int8_t".to_string(),
            CudaType::Int16 => "int16_t".to_string(),
            CudaType::Int32 => "int32_t".to_string(),
            CudaType::Int64 => "int64_t".to_string(),
            CudaType::UInt8 => "uint8_t".to_string(),
            CudaType::UInt16 => "uint16_t".to_string(),
            CudaType::UInt32 => "uint32_t".to_string(),
            CudaType::UInt64 => "uint64_t".to_string(),
            CudaType::Half => "half".to_string(),
            CudaType::BFloat16 => "__nv_bfloat16".to_string(),
            CudaType::Float => "float".to_string(),
            CudaType::Double => "double".to_string(),
            CudaType::Half2 => "half2".to_string(),
            CudaType::Float2 => "float2".to_string(),
            CudaType::Float4 => "float4".to_string(),
            CudaType::Double2 => "double2".to_string(),
            CudaType::Int2 => "int2".to_string(),
            CudaType::Int4 => "int4".to_string(),
            CudaType::Pointer {
                pointee,
                address_space,
            } => {
                let space = match address_space {
                    AddressSpace::Global => "",
                    AddressSpace::Shared => "__shared__ ",
                    AddressSpace::Constant => "__constant__ ",
                    AddressSpace::Local => "",
                    AddressSpace::Generic => "",
                };
                format!("{}{}*", space, pointee.cuda_name())
            }
            CudaType::Array { element, size } => {
                format!("{}[{}]", element.cuda_name(), size)
            }
            CudaType::Struct { name, .. } => name.clone(),
            CudaType::WmmaFragment {
                matrix,
                m,
                n,
                k,
                element_type,
                layout,
            } => {
                let matrix_str = match matrix {
                    WmmaMatrix::A => "matrix_a",
                    WmmaMatrix::B => "matrix_b",
                    WmmaMatrix::C => "accumulator",
                    WmmaMatrix::D => "accumulator",
                };
                let layout_str = match layout {
                    WmmaLayout::RowMajor => "nvcuda::wmma::row_major",
                    WmmaLayout::ColMajor => "nvcuda::wmma::col_major",
                };
                format!(
                    "nvcuda::wmma::fragment<nvcuda::wmma::{}, {}, {}, {}, {}, {}>",
                    matrix_str,
                    m,
                    n,
                    k,
                    element_type.cuda_name(),
                    layout_str
                )
            }
        }
    }

    /// Convert from MIR type
    pub fn from_mir(mir_type: &MirType) -> Self {
        match mir_type {
            MirType::Void => CudaType::Void,
            MirType::Bool => CudaType::Bool,
            MirType::I8 => CudaType::Int8,
            MirType::I16 => CudaType::Int16,
            MirType::I32 => CudaType::Int32,
            MirType::I64 => CudaType::Int64,
            MirType::U8 => CudaType::UInt8,
            MirType::U16 => CudaType::UInt16,
            MirType::U32 => CudaType::UInt32,
            MirType::U64 => CudaType::UInt64,
            MirType::F32 => CudaType::Float,
            MirType::F64 => CudaType::Double,
            MirType::Ptr { pointee, .. } => CudaType::Pointer {
                pointee: Box::new(CudaType::from_mir(pointee)),
                address_space: AddressSpace::Global,
            },
            MirType::Array { element, size } => CudaType::Array {
                element: Box::new(CudaType::from_mir(element)),
                size: *size,
            },
            MirType::Struct { name, fields, .. } => CudaType::Struct {
                name: name.clone(),
                fields: fields
                    .iter()
                    .map(|(n, ty)| (n.clone(), CudaType::from_mir(ty)))
                    .collect(),
            },
            MirType::Vector { element, size } => match (element.as_ref(), size) {
                (MirType::F32, 2) => CudaType::Float2,
                (MirType::F32, 4) => CudaType::Float4,
                (MirType::F64, 2) => CudaType::Double2,
                (MirType::I32, 2) => CudaType::Int2,
                (MirType::I32, 4) => CudaType::Int4,
                _ => CudaType::Array {
                    element: Box::new(CudaType::from_mir(element)),
                    size: *size,
                },
            },
            // Default fallback
            _ => CudaType::Void,
        }
    }
}

/// CUDA memory address spaces
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressSpace {
    /// Global memory (DRAM)
    Global,
    /// Shared memory (per-block SRAM)
    Shared,
    /// Constant memory (cached, read-only)
    Constant,
    /// Local memory (thread-private, spills to DRAM)
    Local,
    /// Generic address space
    Generic,
}

/// WMMA matrix operand type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WmmaMatrix {
    /// Input matrix A
    A,
    /// Input matrix B
    B,
    /// Accumulator (input)
    C,
    /// Accumulator (output)
    D,
}

/// WMMA matrix layout
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WmmaLayout {
    /// Row-major layout
    RowMajor,
    /// Column-major layout
    ColMajor,
}

/// Supported tensor core data types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorCoreDataType {
    /// FP16 input, FP16 accumulator
    Fp16Fp16,
    /// FP16 input, FP32 accumulator
    Fp16Fp32,
    /// BF16 input, FP32 accumulator
    Bf16Fp32,
    /// TF32 input, FP32 accumulator (Ampere+)
    Tf32Fp32,
    /// FP8 E4M3 input, FP32 accumulator (Hopper)
    Fp8E4m3Fp32,
    /// FP8 E5M2 input, FP32 accumulator (Hopper)
    Fp8E5m2Fp32,
    /// INT8 input, INT32 accumulator
    Int8Int32,
    /// INT4 input, INT32 accumulator
    Int4Int32,
    /// Binary (1-bit) input, INT32 accumulator
    BinaryInt32,
}

impl TensorCoreDataType {
    /// Check if supported by architecture
    pub fn is_supported(&self, arch: GpuArch) -> bool {
        match self {
            TensorCoreDataType::Fp16Fp16 | TensorCoreDataType::Fp16Fp32 => true,
            TensorCoreDataType::Bf16Fp32 | TensorCoreDataType::Tf32Fp32 => arch.has_tf32_tensor(),
            TensorCoreDataType::Fp8E4m3Fp32 | TensorCoreDataType::Fp8E5m2Fp32 => {
                arch.has_fp8_tensor()
            }
            TensorCoreDataType::Int8Int32 => arch.has_int8_tensor(),
            TensorCoreDataType::Int4Int32 | TensorCoreDataType::BinaryInt32 => {
                matches!(
                    arch,
                    GpuArch::Sm75 | GpuArch::Sm80 | GpuArch::Sm86 | GpuArch::Sm89 | GpuArch::Sm90
                )
            }
        }
    }

    /// Get input element type
    pub fn input_type(&self) -> CudaType {
        match self {
            TensorCoreDataType::Fp16Fp16 | TensorCoreDataType::Fp16Fp32 => CudaType::Half,
            TensorCoreDataType::Bf16Fp32 => CudaType::BFloat16,
            TensorCoreDataType::Tf32Fp32 => CudaType::Float, // TF32 uses float storage
            TensorCoreDataType::Fp8E4m3Fp32 | TensorCoreDataType::Fp8E5m2Fp32 => CudaType::UInt8,
            TensorCoreDataType::Int8Int32 => CudaType::Int8,
            TensorCoreDataType::Int4Int32 | TensorCoreDataType::BinaryInt32 => CudaType::UInt8,
        }
    }

    /// Get accumulator element type
    pub fn accumulator_type(&self) -> CudaType {
        match self {
            TensorCoreDataType::Fp16Fp16 => CudaType::Half,
            TensorCoreDataType::Fp16Fp32
            | TensorCoreDataType::Bf16Fp32
            | TensorCoreDataType::Tf32Fp32
            | TensorCoreDataType::Fp8E4m3Fp32
            | TensorCoreDataType::Fp8E5m2Fp32 => CudaType::Float,
            TensorCoreDataType::Int8Int32
            | TensorCoreDataType::Int4Int32
            | TensorCoreDataType::BinaryInt32 => CudaType::Int32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_arch_features() {
        assert!(GpuArch::Sm70.has_tensor_cores());
        assert!(GpuArch::Sm80.has_tf32_tensor());
        assert!(!GpuArch::Sm70.has_tf32_tensor());
        assert!(GpuArch::Sm90.has_fp8_tensor());
    }

    #[test]
    fn test_cuda_type_sizes() {
        assert_eq!(CudaType::Float.size_bytes(), 4);
        assert_eq!(CudaType::Double.size_bytes(), 8);
        assert_eq!(CudaType::Half.size_bytes(), 2);
        assert_eq!(CudaType::Float4.size_bytes(), 16);
    }

    #[test]
    fn test_cuda_type_names() {
        assert_eq!(CudaType::Float.cuda_name(), "float");
        assert_eq!(CudaType::Half.cuda_name(), "half");
        assert_eq!(CudaType::BFloat16.cuda_name(), "__nv_bfloat16");
    }

    #[test]
    fn test_mir_to_cuda_type() {
        assert_eq!(CudaType::from_mir(&MirType::F32), CudaType::Float);
        assert_eq!(CudaType::from_mir(&MirType::F64), CudaType::Double);
        assert_eq!(CudaType::from_mir(&MirType::I32), CudaType::Int32);
    }

    #[test]
    fn test_tensor_core_support() {
        assert!(TensorCoreDataType::Fp16Fp32.is_supported(GpuArch::Sm70));
        assert!(TensorCoreDataType::Tf32Fp32.is_supported(GpuArch::Sm80));
        assert!(!TensorCoreDataType::Tf32Fp32.is_supported(GpuArch::Sm70));
    }
}
