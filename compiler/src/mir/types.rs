//! MIR Type System (Post-Erasure)
//!
//! MIR types are monomorphic and machine-representable.
//! All generics, traits, and units have been erased at this level.
//!
//! This module provides the foundational type system for MedLang's
//! intermediate representation, designed for:
//! - Deterministic semantics (no undefined behavior)
//! - Direct mapping to numerical backends (LLVM, Stan, CUDA)
//! - Efficient AD (automatic differentiation) transformation
//! - Verifiable memory safety

use std::collections::HashMap;
use std::fmt;

/// MIR types are monomorphic and machine-representable
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MirType {
    // =========================================================================
    // PRIMITIVE TYPES
    // =========================================================================
    /// Void (for statements, unit return)
    Void,

    /// Boolean (1 bit logical, 8 bit storage)
    Bool,

    /// Signed integers
    I8,
    I16,
    I32,
    I64,
    I128,

    /// Unsigned integers
    U8,
    U16,
    U32,
    U64,
    U128,

    /// IEEE 754 floating point
    F32, // single precision
    F64, // double precision (DEFAULT for medical computations)

    /// Extended precision (for numerical stability)
    F80, // x87 extended
    F128, // quad precision (software emulated)

    // =========================================================================
    // COMPOSITE TYPES
    // =========================================================================
    /// Fixed-size array: [T; N]
    Array {
        element: Box<MirType>,
        size: usize,
    },

    /// Dynamically-sized slice (fat pointer: ptr + len)
    Slice {
        element: Box<MirType>,
    },

    /// Struct (product type with named fields)
    Struct {
        name: String,
        fields: Vec<(String, MirType)>,
        layout: StructLayout,
    },

    /// Tuple (anonymous product type)
    Tuple {
        elements: Vec<MirType>,
    },

    /// Enum/ADT (sum type)
    Enum {
        name: String,
        variants: Vec<EnumVariant>,
        layout: EnumLayout,
    },

    /// Function pointer
    FnPtr {
        params: Vec<MirType>,
        ret: Box<MirType>,
        abi: CallingConvention,
    },

    // =========================================================================
    // POINTER TYPES
    // =========================================================================
    /// Raw pointer (unsafe)
    Ptr {
        pointee: Box<MirType>,
        mutability: Mutability,
    },

    /// Reference (safe, lifetime-checked at HIR level)
    Ref {
        pointee: Box<MirType>,
        mutability: Mutability,
        /// Lifetime has been erased but we track region for optimization
        region: RegionId,
    },

    /// Box (owned heap allocation)
    Box {
        pointee: Box<MirType>,
    },

    // =========================================================================
    // SPECIAL TYPES FOR NUMERICS
    // =========================================================================
    /// Dual number for forward-mode AD (value, derivative)
    DualF64,
    DualF32,

    /// Dual vector for vector AD
    DualVecF64 {
        size: usize,
    },

    /// Dense matrix (row-major by default)
    Matrix {
        element: Box<MirType>,
        rows: usize,
        cols: usize,
        layout: MatrixLayout,
    },

    /// Vector (1D dense array with SIMD alignment)
    Vector {
        element: Box<MirType>,
        size: usize,
    },

    /// Sparse matrix (CSR format)
    SparseMatrix {
        element: Box<MirType>,
        rows: usize,
        cols: usize,
    },

    /// Complex number
    Complex {
        element: Box<MirType>, // F32 or F64
    },

    /// Interval for interval arithmetic
    Interval {
        element: Box<MirType>, // F32 or F64
    },
}

/// Mutability qualifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mutability {
    Const,
    Mut,
}

impl fmt::Display for Mutability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mutability::Const => write!(f, "const"),
            Mutability::Mut => write!(f, "mut"),
        }
    }
}

/// Region identifier for lifetime tracking
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegionId(pub u32);

impl RegionId {
    /// Static region (lives for entire program)
    pub const STATIC: RegionId = RegionId(0);

    /// Create a new region ID
    pub fn new(id: u32) -> Self {
        RegionId(id)
    }
}

/// Enum variant definition
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EnumVariant {
    pub name: String,
    pub discriminant: u64,
    pub fields: Vec<MirType>,
}

impl EnumVariant {
    pub fn new(name: &str, discriminant: u64) -> Self {
        Self {
            name: name.to_string(),
            discriminant,
            fields: Vec::new(),
        }
    }

    pub fn with_fields(mut self, fields: Vec<MirType>) -> Self {
        self.fields = fields;
        self
    }

    /// Payload size in bytes
    pub fn payload_size(&self) -> usize {
        self.fields.iter().map(|f| f.size()).sum()
    }

    /// Payload alignment
    pub fn payload_align(&self) -> usize {
        self.fields.iter().map(|f| f.align()).max().unwrap_or(1)
    }
}

/// Calling convention for function pointers
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum CallingConvention {
    /// MedLang native (Rust-like)
    #[default]
    MedLang,
    /// C ABI for FFI
    C,
    /// Fast call (registers only, no stack)
    Fast,
    /// Vector call (SIMD registers)
    Vectorcall,
    /// GPU kernel
    CUDA,
    /// Stan model block
    Stan,
    /// System V AMD64
    SysV64,
    /// Windows x64
    Win64,
}

impl fmt::Display for CallingConvention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallingConvention::MedLang => write!(f, "medlang"),
            CallingConvention::C => write!(f, "C"),
            CallingConvention::Fast => write!(f, "fastcc"),
            CallingConvention::Vectorcall => write!(f, "vectorcall"),
            CallingConvention::CUDA => write!(f, "cuda"),
            CallingConvention::Stan => write!(f, "stan"),
            CallingConvention::SysV64 => write!(f, "sysv64"),
            CallingConvention::Win64 => write!(f, "win64"),
        }
    }
}

/// Struct memory layout
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StructLayout {
    pub size: usize,
    pub align: usize,
    pub field_offsets: Vec<usize>,
}

impl StructLayout {
    /// Create layout for a struct with given fields
    pub fn compute(fields: &[(String, MirType)]) -> Self {
        let mut offsets = Vec::with_capacity(fields.len());
        let mut current_offset = 0usize;
        let mut max_align = 1usize;

        for (_, ty) in fields {
            let align = ty.align();
            max_align = max_align.max(align);

            // Align current offset
            current_offset = (current_offset + align - 1) & !(align - 1);
            offsets.push(current_offset);
            current_offset += ty.size();
        }

        // Total size must be multiple of alignment
        let size = (current_offset + max_align - 1) & !(max_align - 1);

        StructLayout {
            size,
            align: max_align,
            field_offsets: offsets,
        }
    }

    /// Get offset for field at index
    pub fn offset_of(&self, field_idx: usize) -> Option<usize> {
        self.field_offsets.get(field_idx).copied()
    }
}

/// Enum memory layout (tagged union)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EnumLayout {
    pub size: usize,
    pub align: usize,
    pub discriminant_offset: usize,
    pub discriminant_size: usize,
    pub payload_offset: usize,
}

impl EnumLayout {
    /// Compute layout for enum with given variants
    pub fn compute(variants: &[EnumVariant]) -> Self {
        // Discriminant size based on variant count
        let discriminant_size = if variants.len() <= 256 {
            1
        } else if variants.len() <= 65536 {
            2
        } else {
            4
        };

        // Find max payload size and alignment
        let mut max_payload_size = 0usize;
        let mut max_align = discriminant_size;

        for variant in variants {
            let payload_size = variant.payload_size();
            let payload_align = variant.payload_align();

            max_payload_size = max_payload_size.max(payload_size);
            max_align = max_align.max(payload_align);
        }

        // Layout: [discriminant][padding][payload]
        let discriminant_offset = 0;
        let payload_offset = (discriminant_size + max_align - 1) & !(max_align - 1);
        let size = payload_offset + max_payload_size;
        let size = (size + max_align - 1) & !(max_align - 1);

        EnumLayout {
            size,
            align: max_align,
            discriminant_offset,
            discriminant_size,
            payload_offset,
        }
    }
}

/// Matrix memory layout
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MatrixLayout {
    #[default]
    RowMajor,
    ColMajor, // For Fortran/BLAS compatibility
}

impl MirType {
    /// Size in bytes
    pub fn size(&self) -> usize {
        match self {
            MirType::Void => 0,
            MirType::Bool | MirType::I8 | MirType::U8 => 1,
            MirType::I16 | MirType::U16 => 2,
            MirType::I32 | MirType::U32 | MirType::F32 => 4,
            MirType::I64 | MirType::U64 | MirType::F64 => 8,
            MirType::I128 | MirType::U128 | MirType::F80 => 16,
            MirType::F128 => 16,
            MirType::DualF64 => 16, // value + derivative
            MirType::DualF32 => 8,
            MirType::Array { element, size } => element.size() * size,
            MirType::Slice { .. } => 16, // ptr + len (fat pointer)
            MirType::Struct { layout, .. } => layout.size,
            MirType::Tuple { elements } => {
                if elements.is_empty() {
                    return 0;
                }
                // Compute with alignment padding
                let mut size = 0usize;
                let mut max_align = 1usize;
                for elem in elements {
                    let align = elem.align();
                    max_align = max_align.max(align);
                    size = (size + align - 1) & !(align - 1);
                    size += elem.size();
                }
                (size + max_align - 1) & !(max_align - 1)
            }
            MirType::Enum { layout, .. } => layout.size,
            MirType::FnPtr { .. } => 8,
            MirType::Ptr { .. } | MirType::Ref { .. } | MirType::Box { .. } => 8,
            MirType::DualVecF64 { size } => 16 * size,
            MirType::Matrix {
                element,
                rows,
                cols,
                ..
            } => element.size() * rows * cols,
            MirType::Vector { element, size } => element.size() * size,
            MirType::SparseMatrix { .. } => 24, // ptr to CSR structure
            MirType::Complex { element } => element.size() * 2,
            MirType::Interval { element } => element.size() * 2,
        }
    }

    /// Alignment in bytes
    pub fn align(&self) -> usize {
        match self {
            MirType::Void => 1,
            MirType::Bool | MirType::I8 | MirType::U8 => 1,
            MirType::I16 | MirType::U16 => 2,
            MirType::I32 | MirType::U32 | MirType::F32 => 4,
            MirType::I64 | MirType::U64 | MirType::F64 => 8,
            MirType::I128 | MirType::U128 | MirType::F128 => 16,
            MirType::F80 => 16,
            MirType::DualF64 => 8,
            MirType::DualF32 => 4,
            MirType::Array { element, .. } => element.align(),
            MirType::Slice { element } => element.align().max(8),
            MirType::Struct { layout, .. } => layout.align,
            MirType::Tuple { elements } => elements.iter().map(|e| e.align()).max().unwrap_or(1),
            MirType::Enum { layout, .. } => layout.align,
            MirType::FnPtr { .. }
            | MirType::Ptr { .. }
            | MirType::Ref { .. }
            | MirType::Box { .. } => 8,
            MirType::DualVecF64 { .. } => 8,
            MirType::Matrix { element, .. } | MirType::Vector { element, .. } => {
                // SIMD alignment: 32 bytes for AVX, 64 for AVX-512
                element.align().max(32)
            }
            MirType::SparseMatrix { .. } => 8,
            MirType::Complex { element } | MirType::Interval { element } => element.align(),
        }
    }

    /// Is this a floating-point type?
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            MirType::F32 | MirType::F64 | MirType::F80 | MirType::F128
        )
    }

    /// Is this an integer type?
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            MirType::I8
                | MirType::I16
                | MirType::I32
                | MirType::I64
                | MirType::I128
                | MirType::U8
                | MirType::U16
                | MirType::U32
                | MirType::U64
                | MirType::U128
        )
    }

    /// Is this a signed integer type?
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            MirType::I8 | MirType::I16 | MirType::I32 | MirType::I64 | MirType::I128
        )
    }

    /// Is this an unsigned integer type?
    pub fn is_unsigned(&self) -> bool {
        matches!(
            self,
            MirType::U8 | MirType::U16 | MirType::U32 | MirType::U64 | MirType::U128
        )
    }

    /// Is this a SIMD-friendly type?
    pub fn is_simd(&self) -> bool {
        matches!(self, MirType::Vector { .. } | MirType::Matrix { .. })
    }

    /// Is this a dual number type (for AD)?
    pub fn is_dual(&self) -> bool {
        matches!(
            self,
            MirType::DualF32 | MirType::DualF64 | MirType::DualVecF64 { .. }
        )
    }

    /// Is this a pointer type?
    pub fn is_pointer(&self) -> bool {
        matches!(
            self,
            MirType::Ptr { .. } | MirType::Ref { .. } | MirType::Box { .. }
        )
    }

    /// Is this type Copy (can be freely duplicated)?
    pub fn is_copy(&self) -> bool {
        match self {
            // Primitives are Copy
            MirType::Void
            | MirType::Bool
            | MirType::I8
            | MirType::I16
            | MirType::I32
            | MirType::I64
            | MirType::I128
            | MirType::U8
            | MirType::U16
            | MirType::U32
            | MirType::U64
            | MirType::U128
            | MirType::F32
            | MirType::F64
            | MirType::F80
            | MirType::F128
            | MirType::DualF32
            | MirType::DualF64 => true,

            // Raw pointers are Copy
            MirType::Ptr { .. } => true,

            // References are Copy (the reference itself)
            MirType::Ref { .. } => true,

            // Function pointers are Copy
            MirType::FnPtr { .. } => true,

            // Small fixed arrays of Copy types are Copy
            MirType::Array { element, size } if *size <= 32 => element.is_copy(),

            // Tuples of Copy types are Copy
            MirType::Tuple { elements } => elements.iter().all(|e| e.is_copy()),

            // Complex and interval are Copy if element is
            MirType::Complex { element } | MirType::Interval { element } => element.is_copy(),

            // Everything else is not Copy
            _ => false,
        }
    }

    /// Does this type need a destructor?
    pub fn needs_drop(&self) -> bool {
        match self {
            MirType::Box { .. } => true,
            MirType::Slice { .. } => true,
            MirType::Struct { fields, .. } => fields.iter().any(|(_, ty)| ty.needs_drop()),
            MirType::Enum { variants, .. } => variants
                .iter()
                .any(|v| v.fields.iter().any(|f| f.needs_drop())),
            MirType::Array { element, .. } => element.needs_drop(),
            MirType::Tuple { elements } => elements.iter().any(|e| e.needs_drop()),
            MirType::SparseMatrix { .. } => true,
            _ => false,
        }
    }

    /// Get the element type for aggregate types
    pub fn element_type(&self) -> Option<&MirType> {
        match self {
            MirType::Array { element, .. }
            | MirType::Slice { element }
            | MirType::Vector { element, .. }
            | MirType::Matrix { element, .. }
            | MirType::SparseMatrix { element, .. }
            | MirType::Complex { element }
            | MirType::Interval { element } => Some(element),
            MirType::Ptr { pointee, .. }
            | MirType::Ref { pointee, .. }
            | MirType::Box { pointee } => Some(pointee),
            _ => None,
        }
    }

    /// Get bit width for integer types
    pub fn bit_width(&self) -> Option<usize> {
        match self {
            MirType::Bool => Some(1),
            MirType::I8 | MirType::U8 => Some(8),
            MirType::I16 | MirType::U16 => Some(16),
            MirType::I32 | MirType::U32 | MirType::F32 => Some(32),
            MirType::I64 | MirType::U64 | MirType::F64 => Some(64),
            MirType::F80 => Some(80),
            MirType::I128 | MirType::U128 | MirType::F128 => Some(128),
            _ => None,
        }
    }
}

// ============================================================================
// Type Constructors
// ============================================================================

impl MirType {
    /// Create a pointer type
    pub fn ptr(pointee: MirType, mutable: bool) -> Self {
        MirType::Ptr {
            pointee: Box::new(pointee),
            mutability: if mutable {
                Mutability::Mut
            } else {
                Mutability::Const
            },
        }
    }

    /// Create a reference type
    pub fn reference(pointee: MirType, mutable: bool, region: RegionId) -> Self {
        MirType::Ref {
            pointee: Box::new(pointee),
            mutability: if mutable {
                Mutability::Mut
            } else {
                Mutability::Const
            },
            region,
        }
    }

    /// Create a boxed type
    pub fn boxed(pointee: MirType) -> Self {
        MirType::Box {
            pointee: Box::new(pointee),
        }
    }

    /// Create an array type
    pub fn array(element: MirType, size: usize) -> Self {
        MirType::Array {
            element: Box::new(element),
            size,
        }
    }

    /// Create a slice type
    pub fn slice(element: MirType) -> Self {
        MirType::Slice {
            element: Box::new(element),
        }
    }

    /// Create a vector type (SIMD-aligned)
    pub fn vector(element: MirType, size: usize) -> Self {
        MirType::Vector {
            element: Box::new(element),
            size,
        }
    }

    /// Create a matrix type
    pub fn matrix(element: MirType, rows: usize, cols: usize) -> Self {
        MirType::Matrix {
            element: Box::new(element),
            rows,
            cols,
            layout: MatrixLayout::RowMajor,
        }
    }

    /// Create a matrix type with column-major layout
    pub fn matrix_col_major(element: MirType, rows: usize, cols: usize) -> Self {
        MirType::Matrix {
            element: Box::new(element),
            rows,
            cols,
            layout: MatrixLayout::ColMajor,
        }
    }

    /// Create a struct type
    pub fn structure(name: &str, fields: Vec<(String, MirType)>) -> Self {
        let layout = StructLayout::compute(&fields);
        MirType::Struct {
            name: name.to_string(),
            fields,
            layout,
        }
    }

    /// Create a tuple type
    pub fn tuple(elements: Vec<MirType>) -> Self {
        MirType::Tuple { elements }
    }

    /// Create an enum type
    pub fn enumeration(name: &str, variants: Vec<EnumVariant>) -> Self {
        let layout = EnumLayout::compute(&variants);
        MirType::Enum {
            name: name.to_string(),
            variants,
            layout,
        }
    }

    /// Create a function pointer type
    pub fn fn_ptr(params: Vec<MirType>, ret: MirType) -> Self {
        MirType::FnPtr {
            params,
            ret: Box::new(ret),
            abi: CallingConvention::MedLang,
        }
    }

    /// Create a function pointer with specified ABI
    pub fn fn_ptr_with_abi(params: Vec<MirType>, ret: MirType, abi: CallingConvention) -> Self {
        MirType::FnPtr {
            params,
            ret: Box::new(ret),
            abi,
        }
    }

    /// Create a complex number type
    pub fn complex(element: MirType) -> Self {
        MirType::Complex {
            element: Box::new(element),
        }
    }

    /// Create an interval type
    pub fn interval(element: MirType) -> Self {
        MirType::Interval {
            element: Box::new(element),
        }
    }
}

// ============================================================================
// Display
// ============================================================================

impl fmt::Display for MirType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirType::Void => write!(f, "void"),
            MirType::Bool => write!(f, "bool"),
            MirType::I8 => write!(f, "i8"),
            MirType::I16 => write!(f, "i16"),
            MirType::I32 => write!(f, "i32"),
            MirType::I64 => write!(f, "i64"),
            MirType::I128 => write!(f, "i128"),
            MirType::U8 => write!(f, "u8"),
            MirType::U16 => write!(f, "u16"),
            MirType::U32 => write!(f, "u32"),
            MirType::U64 => write!(f, "u64"),
            MirType::U128 => write!(f, "u128"),
            MirType::F32 => write!(f, "f32"),
            MirType::F64 => write!(f, "f64"),
            MirType::F80 => write!(f, "f80"),
            MirType::F128 => write!(f, "f128"),
            MirType::DualF32 => write!(f, "dual32"),
            MirType::DualF64 => write!(f, "dual64"),
            MirType::DualVecF64 { size } => write!(f, "dual_vec64<{}>", size),
            MirType::Array { element, size } => write!(f, "[{}; {}]", element, size),
            MirType::Slice { element } => write!(f, "[{}]", element),
            MirType::Vector { element, size } => write!(f, "vec<{}, {}>", element, size),
            MirType::Matrix {
                element,
                rows,
                cols,
                layout,
            } => {
                let layout_str = match layout {
                    MatrixLayout::RowMajor => "",
                    MatrixLayout::ColMajor => ", col_major",
                };
                write!(f, "mat<{}, {}x{}{}>", element, rows, cols, layout_str)
            }
            MirType::SparseMatrix {
                element,
                rows,
                cols,
            } => {
                write!(f, "sparse_mat<{}, {}x{}>", element, rows, cols)
            }
            MirType::Struct { name, .. } => write!(f, "struct {}", name),
            MirType::Tuple { elements } => {
                write!(f, "(")?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, ")")
            }
            MirType::Enum { name, .. } => write!(f, "enum {}", name),
            MirType::FnPtr { params, ret, abi } => {
                write!(
                    f,
                    "fn({}) -> {} [{}]",
                    params
                        .iter()
                        .map(|p| p.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    ret,
                    abi
                )
            }
            MirType::Ptr {
                pointee,
                mutability,
            } => {
                write!(f, "*{} {}", mutability, pointee)
            }
            MirType::Ref {
                pointee,
                mutability,
                ..
            } => {
                write!(f, "&{} {}", mutability, pointee)
            }
            MirType::Box { pointee } => write!(f, "box {}", pointee),
            MirType::Complex { element } => write!(f, "complex<{}>", element),
            MirType::Interval { element } => write!(f, "interval<{}>", element),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_sizes() {
        assert_eq!(MirType::Void.size(), 0);
        assert_eq!(MirType::Bool.size(), 1);
        assert_eq!(MirType::I8.size(), 1);
        assert_eq!(MirType::I16.size(), 2);
        assert_eq!(MirType::I32.size(), 4);
        assert_eq!(MirType::I64.size(), 8);
        assert_eq!(MirType::I128.size(), 16);
        assert_eq!(MirType::F32.size(), 4);
        assert_eq!(MirType::F64.size(), 8);
    }

    #[test]
    fn test_primitive_alignment() {
        assert_eq!(MirType::Bool.align(), 1);
        assert_eq!(MirType::I32.align(), 4);
        assert_eq!(MirType::I64.align(), 8);
        assert_eq!(MirType::F64.align(), 8);
    }

    #[test]
    fn test_array_type() {
        let arr = MirType::array(MirType::F64, 10);
        assert_eq!(arr.size(), 80);
        assert_eq!(arr.align(), 8);
        assert!(arr.element_type().is_some());
    }

    #[test]
    fn test_struct_layout() {
        // Struct { i8, i64, i8 } should have padding
        let fields = vec![
            ("a".to_string(), MirType::I8),
            ("b".to_string(), MirType::I64),
            ("c".to_string(), MirType::I8),
        ];
        let layout = StructLayout::compute(&fields);

        assert_eq!(layout.field_offsets[0], 0); // a at 0
        assert_eq!(layout.field_offsets[1], 8); // b at 8 (aligned)
        assert_eq!(layout.field_offsets[2], 16); // c at 16
        assert_eq!(layout.size, 24); // padded to alignment
        assert_eq!(layout.align, 8);
    }

    #[test]
    fn test_enum_layout() {
        let variants = vec![
            EnumVariant::new("None", 0),
            EnumVariant::new("Some", 1).with_fields(vec![MirType::I64]),
        ];
        let layout = EnumLayout::compute(&variants);

        assert_eq!(layout.discriminant_size, 1); // 2 variants fit in 1 byte
        assert!(layout.payload_offset >= 1); // payload after discriminant
        assert!(layout.size >= 9); // discriminant + i64
    }

    #[test]
    fn test_dual_types() {
        assert_eq!(MirType::DualF64.size(), 16);
        assert_eq!(MirType::DualF32.size(), 8);
        assert!(MirType::DualF64.is_dual());
        assert!(MirType::DualF32.is_copy());
    }

    #[test]
    fn test_matrix_type() {
        let mat = MirType::matrix(MirType::F64, 3, 3);
        assert_eq!(mat.size(), 72); // 9 * 8
        assert_eq!(mat.align(), 32); // SIMD alignment
        assert!(mat.is_simd());
    }

    #[test]
    fn test_is_copy() {
        assert!(MirType::I32.is_copy());
        assert!(MirType::F64.is_copy());
        assert!(MirType::ptr(MirType::I32, false).is_copy());
        assert!(!MirType::boxed(MirType::I32).is_copy());
    }

    #[test]
    fn test_needs_drop() {
        assert!(!MirType::I32.needs_drop());
        assert!(MirType::boxed(MirType::I32).needs_drop());

        let struct_with_box = MirType::structure(
            "Test",
            vec![("value".to_string(), MirType::boxed(MirType::I32))],
        );
        assert!(struct_with_box.needs_drop());
    }

    #[test]
    fn test_type_display() {
        assert_eq!(MirType::I32.to_string(), "i32");
        assert_eq!(MirType::F64.to_string(), "f64");
        assert_eq!(MirType::array(MirType::F64, 10).to_string(), "[f64; 10]");
        assert_eq!(MirType::ptr(MirType::I32, true).to_string(), "*mut i32");
        assert_eq!(MirType::ptr(MirType::I32, false).to_string(), "*const i32");
    }

    #[test]
    fn test_function_pointer() {
        let fn_ty = MirType::fn_ptr(vec![MirType::F64, MirType::F64], MirType::F64);
        assert_eq!(fn_ty.size(), 8);
        assert!(fn_ty.is_copy());
    }

    #[test]
    fn test_complex_and_interval() {
        let complex = MirType::complex(MirType::F64);
        assert_eq!(complex.size(), 16);
        assert!(complex.is_copy());

        let interval = MirType::interval(MirType::F64);
        assert_eq!(interval.size(), 16);
        assert!(interval.is_copy());
    }
}
