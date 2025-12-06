//! Data Layout and ABI
//!
//! Defines the data layout and ABI (Application Binary Interface) for MIR.
//! This module handles type sizes, alignments, calling conventions, and
//! parameter passing rules.

use super::types::*;

/// Target data layout specification
#[derive(Clone, Debug)]
pub struct DataLayout {
    /// Endianness
    pub endianness: Endianness,
    /// Pointer size in bits
    pub pointer_size: u32,
    /// Pointer alignment in bytes
    pub pointer_align: u32,
    /// Integer type alignments (width -> alignment)
    pub int_alignments: Vec<(u32, u32)>,
    /// Float type alignments (width -> alignment)
    pub float_alignments: Vec<(u32, u32)>,
    /// Vector alignment (element_size, num_elements -> alignment)
    pub vector_alignments: Vec<(u32, u32, u32)>,
    /// Aggregate alignment minimum
    pub aggregate_align: u32,
    /// Stack alignment
    pub stack_align: u32,
    /// Native integer widths
    pub native_integers: Vec<u32>,
    /// Mangling style
    pub mangling: Mangling,
}

impl DataLayout {
    /// Create a default data layout (LP64, little-endian)
    pub fn default_lp64() -> Self {
        Self {
            endianness: Endianness::Little,
            pointer_size: 64,
            pointer_align: 8,
            int_alignments: vec![
                (1, 1),    // i1
                (8, 1),    // i8
                (16, 2),   // i16
                (32, 4),   // i32
                (64, 8),   // i64
                (128, 16), // i128
            ],
            float_alignments: vec![
                (32, 4),   // f32
                (64, 8),   // f64
                (80, 16),  // f80 (x87)
                (128, 16), // f128
            ],
            vector_alignments: vec![
                (64, 1, 8),   // 64-bit vectors
                (128, 1, 16), // 128-bit vectors (SSE)
                (256, 1, 32), // 256-bit vectors (AVX)
                (512, 1, 64), // 512-bit vectors (AVX-512)
            ],
            aggregate_align: 1,
            stack_align: 16,
            native_integers: vec![8, 16, 32, 64],
            mangling: Mangling::Itanium,
        }
    }

    /// Create x86_64 Linux data layout
    pub fn x86_64_linux() -> Self {
        Self::default_lp64()
    }

    /// Create x86_64 macOS data layout
    pub fn x86_64_macos() -> Self {
        let mut layout = Self::default_lp64();
        layout.mangling = Mangling::MachO;
        layout
    }

    /// Create x86_64 Windows data layout
    pub fn x86_64_windows() -> Self {
        let mut layout = Self::default_lp64();
        layout.mangling = Mangling::MSVC;
        layout
    }

    /// Create AArch64 data layout
    pub fn aarch64() -> Self {
        Self::default_lp64()
    }

    /// Create WASM32 data layout
    pub fn wasm32() -> Self {
        Self {
            endianness: Endianness::Little,
            pointer_size: 32,
            pointer_align: 4,
            int_alignments: vec![(1, 1), (8, 1), (16, 2), (32, 4), (64, 8), (128, 16)],
            float_alignments: vec![(32, 4), (64, 8)],
            vector_alignments: vec![(128, 1, 16)],
            aggregate_align: 1,
            stack_align: 16,
            native_integers: vec![32],
            mangling: Mangling::None,
        }
    }

    /// Get alignment for integer type with given bit width
    pub fn int_align(&self, bits: u32) -> u32 {
        self.int_alignments
            .iter()
            .find(|(w, _)| *w >= bits)
            .map(|(_, a)| *a)
            .unwrap_or(self.pointer_align)
    }

    /// Get alignment for float type with given bit width
    pub fn float_align(&self, bits: u32) -> u32 {
        self.float_alignments
            .iter()
            .find(|(w, _)| *w >= bits)
            .map(|(_, a)| *a)
            .unwrap_or(self.pointer_align)
    }

    /// Get alignment for vector type
    pub fn vector_align(&self, total_bits: u32) -> u32 {
        self.vector_alignments
            .iter()
            .find(|(b, _, _)| *b >= total_bits)
            .map(|(_, _, a)| *a)
            .unwrap_or(self.stack_align)
    }

    /// Get size and alignment for a MIR type
    pub fn type_layout(&self, ty: &MirType) -> TypeLayout {
        let size = self.type_size(ty);
        let align = self.type_align(ty);
        TypeLayout { size, align }
    }

    /// Get size of a type in bytes
    pub fn type_size(&self, ty: &MirType) -> usize {
        match ty {
            MirType::Void => 0,
            MirType::Bool | MirType::I8 | MirType::U8 => 1,
            MirType::I16 | MirType::U16 => 2,
            MirType::I32 | MirType::U32 | MirType::F32 => 4,
            MirType::I64 | MirType::U64 | MirType::F64 => 8,
            MirType::I128 | MirType::U128 | MirType::F128 => 16,
            MirType::F80 => 16, // Padded to 16 bytes
            MirType::DualF32 => 8,
            MirType::DualF64 => 16,
            MirType::DualVecF64 { size } => 16 * size,
            MirType::Array { element, size } => self.type_size(element) * size,
            MirType::Slice { .. } => (self.pointer_size / 8 * 2) as usize, // fat pointer
            MirType::Struct { fields, .. } => {
                let layout = self.struct_layout(fields);
                layout.size
            }
            MirType::Tuple { elements } => {
                if elements.is_empty() {
                    return 0;
                }
                let mut offset = 0;
                let mut max_align = 1;
                for elem in elements {
                    let elem_align = self.type_align(elem);
                    max_align = max_align.max(elem_align);
                    offset = align_to(offset, elem_align);
                    offset += self.type_size(elem);
                }
                align_to(offset, max_align)
            }
            MirType::Enum { variants, .. } => {
                let layout = self.enum_layout(variants);
                layout.size
            }
            MirType::FnPtr { .. }
            | MirType::Ptr { .. }
            | MirType::Ref { .. }
            | MirType::Box { .. } => (self.pointer_size / 8) as usize,
            MirType::Matrix {
                element,
                rows,
                cols,
                ..
            } => self.type_size(element) * rows * cols,
            MirType::Vector { element, size } => self.type_size(element) * size,
            MirType::SparseMatrix { .. } => 24, // ptr + indices + data
            MirType::Complex { element } => self.type_size(element) * 2,
            MirType::Interval { element } => self.type_size(element) * 2,
        }
    }

    /// Get alignment of a type in bytes
    pub fn type_align(&self, ty: &MirType) -> usize {
        match ty {
            MirType::Void => 1,
            MirType::Bool | MirType::I8 | MirType::U8 => 1,
            MirType::I16 | MirType::U16 => 2,
            MirType::I32 | MirType::U32 | MirType::F32 => 4,
            MirType::I64 | MirType::U64 | MirType::F64 => 8,
            MirType::I128 | MirType::U128 | MirType::F128 => 16,
            MirType::F80 => 16,
            MirType::DualF32 => 4,
            MirType::DualF64 => 8,
            MirType::DualVecF64 { .. } => 8,
            MirType::Array { element, .. } => self.type_align(element),
            MirType::Slice { element } => self.type_align(element).max(self.pointer_align as usize),
            MirType::Struct { fields, .. } => fields
                .iter()
                .map(|(_, ty)| self.type_align(ty))
                .max()
                .unwrap_or(1),
            MirType::Tuple { elements } => elements
                .iter()
                .map(|ty| self.type_align(ty))
                .max()
                .unwrap_or(1),
            MirType::Enum { variants, .. } => {
                let disc_align = if variants.len() <= 256 { 1 } else { 4 };
                let payload_align = variants
                    .iter()
                    .flat_map(|v| v.fields.iter())
                    .map(|ty| self.type_align(ty))
                    .max()
                    .unwrap_or(1);
                disc_align.max(payload_align)
            }
            MirType::FnPtr { .. }
            | MirType::Ptr { .. }
            | MirType::Ref { .. }
            | MirType::Box { .. } => self.pointer_align as usize,
            MirType::Matrix { element, .. } | MirType::Vector { element, .. } => {
                // SIMD alignment for vectors/matrices
                let elem_align = self.type_align(element);
                elem_align.max(32) // AVX alignment
            }
            MirType::SparseMatrix { .. } => self.pointer_align as usize,
            MirType::Complex { element } | MirType::Interval { element } => {
                self.type_align(element)
            }
        }
    }

    /// Compute struct layout with field offsets
    pub fn struct_layout(&self, fields: &[(String, MirType)]) -> StructFieldLayout {
        let mut offsets = Vec::with_capacity(fields.len());
        let mut current_offset = 0usize;
        let mut max_align = 1usize;

        for (_, ty) in fields {
            let align = self.type_align(ty);
            max_align = max_align.max(align);

            // Align current offset
            current_offset = align_to(current_offset, align);
            offsets.push(current_offset);
            current_offset += self.type_size(ty);
        }

        // Total size must be multiple of alignment
        let size = align_to(current_offset, max_align);

        StructFieldLayout {
            size,
            align: max_align,
            field_offsets: offsets,
        }
    }

    /// Compute enum layout
    pub fn enum_layout(&self, variants: &[EnumVariant]) -> EnumFieldLayout {
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
            let mut variant_size = 0usize;
            for ty in &variant.fields {
                let align = self.type_align(ty);
                variant_size = align_to(variant_size, align);
                variant_size += self.type_size(ty);
            }

            let variant_align = variant
                .fields
                .iter()
                .map(|ty| self.type_align(ty))
                .max()
                .unwrap_or(1);

            max_payload_size = max_payload_size.max(variant_size);
            max_align = max_align.max(variant_align);
        }

        // Layout: [discriminant][padding][payload]
        let discriminant_offset = 0;
        let payload_offset = align_to(discriminant_size, max_align);
        let size = align_to(payload_offset + max_payload_size, max_align);

        EnumFieldLayout {
            size,
            align: max_align,
            discriminant_offset,
            discriminant_size,
            payload_offset,
        }
    }
}

/// Endianness
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endianness {
    Little,
    Big,
}

/// Mangling style
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mangling {
    /// No mangling (WASM)
    None,
    /// Itanium C++ ABI (Linux, BSD)
    Itanium,
    /// Mach-O (macOS, iOS)
    MachO,
    /// MSVC (Windows)
    MSVC,
}

/// Type layout info
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypeLayout {
    pub size: usize,
    pub align: usize,
}

/// Struct field layout
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructFieldLayout {
    pub size: usize,
    pub align: usize,
    pub field_offsets: Vec<usize>,
}

/// Enum field layout
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumFieldLayout {
    pub size: usize,
    pub align: usize,
    pub discriminant_offset: usize,
    pub discriminant_size: usize,
    pub payload_offset: usize,
}

/// ABI parameter classification
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParamClass {
    /// Pass in integer register
    Integer,
    /// Pass in SSE/float register
    SSE,
    /// Pass in SSE (up part)
    SSEUp,
    /// Pass in x87 register
    X87,
    /// Pass in x87 (up part)
    X87Up,
    /// Pass in memory
    Memory,
    /// Complex x87
    ComplexX87,
    /// No class (void or zero-sized)
    NoClass,
}

/// ABI parameter info
#[derive(Clone, Debug)]
pub struct ParamInfo {
    /// Parameter type
    pub ty: MirType,
    /// Classification for first 8 bytes
    pub class_lo: ParamClass,
    /// Classification for next 8 bytes (if large type)
    pub class_hi: Option<ParamClass>,
    /// Direct pass in register(s)?
    pub direct: bool,
    /// Indirect (by pointer)?
    pub indirect: bool,
    /// Coerce to different type?
    pub coerce_to: Option<MirType>,
    /// Register(s) to use
    pub registers: Vec<Register>,
}

impl ParamInfo {
    pub fn integer(ty: MirType) -> Self {
        Self {
            ty,
            class_lo: ParamClass::Integer,
            class_hi: None,
            direct: true,
            indirect: false,
            coerce_to: None,
            registers: vec![],
        }
    }

    pub fn sse(ty: MirType) -> Self {
        Self {
            ty,
            class_lo: ParamClass::SSE,
            class_hi: None,
            direct: true,
            indirect: false,
            coerce_to: None,
            registers: vec![],
        }
    }

    pub fn memory(ty: MirType) -> Self {
        Self {
            ty,
            class_lo: ParamClass::Memory,
            class_hi: None,
            direct: false,
            indirect: true,
            coerce_to: None,
            registers: vec![],
        }
    }
}

/// Register specification
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Register {
    /// Integer registers
    RAX,
    RBX,
    RCX,
    RDX,
    RSI,
    RDI,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
    /// SSE registers
    XMM0,
    XMM1,
    XMM2,
    XMM3,
    XMM4,
    XMM5,
    XMM6,
    XMM7,
    XMM8,
    XMM9,
    XMM10,
    XMM11,
    XMM12,
    XMM13,
    XMM14,
    XMM15,
    /// Stack slot
    Stack(i32),
}

/// ABI classifier for x86_64 System V
pub struct SystemVABI {
    layout: DataLayout,
}

impl SystemVABI {
    pub fn new() -> Self {
        Self {
            layout: DataLayout::x86_64_linux(),
        }
    }

    pub fn with_layout(layout: DataLayout) -> Self {
        Self { layout }
    }

    /// Classify a type for parameter passing
    pub fn classify(&self, ty: &MirType) -> ParamInfo {
        let size = self.layout.type_size(ty);

        // Types > 16 bytes or with unaligned fields go in memory
        if size > 16 {
            return ParamInfo::memory(ty.clone());
        }

        match ty {
            // Integers
            MirType::Bool
            | MirType::I8
            | MirType::U8
            | MirType::I16
            | MirType::U16
            | MirType::I32
            | MirType::U32
            | MirType::I64
            | MirType::U64 => ParamInfo::integer(ty.clone()),

            // 128-bit integers - two integer registers
            MirType::I128 | MirType::U128 => {
                let mut info = ParamInfo::integer(ty.clone());
                info.class_hi = Some(ParamClass::Integer);
                info
            }

            // Floats
            MirType::F32 | MirType::F64 => ParamInfo::sse(ty.clone()),

            // Extended precision
            MirType::F80 => {
                let mut info = ParamInfo {
                    ty: ty.clone(),
                    class_lo: ParamClass::X87,
                    class_hi: Some(ParamClass::X87Up),
                    direct: false,
                    indirect: true,
                    coerce_to: None,
                    registers: vec![],
                };
                info
            }

            // Quad precision
            MirType::F128 => {
                let mut info = ParamInfo::sse(ty.clone());
                info.class_hi = Some(ParamClass::SSEUp);
                info
            }

            // Pointers
            MirType::Ptr { .. }
            | MirType::Ref { .. }
            | MirType::Box { .. }
            | MirType::FnPtr { .. } => ParamInfo::integer(ty.clone()),

            // Dual numbers
            MirType::DualF32 => {
                let mut info = ParamInfo::sse(ty.clone());
                info.coerce_to = Some(MirType::F64); // Pack into single register
                info
            }
            MirType::DualF64 => {
                let mut info = ParamInfo::sse(ty.clone());
                info.class_hi = Some(ParamClass::SSE);
                info
            }

            // Complex numbers
            MirType::Complex { element } => {
                let elem_size = self.layout.type_size(element);
                if elem_size <= 8 {
                    // Two SSE registers
                    let mut info = ParamInfo::sse(ty.clone());
                    info.class_hi = Some(ParamClass::SSE);
                    info
                } else {
                    ParamInfo::memory(ty.clone())
                }
            }

            // Structs - need field-by-field analysis
            MirType::Struct { fields, .. } => self.classify_struct(ty, fields),

            // Tuples - similar to structs
            MirType::Tuple { elements } => self.classify_tuple(ty, elements),

            // Arrays - memory for large, registers for small
            MirType::Array {
                element,
                size: arr_size,
            } => {
                let total_size = self.layout.type_size(element) * arr_size;
                if total_size <= 16 && element.is_float() {
                    ParamInfo::sse(ty.clone())
                } else if total_size <= 16 {
                    ParamInfo::integer(ty.clone())
                } else {
                    ParamInfo::memory(ty.clone())
                }
            }

            // Everything else goes in memory
            _ => ParamInfo::memory(ty.clone()),
        }
    }

    fn classify_struct(&self, ty: &MirType, fields: &[(String, MirType)]) -> ParamInfo {
        if fields.is_empty() {
            return ParamInfo {
                ty: ty.clone(),
                class_lo: ParamClass::NoClass,
                class_hi: None,
                direct: true,
                indirect: false,
                coerce_to: None,
                registers: vec![],
            };
        }

        let size = self.layout.type_size(ty);
        if size > 16 {
            return ParamInfo::memory(ty.clone());
        }

        // Classify each 8-byte chunk
        let mut class_lo = ParamClass::NoClass;
        let mut class_hi = ParamClass::NoClass;

        let layout = self.layout.struct_layout(fields);

        for (i, (_, field_ty)) in fields.iter().enumerate() {
            let offset = layout.field_offsets[i];
            let field_class = self.field_class(field_ty);

            if offset < 8 {
                class_lo = self.merge_classes(class_lo, field_class);
            } else {
                class_hi = self.merge_classes(class_hi, field_class);
            }
        }

        // Post-merger cleanup
        if class_lo == ParamClass::Memory || class_hi == ParamClass::Memory {
            return ParamInfo::memory(ty.clone());
        }

        ParamInfo {
            ty: ty.clone(),
            class_lo,
            class_hi: if size > 8 { Some(class_hi) } else { None },
            direct: true,
            indirect: false,
            coerce_to: None,
            registers: vec![],
        }
    }

    fn classify_tuple(&self, ty: &MirType, elements: &[MirType]) -> ParamInfo {
        let fields: Vec<(String, MirType)> = elements
            .iter()
            .enumerate()
            .map(|(i, e)| (format!("{}", i), e.clone()))
            .collect();
        self.classify_struct(ty, &fields)
    }

    fn field_class(&self, ty: &MirType) -> ParamClass {
        match ty {
            MirType::F32 | MirType::F64 | MirType::F128 => ParamClass::SSE,
            MirType::F80 => ParamClass::X87,
            _ if ty.is_integer() || ty.is_pointer() => ParamClass::Integer,
            _ => ParamClass::Memory,
        }
    }

    fn merge_classes(&self, a: ParamClass, b: ParamClass) -> ParamClass {
        if a == b {
            return a;
        }

        if a == ParamClass::NoClass {
            return b;
        }
        if b == ParamClass::NoClass {
            return a;
        }

        if a == ParamClass::Memory || b == ParamClass::Memory {
            return ParamClass::Memory;
        }

        if a == ParamClass::Integer || b == ParamClass::Integer {
            return ParamClass::Integer;
        }

        if a == ParamClass::X87
            || a == ParamClass::X87Up
            || a == ParamClass::ComplexX87
            || b == ParamClass::X87
            || b == ParamClass::X87Up
            || b == ParamClass::ComplexX87
        {
            return ParamClass::Memory;
        }

        ParamClass::SSE
    }

    /// Compute argument locations for a function call
    pub fn compute_call_args(&self, params: &[MirType]) -> Vec<(ParamInfo, Option<Register>)> {
        let mut result = Vec::new();
        let mut int_reg_idx = 0;
        let mut sse_reg_idx = 0;
        let mut stack_offset = 0i32;

        let int_regs = [
            Register::RDI,
            Register::RSI,
            Register::RDX,
            Register::RCX,
            Register::R8,
            Register::R9,
        ];
        let sse_regs = [
            Register::XMM0,
            Register::XMM1,
            Register::XMM2,
            Register::XMM3,
            Register::XMM4,
            Register::XMM5,
            Register::XMM6,
            Register::XMM7,
        ];

        for ty in params {
            let info = self.classify(ty);
            let size = self.layout.type_size(ty);

            let reg = match (&info.class_lo, info.indirect) {
                (ParamClass::Integer, false) if int_reg_idx < int_regs.len() => {
                    let r = Some(int_regs[int_reg_idx]);
                    int_reg_idx += 1;
                    if info.class_hi == Some(ParamClass::Integer) && int_reg_idx < int_regs.len() {
                        int_reg_idx += 1;
                    }
                    r
                }
                (ParamClass::SSE, false) if sse_reg_idx < sse_regs.len() => {
                    let r = Some(sse_regs[sse_reg_idx]);
                    sse_reg_idx += 1;
                    if info.class_hi == Some(ParamClass::SSE) && sse_reg_idx < sse_regs.len() {
                        sse_reg_idx += 1;
                    }
                    r
                }
                _ => {
                    let r = Some(Register::Stack(stack_offset));
                    stack_offset += align_to(size, 8) as i32;
                    r
                }
            };

            result.push((info, reg));
        }

        result
    }

    /// Compute return value location
    pub fn compute_return(&self, ty: &MirType) -> ParamInfo {
        let mut info = self.classify(ty);

        // If returning in memory, caller passes pointer in RDI
        if info.indirect {
            info.registers = vec![Register::RDI];
        } else if info.class_lo == ParamClass::Integer {
            info.registers = vec![Register::RAX];
            if info.class_hi == Some(ParamClass::Integer) {
                info.registers.push(Register::RDX);
            }
        } else if info.class_lo == ParamClass::SSE {
            info.registers = vec![Register::XMM0];
            if info.class_hi == Some(ParamClass::SSE) {
                info.registers.push(Register::XMM1);
            }
        }

        info
    }
}

impl Default for SystemVABI {
    fn default() -> Self {
        Self::new()
    }
}

/// Align a value up to the given alignment
fn align_to(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_layout_sizes() {
        let layout = DataLayout::x86_64_linux();

        assert_eq!(layout.type_size(&MirType::I8), 1);
        assert_eq!(layout.type_size(&MirType::I32), 4);
        assert_eq!(layout.type_size(&MirType::I64), 8);
        assert_eq!(layout.type_size(&MirType::F64), 8);
        assert_eq!(layout.type_size(&MirType::I128), 16);
    }

    #[test]
    fn test_data_layout_alignments() {
        let layout = DataLayout::x86_64_linux();

        assert_eq!(layout.type_align(&MirType::I8), 1);
        assert_eq!(layout.type_align(&MirType::I32), 4);
        assert_eq!(layout.type_align(&MirType::I64), 8);
        assert_eq!(layout.type_align(&MirType::F64), 8);
    }

    #[test]
    fn test_struct_layout() {
        let layout = DataLayout::x86_64_linux();

        // Struct { i8, i64, i8 } with padding
        let fields = vec![
            ("a".to_string(), MirType::I8),
            ("b".to_string(), MirType::I64),
            ("c".to_string(), MirType::I8),
        ];

        let struct_layout = layout.struct_layout(&fields);

        assert_eq!(struct_layout.field_offsets[0], 0); // a at 0
        assert_eq!(struct_layout.field_offsets[1], 8); // b at 8 (aligned)
        assert_eq!(struct_layout.field_offsets[2], 16); // c at 16
        assert_eq!(struct_layout.size, 24); // padded to 8
        assert_eq!(struct_layout.align, 8);
    }

    #[test]
    fn test_abi_classify_primitives() {
        let abi = SystemVABI::new();

        let int_info = abi.classify(&MirType::I64);
        assert_eq!(int_info.class_lo, ParamClass::Integer);
        assert!(int_info.direct);

        let float_info = abi.classify(&MirType::F64);
        assert_eq!(float_info.class_lo, ParamClass::SSE);
        assert!(float_info.direct);

        let ptr_info = abi.classify(&MirType::ptr(MirType::I32, false));
        assert_eq!(ptr_info.class_lo, ParamClass::Integer);
    }

    #[test]
    fn test_abi_classify_structs() {
        let abi = SystemVABI::new();

        // Small struct - registers
        let small_struct = MirType::structure(
            "Small",
            vec![
                ("x".to_string(), MirType::I32),
                ("y".to_string(), MirType::I32),
            ],
        );
        let small_info = abi.classify(&small_struct);
        assert_eq!(small_info.class_lo, ParamClass::Integer);
        assert!(small_info.direct);

        // Large struct - memory
        let large_struct = MirType::structure(
            "Large",
            vec![
                ("a".to_string(), MirType::I64),
                ("b".to_string(), MirType::I64),
                ("c".to_string(), MirType::I64),
            ],
        );
        let large_info = abi.classify(&large_struct);
        assert_eq!(large_info.class_lo, ParamClass::Memory);
        assert!(large_info.indirect);
    }

    #[test]
    fn test_call_arg_allocation() {
        let abi = SystemVABI::new();

        let params = vec![
            MirType::I64, // RDI
            MirType::I64, // RSI
            MirType::F64, // XMM0
            MirType::I64, // RDX
            MirType::F64, // XMM1
        ];

        let args = abi.compute_call_args(&params);

        assert_eq!(args[0].1, Some(Register::RDI));
        assert_eq!(args[1].1, Some(Register::RSI));
        assert_eq!(args[2].1, Some(Register::XMM0));
        assert_eq!(args[3].1, Some(Register::RDX));
        assert_eq!(args[4].1, Some(Register::XMM1));
    }

    #[test]
    fn test_return_classification() {
        let abi = SystemVABI::new();

        let int_ret = abi.compute_return(&MirType::I64);
        assert!(int_ret.registers.contains(&Register::RAX));

        let float_ret = abi.compute_return(&MirType::F64);
        assert!(float_ret.registers.contains(&Register::XMM0));
    }

    #[test]
    fn test_wasm_layout() {
        let layout = DataLayout::wasm32();

        assert_eq!(layout.pointer_size, 32);
        assert_eq!(layout.type_size(&MirType::ptr(MirType::I32, false)), 4);
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(0, 8), 0);
        assert_eq!(align_to(1, 8), 8);
        assert_eq!(align_to(7, 8), 8);
        assert_eq!(align_to(8, 8), 8);
        assert_eq!(align_to(9, 8), 16);
    }
}
