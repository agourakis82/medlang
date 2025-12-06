//! MIR Instructions (SSA Form)
//!
//! Defines all MIR operations, organized by category:
//! - Constants
//! - Integer arithmetic
//! - Floating-point arithmetic
//! - Mathematical functions (intrinsics)
//! - Bitwise operations
//! - Comparisons
//! - Type conversions
//! - Memory operations
//! - Aggregate operations
//! - Control flow
//! - Automatic differentiation
//! - Vector/matrix operations
//! - Probability distributions

use super::types::*;
use super::value::*;

/// A MIR instruction
#[derive(Clone, Debug)]
pub struct Instruction {
    /// Result value (None for void operations like Store)
    pub result: Option<ValueId>,
    /// The operation
    pub op: Operation,
    /// Result type
    pub ty: MirType,
    /// Source location
    pub span: Option<Span>,
}

impl Instruction {
    pub fn new(op: Operation, ty: MirType) -> Self {
        Self {
            result: None,
            op,
            ty,
            span: None,
        }
    }

    pub fn with_result(mut self, result: ValueId) -> Self {
        self.result = Some(result);
        self
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Check if this instruction has side effects
    pub fn has_side_effects(&self) -> bool {
        self.op.has_side_effects()
    }

    /// Check if this instruction is pure (no side effects, same inputs = same output)
    pub fn is_pure(&self) -> bool {
        self.op.is_pure()
    }

    /// Get all value operands used by this instruction
    pub fn operands(&self) -> Vec<ValueId> {
        self.op.operands()
    }
}

/// MIR operations
#[derive(Clone, Debug)]
pub enum Operation {
    // =========================================================================
    // CONSTANTS
    // =========================================================================
    /// Integer constant
    ConstInt { value: i128, ty: MirType },

    /// Floating-point constant
    ConstFloat { value: f64, ty: MirType },

    /// Boolean constant
    ConstBool { value: bool },

    /// Zero-initialized value
    ZeroInit { ty: MirType },

    /// Undefined value (for uninitialized locals)
    Undef { ty: MirType },

    // =========================================================================
    // INTEGER ARITHMETIC
    // =========================================================================
    /// Integer addition (may trap on overflow in checked mode)
    IAdd { lhs: ValueId, rhs: ValueId },

    /// Wrapping integer addition
    IAddWrap { lhs: ValueId, rhs: ValueId },

    /// Saturating integer addition
    IAddSat { lhs: ValueId, rhs: ValueId },

    /// Integer subtraction
    ISub { lhs: ValueId, rhs: ValueId },

    /// Wrapping integer subtraction
    ISubWrap { lhs: ValueId, rhs: ValueId },

    /// Integer multiplication
    IMul { lhs: ValueId, rhs: ValueId },

    /// Wrapping integer multiplication
    IMulWrap { lhs: ValueId, rhs: ValueId },

    /// Signed integer division
    IDiv { lhs: ValueId, rhs: ValueId },

    /// Unsigned integer division
    UDiv { lhs: ValueId, rhs: ValueId },

    /// Signed integer remainder
    IRem { lhs: ValueId, rhs: ValueId },

    /// Unsigned integer remainder
    URem { lhs: ValueId, rhs: ValueId },

    /// Integer negation
    INeg { operand: ValueId },

    // =========================================================================
    // FLOATING-POINT ARITHMETIC
    // =========================================================================
    /// Floating-point addition
    FAdd { lhs: ValueId, rhs: ValueId },

    /// Floating-point subtraction
    FSub { lhs: ValueId, rhs: ValueId },

    /// Floating-point multiplication
    FMul { lhs: ValueId, rhs: ValueId },

    /// Floating-point division
    FDiv { lhs: ValueId, rhs: ValueId },

    /// Floating-point remainder
    FRem { lhs: ValueId, rhs: ValueId },

    /// Floating-point negation
    FNeg { operand: ValueId },

    /// Fused multiply-add: a * b + c (single rounding)
    FMA { a: ValueId, b: ValueId, c: ValueId },

    // =========================================================================
    // MATHEMATICAL FUNCTIONS (INTRINSICS)
    // =========================================================================
    /// Square root
    Sqrt { operand: ValueId },

    /// Power: base^exp
    Pow { base: ValueId, exp: ValueId },

    /// Exponential: e^x
    Exp { operand: ValueId },

    /// Exp minus 1: e^x - 1 (accurate for small x)
    Expm1 { operand: ValueId },

    /// Natural logarithm: ln(x)
    Log { operand: ValueId },

    /// Log of 1+x: ln(1+x) (accurate for small x)
    Log1p { operand: ValueId },

    /// Log base 10
    Log10 { operand: ValueId },

    /// Log base 2
    Log2 { operand: ValueId },

    /// Sine
    Sin { operand: ValueId },

    /// Cosine
    Cos { operand: ValueId },

    /// Tangent
    Tan { operand: ValueId },

    /// Arc sine
    Asin { operand: ValueId },

    /// Arc cosine
    Acos { operand: ValueId },

    /// Arc tangent
    Atan { operand: ValueId },

    /// Two-argument arc tangent
    Atan2 { y: ValueId, x: ValueId },

    /// Hyperbolic sine
    Sinh { operand: ValueId },

    /// Hyperbolic cosine
    Cosh { operand: ValueId },

    /// Hyperbolic tangent
    Tanh { operand: ValueId },

    /// Inverse hyperbolic sine
    Asinh { operand: ValueId },

    /// Inverse hyperbolic cosine
    Acosh { operand: ValueId },

    /// Inverse hyperbolic tangent
    Atanh { operand: ValueId },

    /// Absolute value
    Abs { operand: ValueId },

    /// Floor
    Floor { operand: ValueId },

    /// Ceiling
    Ceil { operand: ValueId },

    /// Round to nearest
    Round { operand: ValueId },

    /// Truncate toward zero
    Trunc { operand: ValueId },

    /// Minimum (NaN-propagating)
    FMin { lhs: ValueId, rhs: ValueId },

    /// Maximum (NaN-propagating)
    FMax { lhs: ValueId, rhs: ValueId },

    /// Copy sign
    CopySign { magnitude: ValueId, sign: ValueId },

    // =========================================================================
    // SPECIAL MATHEMATICAL FUNCTIONS
    // =========================================================================
    /// Gamma function
    Gamma { operand: ValueId },

    /// Log-gamma function
    LogGamma { operand: ValueId },

    /// Digamma function (psi)
    Digamma { operand: ValueId },

    /// Error function
    Erf { operand: ValueId },

    /// Complementary error function
    Erfc { operand: ValueId },

    /// Inverse error function
    ErfInv { operand: ValueId },

    /// Bessel function J0
    BesselJ0 { operand: ValueId },

    /// Bessel function J1
    BesselJ1 { operand: ValueId },

    /// Modified Bessel function I0
    BesselI0 { operand: ValueId },

    /// Beta function
    Beta { a: ValueId, b: ValueId },

    /// Incomplete beta function
    BetaInc { a: ValueId, b: ValueId, x: ValueId },

    /// Regularized incomplete beta function
    BetaIncReg { a: ValueId, b: ValueId, x: ValueId },

    /// Incomplete gamma function
    GammaInc { a: ValueId, x: ValueId },

    /// Regularized incomplete gamma function
    GammaIncReg { a: ValueId, x: ValueId },

    // =========================================================================
    // BITWISE OPERATIONS
    // =========================================================================
    /// Bitwise AND
    And { lhs: ValueId, rhs: ValueId },

    /// Bitwise OR
    Or { lhs: ValueId, rhs: ValueId },

    /// Bitwise XOR
    Xor { lhs: ValueId, rhs: ValueId },

    /// Bitwise NOT
    Not { operand: ValueId },

    /// Shift left
    Shl { lhs: ValueId, rhs: ValueId },

    /// Logical shift right
    LShr { lhs: ValueId, rhs: ValueId },

    /// Arithmetic shift right
    AShr { lhs: ValueId, rhs: ValueId },

    /// Count leading zeros
    Clz { operand: ValueId },

    /// Count trailing zeros
    Ctz { operand: ValueId },

    /// Population count
    Popcnt { operand: ValueId },

    /// Byte swap
    Bswap { operand: ValueId },

    /// Bit reverse
    Bitreverse { operand: ValueId },

    // =========================================================================
    // COMPARISON
    // =========================================================================
    /// Integer comparison
    ICmp {
        pred: IntPredicate,
        lhs: ValueId,
        rhs: ValueId,
    },

    /// Floating-point comparison
    FCmp {
        pred: FloatPredicate,
        lhs: ValueId,
        rhs: ValueId,
    },

    /// Check if float is NaN
    IsNaN { operand: ValueId },

    /// Check if float is infinite
    IsInf { operand: ValueId },

    /// Check if float is finite
    IsFinite { operand: ValueId },

    // =========================================================================
    // TYPE CONVERSIONS
    // =========================================================================
    /// Sign extend integer
    SExt { operand: ValueId, to: MirType },

    /// Zero extend integer
    ZExt { operand: ValueId, to: MirType },

    /// Truncate integer
    ITrunc { operand: ValueId, to: MirType },

    /// Float to larger float
    FExt { operand: ValueId, to: MirType },

    /// Float to smaller float
    FTrunc { operand: ValueId, to: MirType },

    /// Signed integer to float
    SIToFP { operand: ValueId, to: MirType },

    /// Unsigned integer to float
    UIToFP { operand: ValueId, to: MirType },

    /// Float to signed integer
    FPToSI { operand: ValueId, to: MirType },

    /// Float to unsigned integer
    FPToUI { operand: ValueId, to: MirType },

    /// Pointer to integer
    PtrToInt { operand: ValueId, to: MirType },

    /// Integer to pointer
    IntToPtr { operand: ValueId, to: MirType },

    /// Bitcast (reinterpret bits)
    Bitcast { operand: ValueId, to: MirType },

    // =========================================================================
    // MEMORY OPERATIONS
    // =========================================================================
    /// Stack allocation
    Alloca {
        ty: MirType,
        count: Option<ValueId>,
        align: usize,
    },

    /// Load from memory
    Load {
        ptr: ValueId,
        ty: MirType,
        volatile: bool,
        align: usize,
    },

    /// Store to memory
    Store {
        ptr: ValueId,
        value: ValueId,
        volatile: bool,
        align: usize,
    },

    /// Get element pointer (GEP)
    GetElementPtr {
        base: ValueId,
        indices: Vec<ValueId>,
        inbounds: bool,
    },

    /// Memory copy
    Memcpy {
        dst: ValueId,
        src: ValueId,
        len: ValueId,
        volatile: bool,
    },

    /// Memory set
    Memset {
        dst: ValueId,
        val: ValueId,
        len: ValueId,
        volatile: bool,
    },

    /// Memory move (handles overlap)
    Memmove {
        dst: ValueId,
        src: ValueId,
        len: ValueId,
        volatile: bool,
    },

    // =========================================================================
    // AGGREGATE OPERATIONS
    // =========================================================================
    /// Extract struct field
    ExtractField { aggregate: ValueId, field_idx: u32 },

    /// Insert struct field (returns new aggregate)
    InsertField {
        aggregate: ValueId,
        field_idx: u32,
        value: ValueId,
    },

    /// Extract array/vector element
    ExtractElement { aggregate: ValueId, index: ValueId },

    /// Insert array/vector element
    InsertElement {
        aggregate: ValueId,
        index: ValueId,
        value: ValueId,
    },

    /// Construct aggregate from values
    Aggregate { ty: MirType, values: Vec<ValueId> },

    // =========================================================================
    // CONTROL FLOW
    // =========================================================================
    /// Select (ternary conditional)
    Select {
        cond: ValueId,
        then_val: ValueId,
        else_val: ValueId,
    },

    /// Function call
    Call {
        callee: Callee,
        args: Vec<ValueId>,
        ret_ty: MirType,
    },

    // =========================================================================
    // AUTOMATIC DIFFERENTIATION
    // =========================================================================
    /// Create dual number: (value, derivative)
    MakeDual { value: ValueId, derivative: ValueId },

    /// Extract primal from dual
    DualPrimal { dual: ValueId },

    /// Extract derivative from dual
    DualTangent { dual: ValueId },

    /// Dual addition
    DualAdd { lhs: ValueId, rhs: ValueId },

    /// Dual subtraction
    DualSub { lhs: ValueId, rhs: ValueId },

    /// Dual multiplication (product rule)
    DualMul { lhs: ValueId, rhs: ValueId },

    /// Dual division (quotient rule)
    DualDiv { lhs: ValueId, rhs: ValueId },

    /// Dual sine (with derivative)
    DualSin { operand: ValueId },

    /// Dual cosine (with derivative)
    DualCos { operand: ValueId },

    /// Dual exponential (with derivative)
    DualExp { operand: ValueId },

    /// Dual logarithm (with derivative)
    DualLog { operand: ValueId },

    /// Dual square root (with derivative)
    DualSqrt { operand: ValueId },

    /// Dual power (with derivative)
    DualPow { base: ValueId, exp: ValueId },

    /// Dual tanh (with derivative)
    DualTanh { operand: ValueId },

    // =========================================================================
    // VECTOR/MATRIX OPERATIONS
    // =========================================================================
    /// Vector dot product
    VecDot { lhs: ValueId, rhs: ValueId },

    /// Vector norm (L2)
    VecNorm { vec: ValueId },

    /// Vector normalize
    VecNormalize { vec: ValueId },

    /// Matrix-vector multiply
    MatVecMul { mat: ValueId, vec: ValueId },

    /// Matrix-matrix multiply
    MatMul { lhs: ValueId, rhs: ValueId },

    /// Matrix transpose
    MatTranspose { mat: ValueId },

    /// Matrix inverse
    MatInverse { mat: ValueId },

    /// Matrix determinant
    MatDet { mat: ValueId },

    /// Matrix trace
    MatTrace { mat: ValueId },

    /// Cholesky decomposition
    MatCholesky { mat: ValueId },

    /// LU decomposition
    MatLU { mat: ValueId },

    /// QR decomposition
    MatQR { mat: ValueId },

    /// Eigendecomposition
    MatEigen { mat: ValueId },

    /// Singular value decomposition
    MatSVD { mat: ValueId },

    /// Solve linear system: Ax = b
    MatSolve { a: ValueId, b: ValueId },

    /// Vector element-wise unary operation
    VecMap { vec: ValueId, op: UnaryOp },

    /// Vector element-wise binary operation
    VecBinOp {
        lhs: ValueId,
        rhs: ValueId,
        op: BinaryOp,
    },

    /// Vector reduce (sum, prod, min, max)
    VecReduce { vec: ValueId, op: ReduceOp },

    /// SIMD shuffle
    VecShuffle {
        vec1: ValueId,
        vec2: ValueId,
        mask: Vec<i32>,
    },

    /// SIMD broadcast scalar to vector
    VecBroadcast { scalar: ValueId, len: usize },

    // =========================================================================
    // PROBABILITY DISTRIBUTIONS
    // =========================================================================
    /// Log probability density/mass
    LogPDF {
        distribution: DistributionKind,
        value: ValueId,
        params: Vec<ValueId>,
    },

    /// Cumulative distribution function
    CDF {
        distribution: DistributionKind,
        value: ValueId,
        params: Vec<ValueId>,
    },

    /// Quantile function (inverse CDF)
    Quantile {
        distribution: DistributionKind,
        prob: ValueId,
        params: Vec<ValueId>,
    },

    /// Random sample (requires RNG state)
    Sample {
        distribution: DistributionKind,
        params: Vec<ValueId>,
        rng: ValueId,
    },

    // =========================================================================
    // SPECIAL OPERATIONS
    // =========================================================================
    /// Assert (trap if false)
    Assert { cond: ValueId, message: String },

    /// Assume (hint to optimizer)
    Assume { cond: ValueId },

    /// Unreachable (UB if reached)
    Unreachable,

    /// Fence (memory barrier)
    Fence { ordering: AtomicOrdering },

    /// Atomic load
    AtomicLoad {
        ptr: ValueId,
        ty: MirType,
        ordering: AtomicOrdering,
    },

    /// Atomic store
    AtomicStore {
        ptr: ValueId,
        value: ValueId,
        ordering: AtomicOrdering,
    },

    /// Atomic compare-and-swap
    AtomicCAS {
        ptr: ValueId,
        expected: ValueId,
        desired: ValueId,
        success_ordering: AtomicOrdering,
        failure_ordering: AtomicOrdering,
    },

    /// Atomic read-modify-write
    AtomicRMW {
        ptr: ValueId,
        value: ValueId,
        op: AtomicRMWOp,
        ordering: AtomicOrdering,
    },

    /// No-op (placeholder)
    Nop,
}

impl Operation {
    /// Check if operation has side effects
    pub fn has_side_effects(&self) -> bool {
        matches!(
            self,
            Operation::Store { .. }
                | Operation::Memcpy { .. }
                | Operation::Memset { .. }
                | Operation::Memmove { .. }
                | Operation::Call { .. }
                | Operation::AtomicStore { .. }
                | Operation::AtomicRMW { .. }
                | Operation::AtomicCAS { .. }
                | Operation::Fence { .. }
                | Operation::Assert { .. }
                | Operation::Sample { .. }
        )
    }

    /// Check if operation is pure
    pub fn is_pure(&self) -> bool {
        !self.has_side_effects()
            && !matches!(
                self,
                Operation::Load { volatile: true, .. }
                    | Operation::AtomicLoad { .. }
                    | Operation::Undef { .. }
            )
    }

    /// Get all value operands
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            // Constants - no operands
            Operation::ConstInt { .. }
            | Operation::ConstFloat { .. }
            | Operation::ConstBool { .. }
            | Operation::ZeroInit { .. }
            | Operation::Undef { .. }
            | Operation::Unreachable
            | Operation::Nop => vec![],

            // Unary
            Operation::INeg { operand }
            | Operation::FNeg { operand }
            | Operation::Sqrt { operand }
            | Operation::Exp { operand }
            | Operation::Expm1 { operand }
            | Operation::Log { operand }
            | Operation::Log1p { operand }
            | Operation::Log10 { operand }
            | Operation::Log2 { operand }
            | Operation::Sin { operand }
            | Operation::Cos { operand }
            | Operation::Tan { operand }
            | Operation::Asin { operand }
            | Operation::Acos { operand }
            | Operation::Atan { operand }
            | Operation::Sinh { operand }
            | Operation::Cosh { operand }
            | Operation::Tanh { operand }
            | Operation::Asinh { operand }
            | Operation::Acosh { operand }
            | Operation::Atanh { operand }
            | Operation::Abs { operand }
            | Operation::Floor { operand }
            | Operation::Ceil { operand }
            | Operation::Round { operand }
            | Operation::Trunc { operand }
            | Operation::Gamma { operand }
            | Operation::LogGamma { operand }
            | Operation::Digamma { operand }
            | Operation::Erf { operand }
            | Operation::Erfc { operand }
            | Operation::ErfInv { operand }
            | Operation::BesselJ0 { operand }
            | Operation::BesselJ1 { operand }
            | Operation::BesselI0 { operand }
            | Operation::Not { operand }
            | Operation::Clz { operand }
            | Operation::Ctz { operand }
            | Operation::Popcnt { operand }
            | Operation::Bswap { operand }
            | Operation::Bitreverse { operand }
            | Operation::IsNaN { operand }
            | Operation::IsInf { operand }
            | Operation::IsFinite { operand }
            | Operation::SExt { operand, .. }
            | Operation::ZExt { operand, .. }
            | Operation::ITrunc { operand, .. }
            | Operation::FExt { operand, .. }
            | Operation::FTrunc { operand, .. }
            | Operation::SIToFP { operand, .. }
            | Operation::UIToFP { operand, .. }
            | Operation::FPToSI { operand, .. }
            | Operation::FPToUI { operand, .. }
            | Operation::PtrToInt { operand, .. }
            | Operation::IntToPtr { operand, .. }
            | Operation::Bitcast { operand, .. }
            | Operation::DualPrimal { dual: operand }
            | Operation::DualTangent { dual: operand }
            | Operation::DualSin { operand }
            | Operation::DualCos { operand }
            | Operation::DualExp { operand }
            | Operation::DualLog { operand }
            | Operation::DualSqrt { operand }
            | Operation::DualTanh { operand }
            | Operation::VecNorm { vec: operand }
            | Operation::VecNormalize { vec: operand }
            | Operation::MatTranspose { mat: operand }
            | Operation::MatInverse { mat: operand }
            | Operation::MatDet { mat: operand }
            | Operation::MatTrace { mat: operand }
            | Operation::MatCholesky { mat: operand }
            | Operation::MatLU { mat: operand }
            | Operation::MatQR { mat: operand }
            | Operation::MatEigen { mat: operand }
            | Operation::MatSVD { mat: operand }
            | Operation::Assume { cond: operand } => vec![*operand],

            // Binary
            Operation::IAdd { lhs, rhs }
            | Operation::IAddWrap { lhs, rhs }
            | Operation::IAddSat { lhs, rhs }
            | Operation::ISub { lhs, rhs }
            | Operation::ISubWrap { lhs, rhs }
            | Operation::IMul { lhs, rhs }
            | Operation::IMulWrap { lhs, rhs }
            | Operation::IDiv { lhs, rhs }
            | Operation::UDiv { lhs, rhs }
            | Operation::IRem { lhs, rhs }
            | Operation::URem { lhs, rhs }
            | Operation::FAdd { lhs, rhs }
            | Operation::FSub { lhs, rhs }
            | Operation::FMul { lhs, rhs }
            | Operation::FDiv { lhs, rhs }
            | Operation::FRem { lhs, rhs }
            | Operation::Pow {
                base: lhs,
                exp: rhs,
            }
            | Operation::Atan2 { y: lhs, x: rhs }
            | Operation::FMin { lhs, rhs }
            | Operation::FMax { lhs, rhs }
            | Operation::CopySign {
                magnitude: lhs,
                sign: rhs,
            }
            | Operation::Beta { a: lhs, b: rhs }
            | Operation::GammaInc { a: lhs, x: rhs }
            | Operation::GammaIncReg { a: lhs, x: rhs }
            | Operation::And { lhs, rhs }
            | Operation::Or { lhs, rhs }
            | Operation::Xor { lhs, rhs }
            | Operation::Shl { lhs, rhs }
            | Operation::LShr { lhs, rhs }
            | Operation::AShr { lhs, rhs }
            | Operation::ICmp { lhs, rhs, .. }
            | Operation::FCmp { lhs, rhs, .. }
            | Operation::MakeDual {
                value: lhs,
                derivative: rhs,
            }
            | Operation::DualAdd { lhs, rhs }
            | Operation::DualSub { lhs, rhs }
            | Operation::DualMul { lhs, rhs }
            | Operation::DualDiv { lhs, rhs }
            | Operation::DualPow {
                base: lhs,
                exp: rhs,
            }
            | Operation::VecDot { lhs, rhs }
            | Operation::MatMul { lhs, rhs }
            | Operation::MatSolve { a: lhs, b: rhs }
            | Operation::VecBinOp { lhs, rhs, .. } => vec![*lhs, *rhs],

            // Ternary
            Operation::FMA { a, b, c }
            | Operation::BetaInc { a, b, x: c }
            | Operation::BetaIncReg { a, b, x: c }
            | Operation::Select {
                cond: a,
                then_val: b,
                else_val: c,
            }
            | Operation::InsertElement {
                aggregate: a,
                index: b,
                value: c,
            } => vec![*a, *b, *c],

            // InsertField has only two operands
            Operation::InsertField {
                aggregate, value, ..
            } => vec![*aggregate, *value],

            // Variable operands
            Operation::Alloca { count, .. } => count.iter().copied().collect(),
            Operation::Load { ptr, .. } => vec![*ptr],
            Operation::Store { ptr, value, .. } => vec![*ptr, *value],
            Operation::GetElementPtr { base, indices, .. } => {
                let mut ops = vec![*base];
                ops.extend(indices);
                ops
            }
            Operation::Memcpy { dst, src, len, .. } | Operation::Memmove { dst, src, len, .. } => {
                vec![*dst, *src, *len]
            }
            Operation::Memset { dst, val, len, .. } => vec![*dst, *val, *len],
            Operation::ExtractField { aggregate, .. } => vec![*aggregate],
            Operation::ExtractElement { aggregate, index } => vec![*aggregate, *index],
            Operation::Aggregate { values, .. } => values.clone(),
            Operation::Call { args, .. } => args.clone(),
            Operation::MatVecMul { mat, vec } => vec![*mat, *vec],
            Operation::VecMap { vec, .. } => vec![*vec],
            Operation::VecReduce { vec, .. } => vec![*vec],
            Operation::VecShuffle { vec1, vec2, .. } => vec![*vec1, *vec2],
            Operation::VecBroadcast { scalar, .. } => vec![*scalar],
            Operation::LogPDF { value, params, .. }
            | Operation::CDF { value, params, .. }
            | Operation::Quantile {
                prob: value,
                params,
                ..
            } => {
                let mut ops = vec![*value];
                ops.extend(params);
                ops
            }
            Operation::Sample { params, rng, .. } => {
                let mut ops = params.clone();
                ops.push(*rng);
                ops
            }
            Operation::Assert { cond, .. } => vec![*cond],
            Operation::Fence { .. } => vec![],
            Operation::AtomicLoad { ptr, .. } => vec![*ptr],
            Operation::AtomicStore { ptr, value, .. } => vec![*ptr, *value],
            Operation::AtomicCAS {
                ptr,
                expected,
                desired,
                ..
            } => vec![*ptr, *expected, *desired],
            Operation::AtomicRMW { ptr, value, .. } => vec![*ptr, *value],
        }
    }
}

/// Call target
#[derive(Clone, Debug)]
pub enum Callee {
    /// Direct function call
    Direct(String),
    /// Indirect call through function pointer
    Indirect(ValueId),
    /// Intrinsic
    Intrinsic(String),
    /// External (FFI)
    External { name: String, lib: String },
}

/// Integer comparison predicate
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntPredicate {
    Eq,
    Ne,
    Slt, // Signed less than
    Sle, // Signed less than or equal
    Sgt, // Signed greater than
    Sge, // Signed greater than or equal
    Ult, // Unsigned less than
    Ule, // Unsigned less than or equal
    Ugt, // Unsigned greater than
    Uge, // Unsigned greater than or equal
}

/// Floating-point comparison predicate
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatPredicate {
    // Ordered (false if NaN)
    OEq,
    ONe,
    OLt,
    OLe,
    OGt,
    OGe,
    // Unordered (true if NaN)
    UEq,
    UNe,
    ULt,
    ULe,
    UGt,
    UGe,
}

/// Unary operation for vector map
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
}

/// Binary operation for vector operations
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Pow,
}

/// Reduce operation
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Add,
    Mul,
    Min,
    Max,
    And,
    Or,
    Xor,
}

/// Distribution kinds for probability operations
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DistributionKind {
    Normal,
    LogNormal,
    Exponential,
    Gamma,
    Beta,
    Uniform,
    Cauchy,
    StudentT,
    Binomial,
    Poisson,
    NegBinomial,
    Bernoulli,
    Categorical,
    Dirichlet,
    MultivariateNormal,
    Wishart,
    InverseWishart,
}

/// Atomic memory ordering
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomicOrdering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

/// Atomic read-modify-write operation
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomicRMWOp {
    Xchg,
    Add,
    Sub,
    And,
    Or,
    Xor,
    Max,
    Min,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_creation() {
        let op = Operation::FAdd {
            lhs: ValueId(0),
            rhs: ValueId(1),
        };
        let inst = Instruction::new(op, MirType::F64).with_result(ValueId(2));

        assert_eq!(inst.result, Some(ValueId(2)));
        assert!(inst.is_pure());
        assert!(!inst.has_side_effects());
    }

    #[test]
    fn test_operation_operands() {
        let add = Operation::FAdd {
            lhs: ValueId(0),
            rhs: ValueId(1),
        };
        assert_eq!(add.operands(), vec![ValueId(0), ValueId(1)]);

        let sqrt = Operation::Sqrt {
            operand: ValueId(5),
        };
        assert_eq!(sqrt.operands(), vec![ValueId(5)]);

        let const_int = Operation::ConstInt {
            value: 42,
            ty: MirType::I32,
        };
        assert!(const_int.operands().is_empty());
    }

    #[test]
    fn test_side_effects() {
        let pure_op = Operation::FMul {
            lhs: ValueId(0),
            rhs: ValueId(1),
        };
        assert!(!pure_op.has_side_effects());
        assert!(pure_op.is_pure());

        let store_op = Operation::Store {
            ptr: ValueId(0),
            value: ValueId(1),
            volatile: false,
            align: 8,
        };
        assert!(store_op.has_side_effects());
        assert!(!store_op.is_pure());

        let call_op = Operation::Call {
            callee: Callee::Direct("foo".to_string()),
            args: vec![],
            ret_ty: MirType::Void,
        };
        assert!(call_op.has_side_effects());
    }

    #[test]
    fn test_dual_operations() {
        let make_dual = Operation::MakeDual {
            value: ValueId(0),
            derivative: ValueId(1),
        };
        assert_eq!(make_dual.operands(), vec![ValueId(0), ValueId(1)]);

        let dual_mul = Operation::DualMul {
            lhs: ValueId(2),
            rhs: ValueId(3),
        };
        assert!(dual_mul.is_pure());
    }

    #[test]
    fn test_vector_operations() {
        let dot = Operation::VecDot {
            lhs: ValueId(0),
            rhs: ValueId(1),
        };
        assert_eq!(dot.operands(), vec![ValueId(0), ValueId(1)]);

        let reduce = Operation::VecReduce {
            vec: ValueId(5),
            op: ReduceOp::Add,
        };
        assert_eq!(reduce.operands(), vec![ValueId(5)]);
    }

    #[test]
    fn test_distribution_operations() {
        let log_pdf = Operation::LogPDF {
            distribution: DistributionKind::Normal,
            value: ValueId(0),
            params: vec![ValueId(1), ValueId(2)],
        };
        assert_eq!(log_pdf.operands(), vec![ValueId(0), ValueId(1), ValueId(2)]);
        assert!(log_pdf.is_pure());

        let sample = Operation::Sample {
            distribution: DistributionKind::Normal,
            params: vec![ValueId(0), ValueId(1)],
            rng: ValueId(2),
        };
        assert!(sample.has_side_effects()); // RNG is stateful
    }
}
