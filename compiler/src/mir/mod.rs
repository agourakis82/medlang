//! MedLang Intermediate Representation (MIR)
//!
//! MIR is the low-level, post-erasure intermediate representation for MedLang.
//! It is designed for:
//!
//! - **Deterministic semantics**: No undefined behavior
//! - **Direct mapping to backends**: LLVM, Stan, CUDA, WebAssembly
//! - **Efficient AD transformation**: First-class support for automatic differentiation
//! - **Verifiable memory safety**: Region-based memory model with ownership tracking
//!
//! # Architecture
//!
//! ```text
//! HIR (High-level IR)
//!        │
//!        │ Type erasure, monomorphization
//!        ▼
//!       MIR (This module)
//!        │
//!        │ Optimizations, lowering
//!        ▼
//! Backend Code (LLVM IR, Stan, CUDA, WASM)
//! ```
//!
//! # Module Structure
//!
//! - [`types`]: Post-erasure type system (monomorphic, machine-representable)
//! - [`value`]: Value and block identifiers for SSA form
//! - [`inst`]: SSA-form instructions (150+ operations)
//! - [`block`]: Basic blocks and terminators
//! - [`function`]: Function definitions with CFG
//! - [`module`]: Module structure with globals and type definitions
//! - [`memory`]: Memory model with ownership tracking
//! - [`layout`]: Data layout and ABI specifications
//!
//! # Example
//!
//! ```rust
//! use medlang::mir::*;
//!
//! // Create a simple add function
//! let mut builder = ModuleBuilder::new("example");
//!
//! builder.function(
//!     "add",
//!     FunctionSignature::new(vec![MirType::F64, MirType::F64], MirType::F64),
//!     |f| {
//!         let x = f.param(0).unwrap();
//!         let y = f.param(1).unwrap();
//!         let sum = f.push_op(Operation::FAdd { lhs: x, rhs: y }, MirType::F64);
//!         f.terminate(Terminator::Return { value: Some(sum) });
//!     },
//! );
//!
//! let module = builder.build();
//! ```
//!
//! # SSA Form
//!
//! MIR uses SSA (Static Single Assignment) form where each value is defined
//! exactly once. Instead of phi nodes, MIR uses block parameters:
//!
//! ```text
//! bb0:
//!   %cond = icmp eq %x, 0
//!   br %cond, bb1(), bb2()
//!
//! bb1:
//!   goto bb3(%const_1)
//!
//! bb2:
//!   goto bb3(%const_0)
//!
//! bb3(%result: i32):  // Block parameter instead of phi
//!   ret %result
//! ```
//!
//! # Automatic Differentiation
//!
//! MIR has first-class support for AD through dual number types and operations:
//!
//! ```text
//! %x_dual = make_dual %x, %dx      // Create dual number
//! %y_dual = dual_sin %x_dual       // sin with derivative
//! %primal = dual_primal %y_dual    // Extract value
//! %tangent = dual_tangent %y_dual  // Extract derivative
//! ```

pub mod ad;
pub mod block;
pub mod cuda;
pub mod deterministic;
pub mod function;
pub mod inst;
pub mod layout;
pub mod memory;
pub mod module;
pub mod regalloc;
pub mod types;
pub mod value;

// Re-export commonly used types
pub use block::{BasicBlock, BlockBuilder, BlockParam, Terminator};
pub use function::{
    FunctionAttributes, FunctionBuilder, FunctionDebugInfo, FunctionSignature, InlineHint,
    LocalDecl, MirFunction, OptLevel, ScopeInfo, ValidationError,
};
pub use inst::{
    AtomicOrdering, AtomicRMWOp, BinaryOp, Callee, DistributionKind, FloatPredicate, Instruction,
    IntPredicate, Operation, ReduceOp, UnaryOp,
};
pub use layout::{
    DataLayout, Endianness, EnumFieldLayout, Mangling, ParamClass, ParamInfo, Register,
    StructFieldLayout, SystemVABI, TypeLayout,
};
pub use memory::{
    AccessPathElement, AliasAnalyzer, AliasResult, Allocation, AllocationId, MemoryError,
    MemoryLocation, MemoryRegion, MemoryState, Ownership, Provenance,
};
pub use module::{
    ConstValue, ConstantDef, ExternalFunction, GlobalDecl, Linkage, MirModule, ModuleAttributes,
    ModuleBuilder, ModuleFlag, SymbolTable, TypeDef,
};
pub use types::{
    CallingConvention, EnumLayout, EnumVariant, MatrixLayout, MirType, Mutability, RegionId,
    StructLayout,
};
pub use value::{BlockId, BlockIdGen, ScopeId, Span, ValueId, ValueIdGen};

/// MIR version
pub const MIR_VERSION: &str = "0.1.0";

/// Check if a MIR module is well-formed
pub fn validate_module(module: &MirModule) -> Result<(), Vec<ValidationError>> {
    module.validate()
}

/// Pretty-print a MIR module
pub fn print_module(module: &MirModule) -> String {
    let mut output = String::new();

    output.push_str(&format!("; MIR Module: {}\n", module.name));
    output.push_str(&format!("; Version: {}\n\n", MIR_VERSION));

    // Print type definitions
    if !module.types.is_empty() {
        output.push_str("; Type Definitions\n");
        for type_def in &module.types {
            output.push_str(&format!("type {} = {}\n", type_def.name, type_def.ty));
        }
        output.push('\n');
    }

    // Print globals
    if !module.globals.is_empty() {
        output.push_str("; Global Variables\n");
        for global in &module.globals {
            let mut_str = if global.mutable { "mut " } else { "" };
            output.push_str(&format!(
                "global @{}: {}{}\n",
                global.name, mut_str, global.ty
            ));
        }
        output.push('\n');
    }

    // Print external functions
    if !module.external_functions.is_empty() {
        output.push_str("; External Functions\n");
        for ext in &module.external_functions {
            output.push_str(&format!(
                "declare @{}({})",
                ext.name,
                ext.signature
                    .params
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            output.push_str(&format!(" -> {}\n", ext.signature.return_type));
        }
        output.push('\n');
    }

    // Print functions
    for func in &module.functions {
        output.push_str(&print_function(func));
        output.push('\n');
    }

    output
}

/// Pretty-print a MIR function
pub fn print_function(func: &MirFunction) -> String {
    let mut output = String::new();

    // Function signature
    let params: Vec<String> = func
        .signature
        .params
        .iter()
        .zip(func.signature.param_names.iter())
        .map(|(ty, name)| format!("{}: {}", name, ty))
        .collect();

    output.push_str(&format!(
        "fn @{}({}) -> {} {{\n",
        func.name,
        params.join(", "),
        func.signature.return_type
    ));

    // Basic blocks
    for block in &func.blocks {
        output.push_str(&print_block(block));
    }

    output.push_str("}\n");
    output
}

/// Pretty-print a basic block
pub fn print_block(block: &BasicBlock) -> String {
    let mut output = String::new();

    // Block header with parameters
    let default_name = format!("bb{}", block.id.0);
    let name = block.name.as_deref().unwrap_or(&default_name);

    if block.params.is_empty() {
        output.push_str(&format!("  {}:\n", name));
    } else {
        let params: Vec<String> = block
            .params
            .iter()
            .map(|p| format!("{}: {}", p.value, p.ty))
            .collect();
        output.push_str(&format!("  {}({}):\n", name, params.join(", ")));
    }

    // Instructions
    for inst in &block.instructions {
        output.push_str(&format!("    {}\n", print_instruction(inst)));
    }

    // Terminator
    output.push_str(&format!("    {}\n", print_terminator(&block.terminator)));

    output
}

/// Pretty-print an instruction
pub fn print_instruction(inst: &Instruction) -> String {
    let result_str = inst.result.map(|v| format!("{} = ", v)).unwrap_or_default();

    format!("{}{}", result_str, print_operation(&inst.op))
}

/// Pretty-print an operation
pub fn print_operation(op: &Operation) -> String {
    match op {
        Operation::ConstInt { value, ty } => format!("const {} : {}", value, ty),
        Operation::ConstFloat { value, ty } => format!("const {} : {}", value, ty),
        Operation::ConstBool { value } => format!("const {}", value),
        Operation::ZeroInit { ty } => format!("zeroinit {}", ty),
        Operation::Undef { ty } => format!("undef {}", ty),
        Operation::FAdd { lhs, rhs } => format!("fadd {}, {}", lhs, rhs),
        Operation::FSub { lhs, rhs } => format!("fsub {}, {}", lhs, rhs),
        Operation::FMul { lhs, rhs } => format!("fmul {}, {}", lhs, rhs),
        Operation::FDiv { lhs, rhs } => format!("fdiv {}, {}", lhs, rhs),
        Operation::IAdd { lhs, rhs } => format!("iadd {}, {}", lhs, rhs),
        Operation::ISub { lhs, rhs } => format!("isub {}, {}", lhs, rhs),
        Operation::IMul { lhs, rhs } => format!("imul {}, {}", lhs, rhs),
        Operation::IDiv { lhs, rhs } => format!("idiv {}, {}", lhs, rhs),
        Operation::Load { ptr, ty, .. } => format!("load {} : {}", ptr, ty),
        Operation::Store { ptr, value, .. } => format!("store {}, {}", value, ptr),
        Operation::Call {
            callee,
            args,
            ret_ty,
            ..
        } => {
            let args_str = args
                .iter()
                .map(|a| a.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let callee_str = match callee {
                Callee::Direct(name) => format!("@{}", name),
                Callee::Indirect(v) => v.to_string(),
                Callee::Intrinsic(name) => format!("intrinsic.{}", name),
                Callee::External { name, lib } => format!("extern.{}.{}", lib, name),
            };
            format!("call {}({}) : {}", callee_str, args_str, ret_ty)
        }
        _ => format!("{:?}", op), // Fallback for complex operations
    }
}

/// Pretty-print a terminator
pub fn print_terminator(term: &Terminator) -> String {
    match term {
        Terminator::Goto { target, args } => {
            if args.is_empty() {
                format!("goto {}", target)
            } else {
                let args_str = args
                    .iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("goto {}({})", target, args_str)
            }
        }
        Terminator::Branch {
            cond,
            then_block,
            else_block,
            ..
        } => format!("br {}, {}, {}", cond, then_block, else_block),
        Terminator::Return { value: Some(v) } => format!("ret {}", v),
        Terminator::Return { value: None } => "ret void".to_string(),
        Terminator::Unreachable => "unreachable".to_string(),
        Terminator::Abort { message } => format!("abort \"{}\"", message),
        _ => format!("{:?}", term),
    }
}
