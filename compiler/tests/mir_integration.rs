//! MIR Integration Tests
//!
//! Tests for the MedLang Intermediate Representation (MIR) system.
//! These tests verify the complete MIR pipeline including type system,
//! SSA instructions, control flow, memory model, and ABI.

use medlangc::mir::*;

// =============================================================================
// Type System Tests
// =============================================================================

#[test]
fn test_mir_primitive_types() {
    // Test all primitive types
    assert_eq!(MirType::Void.size(), 0);
    assert_eq!(MirType::Bool.size(), 1);
    assert_eq!(MirType::I8.size(), 1);
    assert_eq!(MirType::I16.size(), 2);
    assert_eq!(MirType::I32.size(), 4);
    assert_eq!(MirType::I64.size(), 8);
    assert_eq!(MirType::I128.size(), 16);
    assert_eq!(MirType::U8.size(), 1);
    assert_eq!(MirType::U16.size(), 2);
    assert_eq!(MirType::U32.size(), 4);
    assert_eq!(MirType::U64.size(), 8);
    assert_eq!(MirType::U128.size(), 16);
    assert_eq!(MirType::F32.size(), 4);
    assert_eq!(MirType::F64.size(), 8);
}

#[test]
fn test_mir_dual_number_types() {
    // Dual numbers for automatic differentiation
    assert_eq!(MirType::DualF32.size(), 8); // value + derivative
    assert_eq!(MirType::DualF64.size(), 16);
    assert!(MirType::DualF64.is_dual());
    assert!(MirType::DualF32.is_copy());

    let dual_vec = MirType::DualVecF64 { size: 10 };
    assert_eq!(dual_vec.size(), 160); // 10 * 16
}

#[test]
fn test_mir_composite_types() {
    // Array type
    let arr = MirType::array(MirType::F64, 100);
    assert_eq!(arr.size(), 800);
    assert_eq!(arr.align(), 8);

    // Matrix type
    let mat = MirType::matrix(MirType::F64, 4, 4);
    assert_eq!(mat.size(), 128); // 16 * 8
    assert!(mat.is_simd());

    // Vector type
    let vec = MirType::vector(MirType::F32, 8);
    assert_eq!(vec.size(), 32);

    // Complex type
    let complex = MirType::complex(MirType::F64);
    assert_eq!(complex.size(), 16);
}

#[test]
fn test_mir_struct_type() {
    // PK parameter struct
    let pk_params = MirType::structure(
        "PKParams",
        vec![
            ("ka".to_string(), MirType::F64),   // Absorption rate
            ("cl".to_string(), MirType::F64),   // Clearance
            ("v".to_string(), MirType::F64),    // Volume
            ("dose".to_string(), MirType::F64), // Dose
        ],
    );

    assert_eq!(pk_params.size(), 32); // 4 * 8
    assert_eq!(pk_params.align(), 8);
}

#[test]
fn test_mir_enum_type() {
    // Metabolizer phenotype enum
    let phenotype = MirType::enumeration(
        "MetabolizerPhenotype",
        vec![
            EnumVariant::new("PoorMetabolizer", 0),
            EnumVariant::new("IntermediateMetabolizer", 1),
            EnumVariant::new("NormalMetabolizer", 2),
            EnumVariant::new("RapidMetabolizer", 3),
            EnumVariant::new("UltrarapidMetabolizer", 4),
        ],
    );

    // 5 variants fit in 1 byte discriminant
    match &phenotype {
        MirType::Enum { layout, .. } => {
            assert_eq!(layout.discriminant_size, 1);
        }
        _ => panic!("Expected enum type"),
    }
}

#[test]
fn test_mir_pointer_types() {
    let ptr = MirType::ptr(MirType::F64, false);
    assert_eq!(ptr.size(), 8);
    assert!(ptr.is_pointer());
    assert!(ptr.is_copy());

    let boxed = MirType::boxed(MirType::F64);
    assert!(boxed.needs_drop());
    assert!(!boxed.is_copy());
}

// =============================================================================
// SSA Instruction Tests
// =============================================================================

#[test]
fn test_mir_arithmetic_operations() {
    // Create a simple arithmetic function
    let sig = FunctionSignature::new(vec![MirType::F64, MirType::F64], MirType::F64);
    let mut builder = FunctionBuilder::new("compute", sig);

    let x = builder.param(0).unwrap();
    let y = builder.param(1).unwrap();

    // z = x + y
    let z = builder.push_op(Operation::FAdd { lhs: x, rhs: y }, MirType::F64);

    // w = z * x
    let w = builder.push_op(Operation::FMul { lhs: z, rhs: x }, MirType::F64);

    builder.terminate(Terminator::Return { value: Some(w) });

    let func = builder.build_validated().unwrap();
    assert_eq!(func.entry_block().unwrap().instructions.len(), 2);
}

#[test]
fn test_mir_math_intrinsics() {
    let sig = FunctionSignature::new(vec![MirType::F64], MirType::F64);
    let mut builder = FunctionBuilder::new("math_ops", sig);

    let x = builder.param(0).unwrap();

    // Test various math operations
    let sqrt_x = builder.push_op(Operation::Sqrt { operand: x }, MirType::F64);
    let exp_x = builder.push_op(Operation::Exp { operand: x }, MirType::F64);
    let log_x = builder.push_op(Operation::Log { operand: x }, MirType::F64);
    let sin_x = builder.push_op(Operation::Sin { operand: x }, MirType::F64);
    let tanh_x = builder.push_op(Operation::Tanh { operand: x }, MirType::F64);

    // Combine: result = sqrt(x) + exp(x) + log(x) + sin(x) + tanh(x)
    let t1 = builder.push_op(
        Operation::FAdd {
            lhs: sqrt_x,
            rhs: exp_x,
        },
        MirType::F64,
    );
    let t2 = builder.push_op(
        Operation::FAdd {
            lhs: log_x,
            rhs: sin_x,
        },
        MirType::F64,
    );
    let t3 = builder.push_op(Operation::FAdd { lhs: t1, rhs: t2 }, MirType::F64);
    let result = builder.push_op(
        Operation::FAdd {
            lhs: t3,
            rhs: tanh_x,
        },
        MirType::F64,
    );

    builder.terminate(Terminator::Return {
        value: Some(result),
    });

    let func = builder.build_validated().unwrap();
    assert!(func.entry_block().unwrap().instructions.len() >= 9);
}

#[test]
fn test_mir_special_math_functions() {
    // Test special functions for statistical computing
    let sig = FunctionSignature::new(vec![MirType::F64], MirType::F64);
    let mut builder = FunctionBuilder::new("special_math", sig);

    let x = builder.param(0).unwrap();

    // Gamma function
    let gamma = builder.push_op(Operation::Gamma { operand: x }, MirType::F64);
    // Log-gamma
    let lgamma = builder.push_op(Operation::LogGamma { operand: x }, MirType::F64);
    // Error function
    let erf = builder.push_op(Operation::Erf { operand: x }, MirType::F64);
    // Bessel J0
    let j0 = builder.push_op(Operation::BesselJ0 { operand: x }, MirType::F64);

    builder.terminate(Terminator::Return { value: Some(j0) });

    let func = builder.build_validated().unwrap();
    assert_eq!(func.entry_block().unwrap().instructions.len(), 4);
}

// =============================================================================
// Automatic Differentiation Tests
// =============================================================================

#[test]
fn test_mir_dual_number_operations() {
    // Test forward-mode AD operations
    let sig = FunctionSignature::new(vec![MirType::F64, MirType::F64], MirType::DualF64);
    let mut builder = FunctionBuilder::new("ad_forward", sig);

    let x = builder.param(0).unwrap();
    let dx = builder.param(1).unwrap();

    // Create dual number
    let x_dual = builder.push_op(
        Operation::MakeDual {
            value: x,
            derivative: dx,
        },
        MirType::DualF64,
    );

    // Apply operations on dual numbers
    let sin_dual = builder.push_op(Operation::DualSin { operand: x_dual }, MirType::DualF64);
    let exp_dual = builder.push_op(Operation::DualExp { operand: sin_dual }, MirType::DualF64);

    builder.terminate(Terminator::Return {
        value: Some(exp_dual),
    });

    let func = builder.build_validated().unwrap();
    assert_eq!(func.num_blocks(), 1);
}

#[test]
fn test_mir_extract_dual_components() {
    let sig = FunctionSignature::new(vec![MirType::DualF64], MirType::F64);
    let mut builder = FunctionBuilder::new("extract_dual", sig);

    let dual = builder.param(0).unwrap();

    // Extract primal and tangent
    let primal = builder.push_op(Operation::DualPrimal { dual }, MirType::F64);
    let tangent = builder.push_op(Operation::DualTangent { dual }, MirType::F64);

    // Return primal + tangent
    let sum = builder.push_op(
        Operation::FAdd {
            lhs: primal,
            rhs: tangent,
        },
        MirType::F64,
    );

    builder.terminate(Terminator::Return { value: Some(sum) });

    let func = builder.build_validated().unwrap();
    assert_eq!(func.entry_block().unwrap().instructions.len(), 3);
}

// =============================================================================
// Control Flow Tests
// =============================================================================

#[test]
fn test_mir_conditional_branch() {
    let sig = FunctionSignature::new(vec![MirType::F64, MirType::F64], MirType::F64);
    let mut builder = FunctionBuilder::new("max", sig);

    let x = builder.param(0).unwrap();
    let y = builder.param(1).unwrap();

    // Compare x > y
    let cond = builder.push_op(
        Operation::FCmp {
            pred: FloatPredicate::OGt,
            lhs: x,
            rhs: y,
        },
        MirType::Bool,
    );

    let then_block = builder.create_named_block("then");
    let else_block = builder.create_named_block("else");
    let merge_block = builder.create_named_block("merge");

    builder.terminate(Terminator::Branch {
        cond,
        then_block,
        then_args: vec![],
        else_block,
        else_args: vec![],
    });

    // Then: return x
    builder.switch_to(then_block);
    builder.terminate(Terminator::Goto {
        target: merge_block,
        args: vec![x],
    });

    // Else: return y
    builder.switch_to(else_block);
    builder.terminate(Terminator::Goto {
        target: merge_block,
        args: vec![y],
    });

    // Merge with phi via block param
    builder.switch_to(merge_block);
    let result = builder.block_param(MirType::F64);
    builder.terminate(Terminator::Return {
        value: Some(result),
    });

    let func = builder.build_validated().unwrap();
    assert_eq!(func.num_blocks(), 4);
}

#[test]
fn test_mir_loop_structure() {
    // While loop: sum = 0; while (i < n) { sum += i; i++; }
    let sig = FunctionSignature::new(vec![MirType::I64], MirType::I64);
    let mut builder = FunctionBuilder::new("sum_to_n", sig);

    let n = builder.param(0).unwrap();

    // Initialize
    let zero = builder.push_op(
        Operation::ConstInt {
            value: 0,
            ty: MirType::I64,
        },
        MirType::I64,
    );
    let one = builder.push_op(
        Operation::ConstInt {
            value: 1,
            ty: MirType::I64,
        },
        MirType::I64,
    );

    let loop_header = builder.create_named_block("loop_header");
    let loop_body = builder.create_named_block("loop_body");
    let exit = builder.create_named_block("exit");

    builder.terminate(Terminator::Goto {
        target: loop_header,
        args: vec![zero, zero], // i=0, sum=0
    });

    // Loop header: check i < n
    builder.switch_to(loop_header);
    let i = builder.block_param(MirType::I64);
    let sum = builder.block_param(MirType::I64);

    let cond = builder.push_op(
        Operation::ICmp {
            pred: IntPredicate::Slt,
            lhs: i,
            rhs: n,
        },
        MirType::Bool,
    );

    builder.terminate(Terminator::Branch {
        cond,
        then_block: loop_body,
        then_args: vec![],
        else_block: exit,
        else_args: vec![],
    });

    // Loop body: sum += i; i++
    builder.switch_to(loop_body);
    let new_sum = builder.push_op(Operation::IAdd { lhs: sum, rhs: i }, MirType::I64);
    let new_i = builder.push_op(Operation::IAdd { lhs: i, rhs: one }, MirType::I64);

    builder.terminate(Terminator::Goto {
        target: loop_header,
        args: vec![new_i, new_sum],
    });

    // Exit
    builder.switch_to(exit);
    builder.terminate(Terminator::Return { value: Some(sum) });

    let func = builder.build_validated().unwrap();
    assert_eq!(func.num_blocks(), 4);

    // Verify predecessors
    let preds = func.predecessors();
    assert_eq!(preds[&loop_header].len(), 2); // Entry and loop_body
}

// =============================================================================
// Probability Distribution Tests
// =============================================================================

#[test]
fn test_mir_distribution_operations() {
    // Log-probability computation for Normal distribution
    let sig = FunctionSignature::new(vec![MirType::F64, MirType::F64, MirType::F64], MirType::F64);
    let mut builder = FunctionBuilder::new("normal_lpdf", sig);

    let y = builder.param(0).unwrap(); // observed value
    let mu = builder.param(1).unwrap(); // mean
    let sigma = builder.param(2).unwrap(); // std dev

    let log_pdf = builder.push_op(
        Operation::LogPDF {
            distribution: DistributionKind::Normal,
            value: y,
            params: vec![mu, sigma],
        },
        MirType::F64,
    );

    builder.terminate(Terminator::Return {
        value: Some(log_pdf),
    });

    let func = builder.build_validated().unwrap();
    assert_eq!(func.name, "normal_lpdf");
}

#[test]
fn test_mir_multiple_distributions() {
    let sig = FunctionSignature::new(vec![MirType::F64], MirType::F64);
    let mut builder = FunctionBuilder::new("mixed_lpdf", sig);

    let x = builder.param(0).unwrap();

    // Normal
    let mu = builder.push_op(
        Operation::ConstFloat {
            value: 0.0,
            ty: MirType::F64,
        },
        MirType::F64,
    );
    let sigma = builder.push_op(
        Operation::ConstFloat {
            value: 1.0,
            ty: MirType::F64,
        },
        MirType::F64,
    );
    let normal_lp = builder.push_op(
        Operation::LogPDF {
            distribution: DistributionKind::Normal,
            value: x,
            params: vec![mu, sigma],
        },
        MirType::F64,
    );

    // Gamma
    let alpha = builder.push_op(
        Operation::ConstFloat {
            value: 2.0,
            ty: MirType::F64,
        },
        MirType::F64,
    );
    let beta = builder.push_op(
        Operation::ConstFloat {
            value: 1.0,
            ty: MirType::F64,
        },
        MirType::F64,
    );
    let gamma_lp = builder.push_op(
        Operation::LogPDF {
            distribution: DistributionKind::Gamma,
            value: x,
            params: vec![alpha, beta],
        },
        MirType::F64,
    );

    let total = builder.push_op(
        Operation::FAdd {
            lhs: normal_lp,
            rhs: gamma_lp,
        },
        MirType::F64,
    );

    builder.terminate(Terminator::Return { value: Some(total) });

    let func = builder.build_validated().unwrap();
    assert!(func.entry_block().unwrap().instructions.len() >= 7);
}

// =============================================================================
// Matrix/Vector Operations Tests
// =============================================================================

#[test]
fn test_mir_matrix_operations() {
    // Matrix-vector multiplication
    let mat_ty = MirType::matrix(MirType::F64, 3, 3);
    let vec_ty = MirType::vector(MirType::F64, 3);

    let sig = FunctionSignature::new(vec![mat_ty.clone(), vec_ty.clone()], vec_ty.clone());
    let mut builder = FunctionBuilder::new("matvec", sig);

    let mat = builder.param(0).unwrap();
    let vec = builder.param(1).unwrap();

    let result = builder.push_op(Operation::MatVecMul { mat, vec }, vec_ty);

    builder.terminate(Terminator::Return {
        value: Some(result),
    });

    let func = builder.build_validated().unwrap();
    assert_eq!(func.name, "matvec");
}

#[test]
fn test_mir_linear_algebra() {
    let mat_ty = MirType::matrix(MirType::F64, 4, 4);

    let sig = FunctionSignature::new(vec![mat_ty.clone()], MirType::F64);
    let mut builder = FunctionBuilder::new("linalg_ops", sig);

    let mat = builder.param(0).unwrap();

    // Determinant
    let det = builder.push_op(Operation::MatDet { mat }, MirType::F64);

    // Trace
    let trace = builder.push_op(Operation::MatTrace { mat }, MirType::F64);

    let sum = builder.push_op(
        Operation::FAdd {
            lhs: det,
            rhs: trace,
        },
        MirType::F64,
    );

    builder.terminate(Terminator::Return { value: Some(sum) });

    let func = builder.build_validated().unwrap();
    assert_eq!(func.entry_block().unwrap().instructions.len(), 3);
}

// =============================================================================
// Module Structure Tests
// =============================================================================

#[test]
fn test_mir_module_construction() {
    let mut builder = ModuleBuilder::new("pk_model");

    builder
        .target("x86_64-unknown-linux-gnu")
        .source_file("pk_model.med");

    // Add global constants
    builder.constant(
        "LOG_2PI",
        MirType::F64,
        ConstValue::Float(1.8378770664093453),
    );

    // Add type definition
    builder.type_def(
        "PKParams",
        MirType::structure(
            "PKParams",
            vec![
                ("ka".to_string(), MirType::F64),
                ("cl".to_string(), MirType::F64),
                ("v".to_string(), MirType::F64),
            ],
        ),
    );

    // Add external function
    builder.external(
        "printf",
        FunctionSignature::new(vec![MirType::ptr(MirType::I8, false)], MirType::I32).variadic(),
    );

    // Add function
    builder.function(
        "compute_concentration",
        FunctionSignature::new(
            vec![MirType::F64, MirType::F64, MirType::F64, MirType::F64],
            MirType::F64,
        ),
        |f| {
            let dose = f.param(0).unwrap();
            let ka = f.param(1).unwrap();
            let ke = f.param(2).unwrap();
            let t = f.param(3).unwrap();

            // C(t) = dose * ka / (ka - ke) * (exp(-ke*t) - exp(-ka*t))
            let ka_minus_ke = f.push_op(Operation::FSub { lhs: ka, rhs: ke }, MirType::F64);
            let dose_ka = f.push_op(Operation::FMul { lhs: dose, rhs: ka }, MirType::F64);
            let coeff = f.push_op(
                Operation::FDiv {
                    lhs: dose_ka,
                    rhs: ka_minus_ke,
                },
                MirType::F64,
            );

            let neg_ke = f.push_op(Operation::FNeg { operand: ke }, MirType::F64);
            let neg_ka = f.push_op(Operation::FNeg { operand: ka }, MirType::F64);
            let ke_t = f.push_op(
                Operation::FMul {
                    lhs: neg_ke,
                    rhs: t,
                },
                MirType::F64,
            );
            let ka_t = f.push_op(
                Operation::FMul {
                    lhs: neg_ka,
                    rhs: t,
                },
                MirType::F64,
            );

            let exp_ke_t = f.push_op(Operation::Exp { operand: ke_t }, MirType::F64);
            let exp_ka_t = f.push_op(Operation::Exp { operand: ka_t }, MirType::F64);

            let diff = f.push_op(
                Operation::FSub {
                    lhs: exp_ke_t,
                    rhs: exp_ka_t,
                },
                MirType::F64,
            );

            let result = f.push_op(
                Operation::FMul {
                    lhs: coeff,
                    rhs: diff,
                },
                MirType::F64,
            );

            f.terminate(Terminator::Return {
                value: Some(result),
            });
        },
    );

    let module = builder.build_validated().unwrap();

    assert_eq!(module.name, "pk_model");
    assert_eq!(module.num_functions(), 1);
    assert!(module.get_function("compute_concentration").is_some());
    assert!(module.has_external("printf"));
}

// =============================================================================
// Memory Model Tests
// =============================================================================

#[test]
fn test_mir_memory_regions() {
    assert!(MemoryRegion::HEAP.is_heap());
    assert!(MemoryRegion::ARENA.is_heap());
    assert!(MemoryRegion::GPU.is_device());
    assert!(!MemoryRegion::STACK.is_heap());
    assert!(!MemoryRegion::STATIC.is_device());
}

#[test]
fn test_mir_ownership_tracking() {
    let mut state = MemoryState::new();

    // Allocate on heap
    let alloc = state.allocate(MirType::F64, MemoryRegion::HEAP);
    assert!(state.is_live(alloc));

    // Set ownership
    state.set_ownership(ValueId(0), Ownership::Owned);
    assert!(state.can_read(ValueId(0)));
    assert!(state.can_write(ValueId(0)));

    // Borrow
    state.set_ownership(ValueId(1), Ownership::Borrowed);
    assert!(state.can_read(ValueId(1)));
    assert!(!state.can_write(ValueId(1)));

    // Deallocate
    state.deallocate(alloc).unwrap();
    assert!(!state.is_live(alloc));
}

#[test]
fn test_mir_alias_analysis() {
    let mut state = MemoryState::new();

    let alloc1 = state.allocate(MirType::F64, MemoryRegion::HEAP);
    let alloc2 = state.allocate(MirType::F64, MemoryRegion::HEAP);

    // Different allocations don't alias
    state.set_provenance(ValueId(0), Provenance::new(alloc1));
    state.set_provenance(ValueId(1), Provenance::new(alloc2));

    let analyzer = AliasAnalyzer::new(state);
    assert_eq!(analyzer.alias(ValueId(0), ValueId(1)), AliasResult::NoAlias);
    assert_eq!(
        analyzer.alias(ValueId(0), ValueId(0)),
        AliasResult::MustAlias
    );
}

// =============================================================================
// ABI and Layout Tests
// =============================================================================

#[test]
fn test_mir_x86_64_abi() {
    let abi = SystemVABI::new();

    // Integer in register
    let int_info = abi.classify(&MirType::I64);
    assert_eq!(int_info.class_lo, ParamClass::Integer);
    assert!(int_info.direct);

    // Float in SSE register
    let float_info = abi.classify(&MirType::F64);
    assert_eq!(float_info.class_lo, ParamClass::SSE);

    // Large struct in memory
    let large_struct = MirType::structure(
        "Large",
        vec![
            ("a".to_string(), MirType::I64),
            ("b".to_string(), MirType::I64),
            ("c".to_string(), MirType::I64),
        ],
    );
    let large_info = abi.classify(&large_struct);
    assert!(large_info.indirect);
}

#[test]
fn test_mir_data_layout() {
    let layout = DataLayout::x86_64_linux();

    // Pointer size
    assert_eq!(layout.pointer_size, 64);

    // Type layout
    let f64_layout = layout.type_layout(&MirType::F64);
    assert_eq!(f64_layout.size, 8);
    assert_eq!(f64_layout.align, 8);

    // Struct layout with padding
    let fields = vec![
        ("a".to_string(), MirType::I8),
        ("b".to_string(), MirType::I64),
    ];
    let struct_layout = layout.struct_layout(&fields);
    assert_eq!(struct_layout.field_offsets[0], 0);
    assert_eq!(struct_layout.field_offsets[1], 8); // Padded for alignment
}

// =============================================================================
// Pretty Printing Tests
// =============================================================================

#[test]
fn test_mir_pretty_print() {
    let mut builder = ModuleBuilder::new("test");

    builder.function(
        "add",
        FunctionSignature::new(vec![MirType::F64, MirType::F64], MirType::F64),
        |f| {
            let x = f.param(0).unwrap();
            let y = f.param(1).unwrap();
            let sum = f.push_op(Operation::FAdd { lhs: x, rhs: y }, MirType::F64);
            f.terminate(Terminator::Return { value: Some(sum) });
        },
    );

    let module = builder.build();
    let printed = print_module(&module);

    assert!(printed.contains("MIR Module: test"));
    assert!(printed.contains("fn @add"));
    assert!(printed.contains("fadd"));
    assert!(printed.contains("ret"));
}

// =============================================================================
// Integration: Complete PK Model in MIR
// =============================================================================

#[test]
fn test_mir_complete_pk_model() {
    let mut builder = ModuleBuilder::new("one_compartment_pk");

    // Global: standard normal log-pdf constant term
    builder.constant(
        "LOG_2PI",
        MirType::F64,
        ConstValue::Float(1.8378770664093453),
    );

    // Type for individual parameters
    builder.type_def(
        "IndividualParams",
        MirType::structure(
            "IndividualParams",
            vec![
                ("log_ka".to_string(), MirType::F64),
                ("log_cl".to_string(), MirType::F64),
                ("log_v".to_string(), MirType::F64),
            ],
        ),
    );

    // ODE solution function
    builder.function(
        "pk_solution",
        FunctionSignature::new(
            vec![
                MirType::F64, // t
                MirType::F64, // dose
                MirType::F64, // ka
                MirType::F64, // ke
            ],
            MirType::F64,
        ),
        |f| {
            let t = f.param(0).unwrap();
            let dose = f.param(1).unwrap();
            let ka = f.param(2).unwrap();
            let ke = f.param(3).unwrap();

            // C(t) = dose * ka / (ka - ke) * (exp(-ke*t) - exp(-ka*t))
            let ka_minus_ke = f.push_op(Operation::FSub { lhs: ka, rhs: ke }, MirType::F64);
            let dose_ka = f.push_op(Operation::FMul { lhs: dose, rhs: ka }, MirType::F64);
            let coeff = f.push_op(
                Operation::FDiv {
                    lhs: dose_ka,
                    rhs: ka_minus_ke,
                },
                MirType::F64,
            );

            let neg_ke = f.push_op(Operation::FNeg { operand: ke }, MirType::F64);
            let neg_ka = f.push_op(Operation::FNeg { operand: ka }, MirType::F64);
            let ke_t = f.push_op(
                Operation::FMul {
                    lhs: neg_ke,
                    rhs: t,
                },
                MirType::F64,
            );
            let ka_t = f.push_op(
                Operation::FMul {
                    lhs: neg_ka,
                    rhs: t,
                },
                MirType::F64,
            );

            let exp_ke_t = f.push_op(Operation::Exp { operand: ke_t }, MirType::F64);
            let exp_ka_t = f.push_op(Operation::Exp { operand: ka_t }, MirType::F64);

            let diff = f.push_op(
                Operation::FSub {
                    lhs: exp_ke_t,
                    rhs: exp_ka_t,
                },
                MirType::F64,
            );

            let result = f.push_op(
                Operation::FMul {
                    lhs: coeff,
                    rhs: diff,
                },
                MirType::F64,
            );

            f.terminate(Terminator::Return {
                value: Some(result),
            });
        },
    );

    // Log-likelihood function
    builder.function(
        "log_likelihood",
        FunctionSignature::new(
            vec![
                MirType::F64, // y_obs
                MirType::F64, // y_pred
                MirType::F64, // sigma
            ],
            MirType::F64,
        ),
        |f| {
            let y_obs = f.param(0).unwrap();
            let y_pred = f.param(1).unwrap();
            let sigma = f.param(2).unwrap();

            let ll = f.push_op(
                Operation::LogPDF {
                    distribution: DistributionKind::Normal,
                    value: y_obs,
                    params: vec![y_pred, sigma],
                },
                MirType::F64,
            );

            f.terminate(Terminator::Return { value: Some(ll) });
        },
    );

    let module = builder.build_validated().unwrap();

    assert_eq!(module.name, "one_compartment_pk");
    assert_eq!(module.num_functions(), 2);
    assert!(module.get_function("pk_solution").is_some());
    assert!(module.get_function("log_likelihood").is_some());
    assert_eq!(module.constants.len(), 1);
    assert_eq!(module.types.len(), 1);

    // Verify the functions are well-formed
    let pk_fn = module.get_function("pk_solution").unwrap();
    assert_eq!(pk_fn.signature.params.len(), 4);
    assert_eq!(pk_fn.num_blocks(), 1);

    let ll_fn = module.get_function("log_likelihood").unwrap();
    assert_eq!(ll_fn.signature.params.len(), 3);
}
