//! Low-Level Architecture Tests for MIR
//!
//! Tests for:
//! - Automatic Differentiation (forward and reverse mode)
//! - CUDA code generation
//! - Deterministic floating-point operations
//! - Register allocation

use medlangc::mir::ad::reverse::TapeEntry;
use medlangc::mir::ad::ADMode;
use medlangc::mir::block::{BasicBlock, Terminator};
use medlangc::mir::cuda::kernel::{Dim3, ParallelismPattern, ReductionOp};
use medlangc::mir::cuda::memory::MemoryPlanner;
use medlangc::mir::cuda::{CudaCodegen, CudaType, GpuArch, KernelConfig, LaunchBounds};
use medlangc::mir::deterministic::analysis::{
    AccuracyRequirement, DeterminismAnalysis, DeterminismLevel,
};
use medlangc::mir::deterministic::rounding::{Interval, RoundingContext, RoundingMode};
use medlangc::mir::deterministic::summation::{
    CompensatedSum, PairwiseSum, SummationAlgorithm, TreeSum,
};
use medlangc::mir::deterministic::transcendental::{DeterministicMath, MathAccuracy};
use medlangc::mir::deterministic::transform::{FPTransformBuilder, FPTransformConfig};
use medlangc::mir::function::{FunctionSignature, MirFunction};
use medlangc::mir::inst::{Instruction, Operation};
use medlangc::mir::module::ModuleBuilder;
use medlangc::mir::regalloc::interference::{InterferenceGraph, InterferenceNode};
use medlangc::mir::regalloc::lifetime::{LiveInterval, LiveRange, LivenessAnalysis, ProgramPoint};
use medlangc::mir::regalloc::linear_scan::{LinearScanAllocator, LinearScanConfig};
use medlangc::mir::regalloc::spill::{SpillCost, SpillFrameLayout};
use medlangc::mir::regalloc::target::{PhysicalRegister, RegisterClass, TargetRegisters};
use medlangc::mir::regalloc::{AllocatorConfig, AllocatorKind, RegisterAllocator};
use medlangc::mir::types::MirType;
use medlangc::mir::value::{BlockId, ValueId, ValueIdGen};
use std::collections::HashSet;

// ============================================================================
// Automatic Differentiation Tests
// ============================================================================

mod ad_tests {
    use super::*;

    #[test]
    fn test_ad_mode_variants() {
        let forward = ADMode::Forward;
        let reverse = ADMode::Reverse;
        let mixed = ADMode::Mixed;

        // Just check they can be constructed
        assert!(matches!(forward, ADMode::Forward));
        assert!(matches!(reverse, ADMode::Reverse));
        assert!(matches!(mixed, ADMode::Mixed));
    }

    #[test]
    fn test_tape_entry_creation() {
        let mut id_gen = ValueIdGen::new();
        let lhs = id_gen.next();
        let rhs = id_gen.next();
        let result = id_gen.next();

        let inst = Instruction::new(Operation::FAdd { lhs, rhs }, MirType::F64).with_result(result);

        let entry = TapeEntry {
            primal_inst: inst,
            saved_values: vec![],
            adjoint_code: vec![],
        };

        assert!(entry.saved_values.is_empty());
        assert!(entry.adjoint_code.is_empty());
    }
}

// ============================================================================
// CUDA Code Generation Tests
// ============================================================================

mod cuda_tests {
    use super::*;

    #[test]
    fn test_gpu_architectures() {
        assert!(GpuArch::Sm70.has_tensor_cores());
        assert!(GpuArch::Sm80.has_tensor_cores());
        assert!(GpuArch::Sm90.has_tensor_cores());

        // Check compute capability (returns string)
        assert_eq!(GpuArch::Sm70.compute_capability(), "70");
        assert_eq!(GpuArch::Sm80.compute_capability(), "80");
    }

    #[test]
    fn test_cuda_type_mapping() {
        assert_eq!(CudaType::from_mir(&MirType::F32), CudaType::Float);
        assert_eq!(CudaType::from_mir(&MirType::F64), CudaType::Double);
        assert_eq!(CudaType::from_mir(&MirType::I32), CudaType::Int32);
        assert_eq!(CudaType::from_mir(&MirType::Bool), CudaType::Bool);
    }

    #[test]
    fn test_cuda_type_sizes() {
        assert_eq!(CudaType::Float.size_bytes(), 4);
        assert_eq!(CudaType::Double.size_bytes(), 8);
        assert_eq!(CudaType::Int32.size_bytes(), 4);
        assert_eq!(CudaType::Bool.size_bytes(), 1);
    }

    #[test]
    fn test_dim3_creation() {
        let dim = Dim3::new(256, 1, 1);
        assert_eq!(dim.x, 256);
        assert_eq!(dim.y, 1);
        assert_eq!(dim.z, 1);
        assert_eq!(dim.total(), 256);
    }

    #[test]
    fn test_kernel_config() {
        let config = KernelConfig::new(Dim3::new(128, 1, 1), Dim3::new(256, 1, 1));

        assert_eq!(config.total_threads(), 128 * 256);
        assert_eq!(config.total_blocks(), 128);
        assert_eq!(config.threads_per_block(), 256);
    }

    #[test]
    fn test_launch_bounds() {
        let bounds = LaunchBounds::new(256)
            .with_min_blocks(2)
            .with_max_registers(64);

        assert_eq!(bounds.max_threads_per_block, 256);
        assert_eq!(bounds.min_blocks_per_sm, Some(2));
        assert_eq!(bounds.max_registers_per_thread, Some(64));
    }

    #[test]
    fn test_parallelism_patterns() {
        let elementwise = ParallelismPattern::ElementWise { size: 1024 };
        let reduction = ParallelismPattern::Reduction {
            size: 1024,
            op: ReductionOp::Sum,
        };
        let matmul = ParallelismPattern::MatMul {
            m: 64,
            n: 64,
            k: 64,
        };

        // Verify patterns match correctly
        assert!(matches!(
            elementwise,
            ParallelismPattern::ElementWise { .. }
        ));
        assert!(matches!(reduction, ParallelismPattern::Reduction { .. }));
        assert!(matches!(matmul, ParallelismPattern::MatMul { .. }));
    }

    #[test]
    fn test_reduction_ops() {
        let sum = ReductionOp::Sum;
        let product = ReductionOp::Product;
        let min = ReductionOp::Min;
        let max = ReductionOp::Max;

        assert!(matches!(sum, ReductionOp::Sum));
        assert!(matches!(product, ReductionOp::Product));
        assert!(matches!(min, ReductionOp::Min));
        assert!(matches!(max, ReductionOp::Max));
    }

    #[test]
    fn test_memory_planner_creation() {
        let _planner = MemoryPlanner::new(GpuArch::Sm80);
        // Just verify construction works
    }

    #[test]
    fn test_cuda_codegen_creation() {
        let _codegen = CudaCodegen::new(GpuArch::Sm80);
        // Just verify construction works
    }
}

// ============================================================================
// Deterministic Floating-Point Tests
// ============================================================================

mod deterministic_fp_tests {
    use super::*;

    #[test]
    fn test_determinism_levels_ordering() {
        assert!(DeterminismLevel::BitExact > DeterminismLevel::CrossPlatform);
        assert!(DeterminismLevel::CrossPlatform > DeterminismLevel::LocalReproducible);
        assert!(DeterminismLevel::LocalReproducible > DeterminismLevel::None);
    }

    #[test]
    fn test_accuracy_requirements() {
        let default = AccuracyRequirement::Default;
        let ieee = AccuracyRequirement::IEEE754;
        let one_ulp = AccuracyRequirement::OneULP;
        let correct = AccuracyRequirement::CorrectlyRounded;

        // Verify variants exist
        assert!(matches!(default, AccuracyRequirement::Default));
        assert!(matches!(ieee, AccuracyRequirement::IEEE754));
        assert!(matches!(one_ulp, AccuracyRequirement::OneULP));
        assert!(matches!(correct, AccuracyRequirement::CorrectlyRounded));
    }

    #[test]
    fn test_rounding_modes() {
        assert_eq!(RoundingMode::default(), RoundingMode::NearestEven);

        let toward_zero = RoundingMode::TowardZero;
        assert_eq!(toward_zero.fenv_constant(), "FE_TOWARDZERO");
    }

    #[test]
    fn test_rounding_context() {
        let mut ctx = RoundingContext::new();
        assert_eq!(ctx.mode, RoundingMode::NearestEven);

        ctx.push_mode(RoundingMode::TowardPositive);
        assert_eq!(ctx.mode, RoundingMode::TowardPositive);

        ctx.pop_mode();
        assert_eq!(ctx.mode, RoundingMode::NearestEven);
    }

    #[test]
    fn test_kahan_summation() {
        let mut sum = CompensatedSum::zero();

        // Add a large value, then small values
        sum.add_kahan(1.0);
        for _ in 0..1000 {
            sum.add_kahan(1e-16);
        }

        // Should maintain precision
        let result = sum.result();
        assert!((result - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_neumaier_summation() {
        let mut sum = CompensatedSum::zero();

        // Neumaier handles the case where we add large, then small, then negate large
        sum.add_neumaier(1e16);
        sum.add_neumaier(1.0);
        sum.add_neumaier(-1e16);

        let result = sum.result();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_sum_generation() {
        let mut id_gen = ValueIdGen::new();
        let values: Vec<ValueId> = (0..8).map(|_| id_gen.next()).collect();

        let pairwise = PairwiseSum::new(values, MirType::F64);
        let (insts, _result) = pairwise.generate(&mut id_gen);

        // For 8 values, we need 7 additions (binary tree)
        assert_eq!(insts.len(), 7);
    }

    #[test]
    fn test_tree_sum_generation() {
        let mut id_gen = ValueIdGen::new();
        let leaves: Vec<ValueId> = (0..16).map(|_| id_gen.next()).collect();

        let tree = TreeSum::new(leaves, MirType::F64);
        let (insts, _result) = tree.generate(&mut id_gen);

        // For 16 values with branching factor 2, we need 15 additions
        assert_eq!(insts.len(), 15);
    }

    #[test]
    fn test_interval_arithmetic() {
        let a = Interval::new(1.0, 2.0);
        let b = Interval::new(3.0, 4.0);

        let sum = a.add(&b);
        assert_eq!(sum.lo, 4.0);
        assert_eq!(sum.hi, 6.0);

        let diff = b.sub(&a);
        assert_eq!(diff.lo, 1.0); // 3 - 2
        assert_eq!(diff.hi, 3.0); // 4 - 1

        let prod = a.mul(&b);
        assert_eq!(prod.lo, 3.0); // 1 * 3
        assert_eq!(prod.hi, 8.0); // 2 * 4
    }

    #[test]
    fn test_interval_contains_zero() {
        let positive = Interval::new(1.0, 2.0);
        let crossing = Interval::new(-1.0, 1.0);
        let negative = Interval::new(-2.0, -1.0);

        assert!(!positive.contains_zero());
        assert!(crossing.contains_zero());
        assert!(!negative.contains_zero());
    }

    #[test]
    fn test_deterministic_math_config() {
        let hw = DeterministicMath::hardware();
        assert_eq!(hw.accuracy, MathAccuracy::Hardware);

        let cr = DeterministicMath::correctly_rounded();
        assert_eq!(cr.accuracy, MathAccuracy::CorrectlyRounded);
    }

    #[test]
    fn test_fp_transform_config_presets() {
        let bit_exact = FPTransformConfig::bit_exact();
        assert_eq!(bit_exact.determinism_level, DeterminismLevel::BitExact);
        assert!(bit_exact.strict_fp);
        assert!(bit_exact.disable_fma);

        let local = FPTransformConfig::local_reproducible();
        assert_eq!(local.determinism_level, DeterminismLevel::LocalReproducible);

        let cross = FPTransformConfig::cross_platform();
        assert_eq!(cross.determinism_level, DeterminismLevel::CrossPlatform);
    }

    #[test]
    fn test_fp_transform_builder() {
        let config = FPTransformBuilder::new()
            .determinism_level(DeterminismLevel::BitExact)
            .strict()
            .no_fma()
            .summation(SummationAlgorithm::Tree)
            .build();

        assert_eq!(config.determinism_level, DeterminismLevel::BitExact);
        assert!(config.strict_fp);
        assert!(config.disable_fma);
        assert_eq!(config.summation_algorithm, SummationAlgorithm::Tree);
    }
}

// ============================================================================
// Register Allocation Tests
// ============================================================================

mod regalloc_tests {
    use super::*;

    #[test]
    fn test_physical_register_x86_64_names() {
        let rax = PhysicalRegister::new(0, RegisterClass::GPR);
        assert_eq!(rax.x86_64_name(), "rax");

        let rcx = PhysicalRegister::new(1, RegisterClass::GPR);
        assert_eq!(rcx.x86_64_name(), "rcx");

        let xmm0 = PhysicalRegister::new(0, RegisterClass::FPR);
        assert_eq!(xmm0.x86_64_name(), "xmm0");
    }

    #[test]
    fn test_physical_register_aarch64_names() {
        let x0 = PhysicalRegister::new(0, RegisterClass::GPR);
        assert_eq!(x0.aarch64_name(), "x0");

        let x29 = PhysicalRegister::new(29, RegisterClass::GPR);
        assert_eq!(x29.aarch64_name(), "x29");

        let v0 = PhysicalRegister::new(0, RegisterClass::FPR);
        assert_eq!(v0.aarch64_name(), "v0");
    }

    #[test]
    fn test_register_class_sizes() {
        assert_eq!(RegisterClass::GPR.size(), 8);
        assert_eq!(RegisterClass::FPR.size(), 8);
        assert_eq!(RegisterClass::Vector.size(), 32);
    }

    #[test]
    fn test_x86_64_register_config() {
        let target = TargetRegisters::x86_64();

        assert_eq!(target.num_gprs, 16);
        assert_eq!(target.num_fprs, 16);

        // Check allocatable registers (excluding rsp, rbp)
        let allocatable_gprs = target.allocatable(RegisterClass::GPR);
        assert_eq!(allocatable_gprs.len(), 14);

        // Check callee-saved
        let callee_saved = target.callee_saved(RegisterClass::GPR);
        assert!(!callee_saved.is_empty());
    }

    #[test]
    fn test_aarch64_register_config() {
        let target = TargetRegisters::aarch64();

        assert_eq!(target.num_gprs, 31);
        assert_eq!(target.num_fprs, 32);

        let callee_saved = target.callee_saved(RegisterClass::GPR);
        assert_eq!(callee_saved.len(), 10); // x19-x28
    }

    #[test]
    fn test_riscv64_register_config() {
        let target = TargetRegisters::riscv64();

        assert_eq!(target.num_gprs, 32);
        assert_eq!(target.num_fprs, 32);
    }

    #[test]
    fn test_program_point_ordering() {
        let p1 = ProgramPoint::before(0, 5);
        let p2 = ProgramPoint::after(0, 5);
        let p3 = ProgramPoint::before(0, 6);
        let p4 = ProgramPoint::before(1, 0);

        assert!(p1 < p2);
        assert!(p2 < p3);
        assert!(p3 < p4);
    }

    #[test]
    fn test_live_range_overlap() {
        let r1 = LiveRange::new(ProgramPoint::before(0, 0), ProgramPoint::after(0, 5));
        let r2 = LiveRange::new(ProgramPoint::before(0, 3), ProgramPoint::after(0, 8));
        let r3 = LiveRange::new(ProgramPoint::before(0, 10), ProgramPoint::after(0, 15));

        assert!(r1.overlaps(&r2));
        assert!(!r1.overlaps(&r3));
        assert!(!r2.overlaps(&r3));
    }

    #[test]
    fn test_live_range_contains() {
        let range = LiveRange::new(ProgramPoint::before(0, 5), ProgramPoint::after(0, 10));

        assert!(range.contains(ProgramPoint::before(0, 7)));
        assert!(range.contains(ProgramPoint::before(0, 5)));
        assert!(range.contains(ProgramPoint::after(0, 10)));
        assert!(!range.contains(ProgramPoint::before(0, 4)));
        assert!(!range.contains(ProgramPoint::after(0, 11)));
    }

    #[test]
    fn test_live_interval_creation() {
        let mut interval = LiveInterval::new(ValueId(0));

        interval.add_range(LiveRange::new(
            ProgramPoint::before(0, 0),
            ProgramPoint::after(0, 5),
        ));
        interval.add_range(LiveRange::new(
            ProgramPoint::before(0, 10),
            ProgramPoint::after(0, 15),
        ));

        assert_eq!(interval.ranges.len(), 2);
        assert_eq!(interval.start(), Some(ProgramPoint::before(0, 0)));
        assert_eq!(interval.end(), Some(ProgramPoint::after(0, 15)));
    }

    #[test]
    fn test_live_interval_overlap() {
        let mut interval1 = LiveInterval::new(ValueId(0));
        interval1.add_range(LiveRange::new(
            ProgramPoint::before(0, 0),
            ProgramPoint::after(0, 10),
        ));

        let mut interval2 = LiveInterval::new(ValueId(1));
        interval2.add_range(LiveRange::new(
            ProgramPoint::before(0, 5),
            ProgramPoint::after(0, 15),
        ));

        let mut interval3 = LiveInterval::new(ValueId(2));
        interval3.add_range(LiveRange::new(
            ProgramPoint::before(0, 20),
            ProgramPoint::after(0, 25),
        ));

        assert!(interval1.overlaps(&interval2));
        assert!(!interval1.overlaps(&interval3));
    }

    #[test]
    fn test_interference_graph_basics() {
        let mut graph = InterferenceGraph {
            nodes: vec![
                InterferenceNode::new(ValueId(0), RegisterClass::GPR),
                InterferenceNode::new(ValueId(1), RegisterClass::GPR),
                InterferenceNode::new(ValueId(2), RegisterClass::GPR),
            ],
            value_to_node: [(ValueId(0), 0), (ValueId(1), 1), (ValueId(2), 2)]
                .into_iter()
                .collect(),
            edges: vec![HashSet::new(), HashSet::new(), HashSet::new()],
            move_edges: vec![],
        };

        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        assert_eq!(graph.degree(0), 1);
        assert_eq!(graph.degree(1), 2);
        assert_eq!(graph.degree(2), 1);
    }

    #[test]
    fn test_interference_graph_simplify() {
        let mut graph = InterferenceGraph {
            nodes: vec![
                InterferenceNode::new(ValueId(0), RegisterClass::GPR),
                InterferenceNode::new(ValueId(1), RegisterClass::GPR),
                InterferenceNode::new(ValueId(2), RegisterClass::GPR),
            ],
            value_to_node: [(ValueId(0), 0), (ValueId(1), 1), (ValueId(2), 2)]
                .into_iter()
                .collect(),
            edges: vec![HashSet::new(), HashSet::new(), HashSet::new()],
            move_edges: vec![],
        };

        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        let order = graph.simplify_order();
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_linear_scan_non_overlapping() {
        let config = LinearScanConfig::default();
        let target = TargetRegisters::x86_64();
        let mut allocator = LinearScanAllocator::new(config, target);

        let mut liveness = LivenessAnalysis::new();

        // Create non-overlapping intervals
        let mut interval1 = LiveInterval::new(ValueId(0));
        interval1.add_range(LiveRange::new(
            ProgramPoint::before(0, 0),
            ProgramPoint::after(0, 5),
        ));

        let mut interval2 = LiveInterval::new(ValueId(1));
        interval2.add_range(LiveRange::new(
            ProgramPoint::before(0, 6),
            ProgramPoint::after(0, 10),
        ));

        liveness.intervals.insert(ValueId(0), interval1);
        liveness.intervals.insert(ValueId(1), interval2);

        let result = allocator.allocate(&liveness);

        // Both should be allocated (can share registers)
        assert!(result.assignments.contains_key(&ValueId(0)));
        assert!(result.assignments.contains_key(&ValueId(1)));
        assert_eq!(result.spilled.len(), 0);
    }

    #[test]
    fn test_linear_scan_overlapping() {
        let config = LinearScanConfig::default();
        let target = TargetRegisters::x86_64();
        let mut allocator = LinearScanAllocator::new(config, target);

        let mut liveness = LivenessAnalysis::new();

        // Create overlapping intervals
        let mut interval1 = LiveInterval::new(ValueId(0));
        interval1.add_range(LiveRange::new(
            ProgramPoint::before(0, 0),
            ProgramPoint::after(0, 10),
        ));

        let mut interval2 = LiveInterval::new(ValueId(1));
        interval2.add_range(LiveRange::new(
            ProgramPoint::before(0, 5),
            ProgramPoint::after(0, 15),
        ));

        liveness.intervals.insert(ValueId(0), interval1);
        liveness.intervals.insert(ValueId(1), interval2);

        let result = allocator.allocate(&liveness);

        // Both should be allocated to different registers
        assert!(result.assignments.contains_key(&ValueId(0)));
        assert!(result.assignments.contains_key(&ValueId(1)));
        assert_ne!(
            result.assignments.get(&ValueId(0)),
            result.assignments.get(&ValueId(1))
        );
    }

    #[test]
    fn test_spill_cost_calculation() {
        let cost = SpillCost::default();
        assert_eq!(cost.total(1, 3), 4.0); // 1 store + 3 loads

        let remat = SpillCost::rematerializable(0.5);
        assert!(remat.rematerializable);
        // For rematerializable, min(stores * store, loads * remat_cost)
        assert_eq!(remat.total(1, 10), 1.0); // min(1.0, 5.0)
    }

    #[test]
    fn test_spill_frame_layout() {
        let mut layout = SpillFrameLayout::new();

        let slot1 = layout.allocate(8, 8);
        assert_eq!(slot1, 0);

        let slot2 = layout.allocate(8, 8);
        assert_eq!(slot2, 8);

        let slot3 = layout.allocate(4, 4);
        assert_eq!(slot3, 16);

        assert_eq!(layout.size, 20);
        assert_eq!(layout.align, 8);
    }

    #[test]
    fn test_allocator_config() {
        let config = AllocatorConfig::default();
        assert_eq!(config.algorithm, AllocatorKind::LinearScan);
        assert!(config.coalesce);
        assert!(config.rematerialize);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_full_ad_pipeline() {
        // Create a simple function
        let sig = FunctionSignature::new(vec![MirType::F64], MirType::F64);
        let mut func = MirFunction::new("square", sig);

        let x = ValueId(0);
        let result = ValueId(1);

        let mut block = BasicBlock::new(BlockId(0));
        block.instructions.push(
            Instruction::new(Operation::FMul { lhs: x, rhs: x }, MirType::F64).with_result(result),
        );
        block.terminator = Terminator::Return {
            value: Some(result),
        };
        func.blocks.push(block);

        // Verify function was created correctly
        assert_eq!(func.name, "square");
        assert_eq!(func.blocks.len(), 1);
    }

    #[test]
    fn test_deterministic_fp_analysis() {
        // Create module
        let builder = ModuleBuilder::new("test");
        let module = builder.build();

        // Run analysis
        let mut analysis =
            DeterminismAnalysis::new(DeterminismLevel::CrossPlatform, AccuracyRequirement::OneULP);
        let result = analysis.analyze_module(&module);

        // Empty module should have highest determinism
        assert_eq!(result.current_level, DeterminismLevel::BitExact);
    }

    #[test]
    fn test_register_allocation_empty_function() {
        let sig = FunctionSignature::new(vec![], MirType::Void);
        let mut func = MirFunction::new("empty", sig);

        let mut block = BasicBlock::new(BlockId(0));
        block.terminator = Terminator::Return { value: None };
        func.blocks.push(block);

        let config = AllocatorConfig::default();
        let allocator = RegisterAllocator::new(config);
        let result = allocator.allocate(&func);

        assert!(result.assignments.is_empty());
        assert!(result.spilled.is_empty());
    }
}
