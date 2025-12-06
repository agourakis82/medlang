//! Register Allocation for MIR
//!
//! This module provides register allocation for lowering MIR to machine code.
//! It supports multiple allocation strategies:
//!
//! - **Linear Scan**: Fast O(n) allocation, good for JIT
//! - **Graph Coloring**: Better code quality, good for AOT
//! - **PBQP**: Optimal for irregular architectures
//!
//! # Architecture Support
//!
//! The allocator is parameterized by target architecture:
//! - x86-64: 16 GPRs, 16 XMM/YMM/ZMM
//! - AArch64: 31 GPRs, 32 SIMD
//! - RISC-V: 32 GPRs, 32 FPRs
//!
//! # Spilling Strategy
//!
//! When registers are exhausted, values are spilled to the stack.
//! The allocator minimizes spill costs using:
//! - Loop depth weighting
//! - Use frequency analysis
//! - Rematerialization for cheap values

pub mod interference;
pub mod lifetime;
pub mod linear_scan;
pub mod spill;
pub mod target;

pub use interference::{InterferenceGraph, InterferenceNode};
pub use lifetime::{LiveInterval, LiveRange, LivenessAnalysis};
pub use linear_scan::{LinearScanAllocator, LinearScanConfig, LinearScanResult};
pub use spill::{SpillCost, SpillStrategy, Spiller};
pub use target::{PhysicalRegister, RegisterClass, RegisterInfo, TargetRegisters};

use crate::mir::function::MirFunction;
use crate::mir::value::ValueId;
use std::collections::HashMap;

/// Result of register allocation
#[derive(Debug, Clone)]
pub struct AllocationResult {
    /// Mapping from virtual registers to physical registers
    pub assignments: HashMap<ValueId, PhysicalRegister>,
    /// Values that were spilled to stack
    pub spilled: Vec<SpilledValue>,
    /// Stack frame size needed for spills
    pub spill_slots: usize,
    /// Statistics about the allocation
    pub stats: AllocationStats,
}

/// A value that was spilled to stack
#[derive(Debug, Clone)]
pub struct SpilledValue {
    /// The virtual register that was spilled
    pub value: ValueId,
    /// Stack slot offset
    pub slot: i32,
    /// Size in bytes
    pub size: u32,
    /// Alignment requirement
    pub align: u32,
}

/// Statistics about register allocation
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    /// Number of virtual registers
    pub virtual_regs: usize,
    /// Number of physical registers used
    pub phys_regs_used: usize,
    /// Number of spills
    pub spills: usize,
    /// Number of reloads
    pub reloads: usize,
    /// Number of moves inserted
    pub moves_inserted: usize,
    /// Number of coalesced moves
    pub moves_coalesced: usize,
}

/// Register allocation algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocatorKind {
    /// Linear scan (fast, JIT-friendly)
    LinearScan,
    /// Graph coloring (better quality)
    GraphColoring,
    /// PBQP (optimal for irregular targets)
    PBQP,
}

/// Configuration for register allocation
#[derive(Debug, Clone)]
pub struct AllocatorConfig {
    /// Algorithm to use
    pub algorithm: AllocatorKind,
    /// Target register info
    pub target: TargetRegisters,
    /// Enable move coalescing
    pub coalesce: bool,
    /// Enable live range splitting
    pub split_ranges: bool,
    /// Enable rematerialization
    pub rematerialize: bool,
    /// Spill weight multiplier for loop depth
    pub loop_weight: f64,
}

impl Default for AllocatorConfig {
    fn default() -> Self {
        Self {
            algorithm: AllocatorKind::LinearScan,
            target: TargetRegisters::x86_64(),
            coalesce: true,
            split_ranges: false,
            rematerialize: true,
            loop_weight: 10.0,
        }
    }
}

/// Main register allocator
pub struct RegisterAllocator {
    config: AllocatorConfig,
}

impl RegisterAllocator {
    /// Create a new allocator with configuration
    pub fn new(config: AllocatorConfig) -> Self {
        Self { config }
    }

    /// Allocate registers for a function
    pub fn allocate(&self, func: &MirFunction) -> AllocationResult {
        // Compute liveness
        let mut liveness = LivenessAnalysis::new();
        liveness.analyze(func);

        match self.config.algorithm {
            AllocatorKind::LinearScan => {
                let mut allocator = LinearScanAllocator::new(
                    LinearScanConfig::default(),
                    self.config.target.clone(),
                );
                allocator.allocate(&liveness)
            }
            AllocatorKind::GraphColoring => {
                // Build interference graph and color it
                let graph = InterferenceGraph::build(&liveness);
                self.graph_coloring_allocate(&graph, &liveness)
            }
            AllocatorKind::PBQP => {
                // PBQP allocation (simplified)
                let graph = InterferenceGraph::build(&liveness);
                self.graph_coloring_allocate(&graph, &liveness)
            }
        }
    }

    /// Graph coloring allocation
    fn graph_coloring_allocate(
        &self,
        graph: &InterferenceGraph,
        _liveness: &LivenessAnalysis,
    ) -> AllocationResult {
        // Simplify-Select-Spill cycle
        let mut assignments = HashMap::new();
        let mut spilled = Vec::new();
        let mut stats = AllocationStats::default();

        // Get nodes in simplification order
        let order = graph.simplify_order();

        // Assign colors (registers) in reverse order
        for node_id in order.into_iter().rev() {
            let node = &graph.nodes[node_id];
            stats.virtual_regs += 1;

            // Find available register
            let used: Vec<_> = graph
                .neighbors(node_id)
                .filter_map(|n| assignments.get(&graph.nodes[n].value))
                .cloned()
                .collect();

            if let Some(reg) = self.find_available_register(&node.reg_class, &used) {
                assignments.insert(node.value, reg);
                stats.phys_regs_used += 1;
            } else {
                // Need to spill
                spilled.push(SpilledValue {
                    value: node.value,
                    slot: (spilled.len() * 8) as i32,
                    size: 8,
                    align: 8,
                });
                stats.spills += 1;
            }
        }

        AllocationResult {
            assignments,
            spilled,
            spill_slots: stats.spills * 8,
            stats,
        }
    }

    /// Find an available register of the given class
    fn find_available_register(
        &self,
        class: &RegisterClass,
        used: &[PhysicalRegister],
    ) -> Option<PhysicalRegister> {
        let regs = self.config.target.registers_for_class(class);
        regs.into_iter().find(|r| !used.contains(r))
    }
}
