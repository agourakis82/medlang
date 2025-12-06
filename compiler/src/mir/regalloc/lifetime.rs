//! Liveness Analysis and Live Intervals
//!
//! Computes the live range of each virtual register in a function.
//! This information is used by register allocators to determine
//! which registers can share physical storage.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::mir::block::BasicBlock;
use crate::mir::function::MirFunction;
use crate::mir::inst::Operation;
use crate::mir::value::{BlockId, ValueId};

/// A point in the program (for live ranges)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgramPoint {
    /// Block ID
    pub block: u32,
    /// Instruction index within block
    pub index: u32,
    /// Before (0) or after (1) the instruction
    pub position: u8,
}

impl ProgramPoint {
    /// Create a point before an instruction
    pub fn before(block: u32, index: u32) -> Self {
        Self {
            block,
            index,
            position: 0,
        }
    }

    /// Create a point after an instruction
    pub fn after(block: u32, index: u32) -> Self {
        Self {
            block,
            index,
            position: 1,
        }
    }

    /// Create a point at block entry
    pub fn block_entry(block: u32) -> Self {
        Self::before(block, 0)
    }

    /// Create a point at block exit
    pub fn block_exit(block: u32, num_insts: u32) -> Self {
        Self::after(block, num_insts.saturating_sub(1))
    }
}

/// A contiguous range where a value is live
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveRange {
    /// Start of the range
    pub start: ProgramPoint,
    /// End of the range (inclusive)
    pub end: ProgramPoint,
}

impl LiveRange {
    /// Create a new live range
    pub fn new(start: ProgramPoint, end: ProgramPoint) -> Self {
        Self { start, end }
    }

    /// Check if this range contains a point
    pub fn contains(&self, point: ProgramPoint) -> bool {
        self.start <= point && point <= self.end
    }

    /// Check if this range overlaps with another
    pub fn overlaps(&self, other: &LiveRange) -> bool {
        !(self.end < other.start || other.end < self.start)
    }

    /// Merge with another range (assumes they overlap or are adjacent)
    pub fn merge(&self, other: &LiveRange) -> LiveRange {
        LiveRange {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// Complete live interval for a virtual register
#[derive(Debug, Clone)]
pub struct LiveInterval {
    /// The virtual register
    pub value: ValueId,
    /// Live ranges (may be multiple due to holes)
    pub ranges: Vec<LiveRange>,
    /// Use positions
    pub uses: Vec<ProgramPoint>,
    /// Definition positions
    pub defs: Vec<ProgramPoint>,
    /// Spill weight (higher = more expensive to spill)
    pub spill_weight: f64,
}

impl LiveInterval {
    /// Create a new live interval
    pub fn new(value: ValueId) -> Self {
        Self {
            value,
            ranges: Vec::new(),
            uses: Vec::new(),
            defs: Vec::new(),
            spill_weight: 1.0,
        }
    }

    /// Add a range to this interval
    pub fn add_range(&mut self, range: LiveRange) {
        // Try to merge with existing ranges
        let mut merged = false;
        for existing in &mut self.ranges {
            if existing.overlaps(&range)
                || existing.end.block == range.start.block
                    && existing.end.index + 1 == range.start.index
            {
                *existing = existing.merge(&range);
                merged = true;
                break;
            }
        }

        if !merged {
            self.ranges.push(range);
        }

        // Sort ranges
        self.ranges.sort_by_key(|r| r.start);
    }

    /// Get the start point of the interval
    pub fn start(&self) -> Option<ProgramPoint> {
        self.ranges.first().map(|r| r.start)
    }

    /// Get the end point of the interval
    pub fn end(&self) -> Option<ProgramPoint> {
        self.ranges.last().map(|r| r.end)
    }

    /// Check if this interval is live at a point
    pub fn live_at(&self, point: ProgramPoint) -> bool {
        self.ranges.iter().any(|r| r.contains(point))
    }

    /// Check if this interval overlaps with another
    pub fn overlaps(&self, other: &LiveInterval) -> bool {
        for r1 in &self.ranges {
            for r2 in &other.ranges {
                if r1.overlaps(r2) {
                    return true;
                }
            }
        }
        false
    }

    /// Compute spill weight based on uses and loop depth
    pub fn compute_spill_weight(&mut self, loop_depths: &HashMap<BlockId, u32>) {
        let mut weight = 0.0;

        for use_point in &self.uses {
            let depth = loop_depths
                .get(&BlockId(use_point.block))
                .copied()
                .unwrap_or(0);
            weight += 10.0_f64.powi(depth as i32);
        }

        for def_point in &self.defs {
            let depth = loop_depths
                .get(&BlockId(def_point.block))
                .copied()
                .unwrap_or(0);
            weight += 10.0_f64.powi(depth as i32);
        }

        // Normalize by interval length
        if let (Some(start), Some(end)) = (self.start(), self.end()) {
            let length = (end.block - start.block + 1) as f64;
            weight /= length.max(1.0);
        }

        self.spill_weight = weight;
    }
}

/// Liveness analysis for a function
#[derive(Debug, Clone)]
pub struct LivenessAnalysis {
    /// Live intervals for each virtual register
    pub intervals: HashMap<ValueId, LiveInterval>,
    /// Values live at each block entry
    pub live_in: HashMap<BlockId, HashSet<ValueId>>,
    /// Values live at each block exit
    pub live_out: HashMap<BlockId, HashSet<ValueId>>,
    /// Block ordering (topological)
    pub block_order: Vec<BlockId>,
}

impl Default for LivenessAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl LivenessAnalysis {
    /// Create a new liveness analysis
    pub fn new() -> Self {
        Self {
            intervals: HashMap::new(),
            live_in: HashMap::new(),
            live_out: HashMap::new(),
            block_order: Vec::new(),
        }
    }

    /// Analyze a function
    pub fn analyze(&mut self, func: &MirFunction) {
        // Build block ordering (reverse postorder)
        self.block_order = self.compute_block_order(func);

        // Initialize live sets
        for block in &func.blocks {
            self.live_in.insert(block.id, HashSet::new());
            self.live_out.insert(block.id, HashSet::new());
        }

        // Iterate until fixed point
        let mut changed = true;
        while changed {
            changed = false;

            for &block_id in self.block_order.iter().rev() {
                let block = func.blocks.iter().find(|b| b.id == block_id).unwrap();

                // Compute live-out from successors' live-in
                let mut new_live_out = HashSet::new();
                for succ_id in self.get_successors(block) {
                    if let Some(live_in) = self.live_in.get(&succ_id) {
                        new_live_out.extend(live_in.iter().copied());
                    }
                }

                // Compute live-in: (live-out - defs) + uses
                let mut new_live_in = new_live_out.clone();

                // Process instructions in reverse
                for inst in block.instructions.iter().rev() {
                    // Remove definitions
                    if let Some(result) = inst.result {
                        new_live_in.remove(&result);
                    }

                    // Add uses
                    for used in self.get_uses(&inst.op) {
                        new_live_in.insert(used);
                    }
                }

                // Add block parameters as definitions
                for param in &block.params {
                    new_live_in.remove(&param.value);
                }

                // Check for changes
                if self.live_in.get(&block_id) != Some(&new_live_in) {
                    changed = true;
                    self.live_in.insert(block_id, new_live_in);
                }
                if self.live_out.get(&block_id) != Some(&new_live_out) {
                    changed = true;
                    self.live_out.insert(block_id, new_live_out);
                }
            }
        }

        // Build live intervals from live sets
        self.build_intervals(func);
    }

    /// Get successors of a block
    fn get_successors(&self, block: &BasicBlock) -> Vec<BlockId> {
        use crate::mir::block::Terminator;
        match &block.terminator {
            Terminator::Goto { target, .. } => vec![*target],
            Terminator::Branch {
                then_block,
                else_block,
                ..
            } => vec![*then_block, *else_block],
            Terminator::Switch { default, cases, .. } => {
                let mut succs = vec![*default];
                succs.extend(cases.iter().map(|(_, target, _)| *target));
                succs
            }
            Terminator::Call { next, .. } => vec![*next],
            Terminator::Invoke { normal, unwind, .. } => vec![*normal, *unwind],
            Terminator::Return { .. }
            | Terminator::Resume { .. }
            | Terminator::TailCall { .. }
            | Terminator::Unreachable
            | Terminator::Abort { .. } => {
                vec![]
            }
        }
    }

    /// Get values used by an operation
    fn get_uses(&self, op: &Operation) -> Vec<ValueId> {
        let mut uses = Vec::new();

        match op {
            Operation::ConstInt { .. }
            | Operation::ConstFloat { .. }
            | Operation::ConstBool { .. }
            | Operation::ZeroInit { .. }
            | Operation::Undef { .. } => {}

            Operation::FAdd { lhs, rhs }
            | Operation::FSub { lhs, rhs }
            | Operation::FMul { lhs, rhs }
            | Operation::FDiv { lhs, rhs }
            | Operation::IAdd { lhs, rhs }
            | Operation::ISub { lhs, rhs }
            | Operation::IMul { lhs, rhs }
            | Operation::IDiv { lhs, rhs }
            | Operation::UDiv { lhs, rhs }
            | Operation::IRem { lhs, rhs }
            | Operation::URem { lhs, rhs }
            | Operation::Shl { lhs, rhs }
            | Operation::LShr { lhs, rhs }
            | Operation::AShr { lhs, rhs }
            | Operation::And { lhs, rhs }
            | Operation::Or { lhs, rhs }
            | Operation::Xor { lhs, rhs }
            | Operation::FMin { lhs, rhs }
            | Operation::FMax { lhs, rhs }
            | Operation::Pow {
                base: lhs,
                exp: rhs,
            }
            | Operation::CopySign {
                magnitude: lhs,
                sign: rhs,
            } => {
                uses.push(*lhs);
                uses.push(*rhs);
            }

            Operation::FNeg { operand }
            | Operation::Not { operand }
            | Operation::Abs { operand }
            | Operation::Ceil { operand }
            | Operation::Floor { operand }
            | Operation::Round { operand }
            | Operation::Trunc { operand }
            | Operation::Sqrt { operand }
            | Operation::Sin { operand }
            | Operation::Cos { operand }
            | Operation::Tan { operand }
            | Operation::Asin { operand }
            | Operation::Acos { operand }
            | Operation::Atan { operand }
            | Operation::Sinh { operand }
            | Operation::Cosh { operand }
            | Operation::Tanh { operand }
            | Operation::Exp { operand }
            | Operation::Log { operand }
            | Operation::Log2 { operand }
            | Operation::Log10 { operand } => {
                uses.push(*operand);
            }

            Operation::Load { ptr, .. } => {
                uses.push(*ptr);
            }

            Operation::Store { ptr, value, .. } => {
                uses.push(*ptr);
                uses.push(*value);
            }

            Operation::Call { args, .. } => {
                uses.extend(args.iter().copied());
            }

            Operation::Select {
                cond,
                then_val,
                else_val,
            } => {
                uses.push(*cond);
                uses.push(*then_val);
                uses.push(*else_val);
            }

            Operation::ICmp { lhs, rhs, .. } | Operation::FCmp { lhs, rhs, .. } => {
                uses.push(*lhs);
                uses.push(*rhs);
            }

            Operation::FMA { a, b, c } => {
                uses.push(*a);
                uses.push(*b);
                uses.push(*c);
            }

            _ => {
                // Handle other operations conservatively
            }
        }

        uses
    }

    /// Compute reverse postorder of blocks
    fn compute_block_order(&self, func: &MirFunction) -> Vec<BlockId> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();

        if let Some(entry) = func.blocks.first() {
            self.dfs_postorder(func, entry.id, &mut visited, &mut order);
        }

        order.reverse();
        order
    }

    fn dfs_postorder(
        &self,
        func: &MirFunction,
        block_id: BlockId,
        visited: &mut HashSet<BlockId>,
        order: &mut Vec<BlockId>,
    ) {
        if visited.contains(&block_id) {
            return;
        }
        visited.insert(block_id);

        if let Some(block) = func.blocks.iter().find(|b| b.id == block_id) {
            for succ in self.get_successors(block) {
                self.dfs_postorder(func, succ, visited, order);
            }
        }

        order.push(block_id);
    }

    /// Build live intervals from live sets
    fn build_intervals(&mut self, func: &MirFunction) {
        for block in &func.blocks {
            let block_num = block.id.0 as u32;

            // Values live throughout the block
            if let Some(live_out) = self.live_out.get(&block.id) {
                for &value in live_out {
                    let interval = self
                        .intervals
                        .entry(value)
                        .or_insert_with(|| LiveInterval::new(value));
                    interval.add_range(LiveRange::new(
                        ProgramPoint::block_entry(block_num),
                        ProgramPoint::block_exit(block_num, block.instructions.len() as u32),
                    ));
                }
            }

            // Process instructions
            for (inst_idx, inst) in block.instructions.iter().enumerate() {
                let idx = inst_idx as u32;

                // Record definition
                if let Some(result) = inst.result {
                    let interval = self
                        .intervals
                        .entry(result)
                        .or_insert_with(|| LiveInterval::new(result));
                    interval.defs.push(ProgramPoint::after(block_num, idx));
                }

                // Record uses
                for used in self.get_uses(&inst.op) {
                    let interval = self
                        .intervals
                        .entry(used)
                        .or_insert_with(|| LiveInterval::new(used));
                    interval.uses.push(ProgramPoint::before(block_num, idx));
                }
            }
        }
    }

    /// Get all intervals sorted by start point
    pub fn sorted_intervals(&self) -> Vec<&LiveInterval> {
        let mut intervals: Vec<_> = self.intervals.values().collect();
        intervals.sort_by(|a, b| {
            let a_start = a.start().unwrap_or(ProgramPoint::before(0, 0));
            let b_start = b.start().unwrap_or(ProgramPoint::before(0, 0));
            a_start.cmp(&b_start)
        });
        intervals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_range_overlap() {
        let r1 = LiveRange::new(ProgramPoint::before(0, 0), ProgramPoint::after(0, 5));
        let r2 = LiveRange::new(ProgramPoint::before(0, 3), ProgramPoint::after(0, 8));
        let r3 = LiveRange::new(ProgramPoint::before(0, 10), ProgramPoint::after(0, 15));

        assert!(r1.overlaps(&r2));
        assert!(!r1.overlaps(&r3));
    }

    #[test]
    fn test_program_point_ordering() {
        let p1 = ProgramPoint::before(0, 5);
        let p2 = ProgramPoint::after(0, 5);
        let p3 = ProgramPoint::before(0, 6);

        assert!(p1 < p2);
        assert!(p2 < p3);
    }
}
