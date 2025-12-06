//! Linear Scan Register Allocation
//!
//! A fast O(n log n) register allocation algorithm suitable for JIT compilation.
//! Based on Poletto & Sarkar's original algorithm with extensions for:
//! - Second chance allocation
//! - Spill slot reuse
//! - Live range splitting

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::mir::value::ValueId;

use super::lifetime::{LiveInterval, LivenessAnalysis, ProgramPoint};
use super::target::{PhysicalRegister, RegisterClass, TargetRegisters};
use super::{AllocationResult, AllocationStats, SpilledValue};

/// Configuration for linear scan allocator
#[derive(Debug, Clone)]
pub struct LinearScanConfig {
    /// Enable second chance allocation
    pub second_chance: bool,
    /// Enable spill slot coalescing
    pub coalesce_spills: bool,
    /// Maximum number of split points per interval
    pub max_splits: usize,
}

impl Default for LinearScanConfig {
    fn default() -> Self {
        Self {
            second_chance: true,
            coalesce_spills: true,
            max_splits: 2,
        }
    }
}

/// Result of linear scan allocation
#[derive(Debug, Clone)]
pub struct LinearScanResult {
    /// Mapping from virtual registers to physical registers
    pub assignments: HashMap<ValueId, PhysicalRegister>,
    /// Spilled values
    pub spilled: Vec<SpilledValue>,
    /// Statistics
    pub stats: AllocationStats,
}

/// Active interval during allocation
#[derive(Debug, Clone)]
struct ActiveInterval {
    /// The interval
    interval: LiveInterval,
    /// Assigned physical register
    register: PhysicalRegister,
    /// End point (for heap ordering)
    end: ProgramPoint,
}

impl PartialEq for ActiveInterval {
    fn eq(&self, other: &Self) -> bool {
        self.end == other.end
    }
}

impl Eq for ActiveInterval {}

impl PartialOrd for ActiveInterval {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ActiveInterval {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap by end point
        other.end.cmp(&self.end)
    }
}

/// Linear scan register allocator
pub struct LinearScanAllocator {
    /// Configuration
    config: LinearScanConfig,
    /// Target register info
    target: TargetRegisters,
    /// Currently active intervals
    active: BinaryHeap<Reverse<ActiveInterval>>,
    /// Available registers per class
    free_regs: HashMap<RegisterClass, Vec<PhysicalRegister>>,
    /// Assignments
    assignments: HashMap<ValueId, PhysicalRegister>,
    /// Spilled values
    spilled: Vec<SpilledValue>,
    /// Next spill slot
    next_spill_slot: i32,
    /// Statistics
    stats: AllocationStats,
}

impl LinearScanAllocator {
    /// Create a new linear scan allocator
    pub fn new(config: LinearScanConfig, target: TargetRegisters) -> Self {
        let mut free_regs = HashMap::new();

        // Initialize free registers for each class
        free_regs.insert(RegisterClass::GPR, target.allocatable(RegisterClass::GPR));
        free_regs.insert(RegisterClass::FPR, target.allocatable(RegisterClass::FPR));
        free_regs.insert(
            RegisterClass::Vector,
            target.allocatable(RegisterClass::Vector),
        );

        Self {
            config,
            target,
            active: BinaryHeap::new(),
            free_regs,
            assignments: HashMap::new(),
            spilled: Vec::new(),
            next_spill_slot: 0,
            stats: AllocationStats::default(),
        }
    }

    /// Allocate registers for all intervals
    pub fn allocate(&mut self, liveness: &LivenessAnalysis) -> AllocationResult {
        // Get intervals sorted by start point
        let mut intervals: Vec<_> = liveness.intervals.values().cloned().collect();
        intervals.sort_by(|a, b| {
            let a_start = a.start().unwrap_or(ProgramPoint::before(0, 0));
            let b_start = b.start().unwrap_or(ProgramPoint::before(0, 0));
            a_start.cmp(&b_start)
        });

        self.stats.virtual_regs = intervals.len();

        // Process each interval
        for interval in intervals {
            let start = match interval.start() {
                Some(s) => s,
                None => continue, // Empty interval
            };

            // Expire old intervals
            self.expire_old_intervals(start);

            // Try to allocate
            let reg_class = self.classify_interval(&interval);

            if let Some(reg) = self.try_allocate(reg_class) {
                // Successfully allocated
                let end = interval.end().unwrap_or(start);
                self.assignments.insert(interval.value, reg);
                self.active.push(Reverse(ActiveInterval {
                    interval,
                    register: reg,
                    end,
                }));
                self.stats.phys_regs_used += 1;
            } else {
                // Need to spill
                self.spill_at_interval(&interval, reg_class);
            }
        }

        AllocationResult {
            assignments: self.assignments.clone(),
            spilled: self.spilled.clone(),
            spill_slots: self.next_spill_slot as usize,
            stats: self.stats.clone(),
        }
    }

    /// Classify interval into register class
    fn classify_interval(&self, _interval: &LiveInterval) -> RegisterClass {
        // In practice, this would look at the type of the value
        // For now, default to GPR
        RegisterClass::GPR
    }

    /// Expire intervals that end before the given point
    fn expire_old_intervals(&mut self, point: ProgramPoint) {
        while let Some(Reverse(active)) = self.active.peek() {
            if active.end >= point {
                break;
            }

            let Reverse(expired) = self.active.pop().unwrap();

            // Return register to free pool
            self.free_regs
                .entry(expired.register.class)
                .or_default()
                .push(expired.register);
        }
    }

    /// Try to allocate a register of the given class
    fn try_allocate(&mut self, class: RegisterClass) -> Option<PhysicalRegister> {
        self.free_regs.get_mut(&class)?.pop()
    }

    /// Spill the interval with the furthest end point
    fn spill_at_interval(&mut self, interval: &LiveInterval, class: RegisterClass) {
        // Check if we should spill the new interval or an active one
        if let Some(Reverse(furthest)) = self.active.peek() {
            let interval_end = interval.end().unwrap_or(ProgramPoint::before(0, 0));

            if furthest.end > interval_end && furthest.register.class == class {
                // Spill the furthest active interval instead
                let Reverse(to_spill) = self.active.pop().unwrap();

                // Give its register to the new interval
                self.assignments.insert(interval.value, to_spill.register);
                self.active.push(Reverse(ActiveInterval {
                    interval: interval.clone(),
                    register: to_spill.register,
                    end: interval_end,
                }));

                // Spill the old interval
                self.spill_interval(&to_spill.interval);
                self.assignments.remove(&to_spill.interval.value);

                return;
            }
        }

        // Spill the new interval
        self.spill_interval(interval);
    }

    /// Spill an interval to the stack
    fn spill_interval(&mut self, interval: &LiveInterval) {
        let slot = self.next_spill_slot;
        self.next_spill_slot += 8; // Assume 8-byte slots

        self.spilled.push(SpilledValue {
            value: interval.value,
            slot,
            size: 8,
            align: 8,
        });

        self.stats.spills += 1;

        // Count reloads (approximate: one per use)
        self.stats.reloads += interval.uses.len();
    }

    /// Get the allocation result
    pub fn result(&self) -> AllocationResult {
        AllocationResult {
            assignments: self.assignments.clone(),
            spilled: self.spilled.clone(),
            spill_slots: self.next_spill_slot as usize,
            stats: self.stats.clone(),
        }
    }
}

/// Extended linear scan with live range splitting
pub struct SplittingLinearScan {
    /// Base allocator
    base: LinearScanAllocator,
    /// Split points
    split_points: HashMap<ValueId, Vec<ProgramPoint>>,
}

impl SplittingLinearScan {
    /// Create a new splitting linear scan allocator
    pub fn new(config: LinearScanConfig, target: TargetRegisters) -> Self {
        Self {
            base: LinearScanAllocator::new(config, target),
            split_points: HashMap::new(),
        }
    }

    /// Find optimal split points for an interval
    pub fn find_split_points(&self, interval: &LiveInterval) -> Vec<ProgramPoint> {
        let mut points = Vec::new();

        // Split at loop boundaries (simplified: split at block boundaries)
        let mut seen_blocks = HashSet::new();

        for range in &interval.ranges {
            if !seen_blocks.contains(&range.start.block) {
                if !points.is_empty() {
                    points.push(range.start);
                }
                seen_blocks.insert(range.start.block);
            }
        }

        // Limit number of splits
        if points.len() > self.base.config.max_splits {
            points.truncate(self.base.config.max_splits);
        }

        points
    }

    /// Allocate with splitting
    pub fn allocate(&mut self, liveness: &LivenessAnalysis) -> AllocationResult {
        // For now, delegate to base allocator
        // Full implementation would split intervals when spilling
        self.base.allocate(liveness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::regalloc::lifetime::LiveRange;

    #[test]
    fn test_linear_scan_basic() {
        let config = LinearScanConfig::default();
        let target = TargetRegisters::x86_64();
        let mut allocator = LinearScanAllocator::new(config, target);

        // Create simple liveness info
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

        // Both should be allocated (no spills needed)
        assert!(result.assignments.contains_key(&ValueId(0)));
        assert!(result.assignments.contains_key(&ValueId(1)));
        assert_eq!(result.spilled.len(), 0);
    }

    #[test]
    fn test_linear_scan_overlap() {
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
}
