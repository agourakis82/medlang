//! Spill Code Generation
//!
//! Handles insertion of spill and reload instructions when register
//! allocation runs out of physical registers.

use std::collections::HashMap;

use crate::mir::block::BasicBlock;
use crate::mir::function::MirFunction;
use crate::mir::inst::{Instruction, Operation};
use crate::mir::types::MirType;
use crate::mir::value::{ValueId, ValueIdGen};

use super::target::PhysicalRegister;
use super::{AllocationResult, SpilledValue};

/// Cost of spilling a value
#[derive(Debug, Clone, Copy)]
pub struct SpillCost {
    /// Store cost
    pub store: f64,
    /// Load cost
    pub load: f64,
    /// Can this value be rematerialized instead?
    pub rematerializable: bool,
    /// Cost of rematerialization
    pub remat_cost: f64,
}

impl Default for SpillCost {
    fn default() -> Self {
        Self {
            store: 1.0,
            load: 1.0,
            rematerializable: false,
            remat_cost: f64::INFINITY,
        }
    }
}

impl SpillCost {
    /// Create cost for a rematerializable value
    pub fn rematerializable(cost: f64) -> Self {
        Self {
            store: 1.0,
            load: 1.0,
            rematerializable: true,
            remat_cost: cost,
        }
    }

    /// Total cost for a given number of uses
    pub fn total(&self, stores: usize, loads: usize) -> f64 {
        if self.rematerializable {
            (stores as f64 * self.store).min(loads as f64 * self.remat_cost)
        } else {
            stores as f64 * self.store + loads as f64 * self.load
        }
    }
}

/// Strategy for spilling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpillStrategy {
    /// Spill everywhere (simple)
    SpillEverywhere,
    /// Spill only at definition, reload at uses
    SpillAtDef,
    /// Try to rematerialize cheap values
    Rematerialize,
    /// Split live range and spill only part
    SplitAndSpill,
}

/// Spill code generator
pub struct Spiller {
    /// Spill strategy
    strategy: SpillStrategy,
    /// Value ID generator
    id_gen: ValueIdGen,
    /// Spill slot assignments
    spill_slots: HashMap<ValueId, i32>,
    /// Rematerialization instructions
    remat_instrs: HashMap<ValueId, Instruction>,
    /// Frame pointer register
    _frame_pointer: PhysicalRegister,
    /// Stack offset for spill area
    _spill_area_offset: i32,
}

impl Spiller {
    /// Create a new spiller
    pub fn new(
        strategy: SpillStrategy,
        frame_pointer: PhysicalRegister,
        spill_area_offset: i32,
    ) -> Self {
        Self {
            strategy,
            id_gen: ValueIdGen::new(),
            spill_slots: HashMap::new(),
            remat_instrs: HashMap::new(),
            _frame_pointer: frame_pointer,
            _spill_area_offset: spill_area_offset,
        }
    }

    /// Insert spill code into a function
    pub fn insert_spill_code(
        &mut self,
        func: &MirFunction,
        alloc_result: &AllocationResult,
    ) -> MirFunction {
        let mut new_func = func.clone();

        for spilled in &alloc_result.spilled {
            self.spill_slots.insert(spilled.value, spilled.slot);
        }

        if self.strategy == SpillStrategy::Rematerialize {
            self.find_rematerializable(func, alloc_result);
        }

        for block in &mut new_func.blocks {
            *block = self.transform_block(block, alloc_result);
        }

        new_func
    }

    /// Find values that can be rematerialized
    fn find_rematerializable(&mut self, func: &MirFunction, alloc_result: &AllocationResult) {
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(result) = inst.result {
                    if !alloc_result.spilled.iter().any(|s| s.value == result) {
                        continue;
                    }

                    if self.is_rematerializable(&inst.op) {
                        self.remat_instrs.insert(result, inst.clone());
                    }
                }
            }
        }
    }

    /// Check if an operation can be rematerialized
    fn is_rematerializable(&self, op: &Operation) -> bool {
        matches!(
            op,
            Operation::ConstInt { .. }
                | Operation::ConstFloat { .. }
                | Operation::ConstBool { .. }
                | Operation::ZeroInit { .. }
        )
    }

    /// Transform a block to insert spill/reload code
    fn transform_block(
        &mut self,
        block: &BasicBlock,
        _alloc_result: &AllocationResult,
    ) -> BasicBlock {
        let mut new_block = BasicBlock::new(block.id);
        new_block.params = block.params.clone();
        new_block.name = block.name.clone();
        new_block.terminator = block.terminator.clone();

        for inst in &block.instructions {
            // Insert reloads before uses
            let mut reloads = Vec::new();
            for used in self.get_uses(&inst.op) {
                if self.spill_slots.contains_key(&used) {
                    if let Some(reload) = self.generate_reload(used, &inst.ty) {
                        reloads.push(reload);
                    }
                }
            }
            new_block.instructions.extend(reloads);

            // Insert the original instruction
            new_block.instructions.push(inst.clone());

            // Insert spill after definition
            if let Some(result) = inst.result {
                if self.spill_slots.contains_key(&result) {
                    if let Some(spill) = self.generate_spill(result, &inst.ty) {
                        new_block.instructions.push(spill);
                    }
                }
            }
        }

        new_block
    }

    /// Generate a spill instruction
    fn generate_spill(&mut self, value: ValueId, _ty: &MirType) -> Option<Instruction> {
        let _slot = self.spill_slots.get(&value)?;

        // Create store to stack (simplified)
        Some(Instruction::new(
            Operation::Store {
                ptr: value,
                value,
                volatile: false,
                align: 8,
            },
            MirType::Void,
        ))
    }

    /// Generate a reload instruction
    fn generate_reload(&mut self, value: ValueId, ty: &MirType) -> Option<Instruction> {
        // Check if we can rematerialize instead
        if self.strategy == SpillStrategy::Rematerialize {
            if let Some(remat_inst) = self.remat_instrs.get(&value) {
                let new_result = self.id_gen.next();
                return Some(remat_inst.clone().with_result(new_result));
            }
        }

        let _slot = self.spill_slots.get(&value)?;

        // Load from stack (simplified)
        let result = self.id_gen.next();
        Some(
            Instruction::new(
                Operation::Load {
                    ptr: value,
                    ty: ty.clone(),
                    volatile: false,
                    align: 8,
                },
                ty.clone(),
            )
            .with_result(result),
        )
    }

    /// Get values used by an operation
    fn get_uses(&self, op: &Operation) -> Vec<ValueId> {
        let mut uses = Vec::new();

        match op {
            Operation::FAdd { lhs, rhs }
            | Operation::FSub { lhs, rhs }
            | Operation::FMul { lhs, rhs }
            | Operation::FDiv { lhs, rhs }
            | Operation::IAdd { lhs, rhs }
            | Operation::ISub { lhs, rhs }
            | Operation::IMul { lhs, rhs }
            | Operation::IDiv { lhs, rhs } => {
                uses.push(*lhs);
                uses.push(*rhs);
            }

            Operation::FNeg { operand }
            | Operation::Not { operand }
            | Operation::Abs { operand } => {
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

            _ => {}
        }

        uses
    }
}

/// Stack frame layout for spills
#[derive(Debug, Clone)]
pub struct SpillFrameLayout {
    /// Total size of spill area
    pub size: usize,
    /// Alignment of spill area
    pub align: usize,
    /// Slot allocations
    pub slots: Vec<SpillSlot>,
}

/// A single spill slot
#[derive(Debug, Clone)]
pub struct SpillSlot {
    /// Offset from frame pointer
    pub offset: i32,
    /// Size of slot
    pub size: u32,
    /// Values that share this slot
    pub values: Vec<ValueId>,
}

impl SpillFrameLayout {
    /// Create a new spill frame layout
    pub fn new() -> Self {
        Self {
            size: 0,
            align: 8,
            slots: Vec::new(),
        }
    }

    /// Allocate a new spill slot
    pub fn allocate(&mut self, size: u32, align: u32) -> i32 {
        let aligned_size = (self.size + align as usize - 1) & !(align as usize - 1);
        let offset = aligned_size as i32;

        self.slots.push(SpillSlot {
            offset,
            size,
            values: Vec::new(),
        });

        self.size = aligned_size + size as usize;
        self.align = self.align.max(align as usize);

        offset
    }

    /// Try to reuse an existing slot
    pub fn try_reuse(
        &mut self,
        size: u32,
        _align: u32,
        non_interfering: &[ValueId],
    ) -> Option<i32> {
        for slot in &mut self.slots {
            if slot.size >= size {
                let interferes = slot.values.iter().any(|v| !non_interfering.contains(v));
                if !interferes {
                    return Some(slot.offset);
                }
            }
        }
        None
    }
}

impl Default for SpillFrameLayout {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spill_cost() {
        let cost = SpillCost::default();
        assert_eq!(cost.total(1, 3), 4.0);

        let remat = SpillCost::rematerializable(0.5);
        assert!(remat.rematerializable);
    }

    #[test]
    fn test_spill_frame_layout() {
        let mut layout = SpillFrameLayout::new();

        let slot1 = layout.allocate(8, 8);
        assert_eq!(slot1, 0);

        let slot2 = layout.allocate(8, 8);
        assert_eq!(slot2, 8);

        assert_eq!(layout.size, 16);
    }
}
