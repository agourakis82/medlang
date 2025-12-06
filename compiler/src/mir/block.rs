//! MIR Basic Blocks and Terminators
//!
//! Basic blocks are sequences of instructions with a single entry point
//! and a terminator that defines control flow to successor blocks.

use super::inst::*;
use super::types::*;
use super::value::*;

/// A basic block in MIR
#[derive(Clone, Debug)]
pub struct BasicBlock {
    /// Block identifier
    pub id: BlockId,
    /// Block parameters (for SSA phi nodes)
    pub params: Vec<BlockParam>,
    /// Instructions in the block
    pub instructions: Vec<Instruction>,
    /// Block terminator (defines successors)
    pub terminator: Terminator,
    /// Debug name (optional)
    pub name: Option<String>,
}

impl BasicBlock {
    /// Create a new basic block
    pub fn new(id: BlockId) -> Self {
        Self {
            id,
            params: Vec::new(),
            instructions: Vec::new(),
            terminator: Terminator::Unreachable,
            name: None,
        }
    }

    /// Create a new basic block with a name
    pub fn with_name(id: BlockId, name: &str) -> Self {
        Self {
            id,
            params: Vec::new(),
            instructions: Vec::new(),
            terminator: Terminator::Unreachable,
            name: Some(name.to_string()),
        }
    }

    /// Add a block parameter
    pub fn add_param(&mut self, value: ValueId, ty: MirType) {
        self.params.push(BlockParam { value, ty });
    }

    /// Add an instruction to the block
    pub fn push(&mut self, inst: Instruction) {
        self.instructions.push(inst);
    }

    /// Set the terminator
    pub fn terminate(&mut self, terminator: Terminator) {
        self.terminator = terminator;
    }

    /// Get successor block IDs
    pub fn successors(&self) -> Vec<BlockId> {
        self.terminator.successors()
    }

    /// Check if block is empty (no instructions)
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Get the number of instructions
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Check if block has been terminated
    pub fn is_terminated(&self) -> bool {
        !matches!(self.terminator, Terminator::Unreachable)
    }

    /// Get all values defined in this block
    pub fn defined_values(&self) -> Vec<ValueId> {
        let mut values: Vec<ValueId> = self.params.iter().map(|p| p.value).collect();
        for inst in &self.instructions {
            if let Some(result) = inst.result {
                values.push(result);
            }
        }
        values
    }

    /// Get all values used in this block
    pub fn used_values(&self) -> Vec<ValueId> {
        let mut values = Vec::new();
        for inst in &self.instructions {
            values.extend(inst.operands());
        }
        values.extend(self.terminator.operands());
        values
    }
}

/// Block parameter (used instead of phi nodes)
#[derive(Clone, Debug)]
pub struct BlockParam {
    /// Value ID for this parameter
    pub value: ValueId,
    /// Type of the parameter
    pub ty: MirType,
}

impl BlockParam {
    pub fn new(value: ValueId, ty: MirType) -> Self {
        Self { value, ty }
    }
}

/// Block terminator - defines control flow to successors
#[derive(Clone, Debug)]
pub enum Terminator {
    /// Unconditional jump
    Goto { target: BlockId, args: Vec<ValueId> },

    /// Conditional branch
    Branch {
        cond: ValueId,
        then_block: BlockId,
        then_args: Vec<ValueId>,
        else_block: BlockId,
        else_args: Vec<ValueId>,
    },

    /// Multi-way switch
    Switch {
        value: ValueId,
        /// Default target
        default: BlockId,
        default_args: Vec<ValueId>,
        /// (discriminant, target, args)
        cases: Vec<(u64, BlockId, Vec<ValueId>)>,
    },

    /// Return from function
    Return { value: Option<ValueId> },

    /// Function call with continuation
    Call {
        callee: Callee,
        args: Vec<ValueId>,
        ret_ty: MirType,
        /// Result value (if non-void)
        result: Option<ValueId>,
        /// Continuation block
        next: BlockId,
        next_args: Vec<ValueId>,
    },

    /// Invoke with landing pad (for unwinding)
    Invoke {
        callee: Callee,
        args: Vec<ValueId>,
        ret_ty: MirType,
        result: Option<ValueId>,
        /// Normal return block
        normal: BlockId,
        normal_args: Vec<ValueId>,
        /// Exception handler block
        unwind: BlockId,
        unwind_args: Vec<ValueId>,
    },

    /// Resume unwinding
    Resume { exception: ValueId },

    /// Unreachable (undefined behavior if reached)
    Unreachable,

    /// Abort program
    Abort { message: String },

    /// Tail call (no continuation)
    TailCall { callee: Callee, args: Vec<ValueId> },
}

impl Terminator {
    /// Get all successor blocks
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Terminator::Goto { target, .. } => vec![*target],
            Terminator::Branch {
                then_block,
                else_block,
                ..
            } => vec![*then_block, *else_block],
            Terminator::Switch { default, cases, .. } => {
                let mut succs = vec![*default];
                for (_, target, _) in cases {
                    succs.push(*target);
                }
                succs
            }
            Terminator::Return { .. } => vec![],
            Terminator::Call { next, .. } => vec![*next],
            Terminator::Invoke { normal, unwind, .. } => vec![*normal, *unwind],
            Terminator::Resume { .. } => vec![],
            Terminator::Unreachable => vec![],
            Terminator::Abort { .. } => vec![],
            Terminator::TailCall { .. } => vec![],
        }
    }

    /// Get mutable references to successor blocks (for CFG transformation)
    pub fn successors_mut(&mut self) -> Vec<&mut BlockId> {
        match self {
            Terminator::Goto { target, .. } => vec![target],
            Terminator::Branch {
                then_block,
                else_block,
                ..
            } => vec![then_block, else_block],
            Terminator::Switch { default, cases, .. } => {
                let mut succs = vec![default];
                for (_, target, _) in cases {
                    succs.push(target);
                }
                succs
            }
            Terminator::Return { .. } => vec![],
            Terminator::Call { next, .. } => vec![next],
            Terminator::Invoke { normal, unwind, .. } => vec![normal, unwind],
            Terminator::Resume { .. } => vec![],
            Terminator::Unreachable => vec![],
            Terminator::Abort { .. } => vec![],
            Terminator::TailCall { .. } => vec![],
        }
    }

    /// Get all value operands
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            Terminator::Goto { args, .. } => args.clone(),
            Terminator::Branch {
                cond,
                then_args,
                else_args,
                ..
            } => {
                let mut ops = vec![*cond];
                ops.extend(then_args);
                ops.extend(else_args);
                ops
            }
            Terminator::Switch {
                value,
                default_args,
                cases,
                ..
            } => {
                let mut ops = vec![*value];
                ops.extend(default_args);
                for (_, _, args) in cases {
                    ops.extend(args);
                }
                ops
            }
            Terminator::Return { value } => value.iter().copied().collect(),
            Terminator::Call {
                args, next_args, ..
            } => {
                let mut ops = args.clone();
                ops.extend(next_args);
                ops
            }
            Terminator::Invoke {
                args,
                normal_args,
                unwind_args,
                ..
            } => {
                let mut ops = args.clone();
                ops.extend(normal_args);
                ops.extend(unwind_args);
                ops
            }
            Terminator::Resume { exception } => vec![*exception],
            Terminator::Unreachable => vec![],
            Terminator::Abort { .. } => vec![],
            Terminator::TailCall { args, .. } => args.clone(),
        }
    }

    /// Check if this terminator can fall through (has successors)
    pub fn can_fall_through(&self) -> bool {
        !matches!(
            self,
            Terminator::Return { .. }
                | Terminator::Resume { .. }
                | Terminator::Unreachable
                | Terminator::Abort { .. }
                | Terminator::TailCall { .. }
        )
    }

    /// Check if this is a returning terminator
    pub fn is_return(&self) -> bool {
        matches!(
            self,
            Terminator::Return { .. } | Terminator::TailCall { .. }
        )
    }

    /// Check if this is an exception-handling terminator
    pub fn is_exception(&self) -> bool {
        matches!(self, Terminator::Invoke { .. } | Terminator::Resume { .. })
    }
}

/// Builder for constructing basic blocks
pub struct BlockBuilder {
    block: BasicBlock,
    value_gen: ValueIdGen,
}

impl BlockBuilder {
    pub fn new(id: BlockId) -> Self {
        Self {
            block: BasicBlock::new(id),
            value_gen: ValueIdGen::new(),
        }
    }

    pub fn with_value_gen(id: BlockId, value_gen: ValueIdGen) -> Self {
        Self {
            block: BasicBlock::new(id),
            value_gen,
        }
    }

    /// Add a block parameter
    pub fn param(&mut self, ty: MirType) -> ValueId {
        let value = self.value_gen.next();
        self.block.add_param(value, ty);
        value
    }

    /// Add an instruction and return its result
    pub fn push_op(&mut self, op: Operation, ty: MirType) -> ValueId {
        let result = self.value_gen.next();
        let inst = Instruction::new(op, ty).with_result(result);
        self.block.push(inst);
        result
    }

    /// Add a void instruction (no result)
    pub fn push_void(&mut self, op: Operation) {
        let inst = Instruction::new(op, MirType::Void);
        self.block.push(inst);
    }

    /// Set the terminator
    pub fn terminate(&mut self, terminator: Terminator) {
        self.block.terminate(terminator);
    }

    /// Finish building and return the block
    pub fn build(self) -> (BasicBlock, ValueIdGen) {
        (self.block, self.value_gen)
    }

    /// Get the value generator (for passing to next block)
    pub fn value_gen(&self) -> &ValueIdGen {
        &self.value_gen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_block_creation() {
        let mut block = BasicBlock::new(BlockId(0));
        assert!(block.is_empty());
        assert!(!block.is_terminated());

        block.add_param(ValueId(0), MirType::F64);
        block.push(Instruction::new(
            Operation::ConstFloat {
                value: 1.0,
                ty: MirType::F64,
            },
            MirType::F64,
        ));
        block.terminate(Terminator::Return { value: None });

        assert!(!block.is_empty());
        assert!(block.is_terminated());
        assert_eq!(block.len(), 1);
    }

    #[test]
    fn test_terminator_successors() {
        let goto = Terminator::Goto {
            target: BlockId(1),
            args: vec![],
        };
        assert_eq!(goto.successors(), vec![BlockId(1)]);

        let branch = Terminator::Branch {
            cond: ValueId(0),
            then_block: BlockId(1),
            then_args: vec![],
            else_block: BlockId(2),
            else_args: vec![],
        };
        assert_eq!(branch.successors(), vec![BlockId(1), BlockId(2)]);

        let ret = Terminator::Return { value: None };
        assert!(ret.successors().is_empty());
    }

    #[test]
    fn test_block_builder() {
        let mut builder = BlockBuilder::new(BlockId(0));

        let x = builder.param(MirType::F64);
        let y = builder.param(MirType::F64);

        let sum = builder.push_op(Operation::FAdd { lhs: x, rhs: y }, MirType::F64);

        builder.terminate(Terminator::Return { value: Some(sum) });

        let (block, _) = builder.build();
        assert_eq!(block.params.len(), 2);
        assert_eq!(block.instructions.len(), 1);
        assert!(block.is_terminated());
    }

    #[test]
    fn test_defined_and_used_values() {
        let mut block = BasicBlock::new(BlockId(0));
        block.add_param(ValueId(0), MirType::F64);
        block.add_param(ValueId(1), MirType::F64);

        block.push(
            Instruction::new(
                Operation::FAdd {
                    lhs: ValueId(0),
                    rhs: ValueId(1),
                },
                MirType::F64,
            )
            .with_result(ValueId(2)),
        );

        block.terminate(Terminator::Return {
            value: Some(ValueId(2)),
        });

        let defined = block.defined_values();
        assert_eq!(defined, vec![ValueId(0), ValueId(1), ValueId(2)]);

        let used = block.used_values();
        assert!(used.contains(&ValueId(0)));
        assert!(used.contains(&ValueId(1)));
        assert!(used.contains(&ValueId(2)));
    }

    #[test]
    fn test_switch_terminator() {
        let switch = Terminator::Switch {
            value: ValueId(0),
            default: BlockId(99),
            default_args: vec![],
            cases: vec![
                (0, BlockId(1), vec![]),
                (1, BlockId(2), vec![]),
                (2, BlockId(3), vec![]),
            ],
        };

        let succs = switch.successors();
        assert_eq!(succs.len(), 4);
        assert!(succs.contains(&BlockId(99)));
        assert!(succs.contains(&BlockId(1)));
        assert!(succs.contains(&BlockId(2)));
        assert!(succs.contains(&BlockId(3)));
    }

    #[test]
    fn test_invoke_terminator() {
        let invoke = Terminator::Invoke {
            callee: Callee::Direct("may_throw".to_string()),
            args: vec![ValueId(0)],
            ret_ty: MirType::I32,
            result: Some(ValueId(1)),
            normal: BlockId(1),
            normal_args: vec![ValueId(1)],
            unwind: BlockId(2),
            unwind_args: vec![],
        };

        assert!(invoke.is_exception());
        assert!(invoke.can_fall_through());
        assert_eq!(invoke.successors(), vec![BlockId(1), BlockId(2)]);
    }
}
