//! Reproducible Summation Algorithms
//!
//! Provides deterministic summation algorithms that produce identical results
//! regardless of execution order, making them suitable for parallel reduction.

use crate::mir::block::BasicBlock;
use crate::mir::inst::{FloatPredicate, Instruction, Operation};
use crate::mir::types::MirType;
use crate::mir::value::{ValueId, ValueIdGen};

/// Algorithm for compensated summation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SummationAlgorithm {
    /// Simple sequential sum (non-deterministic in parallel)
    Naive,
    /// Kahan summation with error compensation
    Kahan,
    /// Neumaier's improved Kahan algorithm
    Neumaier,
    /// Pairwise (cascade) summation
    Pairwise,
    /// Tree reduction (deterministic parallel)
    Tree,
    /// Exact accumulation using integer representation
    ExactAccumulator,
}

/// State for Kahan summation
#[derive(Debug, Clone)]
pub struct KahanSum {
    /// Running sum
    pub sum: ValueId,
    /// Compensation term
    pub compensation: ValueId,
    /// Type of values being summed
    pub ty: MirType,
}

impl KahanSum {
    /// Create a new Kahan sum state
    pub fn new(sum: ValueId, compensation: ValueId, ty: MirType) -> Self {
        Self {
            sum,
            compensation,
            ty,
        }
    }

    /// Generate instructions to add a value to the sum
    pub fn add(&self, value: ValueId, id_gen: &mut ValueIdGen) -> (Vec<Instruction>, KahanSum) {
        let mut insts = Vec::new();

        // y = value - compensation
        let y = id_gen.next();
        insts.push(
            Instruction::new(
                Operation::FSub {
                    lhs: value,
                    rhs: self.compensation,
                },
                self.ty.clone(),
            )
            .with_result(y),
        );

        // t = sum + y
        let t = id_gen.next();
        insts.push(
            Instruction::new(
                Operation::FAdd {
                    lhs: self.sum,
                    rhs: y,
                },
                self.ty.clone(),
            )
            .with_result(t),
        );

        // new_compensation = (t - sum) - y
        let t_minus_sum = id_gen.next();
        insts.push(
            Instruction::new(
                Operation::FSub {
                    lhs: t,
                    rhs: self.sum,
                },
                self.ty.clone(),
            )
            .with_result(t_minus_sum),
        );

        let new_comp = id_gen.next();
        insts.push(
            Instruction::new(
                Operation::FSub {
                    lhs: t_minus_sum,
                    rhs: y,
                },
                self.ty.clone(),
            )
            .with_result(new_comp),
        );

        let new_state = KahanSum::new(t, new_comp, self.ty.clone());
        (insts, new_state)
    }

    /// Get the final sum value
    pub fn finalize(&self) -> ValueId {
        self.sum
    }
}

/// State for Neumaier summation (improved Kahan)
#[derive(Debug, Clone)]
pub struct NeumaierSum {
    /// Running sum
    pub sum: ValueId,
    /// Compensation term
    pub compensation: ValueId,
    /// Type of values being summed
    pub ty: MirType,
}

impl NeumaierSum {
    /// Create a new Neumaier sum state
    pub fn new(sum: ValueId, compensation: ValueId, ty: MirType) -> Self {
        Self {
            sum,
            compensation,
            ty,
        }
    }

    /// Generate instructions to add a value
    pub fn add(&self, value: ValueId, id_gen: &mut ValueIdGen) -> (Vec<Instruction>, NeumaierSum) {
        let mut insts = Vec::new();

        // t = sum + value
        let t = id_gen.next();
        insts.push(
            Instruction::new(
                Operation::FAdd {
                    lhs: self.sum,
                    rhs: value,
                },
                self.ty.clone(),
            )
            .with_result(t),
        );

        // abs_sum = abs(sum)
        let abs_sum = id_gen.next();
        insts.push(
            Instruction::new(Operation::Abs { operand: self.sum }, self.ty.clone())
                .with_result(abs_sum),
        );

        // abs_value = abs(value)
        let abs_value = id_gen.next();
        insts.push(
            Instruction::new(Operation::Abs { operand: value }, self.ty.clone())
                .with_result(abs_value),
        );

        // Branch 1: (sum - t) + value
        let sum_minus_t = id_gen.next();
        insts.push(
            Instruction::new(
                Operation::FSub {
                    lhs: self.sum,
                    rhs: t,
                },
                self.ty.clone(),
            )
            .with_result(sum_minus_t),
        );

        let comp1 = id_gen.next();
        insts.push(
            Instruction::new(
                Operation::FAdd {
                    lhs: sum_minus_t,
                    rhs: value,
                },
                self.ty.clone(),
            )
            .with_result(comp1),
        );

        // Branch 2: (value - t) + sum
        let val_minus_t = id_gen.next();
        insts.push(
            Instruction::new(Operation::FSub { lhs: value, rhs: t }, self.ty.clone())
                .with_result(val_minus_t),
        );

        let comp2 = id_gen.next();
        insts.push(
            Instruction::new(
                Operation::FAdd {
                    lhs: val_minus_t,
                    rhs: self.sum,
                },
                self.ty.clone(),
            )
            .with_result(comp2),
        );

        // cond = abs_sum >= abs_value
        let cond = id_gen.next();
        insts.push(
            Instruction::new(
                Operation::FCmp {
                    pred: FloatPredicate::OGe,
                    lhs: abs_sum,
                    rhs: abs_value,
                },
                MirType::Bool,
            )
            .with_result(cond),
        );

        // delta = select(cond, comp1, comp2)
        let delta = id_gen.next();
        insts.push(
            Instruction::new(
                Operation::Select {
                    cond,
                    then_val: comp1,
                    else_val: comp2,
                },
                self.ty.clone(),
            )
            .with_result(delta),
        );

        // new_compensation = compensation + delta
        let new_comp = id_gen.next();
        insts.push(
            Instruction::new(
                Operation::FAdd {
                    lhs: self.compensation,
                    rhs: delta,
                },
                self.ty.clone(),
            )
            .with_result(new_comp),
        );

        let new_state = NeumaierSum::new(t, new_comp, self.ty.clone());
        (insts, new_state)
    }

    /// Finalize and return sum + compensation
    pub fn finalize(&self, id_gen: &mut ValueIdGen) -> (Vec<Instruction>, ValueId) {
        let result = id_gen.next();
        let inst = Instruction::new(
            Operation::FAdd {
                lhs: self.sum,
                rhs: self.compensation,
            },
            self.ty.clone(),
        )
        .with_result(result);
        (vec![inst], result)
    }
}

/// Pairwise summation state
#[derive(Debug, Clone)]
pub struct PairwiseSum {
    /// Values to be summed (organized as a tree)
    pub values: Vec<ValueId>,
    /// Type of values
    pub ty: MirType,
}

impl PairwiseSum {
    /// Create a new pairwise sum
    pub fn new(values: Vec<ValueId>, ty: MirType) -> Self {
        Self { values, ty }
    }

    /// Generate pairwise summation instructions
    pub fn generate(&self, id_gen: &mut ValueIdGen) -> (Vec<Instruction>, ValueId) {
        if self.values.is_empty() {
            let zero = id_gen.next();
            let inst = Instruction::new(
                Operation::ConstFloat {
                    value: 0.0,
                    ty: self.ty.clone(),
                },
                self.ty.clone(),
            )
            .with_result(zero);
            return (vec![inst], zero);
        }

        if self.values.len() == 1 {
            return (vec![], self.values[0]);
        }

        let mut insts = Vec::new();
        let mut current_level = self.values.clone();

        while current_level.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in current_level.chunks(2) {
                if chunk.len() == 2 {
                    let sum = id_gen.next();
                    insts.push(
                        Instruction::new(
                            Operation::FAdd {
                                lhs: chunk[0],
                                rhs: chunk[1],
                            },
                            self.ty.clone(),
                        )
                        .with_result(sum),
                    );
                    next_level.push(sum);
                } else {
                    next_level.push(chunk[0]);
                }
            }

            current_level = next_level;
        }

        (insts, current_level[0])
    }
}

/// Tree summation for deterministic parallel reduction
#[derive(Debug, Clone)]
pub struct TreeSum {
    /// Leaf values
    pub leaves: Vec<ValueId>,
    /// Type of values
    pub ty: MirType,
    /// Branching factor (typically 2)
    pub branching_factor: usize,
}

impl TreeSum {
    /// Create a new tree sum
    pub fn new(leaves: Vec<ValueId>, ty: MirType) -> Self {
        Self {
            leaves,
            ty,
            branching_factor: 2,
        }
    }

    /// Set branching factor
    pub fn with_branching_factor(mut self, factor: usize) -> Self {
        self.branching_factor = factor.max(2);
        self
    }

    /// Generate tree reduction with deterministic order
    pub fn generate(&self, id_gen: &mut ValueIdGen) -> (Vec<Instruction>, ValueId) {
        if self.leaves.is_empty() {
            let zero = id_gen.next();
            let inst = Instruction::new(
                Operation::ConstFloat {
                    value: 0.0,
                    ty: self.ty.clone(),
                },
                self.ty.clone(),
            )
            .with_result(zero);
            return (vec![inst], zero);
        }

        if self.leaves.len() == 1 {
            return (vec![], self.leaves[0]);
        }

        let mut insts = Vec::new();
        let mut current = self.leaves.clone();

        while current.len() > 1 {
            let mut next = Vec::new();

            for chunk in current.chunks(self.branching_factor) {
                if chunk.len() == 1 {
                    next.push(chunk[0]);
                } else {
                    let mut acc = chunk[0];
                    for &val in &chunk[1..] {
                        let sum = id_gen.next();
                        insts.push(
                            Instruction::new(
                                Operation::FAdd { lhs: acc, rhs: val },
                                self.ty.clone(),
                            )
                            .with_result(sum),
                        );
                        acc = sum;
                    }
                    next.push(acc);
                }
            }

            current = next;
        }

        (insts, current[0])
    }
}

/// Compensated sum value (for runtime use)
#[derive(Debug, Clone, Copy)]
pub struct CompensatedSum {
    /// The sum value
    pub sum: f64,
    /// Compensation term
    pub compensation: f64,
}

impl CompensatedSum {
    /// Create a new compensated sum initialized to zero
    pub fn zero() -> Self {
        Self {
            sum: 0.0,
            compensation: 0.0,
        }
    }

    /// Add a value using Kahan summation
    pub fn add_kahan(&mut self, value: f64) {
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    /// Add a value using Neumaier summation
    pub fn add_neumaier(&mut self, value: f64) {
        let t = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.compensation += (self.sum - t) + value;
        } else {
            self.compensation += (value - t) + self.sum;
        }
        self.sum = t;
    }

    /// Get the final result
    pub fn result(&self) -> f64 {
        self.sum + self.compensation
    }
}

/// Generator for reproducible reductions
pub struct ReproducibleReduction {
    /// Algorithm to use
    algorithm: SummationAlgorithm,
    /// Type of values
    ty: MirType,
}

impl ReproducibleReduction {
    /// Create a new reproducible reduction
    pub fn new(algorithm: SummationAlgorithm, ty: MirType) -> Self {
        Self { algorithm, ty }
    }

    /// Transform a naive reduction into a reproducible one
    pub fn transform_block(&self, block: &BasicBlock, id_gen: &mut ValueIdGen) -> Vec<Instruction> {
        // For now, pass through - full implementation would detect and transform reductions
        block.instructions.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kahan_sum() {
        let mut sum = CompensatedSum::zero();
        sum.add_kahan(1.0);
        sum.add_kahan(1e-16);
        sum.add_kahan(1e-16);
        assert!((sum.result() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_neumaier_sum() {
        let mut sum = CompensatedSum::zero();
        sum.add_neumaier(1e16);
        sum.add_neumaier(1.0);
        sum.add_neumaier(-1e16);
        assert!((sum.result() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_sum() {
        let mut id_gen = ValueIdGen::new();
        let values: Vec<ValueId> = (0..8).map(|_| id_gen.next()).collect();
        let pairwise = PairwiseSum::new(values, MirType::F64);
        let (insts, _result) = pairwise.generate(&mut id_gen);
        assert_eq!(insts.len(), 7);
    }
}
