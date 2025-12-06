//! Reverse-Mode Automatic Differentiation
//!
//! Reverse mode computes gradients by backpropagating adjoint values.
//! For each output y, we compute ∂L/∂x for all inputs x where L is a scalar loss.
//!
//! Reverse mode is efficient when:
//! - Number of inputs >> number of outputs (especially scalar output)
//! - Computing gradients for optimization (∇f)
//! - Vector-Jacobian products (v^T J)
//!
//! # Algorithm
//!
//! 1. Forward pass: Compute primal values, save values needed for adjoints
//! 2. Backward pass: Propagate adjoints from output to inputs
//!
//! # Tape
//!
//! Reverse mode requires storing intermediate values ("tape") for the backward pass.
//! We optimize tape size by only saving values actually needed for adjoint computation.

use std::collections::{HashMap, HashSet};

use super::activity::ActivityResult;
use crate::mir::block::{BasicBlock, Terminator};
use crate::mir::function::{FunctionSignature, MirFunction};
use crate::mir::inst::{Instruction, Operation};
use crate::mir::types::{MirType, StructLayout};
use crate::mir::value::{BlockId, ValueId, ValueIdGen};

/// Entry in the reverse-mode tape
#[derive(Clone, Debug)]
pub struct TapeEntry {
    /// Original instruction
    pub primal_inst: Instruction,
    /// Values that need to be saved for adjoint computation
    pub saved_values: Vec<ValueId>,
    /// Instructions for adjoint computation
    pub adjoint_code: Vec<Instruction>,
}

/// Reverse-mode AD transformer
pub struct ReverseModeTransform {
    /// Original function
    primal_func: MirFunction,
    /// Activity analysis result
    activity: ActivityResult,
    /// Mapping from primal values to their adjoint values
    adjoints: HashMap<ValueId, ValueId>,
    /// Tape entries (instructions that need adjoint computation)
    tape: Vec<TapeEntry>,
    /// Values that need to be saved in tape
    values_to_save: HashSet<ValueId>,
    /// Value ID generator
    value_gen: ValueIdGen,
}

impl ReverseModeTransform {
    pub fn new(func: MirFunction, activity: ActivityResult) -> Self {
        // Find max value ID
        let mut max_id = 0u32;
        for block in &func.blocks {
            for param in &block.params {
                max_id = max_id.max(param.value.0 + 1);
            }
            for inst in &block.instructions {
                if let Some(r) = inst.result {
                    max_id = max_id.max(r.0 + 1);
                }
            }
        }

        let mut value_gen = ValueIdGen::new();
        for _ in 0..max_id {
            value_gen.next();
        }

        Self {
            primal_func: func,
            activity,
            adjoints: HashMap::new(),
            tape: Vec::new(),
            values_to_save: HashSet::new(),
            value_gen,
        }
    }

    /// Transform function to compute gradients via reverse mode
    pub fn transform(&mut self) -> ReverseModeResult {
        // Phase 1: Analyze what values need to be saved
        self.analyze_tape_requirements();

        // Phase 2: Generate augmented forward pass
        let forward_func = self.generate_forward_pass();

        // Phase 3: Generate backward pass
        let backward_func = self.generate_backward_pass();

        // Phase 4: Generate tape type
        let tape_type = self.generate_tape_type();

        ReverseModeResult {
            forward_func,
            backward_func,
            tape_type,
            tape_entries: self.tape.clone(),
        }
    }

    /// Analyze which values need to be saved for backward pass
    fn analyze_tape_requirements(&mut self) {
        for block in &self.primal_func.blocks {
            for inst in &block.instructions {
                if let Some(result) = inst.result {
                    if self.activity.active_values.contains(&result) {
                        let needed = self.values_needed_for_adjoint(&inst.op, Some(result));
                        self.values_to_save.extend(needed);
                    }
                }
            }
        }
    }

    /// Determine which values are needed to compute adjoints for an operation
    fn values_needed_for_adjoint(&self, op: &Operation, result: Option<ValueId>) -> Vec<ValueId> {
        match op {
            // Multiplication: d/dx (x * y) requires y, d/dy (x * y) requires x
            Operation::FMul { lhs, rhs } => vec![*lhs, *rhs],

            // Division: needs both operands and result
            // d/dx (x / y) = 1/y, d/dy (x / y) = -x/y²
            Operation::FDiv { lhs, rhs } => {
                let mut v = vec![*lhs, *rhs];
                if let Some(r) = result {
                    v.push(r);
                }
                v
            }

            // sqrt: needs the result (d/dx sqrt(x) = 1/(2*sqrt(x)))
            Operation::Sqrt { operand } => {
                let mut v = vec![*operand];
                if let Some(r) = result {
                    v.push(r);
                }
                v
            }

            // exp: needs the result (d/dx exp(x) = exp(x))
            Operation::Exp { .. } => result.map(|r| vec![r]).unwrap_or_default(),

            // log: needs the operand (d/dx log(x) = 1/x)
            Operation::Log { operand } => vec![*operand],

            // sin: needs operand (to compute cos)
            Operation::Sin { operand } => vec![*operand],

            // cos: needs operand (to compute sin)
            Operation::Cos { operand } => vec![*operand],

            // tanh: needs the result (d/dx tanh(x) = 1 - tanh²(x))
            Operation::Tanh { .. } => result.map(|r| vec![r]).unwrap_or_default(),

            // pow: needs base, exp, and result
            Operation::Pow { base, exp } => {
                let mut v = vec![*base, *exp];
                if let Some(r) = result {
                    v.push(r);
                }
                v
            }

            // Addition/subtraction need nothing extra
            Operation::FAdd { .. } | Operation::FSub { .. } => vec![],

            // FMA needs all operands
            Operation::FMA { a, b, c } => vec![*a, *b, *c],

            // atan2 needs both operands
            Operation::Atan2 { y, x } => vec![*y, *x],

            // asin, acos, atan need operand
            Operation::Asin { operand }
            | Operation::Acos { operand }
            | Operation::Atan { operand } => vec![*operand],

            // Hyperbolic functions
            Operation::Sinh { operand } | Operation::Cosh { operand } => vec![*operand],

            Operation::Asinh { operand }
            | Operation::Acosh { operand }
            | Operation::Atanh { operand } => vec![*operand],

            // Gamma and special functions need operand
            Operation::Gamma { operand }
            | Operation::LogGamma { operand }
            | Operation::Digamma { operand }
            | Operation::Erf { operand }
            | Operation::Erfc { operand } => vec![*operand],

            // Default: nothing extra needed
            _ => vec![],
        }
    }

    /// Generate augmented forward pass that saves values to tape
    fn generate_forward_pass(&mut self) -> MirFunction {
        let tape_type = self.generate_tape_type();

        // New signature: (original_params) -> (original_return, tape)
        let mut new_params = self.primal_func.signature.params.clone();
        let original_ret = self.primal_func.signature.return_type.clone();
        let new_ret = MirType::Tuple {
            elements: vec![original_ret.clone(), tape_type.clone()],
        };

        let signature = FunctionSignature {
            params: new_params,
            param_names: self.primal_func.signature.param_names.clone(),
            return_type: new_ret,
            calling_convention: self.primal_func.signature.calling_convention,
            variadic: false,
        };

        // Clone blocks and add tape storage
        let mut blocks = Vec::new();
        for block in &self.primal_func.blocks {
            let mut new_block = BasicBlock::new(block.id);
            new_block.name = block.name.clone();
            new_block.params = block.params.clone();

            for inst in &block.instructions {
                // Add original instruction
                new_block.push(inst.clone());

                // Add tape storage for values that need saving
                if let Some(result) = inst.result {
                    if self.values_to_save.contains(&result) {
                        // In real implementation: store to tape struct
                        // For now, just record the tape entry
                        self.tape.push(TapeEntry {
                            primal_inst: inst.clone(),
                            saved_values: self.values_needed_for_adjoint(&inst.op, Some(result)),
                            adjoint_code: Vec::new(),
                        });
                    }
                }
            }

            new_block.terminator = block.terminator.clone();
            blocks.push(new_block);
        }

        MirFunction {
            name: format!("{}_forward", self.primal_func.name),
            signature,
            blocks,
            locals: self.primal_func.locals.clone(),
            debug_info: None,
            attributes: self.primal_func.attributes.clone(),
        }
    }

    /// Generate backward pass that computes gradients
    fn generate_backward_pass(&mut self) -> MirFunction {
        let tape_type = self.generate_tape_type();
        let original_ret = self.primal_func.signature.return_type.clone();

        // Signature: (tape, output_adjoint) -> (input_adjoints)
        let params = vec![tape_type.clone(), original_ret.clone()];
        let param_names = vec!["tape".to_string(), "output_adjoint".to_string()];

        // Return type: tuple of adjoints for each active input
        let adjoint_types: Vec<MirType> = self
            .primal_func
            .signature
            .params
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                if let Some(entry) = self.primal_func.entry_block() {
                    entry
                        .params
                        .get(*i)
                        .map(|p| self.activity.active_inputs.contains(&p.value))
                        .unwrap_or(false)
                } else {
                    false
                }
            })
            .map(|(_, ty)| ty.clone())
            .collect();

        let return_type = if adjoint_types.len() == 1 {
            adjoint_types[0].clone()
        } else {
            MirType::Tuple {
                elements: adjoint_types,
            }
        };

        let signature = FunctionSignature {
            params,
            param_names,
            return_type,
            calling_convention: self.primal_func.signature.calling_convention,
            variadic: false,
        };

        // Generate backward pass blocks
        let mut blocks = Vec::new();
        let mut entry_block = BasicBlock::new(BlockId(0));
        entry_block.name = Some("backward_entry".to_string());

        // Initialize adjoints to zero
        let mut adjoint_inits = Vec::new();
        for (i, param_ty) in self.primal_func.signature.params.iter().enumerate() {
            if let Some(entry) = self.primal_func.entry_block() {
                if let Some(param) = entry.params.get(i) {
                    if self.activity.active_inputs.contains(&param.value) {
                        let adj_id = self.value_gen.next();
                        self.adjoints.insert(param.value, adj_id);

                        adjoint_inits.push(Instruction {
                            result: Some(adj_id),
                            op: Operation::ConstFloat {
                                value: 0.0,
                                ty: MirType::F64,
                            },
                            ty: param_ty.clone(),
                            span: None,
                        });
                    }
                }
            }
        }

        entry_block.instructions.extend(adjoint_inits);

        // Process tape entries in reverse order
        // Clone tape to avoid borrow conflict with generate_adjoint_code
        let tape_entries: Vec<_> = self.tape.iter().cloned().collect();
        for tape_entry in tape_entries.iter().rev() {
            let adjoint_code = self.generate_adjoint_code(&tape_entry.primal_inst);
            entry_block.instructions.extend(adjoint_code);
        }

        // Return adjoints
        let return_value = if let Some(entry) = self.primal_func.entry_block() {
            entry
                .params
                .get(0)
                .and_then(|p| self.adjoints.get(&p.value).copied())
        } else {
            None
        };

        entry_block.terminator = Terminator::Return {
            value: return_value,
        };

        blocks.push(entry_block);

        MirFunction {
            name: format!("{}_backward", self.primal_func.name),
            signature,
            blocks,
            locals: Vec::new(),
            debug_info: None,
            attributes: self.primal_func.attributes.clone(),
        }
    }

    /// Generate adjoint code for a single instruction
    fn generate_adjoint_code(&mut self, inst: &Instruction) -> Vec<Instruction> {
        let mut code = Vec::new();

        let result = match inst.result {
            Some(r) => r,
            None => return code,
        };

        // Get adjoint of result (∂L/∂result)
        let result_adj = match self.adjoints.get(&result) {
            Some(adj) => *adj,
            None => {
                // Initialize adjoint if not exists
                let adj = self.value_gen.next();
                self.adjoints.insert(result, adj);
                adj
            }
        };

        match &inst.op {
            // Adjoint of z = x + y:
            //   x̄ += z̄, ȳ += z̄
            Operation::FAdd { lhs, rhs } => {
                // x̄ += z̄
                code.extend(self.accumulate_adjoint(*lhs, result_adj, &inst.ty));
                // ȳ += z̄
                code.extend(self.accumulate_adjoint(*rhs, result_adj, &inst.ty));
            }

            // Adjoint of z = x - y:
            //   x̄ += z̄, ȳ -= z̄
            Operation::FSub { lhs, rhs } => {
                // x̄ += z̄
                code.extend(self.accumulate_adjoint(*lhs, result_adj, &inst.ty));

                // ȳ -= z̄ (i.e., ȳ += -z̄)
                let neg_adj = self.value_gen.next();
                code.push(Instruction {
                    result: Some(neg_adj),
                    op: Operation::FNeg {
                        operand: result_adj,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });
                code.extend(self.accumulate_adjoint(*rhs, neg_adj, &inst.ty));
            }

            // Adjoint of z = x * y:
            //   x̄ += z̄ * y, ȳ += z̄ * x
            Operation::FMul { lhs, rhs } => {
                // x̄ += z̄ * y
                let contrib_lhs = self.value_gen.next();
                code.push(Instruction {
                    result: Some(contrib_lhs),
                    op: Operation::FMul {
                        lhs: result_adj,
                        rhs: *rhs,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });
                code.extend(self.accumulate_adjoint(*lhs, contrib_lhs, &inst.ty));

                // ȳ += z̄ * x
                let contrib_rhs = self.value_gen.next();
                code.push(Instruction {
                    result: Some(contrib_rhs),
                    op: Operation::FMul {
                        lhs: result_adj,
                        rhs: *lhs,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });
                code.extend(self.accumulate_adjoint(*rhs, contrib_rhs, &inst.ty));
            }

            // Adjoint of z = x / y:
            //   x̄ += z̄ / y
            //   ȳ += -z̄ * z / y = -z̄ * x / y²
            Operation::FDiv { lhs, rhs } => {
                // x̄ += z̄ / y
                let contrib_lhs = self.value_gen.next();
                code.push(Instruction {
                    result: Some(contrib_lhs),
                    op: Operation::FDiv {
                        lhs: result_adj,
                        rhs: *rhs,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });
                code.extend(self.accumulate_adjoint(*lhs, contrib_lhs, &inst.ty));

                // ȳ += -z̄ * z / y (where z = result = lhs/rhs)
                let neg_adj = self.value_gen.next();
                code.push(Instruction {
                    result: Some(neg_adj),
                    op: Operation::FNeg {
                        operand: result_adj,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });

                let t1 = self.value_gen.next();
                code.push(Instruction {
                    result: Some(t1),
                    op: Operation::FMul {
                        lhs: neg_adj,
                        rhs: result,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });

                let contrib_rhs = self.value_gen.next();
                code.push(Instruction {
                    result: Some(contrib_rhs),
                    op: Operation::FDiv { lhs: t1, rhs: *rhs },
                    ty: inst.ty.clone(),
                    span: None,
                });
                code.extend(self.accumulate_adjoint(*rhs, contrib_rhs, &inst.ty));
            }

            // Adjoint of z = -x:
            //   x̄ += -z̄
            Operation::FNeg { operand } => {
                let neg_adj = self.value_gen.next();
                code.push(Instruction {
                    result: Some(neg_adj),
                    op: Operation::FNeg {
                        operand: result_adj,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });
                code.extend(self.accumulate_adjoint(*operand, neg_adj, &inst.ty));
            }

            // Adjoint of z = exp(x):
            //   x̄ += z̄ * z (since d/dx exp(x) = exp(x) = z)
            Operation::Exp { operand } => {
                let contrib = self.value_gen.next();
                code.push(Instruction {
                    result: Some(contrib),
                    op: Operation::FMul {
                        lhs: result_adj,
                        rhs: result,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });
                code.extend(self.accumulate_adjoint(*operand, contrib, &inst.ty));
            }

            // Adjoint of z = log(x):
            //   x̄ += z̄ / x
            Operation::Log { operand } => {
                let contrib = self.value_gen.next();
                code.push(Instruction {
                    result: Some(contrib),
                    op: Operation::FDiv {
                        lhs: result_adj,
                        rhs: *operand,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });
                code.extend(self.accumulate_adjoint(*operand, contrib, &inst.ty));
            }

            // Adjoint of z = sqrt(x):
            //   x̄ += z̄ / (2 * z)
            Operation::Sqrt { operand } => {
                // 2 * z
                let two = self.value_gen.next();
                code.push(Instruction {
                    result: Some(two),
                    op: Operation::ConstFloat {
                        value: 2.0,
                        ty: MirType::F64,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });

                let two_z = self.value_gen.next();
                code.push(Instruction {
                    result: Some(two_z),
                    op: Operation::FMul {
                        lhs: two,
                        rhs: result,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });

                let contrib = self.value_gen.next();
                code.push(Instruction {
                    result: Some(contrib),
                    op: Operation::FDiv {
                        lhs: result_adj,
                        rhs: two_z,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });
                code.extend(self.accumulate_adjoint(*operand, contrib, &inst.ty));
            }

            // Adjoint of z = sin(x):
            //   x̄ += z̄ * cos(x)
            Operation::Sin { operand } => {
                let cos_x = self.value_gen.next();
                code.push(Instruction {
                    result: Some(cos_x),
                    op: Operation::Cos { operand: *operand },
                    ty: inst.ty.clone(),
                    span: None,
                });

                let contrib = self.value_gen.next();
                code.push(Instruction {
                    result: Some(contrib),
                    op: Operation::FMul {
                        lhs: result_adj,
                        rhs: cos_x,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });
                code.extend(self.accumulate_adjoint(*operand, contrib, &inst.ty));
            }

            // Adjoint of z = cos(x):
            //   x̄ += -z̄ * sin(x)
            Operation::Cos { operand } => {
                let sin_x = self.value_gen.next();
                code.push(Instruction {
                    result: Some(sin_x),
                    op: Operation::Sin { operand: *operand },
                    ty: inst.ty.clone(),
                    span: None,
                });

                let t1 = self.value_gen.next();
                code.push(Instruction {
                    result: Some(t1),
                    op: Operation::FMul {
                        lhs: result_adj,
                        rhs: sin_x,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });

                let contrib = self.value_gen.next();
                code.push(Instruction {
                    result: Some(contrib),
                    op: Operation::FNeg { operand: t1 },
                    ty: inst.ty.clone(),
                    span: None,
                });
                code.extend(self.accumulate_adjoint(*operand, contrib, &inst.ty));
            }

            // Adjoint of z = tanh(x):
            //   x̄ += z̄ * (1 - z²)
            Operation::Tanh { operand } => {
                // z²
                let z_sq = self.value_gen.next();
                code.push(Instruction {
                    result: Some(z_sq),
                    op: Operation::FMul {
                        lhs: result,
                        rhs: result,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });

                // 1
                let one = self.value_gen.next();
                code.push(Instruction {
                    result: Some(one),
                    op: Operation::ConstFloat {
                        value: 1.0,
                        ty: MirType::F64,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });

                // 1 - z²
                let sech_sq = self.value_gen.next();
                code.push(Instruction {
                    result: Some(sech_sq),
                    op: Operation::FSub {
                        lhs: one,
                        rhs: z_sq,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });

                // z̄ * (1 - z²)
                let contrib = self.value_gen.next();
                code.push(Instruction {
                    result: Some(contrib),
                    op: Operation::FMul {
                        lhs: result_adj,
                        rhs: sech_sq,
                    },
                    ty: inst.ty.clone(),
                    span: None,
                });
                code.extend(self.accumulate_adjoint(*operand, contrib, &inst.ty));
            }

            _ => {
                // Non-differentiable or unhandled operations
            }
        }

        code
    }

    /// Generate code to accumulate adjoint contribution
    fn accumulate_adjoint(
        &mut self,
        value: ValueId,
        contribution: ValueId,
        ty: &MirType,
    ) -> Vec<Instruction> {
        let mut code = Vec::new();

        let current_adj = self.adjoints.get(&value).copied().unwrap_or_else(|| {
            let new_adj = self.value_gen.next();
            self.adjoints.insert(value, new_adj);
            new_adj
        });

        // new_adj = current_adj + contribution
        let new_adj = self.value_gen.next();
        code.push(Instruction {
            result: Some(new_adj),
            op: Operation::FAdd {
                lhs: current_adj,
                rhs: contribution,
            },
            ty: ty.clone(),
            span: None,
        });

        // Update adjoint mapping
        self.adjoints.insert(value, new_adj);

        code
    }

    /// Generate tape structure type
    fn generate_tape_type(&self) -> MirType {
        let fields: Vec<(String, MirType)> = self
            .values_to_save
            .iter()
            .enumerate()
            .map(|(i, id)| (format!("v{}", id.0), self.get_value_type(*id)))
            .collect();

        if fields.is_empty() {
            MirType::Void
        } else {
            let layout = StructLayout::compute(&fields);
            MirType::Struct {
                name: format!("{}_tape", self.primal_func.name),
                fields,
                layout,
            }
        }
    }

    /// Get type of a value (simplified - assumes F64 for now)
    fn get_value_type(&self, _id: ValueId) -> MirType {
        MirType::F64
    }
}

/// Result of reverse-mode AD transformation
#[derive(Clone, Debug)]
pub struct ReverseModeResult {
    /// Augmented forward function
    pub forward_func: MirFunction,
    /// Backward (gradient) function
    pub backward_func: MirFunction,
    /// Tape type for storing intermediate values
    pub tape_type: MirType,
    /// Tape entries (for debugging/analysis)
    pub tape_entries: Vec<TapeEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::ad::ActivityAnalysis;
    use crate::mir::function::FunctionBuilder;

    #[test]
    fn test_reverse_mode_simple() {
        // f(x) = x * x
        let sig = FunctionSignature::new(vec![MirType::F64], MirType::F64);
        let mut builder = FunctionBuilder::new("square", sig);

        let x = builder.param(0).unwrap();
        let xx = builder.push_op(Operation::FMul { lhs: x, rhs: x }, MirType::F64);

        builder.terminate(Terminator::Return { value: Some(xx) });

        let func = builder.build();

        let analysis = ActivityAnalysis::new(&func);
        let activity = analysis.analyze();

        let mut transform = ReverseModeTransform::new(func, activity);
        let result = transform.transform();

        assert_eq!(result.forward_func.name, "square_forward");
        assert_eq!(result.backward_func.name, "square_backward");
    }

    #[test]
    fn test_reverse_mode_chain() {
        // f(x) = exp(log(x))
        let sig = FunctionSignature::new(vec![MirType::F64], MirType::F64);
        let mut builder = FunctionBuilder::new("exp_log", sig);

        let x = builder.param(0).unwrap();
        let log_x = builder.push_op(Operation::Log { operand: x }, MirType::F64);
        let result = builder.push_op(Operation::Exp { operand: log_x }, MirType::F64);

        builder.terminate(Terminator::Return {
            value: Some(result),
        });

        let func = builder.build();

        let analysis = ActivityAnalysis::new(&func);
        let activity = analysis.analyze();

        let mut transform = ReverseModeTransform::new(func, activity);
        let result = transform.transform();

        // Should have tape entries for log and exp
        assert!(!result.tape_entries.is_empty());
    }

    #[test]
    fn test_tape_requirements() {
        // f(x) = x * x * x (needs to save intermediate x*x for gradient)
        let sig = FunctionSignature::new(vec![MirType::F64], MirType::F64);
        let mut builder = FunctionBuilder::new("cube", sig);

        let x = builder.param(0).unwrap();
        let xx = builder.push_op(Operation::FMul { lhs: x, rhs: x }, MirType::F64);
        let xxx = builder.push_op(Operation::FMul { lhs: xx, rhs: x }, MirType::F64);

        builder.terminate(Terminator::Return { value: Some(xxx) });

        let func = builder.build();

        let analysis = ActivityAnalysis::new(&func);
        let activity = analysis.analyze();

        let mut transform = ReverseModeTransform::new(func, activity);
        transform.analyze_tape_requirements();

        // x and xx should be in values_to_save for the second multiplication
        assert!(!transform.values_to_save.is_empty());
    }
}
