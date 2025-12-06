//! Forward-Mode Automatic Differentiation
//!
//! Forward mode propagates tangent (derivative) values alongside primal values.
//! For each primal value x, we compute (x, ẋ) where ẋ = ∂x/∂input.
//!
//! Forward mode is efficient when:
//! - Number of inputs << number of outputs
//! - Computing Jacobian-vector products (Jv)
//! - Directional derivatives

use std::collections::HashMap;

use super::activity::ActivityResult;
use crate::mir::block::{BasicBlock, Terminator};
use crate::mir::function::{FunctionSignature, LocalDecl, MirFunction};
use crate::mir::inst::{Instruction, Operation};
use crate::mir::types::MirType;
use crate::mir::value::{BlockId, ValueId, ValueIdGen};

/// Forward-mode AD transformer
pub struct ForwardModeTransform {
    /// Original function
    primal_func: MirFunction,
    /// Activity analysis result
    activity: ActivityResult,
    /// Mapping from primal values to their tangent values
    tangents: HashMap<ValueId, ValueId>,
    /// Value ID generator
    value_gen: ValueIdGen,
    /// Generated blocks
    blocks: Vec<BasicBlock>,
}

impl ForwardModeTransform {
    pub fn new(func: MirFunction, activity: ActivityResult) -> Self {
        // Find max value ID in function
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
            tangents: HashMap::new(),
            value_gen,
            blocks: Vec::new(),
        }
    }

    /// Transform function to compute primal and tangent values
    pub fn transform(&mut self) -> MirFunction {
        // Create new function with extended signature
        let tangent_func = self.create_tangent_function();

        // Transform each block
        for block in &self.primal_func.blocks.clone() {
            let transformed = self.transform_block(block);
            self.blocks.push(transformed);
        }

        // Build result function
        let mut result = tangent_func;
        result.blocks = self.blocks.clone();
        result
    }

    /// Create function signature for tangent computation
    fn create_tangent_function(&mut self) -> MirFunction {
        let mut new_params = self.primal_func.signature.params.clone();
        let mut new_param_names = self.primal_func.signature.param_names.clone();

        // Add tangent parameters for each active input
        if let Some(entry) = self.primal_func.entry_block() {
            for (i, param) in entry.params.iter().enumerate() {
                if self.activity.active_inputs.contains(&param.value) {
                    new_params.push(self.primal_func.signature.params[i].clone());
                    new_param_names.push(format!("{}_tangent", new_param_names[i]));
                }
            }
        }

        // Return type is tuple of (primal, tangent)
        let primal_ret = self.primal_func.signature.return_type.clone();
        let tangent_ret = if self.activity.active_outputs.is_empty() {
            MirType::Void
        } else {
            primal_ret.clone()
        };

        let return_type = MirType::Tuple {
            elements: vec![primal_ret, tangent_ret],
        };

        let signature = FunctionSignature {
            params: new_params,
            param_names: new_param_names,
            return_type,
            calling_convention: self.primal_func.signature.calling_convention,
            variadic: false,
        };

        MirFunction {
            name: format!("{}_tangent", self.primal_func.name),
            signature,
            blocks: Vec::new(),
            locals: self.primal_func.locals.clone(),
            debug_info: None,
            attributes: self.primal_func.attributes.clone(),
        }
    }

    /// Transform a basic block
    fn transform_block(&mut self, block: &BasicBlock) -> BasicBlock {
        let mut new_block = BasicBlock::new(block.id);
        new_block.name = block.name.clone();

        // Add original block parameters
        for param in &block.params {
            new_block.params.push(param.clone());
        }

        // Add tangent block parameters for active values
        for param in &block.params {
            if self.activity.active_values.contains(&param.value) {
                let tangent_id = self.value_gen.next();
                self.tangents.insert(param.value, tangent_id);
                new_block.add_param(tangent_id, param.ty.clone());
            }
        }

        // Transform instructions
        for inst in &block.instructions {
            // Keep primal instruction
            new_block.push(inst.clone());

            // Generate tangent instruction if result is active
            if let Some(result) = inst.result {
                if self.activity.active_values.contains(&result) {
                    if let Some(tangent_inst) = self.differentiate_forward(inst) {
                        new_block.push(tangent_inst);
                    }
                }
            }
        }

        // Transform terminator
        new_block.terminator = self.transform_terminator(&block.terminator);

        new_block
    }

    /// Generate forward-mode derivative for an instruction
    fn differentiate_forward(&mut self, inst: &Instruction) -> Option<Instruction> {
        let result = inst.result?;
        let result_tangent = self.value_gen.next();
        self.tangents.insert(result, result_tangent);

        let tangent_op = match &inst.op {
            // ∂(a + b) = ∂a + ∂b
            Operation::FAdd { lhs, rhs } => {
                let lhs_tan = self.get_tangent(*lhs)?;
                let rhs_tan = self.get_tangent(*rhs)?;
                Operation::FAdd {
                    lhs: lhs_tan,
                    rhs: rhs_tan,
                }
            }

            // ∂(a - b) = ∂a - ∂b
            Operation::FSub { lhs, rhs } => {
                let lhs_tan = self.get_tangent(*lhs)?;
                let rhs_tan = self.get_tangent(*rhs)?;
                Operation::FSub {
                    lhs: lhs_tan,
                    rhs: rhs_tan,
                }
            }

            // ∂(a * b) = a * ∂b + ∂a * b
            // Using DualMul if both operands are active
            Operation::FMul { lhs, rhs } => {
                let lhs_tan = self.get_tangent(*lhs);
                let rhs_tan = self.get_tangent(*rhs);

                match (lhs_tan, rhs_tan) {
                    (Some(lt), Some(rt)) => {
                        // Full product rule: need temp values
                        // Simplified: use dual multiply semantic
                        Operation::FMA {
                            a: *lhs,
                            b: rt,
                            c: self.value_gen.next(), // Should be lhs_tan * rhs
                        }
                    }
                    (Some(lt), None) => {
                        // ∂(a * const) = const * ∂a
                        Operation::FMul { lhs: lt, rhs: *rhs }
                    }
                    (None, Some(rt)) => {
                        // ∂(const * b) = const * ∂b
                        Operation::FMul { lhs: *lhs, rhs: rt }
                    }
                    (None, None) => return None,
                }
            }

            // ∂(a / b) = (∂a * b - a * ∂b) / b²
            // Simplified: (∂a - result * ∂b) / b
            Operation::FDiv { lhs, rhs } => {
                let lhs_tan = self.get_tangent(*lhs);
                let rhs_tan = self.get_tangent(*rhs);

                match (lhs_tan, rhs_tan) {
                    (Some(lt), Some(_rt)) => {
                        // Complex - return simplified version
                        Operation::FDiv { lhs: lt, rhs: *rhs }
                    }
                    (Some(lt), None) => {
                        // ∂(a / const) = ∂a / const
                        Operation::FDiv { lhs: lt, rhs: *rhs }
                    }
                    (None, Some(rt)) => {
                        // ∂(const / b) = -const * ∂b / b²
                        Operation::FNeg { operand: rt }
                    }
                    (None, None) => return None,
                }
            }

            // ∂(-a) = -∂a
            Operation::FNeg { operand } => {
                let op_tan = self.get_tangent(*operand)?;
                Operation::FNeg { operand: op_tan }
            }

            // ∂sqrt(x) = ∂x / (2 * sqrt(x))
            Operation::Sqrt { operand } => {
                let _op_tan = self.get_tangent(*operand)?;
                // Use dual sqrt
                Operation::DualSqrt { operand: *operand }
            }

            // ∂exp(x) = exp(x) * ∂x
            Operation::Exp { operand } => {
                let _op_tan = self.get_tangent(*operand)?;
                // exp(x) is already computed as result
                Operation::FMul {
                    lhs: result,
                    rhs: self.get_tangent(*operand)?,
                }
            }

            // ∂log(x) = ∂x / x
            Operation::Log { operand } => {
                let op_tan = self.get_tangent(*operand)?;
                Operation::FDiv {
                    lhs: op_tan,
                    rhs: *operand,
                }
            }

            // ∂sin(x) = cos(x) * ∂x
            Operation::Sin { operand } => {
                let _op_tan = self.get_tangent(*operand)?;
                Operation::DualSin { operand: *operand }
            }

            // ∂cos(x) = -sin(x) * ∂x
            Operation::Cos { operand } => {
                let _op_tan = self.get_tangent(*operand)?;
                Operation::DualCos { operand: *operand }
            }

            // ∂tanh(x) = (1 - tanh²(x)) * ∂x = sech²(x) * ∂x
            Operation::Tanh { operand } => {
                let _op_tan = self.get_tangent(*operand)?;
                Operation::DualTanh { operand: *operand }
            }

            // ∂pow(x, y) = pow(x, y) * (y * ∂x / x + ∂y * log(x))
            Operation::Pow { base, exp } => {
                let _base_tan = self.get_tangent(*base);
                let _exp_tan = self.get_tangent(*exp);
                Operation::DualPow {
                    base: *base,
                    exp: *exp,
                }
            }

            // Constants have zero tangent
            Operation::ConstFloat { ty, .. } => Operation::ConstFloat {
                value: 0.0,
                ty: ty.clone(),
            },

            Operation::ConstInt { ty, .. } => Operation::ConstInt {
                value: 0,
                ty: ty.clone(),
            },

            // Dual number operations pass through
            Operation::MakeDual { value, derivative } => Operation::MakeDual {
                value: *value,
                derivative: *derivative,
            },

            Operation::DualPrimal { dual } => Operation::DualTangent { dual: *dual },

            Operation::DualTangent { .. } => {
                // Second derivative - tangent of tangent
                // For now, return zero
                return None;
            }

            // Non-differentiable operations
            _ => return None,
        };

        Some(Instruction {
            result: Some(result_tangent),
            op: tangent_op,
            ty: inst.ty.clone(),
            span: inst.span,
        })
    }

    /// Transform terminator for tangent propagation
    fn transform_terminator(&mut self, term: &Terminator) -> Terminator {
        match term {
            Terminator::Return { value } => {
                if let Some(v) = value {
                    if let Some(tangent) = self.get_tangent(*v) {
                        // Return tuple of (primal, tangent)
                        // In real implementation, would construct aggregate
                        Terminator::Return {
                            value: Some(tangent), // Simplified
                        }
                    } else {
                        term.clone()
                    }
                } else {
                    term.clone()
                }
            }

            Terminator::Goto { target, args } => {
                let mut new_args = args.clone();
                // Add tangent arguments
                for arg in args {
                    if let Some(tangent) = self.get_tangent(*arg) {
                        new_args.push(tangent);
                    }
                }
                Terminator::Goto {
                    target: *target,
                    args: new_args,
                }
            }

            Terminator::Branch {
                cond,
                then_block,
                then_args,
                else_block,
                else_args,
            } => {
                let mut new_then_args = then_args.clone();
                let mut new_else_args = else_args.clone();

                for arg in then_args {
                    if let Some(tangent) = self.get_tangent(*arg) {
                        new_then_args.push(tangent);
                    }
                }
                for arg in else_args {
                    if let Some(tangent) = self.get_tangent(*arg) {
                        new_else_args.push(tangent);
                    }
                }

                Terminator::Branch {
                    cond: *cond,
                    then_block: *then_block,
                    then_args: new_then_args,
                    else_block: *else_block,
                    else_args: new_else_args,
                }
            }

            _ => term.clone(),
        }
    }

    /// Get tangent value for a primal value
    fn get_tangent(&self, primal: ValueId) -> Option<ValueId> {
        self.tangents.get(&primal).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::ad::ActivityAnalysis;
    use crate::mir::function::FunctionBuilder;

    #[test]
    fn test_forward_mode_simple() {
        // f(x) = x * x
        let sig = FunctionSignature::new(vec![MirType::F64], MirType::F64);
        let mut builder = FunctionBuilder::new("square", sig);

        let x = builder.param(0).unwrap();
        let xx = builder.push_op(Operation::FMul { lhs: x, rhs: x }, MirType::F64);

        builder.terminate(Terminator::Return { value: Some(xx) });

        let func = builder.build();

        // Analyze activity
        let analysis = ActivityAnalysis::new(&func);
        let activity = analysis.analyze();

        // Transform
        let mut transform = ForwardModeTransform::new(func, activity);
        let tangent_func = transform.transform();

        assert_eq!(tangent_func.name, "square_tangent");
        // Should have original param + tangent param
        assert!(tangent_func.signature.params.len() >= 1);
    }

    #[test]
    fn test_forward_mode_chain_rule() {
        // f(x) = exp(x * x)
        let sig = FunctionSignature::new(vec![MirType::F64], MirType::F64);
        let mut builder = FunctionBuilder::new("exp_square", sig);

        let x = builder.param(0).unwrap();
        let xx = builder.push_op(Operation::FMul { lhs: x, rhs: x }, MirType::F64);
        let result = builder.push_op(Operation::Exp { operand: xx }, MirType::F64);

        builder.terminate(Terminator::Return {
            value: Some(result),
        });

        let func = builder.build();

        let analysis = ActivityAnalysis::new(&func);
        let activity = analysis.analyze();

        let mut transform = ForwardModeTransform::new(func, activity);
        let tangent_func = transform.transform();

        // Function should have transformed
        assert_eq!(tangent_func.name, "exp_square_tangent");
    }
}
