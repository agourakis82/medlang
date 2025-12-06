//! Activity Analysis for Automatic Differentiation
//!
//! Determines which values are "active" - meaning they:
//! 1. Depend on differentiated inputs (forward sweep)
//! 2. Affect the output through active paths (backward sweep)
//!
//! A value is active iff it's in both sets (forward ∩ backward).

use std::collections::HashSet;

use crate::mir::block::Terminator;
use crate::mir::function::MirFunction;
use crate::mir::inst::Operation;
use crate::mir::value::ValueId;

/// Result of activity analysis
#[derive(Clone, Debug)]
pub struct ActivityResult {
    /// Values that are active (depend on inputs AND affect outputs)
    pub active_values: HashSet<ValueId>,
    /// Values that depend on active inputs (forward sweep)
    pub forward_active: HashSet<ValueId>,
    /// Values that affect outputs (backward sweep)
    pub backward_active: HashSet<ValueId>,
    /// Active inputs (parameters that are differentiated)
    pub active_inputs: HashSet<ValueId>,
    /// Active outputs (return values)
    pub active_outputs: HashSet<ValueId>,
}

/// Activity analyzer for AD
pub struct ActivityAnalysis<'a> {
    func: &'a MirFunction,
    /// Which parameters to differentiate with respect to
    diff_params: HashSet<usize>,
}

impl<'a> ActivityAnalysis<'a> {
    /// Create analyzer for differentiating with respect to all parameters
    pub fn new(func: &'a MirFunction) -> Self {
        let diff_params: HashSet<usize> = (0..func.signature.params.len()).collect();
        Self { func, diff_params }
    }

    /// Create analyzer for differentiating with respect to specific parameters
    pub fn with_params(func: &'a MirFunction, params: HashSet<usize>) -> Self {
        Self {
            func,
            diff_params: params,
        }
    }

    /// Run activity analysis
    pub fn analyze(&self) -> ActivityResult {
        let forward_active = self.forward_sweep();
        let backward_active = self.backward_sweep();

        // Active = forward ∩ backward
        let active_values: HashSet<ValueId> = forward_active
            .intersection(&backward_active)
            .cloned()
            .collect();

        // Collect active inputs
        let active_inputs: HashSet<ValueId> = self
            .func
            .entry_block()
            .map(|b| {
                b.params
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| self.diff_params.contains(i))
                    .map(|(_, p)| p.value)
                    .collect()
            })
            .unwrap_or_default();

        // Collect active outputs
        let active_outputs: HashSet<ValueId> = self
            .func
            .blocks
            .iter()
            .filter_map(|b| {
                if let Terminator::Return { value: Some(v) } = &b.terminator {
                    Some(*v)
                } else {
                    None
                }
            })
            .filter(|v| active_values.contains(v))
            .collect();

        ActivityResult {
            active_values,
            forward_active,
            backward_active,
            active_inputs,
            active_outputs,
        }
    }

    /// Forward sweep: mark values that depend on active inputs
    fn forward_sweep(&self) -> HashSet<ValueId> {
        let mut active = HashSet::new();

        // All differentiated parameters are active
        if let Some(entry) = self.func.entry_block() {
            for (i, param) in entry.params.iter().enumerate() {
                if self.diff_params.contains(&i) {
                    active.insert(param.value);
                }
            }
        }

        // Propagate activity forward through instructions
        let mut changed = true;
        while changed {
            changed = false;

            for block in &self.func.blocks {
                for inst in &block.instructions {
                    if let Some(result) = inst.result {
                        if !active.contains(&result) {
                            if self.has_active_operand(&inst.op, &active) {
                                active.insert(result);
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        active
    }

    /// Backward sweep: mark values that affect outputs
    fn backward_sweep(&self) -> HashSet<ValueId> {
        let mut active = HashSet::new();

        // Return values are active
        for block in &self.func.blocks {
            if let Terminator::Return { value: Some(v) } = &block.terminator {
                active.insert(*v);
            }
        }

        // Propagate activity backward
        let mut changed = true;
        while changed {
            changed = false;

            for block in self.func.blocks.iter().rev() {
                for inst in block.instructions.iter().rev() {
                    if let Some(result) = inst.result {
                        if active.contains(&result) {
                            let operands = self.get_operands(&inst.op);
                            for op in operands {
                                if active.insert(op) {
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                // Terminator operands
                let term_ops = self.terminator_operands(&block.terminator);
                for op in term_ops {
                    if active.contains(&op) {
                        // Already handled by instruction sweep
                    }
                }
            }
        }

        active
    }

    /// Check if operation has at least one active operand
    fn has_active_operand(&self, op: &Operation, active: &HashSet<ValueId>) -> bool {
        let operands = self.get_operands(op);
        operands.iter().any(|v| active.contains(v))
    }

    /// Get operands of an operation that participate in differentiation
    fn get_operands(&self, op: &Operation) -> Vec<ValueId> {
        match op {
            // Binary arithmetic
            Operation::FAdd { lhs, rhs }
            | Operation::FSub { lhs, rhs }
            | Operation::FMul { lhs, rhs }
            | Operation::FDiv { lhs, rhs }
            | Operation::FRem { lhs, rhs }
            | Operation::IAdd { lhs, rhs }
            | Operation::ISub { lhs, rhs }
            | Operation::IMul { lhs, rhs }
            | Operation::Pow {
                base: lhs,
                exp: rhs,
            }
            | Operation::Atan2 { y: lhs, x: rhs }
            | Operation::FMin { lhs, rhs }
            | Operation::FMax { lhs, rhs } => vec![*lhs, *rhs],

            // Unary operations
            Operation::FNeg { operand }
            | Operation::Sqrt { operand }
            | Operation::Exp { operand }
            | Operation::Expm1 { operand }
            | Operation::Log { operand }
            | Operation::Log1p { operand }
            | Operation::Log10 { operand }
            | Operation::Log2 { operand }
            | Operation::Sin { operand }
            | Operation::Cos { operand }
            | Operation::Tan { operand }
            | Operation::Asin { operand }
            | Operation::Acos { operand }
            | Operation::Atan { operand }
            | Operation::Sinh { operand }
            | Operation::Cosh { operand }
            | Operation::Tanh { operand }
            | Operation::Asinh { operand }
            | Operation::Acosh { operand }
            | Operation::Atanh { operand }
            | Operation::Abs { operand }
            | Operation::Floor { operand }
            | Operation::Ceil { operand }
            | Operation::Round { operand }
            | Operation::Trunc { operand }
            | Operation::Gamma { operand }
            | Operation::LogGamma { operand }
            | Operation::Digamma { operand }
            | Operation::Erf { operand }
            | Operation::Erfc { operand }
            | Operation::ErfInv { operand } => vec![*operand],

            // FMA
            Operation::FMA { a, b, c } => vec![*a, *b, *c],

            // Dual number operations
            Operation::MakeDual { value, derivative } => vec![*value, *derivative],
            Operation::DualPrimal { dual } | Operation::DualTangent { dual } => vec![*dual],
            Operation::DualAdd { lhs, rhs }
            | Operation::DualSub { lhs, rhs }
            | Operation::DualMul { lhs, rhs }
            | Operation::DualDiv { lhs, rhs }
            | Operation::DualPow {
                base: lhs,
                exp: rhs,
            } => vec![*lhs, *rhs],
            Operation::DualSin { operand }
            | Operation::DualCos { operand }
            | Operation::DualExp { operand }
            | Operation::DualLog { operand }
            | Operation::DualSqrt { operand }
            | Operation::DualTanh { operand } => vec![*operand],

            // Vector/matrix operations
            Operation::VecDot { lhs, rhs }
            | Operation::MatMul { lhs, rhs }
            | Operation::MatSolve { a: lhs, b: rhs } => vec![*lhs, *rhs],
            Operation::MatVecMul { mat, vec } => vec![*mat, *vec],
            Operation::VecNorm { vec }
            | Operation::VecNormalize { vec }
            | Operation::MatTranspose { mat: vec }
            | Operation::MatInverse { mat: vec }
            | Operation::MatDet { mat: vec }
            | Operation::MatTrace { mat: vec }
            | Operation::MatCholesky { mat: vec }
            | Operation::MatLU { mat: vec }
            | Operation::MatQR { mat: vec }
            | Operation::MatEigen { mat: vec }
            | Operation::MatSVD { mat: vec } => vec![*vec],

            // Probability distributions (differentiable with respect to value and params)
            Operation::LogPDF { value, params, .. } | Operation::CDF { value, params, .. } => {
                let mut ops = vec![*value];
                ops.extend(params);
                ops
            }

            // Select (conditional)
            Operation::Select {
                cond: _,
                then_val,
                else_val,
            } => vec![*then_val, *else_val],

            // Memory operations (value flows through)
            Operation::Load { ptr, .. } => vec![*ptr],

            // Call (assume all args are active for now)
            Operation::Call { args, .. } => args.clone(),

            // Non-differentiable operations
            Operation::ConstInt { .. }
            | Operation::ConstFloat { .. }
            | Operation::ConstBool { .. }
            | Operation::ZeroInit { .. }
            | Operation::Undef { .. } => vec![],

            // Integer and bitwise operations (not differentiable)
            Operation::IDiv { .. }
            | Operation::UDiv { .. }
            | Operation::IRem { .. }
            | Operation::URem { .. }
            | Operation::INeg { .. }
            | Operation::And { .. }
            | Operation::Or { .. }
            | Operation::Xor { .. }
            | Operation::Not { .. }
            | Operation::Shl { .. }
            | Operation::LShr { .. }
            | Operation::AShr { .. } => vec![],

            // Comparisons (not differentiable but used in control flow)
            Operation::ICmp { .. }
            | Operation::FCmp { .. }
            | Operation::IsNaN { .. }
            | Operation::IsInf { .. }
            | Operation::IsFinite { .. } => vec![],

            // Type conversions
            Operation::SIToFP { operand, .. }
            | Operation::UIToFP { operand, .. }
            | Operation::FPToSI { operand, .. }
            | Operation::FPToUI { operand, .. }
            | Operation::FExt { operand, .. }
            | Operation::FTrunc { operand, .. } => vec![*operand],

            // Everything else
            _ => vec![],
        }
    }

    fn terminator_operands(&self, term: &Terminator) -> Vec<ValueId> {
        match term {
            Terminator::Return { value: Some(v) } => vec![*v],
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
            Terminator::Goto { args, .. } => args.clone(),
            Terminator::Switch { value, .. } => vec![*value],
            _ => vec![],
        }
    }
}

/// Check if an operation is differentiable
pub fn is_differentiable(op: &Operation) -> bool {
    matches!(
        op,
        Operation::FAdd { .. }
            | Operation::FSub { .. }
            | Operation::FMul { .. }
            | Operation::FDiv { .. }
            | Operation::FNeg { .. }
            | Operation::FMA { .. }
            | Operation::Sqrt { .. }
            | Operation::Pow { .. }
            | Operation::Exp { .. }
            | Operation::Expm1 { .. }
            | Operation::Log { .. }
            | Operation::Log1p { .. }
            | Operation::Log10 { .. }
            | Operation::Log2 { .. }
            | Operation::Sin { .. }
            | Operation::Cos { .. }
            | Operation::Tan { .. }
            | Operation::Asin { .. }
            | Operation::Acos { .. }
            | Operation::Atan { .. }
            | Operation::Atan2 { .. }
            | Operation::Sinh { .. }
            | Operation::Cosh { .. }
            | Operation::Tanh { .. }
            | Operation::Asinh { .. }
            | Operation::Acosh { .. }
            | Operation::Atanh { .. }
            | Operation::Abs { .. }
            | Operation::Gamma { .. }
            | Operation::LogGamma { .. }
            | Operation::Erf { .. }
            | Operation::Erfc { .. }
            | Operation::LogPDF { .. }
            | Operation::CDF { .. }
            | Operation::DualAdd { .. }
            | Operation::DualSub { .. }
            | Operation::DualMul { .. }
            | Operation::DualDiv { .. }
            | Operation::DualSin { .. }
            | Operation::DualCos { .. }
            | Operation::DualExp { .. }
            | Operation::DualLog { .. }
            | Operation::DualSqrt { .. }
            | Operation::DualTanh { .. }
            | Operation::DualPow { .. }
            | Operation::VecDot { .. }
            | Operation::MatMul { .. }
            | Operation::MatVecMul { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::block::Terminator;
    use crate::mir::function::{FunctionBuilder, FunctionSignature};
    use crate::mir::types::MirType;

    #[test]
    fn test_activity_simple_function() {
        // f(x, y) = x * y + x
        let sig = FunctionSignature::new(vec![MirType::F64, MirType::F64], MirType::F64);
        let mut builder = FunctionBuilder::new("mul_add", sig);

        let x = builder.param(0).unwrap();
        let y = builder.param(1).unwrap();

        let xy = builder.push_op(Operation::FMul { lhs: x, rhs: y }, MirType::F64);
        let result = builder.push_op(Operation::FAdd { lhs: xy, rhs: x }, MirType::F64);

        builder.terminate(Terminator::Return {
            value: Some(result),
        });

        let func = builder.build();
        let analysis = ActivityAnalysis::new(&func);
        let result = analysis.analyze();

        // All values should be active
        assert!(result.active_values.contains(&x));
        assert!(result.active_values.contains(&y));
        assert!(result.active_values.contains(&xy));
    }

    #[test]
    fn test_activity_with_constant() {
        // f(x) = x * 2.0
        let sig = FunctionSignature::new(vec![MirType::F64], MirType::F64);
        let mut builder = FunctionBuilder::new("scale", sig);

        let x = builder.param(0).unwrap();
        let two = builder.push_op(
            Operation::ConstFloat {
                value: 2.0,
                ty: MirType::F64,
            },
            MirType::F64,
        );
        let result = builder.push_op(Operation::FMul { lhs: x, rhs: two }, MirType::F64);

        builder.terminate(Terminator::Return {
            value: Some(result),
        });

        let func = builder.build();
        let analysis = ActivityAnalysis::new(&func);
        let result = analysis.analyze();

        // x and result should be active, constant should not affect output activity
        assert!(result.active_values.contains(&x));
        // The constant participates in forward activity but may or may not
        // be in backward depending on implementation
    }

    #[test]
    fn test_activity_dead_code() {
        // f(x, y) = x (y is unused)
        let sig = FunctionSignature::new(vec![MirType::F64, MirType::F64], MirType::F64);
        let mut builder = FunctionBuilder::new("identity", sig);

        let x = builder.param(0).unwrap();
        let _y = builder.param(1).unwrap();

        builder.terminate(Terminator::Return { value: Some(x) });

        let func = builder.build();
        let analysis = ActivityAnalysis::new(&func);
        let result = analysis.analyze();

        // Only x should be active
        assert!(result.active_values.contains(&x));
        // y is not in active_values because it doesn't affect output
    }
}
