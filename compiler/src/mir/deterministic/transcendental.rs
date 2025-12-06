//! Deterministic Transcendental Functions
//!
//! Software implementations of transcendental functions (exp, log, sin, cos, etc.)
//! that produce bitwise identical results across all platforms.

use crate::mir::inst::{Callee, Instruction, Operation};
use crate::mir::types::MirType;
use crate::mir::value::{ValueId, ValueIdGen};

/// Accuracy level for transcendental implementations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathAccuracy {
    /// Use hardware (non-deterministic but fast)
    Hardware,
    /// 1 ULP accuracy (good for most uses)
    OneULP,
    /// 0.5 ULP (correctly rounded)
    CorrectlyRounded,
    /// Arbitrary precision (slow but exact)
    Arbitrary,
}

/// Which implementation to use for transcendentals
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscendentalImpl {
    /// Hardware intrinsic
    Hardware,
    /// MPFR-based implementation
    MPFR,
    /// CRLibm (correctly rounded libm)
    CRLibm,
    /// Custom polynomial approximation
    Polynomial,
    /// Table-based with interpolation
    TableLookup,
}

/// Configuration for deterministic math operations
#[derive(Debug, Clone)]
pub struct DeterministicMath {
    /// Implementation to use
    pub implementation: TranscendentalImpl,
    /// Required accuracy
    pub accuracy: MathAccuracy,
    /// Whether to handle special cases (NaN, Inf) consistently
    pub consistent_special_cases: bool,
    /// Maximum polynomial degree for approximations
    pub max_poly_degree: usize,
}

impl Default for DeterministicMath {
    fn default() -> Self {
        Self {
            implementation: TranscendentalImpl::Polynomial,
            accuracy: MathAccuracy::OneULP,
            consistent_special_cases: true,
            max_poly_degree: 13,
        }
    }
}

impl DeterministicMath {
    /// Create with correctly rounded accuracy
    pub fn correctly_rounded() -> Self {
        Self {
            implementation: TranscendentalImpl::CRLibm,
            accuracy: MathAccuracy::CorrectlyRounded,
            consistent_special_cases: true,
            max_poly_degree: 15,
        }
    }

    /// Create with hardware (fast, non-deterministic)
    pub fn hardware() -> Self {
        Self {
            implementation: TranscendentalImpl::Hardware,
            accuracy: MathAccuracy::Hardware,
            consistent_special_cases: false,
            max_poly_degree: 0,
        }
    }
}

/// Generator for software transcendental implementations
pub struct SoftwareTranscendental {
    /// Configuration
    config: DeterministicMath,
}

impl SoftwareTranscendental {
    /// Create a new generator
    pub fn new(config: DeterministicMath) -> Self {
        Self { config }
    }

    /// Transform an operation to use software implementation
    pub fn transform_op(&self, inst: &Instruction, id_gen: &mut ValueIdGen) -> Vec<Instruction> {
        if self.config.implementation == TranscendentalImpl::Hardware {
            return vec![inst.clone()];
        }

        match &inst.op {
            Operation::Exp { operand } => {
                self.generate_exp(*operand, inst.result, &inst.ty, id_gen)
            }
            Operation::Log { operand } => {
                self.generate_log(*operand, inst.result, &inst.ty, id_gen)
            }
            Operation::Sin { operand } => {
                self.generate_sin(*operand, inst.result, &inst.ty, id_gen)
            }
            Operation::Cos { operand } => {
                self.generate_cos(*operand, inst.result, &inst.ty, id_gen)
            }
            Operation::Tan { operand } => {
                self.generate_tan(*operand, inst.result, &inst.ty, id_gen)
            }
            Operation::Sqrt { operand } => self.generate_sqrt(*operand, inst.result, &inst.ty),
            Operation::Pow { base, exp } => {
                self.generate_pow(*base, *exp, inst.result, &inst.ty, id_gen)
            }
            _ => vec![inst.clone()],
        }
    }

    /// Generate exp(x) using polynomial approximation
    fn generate_exp(
        &self,
        x: ValueId,
        result: Option<ValueId>,
        ty: &MirType,
        id_gen: &mut ValueIdGen,
    ) -> Vec<Instruction> {
        match self.config.implementation {
            TranscendentalImpl::CRLibm => {
                self.generate_external_call("cr_exp", "crlibm", x, result, ty, id_gen)
            }
            TranscendentalImpl::MPFR => {
                self.generate_external_call("mpfr_exp", "mpfr", x, result, ty, id_gen)
            }
            TranscendentalImpl::Polynomial => self.generate_exp_polynomial(x, result, ty, id_gen),
            _ => {
                let out = result.unwrap_or_else(|| id_gen.next());
                vec![Instruction::new(Operation::Exp { operand: x }, ty.clone()).with_result(out)]
            }
        }
    }

    /// Generate exp using polynomial approximation
    fn generate_exp_polynomial(
        &self,
        x: ValueId,
        result: Option<ValueId>,
        ty: &MirType,
        id_gen: &mut ValueIdGen,
    ) -> Vec<Instruction> {
        let mut insts = Vec::new();
        let out = result.unwrap_or_else(|| id_gen.next());

        // Constants for range reduction
        let log2e = id_gen.next();
        insts.push(
            Instruction::new(
                Operation::ConstFloat {
                    value: std::f64::consts::LOG2_E,
                    ty: ty.clone(),
                },
                ty.clone(),
            )
            .with_result(log2e),
        );

        let ln2 = id_gen.next();
        insts.push(
            Instruction::new(
                Operation::ConstFloat {
                    value: std::f64::consts::LN_2,
                    ty: ty.clone(),
                },
                ty.clone(),
            )
            .with_result(ln2),
        );

        // k = round(x * log2(e))
        let x_scaled = id_gen.next();
        insts.push(
            Instruction::new(Operation::FMul { lhs: x, rhs: log2e }, ty.clone())
                .with_result(x_scaled),
        );

        let k_float = id_gen.next();
        insts.push(
            Instruction::new(Operation::Round { operand: x_scaled }, ty.clone())
                .with_result(k_float),
        );

        // r = x - k * ln(2)
        let k_ln2 = id_gen.next();
        insts.push(
            Instruction::new(
                Operation::FMul {
                    lhs: k_float,
                    rhs: ln2,
                },
                ty.clone(),
            )
            .with_result(k_ln2),
        );

        let r = id_gen.next();
        insts.push(
            Instruction::new(Operation::FSub { lhs: x, rhs: k_ln2 }, ty.clone()).with_result(r),
        );

        // Polynomial coefficients
        let coeffs = [
            1.0,
            1.0,
            0.5,
            0.16666666666666666,
            0.041666666666666664,
            0.008333333333333333,
            0.001388888888888889,
        ];

        // Horner's method
        let mut acc = id_gen.next();
        insts.push(
            Instruction::new(
                Operation::ConstFloat {
                    value: coeffs[coeffs.len() - 1],
                    ty: ty.clone(),
                },
                ty.clone(),
            )
            .with_result(acc),
        );

        for &coeff in coeffs[..coeffs.len() - 1].iter().rev() {
            let prod = id_gen.next();
            insts.push(
                Instruction::new(Operation::FMul { lhs: acc, rhs: r }, ty.clone())
                    .with_result(prod),
            );

            let c = id_gen.next();
            insts.push(
                Instruction::new(
                    Operation::ConstFloat {
                        value: coeff,
                        ty: ty.clone(),
                    },
                    ty.clone(),
                )
                .with_result(c),
            );

            let new_acc = id_gen.next();
            insts.push(
                Instruction::new(Operation::FAdd { lhs: prod, rhs: c }, ty.clone())
                    .with_result(new_acc),
            );
            acc = new_acc;
        }

        // Scale by 2^k using ldexp intrinsic
        insts.push(
            Instruction::new(
                Operation::Call {
                    callee: Callee::Intrinsic("ldexp".to_string()),
                    args: vec![acc, k_float],
                    ret_ty: ty.clone(),
                },
                ty.clone(),
            )
            .with_result(out),
        );

        insts
    }

    fn generate_log(
        &self,
        x: ValueId,
        result: Option<ValueId>,
        ty: &MirType,
        id_gen: &mut ValueIdGen,
    ) -> Vec<Instruction> {
        match self.config.implementation {
            TranscendentalImpl::CRLibm => {
                self.generate_external_call("cr_log", "crlibm", x, result, ty, id_gen)
            }
            TranscendentalImpl::MPFR => {
                self.generate_external_call("mpfr_log", "mpfr", x, result, ty, id_gen)
            }
            _ => self.generate_intrinsic_call("deterministic_log", x, result, ty, id_gen),
        }
    }

    fn generate_sin(
        &self,
        x: ValueId,
        result: Option<ValueId>,
        ty: &MirType,
        id_gen: &mut ValueIdGen,
    ) -> Vec<Instruction> {
        match self.config.implementation {
            TranscendentalImpl::CRLibm => {
                self.generate_external_call("cr_sin", "crlibm", x, result, ty, id_gen)
            }
            TranscendentalImpl::MPFR => {
                self.generate_external_call("mpfr_sin", "mpfr", x, result, ty, id_gen)
            }
            _ => self.generate_intrinsic_call("deterministic_sin", x, result, ty, id_gen),
        }
    }

    fn generate_cos(
        &self,
        x: ValueId,
        result: Option<ValueId>,
        ty: &MirType,
        id_gen: &mut ValueIdGen,
    ) -> Vec<Instruction> {
        match self.config.implementation {
            TranscendentalImpl::CRLibm => {
                self.generate_external_call("cr_cos", "crlibm", x, result, ty, id_gen)
            }
            TranscendentalImpl::MPFR => {
                self.generate_external_call("mpfr_cos", "mpfr", x, result, ty, id_gen)
            }
            _ => self.generate_intrinsic_call("deterministic_cos", x, result, ty, id_gen),
        }
    }

    fn generate_tan(
        &self,
        x: ValueId,
        result: Option<ValueId>,
        ty: &MirType,
        id_gen: &mut ValueIdGen,
    ) -> Vec<Instruction> {
        match self.config.implementation {
            TranscendentalImpl::CRLibm => {
                self.generate_external_call("cr_tan", "crlibm", x, result, ty, id_gen)
            }
            TranscendentalImpl::MPFR => {
                self.generate_external_call("mpfr_tan", "mpfr", x, result, ty, id_gen)
            }
            _ => self.generate_intrinsic_call("deterministic_tan", x, result, ty, id_gen),
        }
    }

    fn generate_sqrt(&self, x: ValueId, result: Option<ValueId>, ty: &MirType) -> Vec<Instruction> {
        let mut inst = Instruction::new(Operation::Sqrt { operand: x }, ty.clone());
        if let Some(r) = result {
            inst = inst.with_result(r);
        }
        vec![inst]
    }

    fn generate_pow(
        &self,
        base: ValueId,
        exp: ValueId,
        result: Option<ValueId>,
        ty: &MirType,
        id_gen: &mut ValueIdGen,
    ) -> Vec<Instruction> {
        let out = result.unwrap_or_else(|| id_gen.next());
        vec![Instruction::new(
            Operation::Call {
                callee: Callee::Intrinsic("deterministic_pow".to_string()),
                args: vec![base, exp],
                ret_ty: ty.clone(),
            },
            ty.clone(),
        )
        .with_result(out)]
    }

    fn generate_external_call(
        &self,
        name: &str,
        lib: &str,
        arg: ValueId,
        result: Option<ValueId>,
        ty: &MirType,
        id_gen: &mut ValueIdGen,
    ) -> Vec<Instruction> {
        let out = result.unwrap_or_else(|| id_gen.next());
        vec![Instruction::new(
            Operation::Call {
                callee: Callee::External {
                    name: name.to_string(),
                    lib: lib.to_string(),
                },
                args: vec![arg],
                ret_ty: ty.clone(),
            },
            ty.clone(),
        )
        .with_result(out)]
    }

    fn generate_intrinsic_call(
        &self,
        name: &str,
        arg: ValueId,
        result: Option<ValueId>,
        ty: &MirType,
        id_gen: &mut ValueIdGen,
    ) -> Vec<Instruction> {
        let out = result.unwrap_or_else(|| id_gen.next());
        vec![Instruction::new(
            Operation::Call {
                callee: Callee::Intrinsic(name.to_string()),
                args: vec![arg],
                ret_ty: ty.clone(),
            },
            ty.clone(),
        )
        .with_result(out)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_accuracy_levels() {
        let hw = DeterministicMath::hardware();
        assert_eq!(hw.accuracy, MathAccuracy::Hardware);

        let cr = DeterministicMath::correctly_rounded();
        assert_eq!(cr.accuracy, MathAccuracy::CorrectlyRounded);
    }

    #[test]
    fn test_exp_polynomial_generation() {
        let config = DeterministicMath::default();
        let trans = SoftwareTranscendental::new(config);
        let mut id_gen = ValueIdGen::new();

        let x = id_gen.next();
        let insts = trans.generate_exp_polynomial(x, None, &MirType::F64, &mut id_gen);
        assert!(!insts.is_empty());
    }
}
