//! Floating-Point Determinism Analysis
//!
//! Analyzes MIR code for potential sources of non-determinism in floating-point
//! computations and provides recommendations for achieving reproducibility.

use std::collections::{HashMap, HashSet};

use crate::mir::block::BasicBlock;
use crate::mir::function::MirFunction;
use crate::mir::inst::{Instruction, Operation};
use crate::mir::module::MirModule;
use crate::mir::types::MirType;
use crate::mir::value::ValueId;

/// Level of determinism required for the computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeterminismLevel {
    /// No guarantees - use hardware defaults
    None,
    /// Same result on same hardware with same thread count
    LocalReproducible,
    /// Same result across different x86/ARM CPUs
    CrossPlatform,
    /// Bitwise identical across all platforms including GPU
    BitExact,
}

/// Accuracy requirements for floating-point operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccuracyRequirement {
    /// Use hardware default (fastest)
    Default,
    /// IEEE 754 compliant
    IEEE754,
    /// Maximum 1 ULP error
    OneULP,
    /// Maximum 0.5 ULP error (correctly rounded)
    CorrectlyRounded,
    /// Use arbitrary precision for intermediates
    ExactIntermediate,
}

/// Information about precision loss in a computation
#[derive(Debug, Clone)]
pub struct PrecisionLoss {
    /// Value where precision loss occurs
    pub value: ValueId,
    /// Type of precision loss
    pub kind: PrecisionLossKind,
    /// Estimated magnitude of error (in ULPs)
    pub estimated_ulps: f64,
    /// Suggested mitigation
    pub mitigation: String,
}

/// Kinds of precision loss
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionLossKind {
    /// Catastrophic cancellation (subtracting similar values)
    CatastrophicCancellation,
    /// Accumulation of rounding errors in sums
    AccumulatedRounding,
    /// Loss of precision in large/small number operations
    MagnitudeMismatch,
    /// Non-associative reduction ordering
    ReductionOrdering,
    /// Platform-dependent transcendental
    TranscendentalVariance,
    /// Fused multiply-add vs separate operations
    FMAInconsistency,
}

/// Numerical stability information for a function
#[derive(Debug, Clone)]
pub struct NumericalStabilityInfo {
    /// Condition number estimate (if computable)
    pub condition_number: Option<f64>,
    /// Whether the algorithm is backward stable
    pub backward_stable: bool,
    /// Identified potential instabilities
    pub instabilities: Vec<InstabilityWarning>,
}

/// Warning about potential numerical instability
#[derive(Debug, Clone)]
pub struct InstabilityWarning {
    /// Location in the code
    pub value: ValueId,
    /// Description of the issue
    pub description: String,
    /// Severity (0-10)
    pub severity: u8,
}

/// Result of determinism analysis
#[derive(Debug, Clone)]
pub struct FPAnalysisResult {
    /// Achieved determinism level with current code
    pub current_level: DeterminismLevel,
    /// Required determinism level
    pub required_level: DeterminismLevel,
    /// Non-deterministic operations found
    pub non_deterministic_ops: Vec<NonDeterministicOp>,
    /// Precision loss warnings
    pub precision_warnings: Vec<PrecisionLoss>,
    /// Stability information
    pub stability_info: NumericalStabilityInfo,
    /// Recommended transformations
    pub recommendations: Vec<Recommendation>,
}

/// A non-deterministic operation
#[derive(Debug, Clone)]
pub struct NonDeterministicOp {
    /// The operation
    pub value: ValueId,
    /// Why it's non-deterministic
    pub reason: NonDeterminismReason,
    /// Can it be made deterministic?
    pub fixable: bool,
}

/// Reasons for non-determinism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonDeterminismReason {
    /// Parallel reduction with non-associative operation
    ParallelReduction,
    /// Platform-dependent transcendental function
    Transcendental,
    /// Hardware FMA vs software multiply-add
    FMAVariance,
    /// Denormal handling differences
    DenormalHandling,
    /// Rounding mode differences
    RoundingMode,
    /// Compiler reordering of operations
    ExpressionReordering,
}

/// Recommended transformation
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// Target operation
    pub target: ValueId,
    /// Recommended action
    pub action: RecommendedAction,
    /// Performance impact estimate (negative = slower)
    pub perf_impact: f64,
    /// Accuracy impact (positive = better)
    pub accuracy_impact: f64,
}

/// Recommended actions for determinism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedAction {
    /// Use compensated summation
    UseCompensatedSum,
    /// Use pairwise summation
    UsePairwiseSum,
    /// Use tree reduction
    UseTreeReduction,
    /// Use software transcendental
    UseSoftwareTranscendental,
    /// Disable FMA
    DisableFMA,
    /// Fix operation order
    FixOperationOrder,
    /// Use higher precision intermediate
    UseHigherPrecision,
}

/// Analyzer for floating-point determinism
pub struct DeterminismAnalysis {
    /// Required determinism level
    required_level: DeterminismLevel,
    /// Required accuracy
    required_accuracy: AccuracyRequirement,
    /// Track which values are FP
    fp_values: HashSet<ValueId>,
    /// Track value dependencies
    dependencies: HashMap<ValueId, Vec<ValueId>>,
    /// Found issues
    issues: Vec<NonDeterministicOp>,
    /// Precision warnings
    precision_warnings: Vec<PrecisionLoss>,
}

impl DeterminismAnalysis {
    /// Create a new analysis with specified requirements
    pub fn new(level: DeterminismLevel, accuracy: AccuracyRequirement) -> Self {
        Self {
            required_level: level,
            required_accuracy: accuracy,
            fp_values: HashSet::new(),
            dependencies: HashMap::new(),
            issues: Vec::new(),
            precision_warnings: Vec::new(),
        }
    }

    /// Analyze a module for determinism
    pub fn analyze_module(&mut self, module: &MirModule) -> FPAnalysisResult {
        for func in &module.functions {
            self.analyze_function(func);
        }

        let current_level = self.compute_current_level();
        let recommendations = self.generate_recommendations();
        let stability_info = self.compute_stability_info();

        FPAnalysisResult {
            current_level,
            required_level: self.required_level,
            non_deterministic_ops: self.issues.clone(),
            precision_warnings: self.precision_warnings.clone(),
            stability_info,
            recommendations,
        }
    }

    /// Analyze a function for determinism
    pub fn analyze_function(&mut self, func: &MirFunction) {
        // First pass: identify FP values
        for block in &func.blocks {
            self.identify_fp_values(block);
        }

        // Second pass: check for non-determinism
        for block in &func.blocks {
            self.check_determinism(block);
        }

        // Third pass: check for precision loss
        for block in &func.blocks {
            self.check_precision_loss(block);
        }
    }

    /// Identify floating-point values in a block
    fn identify_fp_values(&mut self, block: &BasicBlock) {
        for inst in &block.instructions {
            if let Some(result) = inst.result {
                if self.is_fp_type(&inst.ty) {
                    self.fp_values.insert(result);
                }

                // Track dependencies
                let deps = self.get_operands(&inst.op);
                self.dependencies.insert(result, deps);
            }
        }
    }

    /// Check for non-deterministic operations
    fn check_determinism(&mut self, block: &BasicBlock) {
        for inst in &block.instructions {
            if let Some(result) = inst.result {
                if !self.fp_values.contains(&result) {
                    continue;
                }

                if let Some(reason) = self.check_op_determinism(&inst.op) {
                    self.issues.push(NonDeterministicOp {
                        value: result,
                        reason,
                        fixable: self.is_fixable(reason),
                    });
                }
            }
        }
    }

    /// Check if an operation is non-deterministic
    fn check_op_determinism(&self, op: &Operation) -> Option<NonDeterminismReason> {
        match op {
            // Transcendental functions vary across platforms
            Operation::Sin { .. }
            | Operation::Cos { .. }
            | Operation::Tan { .. }
            | Operation::Exp { .. }
            | Operation::Log { .. }
            | Operation::Pow { .. }
            | Operation::Sqrt { .. } => {
                if self.required_level >= DeterminismLevel::CrossPlatform {
                    Some(NonDeterminismReason::Transcendental)
                } else {
                    None
                }
            }

            // FMA can vary
            Operation::FMA { .. } => {
                if self.required_level >= DeterminismLevel::CrossPlatform {
                    Some(NonDeterminismReason::FMAVariance)
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    /// Check for precision loss
    fn check_precision_loss(&mut self, block: &BasicBlock) {
        for inst in &block.instructions {
            if let Some(result) = inst.result {
                if !self.fp_values.contains(&result) {
                    continue;
                }

                // Check for catastrophic cancellation in subtraction
                if let Operation::FSub { lhs, rhs } = &inst.op {
                    // If lhs and rhs come from similar computations, warn
                    if self.might_cancel(*lhs, *rhs) {
                        self.precision_warnings.push(PrecisionLoss {
                            value: result,
                            kind: PrecisionLossKind::CatastrophicCancellation,
                            estimated_ulps: 1e6, // Conservative estimate
                            mitigation:
                                "Consider reformulating to avoid subtraction of similar values"
                                    .to_string(),
                        });
                    }
                }

                // Check for accumulated rounding in sums
                if let Operation::FAdd { .. } = &inst.op {
                    if self.is_accumulator(result, block) {
                        self.precision_warnings.push(PrecisionLoss {
                            value: result,
                            kind: PrecisionLossKind::AccumulatedRounding,
                            estimated_ulps: 100.0, // Depends on iteration count
                            mitigation: "Use compensated summation (Kahan or Neumaier)".to_string(),
                        });
                    }
                }
            }
        }
    }

    /// Check if two values might cancel
    fn might_cancel(&self, _lhs: ValueId, _rhs: ValueId) -> bool {
        // Simplified check - in practice would do dataflow analysis
        false
    }

    /// Check if a value is an accumulator (used in a loop)
    fn is_accumulator(&self, _value: ValueId, _block: &BasicBlock) -> bool {
        // Simplified - would check if value feeds back to itself
        false
    }

    /// Compute current determinism level based on found issues
    fn compute_current_level(&self) -> DeterminismLevel {
        if self.issues.is_empty() {
            return DeterminismLevel::BitExact;
        }

        let has_parallel = self
            .issues
            .iter()
            .any(|i| i.reason == NonDeterminismReason::ParallelReduction);
        let has_transcendental = self
            .issues
            .iter()
            .any(|i| i.reason == NonDeterminismReason::Transcendental);
        let has_fma = self
            .issues
            .iter()
            .any(|i| i.reason == NonDeterminismReason::FMAVariance);

        if has_parallel {
            DeterminismLevel::None
        } else if has_transcendental || has_fma {
            DeterminismLevel::LocalReproducible
        } else {
            DeterminismLevel::CrossPlatform
        }
    }

    /// Generate recommendations based on issues
    fn generate_recommendations(&self) -> Vec<Recommendation> {
        let mut recs = Vec::new();

        for issue in &self.issues {
            let (action, perf, acc) = match issue.reason {
                NonDeterminismReason::ParallelReduction => {
                    (RecommendedAction::UseTreeReduction, -0.2, 0.5)
                }
                NonDeterminismReason::Transcendental => {
                    (RecommendedAction::UseSoftwareTranscendental, -0.5, 0.1)
                }
                NonDeterminismReason::FMAVariance => (RecommendedAction::DisableFMA, -0.1, 0.0),
                NonDeterminismReason::ExpressionReordering => {
                    (RecommendedAction::FixOperationOrder, 0.0, 0.0)
                }
                _ => continue,
            };

            recs.push(Recommendation {
                target: issue.value,
                action,
                perf_impact: perf,
                accuracy_impact: acc,
            });
        }

        // Add recommendations for precision warnings
        for warning in &self.precision_warnings {
            let (action, perf, acc) = match warning.kind {
                PrecisionLossKind::AccumulatedRounding => {
                    (RecommendedAction::UseCompensatedSum, -0.3, 1.0)
                }
                PrecisionLossKind::MagnitudeMismatch => {
                    (RecommendedAction::UseHigherPrecision, -0.2, 0.8)
                }
                _ => continue,
            };

            recs.push(Recommendation {
                target: warning.value,
                action,
                perf_impact: perf,
                accuracy_impact: acc,
            });
        }

        recs
    }

    /// Compute stability information
    fn compute_stability_info(&self) -> NumericalStabilityInfo {
        let instabilities: Vec<InstabilityWarning> = self
            .precision_warnings
            .iter()
            .filter(|w| w.estimated_ulps > 100.0)
            .map(|w| InstabilityWarning {
                value: w.value,
                description: w.mitigation.clone(),
                severity: (w.estimated_ulps.log10() as u8).min(10),
            })
            .collect();

        NumericalStabilityInfo {
            condition_number: None, // Would require symbolic analysis
            backward_stable: instabilities.is_empty(),
            instabilities,
        }
    }

    /// Check if a non-determinism issue is fixable
    fn is_fixable(&self, reason: NonDeterminismReason) -> bool {
        matches!(
            reason,
            NonDeterminismReason::ParallelReduction
                | NonDeterminismReason::Transcendental
                | NonDeterminismReason::FMAVariance
                | NonDeterminismReason::ExpressionReordering
        )
    }

    /// Check if a type is floating-point
    fn is_fp_type(&self, ty: &MirType) -> bool {
        matches!(ty, MirType::F32 | MirType::F64)
    }

    /// Get operands of an operation
    fn get_operands(&self, op: &Operation) -> Vec<ValueId> {
        match op {
            Operation::FAdd { lhs, rhs }
            | Operation::FSub { lhs, rhs }
            | Operation::FMul { lhs, rhs }
            | Operation::FDiv { lhs, rhs } => vec![*lhs, *rhs],
            Operation::FNeg { operand }
            | Operation::Sin { operand }
            | Operation::Cos { operand }
            | Operation::Tan { operand }
            | Operation::Exp { operand }
            | Operation::Log { operand }
            | Operation::Sqrt { operand } => vec![*operand],
            Operation::FMA { a, b, c } => vec![*a, *b, *c],
            Operation::Pow { base, exp } => vec![*base, *exp],
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinism_levels() {
        assert!(DeterminismLevel::BitExact > DeterminismLevel::CrossPlatform);
        assert!(DeterminismLevel::CrossPlatform > DeterminismLevel::LocalReproducible);
        assert!(DeterminismLevel::LocalReproducible > DeterminismLevel::None);
    }
}
