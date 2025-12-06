//! AD Transformation Entry Point
//!
//! This module provides the main interface for applying automatic differentiation
//! transformations to MIR functions. It handles mode selection, checkpointing
//! strategies, and produces optimized derivative code.
//!
//! # Usage
//!
//! ```text
//! let transformer = ADTransformer::new(ADConfig::default());
//! let result = transformer.transform(&function, ADMode::Reverse)?;
//! ```

use std::collections::{HashMap, HashSet};

use super::activity::ActivityAnalysis;
use super::forward::ForwardModeTransform;
use super::reverse::{ReverseModeResult, ReverseModeTransform};
use crate::mir::function::MirFunction;
use crate::mir::module::MirModule;
use crate::mir::types::MirType;
use crate::mir::value::ValueId;

/// AD computation mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ADMode {
    /// Forward mode: propagate tangents alongside primals
    /// Efficient for few inputs, many outputs
    Forward,

    /// Reverse mode: backpropagate adjoints from outputs to inputs
    /// Efficient for many inputs, few outputs (gradients)
    Reverse,

    /// Mixed mode: automatically select based on input/output ratio
    /// Uses heuristics to choose optimal mode
    Mixed,

    /// Forward-over-reverse: for Hessian computation
    /// Computes Hessian-vector products efficiently
    ForwardOverReverse,

    /// Reverse-over-forward: alternative Hessian computation
    ReverseOverForward,
}

impl ADMode {
    /// Select optimal mode based on input/output dimensions
    pub fn auto_select(num_inputs: usize, num_outputs: usize) -> Self {
        // Reverse mode is O(outputs) forward passes
        // Forward mode is O(inputs) computations
        // Choose based on which is smaller
        if num_outputs <= num_inputs {
            ADMode::Reverse
        } else {
            ADMode::Forward
        }
    }
}

/// Checkpointing strategy for reverse mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointStrategy {
    /// Store all intermediate values (highest memory, fastest)
    StoreAll,

    /// Recompute all values during backward pass (lowest memory, slowest)
    RecomputeAll,

    /// Binomial checkpointing (Griewank's treeverse/revolve algorithm)
    /// Optimal trade-off between memory and compute
    Binomial {
        /// Number of checkpoints to use
        checkpoints: usize,
    },

    /// Periodic checkpointing every N steps
    Periodic {
        /// Save checkpoint every N operations
        interval: usize,
    },

    /// Automatic selection based on available memory
    Auto {
        /// Maximum memory budget in bytes
        memory_budget: usize,
    },
}

impl Default for CheckpointStrategy {
    fn default() -> Self {
        CheckpointStrategy::StoreAll
    }
}

/// Configuration for AD transformation
#[derive(Clone, Debug)]
pub struct ADConfig {
    /// Computation mode
    pub mode: ADMode,

    /// Checkpointing strategy for reverse mode
    pub checkpoint_strategy: CheckpointStrategy,

    /// Enable sparsity detection and exploitation
    pub exploit_sparsity: bool,

    /// Enable activity analysis to skip non-active computations
    pub activity_analysis: bool,

    /// Generate code for multiple output adjoints (vectorized reverse mode)
    pub vectorized_adjoints: bool,

    /// Maximum tape size before switching to checkpointing
    pub max_tape_size: Option<usize>,

    /// Inline threshold for small functions
    pub inline_threshold: usize,

    /// Enable fusion of adjoint operations
    pub fuse_adjoints: bool,
}

impl Default for ADConfig {
    fn default() -> Self {
        Self {
            mode: ADMode::Mixed,
            checkpoint_strategy: CheckpointStrategy::default(),
            exploit_sparsity: true,
            activity_analysis: true,
            vectorized_adjoints: false,
            max_tape_size: None,
            inline_threshold: 50,
            fuse_adjoints: true,
        }
    }
}

/// Errors that can occur during AD transformation
#[derive(Clone, Debug, PartialEq)]
pub enum ADError {
    /// Function contains non-differentiable operations
    NonDifferentiable {
        operation: String,
        location: Option<(u32, u32)>,
    },

    /// Discontinuity detected (e.g., if/else on active values)
    Discontinuity {
        description: String,
        location: Option<(u32, u32)>,
    },

    /// No active outputs (nothing to differentiate)
    NoActiveOutputs,

    /// Circular dependency in computation graph
    CircularDependency { values: Vec<ValueId> },

    /// Checkpointing failed
    CheckpointError { reason: String },

    /// Unsupported control flow for reverse mode
    UnsupportedControlFlow { description: String },

    /// Type mismatch in AD transformation
    TypeMismatch { expected: MirType, found: MirType },

    /// Internal compiler error
    InternalError { message: String },
}

impl std::fmt::Display for ADError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ADError::NonDifferentiable {
                operation,
                location,
            } => {
                write!(f, "Non-differentiable operation: {}", operation)?;
                if let Some((line, col)) = location {
                    write!(f, " at {}:{}", line, col)?;
                }
                Ok(())
            }
            ADError::Discontinuity {
                description,
                location,
            } => {
                write!(f, "Discontinuity: {}", description)?;
                if let Some((line, col)) = location {
                    write!(f, " at {}:{}", line, col)?;
                }
                Ok(())
            }
            ADError::NoActiveOutputs => {
                write!(f, "No active outputs to differentiate")
            }
            ADError::CircularDependency { values } => {
                write!(f, "Circular dependency involving values: {:?}", values)
            }
            ADError::CheckpointError { reason } => {
                write!(f, "Checkpointing error: {}", reason)
            }
            ADError::UnsupportedControlFlow { description } => {
                write!(f, "Unsupported control flow: {}", description)
            }
            ADError::TypeMismatch { expected, found } => {
                write!(
                    f,
                    "Type mismatch: expected {:?}, found {:?}",
                    expected, found
                )
            }
            ADError::InternalError { message } => {
                write!(f, "Internal error: {}", message)
            }
        }
    }
}

impl std::error::Error for ADError {}

/// Result of AD transformation
#[derive(Clone, Debug)]
pub enum ADTransformResult {
    /// Forward mode result: single function computing primal + tangent
    Forward {
        /// Transformed function
        tangent_func: MirFunction,
    },

    /// Reverse mode result: separate forward and backward passes
    Reverse {
        /// Forward pass with tape recording
        forward_func: MirFunction,
        /// Backward pass computing gradients
        backward_func: MirFunction,
        /// Type of the tape structure
        tape_type: MirType,
    },

    /// Mixed mode result
    Mixed {
        /// Functions using forward mode
        forward_parts: Vec<MirFunction>,
        /// Functions using reverse mode
        reverse_parts: Vec<(MirFunction, MirFunction, MirType)>,
    },

    /// Hessian computation result
    Hessian {
        /// Function computing Hessian-vector product
        hvp_func: MirFunction,
        /// Optional explicit Hessian computation
        hessian_func: Option<MirFunction>,
    },
}

/// Main AD transformer
pub struct ADTransformer {
    /// Configuration
    config: ADConfig,

    /// Cache of transformed functions
    cache: HashMap<String, ADTransformResult>,

    /// Statistics about transformations
    stats: ADStats,
}

/// Statistics collected during AD transformation
#[derive(Clone, Debug, Default)]
pub struct ADStats {
    /// Number of functions transformed
    pub functions_transformed: usize,

    /// Number of instructions in original code
    pub original_instructions: usize,

    /// Number of instructions in transformed code
    pub transformed_instructions: usize,

    /// Number of values saved to tape
    pub tape_values: usize,

    /// Number of active values identified
    pub active_values: usize,

    /// Number of inactive values pruned
    pub pruned_values: usize,

    /// Time spent in activity analysis (microseconds)
    pub activity_analysis_time_us: u64,

    /// Time spent in transformation (microseconds)
    pub transform_time_us: u64,
}

impl ADTransformer {
    /// Create a new AD transformer with the given configuration
    pub fn new(config: ADConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            stats: ADStats::default(),
        }
    }

    /// Create a transformer with default configuration
    pub fn default_transformer() -> Self {
        Self::new(ADConfig::default())
    }

    /// Transform a function using the configured AD mode
    pub fn transform(
        &mut self,
        func: &MirFunction,
        mode: ADMode,
    ) -> Result<ADTransformResult, ADError> {
        // Check cache
        let cache_key = format!("{}_{:?}", func.name, mode);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Perform activity analysis
        let analysis = ActivityAnalysis::new(func);
        let activity = analysis.analyze();

        self.stats.active_values = activity.active_values.len();

        // Check for active outputs
        if activity.active_outputs.is_empty() {
            return Err(ADError::NoActiveOutputs);
        }

        // Determine actual mode
        let actual_mode = match mode {
            ADMode::Mixed => {
                ADMode::auto_select(activity.active_inputs.len(), activity.active_outputs.len())
            }
            other => other,
        };

        // Transform based on mode
        let result = match actual_mode {
            ADMode::Forward => {
                let mut transform = ForwardModeTransform::new(func.clone(), activity);
                let tangent_func = transform.transform();

                self.stats.functions_transformed += 1;

                ADTransformResult::Forward { tangent_func }
            }

            ADMode::Reverse => {
                let mut transform = ReverseModeTransform::new(func.clone(), activity);
                let reverse_result = transform.transform();

                self.stats.functions_transformed += 1;
                self.stats.tape_values = reverse_result.tape_entries.len();

                ADTransformResult::Reverse {
                    forward_func: reverse_result.forward_func,
                    backward_func: reverse_result.backward_func,
                    tape_type: reverse_result.tape_type,
                }
            }

            ADMode::ForwardOverReverse => {
                // First apply reverse mode
                let mut reverse_transform =
                    ReverseModeTransform::new(func.clone(), activity.clone());
                let reverse_result = reverse_transform.transform();

                // Then apply forward mode to the backward function
                let backward_analysis = ActivityAnalysis::new(&reverse_result.backward_func);
                let backward_activity = backward_analysis.analyze();

                let mut forward_transform = ForwardModeTransform::new(
                    reverse_result.backward_func.clone(),
                    backward_activity,
                );
                let hvp_func = forward_transform.transform();

                self.stats.functions_transformed += 2;

                ADTransformResult::Hessian {
                    hvp_func,
                    hessian_func: None,
                }
            }

            ADMode::ReverseOverForward => {
                // First apply forward mode
                let mut forward_transform =
                    ForwardModeTransform::new(func.clone(), activity.clone());
                let tangent_func = forward_transform.transform();

                // Then apply reverse mode to the tangent function
                let tangent_analysis = ActivityAnalysis::new(&tangent_func);
                let tangent_activity = tangent_analysis.analyze();

                let mut reverse_transform =
                    ReverseModeTransform::new(tangent_func.clone(), tangent_activity);
                let reverse_result = reverse_transform.transform();

                self.stats.functions_transformed += 2;

                ADTransformResult::Hessian {
                    hvp_func: reverse_result.backward_func,
                    hessian_func: None,
                }
            }

            ADMode::Mixed => {
                // This should have been resolved above
                unreachable!()
            }
        };

        // Cache the result
        self.cache.insert(cache_key, result.clone());

        Ok(result)
    }

    /// Transform all differentiable functions in a module
    pub fn transform_module(&mut self, module: &MirModule) -> Result<ADModuleResult, ADError> {
        let mut results = HashMap::new();
        let mut errors = Vec::new();

        for func in &module.functions {
            // Check if function is marked as differentiable
            // For now, transform all functions (attribute checking can be added later)
            {
                match self.transform(func, self.config.mode) {
                    Ok(result) => {
                        results.insert(func.name.clone(), result);
                    }
                    Err(e) => {
                        errors.push((func.name.clone(), e));
                    }
                }
            }
        }

        Ok(ADModuleResult {
            transformed_functions: results,
            errors,
            stats: self.stats.clone(),
        })
    }

    /// Get transformation statistics
    pub fn stats(&self) -> &ADStats {
        &self.stats
    }

    /// Clear the transformation cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Validate that a function can be differentiated
    pub fn validate(&self, func: &MirFunction) -> Result<ADValidation, Vec<ADError>> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check for non-differentiable operations
        for block in &func.blocks {
            for inst in &block.instructions {
                if !self.is_differentiable(&inst.op) {
                    errors.push(ADError::NonDifferentiable {
                        operation: format!("{:?}", inst.op),
                        location: inst.span.map(|s| (s.start, s.end)),
                    });
                }
            }

            // Check for problematic control flow
            match &block.terminator {
                crate::mir::block::Terminator::Branch { .. } => {
                    // Branches on active values can introduce discontinuities
                    warnings.push("Branch may introduce discontinuity".to_string());
                }
                crate::mir::block::Terminator::Switch { .. } => {
                    warnings.push("Switch may introduce discontinuity".to_string());
                }
                _ => {}
            }
        }

        if errors.is_empty() {
            Ok(ADValidation {
                is_valid: true,
                warnings,
                estimated_tape_size: self.estimate_tape_size(func),
                recommended_mode: self.recommend_mode(func),
            })
        } else {
            Err(errors)
        }
    }

    /// Check if an operation is differentiable
    fn is_differentiable(&self, op: &crate::mir::inst::Operation) -> bool {
        use crate::mir::inst::Operation::*;

        match op {
            // Arithmetic - differentiable
            FAdd { .. } | FSub { .. } | FMul { .. } | FDiv { .. } | FNeg { .. } => true,

            // Transcendental - differentiable
            Sqrt { .. } | Exp { .. } | Log { .. } | Pow { .. } => true,
            Sin { .. } | Cos { .. } | Tan { .. } | Tanh { .. } => true,
            Asin { .. } | Acos { .. } | Atan { .. } | Atan2 { .. } => true,
            Sinh { .. } | Cosh { .. } => true,
            Asinh { .. } | Acosh { .. } | Atanh { .. } => true,

            // Special functions - differentiable
            Gamma { .. } | LogGamma { .. } | Digamma { .. } => true,
            Erf { .. } | Erfc { .. } => true,

            // FMA - differentiable
            FMA { .. } => true,

            // Constants - differentiable (derivative is zero)
            ConstFloat { .. } | ConstInt { .. } => true,

            // Dual number operations - by definition differentiable
            MakeDual { .. } | DualPrimal { .. } | DualTangent { .. } => true,
            DualAdd { .. } | DualSub { .. } | DualMul { .. } | DualDiv { .. } => true,
            DualSqrt { .. } | DualExp { .. } | DualLog { .. } | DualPow { .. } => true,
            DualSin { .. } | DualCos { .. } | DualTanh { .. } => true,

            // Memory operations - not differentiable but pass-through
            Load { .. } | Store { .. } => true,
            Alloca { .. } => true,
            Memcpy { .. } | Memset { .. } => true,

            // Aggregate operations - pass-through
            ExtractField { .. } | InsertField { .. } => true,
            ExtractElement { .. } | InsertElement { .. } => true,

            // Integer operations - not differentiable (but may be inactive)
            IAdd { .. } | ISub { .. } | IMul { .. } | IDiv { .. } | IRem { .. } => true,
            INeg { .. } | Abs { .. } => true,

            // Comparison operations - not differentiable
            ICmp { .. } | FCmp { .. } => true,

            // Bitwise - not differentiable
            And { .. } | Or { .. } | Xor { .. } | Not { .. } => true,
            Shl { .. } => true,

            // Conversions - some are differentiable
            SIToFP { .. } | FPToSI { .. } => true,
            Trunc { .. } | SExt { .. } | ZExt { .. } => true,

            // Calls - depends on callee
            Call { .. } => true, // Assume differentiable, check callee later

            // Absolute value - has subgradient
            Abs { .. } => true,

            // Min/max - have subgradients
            FMin { .. } | FMax { .. } => true,

            // Probability distributions - differentiable
            Sample { .. } => false, // Sampling is not differentiable in the usual sense

            // Select is differentiable (piecewise)
            Select { .. } => true,

            // Everything else
            _ => true,
        }
    }

    /// Estimate tape size for reverse mode
    fn estimate_tape_size(&self, func: &MirFunction) -> usize {
        let mut size = 0;

        for block in &func.blocks {
            for inst in &block.instructions {
                // Estimate based on operation type
                let inst_size = match &inst.op {
                    crate::mir::inst::Operation::FMul { .. } => 16, // Save both operands
                    crate::mir::inst::Operation::FDiv { .. } => 24, // Save operands + result
                    crate::mir::inst::Operation::Sqrt { .. } => 16, // Save result
                    crate::mir::inst::Operation::Exp { .. } => 8,   // Save result
                    crate::mir::inst::Operation::Log { .. } => 8,   // Save operand
                    crate::mir::inst::Operation::Pow { .. } => 32,  // Save all
                    crate::mir::inst::Operation::Sin { .. }
                    | crate::mir::inst::Operation::Cos { .. } => 8, // Save operand
                    _ => 0,
                };
                size += inst_size;
            }
        }

        size
    }

    /// Recommend AD mode for a function
    fn recommend_mode(&self, func: &MirFunction) -> ADMode {
        let analysis = ActivityAnalysis::new(func);
        let activity = analysis.analyze();

        ADMode::auto_select(activity.active_inputs.len(), activity.active_outputs.len())
    }
}

/// Result of validating a function for AD
#[derive(Clone, Debug)]
pub struct ADValidation {
    /// Whether the function can be differentiated
    pub is_valid: bool,
    /// Warnings about potential issues
    pub warnings: Vec<String>,
    /// Estimated tape size in bytes
    pub estimated_tape_size: usize,
    /// Recommended AD mode
    pub recommended_mode: ADMode,
}

/// Result of transforming a module
#[derive(Clone, Debug)]
pub struct ADModuleResult {
    /// Successfully transformed functions
    pub transformed_functions: HashMap<String, ADTransformResult>,
    /// Functions that failed to transform
    pub errors: Vec<(String, ADError)>,
    /// Overall statistics
    pub stats: ADStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::block::Terminator;
    use crate::mir::function::{FunctionBuilder, FunctionSignature};
    use crate::mir::inst::Operation;

    #[test]
    fn test_mode_auto_select() {
        // Many inputs, one output -> reverse
        assert_eq!(ADMode::auto_select(100, 1), ADMode::Reverse);

        // One input, many outputs -> forward
        assert_eq!(ADMode::auto_select(1, 100), ADMode::Forward);

        // Equal -> reverse (typical for neural networks)
        assert_eq!(ADMode::auto_select(10, 10), ADMode::Reverse);
    }

    #[test]
    fn test_transform_forward() {
        let sig = FunctionSignature::new(vec![MirType::F64], MirType::F64);
        let mut builder = FunctionBuilder::new("square", sig);

        let x = builder.param(0).unwrap();
        let xx = builder.push_op(Operation::FMul { lhs: x, rhs: x }, MirType::F64);

        builder.terminate(Terminator::Return { value: Some(xx) });

        let func = builder.build();

        let mut transformer = ADTransformer::new(ADConfig::default());
        let result = transformer.transform(&func, ADMode::Forward).unwrap();

        match result {
            ADTransformResult::Forward { tangent_func } => {
                assert_eq!(tangent_func.name, "square_tangent");
            }
            _ => panic!("Expected forward mode result"),
        }
    }

    #[test]
    fn test_transform_reverse() {
        let sig = FunctionSignature::new(vec![MirType::F64], MirType::F64);
        let mut builder = FunctionBuilder::new("cube", sig);

        let x = builder.param(0).unwrap();
        let xx = builder.push_op(Operation::FMul { lhs: x, rhs: x }, MirType::F64);
        let xxx = builder.push_op(Operation::FMul { lhs: xx, rhs: x }, MirType::F64);

        builder.terminate(Terminator::Return { value: Some(xxx) });

        let func = builder.build();

        let mut transformer = ADTransformer::new(ADConfig::default());
        let result = transformer.transform(&func, ADMode::Reverse).unwrap();

        match result {
            ADTransformResult::Reverse {
                forward_func,
                backward_func,
                ..
            } => {
                assert_eq!(forward_func.name, "cube_forward");
                assert_eq!(backward_func.name, "cube_backward");
            }
            _ => panic!("Expected reverse mode result"),
        }
    }

    #[test]
    fn test_validation() {
        let sig = FunctionSignature::new(vec![MirType::F64], MirType::F64);
        let mut builder = FunctionBuilder::new("exp_func", sig);

        let x = builder.param(0).unwrap();
        let result = builder.push_op(Operation::Exp { operand: x }, MirType::F64);

        builder.terminate(Terminator::Return {
            value: Some(result),
        });

        let func = builder.build();

        let transformer = ADTransformer::new(ADConfig::default());
        let validation = transformer.validate(&func).unwrap();

        assert!(validation.is_valid);
        assert_eq!(validation.recommended_mode, ADMode::Reverse);
    }

    #[test]
    fn test_error_no_active_outputs() {
        // Function with no active computation
        let sig = FunctionSignature::new(vec![MirType::F64], MirType::I64);
        let mut builder = FunctionBuilder::new("const_func", sig);

        let _x = builder.param(0).unwrap();
        let c = builder.push_op(
            Operation::ConstInt {
                value: 42,
                ty: MirType::I64,
            },
            MirType::I64,
        );

        builder.terminate(Terminator::Return { value: Some(c) });

        let func = builder.build();

        let mut transformer = ADTransformer::new(ADConfig::default());
        let result = transformer.transform(&func, ADMode::Reverse);

        match result {
            Err(ADError::NoActiveOutputs) => {}
            _ => panic!("Expected NoActiveOutputs error"),
        }
    }

    #[test]
    fn test_hessian_computation() {
        let sig = FunctionSignature::new(vec![MirType::F64], MirType::F64);
        let mut builder = FunctionBuilder::new("quadratic", sig);

        let x = builder.param(0).unwrap();
        let xx = builder.push_op(Operation::FMul { lhs: x, rhs: x }, MirType::F64);

        builder.terminate(Terminator::Return { value: Some(xx) });

        let func = builder.build();

        let mut transformer = ADTransformer::new(ADConfig::default());
        let result = transformer
            .transform(&func, ADMode::ForwardOverReverse)
            .unwrap();

        match result {
            ADTransformResult::Hessian { hvp_func, .. } => {
                // Hessian-vector product function should exist
                assert!(hvp_func.name.contains("tangent") || hvp_func.name.contains("backward"));
            }
            _ => panic!("Expected Hessian result"),
        }
    }

    #[test]
    fn test_checkpoint_strategies() {
        // Test different checkpoint strategies are accepted
        let configs = vec![
            ADConfig {
                checkpoint_strategy: CheckpointStrategy::StoreAll,
                ..Default::default()
            },
            ADConfig {
                checkpoint_strategy: CheckpointStrategy::RecomputeAll,
                ..Default::default()
            },
            ADConfig {
                checkpoint_strategy: CheckpointStrategy::Binomial { checkpoints: 10 },
                ..Default::default()
            },
            ADConfig {
                checkpoint_strategy: CheckpointStrategy::Periodic { interval: 5 },
                ..Default::default()
            },
        ];

        for config in configs {
            let transformer = ADTransformer::new(config);
            assert!(transformer.stats().functions_transformed == 0);
        }
    }

    #[test]
    fn test_caching() {
        let sig = FunctionSignature::new(vec![MirType::F64], MirType::F64);
        let mut builder = FunctionBuilder::new("cached_func", sig);

        let x = builder.param(0).unwrap();
        let result = builder.push_op(Operation::Exp { operand: x }, MirType::F64);

        builder.terminate(Terminator::Return {
            value: Some(result),
        });

        let func = builder.build();

        let mut transformer = ADTransformer::new(ADConfig::default());

        // First transform
        let _ = transformer.transform(&func, ADMode::Forward).unwrap();
        assert_eq!(transformer.stats().functions_transformed, 1);

        // Second transform should use cache (counter doesn't increase)
        let _ = transformer.transform(&func, ADMode::Forward).unwrap();
        assert_eq!(transformer.stats().functions_transformed, 1);

        // Different mode should transform again
        let _ = transformer.transform(&func, ADMode::Reverse).unwrap();
        assert_eq!(transformer.stats().functions_transformed, 2);
    }
}
