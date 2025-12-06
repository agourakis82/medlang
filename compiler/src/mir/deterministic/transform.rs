//! Deterministic FP Transformation Pass
//!
//! Transforms MIR code to ensure deterministic floating-point computation.

use std::collections::HashSet;

use crate::mir::function::MirFunction;
use crate::mir::inst::{Instruction, Operation};
use crate::mir::module::MirModule;
use crate::mir::value::{ValueId, ValueIdGen};

use super::analysis::{
    AccuracyRequirement, DeterminismAnalysis, DeterminismLevel, FPAnalysisResult,
};
use super::rounding::{RoundingContext, RoundingMode, RoundingTransform};
use super::summation::SummationAlgorithm;
use super::transcendental::{DeterministicMath, SoftwareTranscendental};

/// Configuration for the deterministic FP transform
#[derive(Debug, Clone)]
pub struct FPTransformConfig {
    /// Required determinism level
    pub determinism_level: DeterminismLevel,
    /// Required accuracy
    pub accuracy: AccuracyRequirement,
    /// Summation algorithm to use
    pub summation_algorithm: SummationAlgorithm,
    /// Math implementation configuration
    pub math_config: DeterministicMath,
    /// Rounding mode to enforce
    pub rounding_mode: RoundingMode,
    /// Whether to use strict FP
    pub strict_fp: bool,
    /// Whether to disable FMA for consistency
    pub disable_fma: bool,
    /// Whether to handle denormals consistently
    pub flush_denormals: bool,
}

impl Default for FPTransformConfig {
    fn default() -> Self {
        Self {
            determinism_level: DeterminismLevel::CrossPlatform,
            accuracy: AccuracyRequirement::OneULP,
            summation_algorithm: SummationAlgorithm::Pairwise,
            math_config: DeterministicMath::default(),
            rounding_mode: RoundingMode::NearestEven,
            strict_fp: false,
            disable_fma: false,
            flush_denormals: false,
        }
    }
}

impl FPTransformConfig {
    /// Configuration for maximum reproducibility (slower)
    pub fn bit_exact() -> Self {
        Self {
            determinism_level: DeterminismLevel::BitExact,
            accuracy: AccuracyRequirement::CorrectlyRounded,
            summation_algorithm: SummationAlgorithm::Tree,
            math_config: DeterministicMath::correctly_rounded(),
            rounding_mode: RoundingMode::NearestEven,
            strict_fp: true,
            disable_fma: true,
            flush_denormals: true,
        }
    }

    /// Configuration for local reproducibility (faster)
    pub fn local_reproducible() -> Self {
        Self {
            determinism_level: DeterminismLevel::LocalReproducible,
            accuracy: AccuracyRequirement::IEEE754,
            summation_algorithm: SummationAlgorithm::Kahan,
            math_config: DeterministicMath::hardware(),
            rounding_mode: RoundingMode::NearestEven,
            strict_fp: false,
            disable_fma: false,
            flush_denormals: false,
        }
    }

    /// Configuration for cross-platform reproducibility
    pub fn cross_platform() -> Self {
        Self {
            determinism_level: DeterminismLevel::CrossPlatform,
            accuracy: AccuracyRequirement::OneULP,
            summation_algorithm: SummationAlgorithm::Pairwise,
            math_config: DeterministicMath::default(),
            rounding_mode: RoundingMode::NearestEven,
            strict_fp: true,
            disable_fma: true,
            flush_denormals: false,
        }
    }
}

/// Result of deterministic FP transformation
#[derive(Debug)]
pub struct FPTransformResult {
    /// Transformed module
    pub module: MirModule,
    /// Analysis results
    pub analysis: FPAnalysisResult,
    /// Number of operations transformed
    pub ops_transformed: usize,
    /// Warnings generated
    pub warnings: Vec<String>,
}

/// Main transformation pass for deterministic FP
pub struct DeterministicFPTransform {
    /// Configuration
    config: FPTransformConfig,
    /// Value ID generator
    id_gen: ValueIdGen,
    /// Values that have been transformed
    transformed_values: HashSet<ValueId>,
    /// Transcendental transformer
    transcendental: SoftwareTranscendental,
    /// Rounding transformer
    rounding: RoundingTransform,
    /// Statistics
    ops_transformed: usize,
    /// Warnings
    warnings: Vec<String>,
}

impl DeterministicFPTransform {
    /// Create a new transform with configuration
    pub fn new(config: FPTransformConfig) -> Self {
        let transcendental = SoftwareTranscendental::new(config.math_config.clone());
        let rounding_ctx = if config.strict_fp {
            RoundingContext::with_mode(config.rounding_mode).strict()
        } else {
            RoundingContext::with_mode(config.rounding_mode)
        };
        let rounding = RoundingTransform::new(rounding_ctx);

        Self {
            config,
            id_gen: ValueIdGen::new(),
            transformed_values: HashSet::new(),
            transcendental,
            rounding,
            ops_transformed: 0,
            warnings: Vec::new(),
        }
    }

    /// Transform an entire module
    pub fn transform_module(&mut self, module: &MirModule) -> FPTransformResult {
        let mut analysis =
            DeterminismAnalysis::new(self.config.determinism_level, self.config.accuracy);
        let analysis_result = analysis.analyze_module(module);

        let mut new_module = module.clone();

        if analysis_result.current_level < self.config.determinism_level {
            for func in &mut new_module.functions {
                *func = self.transform_function(func);
            }
        }

        FPTransformResult {
            module: new_module,
            analysis: analysis_result,
            ops_transformed: self.ops_transformed,
            warnings: self.warnings.clone(),
        }
    }

    /// Transform a single function
    pub fn transform_function(&mut self, func: &MirFunction) -> MirFunction {
        let mut new_func = func.clone();

        if self.config.strict_fp {
            new_func = self
                .rounding
                .transform_function(&new_func, &mut self.id_gen);
        }

        for block in &mut new_func.blocks {
            let mut new_instructions = Vec::new();

            for inst in &block.instructions {
                let transformed = self.transform_instruction(inst);
                new_instructions.extend(transformed);
            }

            block.instructions = new_instructions;
        }

        new_func
    }

    /// Transform a single instruction
    fn transform_instruction(&mut self, inst: &Instruction) -> Vec<Instruction> {
        match &inst.op {
            // Transform transcendentals
            Operation::Sin { .. }
            | Operation::Cos { .. }
            | Operation::Tan { .. }
            | Operation::Exp { .. }
            | Operation::Log { .. }
            | Operation::Pow { .. } => {
                if self.config.determinism_level >= DeterminismLevel::CrossPlatform {
                    self.ops_transformed += 1;
                    self.transcendental.transform_op(inst, &mut self.id_gen)
                } else {
                    vec![inst.clone()]
                }
            }

            // Disable FMA if configured
            Operation::FMA { a, b, c } if self.config.disable_fma => {
                self.ops_transformed += 1;
                self.expand_fma(*a, *b, *c, inst)
            }

            // Pass through other operations
            _ => vec![inst.clone()],
        }
    }

    /// Expand FMA into separate multiply and add
    fn expand_fma(
        &mut self,
        a: ValueId,
        b: ValueId,
        c: ValueId,
        inst: &Instruction,
    ) -> Vec<Instruction> {
        let mut insts = Vec::new();

        let product = self.id_gen.next();
        insts.push(
            Instruction::new(Operation::FMul { lhs: a, rhs: b }, inst.ty.clone())
                .with_result(product),
        );

        let mut add_inst = Instruction::new(
            Operation::FAdd {
                lhs: product,
                rhs: c,
            },
            inst.ty.clone(),
        );
        if let Some(r) = inst.result {
            add_inst = add_inst.with_result(r);
        }
        insts.push(add_inst);

        insts
    }
}

/// Builder for deterministic FP configuration
pub struct FPTransformBuilder {
    config: FPTransformConfig,
}

impl Default for FPTransformBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FPTransformBuilder {
    /// Create a new builder with defaults
    pub fn new() -> Self {
        Self {
            config: FPTransformConfig::default(),
        }
    }

    /// Set determinism level
    pub fn determinism_level(mut self, level: DeterminismLevel) -> Self {
        self.config.determinism_level = level;
        self
    }

    /// Set accuracy requirement
    pub fn accuracy(mut self, accuracy: AccuracyRequirement) -> Self {
        self.config.accuracy = accuracy;
        self
    }

    /// Set summation algorithm
    pub fn summation(mut self, algorithm: SummationAlgorithm) -> Self {
        self.config.summation_algorithm = algorithm;
        self
    }

    /// Enable strict FP
    pub fn strict(mut self) -> Self {
        self.config.strict_fp = true;
        self
    }

    /// Disable FMA
    pub fn no_fma(mut self) -> Self {
        self.config.disable_fma = true;
        self
    }

    /// Enable denormal flushing
    pub fn flush_denormals(mut self) -> Self {
        self.config.flush_denormals = true;
        self
    }

    /// Build the configuration
    pub fn build(self) -> FPTransformConfig {
        self.config
    }

    /// Build and create a transform
    pub fn build_transform(self) -> DeterministicFPTransform {
        DeterministicFPTransform::new(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_presets() {
        let bit_exact = FPTransformConfig::bit_exact();
        assert_eq!(bit_exact.determinism_level, DeterminismLevel::BitExact);
        assert!(bit_exact.strict_fp);
        assert!(bit_exact.disable_fma);

        let local = FPTransformConfig::local_reproducible();
        assert_eq!(local.determinism_level, DeterminismLevel::LocalReproducible);
        assert!(!local.strict_fp);

        let cross = FPTransformConfig::cross_platform();
        assert_eq!(cross.determinism_level, DeterminismLevel::CrossPlatform);
        assert!(cross.disable_fma);
    }

    #[test]
    fn test_builder() {
        let config = FPTransformBuilder::new()
            .determinism_level(DeterminismLevel::BitExact)
            .strict()
            .no_fma()
            .summation(SummationAlgorithm::Tree)
            .build();

        assert_eq!(config.determinism_level, DeterminismLevel::BitExact);
        assert!(config.strict_fp);
        assert!(config.disable_fma);
        assert_eq!(config.summation_algorithm, SummationAlgorithm::Tree);
    }
}
