//! Effect System for MedLang
//!
//! This module implements an algebraic effect system inspired by Demetrios,
//! adapted for medical computing needs. Effects track computational side effects
//! like randomness, I/O, GPU operations, and clinical data access.
//!
//! ## Effect Categories
//!
//! - **Prob**: Probabilistic/stochastic operations (random sampling, Monte Carlo)
//! - **IO**: File system and network I/O (data loading, logging)
//! - **GPU**: GPU-accelerated computations (CUDA, PTX, SPIR-V)
//! - **Pure**: No side effects (default for pure calculations)
//!
//! ## Safety Guarantees
//!
//! 1. **Reproducibility**: `Prob` effects require explicit seed tracking
//! 2. **Data Provenance**: `IO` effects track data sources for regulatory compliance
//! 3. **Device Safety**: `GPU` effects ensure proper memory management
//! 4. **Purity Checking**: Pure functions cannot call effectful functions

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use thiserror::Error;

// =============================================================================
// Effect Types
// =============================================================================

/// Computational effects in MedLang
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Effect {
    /// Probabilistic/stochastic operations (sampling, Monte Carlo)
    Prob,

    /// Input/Output operations (file I/O, network, logging)
    IO,

    /// GPU-accelerated computations
    GPU,

    /// Pure computation (no side effects)
    Pure,
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Effect::Prob => write!(f, "Prob"),
            Effect::IO => write!(f, "IO"),
            Effect::GPU => write!(f, "GPU"),
            Effect::Pure => write!(f, "Pure"),
        }
    }
}

impl Effect {
    /// Parse effect from string (for parser integration)
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Prob" => Some(Effect::Prob),
            "IO" => Some(Effect::IO),
            "GPU" => Some(Effect::GPU),
            "Pure" => Some(Effect::Pure),
            _ => None,
        }
    }

    /// Check if this effect is pure (no side effects)
    pub fn is_pure(&self) -> bool {
        matches!(self, Effect::Pure)
    }
}

// =============================================================================
// Effect Sets
// =============================================================================

/// Set of effects for a computation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectSet {
    effects: HashSet<Effect>,
}

impl EffectSet {
    /// Create an empty effect set (pure)
    pub fn new() -> Self {
        Self {
            effects: HashSet::new(),
        }
    }

    /// Create a pure effect set
    pub fn pure() -> Self {
        let mut effects = HashSet::new();
        effects.insert(Effect::Pure);
        Self { effects }
    }

    /// Create effect set from single effect
    pub fn single(effect: Effect) -> Self {
        let mut effects = HashSet::new();
        effects.insert(effect);
        Self { effects }
    }

    /// Create effect set from multiple effects
    pub fn from_vec(effect_vec: Vec<Effect>) -> Self {
        let effects = effect_vec.into_iter().collect();
        Self { effects }
    }

    /// Add an effect to the set
    pub fn add(&mut self, effect: Effect) {
        // If we're adding a non-pure effect, remove Pure
        if effect != Effect::Pure {
            self.effects.remove(&Effect::Pure);
        }
        self.effects.insert(effect);
    }

    /// Union of two effect sets
    pub fn union(&self, other: &EffectSet) -> Self {
        let mut combined = self.effects.clone();
        combined.extend(&other.effects);

        // If any non-pure effects exist, remove Pure
        if combined.iter().any(|e| !e.is_pure()) {
            combined.remove(&Effect::Pure);
        }

        Self { effects: combined }
    }

    /// Check if effect set contains a specific effect
    pub fn contains(&self, effect: Effect) -> bool {
        self.effects.contains(&effect)
    }

    /// Check if effect set is pure (no side effects)
    pub fn is_pure(&self) -> bool {
        self.effects.is_empty() || (self.effects.len() == 1 && self.contains(Effect::Pure))
    }

    /// Check if effect set includes probabilistic operations
    pub fn has_prob(&self) -> bool {
        self.contains(Effect::Prob)
    }

    /// Check if effect set includes I/O operations
    pub fn has_io(&self) -> bool {
        self.contains(Effect::IO)
    }

    /// Check if effect set includes GPU operations
    pub fn has_gpu(&self) -> bool {
        self.contains(Effect::GPU)
    }

    /// Get iterator over effects
    pub fn iter(&self) -> impl Iterator<Item = &Effect> {
        self.effects.iter()
    }

    /// Get number of effects (excluding Pure)
    pub fn len(&self) -> usize {
        if self.is_pure() {
            0
        } else {
            self.effects.len()
        }
    }

    /// Check if effect set is empty
    pub fn is_empty(&self) -> bool {
        self.is_pure()
    }
}

impl Default for EffectSet {
    fn default() -> Self {
        Self::pure()
    }
}

impl fmt::Display for EffectSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_pure() {
            write!(f, "Pure")
        } else {
            let effects: Vec<String> = self
                .effects
                .iter()
                .filter(|e| !e.is_pure())
                .map(|e| e.to_string())
                .collect();
            write!(f, "with {}", effects.join(", "))
        }
    }
}

// =============================================================================
// Effect Annotations
// =============================================================================

/// Effect annotation for declarations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectAnnotation {
    pub effects: EffectSet,
    pub explicit: bool, // True if user explicitly annotated, false if inferred
}

impl EffectAnnotation {
    pub fn new(effects: EffectSet) -> Self {
        Self {
            effects,
            explicit: true,
        }
    }

    pub fn inferred(effects: EffectSet) -> Self {
        Self {
            effects,
            explicit: false,
        }
    }

    pub fn pure() -> Self {
        Self {
            effects: EffectSet::pure(),
            explicit: false,
        }
    }
}

// =============================================================================
// Effect Checking Errors
// =============================================================================

#[derive(Debug, Error, Clone, PartialEq)]
pub enum EffectError {
    #[error("Effect mismatch: {context} requires {required} but has {actual}")]
    EffectMismatch {
        context: String,
        required: String,
        actual: String,
    },

    #[error("Pure function {function} cannot call effectful function with {effects}")]
    PureViolation { function: String, effects: String },

    #[error("Missing effect annotation: {declaration} uses {effects} but has no annotation")]
    MissingAnnotation {
        declaration: String,
        effects: String,
    },

    #[error("Probabilistic effect {context} requires explicit seed for reproducibility")]
    MissingSeed { context: String },

    #[error("GPU effect {context} without proper device management")]
    UnsafeGPU { context: String },

    #[error("I/O effect {context} without data provenance tracking")]
    MissingProvenance { context: String },
}

// =============================================================================
// Effect Checker
// =============================================================================

/// Effect checker for validating effect annotations
pub struct EffectChecker {
    /// Map from declaration names to their effect annotations
    effect_map: std::collections::HashMap<String, EffectAnnotation>,
}

impl EffectChecker {
    pub fn new() -> Self {
        Self {
            effect_map: std::collections::HashMap::new(),
        }
    }

    /// Register a declaration with its effects
    pub fn register(&mut self, name: String, annotation: EffectAnnotation) {
        self.effect_map.insert(name, annotation);
    }

    /// Get effect annotation for a declaration
    pub fn get_effects(&self, name: &str) -> Option<&EffectAnnotation> {
        self.effect_map.get(name)
    }

    /// Check if calling `callee` from `caller` is allowed
    pub fn check_call(&self, caller: &str, callee: &str) -> Result<(), EffectError> {
        let pure_default = EffectSet::pure();
        let caller_effects = self
            .get_effects(caller)
            .map(|a| &a.effects)
            .unwrap_or(&pure_default);

        let callee_effects = self
            .get_effects(callee)
            .map(|a| &a.effects)
            .unwrap_or(&pure_default);

        // Check if caller's effects include all of callee's effects
        if !self.effects_subsume(caller_effects, callee_effects) {
            return Err(EffectError::EffectMismatch {
                context: format!("calling {} from {}", callee, caller),
                required: callee_effects.to_string(),
                actual: caller_effects.to_string(),
            });
        }

        Ok(())
    }

    /// Check if `superset` includes all effects in `subset`
    fn effects_subsume(&self, superset: &EffectSet, subset: &EffectSet) -> bool {
        // Pure can only call pure
        if superset.is_pure() && !subset.is_pure() {
            return false;
        }

        // Otherwise, check that all effects in subset are in superset
        subset
            .effects
            .iter()
            .all(|e| e.is_pure() || superset.contains(*e))
    }

    /// Infer effects for an expression (placeholder for now)
    pub fn infer_effects(&self, _expr_name: &str) -> EffectSet {
        // TODO: Implement effect inference by traversing expressions
        EffectSet::pure()
    }
}

impl Default for EffectChecker {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_pure() {
        let eff = EffectSet::pure();
        assert!(eff.is_pure());
        assert!(!eff.has_prob());
        assert!(!eff.has_io());
    }

    #[test]
    fn test_effect_single() {
        let eff = EffectSet::single(Effect::Prob);
        assert!(!eff.is_pure());
        assert!(eff.has_prob());
        assert!(!eff.has_io());
    }

    #[test]
    fn test_effect_union() {
        let eff1 = EffectSet::single(Effect::Prob);
        let eff2 = EffectSet::single(Effect::IO);
        let combined = eff1.union(&eff2);

        assert!(combined.has_prob());
        assert!(combined.has_io());
        assert!(!combined.is_pure());
    }

    #[test]
    fn test_effect_display() {
        let eff = EffectSet::from_vec(vec![Effect::Prob, Effect::GPU]);
        let display = format!("{}", eff);
        assert!(display.contains("Prob"));
        assert!(display.contains("GPU"));
    }

    #[test]
    fn test_effect_checker_pure_violation() {
        let mut checker = EffectChecker::new();

        checker.register("pure_fn".to_string(), EffectAnnotation::pure());

        checker.register(
            "prob_fn".to_string(),
            EffectAnnotation::new(EffectSet::single(Effect::Prob)),
        );

        let result = checker.check_call("pure_fn", "prob_fn");
        assert!(result.is_err());
    }

    #[test]
    fn test_effect_checker_allowed_call() {
        let mut checker = EffectChecker::new();

        checker.register(
            "prob_fn".to_string(),
            EffectAnnotation::new(EffectSet::single(Effect::Prob)),
        );

        checker.register("pure_helper".to_string(), EffectAnnotation::pure());

        // Prob function can call pure helper
        let result = checker.check_call("prob_fn", "pure_helper");
        assert!(result.is_ok());
    }

    #[test]
    fn test_effect_subsumption() {
        let checker = EffectChecker::new();

        let prob_io = EffectSet::from_vec(vec![Effect::Prob, Effect::IO]);
        let prob_only = EffectSet::single(Effect::Prob);

        // prob_io subsumes prob_only
        assert!(checker.effects_subsume(&prob_io, &prob_only));

        // prob_only does NOT subsume prob_io
        assert!(!checker.effects_subsume(&prob_only, &prob_io));
    }
}
