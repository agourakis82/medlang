// Week 52: Parametric Polymorphism - Monomorphization
//
// Monomorphization is the process of creating specialized versions of
// generic functions for each set of concrete type arguments used.
//
// For example, `map<Int, String>` and `map<Float, Bool>` would become
// two separate monomorphic functions.

use super::types::{PolyType, Subst, Type, TypeParam, TypeVarId};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

// =============================================================================
// Monomorphization Errors
// =============================================================================

#[derive(Debug, Clone, Error, PartialEq)]
pub enum MonoError {
    #[error("cannot monomorphize: type contains unresolved type variables")]
    UnresolvedTypeVars { vars: Vec<String> },

    #[error("generic function not found: {0}")]
    FunctionNotFound(String),

    #[error("wrong number of type arguments for {name}: expected {expected}, found {found}")]
    WrongTypeArgCount {
        name: String,
        expected: usize,
        found: usize,
    },

    #[error("monomorphization limit exceeded for {0}")]
    LimitExceeded(String),
}

// =============================================================================
// Monomorphization Key
// =============================================================================

/// A unique key for a monomorphized function instance
///
/// Combines the original function name with concrete type arguments
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MonoKey {
    /// Original function name
    pub name: String,
    /// Concrete type arguments
    pub type_args: Vec<Type>,
}

impl MonoKey {
    pub fn new(name: impl Into<String>, type_args: Vec<Type>) -> Self {
        Self {
            name: name.into(),
            type_args,
        }
    }

    /// Generate a mangled name for the monomorphized function
    pub fn mangled_name(&self) -> String {
        if self.type_args.is_empty() {
            return self.name.clone();
        }

        let args_str: Vec<String> = self.type_args.iter().map(mangle_type).collect();
        format!("{}$${}", self.name, args_str.join("$"))
    }
}

/// Mangle a type into a string suitable for function names
fn mangle_type(ty: &Type) -> String {
    match ty {
        Type::Int => "Int".to_string(),
        Type::Float => "Float".to_string(),
        Type::Bool => "Bool".to_string(),
        Type::String => "String".to_string(),
        Type::Unit => "Unit".to_string(),
        Type::List(elem) => format!("List_{}", mangle_type(elem)),
        Type::Option(inner) => format!("Option_{}", mangle_type(inner)),
        Type::Result { ok, err } => {
            format!("Result_{}_{}", mangle_type(ok), mangle_type(err))
        }
        Type::Tuple(elems) => {
            let parts: Vec<_> = elems.iter().map(mangle_type).collect();
            format!("Tuple_{}", parts.join("_"))
        }
        Type::Function { params, ret } => {
            let param_parts: Vec<_> = params.iter().map(mangle_type).collect();
            format!("Fn_{}_{}", param_parts.join("_"), mangle_type(ret))
        }
        Type::Record(fields) => {
            let mut parts: Vec<_> = fields
                .iter()
                .map(|(k, v)| format!("{}_{}", k, mangle_type(v)))
                .collect();
            parts.sort(); // Ensure consistent ordering
            format!("Rec_{}", parts.join("_"))
        }
        Type::Named(name) => name.clone(),
        Type::App { constructor, args } => {
            let arg_parts: Vec<_> = args.iter().map(mangle_type).collect();
            format!("{}_{}", constructor, arg_parts.join("_"))
        }
        Type::Model => "Model".to_string(),
        Type::Protocol => "Protocol".to_string(),
        Type::Policy => "Policy".to_string(),
        Type::EvidenceProgram => "EvidenceProgram".to_string(),
        Type::EvidenceResult => "EvidenceResult".to_string(),
        Type::SimulationResult => "SimulationResult".to_string(),
        Type::FitResult => "FitResult".to_string(),
        Type::SurrogateModel => "SurrogateModel".to_string(),
        Type::RLPolicy => "RLPolicy".to_string(),
        Type::Dual => "Dual".to_string(),
        Type::DualVec => "DualVec".to_string(),
        Type::DualRec => "DualRec".to_string(),
        Type::Var(tv) => format!("T{}", tv.id.0),
        Type::Error => "Error".to_string(),
    }
}

// =============================================================================
// Monomorphized Function
// =============================================================================

/// A monomorphized (specialized) function
#[derive(Debug, Clone)]
pub struct MonoFunction {
    /// Original function name
    pub original_name: String,
    /// Mangled name for this specialization
    pub mangled_name: String,
    /// Concrete type arguments used
    pub type_args: Vec<Type>,
    /// The monomorphized function type
    pub mono_type: Type,
    /// Mapping from type params to concrete types
    pub subst: Subst,
}

impl MonoFunction {
    pub fn new(
        original_name: String,
        type_args: Vec<Type>,
        poly: &PolyType,
    ) -> Result<Self, MonoError> {
        if poly.type_params.len() != type_args.len() {
            return Err(MonoError::WrongTypeArgCount {
                name: original_name,
                expected: poly.type_params.len(),
                found: type_args.len(),
            });
        }

        // Build substitution
        let mut subst = Subst::new();
        for (param, arg) in poly.type_params.iter().zip(type_args.iter()) {
            // Check that argument is fully concrete
            if arg.has_type_vars() {
                let vars: Vec<_> = arg
                    .free_type_vars()
                    .iter()
                    .map(|v| format!("τ{}", v.0))
                    .collect();
                return Err(MonoError::UnresolvedTypeVars { vars });
            }
            subst.insert(param.id, arg.clone());
        }

        let mono_type = subst.apply(&poly.ty);
        let key = MonoKey::new(&original_name, type_args.clone());

        Ok(Self {
            original_name,
            mangled_name: key.mangled_name(),
            type_args,
            mono_type,
            subst,
        })
    }
}

// =============================================================================
// Monomorphization Context
// =============================================================================

/// Context for tracking monomorphization during compilation
#[derive(Debug, Clone, Default)]
pub struct MonoContext {
    /// Generic function definitions: name -> PolyType
    generics: HashMap<String, PolyType>,

    /// Monomorphized instances: MonoKey -> MonoFunction
    instances: HashMap<MonoKey, MonoFunction>,

    /// Pending monomorphizations (worklist)
    pending: Vec<MonoKey>,

    /// Already processed keys
    processed: HashSet<MonoKey>,

    /// Maximum instances per generic (to prevent explosion)
    max_instances_per_generic: usize,

    /// Instance counts per generic
    instance_counts: HashMap<String, usize>,
}

impl MonoContext {
    pub fn new() -> Self {
        Self {
            generics: HashMap::new(),
            instances: HashMap::new(),
            pending: Vec::new(),
            processed: HashSet::new(),
            max_instances_per_generic: 100,
            instance_counts: HashMap::new(),
        }
    }

    /// Set the maximum number of instances allowed per generic
    pub fn with_max_instances(mut self, max: usize) -> Self {
        self.max_instances_per_generic = max;
        self
    }

    /// Register a generic function
    pub fn register_generic(&mut self, name: String, poly: PolyType) {
        self.generics.insert(name, poly);
    }

    /// Request monomorphization of a function with specific type arguments
    pub fn request_mono(&mut self, name: &str, type_args: Vec<Type>) -> Result<String, MonoError> {
        let key = MonoKey::new(name, type_args.clone());

        // Check if already processed
        if let Some(mono) = self.instances.get(&key) {
            return Ok(mono.mangled_name.clone());
        }

        // Check if pending
        if self.pending.contains(&key) {
            return Ok(key.mangled_name());
        }

        // Look up generic
        let poly = self
            .generics
            .get(name)
            .ok_or_else(|| MonoError::FunctionNotFound(name.to_string()))?
            .clone();

        // Check limit
        let count = self.instance_counts.get(name).copied().unwrap_or(0);
        if count >= self.max_instances_per_generic {
            return Err(MonoError::LimitExceeded(name.to_string()));
        }

        // Create monomorphized function
        let mono = MonoFunction::new(name.to_string(), type_args, &poly)?;
        let mangled = mono.mangled_name.clone();

        // Add to pending
        self.pending.push(key.clone());

        // Store instance
        self.instances.insert(key, mono);

        // Update count
        *self.instance_counts.entry(name.to_string()).or_insert(0) += 1;

        Ok(mangled)
    }

    /// Process one pending monomorphization
    pub fn process_one(&mut self) -> Option<MonoKey> {
        if let Some(key) = self.pending.pop() {
            self.processed.insert(key.clone());
            Some(key)
        } else {
            None
        }
    }

    /// Process all pending monomorphizations
    pub fn process_all(&mut self) -> Vec<MonoKey> {
        let mut processed = Vec::new();
        while let Some(key) = self.process_one() {
            processed.push(key);
        }
        processed
    }

    /// Check if there are pending monomorphizations
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Get a monomorphized instance
    pub fn get_instance(&self, key: &MonoKey) -> Option<&MonoFunction> {
        self.instances.get(key)
    }

    /// Get instance by mangled name
    pub fn get_by_mangled_name(&self, mangled: &str) -> Option<&MonoFunction> {
        self.instances.values().find(|m| m.mangled_name == mangled)
    }

    /// Get all instances for a generic function
    pub fn get_instances_for(&self, name: &str) -> Vec<&MonoFunction> {
        self.instances
            .values()
            .filter(|m| m.original_name == name)
            .collect()
    }

    /// Get all monomorphized instances
    pub fn all_instances(&self) -> impl Iterator<Item = &MonoFunction> {
        self.instances.values()
    }

    /// Get the number of instances for a generic
    pub fn instance_count(&self, name: &str) -> usize {
        self.instance_counts.get(name).copied().unwrap_or(0)
    }

    /// Get total number of instances
    pub fn total_instances(&self) -> usize {
        self.instances.len()
    }
}

// =============================================================================
// Monomorphization Collector
// =============================================================================

/// Collects type instantiations from an expression/program
///
/// Walks through the code looking for generic function calls
/// and collects the concrete type arguments used.
#[derive(Debug, Clone, Default)]
pub struct MonoCollector {
    /// Collected instantiations: function name -> set of type argument lists
    instantiations: HashMap<String, HashSet<Vec<Type>>>,
}

impl MonoCollector {
    pub fn new() -> Self {
        Self {
            instantiations: HashMap::new(),
        }
    }

    /// Record an instantiation
    pub fn record(&mut self, name: &str, type_args: Vec<Type>) {
        // Only record if all type args are concrete
        if type_args.iter().all(|t| t.is_monomorphic()) {
            self.instantiations
                .entry(name.to_string())
                .or_default()
                .insert(type_args);
        }
    }

    /// Get all instantiations for a function
    pub fn get(&self, name: &str) -> Option<&HashSet<Vec<Type>>> {
        self.instantiations.get(name)
    }

    /// Get all collected instantiations
    pub fn all(&self) -> &HashMap<String, HashSet<Vec<Type>>> {
        &self.instantiations
    }

    /// Apply collected instantiations to a MonoContext
    pub fn apply_to(&self, ctx: &mut MonoContext) -> Result<(), MonoError> {
        for (name, type_args_set) in &self.instantiations {
            for type_args in type_args_set {
                ctx.request_mono(name, type_args.clone())?;
            }
        }
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generics::types::TypeParam;

    #[test]
    fn test_mono_key_mangled_name() {
        let key = MonoKey::new("map", vec![Type::Int, Type::String]);
        assert_eq!(key.mangled_name(), "map$$Int$String");
    }

    #[test]
    fn test_mono_key_no_args() {
        let key = MonoKey::new("foo", vec![]);
        assert_eq!(key.mangled_name(), "foo");
    }

    #[test]
    fn test_mangle_complex_type() {
        let ty = Type::list(Type::tuple(vec![Type::Int, Type::String]));
        assert_eq!(mangle_type(&ty), "List_Tuple_Int_String");
    }

    #[test]
    fn test_mono_function_creation() {
        // identity<T>(x: T) -> T
        let poly = PolyType::new(
            vec![TypeParam::new("T", 0)],
            Type::function(vec![Type::var(0)], Type::var(0)),
        );

        let mono = MonoFunction::new("identity".to_string(), vec![Type::Int], &poly).unwrap();

        assert_eq!(mono.original_name, "identity");
        assert_eq!(mono.mangled_name, "identity$$Int");

        match &mono.mono_type {
            Type::Function { params, ret } => {
                assert_eq!(params[0], Type::Int);
                assert_eq!(**ret, Type::Int);
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_mono_function_wrong_arg_count() {
        let poly = PolyType::new(
            vec![TypeParam::new("T", 0), TypeParam::new("U", 1)],
            Type::function(vec![Type::var(0)], Type::var(1)),
        );

        let result = MonoFunction::new("foo".to_string(), vec![Type::Int], &poly);
        assert!(matches!(result, Err(MonoError::WrongTypeArgCount { .. })));
    }

    #[test]
    fn test_mono_function_unresolved_vars() {
        let poly = PolyType::new(
            vec![TypeParam::new("T", 0)],
            Type::function(vec![Type::var(0)], Type::var(0)),
        );

        // Try to instantiate with a type that still has variables
        let result = MonoFunction::new("foo".to_string(), vec![Type::var(5)], &poly);
        assert!(matches!(result, Err(MonoError::UnresolvedTypeVars { .. })));
    }

    #[test]
    fn test_mono_context() {
        let mut ctx = MonoContext::new();

        // Register identity<T>
        ctx.register_generic(
            "identity".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", 0)],
                Type::function(vec![Type::var(0)], Type::var(0)),
            ),
        );

        // Request monomorphization
        let name1 = ctx.request_mono("identity", vec![Type::Int]).unwrap();
        let name2 = ctx.request_mono("identity", vec![Type::String]).unwrap();
        let name3 = ctx.request_mono("identity", vec![Type::Int]).unwrap(); // Duplicate

        assert_eq!(name1, "identity$$Int");
        assert_eq!(name2, "identity$$String");
        assert_eq!(name3, name1); // Same as first

        assert_eq!(ctx.instance_count("identity"), 2);
    }

    #[test]
    fn test_mono_context_process_pending() {
        let mut ctx = MonoContext::new();

        ctx.register_generic(
            "id".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", 0)],
                Type::function(vec![Type::var(0)], Type::var(0)),
            ),
        );

        ctx.request_mono("id", vec![Type::Int]).unwrap();
        ctx.request_mono("id", vec![Type::Bool]).unwrap();

        let processed = ctx.process_all();
        assert_eq!(processed.len(), 2);
        assert!(!ctx.has_pending());
    }

    #[test]
    fn test_mono_context_limit() {
        let mut ctx = MonoContext::new().with_max_instances(2);

        ctx.register_generic(
            "id".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", 0)],
                Type::function(vec![Type::var(0)], Type::var(0)),
            ),
        );

        ctx.request_mono("id", vec![Type::Int]).unwrap();
        ctx.request_mono("id", vec![Type::Bool]).unwrap();

        // Third should fail
        let result = ctx.request_mono("id", vec![Type::String]);
        assert!(matches!(result, Err(MonoError::LimitExceeded(_))));
    }

    #[test]
    fn test_mono_collector() {
        let mut collector = MonoCollector::new();

        collector.record("map", vec![Type::Int, Type::String]);
        collector.record("map", vec![Type::Float, Type::Bool]);
        collector.record("map", vec![Type::Int, Type::String]); // Duplicate

        let map_instances = collector.get("map").unwrap();
        assert_eq!(map_instances.len(), 2);
    }

    #[test]
    fn test_mono_collector_apply() {
        let mut collector = MonoCollector::new();
        collector.record("id", vec![Type::Int]);
        collector.record("id", vec![Type::String]);

        let mut ctx = MonoContext::new();
        ctx.register_generic(
            "id".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", 0)],
                Type::function(vec![Type::var(0)], Type::var(0)),
            ),
        );

        collector.apply_to(&mut ctx).unwrap();
        assert_eq!(ctx.instance_count("id"), 2);
    }

    #[test]
    fn test_get_instances_for() {
        let mut ctx = MonoContext::new();

        ctx.register_generic(
            "id".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", 0)],
                Type::function(vec![Type::var(0)], Type::var(0)),
            ),
        );

        ctx.request_mono("id", vec![Type::Int]).unwrap();
        ctx.request_mono("id", vec![Type::Float]).unwrap();

        let instances = ctx.get_instances_for("id");
        assert_eq!(instances.len(), 2);
    }
}
