// Week 52: Parametric Polymorphism - Type System
//
// This module implements the core type representation for generics:
// - Type variables (TypeVar)
// - Polymorphic types (PolyType / type schemes)
// - Substitution maps
// - Type constructors

use std::collections::{HashMap, HashSet};
use std::fmt;

// =============================================================================
// Type Variables
// =============================================================================

/// A unique identifier for type variables
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeVarId(pub u32);

impl TypeVarId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

impl fmt::Display for TypeVarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "τ{}", self.0)
    }
}

/// Type variable with optional name (for user-defined type parameters like T, U)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeVar {
    pub id: TypeVarId,
    pub name: Option<String>,
}

impl TypeVar {
    pub fn fresh(id: u32) -> Self {
        Self {
            id: TypeVarId(id),
            name: None,
        }
    }

    pub fn named(id: u32, name: String) -> Self {
        Self {
            id: TypeVarId(id),
            name: Some(name),
        }
    }

    pub fn display_name(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| format!("τ{}", self.id.0))
    }
}

impl fmt::Display for TypeVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// =============================================================================
// Types
// =============================================================================

/// Core type representation with support for polymorphism
#[derive(Debug, Clone)]
pub enum Type {
    // Primitive types
    Int,
    Float,
    Bool,
    String,
    Unit,

    // Type variable (for polymorphism)
    Var(TypeVar),

    // Function type: (T1, T2, ...) -> R
    Function {
        params: Vec<Type>,
        ret: Box<Type>,
    },

    // Record type: { field1: T1, field2: T2, ... }
    Record(HashMap<String, Type>),

    // List/Array type: [T]
    List(Box<Type>),

    // Option type: Option<T>
    Option(Box<Type>),

    // Result type: Result<T, E>
    Result {
        ok: Box<Type>,
        err: Box<Type>,
    },

    // Tuple type: (T1, T2, ...)
    Tuple(Vec<Type>),

    // Named type (enum, struct, or alias): TypeName
    Named(String),

    // Generic type application: TypeName<T1, T2, ...>
    // e.g., Vec<Int>, Map<String, Float>
    App {
        constructor: String,
        args: Vec<Type>,
    },

    // Domain types (carried over from CoreType)
    Model,
    Protocol,
    Policy,
    EvidenceProgram,
    EvidenceResult,
    SimulationResult,
    FitResult,
    SurrogateModel,
    RLPolicy,

    // AD types (Week 50-51)
    Dual,
    DualVec,
    DualRec,

    // Error/Unknown type (for error recovery)
    Error,
}

impl Type {
    // Constructors
    pub fn var(id: u32) -> Self {
        Type::Var(TypeVar::fresh(id))
    }

    pub fn named_var(id: u32, name: impl Into<String>) -> Self {
        Type::Var(TypeVar::named(id, name.into()))
    }

    pub fn function(params: Vec<Type>, ret: Type) -> Self {
        Type::Function {
            params,
            ret: Box::new(ret),
        }
    }

    pub fn list(elem: Type) -> Self {
        Type::List(Box::new(elem))
    }

    pub fn option(inner: Type) -> Self {
        Type::Option(Box::new(inner))
    }

    pub fn result(ok: Type, err: Type) -> Self {
        Type::Result {
            ok: Box::new(ok),
            err: Box::new(err),
        }
    }

    pub fn tuple(elems: Vec<Type>) -> Self {
        Type::Tuple(elems)
    }

    pub fn app(constructor: impl Into<String>, args: Vec<Type>) -> Self {
        Type::App {
            constructor: constructor.into(),
            args,
        }
    }

    /// Check if this type contains any type variables
    pub fn has_type_vars(&self) -> bool {
        match self {
            Type::Var(_) => true,
            Type::Function { params, ret } => {
                params.iter().any(|p| p.has_type_vars()) || ret.has_type_vars()
            }
            Type::Record(fields) => fields.values().any(|t| t.has_type_vars()),
            Type::List(elem) => elem.has_type_vars(),
            Type::Option(inner) => inner.has_type_vars(),
            Type::Result { ok, err } => ok.has_type_vars() || err.has_type_vars(),
            Type::Tuple(elems) => elems.iter().any(|t| t.has_type_vars()),
            Type::App { args, .. } => args.iter().any(|t| t.has_type_vars()),
            _ => false,
        }
    }

    /// Get all free type variables in this type
    pub fn free_type_vars(&self) -> HashSet<TypeVarId> {
        let mut vars = HashSet::new();
        self.collect_type_vars(&mut vars);
        vars
    }

    fn collect_type_vars(&self, vars: &mut HashSet<TypeVarId>) {
        match self {
            Type::Var(tv) => {
                vars.insert(tv.id);
            }
            Type::Function { params, ret } => {
                for p in params {
                    p.collect_type_vars(vars);
                }
                ret.collect_type_vars(vars);
            }
            Type::Record(fields) => {
                for t in fields.values() {
                    t.collect_type_vars(vars);
                }
            }
            Type::List(elem) => elem.collect_type_vars(vars),
            Type::Option(inner) => inner.collect_type_vars(vars),
            Type::Result { ok, err } => {
                ok.collect_type_vars(vars);
                err.collect_type_vars(vars);
            }
            Type::Tuple(elems) => {
                for e in elems {
                    e.collect_type_vars(vars);
                }
            }
            Type::App { args, .. } => {
                for a in args {
                    a.collect_type_vars(vars);
                }
            }
            _ => {}
        }
    }

    /// Check if this is a monomorphic type (no type variables)
    pub fn is_monomorphic(&self) -> bool {
        !self.has_type_vars()
    }

    /// Check if this is a ground type (concrete, no variables)
    pub fn is_ground(&self) -> bool {
        self.is_monomorphic()
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => write!(f, "Int"),
            Type::Float => write!(f, "Float"),
            Type::Bool => write!(f, "Bool"),
            Type::String => write!(f, "String"),
            Type::Unit => write!(f, "Unit"),
            Type::Var(tv) => write!(f, "{}", tv),
            Type::Function { params, ret } => {
                write!(f, "(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            Type::Record(fields) => {
                write!(f, "{{ ")?;
                let mut first = true;
                for (name, ty) in fields {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", name, ty)?;
                    first = false;
                }
                write!(f, " }}")
            }
            Type::List(elem) => write!(f, "[{}]", elem),
            Type::Option(inner) => write!(f, "Option<{}>", inner),
            Type::Result { ok, err } => write!(f, "Result<{}, {}>", ok, err),
            Type::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e)?;
                }
                write!(f, ")")
            }
            Type::Named(name) => write!(f, "{}", name),
            Type::App { constructor, args } => {
                write!(f, "{}<", constructor)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", a)?;
                }
                write!(f, ">")
            }
            Type::Model => write!(f, "Model"),
            Type::Protocol => write!(f, "Protocol"),
            Type::Policy => write!(f, "Policy"),
            Type::EvidenceProgram => write!(f, "EvidenceProgram"),
            Type::EvidenceResult => write!(f, "EvidenceResult"),
            Type::SimulationResult => write!(f, "SimulationResult"),
            Type::FitResult => write!(f, "FitResult"),
            Type::SurrogateModel => write!(f, "SurrogateModel"),
            Type::RLPolicy => write!(f, "RLPolicy"),
            Type::Dual => write!(f, "Dual"),
            Type::DualVec => write!(f, "DualVec"),
            Type::DualRec => write!(f, "DualRec"),
            Type::Error => write!(f, "<error>"),
        }
    }
}

// Manual PartialEq implementation for Type
impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Type::Int, Type::Int) => true,
            (Type::Float, Type::Float) => true,
            (Type::Bool, Type::Bool) => true,
            (Type::String, Type::String) => true,
            (Type::Unit, Type::Unit) => true,
            (Type::Var(v1), Type::Var(v2)) => v1 == v2,
            (
                Type::Function {
                    params: p1,
                    ret: r1,
                },
                Type::Function {
                    params: p2,
                    ret: r2,
                },
            ) => p1 == p2 && r1 == r2,
            (Type::Record(f1), Type::Record(f2)) => f1 == f2,
            (Type::List(e1), Type::List(e2)) => e1 == e2,
            (Type::Option(i1), Type::Option(i2)) => i1 == i2,
            (Type::Result { ok: o1, err: e1 }, Type::Result { ok: o2, err: e2 }) => {
                o1 == o2 && e1 == e2
            }
            (Type::Tuple(t1), Type::Tuple(t2)) => t1 == t2,
            (Type::Named(n1), Type::Named(n2)) => n1 == n2,
            (
                Type::App {
                    constructor: c1,
                    args: a1,
                },
                Type::App {
                    constructor: c2,
                    args: a2,
                },
            ) => c1 == c2 && a1 == a2,
            (Type::Model, Type::Model) => true,
            (Type::Protocol, Type::Protocol) => true,
            (Type::Policy, Type::Policy) => true,
            (Type::EvidenceProgram, Type::EvidenceProgram) => true,
            (Type::EvidenceResult, Type::EvidenceResult) => true,
            (Type::SimulationResult, Type::SimulationResult) => true,
            (Type::FitResult, Type::FitResult) => true,
            (Type::SurrogateModel, Type::SurrogateModel) => true,
            (Type::RLPolicy, Type::RLPolicy) => true,
            (Type::Dual, Type::Dual) => true,
            (Type::DualVec, Type::DualVec) => true,
            (Type::DualRec, Type::DualRec) => true,
            (Type::Error, Type::Error) => true,
            _ => false,
        }
    }
}

impl Eq for Type {}

// Manual Hash implementation for Type (needed for use in HashSet/HashMap)
impl std::hash::Hash for Type {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Use discriminant for basic distinction
        std::mem::discriminant(self).hash(state);
        match self {
            Type::Var(v) => v.hash(state),
            Type::Function { params, ret } => {
                params.hash(state);
                ret.hash(state);
            }
            Type::Record(fields) => {
                // Sort keys for consistent hashing
                let mut keys: Vec<_> = fields.keys().collect();
                keys.sort();
                for k in keys {
                    k.hash(state);
                    fields[k].hash(state);
                }
            }
            Type::List(elem) => elem.hash(state),
            Type::Option(inner) => inner.hash(state),
            Type::Result { ok, err } => {
                ok.hash(state);
                err.hash(state);
            }
            Type::Tuple(elems) => elems.hash(state),
            Type::Named(name) => name.hash(state),
            Type::App { constructor, args } => {
                constructor.hash(state);
                args.hash(state);
            }
            // Primitive and domain types are distinguished by discriminant only
            _ => {}
        }
    }
}

// =============================================================================
// Polymorphic Types (Type Schemes)
// =============================================================================

/// A type parameter declaration: T, T: Constraint, etc.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeParam {
    pub name: String,
    pub id: TypeVarId,
    pub bounds: Vec<TypeBound>,
}

impl TypeParam {
    pub fn new(name: impl Into<String>, id: u32) -> Self {
        Self {
            name: name.into(),
            id: TypeVarId(id),
            bounds: Vec::new(),
        }
    }

    pub fn with_bound(mut self, bound: TypeBound) -> Self {
        self.bounds.push(bound);
        self
    }
}

impl fmt::Display for TypeParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if !self.bounds.is_empty() {
            write!(f, ": ")?;
            for (i, b) in self.bounds.iter().enumerate() {
                if i > 0 {
                    write!(f, " + ")?;
                }
                write!(f, "{}", b)?;
            }
        }
        Ok(())
    }
}

/// Type bounds/constraints (for future trait-like extensions)
#[derive(Debug, Clone, PartialEq)]
pub enum TypeBound {
    /// Must implement a trait: T: SomeTrait
    Trait(String),
    /// Must be a numeric type: T: Num
    Num,
    /// Must be comparable: T: Ord
    Ord,
    /// Must be equatable: T: Eq
    Eq,
    /// Must be copyable: T: Copy
    Copy,
    /// Must support automatic differentiation: T: Differentiable
    Differentiable,
}

impl fmt::Display for TypeBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeBound::Trait(name) => write!(f, "{}", name),
            TypeBound::Num => write!(f, "Num"),
            TypeBound::Ord => write!(f, "Ord"),
            TypeBound::Eq => write!(f, "Eq"),
            TypeBound::Copy => write!(f, "Copy"),
            TypeBound::Differentiable => write!(f, "Differentiable"),
        }
    }
}

/// Polymorphic type (type scheme): ∀T, U. (T, U) -> T
///
/// Represents a type with bound type variables that can be instantiated
/// to concrete types.
#[derive(Debug, Clone, PartialEq)]
pub struct PolyType {
    /// Bound type parameters (universally quantified)
    pub type_params: Vec<TypeParam>,
    /// The underlying type (may contain type variables)
    pub ty: Type,
}

impl PolyType {
    pub fn new(type_params: Vec<TypeParam>, ty: Type) -> Self {
        Self { type_params, ty }
    }

    /// Create a monomorphic polytype (no type parameters)
    pub fn mono(ty: Type) -> Self {
        Self {
            type_params: Vec::new(),
            ty,
        }
    }

    /// Check if this is a monomorphic type
    pub fn is_monomorphic(&self) -> bool {
        self.type_params.is_empty()
    }

    /// Get the arity (number of type parameters)
    pub fn arity(&self) -> usize {
        self.type_params.len()
    }

    /// Get the bound type variable IDs
    pub fn bound_vars(&self) -> HashSet<TypeVarId> {
        self.type_params.iter().map(|p| p.id).collect()
    }
}

impl fmt::Display for PolyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.type_params.is_empty() {
            write!(f, "∀")?;
            for (i, p) in self.type_params.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", p)?;
            }
            write!(f, ". ")?;
        }
        write!(f, "{}", self.ty)
    }
}

// =============================================================================
// Substitution
// =============================================================================

/// A substitution maps type variables to types
#[derive(Debug, Clone, Default)]
pub struct Subst {
    mappings: HashMap<TypeVarId, Type>,
}

impl Subst {
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
        }
    }

    /// Create a substitution from a single mapping
    pub fn singleton(var: TypeVarId, ty: Type) -> Self {
        let mut s = Self::new();
        s.insert(var, ty);
        s
    }

    /// Insert a mapping
    pub fn insert(&mut self, var: TypeVarId, ty: Type) {
        self.mappings.insert(var, ty);
    }

    /// Look up a type variable
    pub fn get(&self, var: &TypeVarId) -> Option<&Type> {
        self.mappings.get(var)
    }

    /// Check if the substitution is empty
    pub fn is_empty(&self) -> bool {
        self.mappings.is_empty()
    }

    /// Get the number of mappings
    pub fn len(&self) -> usize {
        self.mappings.len()
    }

    /// Apply this substitution to a type
    pub fn apply(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(tv) => {
                if let Some(replacement) = self.mappings.get(&tv.id) {
                    // Avoid infinite loop if var maps to itself
                    if let Type::Var(repl_tv) = replacement {
                        if repl_tv.id == tv.id {
                            return replacement.clone();
                        }
                    }
                    // Apply substitution recursively (in case replacement has vars)
                    self.apply(replacement)
                } else {
                    ty.clone()
                }
            }
            Type::Function { params, ret } => Type::Function {
                params: params.iter().map(|p| self.apply(p)).collect(),
                ret: Box::new(self.apply(ret)),
            },
            Type::Record(fields) => Type::Record(
                fields
                    .iter()
                    .map(|(k, v)| (k.clone(), self.apply(v)))
                    .collect(),
            ),
            Type::List(elem) => Type::List(Box::new(self.apply(elem))),
            Type::Option(inner) => Type::Option(Box::new(self.apply(inner))),
            Type::Result { ok, err } => Type::Result {
                ok: Box::new(self.apply(ok)),
                err: Box::new(self.apply(err)),
            },
            Type::Tuple(elems) => Type::Tuple(elems.iter().map(|e| self.apply(e)).collect()),
            Type::App { constructor, args } => Type::App {
                constructor: constructor.clone(),
                args: args.iter().map(|a| self.apply(a)).collect(),
            },
            // All other types are unchanged
            _ => ty.clone(),
        }
    }

    /// Compose two substitutions: (self ∘ other)(t) = self(other(t))
    /// First apply `other`, then apply `self`
    pub fn compose(&self, other: &Subst) -> Subst {
        let mut result = Subst::new();

        // Apply self to all types in other
        for (var, ty) in &other.mappings {
            result.insert(*var, self.apply(ty));
        }

        // Add mappings from self that don't conflict
        for (var, ty) in &self.mappings {
            if !result.mappings.contains_key(var) {
                result.insert(*var, ty.clone());
            }
        }

        result
    }

    /// Extend this substitution with another (in-place compose)
    pub fn extend(&mut self, other: &Subst) {
        *self = self.compose(other);
    }

    /// Get domain (set of mapped variables)
    pub fn domain(&self) -> HashSet<TypeVarId> {
        self.mappings.keys().copied().collect()
    }

    /// Iterator over mappings
    pub fn iter(&self) -> impl Iterator<Item = (&TypeVarId, &Type)> {
        self.mappings.iter()
    }
}

impl fmt::Display for Subst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        let mut first = true;
        for (var, ty) in &self.mappings {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{} ↦ {}", var, ty)?;
            first = false;
        }
        write!(f, "}}")
    }
}

// =============================================================================
// Type Variable Generator
// =============================================================================

/// Generates fresh type variables with unique IDs
#[derive(Debug, Clone, Default)]
pub struct TypeVarGen {
    next_id: u32,
}

impl TypeVarGen {
    /// Start at 1000 to avoid collisions with user-defined type params (typically 0, 1, 2, ...)
    const FRESH_ID_OFFSET: u32 = 1000;

    pub fn new() -> Self {
        Self {
            next_id: Self::FRESH_ID_OFFSET,
        }
    }

    /// Generate a fresh anonymous type variable
    pub fn fresh(&mut self) -> TypeVar {
        let id = self.next_id;
        self.next_id += 1;
        TypeVar::fresh(id)
    }

    /// Generate a fresh type variable with a name
    pub fn fresh_named(&mut self, name: impl Into<String>) -> TypeVar {
        let id = self.next_id;
        self.next_id += 1;
        TypeVar::named(id, name.into())
    }

    /// Generate a fresh type (Type::Var)
    pub fn fresh_type(&mut self) -> Type {
        Type::Var(self.fresh())
    }

    /// Generate a fresh named type (Type::Var)
    pub fn fresh_named_type(&mut self, name: impl Into<String>) -> Type {
        Type::Var(self.fresh_named(name))
    }

    /// Get the current ID counter (for debugging)
    pub fn current_id(&self) -> u32 {
        self.next_id
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_var_display() {
        let tv1 = TypeVar::fresh(0);
        assert_eq!(tv1.display_name(), "τ0");

        let tv2 = TypeVar::named(1, "T".to_string());
        assert_eq!(tv2.display_name(), "T");
    }

    #[test]
    fn test_type_display() {
        assert_eq!(Type::Int.to_string(), "Int");
        assert_eq!(Type::Float.to_string(), "Float");

        let fn_ty = Type::function(vec![Type::Int, Type::String], Type::Bool);
        assert_eq!(fn_ty.to_string(), "(Int, String) -> Bool");

        let list_ty = Type::list(Type::Int);
        assert_eq!(list_ty.to_string(), "[Int]");

        let var_ty = Type::named_var(0, "T");
        assert_eq!(var_ty.to_string(), "T");
    }

    #[test]
    fn test_free_type_vars() {
        let ty = Type::function(
            vec![Type::named_var(0, "T"), Type::Int],
            Type::named_var(1, "U"),
        );

        let vars = ty.free_type_vars();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&TypeVarId(0)));
        assert!(vars.contains(&TypeVarId(1)));
    }

    #[test]
    fn test_has_type_vars() {
        assert!(!Type::Int.has_type_vars());
        assert!(Type::var(0).has_type_vars());

        let fn_ty = Type::function(vec![Type::Int], Type::var(0));
        assert!(fn_ty.has_type_vars());
    }

    #[test]
    fn test_substitution() {
        let mut subst = Subst::new();
        subst.insert(TypeVarId(0), Type::Int);
        subst.insert(TypeVarId(1), Type::String);

        let ty = Type::function(vec![Type::var(0), Type::var(1)], Type::var(0));

        let result = subst.apply(&ty);

        match result {
            Type::Function { params, ret } => {
                assert_eq!(params[0], Type::Int);
                assert_eq!(params[1], Type::String);
                assert_eq!(*ret, Type::Int);
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_substitution_compose() {
        // s1: T0 -> Int
        let mut s1 = Subst::new();
        s1.insert(TypeVarId(0), Type::Int);

        // s2: T1 -> T0
        let mut s2 = Subst::new();
        s2.insert(TypeVarId(1), Type::var(0));

        // compose: s1 ∘ s2
        // T1 -> s1(T0) = Int
        // T0 -> Int
        let composed = s1.compose(&s2);

        let t1_result = composed.apply(&Type::var(1));
        assert_eq!(t1_result, Type::Int);
    }

    #[test]
    fn test_polytype() {
        let params = vec![TypeParam::new("T", 0), TypeParam::new("U", 1)];
        let ty = Type::function(vec![Type::var(0), Type::var(1)], Type::var(0));
        let poly = PolyType::new(params, ty);

        assert_eq!(poly.arity(), 2);
        assert!(!poly.is_monomorphic());
        assert_eq!(poly.to_string(), "∀T, U. (τ0, τ1) -> τ0");
    }

    #[test]
    fn test_type_var_gen() {
        let mut gen = TypeVarGen::new();
        let start = TypeVarGen::FRESH_ID_OFFSET;

        let v1 = gen.fresh();
        assert_eq!(v1.id.0, start);

        let v2 = gen.fresh_named("T");
        assert_eq!(v2.id.0, start + 1);
        assert_eq!(v2.name, Some("T".to_string()));

        let v3 = gen.fresh();
        assert_eq!(v3.id.0, start + 2);
    }

    #[test]
    fn test_type_param_with_bounds() {
        let param = TypeParam::new("T", 0)
            .with_bound(TypeBound::Num)
            .with_bound(TypeBound::Ord);

        assert_eq!(param.bounds.len(), 2);
        assert_eq!(param.to_string(), "T: Num + Ord");
    }
}
