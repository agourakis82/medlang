// Week 52: Parametric Polymorphism - Type Inference
//
// This module implements Hindley-Milner style type inference
// with support for let-polymorphism and generic functions.

use super::types::{PolyType, Subst, Type, TypeParam, TypeVar, TypeVarGen, TypeVarId};
use super::unify::{unify, Constraint, UnifyError};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

// =============================================================================
// Type Inference Errors
// =============================================================================

#[derive(Debug, Clone, Error, PartialEq)]
pub enum InferError {
    #[error("undefined variable: {0}")]
    UndefinedVar(String),

    #[error("undefined function: {0}")]
    UndefinedFn(String),

    #[error("unification failed: {0}")]
    UnifyError(#[from] UnifyError),

    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    #[error("not a function type: {0}")]
    NotAFunction(String),

    #[error("wrong number of type arguments: expected {expected}, found {found}")]
    WrongTypeArgCount { expected: usize, found: usize },

    #[error("wrong number of arguments: expected {expected}, found {found}")]
    WrongArgCount { expected: usize, found: usize },

    #[error("cannot infer type for: {0}")]
    CannotInfer(String),

    #[error("recursive type not allowed: {0}")]
    RecursiveType(String),

    #[error("type bound not satisfied: {bound} for type {ty}")]
    BoundNotSatisfied { bound: String, ty: String },
}

// =============================================================================
// Type Environment
// =============================================================================

/// Type environment mapping names to polymorphic types
#[derive(Debug, Clone, Default)]
pub struct TypeEnv {
    /// Variable bindings: name -> PolyType
    bindings: HashMap<String, PolyType>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    /// Look up a variable in the environment
    pub fn lookup(&self, name: &str) -> Option<&PolyType> {
        self.bindings.get(name)
    }

    /// Extend the environment with a new binding
    pub fn extend(&mut self, name: String, ty: PolyType) {
        self.bindings.insert(name, ty);
    }

    /// Create an extended environment (functional style)
    pub fn with(&self, name: String, ty: PolyType) -> Self {
        let mut new_env = self.clone();
        new_env.extend(name, ty);
        new_env
    }

    /// Remove a binding
    pub fn remove(&mut self, name: &str) {
        self.bindings.remove(name);
    }

    /// Get all free type variables in the environment
    pub fn free_type_vars(&self) -> HashSet<TypeVarId> {
        let mut vars = HashSet::new();
        for poly in self.bindings.values() {
            let bound = poly.bound_vars();
            for var in poly.ty.free_type_vars() {
                if !bound.contains(&var) {
                    vars.insert(var);
                }
            }
        }
        vars
    }

    /// Apply a substitution to all types in the environment
    pub fn apply_subst(&self, subst: &Subst) -> Self {
        let mut new_bindings = HashMap::new();
        for (name, poly) in &self.bindings {
            let new_ty = subst.apply(&poly.ty);
            // Keep the same type params, just update the underlying type
            let new_poly = PolyType::new(poly.type_params.clone(), new_ty);
            new_bindings.insert(name.clone(), new_poly);
        }
        Self {
            bindings: new_bindings,
        }
    }

    /// Get all bindings
    pub fn bindings(&self) -> &HashMap<String, PolyType> {
        &self.bindings
    }
}

// =============================================================================
// Type Inference Context
// =============================================================================

/// Context for type inference, managing fresh variable generation and constraints
pub struct InferContext {
    /// Fresh type variable generator
    pub var_gen: TypeVarGen,

    /// Collected constraints during inference
    pub constraints: Vec<Constraint>,

    /// Current substitution
    pub subst: Subst,
}

impl InferContext {
    pub fn new() -> Self {
        Self {
            var_gen: TypeVarGen::new(),
            constraints: Vec::new(),
            subst: Subst::new(),
        }
    }

    /// Generate a fresh type variable
    pub fn fresh_var(&mut self) -> TypeVar {
        self.var_gen.fresh()
    }

    /// Generate a fresh type
    pub fn fresh_type(&mut self) -> Type {
        self.var_gen.fresh_type()
    }

    /// Generate a fresh named type variable
    pub fn fresh_named(&mut self, name: impl Into<String>) -> Type {
        self.var_gen.fresh_named_type(name)
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, left: Type, right: Type) {
        self.constraints.push(Constraint::new(left, right));
    }

    /// Solve all constraints and return the final substitution
    pub fn solve(&mut self) -> Result<Subst, InferError> {
        let mut subst = Subst::new();

        for constraint in &self.constraints {
            let left = subst.apply(&constraint.left);
            let right = subst.apply(&constraint.right);
            let s = unify(&left, &right)?;
            subst.extend(&s);
        }

        self.subst = subst.clone();
        Ok(subst)
    }

    /// Apply current substitution to a type
    pub fn apply(&self, ty: &Type) -> Type {
        self.subst.apply(ty)
    }

    /// Unify two types immediately (not deferred)
    pub fn unify(&mut self, t1: &Type, t2: &Type) -> Result<(), InferError> {
        let s = unify(t1, t2)?;
        self.subst.extend(&s);
        Ok(())
    }
}

impl Default for InferContext {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Instantiation and Generalization
// =============================================================================

/// Instantiate a polymorphic type with fresh type variables
///
/// Given a type scheme like `∀T, U. (T, U) -> T`, this replaces
/// the bound type variables with fresh ones.
pub fn instantiate(ctx: &mut InferContext, poly: &PolyType) -> Type {
    if poly.is_monomorphic() {
        return poly.ty.clone();
    }

    let mut subst = Subst::new();
    for param in &poly.type_params {
        let fresh = ctx.fresh_type();
        subst.insert(param.id, fresh);
    }

    subst.apply(&poly.ty)
}

/// Instantiate a polymorphic type with specific type arguments
pub fn instantiate_with(poly: &PolyType, type_args: &[Type]) -> Result<Type, InferError> {
    if poly.type_params.len() != type_args.len() {
        return Err(InferError::WrongTypeArgCount {
            expected: poly.type_params.len(),
            found: type_args.len(),
        });
    }

    let mut subst = Subst::new();
    for (param, arg) in poly.type_params.iter().zip(type_args.iter()) {
        subst.insert(param.id, arg.clone());
    }

    Ok(subst.apply(&poly.ty))
}

/// Generalize a type to a polymorphic type
///
/// Given a type with free variables, create a type scheme by
/// quantifying over variables not free in the environment.
pub fn generalize(env: &TypeEnv, ty: &Type) -> PolyType {
    let env_vars = env.free_type_vars();
    let ty_vars = ty.free_type_vars();

    // Variables to quantify = ty_vars - env_vars
    let to_quantify: Vec<_> = ty_vars.difference(&env_vars).copied().collect();

    if to_quantify.is_empty() {
        return PolyType::mono(ty.clone());
    }

    // Create type parameters for quantified variables
    let type_params: Vec<_> = to_quantify
        .iter()
        .enumerate()
        .map(|(i, id)| {
            // Generate a name like T, U, V, ...
            let name = (b'T' + (i as u8)) as char;
            TypeParam::new(name.to_string(), id.0)
        })
        .collect();

    PolyType::new(type_params, ty.clone())
}

/// Generalize with explicit type parameters (for user-defined generic functions)
pub fn generalize_with_params(type_params: Vec<TypeParam>, ty: Type) -> PolyType {
    PolyType::new(type_params, ty)
}

// =============================================================================
// Built-in Polymorphic Signatures
// =============================================================================

/// Standard library of polymorphic function signatures
pub fn builtin_poly_signatures(gen: &mut TypeVarGen) -> HashMap<String, PolyType> {
    let mut builtins = HashMap::new();

    // identity<T>(x: T) -> T
    {
        let t_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        builtins.insert(
            "identity".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id)],
                Type::function(vec![t.clone()], t),
            ),
        );
    }

    // first<T, U>(a: T, b: U) -> T
    {
        let t_id = gen.fresh().id.0;
        let u_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        let u = Type::named_var(u_id, "U");
        builtins.insert(
            "first".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id), TypeParam::new("U", u_id)],
                Type::function(vec![t.clone(), u], t),
            ),
        );
    }

    // second<T, U>(a: T, b: U) -> U
    {
        let t_id = gen.fresh().id.0;
        let u_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        let u = Type::named_var(u_id, "U");
        builtins.insert(
            "second".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id), TypeParam::new("U", u_id)],
                Type::function(vec![t, u.clone()], u),
            ),
        );
    }

    // map<T, U>(f: (T) -> U, xs: [T]) -> [U]
    {
        let t_id = gen.fresh().id.0;
        let u_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        let u = Type::named_var(u_id, "U");
        builtins.insert(
            "map".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id), TypeParam::new("U", u_id)],
                Type::function(
                    vec![Type::function(vec![t.clone()], u.clone()), Type::list(t)],
                    Type::list(u),
                ),
            ),
        );
    }

    // filter<T>(pred: (T) -> Bool, xs: [T]) -> [T]
    {
        let t_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        builtins.insert(
            "filter".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id)],
                Type::function(
                    vec![
                        Type::function(vec![t.clone()], Type::Bool),
                        Type::list(t.clone()),
                    ],
                    Type::list(t),
                ),
            ),
        );
    }

    // fold<T, U>(f: (U, T) -> U, init: U, xs: [T]) -> U
    {
        let t_id = gen.fresh().id.0;
        let u_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        let u = Type::named_var(u_id, "U");
        builtins.insert(
            "fold".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id), TypeParam::new("U", u_id)],
                Type::function(
                    vec![
                        Type::function(vec![u.clone(), t.clone()], u.clone()),
                        u.clone(),
                        Type::list(t),
                    ],
                    u,
                ),
            ),
        );
    }

    // zip<T, U>(xs: [T], ys: [U]) -> [(T, U)]
    {
        let t_id = gen.fresh().id.0;
        let u_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        let u = Type::named_var(u_id, "U");
        builtins.insert(
            "zip".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id), TypeParam::new("U", u_id)],
                Type::function(
                    vec![Type::list(t.clone()), Type::list(u.clone())],
                    Type::list(Type::tuple(vec![t, u])),
                ),
            ),
        );
    }

    // head<T>(xs: [T]) -> Option<T>
    {
        let t_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        builtins.insert(
            "head".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id)],
                Type::function(vec![Type::list(t.clone())], Type::option(t)),
            ),
        );
    }

    // tail<T>(xs: [T]) -> [T]
    {
        let t_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        builtins.insert(
            "tail".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id)],
                Type::function(vec![Type::list(t.clone())], Type::list(t)),
            ),
        );
    }

    // length<T>(xs: [T]) -> Int
    {
        let t_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        builtins.insert(
            "length".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id)],
                Type::function(vec![Type::list(t)], Type::Int),
            ),
        );
    }

    // reverse<T>(xs: [T]) -> [T]
    {
        let t_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        builtins.insert(
            "reverse".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id)],
                Type::function(vec![Type::list(t.clone())], Type::list(t)),
            ),
        );
    }

    // concat<T>(xs: [T], ys: [T]) -> [T]
    {
        let t_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        builtins.insert(
            "concat".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id)],
                Type::function(
                    vec![Type::list(t.clone()), Type::list(t.clone())],
                    Type::list(t),
                ),
            ),
        );
    }

    // flatten<T>(xss: [[T]]) -> [T]
    {
        let t_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        builtins.insert(
            "flatten".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id)],
                Type::function(vec![Type::list(Type::list(t.clone()))], Type::list(t)),
            ),
        );
    }

    // flatMap<T, U>(f: (T) -> [U], xs: [T]) -> [U]
    {
        let t_id = gen.fresh().id.0;
        let u_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        let u = Type::named_var(u_id, "U");
        builtins.insert(
            "flatMap".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id), TypeParam::new("U", u_id)],
                Type::function(
                    vec![
                        Type::function(vec![t.clone()], Type::list(u.clone())),
                        Type::list(t),
                    ],
                    Type::list(u),
                ),
            ),
        );
    }

    // find<T>(pred: (T) -> Bool, xs: [T]) -> Option<T>
    {
        let t_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        builtins.insert(
            "find".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id)],
                Type::function(
                    vec![
                        Type::function(vec![t.clone()], Type::Bool),
                        Type::list(t.clone()),
                    ],
                    Type::option(t),
                ),
            ),
        );
    }

    // some<T>(value: T) -> Option<T>
    {
        let t_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        builtins.insert(
            "some".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id)],
                Type::function(vec![t.clone()], Type::option(t)),
            ),
        );
    }

    // none<T>() -> Option<T>
    {
        let t_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        builtins.insert(
            "none".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id)],
                Type::function(vec![], Type::option(t)),
            ),
        );
    }

    // ok<T, E>(value: T) -> Result<T, E>
    {
        let t_id = gen.fresh().id.0;
        let e_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        let e = Type::named_var(e_id, "E");
        builtins.insert(
            "ok".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id), TypeParam::new("E", e_id)],
                Type::function(vec![t.clone()], Type::result(t, e)),
            ),
        );
    }

    // err<T, E>(error: E) -> Result<T, E>
    {
        let t_id = gen.fresh().id.0;
        let e_id = gen.fresh().id.0;
        let t = Type::named_var(t_id, "T");
        let e = Type::named_var(e_id, "E");
        builtins.insert(
            "err".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", t_id), TypeParam::new("E", e_id)],
                Type::function(vec![e.clone()], Type::result(t, e)),
            ),
        );
    }

    builtins
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instantiate_mono() {
        let mut ctx = InferContext::new();
        let poly = PolyType::mono(Type::Int);
        let inst = instantiate(&mut ctx, &poly);
        assert_eq!(inst, Type::Int);
    }

    #[test]
    fn test_instantiate_poly() {
        let mut ctx = InferContext::new();

        // ∀T. T -> T
        let poly = PolyType::new(
            vec![TypeParam::new("T", 0)],
            Type::function(vec![Type::var(0)], Type::var(0)),
        );

        let inst = instantiate(&mut ctx, &poly);

        // Should be a function with fresh type variables
        match inst {
            Type::Function { params, ret } => {
                assert_eq!(params.len(), 1);
                // Both should be the same fresh variable
                assert_eq!(params[0], *ret);
                // Should not be the original variable
                assert_ne!(params[0], Type::var(0));
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_instantiate_with() {
        // ∀T, U. (T, U) -> T
        let poly = PolyType::new(
            vec![TypeParam::new("T", 0), TypeParam::new("U", 1)],
            Type::function(vec![Type::var(0), Type::var(1)], Type::var(0)),
        );

        let inst = instantiate_with(&poly, &[Type::Int, Type::String]).unwrap();

        match inst {
            Type::Function { params, ret } => {
                assert_eq!(params[0], Type::Int);
                assert_eq!(params[1], Type::String);
                assert_eq!(*ret, Type::Int);
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_instantiate_with_wrong_count() {
        let poly = PolyType::new(
            vec![TypeParam::new("T", 0)],
            Type::function(vec![Type::var(0)], Type::var(0)),
        );

        let result = instantiate_with(&poly, &[Type::Int, Type::String]);
        assert!(matches!(result, Err(InferError::WrongTypeArgCount { .. })));
    }

    #[test]
    fn test_generalize() {
        let env = TypeEnv::new();

        // Type with free variables: T -> T
        let ty = Type::function(vec![Type::var(0)], Type::var(0));

        let poly = generalize(&env, &ty);

        assert_eq!(poly.type_params.len(), 1);
        assert!(!poly.is_monomorphic());
    }

    #[test]
    fn test_generalize_with_env_vars() {
        let mut env = TypeEnv::new();
        // x: T0 is in the environment
        env.extend("x".to_string(), PolyType::mono(Type::var(0)));

        // Type: T0 -> T1
        let ty = Type::function(vec![Type::var(0)], Type::var(1));

        let poly = generalize(&env, &ty);

        // Only T1 should be generalized (T0 is free in env)
        assert_eq!(poly.type_params.len(), 1);
    }

    #[test]
    fn test_type_env_extend() {
        let mut env = TypeEnv::new();
        env.extend("x".to_string(), PolyType::mono(Type::Int));

        assert!(env.lookup("x").is_some());
        assert!(env.lookup("y").is_none());
    }

    #[test]
    fn test_type_env_apply_subst() {
        let mut env = TypeEnv::new();
        env.extend("x".to_string(), PolyType::mono(Type::var(0)));

        let mut subst = Subst::new();
        subst.insert(TypeVarId(0), Type::Int);

        let new_env = env.apply_subst(&subst);
        let x_type = new_env.lookup("x").unwrap();
        assert_eq!(x_type.ty, Type::Int);
    }

    #[test]
    fn test_infer_context_constraints() {
        let mut ctx = InferContext::new();

        let t1 = ctx.fresh_type();
        let t2 = ctx.fresh_type();

        ctx.add_constraint(t1.clone(), Type::Int);
        ctx.add_constraint(t2.clone(), Type::list(t1.clone()));

        let subst = ctx.solve().unwrap();

        assert_eq!(subst.apply(&t1), Type::Int);
        assert_eq!(subst.apply(&t2), Type::list(Type::Int));
    }

    #[test]
    fn test_builtin_signatures() {
        let mut gen = TypeVarGen::new();
        let builtins = builtin_poly_signatures(&mut gen);

        // Check identity
        let identity = builtins.get("identity").unwrap();
        assert_eq!(identity.arity(), 1);

        // Check map
        let map = builtins.get("map").unwrap();
        assert_eq!(map.arity(), 2);

        // Check fold
        let fold = builtins.get("fold").unwrap();
        assert_eq!(fold.arity(), 2);
    }

    #[test]
    fn test_infer_context_unify() {
        let mut ctx = InferContext::new();

        let t = ctx.fresh_type();
        ctx.unify(&t, &Type::Int).unwrap();

        assert_eq!(ctx.apply(&t), Type::Int);
    }
}
