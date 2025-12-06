// Week 52: Parametric Polymorphism - Type Checker
//
// This module implements type checking for generic functions and expressions,
// integrating type inference with the generics system.

use super::ast::{
    BinaryOpAst, GenericBlock, GenericExpr, GenericFnDecl, GenericParam, GenericStmt, TypeBoundAst,
    TypeExprAst, TypeParamAst, UnaryOpAst,
};
use super::infer::{generalize, instantiate, InferContext, InferError, TypeEnv};
use super::mono::{MonoCollector, MonoContext};
use super::types::{PolyType, Subst, Type, TypeBound, TypeParam, TypeVarGen, TypeVarId};
use super::unify::unify;
use std::collections::HashMap;
use thiserror::Error;

// =============================================================================
// Type Check Errors
// =============================================================================

#[derive(Debug, Clone, Error, PartialEq)]
pub enum CheckError {
    #[error("type inference error: {0}")]
    InferError(#[from] InferError),

    #[error("undefined variable: {0}")]
    UndefinedVar(String),

    #[error("undefined function: {0}")]
    UndefinedFn(String),

    #[error("undefined type: {0}")]
    UndefinedType(String),

    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    #[error("not a function type: {0}")]
    NotAFunction(String),

    #[error("wrong number of type arguments for {name}: expected {expected}, found {found}")]
    WrongTypeArgCount {
        name: String,
        expected: usize,
        found: usize,
    },

    #[error("wrong number of arguments for {name}: expected {expected}, found {found}")]
    WrongArgCount {
        name: String,
        expected: usize,
        found: usize,
    },

    #[error("cannot apply binary operator {op} to types {left} and {right}")]
    InvalidBinaryOp {
        op: String,
        left: String,
        right: String,
    },

    #[error("cannot apply unary operator {op} to type {operand}")]
    InvalidUnaryOp { op: String, operand: String },

    #[error("bound not satisfied: type {ty} does not satisfy {bound}")]
    BoundNotSatisfied { ty: String, bound: String },

    #[error("duplicate type parameter: {0}")]
    DuplicateTypeParam(String),

    #[error("missing type annotation for parameter: {0}")]
    MissingParamType(String),

    #[error("missing return type annotation")]
    MissingReturnType,

    #[error("recursive type detected: {0}")]
    RecursiveType(String),

    #[error("if branches have different types: then={then_ty}, else={else_ty}")]
    IfBranchMismatch { then_ty: String, else_ty: String },
}

// =============================================================================
// Type Checker
// =============================================================================

/// Type checker for generic MedLang code
pub struct GenericTypeChecker {
    /// Type inference context
    infer_ctx: InferContext,
    /// Type environment
    env: TypeEnv,
    /// Function signatures
    fn_sigs: HashMap<String, PolyType>,
    /// Type definitions
    type_defs: HashMap<String, Type>,
    /// Monomorphization collector
    mono_collector: MonoCollector,
    /// Errors collected during checking
    errors: Vec<CheckError>,
}

impl GenericTypeChecker {
    pub fn new() -> Self {
        let mut checker = Self {
            infer_ctx: InferContext::new(),
            env: TypeEnv::new(),
            fn_sigs: HashMap::new(),
            type_defs: HashMap::new(),
            mono_collector: MonoCollector::new(),
            errors: Vec::new(),
        };

        // Add built-in types
        checker.add_builtin_types();
        // Add built-in functions
        checker.add_builtin_functions();

        checker
    }

    fn add_builtin_types(&mut self) {
        self.type_defs.insert("Int".to_string(), Type::Int);
        self.type_defs.insert("Float".to_string(), Type::Float);
        self.type_defs.insert("Bool".to_string(), Type::Bool);
        self.type_defs.insert("String".to_string(), Type::String);
        self.type_defs.insert("Unit".to_string(), Type::Unit);

        // Domain types
        self.type_defs.insert("Model".to_string(), Type::Model);
        self.type_defs
            .insert("Protocol".to_string(), Type::Protocol);
        self.type_defs.insert("Policy".to_string(), Type::Policy);
        self.type_defs
            .insert("EvidenceProgram".to_string(), Type::EvidenceProgram);
        self.type_defs
            .insert("EvidenceResult".to_string(), Type::EvidenceResult);
        self.type_defs
            .insert("SimulationResult".to_string(), Type::SimulationResult);
        self.type_defs
            .insert("FitResult".to_string(), Type::FitResult);
        self.type_defs
            .insert("SurrogateModel".to_string(), Type::SurrogateModel);
        self.type_defs
            .insert("RLPolicy".to_string(), Type::RLPolicy);

        // AD types
        self.type_defs.insert("Dual".to_string(), Type::Dual);
        self.type_defs.insert("DualVec".to_string(), Type::DualVec);
        self.type_defs.insert("DualRec".to_string(), Type::DualRec);
    }

    fn add_builtin_functions(&mut self) {
        use super::infer::builtin_poly_signatures;

        let mut gen = TypeVarGen::new();
        let builtins = builtin_poly_signatures(&mut gen);
        for (name, poly) in builtins {
            self.fn_sigs.insert(name, poly);
        }

        // Add non-generic builtins
        self.fn_sigs.insert(
            "print".to_string(),
            PolyType::mono(Type::function(vec![Type::String], Type::Unit)),
        );
    }

    /// Register a generic function
    pub fn register_function(&mut self, name: String, poly: PolyType) {
        self.fn_sigs.insert(name, poly);
    }

    /// Type check a generic function declaration
    pub fn check_fn_decl(&mut self, fn_decl: &GenericFnDecl) -> Result<PolyType, CheckError> {
        // 1. Check for duplicate type parameters
        let mut seen_params = HashMap::new();
        for tp in &fn_decl.type_params {
            if seen_params.contains_key(&tp.name) {
                return Err(CheckError::DuplicateTypeParam(tp.name.clone()));
            }
            seen_params.insert(tp.name.clone(), tp);
        }

        // 2. Create type parameters
        let type_params: Vec<TypeParam> = fn_decl
            .type_params
            .iter()
            .map(|tp| {
                let id = self.infer_ctx.var_gen.fresh().id.0;
                let bounds = tp.bounds.iter().map(|b| self.resolve_bound(b)).collect();
                TypeParam {
                    name: tp.name.clone(),
                    id: TypeVarId(id),
                    bounds,
                }
            })
            .collect();

        // 3. Build type param mapping
        let mut type_param_map: HashMap<String, Type> = HashMap::new();
        for tp in &type_params {
            type_param_map.insert(
                tp.name.clone(),
                Type::Var(super::types::TypeVar::named(tp.id.0, tp.name.clone())),
            );
        }

        // 4. Resolve parameter types
        let mut param_types = Vec::new();
        let mut local_env = self.env.clone();

        for param in &fn_decl.params {
            let param_ty = match &param.ty {
                Some(ty_expr) => self.resolve_type_expr(ty_expr, &type_param_map)?,
                None => {
                    return Err(CheckError::MissingParamType(param.name.clone()));
                }
            };
            param_types.push(param_ty.clone());
            local_env.extend(param.name.clone(), PolyType::mono(param_ty));
        }

        // 5. Resolve return type
        let ret_type = match &fn_decl.ret_type {
            Some(ty_expr) => self.resolve_type_expr(ty_expr, &type_param_map)?,
            None => Type::Unit, // Default to Unit if not specified
        };

        // 6. Type check body
        let old_env = std::mem::replace(&mut self.env, local_env);
        let body_type = self.check_block(&fn_decl.body, &type_param_map)?;
        self.env = old_env;

        // 7. Check body type matches return type
        let unified_ret = self.unify_types(&body_type, &ret_type)?;

        // 8. Build function type
        let fn_type = Type::function(param_types, unified_ret);

        // 9. Create polymorphic type
        let poly = PolyType::new(type_params, fn_type);

        // 10. Register the function
        self.fn_sigs.insert(fn_decl.name.clone(), poly.clone());

        Ok(poly)
    }

    /// Check a block of statements
    fn check_block(
        &mut self,
        block: &GenericBlock,
        type_param_map: &HashMap<String, Type>,
    ) -> Result<Type, CheckError> {
        let mut last_type = Type::Unit;

        for stmt in &block.stmts {
            last_type = self.check_stmt(stmt, type_param_map)?;
        }

        Ok(last_type)
    }

    /// Check a statement
    fn check_stmt(
        &mut self,
        stmt: &GenericStmt,
        type_param_map: &HashMap<String, Type>,
    ) -> Result<Type, CheckError> {
        match stmt {
            GenericStmt::Let { name, ty, expr } => {
                let expr_type = self.check_expr(expr, type_param_map)?;

                let final_type = if let Some(ty_expr) = ty {
                    let declared = self.resolve_type_expr(ty_expr, type_param_map)?;
                    self.unify_types(&expr_type, &declared)?
                } else {
                    expr_type
                };

                // Add to environment
                let generalized = generalize(&self.env, &final_type);
                self.env.extend(name.clone(), generalized);

                Ok(Type::Unit)
            }
            GenericStmt::Expr(expr) => self.check_expr(expr, type_param_map),
            GenericStmt::Return(expr) => match expr {
                Some(e) => self.check_expr(e, type_param_map),
                None => Ok(Type::Unit),
            },
        }
    }

    /// Check an expression
    pub fn check_expr(
        &mut self,
        expr: &GenericExpr,
        type_param_map: &HashMap<String, Type>,
    ) -> Result<Type, CheckError> {
        match expr {
            GenericExpr::IntLit(_) => Ok(Type::Int),
            GenericExpr::FloatLit(_) => Ok(Type::Float),
            GenericExpr::BoolLit(_) => Ok(Type::Bool),
            GenericExpr::StringLit(_) => Ok(Type::String),

            GenericExpr::Var(name) => {
                // Check type parameters first
                if let Some(ty) = type_param_map.get(name) {
                    return Ok(ty.clone());
                }

                // Check local environment
                if let Some(poly) = self.env.lookup(name) {
                    return Ok(instantiate(&mut self.infer_ctx, poly));
                }

                Err(CheckError::UndefinedVar(name.clone()))
            }

            GenericExpr::Call {
                callee,
                type_args,
                args,
            } => self.check_call(callee, type_args, args, type_param_map),

            GenericExpr::FieldAccess { target, field } => {
                let target_type = self.check_expr(target, type_param_map)?;
                match target_type {
                    Type::Record(fields) => fields
                        .get(field)
                        .cloned()
                        .ok_or_else(|| CheckError::UndefinedVar(field.clone())),
                    _ => Err(CheckError::TypeMismatch {
                        expected: "record".to_string(),
                        found: target_type.to_string(),
                    }),
                }
            }

            GenericExpr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_type = self.check_expr(cond, type_param_map)?;
                self.unify_types(&cond_type, &Type::Bool)?;

                let then_type = self.check_expr(then_branch, type_param_map)?;
                let else_type = self.check_expr(else_branch, type_param_map)?;

                self.unify_types(&then_type, &else_type)
                    .map_err(|_| CheckError::IfBranchMismatch {
                        then_ty: then_type.to_string(),
                        else_ty: else_type.to_string(),
                    })
            }

            GenericExpr::Block(block) => self.check_block(block, type_param_map),

            GenericExpr::Record(fields) => {
                let mut field_types = HashMap::new();
                for (name, expr) in fields {
                    let ty = self.check_expr(expr, type_param_map)?;
                    field_types.insert(name.clone(), ty);
                }
                Ok(Type::Record(field_types))
            }

            GenericExpr::List(elements) => {
                if elements.is_empty() {
                    // Empty list has polymorphic type [T]
                    let elem_type = self.infer_ctx.fresh_type();
                    return Ok(Type::list(elem_type));
                }

                let first_type = self.check_expr(&elements[0], type_param_map)?;
                for elem in elements.iter().skip(1) {
                    let elem_type = self.check_expr(elem, type_param_map)?;
                    self.unify_types(&first_type, &elem_type)?;
                }

                Ok(Type::list(first_type))
            }

            GenericExpr::Tuple(elements) => {
                let types: Result<Vec<_>, _> = elements
                    .iter()
                    .map(|e| self.check_expr(e, type_param_map))
                    .collect();
                Ok(Type::tuple(types?))
            }

            GenericExpr::Lambda {
                params,
                ret_type,
                body,
            } => {
                let mut local_env = self.env.clone();
                let mut param_types = Vec::new();

                for param in params {
                    let param_ty = match &param.ty {
                        Some(ty_expr) => self.resolve_type_expr(ty_expr, type_param_map)?,
                        None => self.infer_ctx.fresh_type(),
                    };
                    param_types.push(param_ty.clone());
                    local_env.extend(param.name.clone(), PolyType::mono(param_ty));
                }

                let old_env = std::mem::replace(&mut self.env, local_env);
                let body_type = self.check_expr(body, type_param_map)?;
                self.env = old_env;

                let final_ret = match ret_type {
                    Some(ty_expr) => {
                        let declared = self.resolve_type_expr(ty_expr, type_param_map)?;
                        self.unify_types(&body_type, &declared)?
                    }
                    None => body_type,
                };

                Ok(Type::function(param_types, final_ret))
            }

            GenericExpr::Binary { op, left, right } => {
                let left_type = self.check_expr(left, type_param_map)?;
                let right_type = self.check_expr(right, type_param_map)?;
                self.check_binary_op(*op, &left_type, &right_type)
            }

            GenericExpr::Unary { op, operand } => {
                let operand_type = self.check_expr(operand, type_param_map)?;
                self.check_unary_op(*op, &operand_type)
            }

            GenericExpr::Ascription { expr, ty } => {
                let expr_type = self.check_expr(expr, type_param_map)?;
                let ascribed = self.resolve_type_expr(ty, type_param_map)?;
                self.unify_types(&expr_type, &ascribed)
            }
        }
    }

    /// Check a function call
    fn check_call(
        &mut self,
        callee: &GenericExpr,
        type_args: &[TypeExprAst],
        args: &[GenericExpr],
        type_param_map: &HashMap<String, Type>,
    ) -> Result<Type, CheckError> {
        // Get the callee name
        let fn_name = match callee {
            GenericExpr::Var(name) => name.clone(),
            _ => {
                // For non-variable callees, check as expression
                let callee_type = self.check_expr(callee, type_param_map)?;
                return self.check_call_on_type(callee_type, args, type_param_map);
            }
        };

        // First check if it's a local variable with function type (e.g., a parameter)
        if let Some(local_poly) = self.env.lookup(&fn_name) {
            let local_type = instantiate(&mut self.infer_ctx, local_poly);
            if matches!(local_type, Type::Function { .. }) {
                return self.check_call_on_type(local_type, args, type_param_map);
            }
        }

        // Look up function signature
        let poly = self
            .fn_sigs
            .get(&fn_name)
            .ok_or_else(|| CheckError::UndefinedFn(fn_name.clone()))?
            .clone();

        // Resolve explicit type arguments
        let resolved_type_args: Vec<Type> = type_args
            .iter()
            .map(|ta| self.resolve_type_expr(ta, type_param_map))
            .collect::<Result<_, _>>()?;

        // Instantiate the function type
        let fn_type = if !resolved_type_args.is_empty() {
            // Explicit type arguments
            if resolved_type_args.len() != poly.type_params.len() {
                return Err(CheckError::WrongTypeArgCount {
                    name: fn_name,
                    expected: poly.type_params.len(),
                    found: resolved_type_args.len(),
                });
            }

            // Build substitution from explicit args
            let mut subst = Subst::new();
            for (param, arg) in poly.type_params.iter().zip(resolved_type_args.iter()) {
                subst.insert(param.id, arg.clone());
            }

            // Record for monomorphization
            if resolved_type_args.iter().all(|t| t.is_monomorphic()) {
                self.mono_collector
                    .record(&fn_name, resolved_type_args.clone());
            }

            subst.apply(&poly.ty)
        } else {
            // Infer type arguments
            instantiate(&mut self.infer_ctx, &poly)
        };

        // Extract parameter and return types
        let (param_types, ret_type) = match fn_type {
            Type::Function { params, ret } => (params, *ret),
            _ => {
                return Err(CheckError::NotAFunction(fn_type.to_string()));
            }
        };

        // Check argument count
        if args.len() != param_types.len() {
            return Err(CheckError::WrongArgCount {
                name: fn_name,
                expected: param_types.len(),
                found: args.len(),
            });
        }

        // Check argument types
        for (arg, expected_ty) in args.iter().zip(param_types.iter()) {
            let arg_type = self.check_expr(arg, type_param_map)?;
            self.unify_types(&arg_type, expected_ty)?;
        }

        Ok(self.infer_ctx.apply(&ret_type))
    }

    /// Check a call on a function type value
    fn check_call_on_type(
        &mut self,
        fn_type: Type,
        args: &[GenericExpr],
        type_param_map: &HashMap<String, Type>,
    ) -> Result<Type, CheckError> {
        let (param_types, ret_type) = match fn_type {
            Type::Function { params, ret } => (params, *ret),
            _ => {
                return Err(CheckError::NotAFunction(fn_type.to_string()));
            }
        };

        if args.len() != param_types.len() {
            return Err(CheckError::WrongArgCount {
                name: "<lambda>".to_string(),
                expected: param_types.len(),
                found: args.len(),
            });
        }

        for (arg, expected_ty) in args.iter().zip(param_types.iter()) {
            let arg_type = self.check_expr(arg, type_param_map)?;
            self.unify_types(&arg_type, expected_ty)?;
        }

        Ok(self.infer_ctx.apply(&ret_type))
    }

    /// Check binary operation
    fn check_binary_op(
        &mut self,
        op: BinaryOpAst,
        left: &Type,
        right: &Type,
    ) -> Result<Type, CheckError> {
        match op {
            // Arithmetic: both must be numeric, result is numeric
            BinaryOpAst::Add | BinaryOpAst::Sub | BinaryOpAst::Mul | BinaryOpAst::Div => {
                self.unify_types(left, right)?;
                match left {
                    Type::Int | Type::Float | Type::Dual => Ok(left.clone()),
                    _ => Err(CheckError::InvalidBinaryOp {
                        op: op.to_string(),
                        left: left.to_string(),
                        right: right.to_string(),
                    }),
                }
            }

            BinaryOpAst::Mod => {
                self.unify_types(left, right)?;
                match left {
                    Type::Int => Ok(Type::Int),
                    _ => Err(CheckError::InvalidBinaryOp {
                        op: op.to_string(),
                        left: left.to_string(),
                        right: right.to_string(),
                    }),
                }
            }

            BinaryOpAst::Pow => {
                // left^right, both must be numeric
                match (left, right) {
                    (Type::Int, Type::Int) => Ok(Type::Int),
                    (Type::Float, _) | (_, Type::Float) => Ok(Type::Float),
                    (Type::Dual, _) | (_, Type::Dual) => Ok(Type::Dual),
                    _ => Err(CheckError::InvalidBinaryOp {
                        op: op.to_string(),
                        left: left.to_string(),
                        right: right.to_string(),
                    }),
                }
            }

            // Logical: both must be Bool
            BinaryOpAst::And | BinaryOpAst::Or => {
                self.unify_types(left, &Type::Bool)?;
                self.unify_types(right, &Type::Bool)?;
                Ok(Type::Bool)
            }

            // Comparison: both must be same type, result is Bool
            BinaryOpAst::Eq | BinaryOpAst::Ne => {
                self.unify_types(left, right)?;
                Ok(Type::Bool)
            }

            BinaryOpAst::Lt | BinaryOpAst::Le | BinaryOpAst::Gt | BinaryOpAst::Ge => {
                self.unify_types(left, right)?;
                // Must be orderable
                match left {
                    Type::Int | Type::Float | Type::String => Ok(Type::Bool),
                    _ => Err(CheckError::InvalidBinaryOp {
                        op: op.to_string(),
                        left: left.to_string(),
                        right: right.to_string(),
                    }),
                }
            }
        }
    }

    /// Check unary operation
    fn check_unary_op(&mut self, op: UnaryOpAst, operand: &Type) -> Result<Type, CheckError> {
        match op {
            UnaryOpAst::Neg => match operand {
                Type::Int | Type::Float | Type::Dual => Ok(operand.clone()),
                _ => Err(CheckError::InvalidUnaryOp {
                    op: op.to_string(),
                    operand: operand.to_string(),
                }),
            },
            UnaryOpAst::Not => {
                self.unify_types(operand, &Type::Bool)?;
                Ok(Type::Bool)
            }
        }
    }

    /// Resolve a type expression to a Type
    fn resolve_type_expr(
        &mut self,
        ty_expr: &TypeExprAst,
        type_param_map: &HashMap<String, Type>,
    ) -> Result<Type, CheckError> {
        match ty_expr {
            TypeExprAst::Simple(name) => {
                // Check type params first
                if let Some(ty) = type_param_map.get(name) {
                    return Ok(ty.clone());
                }
                // Then check type definitions
                self.type_defs
                    .get(name)
                    .cloned()
                    .ok_or_else(|| CheckError::UndefinedType(name.clone()))
            }

            TypeExprAst::Var(name) => type_param_map
                .get(name)
                .cloned()
                .ok_or_else(|| CheckError::UndefinedType(name.clone())),

            TypeExprAst::Function { params, ret } => {
                let param_types: Result<Vec<_>, _> = params
                    .iter()
                    .map(|p| self.resolve_type_expr(p, type_param_map))
                    .collect();
                let ret_type = self.resolve_type_expr(ret, type_param_map)?;
                Ok(Type::function(param_types?, ret_type))
            }

            TypeExprAst::List(elem) => {
                let elem_type = self.resolve_type_expr(elem, type_param_map)?;
                Ok(Type::list(elem_type))
            }

            TypeExprAst::Option(inner) => {
                let inner_type = self.resolve_type_expr(inner, type_param_map)?;
                Ok(Type::option(inner_type))
            }

            TypeExprAst::Result { ok, err } => {
                let ok_type = self.resolve_type_expr(ok, type_param_map)?;
                let err_type = self.resolve_type_expr(err, type_param_map)?;
                Ok(Type::result(ok_type, err_type))
            }

            TypeExprAst::Tuple(elems) => {
                let elem_types: Result<Vec<_>, _> = elems
                    .iter()
                    .map(|e| self.resolve_type_expr(e, type_param_map))
                    .collect();
                Ok(Type::tuple(elem_types?))
            }

            TypeExprAst::Record(fields) => {
                let mut field_types = HashMap::new();
                for (name, ty) in fields {
                    let resolved = self.resolve_type_expr(ty, type_param_map)?;
                    field_types.insert(name.clone(), resolved);
                }
                Ok(Type::Record(field_types))
            }

            TypeExprAst::App { constructor, args } => {
                let arg_types: Result<Vec<_>, _> = args
                    .iter()
                    .map(|a| self.resolve_type_expr(a, type_param_map))
                    .collect();
                Ok(Type::app(constructor.clone(), arg_types?))
            }

            TypeExprAst::Unit => Ok(Type::Unit),

            TypeExprAst::Infer => Ok(self.infer_ctx.fresh_type()),
        }
    }

    /// Resolve a type bound
    fn resolve_bound(&self, bound: &TypeBoundAst) -> TypeBound {
        match bound {
            TypeBoundAst::Trait(name) => TypeBound::Trait(name.clone()),
            TypeBoundAst::Num => TypeBound::Num,
            TypeBoundAst::Ord => TypeBound::Ord,
            TypeBoundAst::Eq => TypeBound::Eq,
            TypeBoundAst::Copy => TypeBound::Copy,
        }
    }

    /// Unify two types
    fn unify_types(&mut self, t1: &Type, t2: &Type) -> Result<Type, CheckError> {
        let t1_applied = self.infer_ctx.apply(t1);
        let t2_applied = self.infer_ctx.apply(t2);

        let subst = unify(&t1_applied, &t2_applied).map_err(|e| CheckError::TypeMismatch {
            expected: t2_applied.to_string(),
            found: t1_applied.to_string(),
        })?;

        self.infer_ctx.subst.extend(&subst);

        Ok(self.infer_ctx.apply(&t1_applied))
    }

    /// Get collected monomorphizations
    pub fn mono_collector(&self) -> &MonoCollector {
        &self.mono_collector
    }

    /// Get the type environment
    pub fn env(&self) -> &TypeEnv {
        &self.env
    }

    /// Get function signatures
    pub fn fn_sigs(&self) -> &HashMap<String, PolyType> {
        &self.fn_sigs
    }

    /// Get errors
    pub fn errors(&self) -> &[CheckError] {
        &self.errors
    }

    /// Take errors
    pub fn take_errors(&mut self) -> Vec<CheckError> {
        std::mem::take(&mut self.errors)
    }
}

impl Default for GenericTypeChecker {
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
    fn test_check_literal_types() {
        let mut checker = GenericTypeChecker::new();
        let type_param_map = HashMap::new();

        let int_type = checker
            .check_expr(&GenericExpr::int(42), &type_param_map)
            .unwrap();
        assert_eq!(int_type, Type::Int);

        let float_type = checker
            .check_expr(&GenericExpr::float(3.14), &type_param_map)
            .unwrap();
        assert_eq!(float_type, Type::Float);

        let bool_type = checker
            .check_expr(&GenericExpr::bool_val(true), &type_param_map)
            .unwrap();
        assert_eq!(bool_type, Type::Bool);

        let string_type = checker
            .check_expr(&GenericExpr::string("hello"), &type_param_map)
            .unwrap();
        assert_eq!(string_type, Type::String);
    }

    #[test]
    fn test_check_binary_ops() {
        let mut checker = GenericTypeChecker::new();
        let type_param_map = HashMap::new();

        // Int + Int = Int
        let add_expr = GenericExpr::Binary {
            op: BinaryOpAst::Add,
            left: Box::new(GenericExpr::int(1)),
            right: Box::new(GenericExpr::int(2)),
        };
        let add_type = checker.check_expr(&add_expr, &type_param_map).unwrap();
        assert_eq!(add_type, Type::Int);

        // Int < Int = Bool
        let lt_expr = GenericExpr::Binary {
            op: BinaryOpAst::Lt,
            left: Box::new(GenericExpr::int(1)),
            right: Box::new(GenericExpr::int(2)),
        };
        let lt_type = checker.check_expr(&lt_expr, &type_param_map).unwrap();
        assert_eq!(lt_type, Type::Bool);
    }

    #[test]
    fn test_check_if_expr() {
        let mut checker = GenericTypeChecker::new();
        let type_param_map = HashMap::new();

        let if_expr = GenericExpr::if_expr(
            GenericExpr::bool_val(true),
            GenericExpr::int(1),
            GenericExpr::int(2),
        );

        let if_type = checker.check_expr(&if_expr, &type_param_map).unwrap();
        assert_eq!(if_type, Type::Int);
    }

    #[test]
    fn test_check_if_branch_mismatch() {
        let mut checker = GenericTypeChecker::new();
        let type_param_map = HashMap::new();

        let if_expr = GenericExpr::if_expr(
            GenericExpr::bool_val(true),
            GenericExpr::int(1),
            GenericExpr::string("two"),
        );

        let result = checker.check_expr(&if_expr, &type_param_map);
        assert!(result.is_err());
    }

    #[test]
    fn test_check_list() {
        let mut checker = GenericTypeChecker::new();
        let type_param_map = HashMap::new();

        let list_expr = GenericExpr::List(vec![
            GenericExpr::int(1),
            GenericExpr::int(2),
            GenericExpr::int(3),
        ]);

        let list_type = checker.check_expr(&list_expr, &type_param_map).unwrap();
        assert_eq!(list_type, Type::list(Type::Int));
    }

    #[test]
    fn test_check_lambda() {
        let mut checker = GenericTypeChecker::new();
        let type_param_map = HashMap::new();

        let lambda = GenericExpr::Lambda {
            params: vec![GenericParam::typed("x", TypeExprAst::simple("Int"))],
            ret_type: None,
            body: Box::new(GenericExpr::var("x")),
        };

        let lambda_type = checker.check_expr(&lambda, &type_param_map).unwrap();
        match lambda_type {
            Type::Function { params, ret } => {
                assert_eq!(params[0], Type::Int);
                assert_eq!(*ret, Type::Int);
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_check_generic_fn_decl() {
        let mut checker = GenericTypeChecker::new();

        // fn identity<T>(x: T) -> T { x }
        let fn_decl = GenericFnDecl::new("identity")
            .with_type_params(vec![TypeParamAst::new("T")])
            .with_params(vec![GenericParam::typed("x", TypeExprAst::var("T"))])
            .with_ret_type(TypeExprAst::var("T"))
            .with_body(GenericBlock::new(vec![GenericStmt::Expr(
                GenericExpr::var("x"),
            )]));

        let poly = checker.check_fn_decl(&fn_decl).unwrap();
        assert_eq!(poly.arity(), 1);
        assert!(!poly.is_monomorphic());
    }

    #[test]
    fn test_check_duplicate_type_param() {
        let mut checker = GenericTypeChecker::new();

        let fn_decl = GenericFnDecl::new("foo")
            .with_type_params(vec![TypeParamAst::new("T"), TypeParamAst::new("T")]);

        let result = checker.check_fn_decl(&fn_decl);
        assert!(matches!(result, Err(CheckError::DuplicateTypeParam(_))));
    }

    #[test]
    fn test_check_undefined_var() {
        let mut checker = GenericTypeChecker::new();
        let type_param_map = HashMap::new();

        let result = checker.check_expr(&GenericExpr::var("undefined"), &type_param_map);
        assert!(matches!(result, Err(CheckError::UndefinedVar(_))));
    }

    #[test]
    fn test_builtin_function_call() {
        let mut checker = GenericTypeChecker::new();
        let type_param_map = HashMap::new();

        // identity<Int>(42)
        let call = GenericExpr::call_generic(
            GenericExpr::var("identity"),
            vec![TypeExprAst::simple("Int")],
            vec![GenericExpr::int(42)],
        );

        let result_type = checker.check_expr(&call, &type_param_map).unwrap();
        assert_eq!(result_type, Type::Int);
    }

    #[test]
    fn test_mono_collection() {
        let mut checker = GenericTypeChecker::new();
        let type_param_map = HashMap::new();

        // Call identity with Int
        let call1 = GenericExpr::call_generic(
            GenericExpr::var("identity"),
            vec![TypeExprAst::simple("Int")],
            vec![GenericExpr::int(42)],
        );
        checker.check_expr(&call1, &type_param_map).unwrap();

        // Call identity with String
        let call2 = GenericExpr::call_generic(
            GenericExpr::var("identity"),
            vec![TypeExprAst::simple("String")],
            vec![GenericExpr::string("hello")],
        );
        checker.check_expr(&call2, &type_param_map).unwrap();

        // Check mono collector
        let instances = checker.mono_collector().get("identity").unwrap();
        assert_eq!(instances.len(), 2);
    }
}
