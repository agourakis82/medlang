// Week 26: Type Checker for L₀ (Core Language)
//
// Implements static type checking for MedLang's host coordination language.

use crate::ast::core_lang::{Block, Expr, FnDef, Ident, LetDecl, Stmt};
use crate::types::core_lang::{resolve_type_ann, CoreType, TypedFnSig};
use std::collections::HashMap;

// =============================================================================
// Type Errors
// =============================================================================

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TypeError {
    #[error("unknown variable `{0}`")]
    UnknownVar(String),

    #[error("unknown function `{0}`")]
    UnknownFn(String),

    #[error("type mismatch: expected {expected}, found {found}")]
    Mismatch { expected: String, found: String },

    #[error("not a function: found type {0}")]
    NotAFunction(String),

    #[error("arity mismatch for function `{fn_name}`: expected {expected} args, found {found}")]
    ArityMismatch {
        fn_name: String,
        expected: usize,
        found: usize,
    },

    #[error("condition must be Bool, found {0}")]
    CondNotBool(String),

    #[error("field `{field}` not found in record type")]
    NoSuchField { field: String },

    #[error("not a record type: {0}")]
    NotARecord(String),

    #[error("missing return type annotation on function `{0}`")]
    MissingReturnType(String),

    #[error("missing parameter type annotation on parameter `{param}` in function `{fn_name}`")]
    MissingParamType { fn_name: String, param: String },

    #[error("expected {expected}, but function body returns {found}")]
    ReturnTypeMismatch { expected: String, found: String },

    // Week 27: Enum type errors
    #[error("unknown enum `{0}`")]
    UnknownEnum(String),

    #[error("unknown variant `{variant_name}` in enum `{enum_name}`: {message}")]
    UnknownEnumVariant {
        enum_name: String,
        variant_name: String,
        message: String,
    },

    #[error("match scrutinee must be an enum type, found {found}")]
    MatchNonEnum { found: String },

    #[error("match arm enum mismatch: scrutinee is `{scrutinee_enum}`, but arm pattern uses `{arm_enum}`")]
    MatchEnumMismatch {
        scrutinee_enum: String,
        arm_enum: String,
    },

    #[error("match arm type mismatch: expected {expected}, found {found}")]
    MatchArmTypeMismatch { expected: String, found: String },

    #[error("non-exhaustive match on enum `{enum_name}`: missing variants [{missing_variants:?}]")]
    NonExhaustiveMatch {
        enum_name: String,
        missing_variants: Vec<String>,
    },
}

// =============================================================================
// Domain Environment
// =============================================================================

/// Environment tracking domain symbols (models, protocols, evidence programs, etc.)
#[derive(Debug, Clone, Default)]
pub struct DomainEnv {
    pub evidence_programs: HashMap<String, CoreType>,
    pub models: HashMap<String, CoreType>,
    pub protocols: HashMap<String, CoreType>,
    pub policies: HashMap<String, CoreType>,
}

impl DomainEnv {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_evidence_program(&mut self, name: String) {
        self.evidence_programs
            .insert(name, CoreType::EvidenceProgram);
    }

    pub fn add_model(&mut self, name: String) {
        self.models.insert(name, CoreType::Model);
    }

    pub fn add_protocol(&mut self, name: String) {
        self.protocols.insert(name, CoreType::Protocol);
    }

    pub fn add_policy(&mut self, name: String) {
        self.policies.insert(name, CoreType::Policy);
    }

    pub fn lookup(&self, name: &str) -> Option<CoreType> {
        if let Some(ty) = self.evidence_programs.get(name) {
            return Some(ty.clone());
        }
        if let Some(ty) = self.models.get(name) {
            return Some(ty.clone());
        }
        if let Some(ty) = self.protocols.get(name) {
            return Some(ty.clone());
        }
        if let Some(ty) = self.policies.get(name) {
            return Some(ty.clone());
        }
        None
    }
}

// =============================================================================
// Function Environment
// =============================================================================

/// Environment tracking function signatures (user-defined + built-ins)
#[derive(Debug, Clone)]
pub struct FnEnv {
    pub user_fns: HashMap<String, TypedFnSig>,
    pub builtin_fns: HashMap<String, TypedFnSig>,
    pub domain_env: DomainEnv,
}

impl FnEnv {
    pub fn new(domain_env: DomainEnv) -> Self {
        Self {
            user_fns: HashMap::new(),
            builtin_fns: build_builtin_signatures(),
            domain_env,
        }
    }

    pub fn lookup_fn(&self, name: &str) -> Result<TypedFnSig, TypeError> {
        if let Some(sig) = self.user_fns.get(name) {
            return Ok(sig.clone());
        }
        if let Some(sig) = self.builtin_fns.get(name) {
            return Ok(sig.clone());
        }
        Err(TypeError::UnknownFn(name.to_string()))
    }

    pub fn lookup_domain_symbol(&self, name: &str) -> Option<CoreType> {
        self.domain_env.lookup(name)
    }

    pub fn add_user_fn(&mut self, name: String, sig: TypedFnSig) {
        self.user_fns.insert(name, sig);
    }
}

/// Build built-in function signatures
fn build_builtin_signatures() -> HashMap<String, TypedFnSig> {
    use CoreType::*;

    let mut builtins = HashMap::new();

    // run_evidence(ev: EvidenceProgram, backend: String) -> EvidenceResult
    builtins.insert(
        "run_evidence".to_string(),
        TypedFnSig::new(vec![EvidenceProgram, String], EvidenceResult),
    );

    // export_results(res: EvidenceResult, path: String) -> Unit
    builtins.insert(
        "export_results".to_string(),
        TypedFnSig::new(vec![EvidenceResult, String], Unit),
    );

    // print(msg: String) -> Unit
    builtins.insert("print".to_string(), TypedFnSig::new(vec![String], Unit));

    // run_simulation(protocol: Protocol, n_subjects: Int) -> SimulationResult
    builtins.insert(
        "run_simulation".to_string(),
        TypedFnSig::new(vec![Protocol, Int], SimulationResult),
    );

    // fit_model(model: Model, data_path: String) -> FitResult
    builtins.insert(
        "fit_model".to_string(),
        TypedFnSig::new(vec![Model, String], FitResult),
    );

    // Week 29: First-class surrogate built-ins
    // Note: BackendKind will be Enum("BackendKind") when we add enum support to builtins
    // For now we use a placeholder approach - this will be refined when enums are fully integrated

    // train_surrogate(ev: EvidenceProgram, cfg: Record) -> SurrogateModel
    // cfg should be SurrogateTrainConfig but we use generic Record for now
    builtins.insert(
        "train_surrogate".to_string(),
        TypedFnSig::new(
            vec![
                EvidenceProgram,
                Record(vec![
                    ("n_train".to_string(), Int),
                    ("backend".to_string(), String), // Will become Enum("BackendKind")
                    ("seed".to_string(), Int),
                    ("max_epochs".to_string(), Int),
                    ("batch_size".to_string(), Int),
                ]),
            ],
            SurrogateModel,
        ),
    );

    // run_evidence_typed(ev: EvidenceProgram, backend: String) -> EvidenceResult
    // Updated version of run_evidence that will take BackendKind enum instead of String
    builtins.insert(
        "run_evidence_typed".to_string(),
        TypedFnSig::new(
            vec![EvidenceProgram, String], // String will become Enum("BackendKind")
            EvidenceResult,
        ),
    );

    // run_evidence_with_surrogate(ev: EvidenceProgram, s: SurrogateModel, backend: String) -> EvidenceResult
    builtins.insert(
        "run_evidence_with_surrogate".to_string(),
        TypedFnSig::new(
            vec![EvidenceProgram, SurrogateModel, String], // String will become Enum("BackendKind")
            EvidenceResult,
        ),
    );

    // Week 30: Surrogate evaluation built-in
    // evaluate_surrogate(ev: EvidenceProgram, surr: SurrogateModel, cfg: SurrogateEvalConfig) -> SurrogateEvalReport
    use crate::types::core_lang::{
        build_surrogate_eval_cfg_type, build_surrogate_eval_report_type,
    };
    builtins.insert(
        "evaluate_surrogate".to_string(),
        TypedFnSig::new(
            vec![
                EvidenceProgram,
                SurrogateModel,
                build_surrogate_eval_cfg_type(),
            ],
            build_surrogate_eval_report_type(),
        ),
    );

    builtins
}

// =============================================================================
// Variable Environment
// =============================================================================

/// Variable information in current scope
#[derive(Debug, Clone)]
pub struct VarInfo {
    pub ty: CoreType,
}

/// Type environment for expressions and statements
#[derive(Debug, Clone)]
pub struct TypeEnv<'a> {
    pub vars: HashMap<String, VarInfo>,
    pub fn_env: &'a FnEnv,
}

impl<'a> TypeEnv<'a> {
    pub fn new(fn_env: &'a FnEnv) -> Self {
        Self {
            vars: HashMap::new(),
            fn_env,
        }
    }

    pub fn add_var(&mut self, name: String, ty: CoreType) {
        self.vars.insert(name, VarInfo { ty });
    }

    pub fn lookup_var(&self, name: &str) -> Option<&VarInfo> {
        self.vars.get(name)
    }
}

// =============================================================================
// Type Checking Functions
// =============================================================================

/// Type check a function definition
pub fn typecheck_fn(fn_def: &FnDef, fn_env: &FnEnv) -> Result<TypedFnSig, TypeError> {
    // 1. Resolve parameter types
    let mut param_types = Vec::new();
    let mut var_env = HashMap::new();

    for p in &fn_def.params {
        let ann = p.ty.as_ref().ok_or_else(|| TypeError::MissingParamType {
            fn_name: fn_def.name.clone(),
            param: p.name.clone(),
        })?;
        let ty = resolve_type_ann(ann);
        param_types.push(ty.clone());
        var_env.insert(p.name.clone(), VarInfo { ty });
    }

    // 2. Resolve return type
    let ret_ann = fn_def
        .ret_type
        .as_ref()
        .ok_or_else(|| TypeError::MissingReturnType(fn_def.name.clone()))?;
    let ret_type = resolve_type_ann(ret_ann);

    // 3. Type check body
    let mut env = TypeEnv {
        vars: var_env,
        fn_env,
    };

    let body_ty = typecheck_block(&mut env, &fn_def.body)?;

    // 4. Check that body type matches declared return type
    // For now, allow Unit body for any return type (implicit return)
    if body_ty != ret_type && body_ty != CoreType::Unit {
        return Err(TypeError::ReturnTypeMismatch {
            expected: ret_type.as_str(),
            found: body_ty.as_str(),
        });
    }

    Ok(TypedFnSig::new(param_types, ret_type))
}

/// Type check a block of statements
pub fn typecheck_block(env: &mut TypeEnv, block: &Block) -> Result<CoreType, TypeError> {
    let mut last_ty = CoreType::Unit;

    for stmt in &block.stmts {
        match stmt {
            Stmt::Let(let_decl) => {
                last_ty = typecheck_let(env, let_decl)?;
            }
            Stmt::Assert(assert_stmt) => {
                // Week 28: Type check assert statements
                // The condition must be Bool
                let cond_ty = typecheck_expr(env, &assert_stmt.condition)?;
                if cond_ty != CoreType::Bool {
                    return Err(TypeError::Mismatch {
                        expected: "Bool",
                        found: cond_ty.as_str(),
                    });
                }
                last_ty = CoreType::Unit;
            }
            Stmt::Expr(expr) => {
                last_ty = typecheck_expr(env, expr)?;
            }
        }
    }

    Ok(last_ty)
}

/// Type check a let declaration
fn typecheck_let(env: &mut TypeEnv, let_decl: &LetDecl) -> Result<CoreType, TypeError> {
    let expr_ty = typecheck_expr(env, &let_decl.expr)?;

    // If type annotation provided, check compatibility
    if let Some(ann) = &let_decl.ty {
        let declared_ty = resolve_type_ann(ann);
        if declared_ty != expr_ty {
            return Err(TypeError::Mismatch {
                expected: declared_ty.as_str(),
                found: expr_ty.as_str(),
            });
        }
        env.add_var(let_decl.name.clone(), declared_ty);
    } else {
        env.add_var(let_decl.name.clone(), expr_ty.clone());
    }

    Ok(CoreType::Unit)
}

/// Type check an expression
pub fn typecheck_expr(env: &mut TypeEnv, expr: &Expr) -> Result<CoreType, TypeError> {
    use CoreType::*;

    match expr {
        Expr::IntLiteral(_) => Ok(Int),
        Expr::FloatLiteral(_) => Ok(Float),
        Expr::BoolLiteral(_) => Ok(Bool),
        Expr::StringLiteral(_) => Ok(String),

        Expr::Var(id) => {
            // Try local vars first
            if let Some(info) = env.lookup_var(id) {
                return Ok(info.ty.clone());
            }

            // Try domain symbols (EvidenceProgram names, Model names, etc.)
            if let Some(ty) = env.fn_env.lookup_domain_symbol(id) {
                return Ok(ty);
            }

            Err(TypeError::UnknownVar(id.clone()))
        }

        Expr::Record(fields) => {
            let mut map = HashMap::new();
            for (name, e) in fields {
                let ty = typecheck_expr(env, e)?;
                map.insert(name.clone(), ty);
            }
            Ok(Record(map))
        }

        Expr::FieldAccess { target, field } => {
            let t_ty = typecheck_expr(env, target)?;
            match t_ty {
                Record(map) => map.get(field).cloned().ok_or(TypeError::NoSuchField {
                    field: field.clone(),
                }),
                other => Err(TypeError::NotARecord(other.as_str())),
            }
        }

        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let c_ty = typecheck_expr(env, cond)?;
            if c_ty != Bool {
                return Err(TypeError::CondNotBool(c_ty.as_str()));
            }

            let t_ty = typecheck_expr(env, then_branch)?;
            let e_ty = typecheck_expr(env, else_branch)?;

            // Require exactly equal types for branches in Week 26
            if t_ty != e_ty {
                return Err(TypeError::Mismatch {
                    expected: t_ty.as_str(),
                    found: e_ty.as_str(),
                });
            }

            Ok(t_ty)
        }

        Expr::Call { callee, args } => {
            // Only allow Var as callee in Week 26
            let fn_name = match &**callee {
                Expr::Var(id) => id.clone(),
                _ => {
                    return Err(TypeError::UnknownFn("<non-var callee>".to_string()));
                }
            };

            // Look up function signature
            let sig = env.fn_env.lookup_fn(&fn_name)?;

            // Check arity
            if args.len() != sig.params.len() {
                return Err(TypeError::ArityMismatch {
                    fn_name,
                    expected: sig.params.len(),
                    found: args.len(),
                });
            }

            // Check argument types
            for (i, (arg_expr, expected_ty)) in args.iter().zip(sig.params.iter()).enumerate() {
                let arg_ty = typecheck_expr(env, arg_expr)?;
                if &arg_ty != expected_ty {
                    return Err(TypeError::Mismatch {
                        expected: expected_ty.as_str(),
                        found: arg_ty.as_str(),
                    });
                }
            }

            Ok(sig.ret.clone())
        }

        Expr::BlockExpr(block) => typecheck_block(env, block),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::core_lang::{Param, TypeAnn};

    fn make_test_domain_env() -> DomainEnv {
        let mut env = DomainEnv::new();
        env.add_evidence_program("OncologyEvidence".to_string());
        env.add_model("OneCmptIV".to_string());
        env.add_protocol("Phase2Protocol".to_string());
        env
    }

    #[test]
    fn test_typecheck_int_literal() {
        let domain_env = DomainEnv::new();
        let fn_env = FnEnv::new(domain_env);
        let mut env = TypeEnv::new(&fn_env);

        let expr = Expr::int(42);
        let ty = typecheck_expr(&mut env, &expr).unwrap();
        assert_eq!(ty, CoreType::Int);
    }

    #[test]
    fn test_typecheck_string_literal() {
        let domain_env = DomainEnv::new();
        let fn_env = FnEnv::new(domain_env);
        let mut env = TypeEnv::new(&fn_env);

        let expr = Expr::string("hello".to_string());
        let ty = typecheck_expr(&mut env, &expr).unwrap();
        assert_eq!(ty, CoreType::String);
    }

    #[test]
    fn test_typecheck_var_local() {
        let domain_env = DomainEnv::new();
        let fn_env = FnEnv::new(domain_env);
        let mut env = TypeEnv::new(&fn_env);

        env.add_var("x".to_string(), CoreType::Int);

        let expr = Expr::var("x".to_string());
        let ty = typecheck_expr(&mut env, &expr).unwrap();
        assert_eq!(ty, CoreType::Int);
    }

    #[test]
    fn test_typecheck_var_domain() {
        let domain_env = make_test_domain_env();
        let fn_env = FnEnv::new(domain_env);
        let mut env = TypeEnv::new(&fn_env);

        let expr = Expr::var("OncologyEvidence".to_string());
        let ty = typecheck_expr(&mut env, &expr).unwrap();
        assert_eq!(ty, CoreType::EvidenceProgram);
    }

    #[test]
    fn test_typecheck_unknown_var() {
        let domain_env = DomainEnv::new();
        let fn_env = FnEnv::new(domain_env);
        let mut env = TypeEnv::new(&fn_env);

        let expr = Expr::var("unknown".to_string());
        let result = typecheck_expr(&mut env, &expr);
        assert!(result.is_err());
        match result.unwrap_err() {
            TypeError::UnknownVar(name) => assert_eq!(name, "unknown"),
            _ => panic!("Expected UnknownVar error"),
        }
    }

    #[test]
    fn test_typecheck_if_expr() {
        let domain_env = DomainEnv::new();
        let fn_env = FnEnv::new(domain_env);
        let mut env = TypeEnv::new(&fn_env);

        let expr = Expr::if_expr(Expr::bool_val(true), Expr::int(1), Expr::int(2));
        let ty = typecheck_expr(&mut env, &expr).unwrap();
        assert_eq!(ty, CoreType::Int);
    }

    #[test]
    fn test_typecheck_if_type_mismatch() {
        let domain_env = DomainEnv::new();
        let fn_env = FnEnv::new(domain_env);
        let mut env = TypeEnv::new(&fn_env);

        let expr = Expr::if_expr(
            Expr::bool_val(true),
            Expr::int(1),
            Expr::string("two".to_string()),
        );
        let result = typecheck_expr(&mut env, &expr);
        assert!(result.is_err());
    }

    #[test]
    fn test_typecheck_if_cond_not_bool() {
        let domain_env = DomainEnv::new();
        let fn_env = FnEnv::new(domain_env);
        let mut env = TypeEnv::new(&fn_env);

        let expr = Expr::if_expr(Expr::int(1), Expr::int(2), Expr::int(3));
        let result = typecheck_expr(&mut env, &expr);
        assert!(result.is_err());
        match result.unwrap_err() {
            TypeError::CondNotBool(_) => {}
            _ => panic!("Expected CondNotBool error"),
        }
    }

    #[test]
    fn test_typecheck_builtin_call() {
        let domain_env = make_test_domain_env();
        let fn_env = FnEnv::new(domain_env);
        let mut env = TypeEnv::new(&fn_env);

        let call = Expr::call(
            Expr::var("run_evidence".to_string()),
            vec![
                Expr::var("OncologyEvidence".to_string()),
                Expr::string("surrogate".to_string()),
            ],
        );

        let ty = typecheck_expr(&mut env, &call).unwrap();
        assert_eq!(ty, CoreType::EvidenceResult);
    }

    #[test]
    fn test_typecheck_builtin_call_wrong_type() {
        let domain_env = make_test_domain_env();
        let fn_env = FnEnv::new(domain_env);
        let mut env = TypeEnv::new(&fn_env);

        // run_evidence expects (EvidenceProgram, String), pass (String, String)
        let call = Expr::call(
            Expr::var("run_evidence".to_string()),
            vec![
                Expr::string("oops".to_string()),
                Expr::string("surrogate".to_string()),
            ],
        );

        let result = typecheck_expr(&mut env, &call);
        assert!(result.is_err());
        match result.unwrap_err() {
            TypeError::Mismatch { expected, found } => {
                assert_eq!(expected, "EvidenceProgram");
                assert_eq!(found, "String");
            }
            _ => panic!("Expected Mismatch error"),
        }
    }

    #[test]
    fn test_typecheck_builtin_call_arity_mismatch() {
        let domain_env = make_test_domain_env();
        let fn_env = FnEnv::new(domain_env);
        let mut env = TypeEnv::new(&fn_env);

        // run_evidence expects 2 args, pass 1
        let call = Expr::call(
            Expr::var("run_evidence".to_string()),
            vec![Expr::var("OncologyEvidence".to_string())],
        );

        let result = typecheck_expr(&mut env, &call);
        assert!(result.is_err());
        match result.unwrap_err() {
            TypeError::ArityMismatch {
                fn_name,
                expected,
                found,
            } => {
                assert_eq!(fn_name, "run_evidence");
                assert_eq!(expected, 2);
                assert_eq!(found, 1);
            }
            _ => panic!("Expected ArityMismatch error"),
        }
    }

    #[test]
    fn test_typecheck_let_with_annotation() {
        let domain_env = make_test_domain_env();
        let fn_env = FnEnv::new(domain_env);
        let mut env = TypeEnv::new(&fn_env);

        let let_decl = LetDecl {
            name: "ev".to_string(),
            ty: Some(TypeAnn::EvidenceProgram),
            expr: Expr::var("OncologyEvidence".to_string()),
        };

        let result = typecheck_let(&mut env, &let_decl);
        assert!(result.is_ok());

        // Verify variable was added with correct type
        let var_ty = env.lookup_var("ev").unwrap();
        assert_eq!(var_ty.ty, CoreType::EvidenceProgram);
    }

    #[test]
    fn test_typecheck_let_type_mismatch() {
        let domain_env = make_test_domain_env();
        let fn_env = FnEnv::new(domain_env);
        let mut env = TypeEnv::new(&fn_env);

        let let_decl = LetDecl {
            name: "x".to_string(),
            ty: Some(TypeAnn::Int),
            expr: Expr::string("not an int".to_string()),
        };

        let result = typecheck_let(&mut env, &let_decl);
        assert!(result.is_err());
    }

    #[test]
    fn test_typecheck_fn() {
        let domain_env = make_test_domain_env();
        let fn_env = FnEnv::new(domain_env);

        let fn_def = FnDef::new(
            "run_phase2".to_string(),
            vec![],
            Some(TypeAnn::EvidenceResult),
            Block::new(vec![
                Stmt::Let(LetDecl {
                    name: "ev".to_string(),
                    ty: Some(TypeAnn::EvidenceProgram),
                    expr: Expr::var("OncologyEvidence".to_string()),
                }),
                Stmt::Expr(Expr::call(
                    Expr::var("run_evidence".to_string()),
                    vec![
                        Expr::var("ev".to_string()),
                        Expr::string("surrogate".to_string()),
                    ],
                )),
            ]),
        );

        let sig = typecheck_fn(&fn_def, &fn_env).unwrap();
        assert_eq!(sig.params.len(), 0);
        assert_eq!(sig.ret, CoreType::EvidenceResult);
    }
}
