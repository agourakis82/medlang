// Week 53: Trait Lowering
//
// Converts trait implementations to plain functions and rewrites
// trait method calls to direct function calls.
//
// This pass erases all trait-related constructs, preparing the IR
// for monomorphization and code generation.

use super::ast::{ImplMethod, TraitImpl};
use super::types::{TraitImplIndex, TraitImplTy, TraitRefTy, TraitResolver};
use std::collections::HashMap;
use std::fmt;

// =============================================================================
// Symbol Mangling
// =============================================================================

/// Generate a mangled symbol for a trait method implementation
///
/// Format: {TraitName}_{TypeArgs}_{MethodName}
///
/// Examples:
/// - Numeric_Float_add
/// - Ord_Vector_Int_compare
/// - Diffable_Dual_tangent
pub fn mangle_trait_method_symbol(
    trait_name: &str,
    type_args: &[String],
    method_name: &str,
) -> String {
    let type_part = if type_args.is_empty() {
        String::new()
    } else {
        format!(
            "_{}",
            type_args
                .iter()
                .map(|t| mangle_type_name(t))
                .collect::<Vec<_>>()
                .join("_")
        )
    };

    format!("{}{}_{}", trait_name, type_part, method_name)
}

/// Mangle a type name for use in symbols
fn mangle_type_name(ty: &str) -> String {
    // Remove angle brackets and replace with underscores
    ty.replace('<', "_")
        .replace('>', "")
        .replace(',', "_")
        .replace(' ', "")
}

/// Generate mangled name for a monomorphized generic function with trait bounds
pub fn mangle_bounded_generic(fn_name: &str, type_args: &[String]) -> String {
    if type_args.is_empty() {
        fn_name.to_string()
    } else {
        format!(
            "{}_{}",
            fn_name,
            type_args
                .iter()
                .map(|t| mangle_type_name(t))
                .collect::<Vec<_>>()
                .join("_")
        )
    }
}

// =============================================================================
// Generated Function
// =============================================================================

/// A function generated from a trait impl method
#[derive(Clone, Debug)]
pub struct GeneratedFn {
    /// The mangled function name
    pub name: String,

    /// Original trait name
    pub trait_name: String,

    /// Original method name
    pub method_name: String,

    /// Type arguments the impl was instantiated with
    pub type_args: Vec<String>,

    /// Parameter names
    pub param_names: Vec<String>,

    /// Parameter types
    pub param_types: Vec<String>,

    /// Return type
    pub ret_type: String,

    /// Function body (as string for now)
    pub body: String,
}

impl GeneratedFn {
    pub fn from_impl_method(impl_: &TraitImpl, method: &ImplMethod) -> Self {
        let name = mangle_trait_method_symbol(
            &impl_.trait_ref.trait_name,
            &impl_.trait_ref.type_args,
            &method.name,
        );

        Self {
            name,
            trait_name: impl_.trait_ref.trait_name.clone(),
            method_name: method.name.clone(),
            type_args: impl_.trait_ref.type_args.clone(),
            param_names: method.params.iter().map(|p| p.name.clone()).collect(),
            param_types: method.params.iter().map(|p| p.ty.clone()).collect(),
            ret_type: method.ret_type.clone(),
            body: method.body.clone(),
        }
    }
}

// =============================================================================
// Trait Impl Lowering
// =============================================================================

/// Lower all trait implementations to plain functions
pub fn lower_trait_impls(impls: &[TraitImpl]) -> Result<Vec<GeneratedFn>, LowerError> {
    let mut generated = Vec::new();

    for impl_ in impls {
        for method in &impl_.methods {
            generated.push(GeneratedFn::from_impl_method(impl_, method));
        }
    }

    Ok(generated)
}

/// Lower a single trait impl to functions
pub fn lower_single_impl(impl_: &TraitImpl) -> Vec<GeneratedFn> {
    impl_
        .methods
        .iter()
        .map(|method| GeneratedFn::from_impl_method(impl_, method))
        .collect()
}

// =============================================================================
// Trait Call Rewriting
// =============================================================================

/// Represents a trait method call in the source
#[derive(Clone, Debug)]
pub struct TraitMethodCall {
    /// Trait name
    pub trait_name: String,

    /// Method name
    pub method_name: String,

    /// Type arguments (may be inferred)
    pub type_args: Vec<String>,

    /// Arguments to the method
    pub args: Vec<String>, // Simplified: argument expressions as strings
}

/// Result of rewriting a trait method call
#[derive(Clone, Debug)]
pub struct RewrittenCall {
    /// The resolved function name
    pub fn_name: String,

    /// Arguments (unchanged)
    pub args: Vec<String>,
}

/// Rewrite a trait method call to a plain function call
pub fn rewrite_trait_call(
    call: &TraitMethodCall,
    traits: &HashMap<String, super::types::TraitDeclTy>,
    impls: &TraitImplIndex,
) -> Result<RewrittenCall, LowerError> {
    let resolver = TraitResolver::new(traits, impls);

    // Resolve the method
    let resolved = resolver
        .resolve_method(&call.trait_name, &call.method_name, &call.type_args)
        .map_err(|e| LowerError::TraitResolution(e.to_string()))?;

    Ok(RewrittenCall {
        fn_name: resolved.symbol,
        args: call.args.clone(),
    })
}

/// Rewrite all trait method calls in an expression
/// This is a simplified version - a real implementation would traverse the AST
pub fn lower_trait_calls(
    calls: &[TraitMethodCall],
    traits: &HashMap<String, super::types::TraitDeclTy>,
    impls: &TraitImplIndex,
) -> Result<Vec<RewrittenCall>, LowerError> {
    calls
        .iter()
        .map(|call| rewrite_trait_call(call, traits, impls))
        .collect()
}

// =============================================================================
// Expression Rewriter (Simplified)
// =============================================================================

/// A simplified expression type for demonstrating trait call rewriting
#[derive(Clone, Debug)]
pub enum SimpleExpr {
    /// Integer literal
    Int(i64),

    /// Float literal
    Float(f64),

    /// String literal
    String(String),

    /// Variable reference
    Var(String),

    /// Binary operation
    BinOp {
        op: String,
        left: Box<SimpleExpr>,
        right: Box<SimpleExpr>,
    },

    /// Function call
    Call {
        fn_name: String,
        args: Vec<SimpleExpr>,
    },

    /// Trait method call (to be lowered)
    TraitCall {
        trait_name: String,
        method_name: String,
        type_args: Vec<String>,
        args: Vec<SimpleExpr>,
    },
}

/// Rewrite all trait calls in an expression tree
pub fn rewrite_expr(
    expr: &SimpleExpr,
    traits: &HashMap<String, super::types::TraitDeclTy>,
    impls: &TraitImplIndex,
) -> Result<SimpleExpr, LowerError> {
    match expr {
        SimpleExpr::TraitCall {
            trait_name,
            method_name,
            type_args,
            args,
        } => {
            let resolver = TraitResolver::new(traits, impls);

            // Resolve the method
            let resolved = resolver
                .resolve_method(trait_name, method_name, type_args)
                .map_err(|e| LowerError::TraitResolution(e.to_string()))?;

            // Rewrite arguments recursively
            let rewritten_args: Result<Vec<_>, _> = args
                .iter()
                .map(|a| rewrite_expr(a, traits, impls))
                .collect();

            Ok(SimpleExpr::Call {
                fn_name: resolved.symbol,
                args: rewritten_args?,
            })
        }

        SimpleExpr::BinOp { op, left, right } => Ok(SimpleExpr::BinOp {
            op: op.clone(),
            left: Box::new(rewrite_expr(left, traits, impls)?),
            right: Box::new(rewrite_expr(right, traits, impls)?),
        }),

        SimpleExpr::Call { fn_name, args } => {
            let rewritten_args: Result<Vec<_>, _> = args
                .iter()
                .map(|a| rewrite_expr(a, traits, impls))
                .collect();

            Ok(SimpleExpr::Call {
                fn_name: fn_name.clone(),
                args: rewritten_args?,
            })
        }

        // Literals and variables pass through unchanged
        _ => Ok(expr.clone()),
    }
}

// =============================================================================
// Verification
// =============================================================================

/// Check that all trait-related constructs have been lowered
pub fn verify_no_trait_constructs(expr: &SimpleExpr) -> Result<(), LowerError> {
    match expr {
        SimpleExpr::TraitCall {
            trait_name,
            method_name,
            ..
        } => Err(LowerError::UnloweredTraitCall {
            trait_name: trait_name.clone(),
            method_name: method_name.clone(),
        }),

        SimpleExpr::BinOp { left, right, .. } => {
            verify_no_trait_constructs(left)?;
            verify_no_trait_constructs(right)
        }

        SimpleExpr::Call { args, .. } => {
            for arg in args {
                verify_no_trait_constructs(arg)?;
            }
            Ok(())
        }

        _ => Ok(()),
    }
}

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug, Clone)]
pub enum LowerError {
    TraitResolution(String),
    UnloweredTraitCall {
        trait_name: String,
        method_name: String,
    },
    Other(String),
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LowerError::TraitResolution(msg) => write!(f, "trait resolution error: {}", msg),
            LowerError::UnloweredTraitCall {
                trait_name,
                method_name,
            } => {
                write!(f, "unlowered trait call: {}::{}", trait_name, method_name)
            }
            LowerError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for LowerError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ast::*;
    use crate::traits::check::TraitChecker;

    #[test]
    fn test_mangle_trait_method_symbol() {
        assert_eq!(
            mangle_trait_method_symbol("Numeric", &["Float".to_string()], "add"),
            "Numeric_Float_add"
        );

        assert_eq!(
            mangle_trait_method_symbol("Ord", &["Int".to_string()], "lt"),
            "Ord_Int_lt"
        );

        assert_eq!(mangle_trait_method_symbol("Eq", &[], "eq"), "Eq_eq");
    }

    #[test]
    fn test_mangle_complex_types() {
        assert_eq!(
            mangle_trait_method_symbol("Container", &["Vector<Int>".to_string()], "push"),
            "Container_Vector_Int_push"
        );
    }

    #[test]
    fn test_lower_trait_impl() {
        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float"))
            .with_method_body("zero", vec![], "Float", "0.0")
            .with_method_body(
                "add",
                vec![("x", "Float"), ("y", "Float")],
                "Float",
                "x + y",
            );

        let generated = lower_single_impl(&impl_);

        assert_eq!(generated.len(), 2);

        let zero_fn = generated.iter().find(|f| f.method_name == "zero").unwrap();
        assert_eq!(zero_fn.name, "Numeric_Float_zero");
        assert!(zero_fn.param_names.is_empty());
        assert_eq!(zero_fn.ret_type, "Float");

        let add_fn = generated.iter().find(|f| f.method_name == "add").unwrap();
        assert_eq!(add_fn.name, "Numeric_Float_add");
        assert_eq!(add_fn.param_names, vec!["x", "y"]);
        assert_eq!(add_fn.param_types, vec!["Float", "Float"]);
    }

    #[test]
    fn test_rewrite_trait_call() {
        let mut checker = TraitChecker::new();

        // Register trait
        let trait_decl = TraitDecl::new("Numeric").with_type_param("T").with_method(
            TraitMethod::new("add")
                .with_param("x", "T")
                .with_param("y", "T")
                .with_ret_type("T"),
        );
        checker.check_trait_decl(&trait_decl).unwrap();

        // Register impl
        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float")).with_method_body(
            "add",
            vec![("x", "Float"), ("y", "Float")],
            "Float",
            "x + y",
        );
        checker.check_trait_impl(&impl_).unwrap();

        // Create a trait method call
        let call = TraitMethodCall {
            trait_name: "Numeric".to_string(),
            method_name: "add".to_string(),
            type_args: vec!["Float".to_string()],
            args: vec!["a".to_string(), "b".to_string()],
        };

        // Rewrite it
        let result = rewrite_trait_call(&call, checker.all_traits(), checker.all_impls());
        assert!(result.is_ok());

        let rewritten = result.unwrap();
        assert_eq!(rewritten.fn_name, "Numeric_Float_add");
        assert_eq!(rewritten.args, vec!["a", "b"]);
    }

    #[test]
    fn test_rewrite_expr_tree() {
        let mut checker = TraitChecker::new();

        // Register trait and impl
        let trait_decl = TraitDecl::new("Numeric").with_type_param("T").with_method(
            TraitMethod::new("add")
                .with_param("x", "T")
                .with_param("y", "T")
                .with_ret_type("T"),
        );
        checker.check_trait_decl(&trait_decl).unwrap();

        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float")).with_method_body(
            "add",
            vec![("x", "Float"), ("y", "Float")],
            "Float",
            "x + y",
        );
        checker.check_trait_impl(&impl_).unwrap();

        // Create expression: Numeric::add(x, Numeric::add(y, z))
        let expr = SimpleExpr::TraitCall {
            trait_name: "Numeric".to_string(),
            method_name: "add".to_string(),
            type_args: vec!["Float".to_string()],
            args: vec![
                SimpleExpr::Var("x".to_string()),
                SimpleExpr::TraitCall {
                    trait_name: "Numeric".to_string(),
                    method_name: "add".to_string(),
                    type_args: vec!["Float".to_string()],
                    args: vec![
                        SimpleExpr::Var("y".to_string()),
                        SimpleExpr::Var("z".to_string()),
                    ],
                },
            ],
        };

        // Rewrite
        let result = rewrite_expr(&expr, checker.all_traits(), checker.all_impls());
        assert!(result.is_ok());

        let rewritten = result.unwrap();

        // Should be: Numeric_Float_add(x, Numeric_Float_add(y, z))
        match &rewritten {
            SimpleExpr::Call { fn_name, args } => {
                assert_eq!(fn_name, "Numeric_Float_add");
                assert_eq!(args.len(), 2);

                match &args[1] {
                    SimpleExpr::Call {
                        fn_name: inner_fn, ..
                    } => {
                        assert_eq!(inner_fn, "Numeric_Float_add");
                    }
                    _ => panic!("Expected nested call"),
                }
            }
            _ => panic!("Expected Call"),
        }

        // Verify no trait constructs remain
        assert!(verify_no_trait_constructs(&rewritten).is_ok());
    }

    #[test]
    fn test_verify_no_trait_constructs() {
        // Clean expression
        let clean = SimpleExpr::Call {
            fn_name: "add".to_string(),
            args: vec![SimpleExpr::Int(1), SimpleExpr::Int(2)],
        };
        assert!(verify_no_trait_constructs(&clean).is_ok());

        // Expression with trait call
        let dirty = SimpleExpr::TraitCall {
            trait_name: "Numeric".to_string(),
            method_name: "add".to_string(),
            type_args: vec![],
            args: vec![],
        };
        assert!(verify_no_trait_constructs(&dirty).is_err());
    }

    #[test]
    fn test_mangle_bounded_generic() {
        assert_eq!(
            mangle_bounded_generic("sum", &["Float".to_string()]),
            "sum_Float"
        );
        assert_eq!(
            mangle_bounded_generic("map", &["Int".to_string(), "String".to_string()]),
            "map_Int_String"
        );
        assert_eq!(mangle_bounded_generic("identity", &[]), "identity");
    }
}
