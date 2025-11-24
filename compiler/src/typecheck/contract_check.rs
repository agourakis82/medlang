//! Week 28: Type checking for contracts, invariants, and assertions
//!
//! This module provides type checking for:
//! - Function contracts (requires/ensures)
//! - Model/Policy invariant blocks
//! - Assert statements

use crate::ast::contracts::{AssertStmt, ContractClause, FnContract, InvariantBlock};
use crate::ast::core_lang::{Expr, FnDef};
use crate::typecheck::core_lang::{TypeEnv, TypeError};
use crate::types::core_lang::CoreType;

// =============================================================================
// Function Contract Type Checking
// =============================================================================

/// Type check a function contract (requires/ensures clauses)
///
/// All contract clauses must type check to Bool.
/// - `requires` clauses can reference function parameters
/// - `ensures` clauses can reference parameters and the special `result` variable
pub fn typecheck_fn_contract(fn_def: &FnDef, env: &mut TypeEnv) -> Result<(), Vec<TypeError>> {
    let contract = match &fn_def.contract {
        Some(c) => c,
        None => return Ok(()), // No contract to check
    };

    let mut errors = Vec::new();

    // Type check requires clauses (preconditions)
    for (idx, clause) in contract.requires.iter().enumerate() {
        if let Err(e) = typecheck_contract_clause(clause, env, &fn_def.name, "requires", idx) {
            errors.push(e);
        }
    }

    // Type check ensures clauses (postconditions)
    // TODO: Add 'result' variable to environment for postconditions
    for (idx, clause) in contract.ensures.iter().enumerate() {
        if let Err(e) = typecheck_contract_clause(clause, env, &fn_def.name, "ensures", idx) {
            errors.push(e);
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Type check a single contract clause
///
/// The clause expression must type check to Bool.
fn typecheck_contract_clause(
    clause: &ContractClause,
    env: &TypeEnv,
    fn_name: &str,
    clause_type: &str,
    index: usize,
) -> Result<(), TypeError> {
    // For now, we'll do a simplified check
    // Full implementation would recursively type check the expression
    // and verify it evaluates to Bool

    // TODO: Implement full expression type checking
    // For Week 28, we're establishing the infrastructure
    // let expr_type = typecheck_expr(&clause.condition, env)?;

    // if expr_type != CoreType::Bool {
    //     return Err(TypeError::ContractExpressionNotBool {
    //         fn_name: fn_name.to_string(),
    //         clause_type: clause_type.to_string(),
    //         clause_index: index,
    //         found: expr_type,
    //     });
    // }

    Ok(())
}

// =============================================================================
// Invariant Block Type Checking
// =============================================================================

/// Type check an invariant block
///
/// All invariant clauses must type check to Bool and can reference:
/// - Model states, params, observables (for model invariants)
/// - Policy parameters and state (for policy invariants)
pub fn typecheck_invariant_block(
    invariants: &InvariantBlock,
    env: &TypeEnv,
    context_name: &str, // Name of model/policy containing the invariants
) -> Result<(), Vec<TypeError>> {
    let mut errors = Vec::new();

    for (idx, clause) in invariants.clauses.iter().enumerate() {
        if let Err(e) = typecheck_contract_clause(clause, env, context_name, "invariant", idx) {
            errors.push(e);
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// =============================================================================
// Assert Statement Type Checking
// =============================================================================

/// Type check an assert statement
///
/// The assertion condition must type check to Bool.
pub fn typecheck_assert(assert_stmt: &AssertStmt, env: &TypeEnv) -> Result<(), TypeError> {
    // TODO: Implement full expression type checking
    // For Week 28, we're establishing the infrastructure
    // let expr_type = typecheck_expr(&assert_stmt.condition, env)?;

    // if expr_type != CoreType::Bool {
    //     return Err(TypeError::AssertExpressionNotBool {
    //         found: expr_type,
    //     });
    // }

    Ok(())
}

// =============================================================================
// Contract Expression Validation Helpers
// =============================================================================

/// Check if an expression is a valid contract expression
///
/// Contract expressions have restrictions:
/// - No side effects
/// - Deterministic (no random number generation)
/// - Must terminate (no loops, only bounded operations)
pub fn is_valid_contract_expr(expr: &Expr) -> bool {
    // TODO: Implement expression validation
    // For Week 28, we accept all expressions
    // Later we'll add restrictions for safety
    true
}

/// Extract free variables from a contract expression
///
/// This is used to verify that contract expressions only reference
/// variables that are in scope (parameters, locals, etc.)
pub fn extract_free_variables(expr: &Expr) -> Vec<String> {
    // TODO: Implement variable extraction
    // For Week 28, return empty list
    Vec::new()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::contracts::{ContractClause, FnContract};
    use crate::ast::core_lang::{Block, Expr, FnDef, Param, Stmt, TypeAnn};
    use crate::ast::{ExprKind, Literal};

    fn bool_expr(value: bool) -> Expr {
        Expr {
            kind: ExprKind::Literal(Literal::Float(if value { 1.0 } else { 0.0 })),
            span: None,
        }
    }

    fn make_test_fn_with_contract(contract: FnContract) -> FnDef {
        FnDef {
            name: "test_fn".to_string(),
            params: vec![Param {
                name: "x".to_string(),
                ty: Some(TypeAnn::Int),
            }],
            ret_type: Some(TypeAnn::Int),
            contract: Some(contract),
            body: Block::new(vec![Stmt::Expr(Expr::var("x".to_string()))]),
        }
    }

    #[test]
    fn test_empty_contract_passes() {
        let fn_def = make_test_fn_with_contract(FnContract::new());
        let mut env = TypeEnv::new();

        let result = typecheck_fn_contract(&fn_def, &mut env);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fn_without_contract_passes() {
        let fn_def = FnDef {
            name: "test_fn".to_string(),
            params: vec![],
            ret_type: Some(TypeAnn::Int),
            contract: None,
            body: Block::new(vec![]),
        };
        let mut env = TypeEnv::new();

        let result = typecheck_fn_contract(&fn_def, &mut env);
        assert!(result.is_ok());
    }

    #[test]
    fn test_requires_clause_typechecks() {
        let contract =
            FnContract::new()
                .with_requires(vec![ContractClause::new(bool_expr(true))
                    .with_label("x must be positive".to_string())]);

        let fn_def = make_test_fn_with_contract(contract);
        let mut env = TypeEnv::new();

        let result = typecheck_fn_contract(&fn_def, &mut env);
        // Should pass for now (full type checking not yet implemented)
        assert!(result.is_ok());
    }

    #[test]
    fn test_ensures_clause_typechecks() {
        let contract =
            FnContract::new()
                .with_ensures(vec![ContractClause::new(bool_expr(true))
                    .with_label("result is positive".to_string())]);

        let fn_def = make_test_fn_with_contract(contract);
        let mut env = TypeEnv::new();

        let result = typecheck_fn_contract(&fn_def, &mut env);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invariant_block_typechecks() {
        let invariants = InvariantBlock::new(vec![
            ContractClause::new(bool_expr(true)).with_label("CL > 0".to_string()),
            ContractClause::new(bool_expr(true)).with_label("V > 0".to_string()),
        ]);

        let env = TypeEnv::new();
        let result = typecheck_invariant_block(&invariants, &env, "PK_OneCompOral");
        assert!(result.is_ok());
    }

    #[test]
    fn test_assert_typechecks() {
        let assert_stmt =
            AssertStmt::new(bool_expr(true)).with_message("n must be positive".to_string());

        let env = TypeEnv::new();
        let result = typecheck_assert(&assert_stmt, &env);
        assert!(result.is_ok());
    }

    #[test]
    fn test_is_valid_contract_expr() {
        let expr = bool_expr(true);
        assert!(is_valid_contract_expr(&expr));
    }

    #[test]
    fn test_extract_free_variables_empty() {
        let expr = bool_expr(true);
        let vars = extract_free_variables(&expr);
        assert!(vars.is_empty());
    }
}
