// Week 49: Pattern Matching Type Checker with Exhaustiveness Analysis
//
// This module provides comprehensive type checking for pattern matching:
// - Type validation of patterns against scrutinee type
// - Exhaustiveness checking (all cases covered)
// - Usefulness checking (no unreachable/redundant patterns)
// - OR pattern binding consistency validation
// - Guard expression type checking
//
// The exhaustiveness algorithm is based on the classic pattern matrix approach,
// adapted for MedLang's type system.

use crate::ast::pattern::{
    GuardExpr, MatchArmV2, MatchBody, MatchExprV2, Pattern, PatternError, PatternKind,
};
use crate::ast::Span;
use crate::typecheck::{FnEnv, TypeEnv};
use crate::types::core_lang::CoreType;
use crate::types::enum_types::EnumEnv;
use std::collections::HashSet;

// =============================================================================
// Main Type Checking Entry Points
// =============================================================================

/// Type check an enhanced match expression (v2)
/// Returns the result type if successful
pub fn typecheck_match_v2(
    env: &mut TypeEnv,
    _fn_env: &FnEnv,
    enum_env: &EnumEnv,
    match_expr: &MatchExprV2,
) -> Result<CoreType, PatternError> {
    // 1. Type check the scrutinee
    let scrutinee_ty = crate::typecheck::core_lang::typecheck_expr(env, &match_expr.scrutinee)
        .map_err(|e| PatternError::TypeMismatch {
            expected: "valid expression".to_string(),
            found: format!("{}", e),
            span: match_expr.span.clone(),
        })?;

    // 2. Validate each arm's pattern against scrutinee type
    let mut arm_types: Vec<CoreType> = Vec::new();

    for (arm_idx, arm) in match_expr.arms.iter().enumerate() {
        // Check pattern type compatibility
        check_pattern_type(env, enum_env, &arm.pattern, &scrutinee_ty)?;

        // Check OR pattern binding consistency
        check_or_pattern_bindings(&arm.pattern)?;

        // Add pattern bindings to environment for body/guard checking
        let bindings = collect_pattern_bindings(&arm.pattern, &scrutinee_ty, enum_env)?;
        let mut arm_env = env.clone();
        for (name, ty) in &bindings {
            arm_env.add_var(name.clone(), ty.clone());
        }

        // Check guard if present
        if let Some(guard) = &arm.guard {
            check_guard_expr(&mut arm_env, guard)?;
        }

        // Check body expression
        let body_ty = match &arm.body {
            MatchBody::Expr(expr) => {
                crate::typecheck::core_lang::typecheck_expr(&mut arm_env, expr).map_err(|e| {
                    PatternError::TypeMismatch {
                        expected: "valid expression".to_string(),
                        found: format!("{}", e),
                        span: arm.span.clone(),
                    }
                })?
            }
            MatchBody::Block(block) => {
                crate::typecheck::core_lang::typecheck_block(&mut arm_env, block).map_err(|e| {
                    PatternError::TypeMismatch {
                        expected: "valid block".to_string(),
                        found: format!("{}", e),
                        span: arm.span.clone(),
                    }
                })?
            }
        };

        arm_types.push(body_ty);
    }

    // 3. Check all arms have the same type
    if let Some(first_ty) = arm_types.first() {
        for (i, ty) in arm_types.iter().enumerate().skip(1) {
            if ty != first_ty {
                return Err(PatternError::TypeMismatch {
                    expected: first_ty.as_str().to_string(),
                    found: ty.as_str().to_string(),
                    span: match_expr.arms.get(i).and_then(|a| a.span.clone()),
                });
            }
        }
    }

    // 4. Check exhaustiveness
    check_exhaustiveness(
        enum_env,
        &scrutinee_ty,
        &match_expr.arms,
        match_expr.span.clone(),
    )?;

    // 5. Check for unreachable patterns
    check_usefulness(&match_expr.arms)?;

    // Return result type
    Ok(arm_types.into_iter().next().unwrap_or(CoreType::Unit))
}

// =============================================================================
// Pattern Type Checking
// =============================================================================

/// Check that a pattern is compatible with the expected type
fn check_pattern_type(
    env: &TypeEnv,
    enum_env: &EnumEnv,
    pattern: &Pattern,
    expected_ty: &CoreType,
) -> Result<(), PatternError> {
    match &pattern.kind {
        PatternKind::Wildcard => Ok(()),
        PatternKind::Binding { .. } => Ok(()),

        PatternKind::EnumVariant {
            enum_name,
            variant,
            payloads,
        } => {
            // Get expected enum name
            let expected_enum = match expected_ty {
                CoreType::Enum(name) => name,
                _ => {
                    return Err(PatternError::TypeMismatch {
                        expected: expected_ty.as_str().to_string(),
                        found: format!(
                            "enum pattern {}::{}",
                            enum_name.as_deref().unwrap_or("?"),
                            variant
                        ),
                        span: pattern.span.clone(),
                    });
                }
            };

            // Check enum name if specified
            if let Some(en) = enum_name {
                if en != expected_enum {
                    return Err(PatternError::TypeMismatch {
                        expected: expected_enum.clone(),
                        found: en.clone(),
                        span: pattern.span.clone(),
                    });
                }
            }

            // Check variant exists
            let enum_info =
                enum_env
                    .get_enum(expected_enum)
                    .ok_or_else(|| PatternError::UnknownEnum {
                        enum_name: expected_enum.clone(),
                        span: pattern.span.clone(),
                    })?;

            if !enum_info.has_variant(variant) {
                return Err(PatternError::UnknownVariant {
                    enum_name: expected_enum.clone(),
                    variant: variant.clone(),
                    span: pattern.span.clone(),
                });
            }

            // For now, Week 49 only supports nullary variants
            // Future: check payload types for data-carrying variants
            if !payloads.is_empty() {
                return Err(PatternError::ArityMismatch {
                    variant: variant.clone(),
                    expected: 0,
                    found: payloads.len(),
                    span: pattern.span.clone(),
                });
            }

            Ok(())
        }

        PatternKind::BoolLit(_) => {
            if *expected_ty != CoreType::Bool {
                return Err(PatternError::TypeMismatch {
                    expected: expected_ty.as_str().to_string(),
                    found: "Bool".to_string(),
                    span: pattern.span.clone(),
                });
            }
            Ok(())
        }

        PatternKind::IntLit(_) => {
            if *expected_ty != CoreType::Int {
                return Err(PatternError::TypeMismatch {
                    expected: expected_ty.as_str().to_string(),
                    found: "Int".to_string(),
                    span: pattern.span.clone(),
                });
            }
            Ok(())
        }

        PatternKind::FloatLit(_) => {
            if *expected_ty != CoreType::Float {
                return Err(PatternError::TypeMismatch {
                    expected: expected_ty.as_str().to_string(),
                    found: "Float".to_string(),
                    span: pattern.span.clone(),
                });
            }
            Ok(())
        }

        PatternKind::StringLit(_) => {
            if *expected_ty != CoreType::String {
                return Err(PatternError::TypeMismatch {
                    expected: expected_ty.as_str().to_string(),
                    found: "String".to_string(),
                    span: pattern.span.clone(),
                });
            }
            Ok(())
        }

        PatternKind::Or(alternatives) => {
            for alt in alternatives {
                check_pattern_type(env, enum_env, alt, expected_ty)?;
            }
            Ok(())
        }
    }
}

/// Check that OR pattern alternatives bind the same names
fn check_or_pattern_bindings(pattern: &Pattern) -> Result<(), PatternError> {
    if let PatternKind::Or(alternatives) = &pattern.kind {
        if alternatives.is_empty() {
            return Ok(());
        }

        let first_bindings: HashSet<String> = alternatives[0].binding_names().into_iter().collect();

        for (_i, alt) in alternatives.iter().enumerate().skip(1) {
            let alt_bindings: HashSet<String> = alt.binding_names().into_iter().collect();

            if first_bindings != alt_bindings {
                return Err(PatternError::OrBindingMismatch {
                    left: first_bindings.iter().cloned().collect(),
                    right: alt_bindings.iter().cloned().collect(),
                    span: pattern.span.clone(),
                });
            }
        }

        // Recursively check nested OR patterns
        for alt in alternatives {
            check_or_pattern_bindings(alt)?;
        }
    }

    Ok(())
}

/// Collect bindings introduced by a pattern with their types
fn collect_pattern_bindings(
    pattern: &Pattern,
    scrutinee_ty: &CoreType,
    enum_env: &EnumEnv,
) -> Result<Vec<(String, CoreType)>, PatternError> {
    let mut bindings = Vec::new();

    match &pattern.kind {
        PatternKind::Wildcard => {}
        PatternKind::Binding { name, .. } => {
            bindings.push((name.clone(), scrutinee_ty.clone()));
        }
        PatternKind::EnumVariant { payloads, .. } => {
            // For data-carrying variants (future), collect payload bindings
            for payload in payloads {
                // Would need payload type info here
                let payload_bindings =
                    collect_pattern_bindings(payload, &CoreType::Unit, enum_env)?;
                bindings.extend(payload_bindings);
            }
        }
        PatternKind::BoolLit(_)
        | PatternKind::IntLit(_)
        | PatternKind::FloatLit(_)
        | PatternKind::StringLit(_) => {}
        PatternKind::Or(alternatives) => {
            // All alternatives bind the same names (checked elsewhere)
            if let Some(first) = alternatives.first() {
                bindings = collect_pattern_bindings(first, scrutinee_ty, enum_env)?;
            }
        }
    }

    Ok(bindings)
}

/// Check that a guard expression has type Bool
fn check_guard_expr(env: &mut TypeEnv, guard: &GuardExpr) -> Result<(), PatternError> {
    let guard_ty =
        crate::typecheck::core_lang::typecheck_expr(env, &guard.condition).map_err(|e| {
            PatternError::GuardNotBool {
                found: format!("{}", e),
                span: guard.span.clone(),
            }
        })?;

    if guard_ty != CoreType::Bool {
        return Err(PatternError::GuardNotBool {
            found: guard_ty.as_str().to_string(),
            span: guard.span.clone(),
        });
    }

    Ok(())
}

// =============================================================================
// Exhaustiveness Checking
// =============================================================================

/// Check that match arms cover all possible values
fn check_exhaustiveness(
    enum_env: &EnumEnv,
    scrutinee_ty: &CoreType,
    arms: &[MatchArmV2],
    match_span: Option<Span>,
) -> Result<(), PatternError> {
    // Build the set of values we need to cover
    let required_coverage = build_required_coverage(enum_env, scrutinee_ty);

    // Track which values are covered (excluding guarded arms)
    let mut covered: HashSet<String> = HashSet::new();
    let mut has_catch_all = false;

    for arm in arms {
        // Guarded arms don't contribute to exhaustiveness
        if arm.has_guard() {
            continue;
        }

        let arm_coverage = pattern_coverage(&arm.pattern);

        if arm_coverage.contains(&"*".to_string()) {
            has_catch_all = true;
            break;
        }

        covered.extend(arm_coverage);
    }

    // Check coverage
    if !has_catch_all {
        match &required_coverage {
            RequiredCoverage::Enum { variants, .. } => {
                let missing: Vec<String> = variants
                    .iter()
                    .filter(|v| !covered.contains(*v))
                    .cloned()
                    .collect();

                if !missing.is_empty() {
                    return Err(PatternError::NonExhaustive {
                        missing,
                        span: match_span,
                    });
                }
            }
            RequiredCoverage::Bool => {
                let mut missing = Vec::new();
                if !covered.contains("true") {
                    missing.push("true".to_string());
                }
                if !covered.contains("false") {
                    missing.push("false".to_string());
                }
                if !missing.is_empty() {
                    return Err(PatternError::NonExhaustive {
                        missing,
                        span: match_span,
                    });
                }
            }
            RequiredCoverage::Unbounded => {
                // Int, Float, String - need wildcard/binding
                return Err(PatternError::NonExhaustive {
                    missing: vec!["_".to_string()],
                    span: match_span,
                });
            }
        }
    }

    Ok(())
}

/// Represents what values need to be covered for exhaustiveness
#[derive(Debug)]
enum RequiredCoverage {
    /// Enum with finite variants
    Enum {
        enum_name: String,
        variants: Vec<String>,
    },
    /// Boolean (true/false)
    Bool,
    /// Unbounded type (Int, Float, String) - requires wildcard
    Unbounded,
}

/// Build the required coverage for a type
fn build_required_coverage(enum_env: &EnumEnv, ty: &CoreType) -> RequiredCoverage {
    match ty {
        CoreType::Enum(name) => {
            if let Some(info) = enum_env.get_enum(name) {
                RequiredCoverage::Enum {
                    enum_name: name.clone(),
                    variants: info.variants.clone(),
                }
            } else {
                RequiredCoverage::Unbounded
            }
        }
        CoreType::Bool => RequiredCoverage::Bool,
        _ => RequiredCoverage::Unbounded,
    }
}

/// Get the coverage provided by a pattern
/// Returns set of covered values, or "*" for wildcard/binding
fn pattern_coverage(pattern: &Pattern) -> HashSet<String> {
    let mut coverage = HashSet::new();

    match &pattern.kind {
        PatternKind::Wildcard | PatternKind::Binding { .. } => {
            coverage.insert("*".to_string());
        }
        PatternKind::EnumVariant { variant, .. } => {
            coverage.insert(variant.clone());
        }
        PatternKind::BoolLit(b) => {
            coverage.insert(b.to_string());
        }
        PatternKind::IntLit(i) => {
            coverage.insert(i.to_string());
        }
        PatternKind::FloatLit(f) => {
            coverage.insert(f.to_string());
        }
        PatternKind::StringLit(s) => {
            coverage.insert(s.clone());
        }
        PatternKind::Or(alternatives) => {
            for alt in alternatives {
                coverage.extend(pattern_coverage(alt));
            }
        }
    }

    coverage
}

// =============================================================================
// Usefulness Checking (Unreachable/Redundant Patterns)
// =============================================================================

/// Check for unreachable patterns
fn check_usefulness(arms: &[MatchArmV2]) -> Result<(), PatternError> {
    // Build pattern matrix incrementally
    let mut seen_patterns: Vec<(usize, &Pattern, bool)> = Vec::new();

    for (arm_idx, arm) in arms.iter().enumerate() {
        // Check if this pattern is useful (not already covered)
        if let Some(covering_idx) = find_covering_pattern(&seen_patterns, &arm.pattern) {
            // Guarded patterns may still be useful even if covered
            if !arm.has_guard() {
                return Err(PatternError::Redundant {
                    arm_index: arm_idx,
                    covered_by: covering_idx,
                    span: arm.span.clone(),
                });
            }
        }

        seen_patterns.push((arm_idx, &arm.pattern, arm.has_guard()));
    }

    Ok(())
}

/// Find a pattern that already covers the given pattern
fn find_covering_pattern(seen: &[(usize, &Pattern, bool)], pattern: &Pattern) -> Option<usize> {
    for (idx, seen_pattern, has_guard) in seen {
        // Guarded patterns don't fully cover
        if *has_guard {
            continue;
        }

        // Check if seen_pattern subsumes pattern
        if crate::ast::pattern::pattern_subsumes(pattern, seen_pattern) {
            return Some(*idx);
        }
    }
    None
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Format missing patterns for error messages
pub fn format_missing_patterns(
    _enum_env: &EnumEnv,
    scrutinee_ty: &CoreType,
    missing: &[String],
) -> String {
    match scrutinee_ty {
        CoreType::Enum(name) => missing
            .iter()
            .map(|v| format!("{}::{}", name, v))
            .collect::<Vec<_>>()
            .join(", "),
        _ => missing.join(", "),
    }
}

/// Get a witness pattern for a missing case (for error messages)
pub fn witness_pattern(_enum_env: &EnumEnv, scrutinee_ty: &CoreType, missing: &str) -> Pattern {
    match scrutinee_ty {
        CoreType::Enum(name) => Pattern::enum_variant(Some(name.clone()), missing.to_string()),
        CoreType::Bool => {
            let b = missing.parse::<bool>().unwrap_or(false);
            Pattern::bool_lit(b)
        }
        CoreType::Int => {
            let i = missing.parse::<i64>().unwrap_or(0);
            Pattern::int_lit(i)
        }
        _ => Pattern::wildcard(),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::core_lang::{Block, Expr, Stmt};
    use crate::typecheck::DomainEnv;
    use crate::types::enum_types::EnumInfo;

    fn setup_response_enum() -> EnumEnv {
        let mut env = EnumEnv::new();
        env.register_enum(EnumInfo::new(
            "Response".to_string(),
            vec![
                "CR".to_string(),
                "PR".to_string(),
                "SD".to_string(),
                "PD".to_string(),
            ],
        ));
        env
    }

    fn make_test_env() -> (TypeEnv<'static>, &'static FnEnv, EnumEnv) {
        let enum_env = setup_response_enum();
        let domain_env = DomainEnv::new();
        let fn_env = Box::leak(Box::new(FnEnv::new(domain_env)));
        let type_env = TypeEnv::new(fn_env);
        (type_env, fn_env, enum_env)
    }

    #[test]
    fn test_exhaustive_enum_match() {
        let enum_env = setup_response_enum();
        let scrutinee_ty = CoreType::Enum("Response".to_string());

        // All variants covered
        let arms = vec![
            MatchArmV2::new(
                Pattern::enum_variant(Some("Response".to_string()), "CR".to_string()),
                MatchBody::expr(Expr::float(1.0)),
            ),
            MatchArmV2::new(
                Pattern::enum_variant(Some("Response".to_string()), "PR".to_string()),
                MatchBody::expr(Expr::float(0.7)),
            ),
            MatchArmV2::new(
                Pattern::enum_variant(Some("Response".to_string()), "SD".to_string()),
                MatchBody::expr(Expr::float(0.3)),
            ),
            MatchArmV2::new(
                Pattern::enum_variant(Some("Response".to_string()), "PD".to_string()),
                MatchBody::expr(Expr::float(0.0)),
            ),
        ];

        let result = check_exhaustiveness(&enum_env, &scrutinee_ty, &arms, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_non_exhaustive_enum_match() {
        let enum_env = setup_response_enum();
        let scrutinee_ty = CoreType::Enum("Response".to_string());

        // Missing PD
        let arms = vec![
            MatchArmV2::new(
                Pattern::enum_variant(Some("Response".to_string()), "CR".to_string()),
                MatchBody::expr(Expr::float(1.0)),
            ),
            MatchArmV2::new(
                Pattern::enum_variant(Some("Response".to_string()), "PR".to_string()),
                MatchBody::expr(Expr::float(0.7)),
            ),
            MatchArmV2::new(
                Pattern::enum_variant(Some("Response".to_string()), "SD".to_string()),
                MatchBody::expr(Expr::float(0.3)),
            ),
        ];

        let result = check_exhaustiveness(&enum_env, &scrutinee_ty, &arms, None);
        assert!(result.is_err());
        match result.unwrap_err() {
            PatternError::NonExhaustive { missing, .. } => {
                assert!(missing.contains(&"PD".to_string()));
            }
            _ => panic!("Expected NonExhaustive error"),
        }
    }

    #[test]
    fn test_wildcard_makes_exhaustive() {
        let enum_env = setup_response_enum();
        let scrutinee_ty = CoreType::Enum("Response".to_string());

        let arms = vec![
            MatchArmV2::new(
                Pattern::enum_variant(Some("Response".to_string()), "CR".to_string()),
                MatchBody::expr(Expr::float(1.0)),
            ),
            MatchArmV2::new(Pattern::wildcard(), MatchBody::expr(Expr::float(0.0))),
        ];

        let result = check_exhaustiveness(&enum_env, &scrutinee_ty, &arms, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_or_pattern_coverage() {
        let enum_env = setup_response_enum();
        let scrutinee_ty = CoreType::Enum("Response".to_string());

        let arms = vec![
            MatchArmV2::new(
                Pattern::or(vec![
                    Pattern::enum_variant(Some("Response".to_string()), "CR".to_string()),
                    Pattern::enum_variant(Some("Response".to_string()), "PR".to_string()),
                ]),
                MatchBody::expr(Expr::float(1.0)),
            ),
            MatchArmV2::new(
                Pattern::or(vec![
                    Pattern::enum_variant(Some("Response".to_string()), "SD".to_string()),
                    Pattern::enum_variant(Some("Response".to_string()), "PD".to_string()),
                ]),
                MatchBody::expr(Expr::float(0.0)),
            ),
        ];

        let result = check_exhaustiveness(&enum_env, &scrutinee_ty, &arms, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_redundant_pattern_detection() {
        let arms = vec![
            MatchArmV2::new(Pattern::wildcard(), MatchBody::expr(Expr::float(0.0))),
            MatchArmV2::new(
                Pattern::enum_variant(Some("Response".to_string()), "CR".to_string()),
                MatchBody::expr(Expr::float(1.0)),
            ),
        ];

        let result = check_usefulness(&arms);
        assert!(result.is_err());
        match result.unwrap_err() {
            PatternError::Redundant {
                arm_index,
                covered_by,
                ..
            } => {
                assert_eq!(arm_index, 1);
                assert_eq!(covered_by, 0);
            }
            _ => panic!("Expected Redundant error"),
        }
    }

    #[test]
    fn test_guarded_arms_not_exhaustive() {
        let enum_env = setup_response_enum();
        let scrutinee_ty = CoreType::Enum("Response".to_string());

        // All variants have guards - not exhaustive
        let arms = vec![
            MatchArmV2::with_guard(
                Pattern::enum_variant(Some("Response".to_string()), "CR".to_string()),
                GuardExpr::new(Expr::bool_val(true)),
                MatchBody::expr(Expr::float(1.0)),
            ),
            MatchArmV2::with_guard(
                Pattern::enum_variant(Some("Response".to_string()), "PR".to_string()),
                GuardExpr::new(Expr::bool_val(true)),
                MatchBody::expr(Expr::float(0.7)),
            ),
            MatchArmV2::with_guard(
                Pattern::enum_variant(Some("Response".to_string()), "SD".to_string()),
                GuardExpr::new(Expr::bool_val(true)),
                MatchBody::expr(Expr::float(0.3)),
            ),
            MatchArmV2::with_guard(
                Pattern::enum_variant(Some("Response".to_string()), "PD".to_string()),
                GuardExpr::new(Expr::bool_val(true)),
                MatchBody::expr(Expr::float(0.0)),
            ),
        ];

        let result = check_exhaustiveness(&enum_env, &scrutinee_ty, &arms, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_bool_exhaustiveness() {
        let enum_env = EnumEnv::new();
        let scrutinee_ty = CoreType::Bool;

        // Both true and false covered
        let arms = vec![
            MatchArmV2::new(Pattern::bool_lit(true), MatchBody::expr(Expr::int(1))),
            MatchArmV2::new(Pattern::bool_lit(false), MatchBody::expr(Expr::int(0))),
        ];

        let result = check_exhaustiveness(&enum_env, &scrutinee_ty, &arms, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bool_non_exhaustive() {
        let enum_env = EnumEnv::new();
        let scrutinee_ty = CoreType::Bool;

        // Only true covered
        let arms = vec![MatchArmV2::new(
            Pattern::bool_lit(true),
            MatchBody::expr(Expr::int(1)),
        )];

        let result = check_exhaustiveness(&enum_env, &scrutinee_ty, &arms, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_or_binding_consistency() {
        // Bindings match
        let good_pattern = Pattern::or(vec![
            Pattern::binding("x".to_string(), false),
            Pattern::binding("x".to_string(), false),
        ]);
        assert!(check_or_pattern_bindings(&good_pattern).is_ok());

        // Bindings don't match
        let bad_pattern = Pattern::or(vec![
            Pattern::binding("x".to_string(), false),
            Pattern::binding("y".to_string(), false),
        ]);
        assert!(check_or_pattern_bindings(&bad_pattern).is_err());
    }

    #[test]
    fn test_pattern_coverage() {
        let wildcard_coverage = pattern_coverage(&Pattern::wildcard());
        assert!(wildcard_coverage.contains("*"));

        let variant_coverage = pattern_coverage(&Pattern::enum_variant(
            Some("Response".to_string()),
            "CR".to_string(),
        ));
        assert!(variant_coverage.contains("CR"));

        let or_coverage = pattern_coverage(&Pattern::or(vec![
            Pattern::enum_variant(Some("Response".to_string()), "CR".to_string()),
            Pattern::enum_variant(Some("Response".to_string()), "PR".to_string()),
        ]));
        assert!(or_coverage.contains("CR"));
        assert!(or_coverage.contains("PR"));
    }

    #[test]
    fn test_unknown_enum_error() {
        let enum_env = EnumEnv::new(); // Empty
        let domain_env = DomainEnv::new();
        let fn_env = FnEnv::new(domain_env);
        let type_env = TypeEnv::new(&fn_env);

        let result = check_pattern_type(
            &type_env,
            &enum_env,
            &Pattern::enum_variant(Some("Unknown".to_string()), "Var".to_string()),
            &CoreType::Enum("Unknown".to_string()),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_type_mismatch_error() {
        let enum_env = setup_response_enum();
        let domain_env = DomainEnv::new();
        let fn_env = FnEnv::new(domain_env);
        let type_env = TypeEnv::new(&fn_env);

        // Bool pattern against enum type
        let result = check_pattern_type(
            &type_env,
            &enum_env,
            &Pattern::bool_lit(true),
            &CoreType::Enum("Response".to_string()),
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            PatternError::TypeMismatch {
                expected, found, ..
            } => {
                assert!(expected.contains("Response"));
                assert_eq!(found, "Bool");
            }
            _ => panic!("Expected TypeMismatch error"),
        }
    }
}
