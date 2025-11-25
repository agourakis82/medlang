// Week 27: Type Checking for Enums and Pattern Matching
//
// This module provides type checking for:
// - Enum variant constructors (Response::CR)
// - Match expressions with exhaustiveness checking

use crate::ast::core_lang::{Expr, MatchArm, MatchPattern};
use crate::typecheck::{FnEnv, TypeEnv, TypeError};
use crate::types::core_lang::CoreType;
use crate::types::enum_types::EnumEnv;
use std::collections::HashSet;

/// Type check an enum variant constructor: Response::CR
pub fn typecheck_enum_variant(
    enum_env: &EnumEnv,
    enum_name: &str,
    variant_name: &str,
) -> Result<CoreType, TypeError> {
    // Verify enum exists
    if !enum_env.has_enum(enum_name) {
        return Err(TypeError::UnknownEnum(enum_name.to_string()));
    }

    // Verify variant exists
    if let Err(msg) = enum_env.verify_variant(enum_name, variant_name) {
        return Err(TypeError::UnknownEnumVariant {
            enum_name: enum_name.to_string(),
            variant_name: variant_name.to_string(),
            message: msg,
        });
    }

    Ok(CoreType::Enum(enum_name.to_string()))
}

/// Type check a match expression
pub fn typecheck_match(
    env: &mut TypeEnv,
    _fn_env: &FnEnv, // Unused but kept for API consistency if needed
    enum_env: &EnumEnv,
    scrutinee: &Expr,
    arms: &[MatchArm],
) -> Result<CoreType, TypeError> {
    // Type check the scrutinee
    let scrutinee_ty = crate::typecheck::core_lang::typecheck_expr(env, scrutinee)?;

    // Scrutinee must be an enum type
    let enum_name = match &scrutinee_ty {
        CoreType::Enum(name) => name.clone(),
        other => {
            return Err(TypeError::MatchNonEnum {
                found: other.as_str(),
            });
        }
    };

    // Get enum info
    let enum_info = enum_env
        .get_enum(&enum_name)
        .ok_or_else(|| TypeError::UnknownEnum(enum_name.clone()))?;

    // Track which variants are covered
    let mut covered_variants: HashSet<String> = HashSet::new();
    let mut has_wildcard = false;

    // Track result type (all arms must have the same type)
    let mut result_type: Option<CoreType> = None;

    for arm in arms {
        // Check pattern validity
        match &arm.pattern {
            MatchPattern::Variant {
                enum_name: pattern_enum,
                variant_name: pattern_variant,
            } => {
                // Pattern enum must match scrutinee enum
                if pattern_enum != &enum_name {
                    return Err(TypeError::MatchEnumMismatch {
                        scrutinee_enum: enum_name.clone(),
                        arm_enum: pattern_enum.clone(),
                    });
                }

                // Variant must exist
                if !enum_info.has_variant(pattern_variant) {
                    return Err(TypeError::UnknownEnumVariant {
                        enum_name: enum_name.clone(),
                        variant_name: pattern_variant.clone(),
                        message: format!(
                            "Enum {} does not have variant {}",
                            enum_name, pattern_variant
                        ),
                    });
                }

                // Track coverage
                covered_variants.insert(pattern_variant.clone());
            }
            MatchPattern::Wildcard => {
                has_wildcard = true;
            }
        }

        // Type check arm body
        let arm_ty = crate::typecheck::core_lang::typecheck_expr(env, &arm.body)?;

        // All arms must have the same type
        if let Some(ref expected_ty) = result_type {
            if &arm_ty != expected_ty {
                return Err(TypeError::MatchArmTypeMismatch {
                    expected: expected_ty.as_str(),
                    found: arm_ty.as_str(),
                });
            }
        } else {
            result_type = Some(arm_ty);
        }
    }

    // Check exhaustiveness (all variants covered or wildcard present)
    if !has_wildcard {
        let missing_variants: Vec<String> = enum_info
            .variants
            .iter()
            .filter(|v| !covered_variants.contains(*v))
            .cloned()
            .collect();

        if !missing_variants.is_empty() {
            return Err(TypeError::NonExhaustiveMatch {
                enum_name: enum_name.clone(),
                missing_variants,
            });
        }
    }

    // Return result type (or Unit if no arms)
    Ok(result_type.unwrap_or(CoreType::Unit))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::core_lang::Expr;
    use crate::typecheck::{DomainEnv, FnEnv, TypeEnv};
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

    #[test]
    fn test_typecheck_enum_variant_success() {
        let enum_env = setup_response_enum();

        let result = typecheck_enum_variant(&enum_env, "Response", "CR");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), CoreType::Enum("Response".to_string()));
    }

    #[test]
    fn test_typecheck_enum_variant_unknown_enum() {
        let enum_env = setup_response_enum();

        let result = typecheck_enum_variant(&enum_env, "UnknownEnum", "CR");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TypeError::UnknownEnum(_)));
    }

    #[test]
    fn test_typecheck_enum_variant_unknown_variant() {
        let enum_env = setup_response_enum();

        let result = typecheck_enum_variant(&enum_env, "Response", "Unknown");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TypeError::UnknownEnumVariant { .. }
        ));
    }

    #[test]
    fn test_typecheck_match_success() {
        let enum_env = setup_response_enum();
        let domain_env = DomainEnv::new();
        let fn_env = FnEnv::new(domain_env);
        let mut type_env = TypeEnv::new(&fn_env);
        
        type_env.add_var("resp".to_string(), CoreType::Enum("Response".to_string()));

        let scrutinee = Expr::var("resp".to_string());
        let arms = vec![
            MatchArm::new(
                MatchPattern::variant("Response".to_string(), "CR".to_string()),
                Expr::float(1.0),
            ),
            MatchArm::new(
                MatchPattern::variant("Response".to_string(), "PR".to_string()),
                Expr::float(0.7),
            ),
            MatchArm::new(MatchPattern::wildcard(), Expr::float(0.0)),
        ];

        let result = typecheck_match(&mut type_env, &fn_env, &enum_env, &scrutinee, &arms);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), CoreType::Float);
    }

    #[test]
    fn test_typecheck_match_non_enum_scrutinee() {
        let enum_env = setup_response_enum();
        let domain_env = DomainEnv::new();
        let fn_env = FnEnv::new(domain_env);
        let mut type_env = TypeEnv::new(&fn_env);
        
        type_env.add_var("x".to_string(), CoreType::Int);

        let scrutinee = Expr::var("x".to_string());
        let arms = vec![MatchArm::new(
            MatchPattern::variant("Response".to_string(), "CR".to_string()),
            Expr::float(1.0),
        )];

        let result = typecheck_match(&mut type_env, &fn_env, &enum_env, &scrutinee, &arms);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TypeError::MatchNonEnum { .. }
        ));
    }

    #[test]
    fn test_typecheck_match_arm_type_mismatch() {
        let enum_env = setup_response_enum();
        let domain_env = DomainEnv::new();
        let fn_env = FnEnv::new(domain_env);
        let mut type_env = TypeEnv::new(&fn_env);
        
        type_env.add_var("resp".to_string(), CoreType::Enum("Response".to_string()));

        let scrutinee = Expr::var("resp".to_string());
        let arms = vec![
            MatchArm::new(
                MatchPattern::variant("Response".to_string(), "CR".to_string()),
                Expr::float(1.0),
            ),
            MatchArm::new(
                MatchPattern::variant("Response".to_string(), "PR".to_string()),
                // Int literal? CoreLang has IntLiteral. 
                // CoreType::Int != CoreType::Float.
                Expr::int(1), 
            ),
            MatchArm::new(MatchPattern::wildcard(), Expr::float(0.0)),
        ];

        let result = typecheck_match(&mut type_env, &fn_env, &enum_env, &scrutinee, &arms);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TypeError::MatchArmTypeMismatch { .. }
        ));
    }
}