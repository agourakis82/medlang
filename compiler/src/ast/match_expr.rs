// Week 27: Pattern Matching on Enums
//
// Match expressions provide type-safe pattern matching on enum variants:
//
// ```medlang
// match resp {
//   Response::CR => 1.0,
//   Response::PR => 0.7,
//   Response::SD => 0.0,
//   Response::PD => 0.0,
// }
// ```
//
// Or with a wildcard:
//
// ```medlang
// match resp {
//   Response::CR => 1.0,
//   Response::PR => 0.7,
//   _            => 0.0,
// }
// ```
//
// Note: This module provides helper types and utilities for match expressions.
// The actual MatchArm and MatchPattern types are defined in ast/mod.rs
// to be part of the main AST.

use crate::ast::{Expr, ExprKind, MatchArm, MatchPattern};

/// Helper functions for working with match expressions

/// Check if a match expression has a wildcard arm
pub fn has_wildcard(arms: &[MatchArm]) -> bool {
    arms.iter()
        .any(|arm| matches!(arm.pattern, MatchPattern::Wildcard))
}

/// Get all explicit variant patterns (non-wildcard) from match arms
pub fn variant_patterns(arms: &[MatchArm]) -> Vec<(&str, &str)> {
    arms.iter()
        .filter_map(|arm| match &arm.pattern {
            MatchPattern::Variant {
                enum_name,
                variant_name,
            } => Some((enum_name.as_str(), variant_name.as_str())),
            MatchPattern::Wildcard => None,
        })
        .collect()
}

impl MatchArm {
    /// Create a new match arm
    pub fn new(pattern: MatchPattern, body: Expr) -> Self {
        MatchArm { pattern, body }
    }
}

impl MatchPattern {
    /// Create a variant pattern
    pub fn variant(enum_name: Ident, variant_name: Ident) -> Self {
        MatchPattern::Variant {
            enum_name,
            variant_name,
        }
    }

    /// Create a wildcard pattern
    pub fn wildcard() -> Self {
        MatchPattern::Wildcard
    }

    /// Check if pattern is a wildcard
    pub fn is_wildcard(&self) -> bool {
        matches!(self, MatchPattern::Wildcard)
    }

    /// Get enum and variant names if this is a variant pattern
    pub fn as_variant(&self) -> Option<(&str, &str)> {
        match self {
            MatchPattern::Variant {
                enum_name,
                variant_name,
            } => Some((enum_name.as_str(), variant_name.as_str())),
            MatchPattern::Wildcard => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr;

    #[test]
    fn test_match_pattern_creation() {
        let variant = MatchPattern::variant("Response".to_string(), "CR".to_string());
        assert!(!variant.is_wildcard());
        assert_eq!(variant.as_variant(), Some(("Response", "CR")));

        let wildcard = MatchPattern::wildcard();
        assert!(wildcard.is_wildcard());
        assert_eq!(wildcard.as_variant(), None);
    }

    #[test]
    fn test_match_arm_creation() {
        let arm = MatchArm::new(
            MatchPattern::variant("Response".to_string(), "CR".to_string()),
            Expr::literal(1.0),
        );

        assert!(!arm.pattern.is_wildcard());
    }

    #[test]
    fn test_has_wildcard_helper() {
        let arms = vec![
            MatchArm::new(
                MatchPattern::variant("Response".to_string(), "CR".to_string()),
                Expr::literal(1.0),
            ),
            MatchArm::new(
                MatchPattern::variant("Response".to_string(), "PR".to_string()),
                Expr::literal(0.7),
            ),
            MatchArm::new(MatchPattern::wildcard(), Expr::literal(0.0)),
        ];

        assert!(has_wildcard(&arms));
    }

    #[test]
    fn test_variant_patterns_helper() {
        let arms = vec![
            MatchArm::new(
                MatchPattern::variant("Response".to_string(), "CR".to_string()),
                Expr::literal(1.0),
            ),
            MatchArm::new(
                MatchPattern::variant("Response".to_string(), "PR".to_string()),
                Expr::literal(0.7),
            ),
            MatchArm::new(MatchPattern::wildcard(), Expr::literal(0.0)),
        ];

        let patterns = variant_patterns(&arms);
        assert_eq!(patterns.len(), 2);
        assert!(patterns.contains(&("Response", "CR")));
        assert!(patterns.contains(&("Response", "PR")));
    }
}
