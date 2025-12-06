// Week 49: Enhanced Pattern Matching AST
//
// This module defines the AST for MedLang's pattern matching system with:
// - Pattern matching on enums, booleans, and literals
// - OR patterns (e.g., `Response::CR | Response::PR`)
// - Optional guards (e.g., `Response::CR if score > 0.5`)
// - Wildcard and binding patterns
// - Source locations for error reporting
// - Extensible design for future data-carrying variants
//
// Example usage:
// ```medlang
// match response {
//   Response::CR | Response::PR => "responder",
//   Response::SD if duration > 6 => "stable_long",
//   Response::SD => "stable_short",
//   _ => "non_responder",
// }
// ```

use crate::ast::{Ident, Span};
use serde::{Deserialize, Serialize};

// =============================================================================
// Pattern AST
// =============================================================================

/// A pattern with source location information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pattern {
    /// The kind of pattern
    pub kind: PatternKind,
    /// Source location for error reporting
    pub span: Option<Span>,
}

impl Pattern {
    pub fn new(kind: PatternKind) -> Self {
        Self { kind, span: None }
    }

    pub fn with_span(kind: PatternKind, span: Span) -> Self {
        Self {
            kind,
            span: Some(span),
        }
    }

    /// Create a wildcard pattern
    pub fn wildcard() -> Self {
        Self::new(PatternKind::Wildcard)
    }

    /// Create a binding pattern
    pub fn binding(name: Ident, mutable: bool) -> Self {
        Self::new(PatternKind::Binding { name, mutable })
    }

    /// Create an enum variant pattern (nullary)
    pub fn enum_variant(enum_name: Option<Ident>, variant: Ident) -> Self {
        Self::new(PatternKind::EnumVariant {
            enum_name,
            variant,
            payloads: Vec::new(),
        })
    }

    /// Create an enum variant pattern with payloads (for future data-carrying variants)
    pub fn enum_variant_with_payloads(
        enum_name: Option<Ident>,
        variant: Ident,
        payloads: Vec<Pattern>,
    ) -> Self {
        Self::new(PatternKind::EnumVariant {
            enum_name,
            variant,
            payloads,
        })
    }

    /// Create a boolean literal pattern
    pub fn bool_lit(value: bool) -> Self {
        Self::new(PatternKind::BoolLit(value))
    }

    /// Create an integer literal pattern
    pub fn int_lit(value: i64) -> Self {
        Self::new(PatternKind::IntLit(value))
    }

    /// Create a float literal pattern
    pub fn float_lit(value: f64) -> Self {
        Self::new(PatternKind::FloatLit(value))
    }

    /// Create a string literal pattern
    pub fn string_lit(value: String) -> Self {
        Self::new(PatternKind::StringLit(value))
    }

    /// Create an OR pattern
    pub fn or(patterns: Vec<Pattern>) -> Self {
        Self::new(PatternKind::Or(patterns))
    }

    /// Check if this pattern is a wildcard
    pub fn is_wildcard(&self) -> bool {
        matches!(self.kind, PatternKind::Wildcard)
    }

    /// Check if this pattern is a binding
    pub fn is_binding(&self) -> bool {
        matches!(self.kind, PatternKind::Binding { .. })
    }

    /// Check if this pattern is an OR pattern
    pub fn is_or(&self) -> bool {
        matches!(self.kind, PatternKind::Or(_))
    }

    /// Get all bindings introduced by this pattern
    pub fn bindings(&self) -> Vec<(&Ident, bool)> {
        match &self.kind {
            PatternKind::Wildcard => vec![],
            PatternKind::Binding { name, mutable } => vec![(name, *mutable)],
            PatternKind::EnumVariant { payloads, .. } => {
                payloads.iter().flat_map(|p| p.bindings()).collect()
            }
            PatternKind::BoolLit(_)
            | PatternKind::IntLit(_)
            | PatternKind::FloatLit(_)
            | PatternKind::StringLit(_) => vec![],
            PatternKind::Or(patterns) => {
                // For OR patterns, all alternatives must bind the same names
                // Return bindings from first alternative (validated elsewhere)
                patterns.first().map(|p| p.bindings()).unwrap_or_default()
            }
        }
    }

    /// Get all binding names as owned strings
    pub fn binding_names(&self) -> Vec<String> {
        self.bindings()
            .into_iter()
            .map(|(n, _)| n.clone())
            .collect()
    }
}

/// The different kinds of patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PatternKind {
    /// Wildcard pattern: `_`
    /// Matches anything, binds nothing
    Wildcard,

    /// Binding pattern: `x` or `mut x`
    /// Matches anything and binds to the name
    Binding { name: Ident, mutable: bool },

    /// Enum variant pattern: `Response::CR` or `Option::Some(x)`
    /// enum_name is optional when it can be inferred from context
    EnumVariant {
        enum_name: Option<Ident>,
        variant: Ident,
        /// Payload patterns for data-carrying variants (Week 49+)
        payloads: Vec<Pattern>,
    },

    /// Boolean literal pattern: `true` or `false`
    BoolLit(bool),

    /// Integer literal pattern: `0`, `1`, `-5`, etc.
    IntLit(i64),

    /// Float literal pattern: `0.0`, `1.5`, etc.
    FloatLit(f64),

    /// String literal pattern: `"hello"`
    StringLit(String),

    /// OR pattern: `p1 | p2 | p3`
    /// Matches if any of the sub-patterns match
    /// All alternatives must bind the same set of names with the same types
    Or(Vec<Pattern>),
}

// =============================================================================
// Match Arms with Guards
// =============================================================================

/// A match arm with pattern, optional guard, and body
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchArmV2 {
    /// The pattern to match
    pub pattern: Pattern,
    /// Optional guard expression (must evaluate to Bool)
    pub guard: Option<GuardExpr>,
    /// The body expression to evaluate if pattern matches (and guard passes)
    pub body: MatchBody,
    /// Source location
    pub span: Option<Span>,
}

impl MatchArmV2 {
    pub fn new(pattern: Pattern, body: MatchBody) -> Self {
        Self {
            pattern,
            guard: None,
            body,
            span: None,
        }
    }

    pub fn with_guard(pattern: Pattern, guard: GuardExpr, body: MatchBody) -> Self {
        Self {
            pattern,
            guard: Some(guard),
            body,
            span: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Check if this arm has a guard
    pub fn has_guard(&self) -> bool {
        self.guard.is_some()
    }

    /// Check if this arm is a catch-all (wildcard/binding without guard)
    pub fn is_catch_all(&self) -> bool {
        self.guard.is_none() && (self.pattern.is_wildcard() || self.pattern.is_binding())
    }
}

/// Guard expression for conditional pattern matching
/// `if <expr>` where expr must evaluate to Bool
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardExpr {
    /// The guard condition expression
    pub condition: Box<crate::ast::core_lang::Expr>,
    /// Source location
    pub span: Option<Span>,
}

impl GuardExpr {
    pub fn new(condition: crate::ast::core_lang::Expr) -> Self {
        Self {
            condition: Box::new(condition),
            span: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }
}

/// Match body - currently just an expression, but extensible for blocks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MatchBody {
    /// Simple expression body: `=> expr`
    Expr(crate::ast::core_lang::Expr),
    /// Block body: `=> { stmts }` (for future use)
    Block(crate::ast::core_lang::Block),
}

impl MatchBody {
    pub fn expr(e: crate::ast::core_lang::Expr) -> Self {
        MatchBody::Expr(e)
    }

    pub fn block(b: crate::ast::core_lang::Block) -> Self {
        MatchBody::Block(b)
    }
}

// =============================================================================
// Enhanced Match Expression
// =============================================================================

/// Enhanced match expression with v2 arms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchExprV2 {
    /// The expression being matched (scrutinee)
    pub scrutinee: Box<crate::ast::core_lang::Expr>,
    /// Match arms
    pub arms: Vec<MatchArmV2>,
    /// Source location
    pub span: Option<Span>,
}

impl MatchExprV2 {
    pub fn new(scrutinee: crate::ast::core_lang::Expr, arms: Vec<MatchArmV2>) -> Self {
        Self {
            scrutinee: Box::new(scrutinee),
            arms,
            span: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Check if any arm has a guard
    pub fn has_guards(&self) -> bool {
        self.arms.iter().any(|arm| arm.has_guard())
    }

    /// Check if there's a catch-all arm
    pub fn has_catch_all(&self) -> bool {
        self.arms.iter().any(|arm| arm.is_catch_all())
    }
}

// =============================================================================
// Pattern Matrix for Exhaustiveness Checking
// =============================================================================

/// Represents a column in the pattern matrix (for exhaustiveness checking)
#[derive(Debug, Clone, PartialEq)]
pub enum PatternColumn {
    /// Enum type column with set of possible variants
    Enum {
        enum_name: String,
        variants: Vec<String>,
    },
    /// Boolean column (true/false)
    Bool,
    /// Integer column (unbounded, needs wildcard)
    Int,
    /// Float column (unbounded, needs wildcard)
    Float,
    /// String column (unbounded, needs wildcard)
    String,
    /// Any type (used for wildcards)
    Any,
}

/// A row in the pattern matrix
#[derive(Debug, Clone)]
pub struct PatternRow {
    /// Patterns in this row (one per column)
    pub patterns: Vec<Pattern>,
    /// Index of the original match arm
    pub arm_index: usize,
    /// Whether this row has a guard (guards don't contribute to exhaustiveness)
    pub has_guard: bool,
}

impl PatternRow {
    pub fn new(patterns: Vec<Pattern>, arm_index: usize, has_guard: bool) -> Self {
        Self {
            patterns,
            arm_index,
            has_guard,
        }
    }
}

/// Pattern matrix for exhaustiveness and usefulness analysis
#[derive(Debug, Clone)]
pub struct PatternMatrix {
    /// Column types
    pub columns: Vec<PatternColumn>,
    /// Pattern rows
    pub rows: Vec<PatternRow>,
}

impl PatternMatrix {
    pub fn new(columns: Vec<PatternColumn>) -> Self {
        Self {
            columns,
            rows: Vec::new(),
        }
    }

    pub fn add_row(&mut self, row: PatternRow) {
        self.rows.push(row);
    }

    /// Check if matrix is empty (all patterns matched)
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Get number of columns
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// Get number of rows
    pub fn height(&self) -> usize {
        self.rows.len()
    }
}

// =============================================================================
// Error Types for Pattern Checking
// =============================================================================

/// Errors that can occur during pattern checking
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PatternError {
    /// Non-exhaustive match - missing patterns
    #[error("non-exhaustive match: missing patterns {missing:?}")]
    NonExhaustive {
        missing: Vec<String>,
        span: Option<Span>,
    },

    /// Unreachable pattern - pattern can never match
    #[error("unreachable pattern at arm {arm_index}")]
    Unreachable {
        arm_index: usize,
        span: Option<Span>,
    },

    /// Redundant pattern - pattern already covered by earlier arm
    #[error("redundant pattern at arm {arm_index}, already covered by arm {covered_by}")]
    Redundant {
        arm_index: usize,
        covered_by: usize,
        span: Option<Span>,
    },

    /// OR pattern binding mismatch - alternatives bind different names
    #[error("OR pattern alternatives bind different names: {left:?} vs {right:?}")]
    OrBindingMismatch {
        left: Vec<String>,
        right: Vec<String>,
        span: Option<Span>,
    },

    /// OR pattern type mismatch - alternatives have different types for same binding
    #[error("OR pattern binding '{name}' has different types in alternatives")]
    OrTypeMismatch { name: String, span: Option<Span> },

    /// Guard type error - guard doesn't evaluate to Bool
    #[error("guard expression must be Bool, found {found}")]
    GuardNotBool { found: String, span: Option<Span> },

    /// Unknown enum in pattern
    #[error("unknown enum '{enum_name}' in pattern")]
    UnknownEnum {
        enum_name: String,
        span: Option<Span>,
    },

    /// Unknown variant in pattern
    #[error("unknown variant '{variant}' in enum '{enum_name}'")]
    UnknownVariant {
        enum_name: String,
        variant: String,
        span: Option<Span>,
    },

    /// Type mismatch in pattern
    #[error("pattern type mismatch: expected {expected}, found {found}")]
    TypeMismatch {
        expected: String,
        found: String,
        span: Option<Span>,
    },

    /// Arity mismatch for data-carrying variant
    #[error("variant '{variant}' expects {expected} fields, found {found}")]
    ArityMismatch {
        variant: String,
        expected: usize,
        found: usize,
        span: Option<Span>,
    },
}

impl PatternError {
    /// Get the span associated with this error
    pub fn span(&self) -> Option<&Span> {
        match self {
            PatternError::NonExhaustive { span, .. } => span.as_ref(),
            PatternError::Unreachable { span, .. } => span.as_ref(),
            PatternError::Redundant { span, .. } => span.as_ref(),
            PatternError::OrBindingMismatch { span, .. } => span.as_ref(),
            PatternError::OrTypeMismatch { span, .. } => span.as_ref(),
            PatternError::GuardNotBool { span, .. } => span.as_ref(),
            PatternError::UnknownEnum { span, .. } => span.as_ref(),
            PatternError::UnknownVariant { span, .. } => span.as_ref(),
            PatternError::TypeMismatch { span, .. } => span.as_ref(),
            PatternError::ArityMismatch { span, .. } => span.as_ref(),
        }
    }

    /// Format error with source location
    pub fn format_with_location(&self) -> String {
        if let Some(span) = self.span() {
            format!("{}:{}: {}", span.line, span.column, self)
        } else {
            format!("{}", self)
        }
    }
}

// =============================================================================
// Pattern Utilities
// =============================================================================

/// Check if two patterns are compatible (could match the same value)
pub fn patterns_overlap(p1: &Pattern, p2: &Pattern) -> bool {
    use PatternKind::*;

    match (&p1.kind, &p2.kind) {
        // Wildcards and bindings overlap with everything
        (Wildcard, _) | (_, Wildcard) => true,
        (Binding { .. }, _) | (_, Binding { .. }) => true,

        // Enum variants overlap only if same variant
        (
            EnumVariant {
                enum_name: e1,
                variant: v1,
                payloads: p1,
            },
            EnumVariant {
                enum_name: e2,
                variant: v2,
                payloads: p2,
            },
        ) => {
            // If enum names are specified, they must match
            let enum_match = match (e1, e2) {
                (Some(n1), Some(n2)) => n1 == n2,
                _ => true, // Inferred enums assumed compatible
            };
            if !enum_match || v1 != v2 {
                return false;
            }
            // Check payload patterns
            if p1.len() != p2.len() {
                return false;
            }
            p1.iter()
                .zip(p2.iter())
                .all(|(a, b)| patterns_overlap(a, b))
        }

        // Literals overlap only if equal
        (BoolLit(b1), BoolLit(b2)) => b1 == b2,
        (IntLit(i1), IntLit(i2)) => i1 == i2,
        (FloatLit(f1), FloatLit(f2)) => (f1 - f2).abs() < f64::EPSILON,
        (StringLit(s1), StringLit(s2)) => s1 == s2,

        // OR patterns overlap if any alternative overlaps
        (Or(alts), other) | (other, Or(alts)) => alts
            .iter()
            .any(|alt| patterns_overlap(alt, &Pattern::new(other.clone()))),

        // Different pattern kinds don't overlap
        _ => false,
    }
}

/// Check if pattern p1 is more specific than or equal to p2
/// (i.e., every value matching p1 also matches p2)
pub fn pattern_subsumes(specific: &Pattern, general: &Pattern) -> bool {
    use PatternKind::*;

    match (&general.kind, &specific.kind) {
        // Wildcard/binding subsumes everything
        (Wildcard, _) | (Binding { .. }, _) => true,

        // Nothing subsumes wildcard except wildcard
        (_, Wildcard) => matches!(general.kind, Wildcard),
        (_, Binding { .. }) => matches!(general.kind, Wildcard | Binding { .. }),

        // Enum variant: must be same variant with subsuming payloads
        (
            EnumVariant {
                variant: v1,
                payloads: p1,
                ..
            },
            EnumVariant {
                variant: v2,
                payloads: p2,
                ..
            },
        ) => {
            if v1 != v2 || p1.len() != p2.len() {
                return false;
            }
            p1.iter()
                .zip(p2.iter())
                .all(|(g, s)| pattern_subsumes(s, g))
        }

        // Literals: must be equal
        (BoolLit(b1), BoolLit(b2)) => b1 == b2,
        (IntLit(i1), IntLit(i2)) => i1 == i2,
        (FloatLit(f1), FloatLit(f2)) => (f1 - f2).abs() < f64::EPSILON,
        (StringLit(s1), StringLit(s2)) => s1 == s2,

        // OR pattern on general side: specific must subsume at least one
        (Or(alts), _) => alts.iter().any(|alt| pattern_subsumes(specific, alt)),

        // OR pattern on specific side: all alternatives must be subsumed
        (_, Or(alts)) => alts.iter().all(|alt| pattern_subsumes(alt, general)),

        _ => false,
    }
}

/// Flatten OR patterns into a list of alternatives
pub fn flatten_or_pattern(pattern: &Pattern) -> Vec<&Pattern> {
    match &pattern.kind {
        PatternKind::Or(alts) => alts.iter().flat_map(flatten_or_pattern).collect(),
        _ => vec![pattern],
    }
}

/// Collect all enum variants mentioned in a pattern
pub fn collect_variants(pattern: &Pattern) -> Vec<(Option<&str>, &str)> {
    match &pattern.kind {
        PatternKind::EnumVariant {
            enum_name,
            variant,
            payloads,
        } => {
            let mut result = vec![(enum_name.as_deref(), variant.as_str())];
            for p in payloads {
                result.extend(collect_variants(p));
            }
            result
        }
        PatternKind::Or(alts) => alts.iter().flat_map(collect_variants).collect(),
        _ => vec![],
    }
}

// =============================================================================
// Display implementations
// =============================================================================

impl std::fmt::Display for Pattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::fmt::Display for PatternKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternKind::Wildcard => write!(f, "_"),
            PatternKind::Binding { name, mutable } => {
                if *mutable {
                    write!(f, "mut {}", name)
                } else {
                    write!(f, "{}", name)
                }
            }
            PatternKind::EnumVariant {
                enum_name,
                variant,
                payloads,
            } => {
                if let Some(en) = enum_name {
                    write!(f, "{}::{}", en, variant)?;
                } else {
                    write!(f, "{}", variant)?;
                }
                if !payloads.is_empty() {
                    write!(f, "(")?;
                    for (i, p) in payloads.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", p)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            PatternKind::BoolLit(b) => write!(f, "{}", b),
            PatternKind::IntLit(i) => write!(f, "{}", i),
            PatternKind::FloatLit(fl) => write!(f, "{}", fl),
            PatternKind::StringLit(s) => write!(f, "\"{}\"", s),
            PatternKind::Or(alts) => {
                for (i, alt) in alts.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{}", alt)?;
                }
                Ok(())
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_creation() {
        let wildcard = Pattern::wildcard();
        assert!(wildcard.is_wildcard());

        let binding = Pattern::binding("x".to_string(), false);
        assert!(binding.is_binding());
        assert_eq!(binding.binding_names(), vec!["x"]);

        let variant = Pattern::enum_variant(Some("Response".to_string()), "CR".to_string());
        assert!(!variant.is_wildcard());
        assert!(!variant.is_binding());
    }

    #[test]
    fn test_or_pattern() {
        let or_pat = Pattern::or(vec![
            Pattern::enum_variant(Some("Response".to_string()), "CR".to_string()),
            Pattern::enum_variant(Some("Response".to_string()), "PR".to_string()),
        ]);
        assert!(or_pat.is_or());
        assert_eq!(format!("{}", or_pat), "Response::CR | Response::PR");
    }

    #[test]
    fn test_literal_patterns() {
        let bool_pat = Pattern::bool_lit(true);
        assert_eq!(format!("{}", bool_pat), "true");

        let int_pat = Pattern::int_lit(42);
        assert_eq!(format!("{}", int_pat), "42");

        let float_pat = Pattern::float_lit(3.14);
        assert_eq!(format!("{}", float_pat), "3.14");

        let string_pat = Pattern::string_lit("hello".to_string());
        assert_eq!(format!("{}", string_pat), "\"hello\"");
    }

    #[test]
    fn test_patterns_overlap() {
        let wildcard = Pattern::wildcard();
        let variant_cr = Pattern::enum_variant(Some("Response".to_string()), "CR".to_string());
        let variant_pr = Pattern::enum_variant(Some("Response".to_string()), "PR".to_string());

        // Wildcard overlaps with everything
        assert!(patterns_overlap(&wildcard, &variant_cr));
        assert!(patterns_overlap(&variant_cr, &wildcard));

        // Same variant overlaps
        assert!(patterns_overlap(&variant_cr, &variant_cr));

        // Different variants don't overlap
        assert!(!patterns_overlap(&variant_cr, &variant_pr));
    }

    #[test]
    fn test_pattern_subsumes() {
        let wildcard = Pattern::wildcard();
        let variant_cr = Pattern::enum_variant(Some("Response".to_string()), "CR".to_string());

        // Wildcard subsumes everything
        assert!(pattern_subsumes(&variant_cr, &wildcard));
        assert!(pattern_subsumes(&wildcard, &wildcard));

        // Specific doesn't subsume wildcard
        assert!(!pattern_subsumes(&wildcard, &variant_cr));

        // Variant subsumes itself
        assert!(pattern_subsumes(&variant_cr, &variant_cr));
    }

    #[test]
    fn test_match_arm_v2() {
        use crate::ast::core_lang::Expr;

        let arm = MatchArmV2::new(
            Pattern::enum_variant(Some("Response".to_string()), "CR".to_string()),
            MatchBody::expr(Expr::float(1.0)),
        );
        assert!(!arm.has_guard());
        assert!(!arm.is_catch_all());

        let catch_all = MatchArmV2::new(Pattern::wildcard(), MatchBody::expr(Expr::float(0.0)));
        assert!(catch_all.is_catch_all());
    }

    #[test]
    fn test_match_arm_with_guard() {
        use crate::ast::core_lang::Expr;

        let guard = GuardExpr::new(Expr::bool_val(true));
        let arm = MatchArmV2::with_guard(
            Pattern::enum_variant(Some("Response".to_string()), "SD".to_string()),
            guard,
            MatchBody::expr(Expr::float(0.5)),
        );
        assert!(arm.has_guard());
        assert!(!arm.is_catch_all());
    }

    #[test]
    fn test_flatten_or_pattern() {
        let nested_or = Pattern::or(vec![
            Pattern::or(vec![
                Pattern::enum_variant(Some("R".to_string()), "A".to_string()),
                Pattern::enum_variant(Some("R".to_string()), "B".to_string()),
            ]),
            Pattern::enum_variant(Some("R".to_string()), "C".to_string()),
        ]);

        let flattened = flatten_or_pattern(&nested_or);
        assert_eq!(flattened.len(), 3);
    }

    #[test]
    fn test_collect_variants() {
        let pattern = Pattern::or(vec![
            Pattern::enum_variant(Some("Response".to_string()), "CR".to_string()),
            Pattern::enum_variant(Some("Response".to_string()), "PR".to_string()),
        ]);

        let variants = collect_variants(&pattern);
        assert_eq!(variants.len(), 2);
        assert!(variants.contains(&(Some("Response"), "CR")));
        assert!(variants.contains(&(Some("Response"), "PR")));
    }

    #[test]
    fn test_pattern_error_formatting() {
        let error = PatternError::NonExhaustive {
            missing: vec!["Response::PD".to_string()],
            span: Some(Span::new(10, 5, 20)),
        };
        let formatted = error.format_with_location();
        assert!(formatted.contains("10:5"));
        assert!(formatted.contains("non-exhaustive"));
    }

    #[test]
    fn test_binding_in_enum_payload() {
        let pattern = Pattern::enum_variant_with_payloads(
            Some("Option".to_string()),
            "Some".to_string(),
            vec![Pattern::binding("x".to_string(), false)],
        );

        let bindings = pattern.binding_names();
        assert_eq!(bindings, vec!["x"]);
    }
}
