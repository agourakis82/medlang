// Week 49: IR Lowering for Pattern Matching
//
// This module lowers enhanced pattern matching (MatchExprV2) to a simplified
// IR representation suitable for code generation. The lowering process:
//
// 1. Converts pattern matching to decision trees or switch statements
// 2. Handles OR patterns by flattening alternatives
// 3. Compiles guards as conditional checks
// 4. Optimizes for common cases (single-arm, all-wildcard, etc.)
//
// The output IR uses a combination of:
// - IRSwitch: For enum/bool dispatch (jump table)
// - IRCond: For guards and fallback chains
// - IRLet: For binding pattern variables

use crate::ast::core_lang::Expr;
use crate::ast::pattern::{MatchArmV2, MatchBody, MatchExprV2, Pattern, PatternKind};
use crate::ast::Span;
use crate::ir::{IRBinaryOp, IRExpr};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Pattern IR Types
// =============================================================================

/// Lowered match expression in IR form
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRMatch {
    /// The scrutinee expression
    pub scrutinee: IRExpr,
    /// Type tag for the scrutinee (enum name, "bool", "int", etc.)
    pub scrutinee_type: String,
    /// The lowered match body
    pub body: IRMatchBody,
    /// Source span for error reporting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<Span>,
}

/// Body of a lowered match (decision tree form)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IRMatchBody {
    /// Switch on enum discriminant or bool value
    Switch {
        /// Map from variant/value tag to arm body
        cases: Vec<IRSwitchCase>,
        /// Default case (for wildcards)
        default: Option<Box<IRMatchArm>>,
    },

    /// Linear chain of conditionals (for guards or complex patterns)
    Cond {
        /// Conditions to check in order
        arms: Vec<IRCondArm>,
        /// Fallback if no condition matches
        fallback: Option<Box<IRExpr>>,
    },

    /// Single arm (optimized case)
    Single(Box<IRMatchArm>),

    /// Direct expression (no matching needed)
    Direct(IRExpr),
}

/// A case in a switch statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRSwitchCase {
    /// The tag value (variant index, bool as 0/1, int value)
    pub tag: IRTag,
    /// The arm body
    pub arm: IRMatchArm,
}

/// Tag value for switch dispatch
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum IRTag {
    /// Enum variant index
    EnumVariant {
        enum_name: String,
        variant: String,
        index: usize,
    },
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// String value
    String(String),
}

impl IRTag {
    pub fn enum_variant(enum_name: &str, variant: &str, index: usize) -> Self {
        IRTag::EnumVariant {
            enum_name: enum_name.to_string(),
            variant: variant.to_string(),
            index,
        }
    }

    pub fn as_int(&self) -> i64 {
        match self {
            IRTag::EnumVariant { index, .. } => *index as i64,
            IRTag::Bool(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
            IRTag::Int(i) => *i,
            IRTag::String(_) => 0, // Strings use hash-based dispatch
        }
    }
}

/// A conditional arm (guard-based dispatch)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRCondArm {
    /// Condition to check
    pub condition: IRExpr,
    /// Body if condition is true
    pub body: IRMatchArm,
}

/// A match arm body with optional bindings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRMatchArm {
    /// Bindings to introduce (name -> expression)
    pub bindings: Vec<IRBinding>,
    /// Optional guard condition
    pub guard: Option<IRExpr>,
    /// The body expression
    pub body: IRExpr,
    /// Original arm index (for debugging)
    pub source_arm: usize,
}

/// A binding introduced by pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRBinding {
    /// Variable name
    pub name: String,
    /// Expression to bind (usually projection from scrutinee)
    pub value: IRExpr,
    /// Whether the binding is mutable
    pub mutable: bool,
}

// =============================================================================
// Lowering Context
// =============================================================================

/// Context for lowering patterns to IR
pub struct LoweringContext {
    /// Map from enum name to variant indices
    enum_indices: HashMap<String, HashMap<String, usize>>,
    /// Counter for generating unique names
    name_counter: usize,
}

impl LoweringContext {
    pub fn new() -> Self {
        Self {
            enum_indices: HashMap::new(),
            name_counter: 0,
        }
    }

    /// Register an enum with its variants
    pub fn register_enum(&mut self, name: &str, variants: &[String]) {
        let indices: HashMap<String, usize> = variants
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();
        self.enum_indices.insert(name.to_string(), indices);
    }

    /// Get the index of a variant in an enum
    pub fn variant_index(&self, enum_name: &str, variant: &str) -> Option<usize> {
        self.enum_indices
            .get(enum_name)
            .and_then(|m| m.get(variant).copied())
    }

    /// Generate a fresh temporary name
    pub fn fresh_name(&mut self, prefix: &str) -> String {
        self.name_counter += 1;
        format!("{}_{}", prefix, self.name_counter)
    }
}

impl Default for LoweringContext {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Main Lowering Functions
// =============================================================================

/// Lower a MatchExprV2 to IR
pub fn lower_match(ctx: &mut LoweringContext, match_expr: &MatchExprV2) -> IRMatch {
    let scrutinee = lower_expr(&match_expr.scrutinee);
    let scrutinee_type = infer_scrutinee_type(&match_expr.arms);

    // Optimize for common cases
    let body = if match_expr.arms.is_empty() {
        // Empty match - should be caught by type checker
        IRMatchBody::Direct(IRExpr::Literal(0.0))
    } else if match_expr.arms.len() == 1 && is_catch_all(&match_expr.arms[0].pattern) {
        // Single catch-all arm
        let arm = lower_arm(ctx, &match_expr.arms[0], &scrutinee, 0);
        IRMatchBody::Single(Box::new(arm))
    } else if all_simple_enum_patterns(&match_expr.arms) && !has_guards(&match_expr.arms) {
        // All simple enum patterns without guards - use switch
        lower_to_switch(ctx, &match_expr.arms, &scrutinee, &scrutinee_type)
    } else {
        // General case - use conditional chain
        lower_to_cond(ctx, &match_expr.arms, &scrutinee, &scrutinee_type)
    };

    IRMatch {
        scrutinee,
        scrutinee_type,
        body,
        span: match_expr.span.clone(),
    }
}

/// Lower arms to a switch statement
fn lower_to_switch(
    ctx: &mut LoweringContext,
    arms: &[MatchArmV2],
    scrutinee: &IRExpr,
    scrutinee_type: &str,
) -> IRMatchBody {
    let mut cases: Vec<IRSwitchCase> = Vec::new();
    let mut default: Option<Box<IRMatchArm>> = None;

    for (arm_idx, arm) in arms.iter().enumerate() {
        if is_catch_all(&arm.pattern) {
            default = Some(Box::new(lower_arm(ctx, arm, scrutinee, arm_idx)));
            break;
        }

        // Handle OR patterns by adding multiple cases for the same arm
        let tags = pattern_to_tags(ctx, &arm.pattern, scrutinee_type);
        let lowered_arm = lower_arm(ctx, arm, scrutinee, arm_idx);

        for tag in tags {
            cases.push(IRSwitchCase {
                tag,
                arm: lowered_arm.clone(),
            });
        }
    }

    IRMatchBody::Switch { cases, default }
}

/// Lower arms to a conditional chain
fn lower_to_cond(
    ctx: &mut LoweringContext,
    arms: &[MatchArmV2],
    scrutinee: &IRExpr,
    scrutinee_type: &str,
) -> IRMatchBody {
    let mut cond_arms: Vec<IRCondArm> = Vec::new();
    let mut fallback: Option<Box<IRExpr>> = None;

    for (arm_idx, arm) in arms.iter().enumerate() {
        if is_catch_all(&arm.pattern) && arm.guard.is_none() {
            // Unconditional catch-all becomes the fallback
            let lowered = lower_arm(ctx, arm, scrutinee, arm_idx);
            fallback = Some(Box::new(wrap_arm_with_bindings(&lowered)));
            break;
        }

        let condition = build_pattern_condition(ctx, &arm.pattern, scrutinee, scrutinee_type);
        let lowered_arm = lower_arm(ctx, arm, scrutinee, arm_idx);

        cond_arms.push(IRCondArm {
            condition,
            body: lowered_arm,
        });
    }

    IRMatchBody::Cond {
        arms: cond_arms,
        fallback,
    }
}

/// Lower a single match arm
fn lower_arm(
    ctx: &mut LoweringContext,
    arm: &MatchArmV2,
    scrutinee: &IRExpr,
    arm_idx: usize,
) -> IRMatchArm {
    let bindings = extract_bindings(&arm.pattern, scrutinee);
    let guard = arm.guard.as_ref().map(|g| lower_expr(&g.condition));
    let body = match &arm.body {
        MatchBody::Expr(e) => lower_expr(e),
        MatchBody::Block(b) => lower_block(b),
    };

    IRMatchArm {
        bindings,
        guard,
        body,
        source_arm: arm_idx,
    }
}

/// Extract bindings from a pattern
fn extract_bindings(pattern: &Pattern, scrutinee: &IRExpr) -> Vec<IRBinding> {
    let mut bindings = Vec::new();

    match &pattern.kind {
        PatternKind::Binding { name, mutable } => {
            bindings.push(IRBinding {
                name: name.clone(),
                value: scrutinee.clone(),
                mutable: *mutable,
            });
        }
        PatternKind::EnumVariant { payloads, .. } => {
            // For data-carrying variants, extract payload bindings
            for (i, payload) in payloads.iter().enumerate() {
                let field_access = IRExpr::Call(format!("__field_{}", i), vec![scrutinee.clone()]);
                bindings.extend(extract_bindings(payload, &field_access));
            }
        }
        PatternKind::Or(alternatives) => {
            // OR patterns bind from the first alternative
            if let Some(first) = alternatives.first() {
                bindings.extend(extract_bindings(first, scrutinee));
            }
        }
        _ => {}
    }

    bindings
}

/// Convert a pattern to IR tags for switch dispatch
fn pattern_to_tags(ctx: &LoweringContext, pattern: &Pattern, scrutinee_type: &str) -> Vec<IRTag> {
    match &pattern.kind {
        PatternKind::EnumVariant {
            enum_name, variant, ..
        } => {
            let en = enum_name
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or(scrutinee_type);
            let idx = ctx.variant_index(en, variant).unwrap_or(0);
            vec![IRTag::enum_variant(en, variant, idx)]
        }
        PatternKind::BoolLit(b) => vec![IRTag::Bool(*b)],
        PatternKind::IntLit(i) => vec![IRTag::Int(*i)],
        PatternKind::StringLit(s) => vec![IRTag::String(s.clone())],
        PatternKind::Or(alternatives) => alternatives
            .iter()
            .flat_map(|alt| pattern_to_tags(ctx, alt, scrutinee_type))
            .collect(),
        PatternKind::Wildcard | PatternKind::Binding { .. } => {
            // These don't generate specific tags
            vec![]
        }
        PatternKind::FloatLit(_) => {
            // Floats shouldn't be used in switch, handled in cond
            vec![]
        }
    }
}

/// Build a condition expression for a pattern
fn build_pattern_condition(
    ctx: &LoweringContext,
    pattern: &Pattern,
    scrutinee: &IRExpr,
    scrutinee_type: &str,
) -> IRExpr {
    match &pattern.kind {
        PatternKind::Wildcard | PatternKind::Binding { .. } => {
            // Always matches
            IRExpr::Literal(1.0) // true
        }
        PatternKind::EnumVariant {
            enum_name, variant, ..
        } => {
            let en = enum_name
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or(scrutinee_type);
            let idx = ctx.variant_index(en, variant).unwrap_or(0);
            // scrutinee.__tag == idx
            IRExpr::Binary(
                IRBinaryOp::Sub, // Using Sub as equality check placeholder
                Box::new(IRExpr::Call("__tag".to_string(), vec![scrutinee.clone()])),
                Box::new(IRExpr::Literal(idx as f64)),
            )
        }
        PatternKind::BoolLit(b) => {
            let val = if *b { 1.0 } else { 0.0 };
            IRExpr::Binary(
                IRBinaryOp::Sub,
                Box::new(scrutinee.clone()),
                Box::new(IRExpr::Literal(val)),
            )
        }
        PatternKind::IntLit(i) => IRExpr::Binary(
            IRBinaryOp::Sub,
            Box::new(scrutinee.clone()),
            Box::new(IRExpr::Literal(*i as f64)),
        ),
        PatternKind::FloatLit(f) => IRExpr::Binary(
            IRBinaryOp::Sub,
            Box::new(scrutinee.clone()),
            Box::new(IRExpr::Literal(*f)),
        ),
        PatternKind::StringLit(s) => {
            // String comparison - use call to __str_eq
            IRExpr::Call(
                "__str_eq".to_string(),
                vec![scrutinee.clone(), IRExpr::Var(format!("\"{}\"", s))],
            )
        }
        PatternKind::Or(alternatives) => {
            // Any alternative matches
            let conditions: Vec<IRExpr> = alternatives
                .iter()
                .map(|alt| build_pattern_condition(ctx, alt, scrutinee, scrutinee_type))
                .collect();

            // Combine with logical OR (using addition as placeholder)
            conditions
                .into_iter()
                .reduce(|acc, cond| IRExpr::Binary(IRBinaryOp::Add, Box::new(acc), Box::new(cond)))
                .unwrap_or(IRExpr::Literal(0.0))
        }
    }
}

/// Wrap an arm with its bindings as let expressions
fn wrap_arm_with_bindings(arm: &IRMatchArm) -> IRExpr {
    // In a real implementation, this would generate nested lets
    // For now, we just return the body
    arm.body.clone()
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Lower an AST expression to IR expression
fn lower_expr(expr: &Expr) -> IRExpr {
    match expr {
        Expr::IntLiteral(i) => IRExpr::Literal(*i as f64),
        Expr::FloatLiteral(f) => IRExpr::Literal(*f),
        Expr::BoolLiteral(b) => IRExpr::Literal(if *b { 1.0 } else { 0.0 }),
        Expr::StringLiteral(s) => IRExpr::Var(format!("\"{}\"", s)),
        Expr::Var(name) => IRExpr::Var(name.clone()),
        Expr::Call { callee, args } => {
            let callee_name = match &**callee {
                Expr::Var(name) => name.clone(),
                _ => "__unknown".to_string(),
            };
            IRExpr::Call(callee_name, args.iter().map(lower_expr).collect())
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => IRExpr::Call(
            "__if".to_string(),
            vec![
                lower_expr(cond),
                lower_expr(then_branch),
                lower_expr(else_branch),
            ],
        ),
        Expr::EnumVariant(enum_name, variant) => {
            IRExpr::Call(format!("{}::{}", enum_name, variant), vec![])
        }
        Expr::Match { scrutinee, arms } => {
            // Legacy match - convert to call for now
            IRExpr::Call("__match".to_string(), vec![lower_expr(scrutinee)])
        }
        Expr::Record(fields) => IRExpr::Call(
            "__record".to_string(),
            fields.iter().map(|(_, e)| lower_expr(e)).collect(),
        ),
        Expr::FieldAccess { target, field } => {
            IRExpr::Call(format!("__field_{}", field), vec![lower_expr(target)])
        }
        Expr::BlockExpr(block) => lower_block(block),
    }
}

/// Lower a block to IR
fn lower_block(block: &crate::ast::core_lang::Block) -> IRExpr {
    // For simplicity, return the last expression or a unit value
    if let Some(last_stmt) = block.stmts.last() {
        match last_stmt {
            crate::ast::core_lang::Stmt::Expr(e) => lower_expr(e),
            _ => IRExpr::Literal(0.0), // Unit
        }
    } else {
        IRExpr::Literal(0.0) // Unit
    }
}

/// Infer the scrutinee type from arm patterns
fn infer_scrutinee_type(arms: &[MatchArmV2]) -> String {
    for arm in arms {
        if let Some(ty) = pattern_type(&arm.pattern) {
            return ty;
        }
    }
    "unknown".to_string()
}

/// Get the type implied by a pattern
fn pattern_type(pattern: &Pattern) -> Option<String> {
    match &pattern.kind {
        PatternKind::EnumVariant { enum_name, .. } => {
            enum_name.clone().or(Some("enum".to_string()))
        }
        PatternKind::BoolLit(_) => Some("bool".to_string()),
        PatternKind::IntLit(_) => Some("int".to_string()),
        PatternKind::FloatLit(_) => Some("float".to_string()),
        PatternKind::StringLit(_) => Some("string".to_string()),
        PatternKind::Or(alts) => alts.first().and_then(pattern_type),
        PatternKind::Wildcard | PatternKind::Binding { .. } => None,
    }
}

/// Check if a pattern is a catch-all (wildcard or binding)
fn is_catch_all(pattern: &Pattern) -> bool {
    matches!(
        pattern.kind,
        PatternKind::Wildcard | PatternKind::Binding { .. }
    )
}

/// Check if all patterns are simple enum patterns
fn all_simple_enum_patterns(arms: &[MatchArmV2]) -> bool {
    arms.iter().all(|arm| {
        matches!(
            arm.pattern.kind,
            PatternKind::EnumVariant { .. }
                | PatternKind::Wildcard
                | PatternKind::Binding { .. }
                | PatternKind::Or(_)
        )
    })
}

/// Check if any arm has a guard
fn has_guards(arms: &[MatchArmV2]) -> bool {
    arms.iter().any(|arm| arm.guard.is_some())
}

// =============================================================================
// Code Generation Helpers
// =============================================================================

/// Generate a unique label for a match arm
pub fn arm_label(match_id: usize, arm_idx: usize) -> String {
    format!("match_{}_arm_{}", match_id, arm_idx)
}

/// Generate the default case label
pub fn default_label(match_id: usize) -> String {
    format!("match_{}_default", match_id)
}

/// Generate the exit label for a match
pub fn exit_label(match_id: usize) -> String {
    format!("match_{}_exit", match_id)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::core_lang::Expr;
    use crate::ast::pattern::{GuardExpr, MatchArmV2, MatchBody, MatchExprV2, Pattern};

    fn setup_context() -> LoweringContext {
        let mut ctx = LoweringContext::new();
        ctx.register_enum(
            "Response",
            &[
                "CR".to_string(),
                "PR".to_string(),
                "SD".to_string(),
                "PD".to_string(),
            ],
        );
        ctx
    }

    #[test]
    fn test_lower_simple_enum_match() {
        let mut ctx = setup_context();

        let match_expr = MatchExprV2::new(
            Expr::var("resp".to_string()),
            vec![
                MatchArmV2::new(
                    Pattern::enum_variant(Some("Response".to_string()), "CR".to_string()),
                    MatchBody::expr(Expr::float(1.0)),
                ),
                MatchArmV2::new(
                    Pattern::enum_variant(Some("Response".to_string()), "PR".to_string()),
                    MatchBody::expr(Expr::float(0.7)),
                ),
                MatchArmV2::new(Pattern::wildcard(), MatchBody::expr(Expr::float(0.0))),
            ],
        );

        let ir = lower_match(&mut ctx, &match_expr);

        assert_eq!(ir.scrutinee_type, "Response");
        match ir.body {
            IRMatchBody::Switch { cases, default } => {
                assert_eq!(cases.len(), 2);
                assert!(default.is_some());
            }
            _ => panic!("Expected Switch body"),
        }
    }

    #[test]
    fn test_lower_or_pattern() {
        let mut ctx = setup_context();

        let match_expr = MatchExprV2::new(
            Expr::var("resp".to_string()),
            vec![
                MatchArmV2::new(
                    Pattern::or(vec![
                        Pattern::enum_variant(Some("Response".to_string()), "CR".to_string()),
                        Pattern::enum_variant(Some("Response".to_string()), "PR".to_string()),
                    ]),
                    MatchBody::expr(Expr::float(1.0)),
                ),
                MatchArmV2::new(Pattern::wildcard(), MatchBody::expr(Expr::float(0.0))),
            ],
        );

        let ir = lower_match(&mut ctx, &match_expr);

        match ir.body {
            IRMatchBody::Switch { cases, .. } => {
                // OR pattern should create 2 cases pointing to same arm
                assert_eq!(cases.len(), 2);
                assert_eq!(cases[0].arm.source_arm, cases[1].arm.source_arm);
            }
            _ => panic!("Expected Switch body"),
        }
    }

    #[test]
    fn test_lower_with_guard() {
        let mut ctx = setup_context();

        let match_expr = MatchExprV2::new(
            Expr::var("resp".to_string()),
            vec![
                MatchArmV2::with_guard(
                    Pattern::enum_variant(Some("Response".to_string()), "CR".to_string()),
                    GuardExpr::new(Expr::bool_val(true)),
                    MatchBody::expr(Expr::float(1.0)),
                ),
                MatchArmV2::new(Pattern::wildcard(), MatchBody::expr(Expr::float(0.0))),
            ],
        );

        let ir = lower_match(&mut ctx, &match_expr);

        // Guards force conditional lowering
        match ir.body {
            IRMatchBody::Cond { arms, fallback } => {
                assert_eq!(arms.len(), 1);
                assert!(arms[0].body.guard.is_some());
                assert!(fallback.is_some());
            }
            _ => panic!("Expected Cond body for guarded match"),
        }
    }

    #[test]
    fn test_lower_single_catch_all() {
        let mut ctx = setup_context();

        let match_expr = MatchExprV2::new(
            Expr::var("x".to_string()),
            vec![MatchArmV2::new(
                Pattern::binding("y".to_string(), false),
                MatchBody::expr(Expr::var("y".to_string())),
            )],
        );

        let ir = lower_match(&mut ctx, &match_expr);

        match ir.body {
            IRMatchBody::Single(arm) => {
                assert_eq!(arm.bindings.len(), 1);
                assert_eq!(arm.bindings[0].name, "y");
            }
            _ => panic!("Expected Single body"),
        }
    }

    #[test]
    fn test_variant_index_lookup() {
        let ctx = setup_context();

        assert_eq!(ctx.variant_index("Response", "CR"), Some(0));
        assert_eq!(ctx.variant_index("Response", "PR"), Some(1));
        assert_eq!(ctx.variant_index("Response", "SD"), Some(2));
        assert_eq!(ctx.variant_index("Response", "PD"), Some(3));
        assert_eq!(ctx.variant_index("Response", "Unknown"), None);
    }

    #[test]
    fn test_ir_tag_as_int() {
        let tag1 = IRTag::enum_variant("Response", "CR", 0);
        assert_eq!(tag1.as_int(), 0);

        let tag2 = IRTag::Bool(true);
        assert_eq!(tag2.as_int(), 1);

        let tag3 = IRTag::Int(42);
        assert_eq!(tag3.as_int(), 42);
    }

    #[test]
    fn test_extract_bindings() {
        let scrutinee = IRExpr::Var("x".to_string());

        let pattern = Pattern::binding("y".to_string(), true);
        let bindings = extract_bindings(&pattern, &scrutinee);

        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].name, "y");
        assert!(bindings[0].mutable);
    }

    #[test]
    fn test_pattern_to_tags() {
        let ctx = setup_context();

        let pattern = Pattern::or(vec![
            Pattern::enum_variant(Some("Response".to_string()), "CR".to_string()),
            Pattern::enum_variant(Some("Response".to_string()), "PR".to_string()),
        ]);

        let tags = pattern_to_tags(&ctx, &pattern, "Response");
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn test_fresh_name_generation() {
        let mut ctx = LoweringContext::new();

        let name1 = ctx.fresh_name("tmp");
        let name2 = ctx.fresh_name("tmp");
        let name3 = ctx.fresh_name("var");

        assert_eq!(name1, "tmp_1");
        assert_eq!(name2, "tmp_2");
        assert_eq!(name3, "var_3");
    }
}
