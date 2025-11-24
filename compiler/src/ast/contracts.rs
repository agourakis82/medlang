//! Week 28: Contracts, Invariants & Assertions AST
//!
//! This module defines AST structures for design-by-contract features:
//! - Function contracts (requires/ensures)
//! - Model/Policy invariant blocks
//! - Assert statements/expressions

use crate::ast::{Expr, Ident, Span};
use serde::{Deserialize, Serialize};

// =============================================================================
// Function Contracts
// =============================================================================

/// Function contract: preconditions (requires) and postconditions (ensures)
///
/// Example:
/// ```medlang
/// fn fit_model(model: Model, data: EvidenceResult) -> FitResult
///   requires data.n_patients > 0
///   requires model.is_valid()
///   ensures result.converged == true
/// { ... }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FnContract {
    /// Preconditions: must hold when function is called
    pub requires: Vec<ContractClause>,

    /// Postconditions: must hold when function returns
    pub ensures: Vec<ContractClause>,
}

impl FnContract {
    pub fn new() -> Self {
        Self {
            requires: Vec::new(),
            ensures: Vec::new(),
        }
    }

    pub fn with_requires(mut self, clauses: Vec<ContractClause>) -> Self {
        self.requires = clauses;
        self
    }

    pub fn with_ensures(mut self, clauses: Vec<ContractClause>) -> Self {
        self.ensures = clauses;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.requires.is_empty() && self.ensures.is_empty()
    }

    pub fn has_requires(&self) -> bool {
        !self.requires.is_empty()
    }

    pub fn has_ensures(&self) -> bool {
        !self.ensures.is_empty()
    }
}

impl Default for FnContract {
    fn default() -> Self {
        Self::new()
    }
}

/// A single contract clause: an expression with optional label
///
/// Example:
/// ```medlang
/// requires data.n_patients > 0, "need at least one patient"
/// ensures result.converged, "fit must converge"
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractClause {
    /// The boolean condition that must hold
    pub condition: Expr,

    /// Optional human-readable label for error messages
    pub label: Option<String>,

    /// Source location for error reporting
    pub span: Option<Span>,
}

impl ContractClause {
    pub fn new(condition: Expr) -> Self {
        Self {
            condition,
            label: None,
            span: None,
        }
    }

    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }
}

// =============================================================================
// Invariant Blocks
// =============================================================================

/// Invariant block for models, policies, or other domain constructs
///
/// Example:
/// ```medlang
/// model PK_OneCompOral {
///   state A_gut : DoseMass;
///   param CL : Clearance;
///
///   invariants {
///     CL > 0.0_L_per_h, "clearance must be positive";
///     V > 0.0_L, "volume must be positive";
///     A_gut >= 0.0_mg, "drug amount cannot be negative";
///   }
///
///   dA_gut/dt = -ka * A_gut;
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InvariantBlock {
    /// List of invariant clauses that must always hold
    pub clauses: Vec<ContractClause>,

    /// Source location for error reporting
    pub span: Option<Span>,
}

impl InvariantBlock {
    pub fn new(clauses: Vec<ContractClause>) -> Self {
        Self {
            clauses,
            span: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }

    pub fn len(&self) -> usize {
        self.clauses.len()
    }
}

// =============================================================================
// Assert Statements
// =============================================================================

/// Assert statement/expression for runtime checks
///
/// Can be used as:
/// 1. Statement: `assert x > 0, "x must be positive";`
/// 2. Expression: `let y = assert x > 0;` (returns unit on success, panics on failure)
///
/// Example:
/// ```medlang
/// fn simulate_trial(protocol: Protocol, n_patients: Int) -> SimulationResult {
///   assert n_patients > 0, "need at least one patient";
///   assert protocol.arms.len() > 0, "protocol must have arms";
///
///   // ... simulation logic
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssertStmt {
    /// The boolean condition to check
    pub condition: Expr,

    /// Optional failure message (for better error reporting)
    pub message: Option<String>,

    /// Source location for error reporting
    pub span: Option<Span>,
}

impl AssertStmt {
    pub fn new(condition: Expr) -> Self {
        Self {
            condition,
            message: None,
            span: None,
        }
    }

    pub fn with_message(mut self, message: String) -> Self {
        self.message = Some(message);
        self
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }
}

// =============================================================================
// Contract Violation Tracking
// =============================================================================

/// Information about a contract violation (for runtime error reporting)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractViolation {
    /// Type of contract that was violated
    pub kind: ContractKind,

    /// The function, model, or policy where the violation occurred
    pub location: String,

    /// The specific clause that failed
    pub clause: String,

    /// Optional user-provided label from the contract
    pub label: Option<String>,

    /// Runtime values that caused the violation (for debugging)
    pub context: Vec<(Ident, String)>, // (variable_name, value_repr)
}

impl ContractViolation {
    pub fn format_error(&self) -> String {
        let kind_str = match self.kind {
            ContractKind::Precondition => "Precondition",
            ContractKind::Postcondition => "Postcondition",
            ContractKind::Invariant => "Invariant",
            ContractKind::Assertion => "Assertion",
        };

        let label_str = self
            .label
            .as_ref()
            .map(|l| format!(": {}", l))
            .unwrap_or_default();

        let context_str = if self.context.is_empty() {
            String::new()
        } else {
            let ctx = self
                .context
                .iter()
                .map(|(name, val)| format!("  {} = {}", name, val))
                .collect::<Vec<_>>()
                .join("\n");
            format!("\n\nContext:\n{}", ctx)
        };

        format!(
            "{} violated in {}{}\n\nFailed clause: {}{}",
            kind_str, self.location, label_str, self.clause, context_str
        )
    }
}

/// Kind of contract that can be violated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContractKind {
    /// requires clause failed
    Precondition,

    /// ensures clause failed
    Postcondition,

    /// invariants block clause failed
    Invariant,

    /// assert statement failed
    Assertion,
}

impl ContractKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ContractKind::Precondition => "precondition",
            ContractKind::Postcondition => "postcondition",
            ContractKind::Invariant => "invariant",
            ContractKind::Assertion => "assertion",
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ExprKind;

    fn bool_expr(value: bool) -> Expr {
        Expr {
            kind: ExprKind::Literal(crate::ast::Literal::Float(if value { 1.0 } else { 0.0 })),
            span: None,
        }
    }

    #[test]
    fn test_fn_contract_creation() {
        let contract =
            FnContract::new()
                .with_requires(vec![ContractClause::new(bool_expr(true))
                    .with_label("test precondition".to_string())])
                .with_ensures(vec![ContractClause::new(bool_expr(true))
                    .with_label("test postcondition".to_string())]);

        assert!(contract.has_requires());
        assert!(contract.has_ensures());
        assert!(!contract.is_empty());
        assert_eq!(contract.requires.len(), 1);
        assert_eq!(contract.ensures.len(), 1);
    }

    #[test]
    fn test_empty_contract() {
        let contract = FnContract::new();
        assert!(contract.is_empty());
        assert!(!contract.has_requires());
        assert!(!contract.has_ensures());
    }

    #[test]
    fn test_contract_clause_with_label() {
        let clause = ContractClause::new(bool_expr(true))
            .with_label("patients must be positive".to_string());

        assert!(clause.label.is_some());
        assert_eq!(
            clause.label.unwrap(),
            "patients must be positive".to_string()
        );
    }

    #[test]
    fn test_invariant_block() {
        let clauses = vec![
            ContractClause::new(bool_expr(true)).with_label("CL > 0".to_string()),
            ContractClause::new(bool_expr(true)).with_label("V > 0".to_string()),
        ];

        let inv_block = InvariantBlock::new(clauses);
        assert_eq!(inv_block.len(), 2);
        assert!(!inv_block.is_empty());
    }

    #[test]
    fn test_assert_stmt() {
        let assert_stmt = AssertStmt::new(bool_expr(true))
            .with_message("n_patients must be positive".to_string());

        assert!(assert_stmt.message.is_some());
        assert_eq!(
            assert_stmt.message.unwrap(),
            "n_patients must be positive".to_string()
        );
    }

    #[test]
    fn test_contract_violation_formatting() {
        let violation = ContractViolation {
            kind: ContractKind::Precondition,
            location: "fit_model".to_string(),
            clause: "data.n_patients > 0".to_string(),
            label: Some("need at least one patient".to_string()),
            context: vec![("data.n_patients".to_string(), "0".to_string())],
        };

        let error_msg = violation.format_error();
        assert!(error_msg.contains("Precondition violated"));
        assert!(error_msg.contains("fit_model"));
        assert!(error_msg.contains("need at least one patient"));
        assert!(error_msg.contains("data.n_patients > 0"));
        assert!(error_msg.contains("data.n_patients = 0"));
    }

    #[test]
    fn test_contract_kind_display() {
        assert_eq!(ContractKind::Precondition.as_str(), "precondition");
        assert_eq!(ContractKind::Postcondition.as_str(), "postcondition");
        assert_eq!(ContractKind::Invariant.as_str(), "invariant");
        assert_eq!(ContractKind::Assertion.as_str(), "assertion");
    }
}
