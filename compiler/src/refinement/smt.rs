//! SMT Solver Interface for Refinement Types
//!
//! Translates refinement predicates to SMT-LIB2 format and invokes Z3.

use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

use super::error::Counterexample;
use super::syntax::{ArithOp, CompareOp, Predicate, RefinedVar, RefinementExpr, RefinementType};

/// SMT solver result
#[derive(Clone, Debug)]
pub enum SmtResult {
    /// Satisfiable with optional model
    Sat(Option<Model>),
    /// Unsatisfiable
    Unsat,
    /// Unknown (timeout, etc.)
    Unknown(String),
    /// Error during solving
    Error(String),
}

/// Model from SAT result
#[derive(Clone, Debug, Default)]
pub struct Model {
    pub assignments: HashMap<String, String>,
}

impl Model {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_assignment(mut self, var: impl Into<String>, val: impl Into<String>) -> Self {
        self.assignments.insert(var.into(), val.into());
        self
    }

    pub fn to_counterexample(&self) -> Counterexample {
        let mut ce = Counterexample::new();
        for (var, val) in &self.assignments {
            ce = ce.with_assignment(var.clone(), val.clone());
        }
        ce
    }
}

/// SMT-LIB2 logics
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmtLogic {
    /// Quantifier-free linear integer arithmetic
    QFLIA,
    /// Quantifier-free linear real arithmetic
    QFLRA,
    /// Quantifier-free nonlinear real arithmetic
    QFNRA,
    /// Linear integer and real arithmetic
    LIRA,
    /// All theories
    ALL,
}

impl SmtLogic {
    pub fn as_str(&self) -> &'static str {
        match self {
            SmtLogic::QFLIA => "QF_LIA",
            SmtLogic::QFLRA => "QF_LRA",
            SmtLogic::QFNRA => "QF_NRA",
            SmtLogic::LIRA => "LIRA",
            SmtLogic::ALL => "ALL",
        }
    }
}

/// SMT sorts
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmtSort {
    Int,
    Real,
    Bool,
}

impl SmtSort {
    pub fn as_str(&self) -> &'static str {
        match self {
            SmtSort::Int => "Int",
            SmtSort::Real => "Real",
            SmtSort::Bool => "Bool",
        }
    }
}

/// SMT context for building queries
#[derive(Clone, Debug)]
pub struct SmtContext {
    declarations: HashMap<String, SmtSort>,
    assertions: Vec<String>,
    logic: SmtLogic,
    timeout_ms: u64,
}

impl Default for SmtContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SmtContext {
    pub fn new() -> Self {
        Self {
            declarations: HashMap::new(),
            assertions: Vec::new(),
            logic: SmtLogic::QFLRA,
            timeout_ms: 5000,
        }
    }

    pub fn with_logic(mut self, logic: SmtLogic) -> Self {
        self.logic = logic;
        self
    }

    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    pub fn declare(&mut self, name: &str, sort: SmtSort) {
        self.declarations.insert(name.to_string(), sort);
    }

    pub fn assert(&mut self, formula: &str) {
        self.assertions.push(formula.to_string());
    }

    /// Translate a predicate to SMT-LIB2 format and auto-declare variables
    pub fn translate_predicate(&mut self, pred: &Predicate) -> String {
        // Collect and declare free variables
        let vars = pred.free_vars();
        for var in &vars {
            if !self.declarations.contains_key(var) {
                self.declare(var, SmtSort::Real);
            }
        }
        self.pred_to_smt(pred)
    }

    fn pred_to_smt(&self, pred: &Predicate) -> String {
        match pred {
            Predicate::Bool(b) => if *b { "true" } else { "false" }.to_string(),

            Predicate::Compare { left, op, right } => {
                let l = self.expr_to_smt(left);
                let r = self.expr_to_smt(right);
                let op_str = match op {
                    CompareOp::Eq => "=",
                    CompareOp::Ne => "distinct",
                    CompareOp::Lt => "<",
                    CompareOp::Le => "<=",
                    CompareOp::Gt => ">",
                    CompareOp::Ge => ">=",
                };
                format!("({} {} {})", op_str, l, r)
            }

            Predicate::And(p1, p2) => {
                format!("(and {} {})", self.pred_to_smt(p1), self.pred_to_smt(p2))
            }

            Predicate::Or(p1, p2) => {
                format!("(or {} {})", self.pred_to_smt(p1), self.pred_to_smt(p2))
            }

            Predicate::Not(p) => {
                format!("(not {})", self.pred_to_smt(p))
            }

            Predicate::Implies(p1, p2) => {
                format!("(=> {} {})", self.pred_to_smt(p1), self.pred_to_smt(p2))
            }

            Predicate::Ite {
                cond,
                then_pred,
                else_pred,
            } => {
                format!(
                    "(ite {} {} {})",
                    self.pred_to_smt(cond),
                    self.pred_to_smt(then_pred),
                    self.pred_to_smt(else_pred)
                )
            }

            Predicate::Forall { var, ty: _, body } => {
                let sort = "Real"; // Default to Real
                format!(
                    "(forall (({} {})) {})",
                    var.name,
                    sort,
                    self.pred_to_smt(body)
                )
            }

            Predicate::Exists { var, ty: _, body } => {
                let sort = "Real";
                format!(
                    "(exists (({} {})) {})",
                    var.name,
                    sort,
                    self.pred_to_smt(body)
                )
            }

            Predicate::Call { func, args } => {
                let args_smt: Vec<_> = args.iter().map(|a| self.expr_to_smt(a)).collect();
                format!("({} {})", func, args_smt.join(" "))
            }

            Predicate::Var(v) => v.name.clone(),
        }
    }

    fn expr_to_smt(&self, expr: &RefinementExpr) -> String {
        match expr {
            RefinementExpr::Int(i) => {
                if *i < 0 {
                    format!("(- {})", -i)
                } else {
                    i.to_string()
                }
            }

            RefinementExpr::Float(f) => {
                if *f < 0.0 {
                    format!("(- {})", -f)
                } else {
                    format!("{}", f)
                }
            }

            RefinementExpr::Var(v) => v.name.clone(),

            RefinementExpr::Arith { left, op, right } => {
                let l = self.expr_to_smt(left);
                let r = self.expr_to_smt(right);
                let op_str = match op {
                    ArithOp::Add => "+",
                    ArithOp::Sub => "-",
                    ArithOp::Mul => "*",
                    ArithOp::Div => "/",
                    ArithOp::Mod => "mod",
                    ArithOp::Pow => "^",
                };
                format!("({} {} {})", op_str, l, r)
            }

            RefinementExpr::Neg(e) => {
                format!("(- {})", self.expr_to_smt(e))
            }

            RefinementExpr::Field { base, field } => {
                format!("(field_{} {})", field, self.expr_to_smt(base))
            }

            RefinementExpr::Index { base, index } => {
                format!(
                    "(select {} {})",
                    self.expr_to_smt(base),
                    self.expr_to_smt(index)
                )
            }

            RefinementExpr::Ite {
                cond,
                then_expr,
                else_expr,
            } => {
                format!(
                    "(ite {} {} {})",
                    self.pred_to_smt(cond),
                    self.expr_to_smt(then_expr),
                    self.expr_to_smt(else_expr)
                )
            }

            RefinementExpr::App { func, args } => {
                let args_smt: Vec<_> = args.iter().map(|a| self.expr_to_smt(a)).collect();
                if args_smt.is_empty() {
                    func.clone()
                } else {
                    format!("({} {})", func, args_smt.join(" "))
                }
            }

            RefinementExpr::Len(arr) => {
                format!("(len {})", self.expr_to_smt(arr))
            }

            RefinementExpr::Old(e) => {
                format!("(old {})", self.expr_to_smt(e))
            }
        }
    }

    /// Generate the full SMT-LIB2 query
    pub fn generate_query(&self) -> String {
        let mut lines = Vec::new();

        lines.push(format!("(set-logic {})", self.logic.as_str()));
        lines.push(format!("(set-option :timeout {})", self.timeout_ms));

        for (name, sort) in &self.declarations {
            lines.push(format!("(declare-const {} {})", name, sort.as_str()));
        }

        for assertion in &self.assertions {
            lines.push(format!("(assert {})", assertion));
        }

        lines.push("(check-sat)".to_string());
        lines.push("(get-model)".to_string());

        lines.join("\n")
    }
}

/// SMT solver wrapper
pub struct SmtSolver {
    z3_path: Option<String>,
    timeout_ms: u64,
}

impl Default for SmtSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SmtSolver {
    pub fn new() -> Self {
        Self {
            z3_path: Self::find_z3(),
            timeout_ms: 5000,
        }
    }

    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    fn find_z3() -> Option<String> {
        if Command::new("z3").arg("--version").output().is_ok() {
            return Some("z3".to_string());
        }

        let paths = ["/usr/bin/z3", "/usr/local/bin/z3", "/opt/homebrew/bin/z3"];

        for path in paths {
            if std::path::Path::new(path).exists() {
                return Some(path.to_string());
            }
        }

        None
    }

    pub fn is_available(&self) -> bool {
        self.z3_path.is_some()
    }

    /// Check if predicate is valid (always true)
    /// Returns Unsat if valid (negation is unsat)
    pub fn check_valid(&self, pred: &Predicate) -> SmtResult {
        // Valid means ¬pred is unsat
        let negated = Predicate::not(pred.clone());
        self.check_sat(&negated)
    }

    /// Check if predicate is satisfiable
    pub fn check_sat(&self, pred: &Predicate) -> SmtResult {
        let z3_path = match &self.z3_path {
            Some(p) => p,
            None => return SmtResult::Unknown("Z3 not available".to_string()),
        };

        let mut ctx = SmtContext::new().with_timeout(self.timeout_ms);
        let smt_pred = ctx.translate_predicate(pred);
        ctx.assert(&smt_pred);

        let query = ctx.generate_query();

        match self.run_z3(z3_path, &query) {
            Ok(output) => self.parse_result(&output),
            Err(e) => SmtResult::Error(e),
        }
    }

    /// Verify subtype relation
    pub fn verify_subtype(&self, sub: &RefinementType, sup: &RefinementType) -> SmtResult {
        // { x: T | P } <: { y: T | Q } iff ∀x. P(x) ⟹ Q[y := x]
        let sup_pred = sup
            .predicate
            .substitute(&sup.binder.name, &RefinementExpr::Var(sub.binder.clone()));

        let implication = Predicate::implies(sub.predicate.clone(), sup_pred);

        self.check_valid(&implication)
    }

    fn run_z3(&self, z3_path: &str, query: &str) -> Result<String, String> {
        let mut child = Command::new(z3_path)
            .arg("-in")
            .arg("-smt2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn Z3: {}", e))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(query.as_bytes())
                .map_err(|e| format!("Failed to write to Z3: {}", e))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| format!("Failed to wait for Z3: {}", e))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn parse_result(&self, output: &str) -> SmtResult {
        let lines: Vec<&str> = output.lines().collect();

        if lines.is_empty() {
            return SmtResult::Error("Empty output from Z3".to_string());
        }

        match lines[0].trim() {
            "sat" => {
                let model = self.parse_model(&lines[1..]);
                SmtResult::Sat(Some(model))
            }
            "unsat" => SmtResult::Unsat,
            "unknown" => SmtResult::Unknown("Solver returned unknown".to_string()),
            other => SmtResult::Error(format!("Unexpected Z3 output: {}", other)),
        }
    }

    fn parse_model(&self, lines: &[&str]) -> Model {
        let mut model = Model::new();

        // Simple parsing of (model (define-fun name () Sort value))
        let full = lines.join("\n");

        // Very basic parsing - just extract variable assignments
        for line in lines {
            let line = line.trim();
            if line.starts_with("(define-fun") {
                // Extract name and value
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    let name = parts[1];
                    // Value is typically the last token before the closing paren
                    if let Some(val) = parts.last() {
                        let val = val.trim_end_matches(')');
                        model = model.with_assignment(name.to_string(), val.to_string());
                    }
                }
            }
        }

        model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smt_context_creation() {
        let ctx = SmtContext::new();
        assert!(ctx.declarations.is_empty());
    }

    #[test]
    fn test_smt_translate_simple() {
        let mut ctx = SmtContext::new();
        let pred = Predicate::gt(
            RefinementExpr::Var(RefinedVar::new("x")),
            RefinementExpr::Int(0),
        );
        let smt = ctx.translate_predicate(&pred);
        assert!(smt.contains(">"));
        assert!(smt.contains("x"));
    }

    #[test]
    fn test_smt_logic_variants() {
        assert_eq!(SmtLogic::QFLIA.as_str(), "QF_LIA");
        assert_eq!(SmtLogic::QFLRA.as_str(), "QF_LRA");
    }

    #[test]
    fn test_model_creation() {
        let model = Model::new()
            .with_assignment("x", "5")
            .with_assignment("y", "10");
        assert_eq!(model.assignments.len(), 2);
    }
}
