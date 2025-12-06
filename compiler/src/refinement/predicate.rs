//! Predicate representation for refinement types
//!
//! Predicates are boolean expressions that constrain values in refinement types.
//! They support arithmetic, comparison, logical operations, and quantifiers.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A predicate in a refinement type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Predicate {
    pub kind: PredicateKind,
}

/// The kind of predicate
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PredicateKind {
    // Literals
    /// Boolean literal (true/false)
    BoolLit(bool),
    
    /// Floating point literal
    FloatLit(f64),
    
    /// Integer literal
    IntLit(i64),
    
    // Variables
    /// Variable reference
    Var(String),
    
    // Arithmetic operations
    /// Arithmetic operation
    Arith(ArithOp, Box<Predicate>, Box<Predicate>),
    
    /// Unary negation (arithmetic)
    Neg(Box<Predicate>),
    
    // Comparison operations
    /// Comparison operation
    Compare(CompareOp, Box<Predicate>, Box<Predicate>),
    
    // Logical operations
    /// Logical AND
    And(Box<Predicate>, Box<Predicate>),
    
    /// Logical OR  
    Or(Box<Predicate>, Box<Predicate>),
    
    /// Logical NOT
    Not(Box<Predicate>),
    
    /// Implication (P => Q)
    Implies(Box<Predicate>, Box<Predicate>),
    
    /// If-then-else (ite condition then else)
    Ite(Box<Predicate>, Box<Predicate>, Box<Predicate>),
    
    // Function applications
    /// Built-in function call (e.g., abs, sqrt, exp, log, pow)
    BuiltinFn(BuiltinFn, Vec<Predicate>),
    
    // Quantifiers (for advanced refinements)
    /// Universal quantification: ∀x. P(x)
    Forall(String, Box<Predicate>),
    
    /// Existential quantification: ∃x. P(x)
    Exists(String, Box<Predicate>),
}

/// Arithmetic operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
}

/// Comparison operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompareOp {
    Eq,  // ==
    Ne,  // !=
    Lt,  // <
    Le,  // <=
    Gt,  // >
    Ge,  // >=
}

/// Logical operations (for external use)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogicalOp {
    And,
    Or,
    Not,
    Implies,
}

/// Built-in functions supported in predicates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinFn {
    // Math functions
    Abs,
    Sqrt,
    Exp,
    Log,
    Log10,
    Sin,
    Cos,
    Tan,
    Floor,
    Ceil,
    Round,
    
    // Min/Max
    Min,
    Max,
    
    // Type conversions
    ToInt,
    ToReal,
}

impl Predicate {
    // =========================================================================
    // Constructors
    // =========================================================================
    
    /// Create a boolean literal
    pub fn bool_literal(b: bool) -> Self {
        Self { kind: PredicateKind::BoolLit(b) }
    }
    
    /// Create a float literal
    pub fn float(f: f64) -> Self {
        Self { kind: PredicateKind::FloatLit(f) }
    }
    
    /// Create an integer literal
    pub fn int(i: i64) -> Self {
        Self { kind: PredicateKind::IntLit(i) }
    }
    
    /// Create a variable reference
    pub fn var(name: impl Into<String>) -> Self {
        Self { kind: PredicateKind::Var(name.into()) }
    }
    
    // =========================================================================
    // Arithmetic operations
    // =========================================================================
    
    /// Addition
    pub fn add(lhs: Predicate, rhs: Predicate) -> Self {
        Self { kind: PredicateKind::Arith(ArithOp::Add, Box::new(lhs), Box::new(rhs)) }
    }
    
    /// Subtraction
    pub fn sub(lhs: Predicate, rhs: Predicate) -> Self {
        Self { kind: PredicateKind::Arith(ArithOp::Sub, Box::new(lhs), Box::new(rhs)) }
    }
    
    /// Multiplication
    pub fn mul(lhs: Predicate, rhs: Predicate) -> Self {
        Self { kind: PredicateKind::Arith(ArithOp::Mul, Box::new(lhs), Box::new(rhs)) }
    }
    
    /// Division
    pub fn div(lhs: Predicate, rhs: Predicate) -> Self {
        Self { kind: PredicateKind::Arith(ArithOp::Div, Box::new(lhs), Box::new(rhs)) }
    }
    
    /// Modulo
    pub fn modulo(lhs: Predicate, rhs: Predicate) -> Self {
        Self { kind: PredicateKind::Arith(ArithOp::Mod, Box::new(lhs), Box::new(rhs)) }
    }
    
    /// Power
    pub fn pow(base: Predicate, exp: Predicate) -> Self {
        Self { kind: PredicateKind::Arith(ArithOp::Pow, Box::new(base), Box::new(exp)) }
    }
    
    /// Unary negation
    pub fn neg(p: Predicate) -> Self {
        Self { kind: PredicateKind::Neg(Box::new(p)) }
    }
    
    // =========================================================================
    // Comparison operations
    // =========================================================================
    
    /// Equal
    pub fn eq(lhs: Predicate, rhs: Predicate) -> Self {
        Self { kind: PredicateKind::Compare(CompareOp::Eq, Box::new(lhs), Box::new(rhs)) }
    }
    
    /// Not equal
    pub fn ne(lhs: Predicate, rhs: Predicate) -> Self {
        Self { kind: PredicateKind::Compare(CompareOp::Ne, Box::new(lhs), Box::new(rhs)) }
    }
    
    /// Less than
    pub fn lt(lhs: Predicate, rhs: Predicate) -> Self {
        Self { kind: PredicateKind::Compare(CompareOp::Lt, Box::new(lhs), Box::new(rhs)) }
    }
    
    /// Less than or equal
    pub fn le(lhs: Predicate, rhs: Predicate) -> Self {
        Self { kind: PredicateKind::Compare(CompareOp::Le, Box::new(lhs), Box::new(rhs)) }
    }
    
    /// Greater than
    pub fn gt(lhs: Predicate, rhs: Predicate) -> Self {
        Self { kind: PredicateKind::Compare(CompareOp::Gt, Box::new(lhs), Box::new(rhs)) }
    }
    
    /// Greater than or equal
    pub fn ge(lhs: Predicate, rhs: Predicate) -> Self {
        Self { kind: PredicateKind::Compare(CompareOp::Ge, Box::new(lhs), Box::new(rhs)) }
    }
    
    // =========================================================================
    // Logical operations
    // =========================================================================
    
    /// Logical AND
    pub fn and(lhs: Predicate, rhs: Predicate) -> Self {
        Self { kind: PredicateKind::And(Box::new(lhs), Box::new(rhs)) }
    }
    
    /// Logical OR
    pub fn or(lhs: Predicate, rhs: Predicate) -> Self {
        Self { kind: PredicateKind::Or(Box::new(lhs), Box::new(rhs)) }
    }
    
    /// Logical NOT
    pub fn not(p: Predicate) -> Self {
        Self { kind: PredicateKind::Not(Box::new(p)) }
    }
    
    /// Implication (P => Q)
    pub fn implies(antecedent: Predicate, consequent: Predicate) -> Self {
        Self { kind: PredicateKind::Implies(Box::new(antecedent), Box::new(consequent)) }
    }
    
    /// If-then-else
    pub fn ite(cond: Predicate, then_: Predicate, else_: Predicate) -> Self {
        Self { kind: PredicateKind::Ite(Box::new(cond), Box::new(then_), Box::new(else_)) }
    }
    
    // =========================================================================
    // Built-in functions
    // =========================================================================
    
    /// Absolute value
    pub fn abs(p: Predicate) -> Self {
        Self { kind: PredicateKind::BuiltinFn(BuiltinFn::Abs, vec![p]) }
    }
    
    /// Square root
    pub fn sqrt(p: Predicate) -> Self {
        Self { kind: PredicateKind::BuiltinFn(BuiltinFn::Sqrt, vec![p]) }
    }
    
    /// Exponential
    pub fn exp(p: Predicate) -> Self {
        Self { kind: PredicateKind::BuiltinFn(BuiltinFn::Exp, vec![p]) }
    }
    
    /// Natural logarithm
    pub fn log(p: Predicate) -> Self {
        Self { kind: PredicateKind::BuiltinFn(BuiltinFn::Log, vec![p]) }
    }
    
    /// Minimum
    pub fn min(a: Predicate, b: Predicate) -> Self {
        Self { kind: PredicateKind::BuiltinFn(BuiltinFn::Min, vec![a, b]) }
    }
    
    /// Maximum
    pub fn max(a: Predicate, b: Predicate) -> Self {
        Self { kind: PredicateKind::BuiltinFn(BuiltinFn::Max, vec![a, b]) }
    }
    
    // =========================================================================
    // Quantifiers
    // =========================================================================
    
    /// Universal quantification
    pub fn forall(var: impl Into<String>, body: Predicate) -> Self {
        Self { kind: PredicateKind::Forall(var.into(), Box::new(body)) }
    }
    
    /// Existential quantification
    pub fn exists(var: impl Into<String>, body: Predicate) -> Self {
        Self { kind: PredicateKind::Exists(var.into(), Box::new(body)) }
    }
    
    // =========================================================================
    // Utility methods
    // =========================================================================
    
    /// Get all free variables in this predicate
    pub fn free_vars(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_free_vars(&mut vars, &HashSet::new());
        vars
    }
    
    fn collect_free_vars(&self, free: &mut HashSet<String>, bound: &HashSet<String>) {
        match &self.kind {
            PredicateKind::BoolLit(_) | 
            PredicateKind::FloatLit(_) | 
            PredicateKind::IntLit(_) => {}
            
            PredicateKind::Var(name) => {
                if !bound.contains(name) {
                    free.insert(name.clone());
                }
            }
            
            PredicateKind::Arith(_, lhs, rhs) |
            PredicateKind::Compare(_, lhs, rhs) |
            PredicateKind::And(lhs, rhs) |
            PredicateKind::Or(lhs, rhs) |
            PredicateKind::Implies(lhs, rhs) => {
                lhs.collect_free_vars(free, bound);
                rhs.collect_free_vars(free, bound);
            }
            
            PredicateKind::Neg(p) |
            PredicateKind::Not(p) => {
                p.collect_free_vars(free, bound);
            }
            
            PredicateKind::Ite(c, t, e) => {
                c.collect_free_vars(free, bound);
                t.collect_free_vars(free, bound);
                e.collect_free_vars(free, bound);
            }
            
            PredicateKind::BuiltinFn(_, args) => {
                for arg in args {
                    arg.collect_free_vars(free, bound);
                }
            }
            
            PredicateKind::Forall(var, body) |
            PredicateKind::Exists(var, body) => {
                let mut new_bound = bound.clone();
                new_bound.insert(var.clone());
                body.collect_free_vars(free, &new_bound);
            }
        }
    }
    
    /// Substitute a variable with another predicate
    pub fn substitute(&self, var: &str, replacement: &Predicate) -> Self {
        match &self.kind {
            PredicateKind::BoolLit(_) | 
            PredicateKind::FloatLit(_) | 
            PredicateKind::IntLit(_) => self.clone(),
            
            PredicateKind::Var(name) => {
                if name == var {
                    replacement.clone()
                } else {
                    self.clone()
                }
            }
            
            PredicateKind::Arith(op, lhs, rhs) => {
                Predicate {
                    kind: PredicateKind::Arith(
                        *op,
                        Box::new(lhs.substitute(var, replacement)),
                        Box::new(rhs.substitute(var, replacement)),
                    )
                }
            }
            
            PredicateKind::Compare(op, lhs, rhs) => {
                Predicate {
                    kind: PredicateKind::Compare(
                        *op,
                        Box::new(lhs.substitute(var, replacement)),
                        Box::new(rhs.substitute(var, replacement)),
                    )
                }
            }
            
            PredicateKind::And(lhs, rhs) => Predicate::and(
                lhs.substitute(var, replacement),
                rhs.substitute(var, replacement),
            ),
            
            PredicateKind::Or(lhs, rhs) => Predicate::or(
                lhs.substitute(var, replacement),
                rhs.substitute(var, replacement),
            ),
            
            PredicateKind::Not(p) => Predicate::not(p.substitute(var, replacement)),
            
            PredicateKind::Neg(p) => Predicate::neg(p.substitute(var, replacement)),
            
            PredicateKind::Implies(lhs, rhs) => Predicate::implies(
                lhs.substitute(var, replacement),
                rhs.substitute(var, replacement),
            ),
            
            PredicateKind::Ite(c, t, e) => Predicate::ite(
                c.substitute(var, replacement),
                t.substitute(var, replacement),
                e.substitute(var, replacement),
            ),
            
            PredicateKind::BuiltinFn(f, args) => {
                Predicate {
                    kind: PredicateKind::BuiltinFn(
                        *f,
                        args.iter().map(|a| a.substitute(var, replacement)).collect(),
                    )
                }
            }
            
            PredicateKind::Forall(bound_var, body) |
            PredicateKind::Exists(bound_var, body) => {
                if bound_var == var {
                    // Variable is shadowed, don't substitute in body
                    self.clone()
                } else {
                    let new_body = body.substitute(var, replacement);
                    match &self.kind {
                        PredicateKind::Forall(_, _) => Predicate::forall(bound_var, new_body),
                        PredicateKind::Exists(_, _) => Predicate::exists(bound_var, new_body),
                        _ => unreachable!(),
                    }
                }
            }
        }
    }
    
    /// Check if this predicate is a simple boolean literal
    pub fn is_true(&self) -> bool {
        matches!(self.kind, PredicateKind::BoolLit(true))
    }
    
    pub fn is_false(&self) -> bool {
        matches!(self.kind, PredicateKind::BoolLit(false))
    }
    
    /// Simplify the predicate (basic algebraic simplifications)
    pub fn simplify(&self) -> Self {
        match &self.kind {
            // true && P => P
            PredicateKind::And(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                if lhs.is_true() { return rhs; }
                if rhs.is_true() { return lhs; }
                if lhs.is_false() || rhs.is_false() { return Predicate::bool_literal(false); }
                Predicate::and(lhs, rhs)
            }
            
            // false || P => P
            PredicateKind::Or(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                if lhs.is_false() { return rhs; }
                if rhs.is_false() { return lhs; }
                if lhs.is_true() || rhs.is_true() { return Predicate::bool_literal(true); }
                Predicate::or(lhs, rhs)
            }
            
            // !true => false, !false => true
            PredicateKind::Not(p) => {
                let p = p.simplify();
                if p.is_true() { return Predicate::bool_literal(false); }
                if p.is_false() { return Predicate::bool_literal(true); }
                Predicate::not(p)
            }
            
            // true => P simplifies to P
            PredicateKind::Implies(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                if lhs.is_true() { return rhs; }
                if lhs.is_false() { return Predicate::bool_literal(true); }
                if rhs.is_true() { return Predicate::bool_literal(true); }
                Predicate::implies(lhs, rhs)
            }
            
            _ => self.clone(),
        }
    }
}

impl std::fmt::Display for Predicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            PredicateKind::BoolLit(b) => write!(f, "{}", b),
            PredicateKind::FloatLit(n) => write!(f, "{}", n),
            PredicateKind::IntLit(n) => write!(f, "{}", n),
            PredicateKind::Var(name) => write!(f, "{}", name),
            
            PredicateKind::Arith(op, lhs, rhs) => {
                let op_str = match op {
                    ArithOp::Add => "+",
                    ArithOp::Sub => "-",
                    ArithOp::Mul => "*",
                    ArithOp::Div => "/",
                    ArithOp::Mod => "%",
                    ArithOp::Pow => "^",
                };
                write!(f, "({} {} {})", lhs, op_str, rhs)
            }
            
            PredicateKind::Neg(p) => write!(f, "(-{})", p),
            
            PredicateKind::Compare(op, lhs, rhs) => {
                let op_str = match op {
                    CompareOp::Eq => "==",
                    CompareOp::Ne => "!=",
                    CompareOp::Lt => "<",
                    CompareOp::Le => "<=",
                    CompareOp::Gt => ">",
                    CompareOp::Ge => ">=",
                };
                write!(f, "({} {} {})", lhs, op_str, rhs)
            }
            
            PredicateKind::And(lhs, rhs) => write!(f, "({} && {})", lhs, rhs),
            PredicateKind::Or(lhs, rhs) => write!(f, "({} || {})", lhs, rhs),
            PredicateKind::Not(p) => write!(f, "!{}", p),
            PredicateKind::Implies(lhs, rhs) => write!(f, "({} => {})", lhs, rhs),
            PredicateKind::Ite(c, t, e) => write!(f, "(if {} then {} else {})", c, t, e),
            
            PredicateKind::BuiltinFn(func, args) => {
                let name = match func {
                    BuiltinFn::Abs => "abs",
                    BuiltinFn::Sqrt => "sqrt",
                    BuiltinFn::Exp => "exp",
                    BuiltinFn::Log => "log",
                    BuiltinFn::Log10 => "log10",
                    BuiltinFn::Sin => "sin",
                    BuiltinFn::Cos => "cos",
                    BuiltinFn::Tan => "tan",
                    BuiltinFn::Floor => "floor",
                    BuiltinFn::Ceil => "ceil",
                    BuiltinFn::Round => "round",
                    BuiltinFn::Min => "min",
                    BuiltinFn::Max => "max",
                    BuiltinFn::ToInt => "to_int",
                    BuiltinFn::ToReal => "to_real",
                };
                let args_str: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "{}({})", name, args_str.join(", "))
            }
            
            PredicateKind::Forall(var, body) => write!(f, "∀{}. {}", var, body),
            PredicateKind::Exists(var, body) => write!(f, "∃{}. {}", var, body),
        }
    }
}

impl std::fmt::Display for ArithOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArithOp::Add => write!(f, "+"),
            ArithOp::Sub => write!(f, "-"),
            ArithOp::Mul => write!(f, "*"),
            ArithOp::Div => write!(f, "/"),
            ArithOp::Mod => write!(f, "%"),
            ArithOp::Pow => write!(f, "^"),
        }
    }
}

impl std::fmt::Display for CompareOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompareOp::Eq => write!(f, "=="),
            CompareOp::Ne => write!(f, "!="),
            CompareOp::Lt => write!(f, "<"),
            CompareOp::Le => write!(f, "<="),
            CompareOp::Gt => write!(f, ">"),
            CompareOp::Ge => write!(f, ">="),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_predicate_display() {
        let p = Predicate::and(
            Predicate::ge(Predicate::var("x"), Predicate::float(0.0)),
            Predicate::le(Predicate::var("x"), Predicate::float(10.0)),
        );
        let s = p.to_string();
        assert!(s.contains("x"));
        assert!(s.contains(">="));
        assert!(s.contains("<="));
    }
    
    #[test]
    fn test_free_vars() {
        let p = Predicate::and(
            Predicate::gt(Predicate::var("x"), Predicate::float(0.0)),
            Predicate::lt(Predicate::var("y"), Predicate::float(10.0)),
        );
        let vars = p.free_vars();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert_eq!(vars.len(), 2);
    }
    
    #[test]
    fn test_free_vars_with_quantifier() {
        // ∀x. (x > 0 && y < 10)
        // x is bound, y is free
        let p = Predicate::forall(
            "x",
            Predicate::and(
                Predicate::gt(Predicate::var("x"), Predicate::float(0.0)),
                Predicate::lt(Predicate::var("y"), Predicate::float(10.0)),
            ),
        );
        let vars = p.free_vars();
        assert!(!vars.contains("x")); // bound
        assert!(vars.contains("y"));  // free
        assert_eq!(vars.len(), 1);
    }
    
    #[test]
    fn test_substitute() {
        let p = Predicate::gt(Predicate::var("x"), Predicate::float(0.0));
        let subst = p.substitute("x", &Predicate::float(5.0));
        
        match subst.kind {
            PredicateKind::Compare(CompareOp::Gt, ref lhs, _) => {
                assert!(matches!(lhs.kind, PredicateKind::FloatLit(5.0)));
            }
            _ => panic!("Wrong predicate kind after substitution"),
        }
    }
    
    #[test]
    fn test_simplify() {
        // true && x > 0 => x > 0
        let p = Predicate::and(
            Predicate::bool_literal(true),
            Predicate::gt(Predicate::var("x"), Predicate::float(0.0)),
        );
        let simplified = p.simplify();
        assert!(matches!(simplified.kind, PredicateKind::Compare(CompareOp::Gt, _, _)));
        
        // false || x > 0 => x > 0
        let p2 = Predicate::or(
            Predicate::bool_literal(false),
            Predicate::gt(Predicate::var("x"), Predicate::float(0.0)),
        );
        let simplified2 = p2.simplify();
        assert!(matches!(simplified2.kind, PredicateKind::Compare(CompareOp::Gt, _, _)));
    }
}
