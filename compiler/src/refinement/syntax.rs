//! Refinement Type Syntax
//!
//! Defines the AST for refinement types and predicates.
//!
//! Syntax:
//! ```text
//! { binder: BaseType | predicate }
//!
//! predicate ::= expr cmp expr
//!            | predicate && predicate
//!            | predicate || predicate
//!            | !predicate
//!            | forall(x: T, predicate)
//!            | exists(x: T, predicate)
//!            | function(args)           -- decidable predicates
//!
//! cmp ::= == | != | < | <= | > | >=
//!
//! expr ::= literal
//!       | variable
//!       | expr arith expr
//!       | -expr
//!       | field_access
//!       | array_index
//!       | if predicate then expr else expr
//!
//! arith ::= + | - | * | / | %
//! ```

use std::collections::HashSet;
use std::fmt;

use crate::ast::Span;

/// A refinement type: { binder: base | predicate }
#[derive(Clone, Debug, PartialEq)]
pub struct RefinementType {
    /// The bound variable (e.g., "dose" in { dose: mg | ... })
    pub binder: RefinedVar,
    /// The base type being refined
    pub base_type: BaseTypeRef,
    /// The refinement predicate
    pub predicate: Predicate,
    /// Source location
    pub span: Option<Span>,
}

/// Reference to a base type (before full type resolution)
#[derive(Clone, Debug, PartialEq)]
pub enum BaseTypeRef {
    /// Named type (Int, Float, mg, mL/min, etc.)
    Named(String),
    /// Type variable
    Var(String),
    /// Unit-qualified numeric (e.g., Float<mg>)
    WithUnit { base: String, unit: String },
}

/// A variable in a refinement
#[derive(Clone, Debug)]
pub struct RefinedVar {
    pub name: String,
    pub span: Option<Span>,
}

// Manual impls to ignore span in equality/hashing
impl PartialEq for RefinedVar {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for RefinedVar {}

impl std::hash::Hash for RefinedVar {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl RefinedVar {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            span: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }
}

/// A refinement predicate
#[derive(Clone, Debug, PartialEq)]
pub enum Predicate {
    /// Boolean literal
    Bool(bool),

    /// Comparison: expr op expr
    Compare {
        left: Box<RefinementExpr>,
        op: CompareOp,
        right: Box<RefinementExpr>,
    },

    /// Logical AND
    And(Box<Predicate>, Box<Predicate>),

    /// Logical OR
    Or(Box<Predicate>, Box<Predicate>),

    /// Logical NOT
    Not(Box<Predicate>),

    /// Implication: p => q
    Implies(Box<Predicate>, Box<Predicate>),

    /// If-then-else predicate
    Ite {
        cond: Box<Predicate>,
        then_pred: Box<Predicate>,
        else_pred: Box<Predicate>,
    },

    /// Universal quantifier: forall(x: T, P(x))
    Forall {
        var: RefinedVar,
        ty: BaseTypeRef,
        body: Box<Predicate>,
    },

    /// Existential quantifier: exists(x: T, P(x))
    Exists {
        var: RefinedVar,
        ty: BaseTypeRef,
        body: Box<Predicate>,
    },

    /// Decidable predicate function call (e.g., is_valid_dose(d))
    Call {
        func: String,
        args: Vec<RefinementExpr>,
    },

    /// Variable reference (boolean variable)
    Var(RefinedVar),
}

/// Comparison operators
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompareOp {
    Eq, // ==
    Ne, // !=
    Lt, // <
    Le, // <=
    Gt, // >
    Ge, // >=
}

/// Expressions in refinements (subset of full expressions)
#[derive(Clone, Debug, PartialEq)]
pub enum RefinementExpr {
    /// Integer literal
    Int(i64),

    /// Float literal
    Float(f64),

    /// Variable reference
    Var(RefinedVar),

    /// Arithmetic operation
    Arith {
        left: Box<RefinementExpr>,
        op: ArithOp,
        right: Box<RefinementExpr>,
    },

    /// Unary negation
    Neg(Box<RefinementExpr>),

    /// Field access: expr.field
    Field {
        base: Box<RefinementExpr>,
        field: String,
    },

    /// Array/vector indexing: expr[index]
    Index {
        base: Box<RefinementExpr>,
        index: Box<RefinementExpr>,
    },

    /// Conditional expression: if p then e1 else e2
    Ite {
        cond: Box<Predicate>,
        then_expr: Box<RefinementExpr>,
        else_expr: Box<RefinementExpr>,
    },

    /// Function application (for interpreted functions)
    App {
        func: String,
        args: Vec<RefinementExpr>,
    },

    /// Length of array/vector
    Len(Box<RefinementExpr>),

    /// Old value (for postconditions): old(expr)
    Old(Box<RefinementExpr>),
}

/// Arithmetic operators
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArithOp {
    Add, // +
    Sub, // -
    Mul, // *
    Div, // /
    Mod, // %
    Pow, // ^
}

/// Built-in functions supported in refinement predicates
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

/// Predicate operator (for constraint generation)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredicateOp {
    And,
    Or,
    Not,
    Implies,
}

// ============================================================================
// Constructors and helpers
// ============================================================================

impl RefinementType {
    /// Create a new refinement type
    pub fn new(binder: RefinedVar, base_type: BaseTypeRef, predicate: Predicate) -> Self {
        Self {
            binder,
            base_type,
            predicate,
            span: None,
        }
    }

    /// Create a trivially refined type (predicate = true)
    pub fn trivial(binder: RefinedVar, base_type: BaseTypeRef) -> Self {
        Self::new(binder, base_type, Predicate::Bool(true))
    }

    /// Get free variables in the predicate
    pub fn free_vars(&self) -> HashSet<String> {
        let mut vars = self.predicate.free_vars();
        vars.remove(&self.binder.name);
        vars
    }

    /// Substitute a variable in the predicate
    pub fn substitute(&self, from: &str, to: &RefinementExpr) -> Self {
        Self {
            binder: self.binder.clone(),
            base_type: self.base_type.clone(),
            predicate: self.predicate.substitute(from, to),
            span: self.span.clone(),
        }
    }
}

impl Predicate {
    // =========================================================================
    // Convenience constructors (for compatibility with constraint module)
    // These wrap expressions to create predicates
    // =========================================================================

    /// Create a variable reference (as expression, wrapped in a trivial comparison)
    /// For use in constraint generation where we need Predicate but have a var name
    pub fn var(name: &str) -> RefinementExpr {
        RefinementExpr::Var(RefinedVar::new(name))
    }

    /// Create a float literal expression
    pub fn float(val: f64) -> RefinementExpr {
        RefinementExpr::Float(val)
    }

    /// Create an integer literal expression
    pub fn int(val: i64) -> RefinementExpr {
        RefinementExpr::Int(val)
    }

    /// Create a boolean literal predicate
    pub fn bool_literal(val: bool) -> Predicate {
        Predicate::Bool(val)
    }

    /// Not-equal comparison
    pub fn ne(e1: RefinementExpr, e2: RefinementExpr) -> Self {
        Predicate::Compare {
            left: Box::new(e1),
            op: CompareOp::Ne,
            right: Box::new(e2),
        }
    }

    // =========================================================================
    // Constructors for common patterns
    // =========================================================================

    pub fn and(p1: Predicate, p2: Predicate) -> Self {
        Predicate::And(Box::new(p1), Box::new(p2))
    }

    pub fn or(p1: Predicate, p2: Predicate) -> Self {
        Predicate::Or(Box::new(p1), Box::new(p2))
    }

    pub fn not(p: Predicate) -> Self {
        Predicate::Not(Box::new(p))
    }

    pub fn implies(p1: Predicate, p2: Predicate) -> Self {
        Predicate::Implies(Box::new(p1), Box::new(p2))
    }

    pub fn eq(e1: RefinementExpr, e2: RefinementExpr) -> Self {
        Predicate::Compare {
            left: Box::new(e1),
            op: CompareOp::Eq,
            right: Box::new(e2),
        }
    }

    pub fn lt(e1: RefinementExpr, e2: RefinementExpr) -> Self {
        Predicate::Compare {
            left: Box::new(e1),
            op: CompareOp::Lt,
            right: Box::new(e2),
        }
    }

    pub fn le(e1: RefinementExpr, e2: RefinementExpr) -> Self {
        Predicate::Compare {
            left: Box::new(e1),
            op: CompareOp::Le,
            right: Box::new(e2),
        }
    }

    pub fn gt(e1: RefinementExpr, e2: RefinementExpr) -> Self {
        Predicate::Compare {
            left: Box::new(e1),
            op: CompareOp::Gt,
            right: Box::new(e2),
        }
    }

    pub fn ge(e1: RefinementExpr, e2: RefinementExpr) -> Self {
        Predicate::Compare {
            left: Box::new(e1),
            op: CompareOp::Ge,
            right: Box::new(e2),
        }
    }

    /// Range predicate: lo <= x && x <= hi
    pub fn in_range(var: &str, lo: RefinementExpr, hi: RefinementExpr) -> Self {
        let v = RefinementExpr::Var(RefinedVar::new(var));
        Predicate::and(Predicate::le(lo, v.clone()), Predicate::le(v, hi))
    }

    /// Positive: x > 0
    pub fn positive(var: &str) -> Self {
        Predicate::gt(
            RefinementExpr::Var(RefinedVar::new(var)),
            RefinementExpr::Int(0),
        )
    }

    /// Non-negative: x >= 0
    pub fn non_negative(var: &str) -> Self {
        Predicate::ge(
            RefinementExpr::Var(RefinedVar::new(var)),
            RefinementExpr::Int(0),
        )
    }

    /// Check if this is the trivial true predicate
    pub fn is_true(&self) -> bool {
        matches!(self, Predicate::Bool(true))
    }

    /// Check if this is the trivial false predicate
    pub fn is_false(&self) -> bool {
        matches!(self, Predicate::Bool(false))
    }

    /// Get free variables
    pub fn free_vars(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_free_vars(&mut vars);
        vars
    }

    fn collect_free_vars(&self, vars: &mut HashSet<String>) {
        match self {
            Predicate::Bool(_) => {}
            Predicate::Compare { left, right, .. } => {
                left.collect_free_vars(vars);
                right.collect_free_vars(vars);
            }
            Predicate::And(p1, p2) | Predicate::Or(p1, p2) | Predicate::Implies(p1, p2) => {
                p1.collect_free_vars(vars);
                p2.collect_free_vars(vars);
            }
            Predicate::Not(p) => p.collect_free_vars(vars),
            Predicate::Ite {
                cond,
                then_pred,
                else_pred,
            } => {
                cond.collect_free_vars(vars);
                then_pred.collect_free_vars(vars);
                else_pred.collect_free_vars(vars);
            }
            Predicate::Forall { var, body, .. } | Predicate::Exists { var, body, .. } => {
                let mut inner = HashSet::new();
                body.collect_free_vars(&mut inner);
                inner.remove(&var.name);
                vars.extend(inner);
            }
            Predicate::Call { args, .. } => {
                for arg in args {
                    arg.collect_free_vars(vars);
                }
            }
            Predicate::Var(v) => {
                vars.insert(v.name.clone());
            }
        }
    }

    /// Substitute a variable with an expression
    pub fn substitute(&self, from: &str, to: &RefinementExpr) -> Self {
        match self {
            Predicate::Bool(b) => Predicate::Bool(*b),
            Predicate::Compare { left, op, right } => Predicate::Compare {
                left: Box::new(left.substitute(from, to)),
                op: *op,
                right: Box::new(right.substitute(from, to)),
            },
            Predicate::And(p1, p2) => {
                Predicate::and(p1.substitute(from, to), p2.substitute(from, to))
            }
            Predicate::Or(p1, p2) => {
                Predicate::or(p1.substitute(from, to), p2.substitute(from, to))
            }
            Predicate::Not(p) => Predicate::not(p.substitute(from, to)),
            Predicate::Implies(p1, p2) => {
                Predicate::implies(p1.substitute(from, to), p2.substitute(from, to))
            }
            Predicate::Ite {
                cond,
                then_pred,
                else_pred,
            } => Predicate::Ite {
                cond: Box::new(cond.substitute(from, to)),
                then_pred: Box::new(then_pred.substitute(from, to)),
                else_pred: Box::new(else_pred.substitute(from, to)),
            },
            Predicate::Forall { var, ty, body } => {
                if var.name == from {
                    // Bound variable shadows, don't substitute in body
                    self.clone()
                } else {
                    Predicate::Forall {
                        var: var.clone(),
                        ty: ty.clone(),
                        body: Box::new(body.substitute(from, to)),
                    }
                }
            }
            Predicate::Exists { var, ty, body } => {
                if var.name == from {
                    self.clone()
                } else {
                    Predicate::Exists {
                        var: var.clone(),
                        ty: ty.clone(),
                        body: Box::new(body.substitute(from, to)),
                    }
                }
            }
            Predicate::Call { func, args } => Predicate::Call {
                func: func.clone(),
                args: args.iter().map(|a| a.substitute(from, to)).collect(),
            },
            Predicate::Var(v) => {
                if v.name == from {
                    // This is a boolean variable being substituted
                    // to should be a boolean expression - wrap in comparison
                    Predicate::Var(v.clone()) // For now, keep as is
                } else {
                    Predicate::Var(v.clone())
                }
            }
        }
    }

    /// Simplify predicate (basic simplifications)
    pub fn simplify(&self) -> Self {
        match self {
            Predicate::And(p1, p2) => {
                let s1 = p1.simplify();
                let s2 = p2.simplify();
                match (&s1, &s2) {
                    (Predicate::Bool(true), _) => s2,
                    (_, Predicate::Bool(true)) => s1,
                    (Predicate::Bool(false), _) | (_, Predicate::Bool(false)) => {
                        Predicate::Bool(false)
                    }
                    _ => Predicate::and(s1, s2),
                }
            }
            Predicate::Or(p1, p2) => {
                let s1 = p1.simplify();
                let s2 = p2.simplify();
                match (&s1, &s2) {
                    (Predicate::Bool(true), _) | (_, Predicate::Bool(true)) => {
                        Predicate::Bool(true)
                    }
                    (Predicate::Bool(false), _) => s2,
                    (_, Predicate::Bool(false)) => s1,
                    _ => Predicate::or(s1, s2),
                }
            }
            Predicate::Not(p) => {
                let s = p.simplify();
                match s {
                    Predicate::Bool(b) => Predicate::Bool(!b),
                    Predicate::Not(inner) => *inner,
                    _ => Predicate::not(s),
                }
            }
            Predicate::Implies(p1, p2) => {
                let s1 = p1.simplify();
                let s2 = p2.simplify();
                match (&s1, &s2) {
                    (Predicate::Bool(false), _) => Predicate::Bool(true),
                    (Predicate::Bool(true), _) => s2,
                    (_, Predicate::Bool(true)) => Predicate::Bool(true),
                    _ => Predicate::implies(s1, s2),
                }
            }
            _ => self.clone(),
        }
    }
}

impl RefinementExpr {
    pub fn var(name: &str) -> Self {
        RefinementExpr::Var(RefinedVar::new(name))
    }

    pub fn int(n: i64) -> Self {
        RefinementExpr::Int(n)
    }

    pub fn float(f: f64) -> Self {
        RefinementExpr::Float(f)
    }

    pub fn add(e1: RefinementExpr, e2: RefinementExpr) -> Self {
        RefinementExpr::Arith {
            left: Box::new(e1),
            op: ArithOp::Add,
            right: Box::new(e2),
        }
    }

    pub fn sub(e1: RefinementExpr, e2: RefinementExpr) -> Self {
        RefinementExpr::Arith {
            left: Box::new(e1),
            op: ArithOp::Sub,
            right: Box::new(e2),
        }
    }

    pub fn mul(e1: RefinementExpr, e2: RefinementExpr) -> Self {
        RefinementExpr::Arith {
            left: Box::new(e1),
            op: ArithOp::Mul,
            right: Box::new(e2),
        }
    }

    pub fn div(e1: RefinementExpr, e2: RefinementExpr) -> Self {
        RefinementExpr::Arith {
            left: Box::new(e1),
            op: ArithOp::Div,
            right: Box::new(e2),
        }
    }

    pub fn collect_free_vars(&self, vars: &mut HashSet<String>) {
        match self {
            RefinementExpr::Int(_) | RefinementExpr::Float(_) => {}
            RefinementExpr::Var(v) => {
                vars.insert(v.name.clone());
            }
            RefinementExpr::Arith { left, right, .. } => {
                left.collect_free_vars(vars);
                right.collect_free_vars(vars);
            }
            RefinementExpr::Neg(e) | RefinementExpr::Len(e) | RefinementExpr::Old(e) => {
                e.collect_free_vars(vars);
            }
            RefinementExpr::Field { base, .. } => {
                base.collect_free_vars(vars);
            }
            RefinementExpr::Index { base, index } => {
                base.collect_free_vars(vars);
                index.collect_free_vars(vars);
            }
            RefinementExpr::Ite {
                cond,
                then_expr,
                else_expr,
            } => {
                cond.collect_free_vars(vars);
                then_expr.collect_free_vars(vars);
                else_expr.collect_free_vars(vars);
            }
            RefinementExpr::App { args, .. } => {
                for arg in args {
                    arg.collect_free_vars(vars);
                }
            }
        }
    }

    pub fn substitute(&self, from: &str, to: &RefinementExpr) -> Self {
        match self {
            RefinementExpr::Int(_) | RefinementExpr::Float(_) => self.clone(),
            RefinementExpr::Var(v) => {
                if v.name == from {
                    to.clone()
                } else {
                    self.clone()
                }
            }
            RefinementExpr::Arith { left, op, right } => RefinementExpr::Arith {
                left: Box::new(left.substitute(from, to)),
                op: *op,
                right: Box::new(right.substitute(from, to)),
            },
            RefinementExpr::Neg(e) => RefinementExpr::Neg(Box::new(e.substitute(from, to))),
            RefinementExpr::Len(e) => RefinementExpr::Len(Box::new(e.substitute(from, to))),
            RefinementExpr::Old(e) => RefinementExpr::Old(Box::new(e.substitute(from, to))),
            RefinementExpr::Field { base, field } => RefinementExpr::Field {
                base: Box::new(base.substitute(from, to)),
                field: field.clone(),
            },
            RefinementExpr::Index { base, index } => RefinementExpr::Index {
                base: Box::new(base.substitute(from, to)),
                index: Box::new(index.substitute(from, to)),
            },
            RefinementExpr::Ite {
                cond,
                then_expr,
                else_expr,
            } => RefinementExpr::Ite {
                cond: Box::new(cond.substitute(from, to)),
                then_expr: Box::new(then_expr.substitute(from, to)),
                else_expr: Box::new(else_expr.substitute(from, to)),
            },
            RefinementExpr::App { func, args } => RefinementExpr::App {
                func: func.clone(),
                args: args.iter().map(|a| a.substitute(from, to)).collect(),
            },
        }
    }
}

// ============================================================================
// Display implementations
// ============================================================================

impl fmt::Display for RefinementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{ {}: {} | {} }}",
            self.binder.name, self.base_type, self.predicate
        )
    }
}

impl fmt::Display for BaseTypeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BaseTypeRef::Named(n) => write!(f, "{}", n),
            BaseTypeRef::Var(v) => write!(f, "{}", v),
            BaseTypeRef::WithUnit { base, unit } => write!(f, "{}<{}>", base, unit),
        }
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Predicate::Bool(b) => write!(f, "{}", b),
            Predicate::Compare { left, op, right } => write!(f, "{} {} {}", left, op, right),
            Predicate::And(p1, p2) => write!(f, "({} && {})", p1, p2),
            Predicate::Or(p1, p2) => write!(f, "({} || {})", p1, p2),
            Predicate::Not(p) => write!(f, "!{}", p),
            Predicate::Implies(p1, p2) => write!(f, "({} => {})", p1, p2),
            Predicate::Ite {
                cond,
                then_pred,
                else_pred,
            } => write!(f, "(if {} then {} else {})", cond, then_pred, else_pred),
            Predicate::Forall { var, ty, body } => {
                write!(f, "forall({}: {}, {})", var.name, ty, body)
            }
            Predicate::Exists { var, ty, body } => {
                write!(f, "exists({}: {}, {})", var.name, ty, body)
            }
            Predicate::Call { func, args } => {
                write!(f, "{}(", func)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Predicate::Var(v) => write!(f, "{}", v.name),
        }
    }
}

impl fmt::Display for CompareOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

impl fmt::Display for RefinementExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RefinementExpr::Int(n) => write!(f, "{}", n),
            RefinementExpr::Float(x) => write!(f, "{}", x),
            RefinementExpr::Var(v) => write!(f, "{}", v.name),
            RefinementExpr::Arith { left, op, right } => write!(f, "({} {} {})", left, op, right),
            RefinementExpr::Neg(e) => write!(f, "-{}", e),
            RefinementExpr::Field { base, field } => write!(f, "{}.{}", base, field),
            RefinementExpr::Index { base, index } => write!(f, "{}[{}]", base, index),
            RefinementExpr::Ite {
                cond,
                then_expr,
                else_expr,
            } => write!(f, "(if {} then {} else {})", cond, then_expr, else_expr),
            RefinementExpr::App { func, args } => {
                write!(f, "{}(", func)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            RefinementExpr::Len(e) => write!(f, "len({})", e),
            RefinementExpr::Old(e) => write!(f, "old({})", e),
        }
    }
}

impl fmt::Display for ArithOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

impl fmt::Display for BuiltinFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuiltinFn::Abs => write!(f, "abs"),
            BuiltinFn::Sqrt => write!(f, "sqrt"),
            BuiltinFn::Exp => write!(f, "exp"),
            BuiltinFn::Log => write!(f, "log"),
            BuiltinFn::Log10 => write!(f, "log10"),
            BuiltinFn::Sin => write!(f, "sin"),
            BuiltinFn::Cos => write!(f, "cos"),
            BuiltinFn::Tan => write!(f, "tan"),
            BuiltinFn::Floor => write!(f, "floor"),
            BuiltinFn::Ceil => write!(f, "ceil"),
            BuiltinFn::Round => write!(f, "round"),
            BuiltinFn::Min => write!(f, "min"),
            BuiltinFn::Max => write!(f, "max"),
            BuiltinFn::ToInt => write!(f, "to_int"),
            BuiltinFn::ToReal => write!(f, "to_real"),
        }
    }
}

// ============================================================================
// Common medical refinement types
// ============================================================================

impl RefinementType {
    /// Positive value: { x: T | x > 0 }
    pub fn positive(name: &str, base: BaseTypeRef) -> Self {
        Self::new(RefinedVar::new(name), base, Predicate::positive(name))
    }

    /// Non-negative value: { x: T | x >= 0 }
    pub fn non_negative(name: &str, base: BaseTypeRef) -> Self {
        Self::new(RefinedVar::new(name), base, Predicate::non_negative(name))
    }

    /// Bounded value: { x: T | lo <= x && x <= hi }
    pub fn bounded(name: &str, base: BaseTypeRef, lo: f64, hi: f64) -> Self {
        Self::new(
            RefinedVar::new(name),
            base,
            Predicate::in_range(name, RefinementExpr::Float(lo), RefinementExpr::Float(hi)),
        )
    }

    /// Safe dose range for common medications
    pub fn safe_dose_mg(name: &str, min: f64, max: f64) -> Self {
        Self::bounded(
            name,
            BaseTypeRef::WithUnit {
                base: "Float".to_string(),
                unit: "mg".to_string(),
            },
            min,
            max,
        )
    }

    /// Valid creatinine clearance: { crcl: mL/min | crcl > 0 && crcl < 200 }
    pub fn valid_crcl(name: &str) -> Self {
        Self::new(
            RefinedVar::new(name),
            BaseTypeRef::WithUnit {
                base: "Float".to_string(),
                unit: "mL/min".to_string(),
            },
            Predicate::and(
                Predicate::gt(RefinementExpr::var(name), RefinementExpr::Int(0)),
                Predicate::lt(RefinementExpr::var(name), RefinementExpr::Int(200)),
            ),
        )
    }

    /// Valid body weight: { weight: kg | weight > 0 && weight < 500 }
    pub fn valid_weight(name: &str) -> Self {
        Self::new(
            RefinedVar::new(name),
            BaseTypeRef::WithUnit {
                base: "Float".to_string(),
                unit: "kg".to_string(),
            },
            Predicate::and(
                Predicate::gt(RefinementExpr::var(name), RefinementExpr::Int(0)),
                Predicate::lt(RefinementExpr::var(name), RefinementExpr::Int(500)),
            ),
        )
    }

    /// Valid age: { age: years | age >= 0 && age <= 150 }
    pub fn valid_age(name: &str) -> Self {
        Self::new(
            RefinedVar::new(name),
            BaseTypeRef::WithUnit {
                base: "Int".to_string(),
                unit: "years".to_string(),
            },
            Predicate::and(
                Predicate::ge(RefinementExpr::var(name), RefinementExpr::Int(0)),
                Predicate::le(RefinementExpr::var(name), RefinementExpr::Int(150)),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refinement_type_display() {
        let rt = RefinementType::bounded("dose", BaseTypeRef::Named("mg".to_string()), 0.5, 10.0);
        let s = format!("{}", rt);
        assert!(s.contains("dose"));
        assert!(s.contains("mg"));
    }

    #[test]
    fn test_predicate_simplify() {
        let p = Predicate::and(Predicate::Bool(true), Predicate::positive("x"));
        let simplified = p.simplify();
        assert!(matches!(simplified, Predicate::Compare { .. }));

        let p2 = Predicate::and(Predicate::Bool(false), Predicate::positive("x"));
        let simplified2 = p2.simplify();
        assert_eq!(simplified2, Predicate::Bool(false));
    }

    #[test]
    fn test_free_vars() {
        let p = Predicate::and(
            Predicate::gt(RefinementExpr::var("x"), RefinementExpr::Int(0)),
            Predicate::lt(RefinementExpr::var("y"), RefinementExpr::var("z")),
        );
        let vars = p.free_vars();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(vars.contains("z"));
        assert_eq!(vars.len(), 3);
    }

    #[test]
    fn test_substitute() {
        let p = Predicate::gt(RefinementExpr::var("x"), RefinementExpr::Int(0));
        let subst = p.substitute("x", &RefinementExpr::var("dose"));
        if let Predicate::Compare { left, .. } = subst {
            if let RefinementExpr::Var(v) = *left {
                assert_eq!(v.name, "dose");
            } else {
                panic!("Expected Var");
            }
        } else {
            panic!("Expected Compare");
        }
    }

    #[test]
    fn test_medical_refinements() {
        let crcl = RefinementType::valid_crcl("crcl");
        assert_eq!(crcl.binder.name, "crcl");

        let weight = RefinementType::valid_weight("w");
        let vars = weight.predicate.free_vars();
        assert!(vars.contains("w"));
    }
}
