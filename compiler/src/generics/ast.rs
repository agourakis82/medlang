// Week 52: Parametric Polymorphism - AST Extensions
//
// This module defines AST nodes for generic type annotations,
// type parameters, and generic function declarations.

use serde::{Deserialize, Serialize};

/// Identifier type
pub type Ident = String;

// =============================================================================
// Type Expressions (Syntax)
// =============================================================================

/// Type expression in source code
///
/// This represents the syntactic form of types as they appear in source.
/// These are later resolved to semantic `Type` values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeExprAst {
    /// Simple type name: Int, Float, Bool, etc.
    Simple(Ident),

    /// Type variable: T, U, etc.
    Var(Ident),

    /// Function type: (T1, T2) -> R
    Function {
        params: Vec<TypeExprAst>,
        ret: Box<TypeExprAst>,
    },

    /// List/Array type: [T]
    List(Box<TypeExprAst>),

    /// Option type: Option<T> or T?
    Option(Box<TypeExprAst>),

    /// Result type: Result<T, E>
    Result {
        ok: Box<TypeExprAst>,
        err: Box<TypeExprAst>,
    },

    /// Tuple type: (T1, T2, T3)
    Tuple(Vec<TypeExprAst>),

    /// Record type: { field1: T1, field2: T2 }
    Record(Vec<(Ident, TypeExprAst)>),

    /// Generic type application: Vec<T>, Map<K, V>
    App {
        constructor: Ident,
        args: Vec<TypeExprAst>,
    },

    /// Unit type: ()
    Unit,

    /// Inferred type (placeholder): _
    Infer,
}

impl TypeExprAst {
    /// Simple type constructor
    pub fn simple(name: impl Into<String>) -> Self {
        TypeExprAst::Simple(name.into())
    }

    /// Type variable constructor
    pub fn var(name: impl Into<String>) -> Self {
        TypeExprAst::Var(name.into())
    }

    /// Function type constructor
    pub fn function(params: Vec<TypeExprAst>, ret: TypeExprAst) -> Self {
        TypeExprAst::Function {
            params,
            ret: Box::new(ret),
        }
    }

    /// List type constructor
    pub fn list(elem: TypeExprAst) -> Self {
        TypeExprAst::List(Box::new(elem))
    }

    /// Option type constructor
    pub fn option(inner: TypeExprAst) -> Self {
        TypeExprAst::Option(Box::new(inner))
    }

    /// Result type constructor
    pub fn result(ok: TypeExprAst, err: TypeExprAst) -> Self {
        TypeExprAst::Result {
            ok: Box::new(ok),
            err: Box::new(err),
        }
    }

    /// Generic application constructor
    pub fn app(constructor: impl Into<String>, args: Vec<TypeExprAst>) -> Self {
        TypeExprAst::App {
            constructor: constructor.into(),
            args,
        }
    }

    /// Check if this is a type variable
    pub fn is_var(&self) -> bool {
        matches!(self, TypeExprAst::Var(_))
    }

    /// Get the name if this is a simple type or variable
    pub fn as_name(&self) -> Option<&str> {
        match self {
            TypeExprAst::Simple(name) | TypeExprAst::Var(name) => Some(name),
            _ => None,
        }
    }
}

impl std::fmt::Display for TypeExprAst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeExprAst::Simple(name) => write!(f, "{}", name),
            TypeExprAst::Var(name) => write!(f, "{}", name),
            TypeExprAst::Function { params, ret } => {
                write!(f, "(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            TypeExprAst::List(elem) => write!(f, "[{}]", elem),
            TypeExprAst::Option(inner) => write!(f, "Option<{}>", inner),
            TypeExprAst::Result { ok, err } => write!(f, "Result<{}, {}>", ok, err),
            TypeExprAst::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e)?;
                }
                write!(f, ")")
            }
            TypeExprAst::Record(fields) => {
                write!(f, "{{ ")?;
                for (i, (name, ty)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", name, ty)?;
                }
                write!(f, " }}")
            }
            TypeExprAst::App { constructor, args } => {
                write!(f, "{}<", constructor)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", a)?;
                }
                write!(f, ">")
            }
            TypeExprAst::Unit => write!(f, "()"),
            TypeExprAst::Infer => write!(f, "_"),
        }
    }
}

// =============================================================================
// Type Parameters
// =============================================================================

/// Type bound in source code: T: Bound
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeBoundAst {
    /// Trait bound: T: SomeTrait
    Trait(Ident),
    /// Numeric bound: T: Num
    Num,
    /// Ordering bound: T: Ord
    Ord,
    /// Equality bound: T: Eq
    Eq,
    /// Copy bound: T: Copy
    Copy,
}

impl std::fmt::Display for TypeBoundAst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeBoundAst::Trait(name) => write!(f, "{}", name),
            TypeBoundAst::Num => write!(f, "Num"),
            TypeBoundAst::Ord => write!(f, "Ord"),
            TypeBoundAst::Eq => write!(f, "Eq"),
            TypeBoundAst::Copy => write!(f, "Copy"),
        }
    }
}

/// Type parameter declaration: T or T: Bound
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeParamAst {
    /// Parameter name (T, U, etc.)
    pub name: Ident,
    /// Optional bounds
    pub bounds: Vec<TypeBoundAst>,
}

impl TypeParamAst {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            bounds: Vec::new(),
        }
    }

    pub fn with_bound(mut self, bound: TypeBoundAst) -> Self {
        self.bounds.push(bound);
        self
    }

    pub fn with_bounds(mut self, bounds: Vec<TypeBoundAst>) -> Self {
        self.bounds = bounds;
        self
    }
}

impl std::fmt::Display for TypeParamAst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)?;
        if !self.bounds.is_empty() {
            write!(f, ": ")?;
            for (i, b) in self.bounds.iter().enumerate() {
                if i > 0 {
                    write!(f, " + ")?;
                }
                write!(f, "{}", b)?;
            }
        }
        Ok(())
    }
}

// =============================================================================
// Generic Function Declaration
// =============================================================================

/// A generic function declaration
///
/// Syntax: fn name<T, U>(params) -> RetType { body }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenericFnDecl {
    /// Function name
    pub name: Ident,
    /// Type parameters: <T, U, ...>
    pub type_params: Vec<TypeParamAst>,
    /// Value parameters: (x: T, y: U, ...)
    pub params: Vec<GenericParam>,
    /// Return type
    pub ret_type: Option<TypeExprAst>,
    /// Function body
    pub body: GenericBlock,
}

impl GenericFnDecl {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            type_params: Vec::new(),
            params: Vec::new(),
            ret_type: None,
            body: GenericBlock::empty(),
        }
    }

    pub fn with_type_params(mut self, params: Vec<TypeParamAst>) -> Self {
        self.type_params = params;
        self
    }

    pub fn with_params(mut self, params: Vec<GenericParam>) -> Self {
        self.params = params;
        self
    }

    pub fn with_ret_type(mut self, ty: TypeExprAst) -> Self {
        self.ret_type = Some(ty);
        self
    }

    pub fn with_body(mut self, body: GenericBlock) -> Self {
        self.body = body;
        self
    }

    /// Check if this is a generic function
    pub fn is_generic(&self) -> bool {
        !self.type_params.is_empty()
    }

    /// Get the arity (number of type parameters)
    pub fn type_arity(&self) -> usize {
        self.type_params.len()
    }
}

impl std::fmt::Display for GenericFnDecl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fn {}", self.name)?;
        if !self.type_params.is_empty() {
            write!(f, "<")?;
            for (i, tp) in self.type_params.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", tp)?;
            }
            write!(f, ">")?;
        }
        write!(f, "(")?;
        for (i, p) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", p)?;
        }
        write!(f, ")")?;
        if let Some(ret) = &self.ret_type {
            write!(f, " -> {}", ret)?;
        }
        Ok(())
    }
}

/// A parameter in a generic function
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenericParam {
    pub name: Ident,
    pub ty: Option<TypeExprAst>,
}

impl GenericParam {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: None,
        }
    }

    pub fn typed(name: impl Into<String>, ty: TypeExprAst) -> Self {
        Self {
            name: name.into(),
            ty: Some(ty),
        }
    }
}

impl std::fmt::Display for GenericParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)?;
        if let Some(ty) = &self.ty {
            write!(f, ": {}", ty)?;
        }
        Ok(())
    }
}

// =============================================================================
// Generic Expressions and Statements
// =============================================================================

/// A block of statements in a generic function
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenericBlock {
    pub stmts: Vec<GenericStmt>,
}

impl GenericBlock {
    pub fn new(stmts: Vec<GenericStmt>) -> Self {
        Self { stmts }
    }

    pub fn empty() -> Self {
        Self { stmts: Vec::new() }
    }
}

/// Statement in a generic function
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GenericStmt {
    /// Let binding: let x: T = expr
    Let {
        name: Ident,
        ty: Option<TypeExprAst>,
        expr: GenericExpr,
    },
    /// Expression statement
    Expr(GenericExpr),
    /// Return statement
    Return(Option<GenericExpr>),
}

/// Expression in a generic context
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GenericExpr {
    /// Literal values
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    StringLit(String),

    /// Variable reference
    Var(Ident),

    /// Function call with optional type arguments: f<T, U>(args)
    Call {
        callee: Box<GenericExpr>,
        type_args: Vec<TypeExprAst>,
        args: Vec<GenericExpr>,
    },

    /// Field access: expr.field
    FieldAccess {
        target: Box<GenericExpr>,
        field: Ident,
    },

    /// If expression
    If {
        cond: Box<GenericExpr>,
        then_branch: Box<GenericExpr>,
        else_branch: Box<GenericExpr>,
    },

    /// Block expression
    Block(GenericBlock),

    /// Record literal
    Record(Vec<(Ident, GenericExpr)>),

    /// List literal
    List(Vec<GenericExpr>),

    /// Tuple literal
    Tuple(Vec<GenericExpr>),

    /// Lambda expression: |x, y| expr or |x: T, y: U| -> R expr
    Lambda {
        params: Vec<GenericParam>,
        ret_type: Option<TypeExprAst>,
        body: Box<GenericExpr>,
    },

    /// Binary operation
    Binary {
        op: BinaryOpAst,
        left: Box<GenericExpr>,
        right: Box<GenericExpr>,
    },

    /// Unary operation
    Unary {
        op: UnaryOpAst,
        operand: Box<GenericExpr>,
    },

    /// Type ascription: expr : Type
    Ascription {
        expr: Box<GenericExpr>,
        ty: TypeExprAst,
    },
}

impl GenericExpr {
    pub fn int(value: i64) -> Self {
        GenericExpr::IntLit(value)
    }

    pub fn float(value: f64) -> Self {
        GenericExpr::FloatLit(value)
    }

    pub fn bool_val(value: bool) -> Self {
        GenericExpr::BoolLit(value)
    }

    pub fn string(value: impl Into<String>) -> Self {
        GenericExpr::StringLit(value.into())
    }

    pub fn var(name: impl Into<String>) -> Self {
        GenericExpr::Var(name.into())
    }

    pub fn call(callee: GenericExpr, args: Vec<GenericExpr>) -> Self {
        GenericExpr::Call {
            callee: Box::new(callee),
            type_args: Vec::new(),
            args,
        }
    }

    pub fn call_generic(
        callee: GenericExpr,
        type_args: Vec<TypeExprAst>,
        args: Vec<GenericExpr>,
    ) -> Self {
        GenericExpr::Call {
            callee: Box::new(callee),
            type_args,
            args,
        }
    }

    pub fn if_expr(cond: GenericExpr, then_branch: GenericExpr, else_branch: GenericExpr) -> Self {
        GenericExpr::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        }
    }

    pub fn lambda(params: Vec<GenericParam>, body: GenericExpr) -> Self {
        GenericExpr::Lambda {
            params,
            ret_type: None,
            body: Box::new(body),
        }
    }
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOpAst {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    And,
    Or,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl std::fmt::Display for BinaryOpAst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOpAst::Add => write!(f, "+"),
            BinaryOpAst::Sub => write!(f, "-"),
            BinaryOpAst::Mul => write!(f, "*"),
            BinaryOpAst::Div => write!(f, "/"),
            BinaryOpAst::Mod => write!(f, "%"),
            BinaryOpAst::Pow => write!(f, "^"),
            BinaryOpAst::And => write!(f, "&&"),
            BinaryOpAst::Or => write!(f, "||"),
            BinaryOpAst::Eq => write!(f, "=="),
            BinaryOpAst::Ne => write!(f, "!="),
            BinaryOpAst::Lt => write!(f, "<"),
            BinaryOpAst::Le => write!(f, "<="),
            BinaryOpAst::Gt => write!(f, ">"),
            BinaryOpAst::Ge => write!(f, ">="),
        }
    }
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOpAst {
    Neg,
    Not,
}

impl std::fmt::Display for UnaryOpAst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOpAst::Neg => write!(f, "-"),
            UnaryOpAst::Not => write!(f, "!"),
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
    fn test_type_expr_display() {
        let simple = TypeExprAst::simple("Int");
        assert_eq!(simple.to_string(), "Int");

        let func = TypeExprAst::function(
            vec![TypeExprAst::var("T"), TypeExprAst::simple("Int")],
            TypeExprAst::var("U"),
        );
        assert_eq!(func.to_string(), "(T, Int) -> U");

        let list = TypeExprAst::list(TypeExprAst::simple("String"));
        assert_eq!(list.to_string(), "[String]");

        let app = TypeExprAst::app(
            "Map",
            vec![TypeExprAst::simple("String"), TypeExprAst::simple("Int")],
        );
        assert_eq!(app.to_string(), "Map<String, Int>");
    }

    #[test]
    fn test_type_param_display() {
        let simple = TypeParamAst::new("T");
        assert_eq!(simple.to_string(), "T");

        let bounded = TypeParamAst::new("T")
            .with_bound(TypeBoundAst::Num)
            .with_bound(TypeBoundAst::Ord);
        assert_eq!(bounded.to_string(), "T: Num + Ord");
    }

    #[test]
    fn test_generic_fn_decl_display() {
        let fn_decl = GenericFnDecl::new("map")
            .with_type_params(vec![TypeParamAst::new("T"), TypeParamAst::new("U")])
            .with_params(vec![
                GenericParam::typed(
                    "f",
                    TypeExprAst::function(vec![TypeExprAst::var("T")], TypeExprAst::var("U")),
                ),
                GenericParam::typed("xs", TypeExprAst::list(TypeExprAst::var("T"))),
            ])
            .with_ret_type(TypeExprAst::list(TypeExprAst::var("U")));

        let expected = "fn map<T, U>(f: (T) -> U, xs: [T]) -> [U]";
        assert_eq!(fn_decl.to_string(), expected);
    }

    #[test]
    fn test_generic_fn_is_generic() {
        let non_generic = GenericFnDecl::new("foo");
        assert!(!non_generic.is_generic());

        let generic = GenericFnDecl::new("foo").with_type_params(vec![TypeParamAst::new("T")]);
        assert!(generic.is_generic());
    }

    #[test]
    fn test_generic_expr_call() {
        let call = GenericExpr::call_generic(
            GenericExpr::var("map"),
            vec![TypeExprAst::simple("Int"), TypeExprAst::simple("String")],
            vec![GenericExpr::var("f"), GenericExpr::var("xs")],
        );

        match call {
            GenericExpr::Call {
                type_args, args, ..
            } => {
                assert_eq!(type_args.len(), 2);
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Call"),
        }
    }

    #[test]
    fn test_generic_lambda() {
        let lambda = GenericExpr::lambda(
            vec![GenericParam::typed("x", TypeExprAst::var("T"))],
            GenericExpr::var("x"),
        );

        match lambda {
            GenericExpr::Lambda { params, body, .. } => {
                assert_eq!(params.len(), 1);
                assert_eq!(params[0].name, "x");
                match *body {
                    GenericExpr::Var(name) => assert_eq!(name, "x"),
                    _ => panic!("Expected Var"),
                }
            }
            _ => panic!("Expected Lambda"),
        }
    }
}
