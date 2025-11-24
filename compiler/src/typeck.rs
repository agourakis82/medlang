//! Type checking and dimensional analysis for MedLang.
//!
//! This module implements:
//! - Type inference for expressions
//! - Dimensional analysis for unit checking
//! - Compatibility checking for assignments and operations

use crate::ast::*;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum TypeError {
    #[error("Undefined variable: {0}")]
    UndefinedVariable(String),

    #[error("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    #[error("Unit mismatch: {operation} requires compatible units, got {left} and {right}")]
    UnitMismatch {
        operation: String,
        left: String,
        right: String,
    },

    #[error("Dimension mismatch in {context}: expected {expected}, found {found}")]
    DimensionMismatch {
        context: String,
        expected: String,
        found: String,
    },

    #[error("Cannot apply {operation} to type {type_name}")]
    InvalidOperation {
        operation: String,
        type_name: String,
    },

    #[error("Undefined model: {0}")]
    UndefinedModel(String),

    #[error("Undefined field: {0} in {1}")]
    UndefinedField(String, String),

    #[error("Function {function} expects {expected} arguments, got {found}")]
    WrongArgumentCount {
        function: String,
        expected: usize,
        found: usize,
    },
}

/// Represents the inferred type of an expression
#[derive(Debug, Clone, PartialEq)]
pub enum InferredType {
    /// Scalar floating-point (dimensionless)
    Float,

    /// Quantity with specific unit dimensions
    Quantity(UnitDimension),

    /// Boolean (for comparisons)
    Bool,

    /// Unknown/uninferred type
    Unknown,
}

impl std::fmt::Display for InferredType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferredType::Float => write!(f, "f64"),
            InferredType::Quantity(dim) => write!(f, "{}", dim),
            InferredType::Bool => write!(f, "bool"),
            InferredType::Unknown => write!(f, "?"),
        }
    }
}

/// Represents unit dimensions for dimensional analysis
#[derive(Debug, Clone, PartialEq)]
pub struct UnitDimension {
    pub mass: i32,   // M
    pub length: i32, // L (via Volume = L^3)
    pub time: i32,   // T
}

impl UnitDimension {
    pub fn dimensionless() -> Self {
        Self {
            mass: 0,
            length: 0,
            time: 0,
        }
    }

    pub fn mass() -> Self {
        Self {
            mass: 1,
            length: 0,
            time: 0,
        }
    }

    pub fn volume() -> Self {
        Self {
            mass: 0,
            length: 3,
            time: 0,
        }
    }

    pub fn time() -> Self {
        Self {
            mass: 0,
            length: 0,
            time: 1,
        }
    }

    pub fn clearance() -> Self {
        // Clearance = Volume / Time = L^3 / T
        Self {
            mass: 0,
            length: 3,
            time: -1,
        }
    }

    pub fn rate_const() -> Self {
        // RateConst = 1 / Time
        Self {
            mass: 0,
            length: 0,
            time: -1,
        }
    }

    pub fn conc_mass() -> Self {
        // ConcMass = Mass / Volume = M / L^3
        Self {
            mass: 1,
            length: -3,
            time: 0,
        }
    }

    /// Multiply dimensions (for multiplication operation)
    pub fn multiply(&self, other: &UnitDimension) -> UnitDimension {
        UnitDimension {
            mass: self.mass + other.mass,
            length: self.length + other.length,
            time: self.time + other.time,
        }
    }

    /// Divide dimensions (for division operation)
    pub fn divide(&self, other: &UnitDimension) -> UnitDimension {
        UnitDimension {
            mass: self.mass - other.mass,
            length: self.length - other.length,
            time: self.time - other.time,
        }
    }

    /// Check if two dimensions are compatible (same)
    pub fn compatible_with(&self, other: &UnitDimension) -> bool {
        self.mass == other.mass && self.length == other.length && self.time == other.time
    }
}

impl std::fmt::Display for UnitDimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.mass == 0 && self.length == 0 && self.time == 0 {
            return write!(f, "dimensionless");
        }

        let mut parts = Vec::new();
        if self.mass != 0 {
            if self.mass == 1 {
                parts.push("M".to_string());
            } else {
                parts.push(format!("M^{}", self.mass));
            }
        }
        if self.length != 0 {
            if self.length == 1 {
                parts.push("L".to_string());
            } else {
                parts.push(format!("L^{}", self.length));
            }
        }
        if self.time != 0 {
            if self.time == 1 {
                parts.push("T".to_string());
            } else {
                parts.push(format!("T^{}", self.time));
            }
        }

        write!(f, "{}", parts.join("·"))
    }
}

/// Type checking context with symbol table
pub struct TypeContext {
    /// Variable name -> inferred type
    variables: HashMap<String, InferredType>,

    /// Model definitions (name -> ModelDef)
    models: HashMap<String, ModelDef>,

    /// Scoped variables (for bind_params blocks)
    scopes: Vec<HashMap<String, InferredType>>,
}

impl TypeContext {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            models: HashMap::new(),
            scopes: Vec::new(),
        }
    }

    /// Register a model definition
    pub fn register_model(&mut self, model: &ModelDef) {
        self.models.insert(model.name.clone(), model.clone());
    }

    /// Declare a variable with a type
    pub fn declare(&mut self, name: String, ty: InferredType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ty);
        } else {
            self.variables.insert(name, ty);
        }
    }

    /// Look up a variable's type
    pub fn lookup(&self, name: &str) -> Option<&InferredType> {
        // Check scopes from innermost to outermost
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        self.variables.get(name)
    }

    /// Push a new scope
    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Pop the current scope
    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Convert TypeExpr to InferredType
    pub fn type_expr_to_inferred(&self, ty: &TypeExpr) -> InferredType {
        match ty {
            TypeExpr::Simple(name) if name == "f64" => InferredType::Float,
            TypeExpr::Unit(ut) => InferredType::Quantity(self.unit_type_to_dimension(ut)),
            TypeExpr::Quantity(_, _scalar) => {
                // For now, treat Quantity<unit, f64> as the unit type
                // TODO: handle the unit expression properly
                InferredType::Float
            }
            _ => InferredType::Unknown,
        }
    }

    fn unit_type_to_dimension(&self, ut: &UnitType) -> UnitDimension {
        match ut {
            UnitType::Mass => UnitDimension::mass(),
            UnitType::Volume => UnitDimension::volume(),
            UnitType::Time => UnitDimension::time(),
            UnitType::DoseMass => UnitDimension::mass(),
            UnitType::ConcMass => UnitDimension::conc_mass(),
            UnitType::Clearance => UnitDimension::clearance(),
            UnitType::RateConst => UnitDimension::rate_const(),
            UnitType::TumourVolume => UnitDimension::volume(), // TumourVolume is a volume (mm³ or cm³)
        }
    }
}

impl Default for TypeContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Type checker for MedLang programs
pub struct TypeChecker {
    ctx: TypeContext,
    errors: Vec<TypeError>,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            ctx: TypeContext::new(),
            errors: Vec::new(),
        }
    }

    /// Check a complete program
    pub fn check_program(&mut self, program: &Program) -> Result<(), Vec<TypeError>> {
        // First pass: register all models
        for decl in &program.declarations {
            if let Declaration::Model(model) = decl {
                self.ctx.register_model(model);
            }
        }

        // Second pass: check each declaration
        for decl in &program.declarations {
            self.check_declaration(decl);
        }

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.clone())
        }
    }

    fn check_declaration(&mut self, decl: &Declaration) {
        match decl {
            Declaration::Model(model) => self.check_model(model),
            Declaration::Population(pop) => self.check_population(pop),
            Declaration::Measure(measure) => self.check_measure(measure),
            Declaration::Timeline(_) => {} // Timeline doesn't need deep type checking for V0
            Declaration::Cohort(_) => {}   // Cohort doesn't need deep type checking for V0
            Declaration::Protocol(_) => {} // Protocol type checking to be added in Week 8
            Declaration::Evidence(_) => {} // Evidence program type checking (Week 24)
            Declaration::Enum(_) => {} // Enum type checking (Week 27) - handled by core_lang typecheck
        }
    }

    fn check_model(&mut self, model: &ModelDef) {
        // Declare all states and params
        for item in &model.items {
            match item {
                ModelItem::State(state) => {
                    let ty = self.ctx.type_expr_to_inferred(&state.ty);
                    self.ctx.declare(state.name.clone(), ty);
                }
                ModelItem::Param(param) => {
                    let ty = self.ctx.type_expr_to_inferred(&param.ty);
                    self.ctx.declare(param.name.clone(), ty);
                }
                _ => {}
            }
        }

        // Check ODEs and observables
        for item in &model.items {
            match item {
                ModelItem::ODE(ode) => self.check_ode(ode),
                ModelItem::Observable(obs) => self.check_observable(obs),
                _ => {}
            }
        }
    }

    fn check_ode(&mut self, ode: &ODEEquation) {
        // Check that state variable exists
        let state_ty = match self.ctx.lookup(&ode.state_name) {
            Some(ty) => ty.clone(),
            None => {
                self.errors
                    .push(TypeError::UndefinedVariable(ode.state_name.clone()));
                return;
            }
        };

        // Infer RHS type
        let rhs_ty = self.infer_expr(&ode.rhs);

        // ODE: dS/dt = expr
        // expr should have dimensions of S / Time
        match (&state_ty, &rhs_ty) {
            (InferredType::Quantity(state_dim), InferredType::Quantity(rhs_dim)) => {
                // Expected: state_dim / time
                let expected = state_dim.divide(&UnitDimension::time());
                if !expected.compatible_with(rhs_dim) {
                    self.errors.push(TypeError::DimensionMismatch {
                        context: format!("ODE d{}/dt", ode.state_name),
                        expected: expected.to_string(),
                        found: rhs_dim.to_string(),
                    });
                }
            }
            _ => {
                // For now, accept Float on either side
            }
        }
    }

    fn check_observable(&mut self, obs: &ObservableDecl) {
        let declared_ty = self.ctx.type_expr_to_inferred(&obs.ty);
        let expr_ty = self.infer_expr(&obs.expr);

        // Check compatibility
        match (&declared_ty, &expr_ty) {
            (InferredType::Quantity(decl_dim), InferredType::Quantity(expr_dim)) => {
                if !decl_dim.compatible_with(expr_dim) {
                    self.errors.push(TypeError::DimensionMismatch {
                        context: format!("Observable {}", obs.name),
                        expected: decl_dim.to_string(),
                        found: expr_dim.to_string(),
                    });
                }
            }
            _ => {}
        }

        // Register the observable
        self.ctx.declare(obs.name.clone(), declared_ty);
    }

    fn check_population(&mut self, _pop: &PopulationDef) {
        // For V0, we'll skip detailed population type checking
        // This would include checking random effect distributions, bind_params logic, etc.
    }

    fn check_measure(&mut self, _measure: &MeasureDef) {
        // For V0, we'll skip detailed measure type checking
    }

    /// Infer the type of an expression
    pub fn infer_expr(&self, expr: &Expr) -> InferredType {
        match &expr.kind {
            ExprKind::Literal(lit) => self.infer_literal(lit),
            ExprKind::Ident(name) => self
                .ctx
                .lookup(name)
                .cloned()
                .unwrap_or(InferredType::Unknown),
            ExprKind::QualifiedName(_) => {
                // For now, return unknown
                // TODO: resolve qualified names
                InferredType::Unknown
            }
            ExprKind::Unary(op, operand) => self.infer_unary(op, operand),
            ExprKind::Binary(op, left, right) => self.infer_binary(op, left, right),
            ExprKind::Call(name, args) => self.infer_call(name, args),
            // Week 27: Enum variants and match expressions
            // These are type-checked by core_lang::typecheck_expr
            ExprKind::EnumVariant { .. } => InferredType::Unknown,
            ExprKind::Match { .. } => InferredType::Unknown,
        }
    }

    pub fn infer_literal(&self, lit: &Literal) -> InferredType {
        match lit {
            Literal::Float(_) => InferredType::Float,
            Literal::UnitFloat { unit, .. } => {
                // Parse unit to determine dimension
                // For V0, we'll do simple string matching
                match unit.as_str() {
                    "mg" | "kg" | "g" => InferredType::Quantity(UnitDimension::mass()),
                    "L" | "mL" => InferredType::Quantity(UnitDimension::volume()),
                    "h" | "hr" | "min" | "s" => InferredType::Quantity(UnitDimension::time()),
                    _ => InferredType::Unknown,
                }
            }
        }
    }

    fn infer_unary(&self, op: &UnaryOp, operand: &Expr) -> InferredType {
        let operand_ty = self.infer_expr(operand);
        match op {
            UnaryOp::Neg | UnaryOp::Pos => operand_ty, // Preserves type
        }
    }

    fn infer_binary(&self, op: &BinaryOp, left: &Expr, right: &Expr) -> InferredType {
        let left_ty = self.infer_expr(left);
        let right_ty = self.infer_expr(right);

        match op {
            BinaryOp::Add | BinaryOp::Sub => {
                // Addition/subtraction requires compatible units
                match (&left_ty, &right_ty) {
                    (InferredType::Quantity(l), InferredType::Quantity(r)) => {
                        if l.compatible_with(r) {
                            left_ty
                        } else {
                            InferredType::Unknown
                        }
                    }
                    (InferredType::Float, InferredType::Float) => InferredType::Float,
                    _ => InferredType::Unknown,
                }
            }
            BinaryOp::Mul => {
                // Multiplication combines dimensions
                match (&left_ty, &right_ty) {
                    (InferredType::Quantity(l), InferredType::Quantity(r)) => {
                        InferredType::Quantity(l.multiply(r))
                    }
                    (InferredType::Quantity(q), InferredType::Float)
                    | (InferredType::Float, InferredType::Quantity(q)) => {
                        InferredType::Quantity(q.clone())
                    }
                    (InferredType::Float, InferredType::Float) => InferredType::Float,
                    _ => InferredType::Unknown,
                }
            }
            BinaryOp::Div => {
                // Division subtracts dimensions
                match (&left_ty, &right_ty) {
                    (InferredType::Quantity(l), InferredType::Quantity(r)) => {
                        InferredType::Quantity(l.divide(r))
                    }
                    (InferredType::Quantity(q), InferredType::Float) => {
                        InferredType::Quantity(q.clone())
                    }
                    (InferredType::Float, InferredType::Quantity(q)) => {
                        InferredType::Quantity(UnitDimension::dimensionless().divide(q))
                    }
                    (InferredType::Float, InferredType::Float) => InferredType::Float,
                    _ => InferredType::Unknown,
                }
            }
            BinaryOp::Pow => {
                // Power: base type, exponent must be dimensionless
                left_ty
            }
            BinaryOp::Lt | BinaryOp::Gt | BinaryOp::Eq | BinaryOp::Ne => InferredType::Bool,
        }
    }

    fn infer_call(&self, name: &str, _args: &[Argument]) -> InferredType {
        // Built-in functions
        match name {
            "exp" | "log" | "sqrt" => InferredType::Float,
            "pow" => InferredType::Float,
            _ => InferredType::Unknown,
        }
    }

    pub fn get_errors(&self) -> &[TypeError] {
        &self.errors
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_dimension_operations() {
        let mass = UnitDimension::mass();
        let volume = UnitDimension::volume();
        let time = UnitDimension::time();

        // Mass / Volume = Concentration
        let conc = mass.divide(&volume);
        assert_eq!(conc.mass, 1);
        assert_eq!(conc.length, -3);
        assert_eq!(conc.time, 0);

        // Volume / Time = Clearance
        let clearance = volume.divide(&time);
        assert_eq!(clearance.mass, 0);
        assert_eq!(clearance.length, 3);
        assert_eq!(clearance.time, -1);
    }

    #[test]
    fn test_unit_compatibility() {
        let mass1 = UnitDimension::mass();
        let mass2 = UnitDimension::mass();
        let volume = UnitDimension::volume();

        assert!(mass1.compatible_with(&mass2));
        assert!(!mass1.compatible_with(&volume));
    }

    #[test]
    fn test_type_context() {
        let mut ctx = TypeContext::new();

        ctx.declare(
            "CL".to_string(),
            InferredType::Quantity(UnitDimension::clearance()),
        );

        let ty = ctx.lookup("CL");
        assert!(matches!(ty, Some(InferredType::Quantity(_))));
    }

    #[test]
    fn test_literal_inference() {
        let checker = TypeChecker::new();

        let float_lit = Literal::Float(100.0);
        let ty = checker.infer_literal(&float_lit);
        assert!(matches!(ty, InferredType::Float));

        let unit_lit = Literal::UnitFloat {
            value: 100.0,
            unit: "mg".to_string(),
        };
        let ty = checker.infer_literal(&unit_lit);
        assert!(matches!(ty, InferredType::Quantity(_)));
    }
}
