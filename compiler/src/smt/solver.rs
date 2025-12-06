// Phase V2: Z3 Solver Interface
//
// Provides a high-level interface to the Z3 SMT solver
// When compiled without the smt-verification feature, provides a mock implementation

use super::translator::{ComparisonOp, SMTExpr, SMTFormula};
use super::vc_gen::VerificationCondition;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Result of SMT solving
#[derive(Debug)]
pub enum Z3Result {
    /// Constraint is proven (UNSAT when goal is negated)
    Proven,
    /// Counterexample found (SAT when goal is negated)
    Counterexample(Z3Model),
    /// Unknown (timeout or too complex)
    Unknown,
}

/// Z3 model (counterexample or satisfying assignment)
#[derive(Debug, Clone)]
pub struct Z3Model {
    pub assignments: HashMap<String, Z3Value>,
}

#[derive(Debug, Clone)]
pub enum Z3Value {
    Real(f64),
    Int(i64),
    Bool(bool),
}

// ====================================================================================
// Implementation WITH Z3 support (when smt-verification feature is enabled)
// ====================================================================================

#[cfg(feature = "smt-verification")]
mod z3_impl {
    use super::*;
    use z3::ast::{Ast, Bool, Dynamic, Int, Real};
    use z3::{Config, Context, Model, SatResult, Solver};

    pub struct Z3Solver {
        ctx: Context,
        solver: Solver<'static>,
        var_map: HashMap<String, Dynamic<'static>>,
    }

    impl Z3Solver {
        pub fn new() -> Self {
            let cfg = Config::new();
            let ctx = Context::new(&cfg);
            let solver = Solver::new(&ctx);

            // Leak context to get 'static lifetime
            let ctx_static = Box::leak(Box::new(ctx));
            let solver_static = Box::leak(Box::new(solver));

            Z3Solver {
                ctx: unsafe { std::ptr::read(ctx_static as *const Context) },
                solver: unsafe { std::ptr::read(solver_static as *const Solver) },
                var_map: HashMap::new(),
            }
        }

        pub fn assert(&mut self, formula: SMTFormula) {
            if let Ok(z3_formula) = self.translate_formula(&formula) {
                if let Some(bool_ast) = z3_formula.as_bool() {
                    self.solver.assert(bool_ast);
                }
            }
        }

        pub fn check_vc(&mut self, vc: &VerificationCondition) -> Z3Result {
            self.solver.push();

            for assumption in &vc.assumptions {
                self.assert(assumption.clone());
            }

            let negated_goal = SMTFormula::Not(Box::new(vc.goal.clone()));
            self.assert(negated_goal);

            let result = match self.solver.check() {
                SatResult::Unsat => Z3Result::Proven,
                SatResult::Sat => {
                    if let Some(model) = self.solver.get_model() {
                        Z3Result::Counterexample(self.extract_model(&model))
                    } else {
                        Z3Result::Unknown
                    }
                }
                SatResult::Unknown => Z3Result::Unknown,
            };

            self.solver.pop(1);
            result
        }

        pub fn reset(&mut self) {
            self.solver.reset();
            self.var_map.clear();
        }

        fn translate_formula(&mut self, formula: &SMTFormula) -> Result<Dynamic<'static>> {
            match formula {
                SMTFormula::Bool(b) => Ok(Dynamic::from_ast(&Bool::from_bool(&self.ctx, *b))),

                SMTFormula::Comparison { lhs, op, rhs } => {
                    let lhs_ast = self.translate_expr(lhs)?;
                    let rhs_ast = self.translate_expr(rhs)?;

                    if let (Some(lhs_real), Some(rhs_real)) = (lhs_ast.as_real(), rhs_ast.as_real())
                    {
                        let result = match op {
                            ComparisonOp::Lt => lhs_real.lt(rhs_real),
                            ComparisonOp::Le => lhs_real.le(rhs_real),
                            ComparisonOp::Gt => lhs_real.gt(rhs_real),
                            ComparisonOp::Ge => lhs_real.ge(rhs_real),
                            ComparisonOp::Eq => lhs_real._eq(rhs_real),
                            ComparisonOp::Ne => lhs_real._eq(rhs_real).not(),
                        };
                        Ok(Dynamic::from_ast(&result))
                    } else if let (Some(lhs_int), Some(rhs_int)) =
                        (lhs_ast.as_int(), rhs_ast.as_int())
                    {
                        let result = match op {
                            ComparisonOp::Lt => lhs_int.lt(rhs_int),
                            ComparisonOp::Le => lhs_int.le(rhs_int),
                            ComparisonOp::Gt => lhs_int.gt(rhs_int),
                            ComparisonOp::Ge => lhs_int.ge(rhs_int),
                            ComparisonOp::Eq => lhs_int._eq(rhs_int),
                            ComparisonOp::Ne => lhs_int._eq(rhs_int).not(),
                        };
                        Ok(Dynamic::from_ast(&result))
                    } else {
                        Err(anyhow!("Type mismatch in comparison"))
                    }
                }

                SMTFormula::Logical { op, operands } => {
                    let mut z3_operands = Vec::new();
                    for operand in operands {
                        let z3_op = self.translate_formula(operand)?;
                        if let Some(bool_ast) = z3_op.as_bool() {
                            z3_operands.push(bool_ast);
                        } else {
                            return Err(anyhow!("Expected boolean operand"));
                        }
                    }

                    use crate::ast::phase_v1::LogicalOp;
                    let result = match op {
                        LogicalOp::And => {
                            Bool::and(&self.ctx, &z3_operands.iter().collect::<Vec<_>>())
                        }
                        LogicalOp::Or => {
                            Bool::or(&self.ctx, &z3_operands.iter().collect::<Vec<_>>())
                        }
                    };
                    Ok(Dynamic::from_ast(&result))
                }

                SMTFormula::Not(inner) => {
                    let inner_ast = self.translate_formula(inner)?;
                    if let Some(bool_ast) = inner_ast.as_bool() {
                        Ok(Dynamic::from_ast(&bool_ast.not()))
                    } else {
                        Err(anyhow!("Expected boolean expression in negation"))
                    }
                }

                SMTFormula::Quantified { .. } => Err(anyhow!("Quantifiers not yet supported")),
            }
        }

        fn translate_expr(&mut self, expr: &SMTExpr) -> Result<Dynamic<'static>> {
            match expr {
                SMTExpr::Var(name) => {
                    if let Some(var) = self.var_map.get(name) {
                        Ok(var.clone())
                    } else {
                        let var = Real::new_const(&self.ctx, name.as_str());
                        let var_dyn = Dynamic::from_ast(&var);
                        self.var_map.insert(name.clone(), var_dyn.clone());
                        Ok(var_dyn)
                    }
                }

                SMTExpr::RealLit(val) => {
                    let real = Real::from_real(&self.ctx, (*val as i32) as i32, 1);
                    Ok(Dynamic::from_ast(&real))
                }

                SMTExpr::IntLit(val) => {
                    let int = Int::from_i64(&self.ctx, *val);
                    Ok(Dynamic::from_ast(&int))
                }

                SMTExpr::BoolLit(val) => {
                    let bool = Bool::from_bool(&self.ctx, *val);
                    Ok(Dynamic::from_ast(&bool))
                }

                SMTExpr::BinOp { op, lhs, rhs } => {
                    let lhs_ast = self.translate_expr(lhs)?;
                    let rhs_ast = self.translate_expr(rhs)?;

                    if let (Some(lhs_real), Some(rhs_real)) = (lhs_ast.as_real(), rhs_ast.as_real())
                    {
                        let result = match op.as_str() {
                            "+" => lhs_real + rhs_real,
                            "-" => lhs_real - rhs_real,
                            "*" => lhs_real * rhs_real,
                            "/" => lhs_real / rhs_real,
                            _ => return Err(anyhow!("Unsupported binary operator: {}", op)),
                        };
                        Ok(Dynamic::from_ast(&result))
                    } else if let (Some(lhs_int), Some(rhs_int)) =
                        (lhs_ast.as_int(), rhs_ast.as_int())
                    {
                        let result = match op.as_str() {
                            "+" => lhs_int + rhs_int,
                            "-" => lhs_int - rhs_int,
                            "*" => lhs_int * rhs_int,
                            "/" => lhs_int / rhs_int,
                            _ => return Err(anyhow!("Unsupported binary operator: {}", op)),
                        };
                        Ok(Dynamic::from_ast(&result))
                    } else {
                        Err(anyhow!("Type mismatch in binary operation"))
                    }
                }

                SMTExpr::UnOp { op, expr } => {
                    let expr_ast = self.translate_expr(expr)?;

                    if let Some(real) = expr_ast.as_real() {
                        let result = match op.as_str() {
                            "-" => -real,
                            _ => return Err(anyhow!("Unsupported unary operator: {}", op)),
                        };
                        Ok(Dynamic::from_ast(&result))
                    } else if let Some(int) = expr_ast.as_int() {
                        let result = match op.as_str() {
                            "-" => -int,
                            _ => return Err(anyhow!("Unsupported unary operator: {}", op)),
                        };
                        Ok(Dynamic::from_ast(&result))
                    } else {
                        Err(anyhow!("Type mismatch in unary operation"))
                    }
                }
            }
        }

        fn extract_model(&self, model: &Model) -> Z3Model {
            let mut assignments = HashMap::new();

            for (name, var) in &self.var_map {
                if let Some(value) = model.eval(var, true) {
                    if let Some(real) = value.as_real() {
                        if let Some((num, den)) = real.as_real() {
                            let val = num as f64 / den as f64;
                            assignments.insert(name.clone(), Z3Value::Real(val));
                        }
                    } else if let Some(int) = value.as_int() {
                        if let Some(val) = int.as_i64() {
                            assignments.insert(name.clone(), Z3Value::Int(val));
                        }
                    } else if let Some(bool_val) = value.as_bool() {
                        assignments
                            .insert(name.clone(), Z3Value::Bool(bool_val.as_bool().unwrap()));
                    }
                }
            }

            Z3Model { assignments }
        }
    }

    impl Default for Z3Solver {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(feature = "smt-verification")]
pub use z3_impl::Z3Solver;

// ====================================================================================
// Mock implementation WITHOUT Z3 support (default, for development)
// ====================================================================================

#[cfg(not(feature = "smt-verification"))]
pub struct Z3Solver {
    _dummy: (),
}

#[cfg(not(feature = "smt-verification"))]
impl Z3Solver {
    pub fn new() -> Self {
        Z3Solver { _dummy: () }
    }

    pub fn assert(&mut self, _formula: SMTFormula) {
        // No-op in mock mode
    }

    pub fn check_vc(&mut self, vc: &VerificationCondition) -> Z3Result {
        // In mock mode, always return Unknown
        eprintln!("⚠ SMT verification disabled (compile with --features smt-verification)");
        eprintln!("  Would verify: {}", vc.description);
        Z3Result::Unknown
    }

    pub fn reset(&mut self) {
        // No-op in mock mode
    }
}

#[cfg(not(feature = "smt-verification"))]
impl Default for Z3Solver {
    fn default() -> Self {
        Self::new()
    }
}

// ====================================================================================
// Tests (only when Z3 is available)
// ====================================================================================

#[cfg(all(test, feature = "smt-verification"))]
mod tests {
    use super::*;
    use crate::ast::phase_v1::ConstraintLiteral;

    #[test]
    fn test_simple_sat() {
        let mut solver = Z3Solver::new();

        let formula = SMTFormula::Comparison {
            lhs: SMTExpr::Var("x".to_string()),
            op: ComparisonOp::Gt,
            rhs: SMTExpr::RealLit(0.0),
        };

        solver.assert(formula);

        let vc = VerificationCondition {
            assumptions: vec![SMTFormula::Comparison {
                lhs: SMTExpr::Var("x".to_string()),
                op: ComparisonOp::Gt,
                rhs: SMTExpr::RealLit(0.0),
            }],
            goal: SMTFormula::Comparison {
                lhs: SMTExpr::Var("x".to_string()),
                op: ComparisonOp::Ge,
                rhs: SMTExpr::RealLit(0.0),
            },
            location: None,
            description: "x > 0 implies x >= 0".to_string(),
        };

        match solver.check_vc(&vc) {
            Z3Result::Proven => {
                // Success!
            }
            other => panic!("Expected Proven, got {:?}", other),
        }
    }

    #[test]
    fn test_counterexample() {
        let mut solver = Z3Solver::new();

        let vc = VerificationCondition {
            assumptions: vec![SMTFormula::Comparison {
                lhs: SMTExpr::Var("x".to_string()),
                op: ComparisonOp::Gt,
                rhs: SMTExpr::RealLit(0.0),
            }],
            goal: SMTFormula::Comparison {
                lhs: SMTExpr::Var("x".to_string()),
                op: ComparisonOp::Gt,
                rhs: SMTExpr::RealLit(10.0),
            },
            location: None,
            description: "x > 0 implies x > 10 (FALSE)".to_string(),
        };

        match solver.check_vc(&vc) {
            Z3Result::Counterexample(_model) => {
                // Expected
            }
            other => panic!("Expected Counterexample, got {:?}", other),
        }
    }
}
