//! AST → IR lowering pass
//!
//! This module transforms the high-level AST into a simplified IR suitable
//! for code generation. It performs:
//! - Name resolution
//! - Scope flattening
//! - Type erasure (keeping only dimension info)
//! - Expression simplification

pub mod evidence;

use crate::ast::*;
use crate::ir::*;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LowerError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Population not found: {0}")]
    PopulationNotFound(String),

    #[error("Measure not found: {0}")]
    MeasureNotFound(String),

    #[error("Multiple models defined (V0 supports only one)")]
    MultipleModels,

    #[error("Multiple populations defined (V0 supports only one)")]
    MultiplePopulations,

    #[error("No model defined")]
    NoModel,

    #[error("No population defined")]
    NoPopulation,

    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),
}

/// Context for lowering AST to IR
struct LowerContext {
    /// Model definitions by name
    models: HashMap<String, ModelDef>,

    /// Population definitions by name
    populations: HashMap<String, PopulationDef>,

    /// Measure definitions by name
    measures: HashMap<String, MeasureDef>,
}

impl LowerContext {
    fn new() -> Self {
        Self {
            models: HashMap::new(),
            populations: HashMap::new(),
            measures: HashMap::new(),
        }
    }

    fn register_declarations(&mut self, program: &Program) {
        for decl in &program.declarations {
            match decl {
                Declaration::Model(m) => {
                    self.models.insert(m.name.clone(), m.clone());
                }
                Declaration::Population(p) => {
                    self.populations.insert(p.name.clone(), p.clone());
                }
                Declaration::Measure(m) => {
                    self.measures.insert(m.name.clone(), m.clone());
                }
                _ => {}
            }
        }
    }
}

/// Lower an AST program to IR
pub fn lower_program(program: &Program) -> Result<IRProgram, LowerError> {
    let mut ctx = LowerContext::new();
    ctx.register_declarations(program);

    // V0: Expect at least one model and exactly one population
    if ctx.models.is_empty() {
        return Err(LowerError::NoModel);
    }
    if ctx.populations.is_empty() {
        return Err(LowerError::NoPopulation);
    }
    if ctx.populations.len() > 1 {
        return Err(LowerError::MultiplePopulations);
    }

    // Get the population (there's only one)
    let pop_def = ctx.populations.values().next().unwrap();

    // Find the model referenced by the population
    let model_name = extract_model_ref(pop_def)?;
    let model_def = ctx
        .models
        .get(&model_name)
        .ok_or_else(|| LowerError::ModelNotFound(model_name.clone()))?;

    // Find the measure referenced in the population
    let measure_name = extract_measure_name(pop_def)?;
    let measure_def = ctx
        .measures
        .get(&measure_name)
        .ok_or_else(|| LowerError::MeasureNotFound(measure_name.clone()))?;

    let ir_model = lower_model(&ctx, model_def, pop_def)?;
    let ir_measure = lower_measure(measure_def)?;
    let ir_data_spec = create_data_spec();

    Ok(IRProgram {
        model: ir_model,
        measures: vec![ir_measure], // For now, single measure; will support multiple in Phase 2
        data_spec: ir_data_spec,
        externals: vec![], // No QM stub by default; set via lower_program_with_qm
    })
}

/// Lower an AST program to IR with quantum stub integration
///
/// This extends `lower_program` by injecting external quantum constants
/// from a QM stub (e.g., Kd_QM, Kp_tumor_QM) into the IR.
pub fn lower_program_with_qm(
    program: &Program,
    qm_stub: Option<&crate::qm_stub::QuantumStub>,
) -> Result<IRProgram, LowerError> {
    let mut ir_program = lower_program(program)?;

    if let Some(stub) = qm_stub {
        // Add Kd_QM as external constant
        ir_program.externals.push(IRExternalScalar {
            name: "Kd_QM".to_string(),
            value: stub.Kd_M,
            source: format!("qm_stub:{}:{}", stub.drug_id, stub.target_id),
            dimension: Some("ConcMass".to_string()), // Kd has concentration units
        });

        // Add Kp_tumor_QM if partition data is available
        if let Some(kp) = stub.kp_tumor_from_dg() {
            ir_program.externals.push(IRExternalScalar {
                name: "Kp_tumor_QM".to_string(),
                value: kp,
                source: format!("qm_stub:{}:dG_part", stub.drug_id),
                dimension: None, // Kp is dimensionless (ratio of concentrations)
            });
        }
    }

    Ok(ir_program)
}

fn extract_model_ref(pop: &PopulationDef) -> Result<String, LowerError> {
    for item in &pop.items {
        if let PopulationItem::ModelRef(model_name) = item {
            return Ok(model_name.clone());
        }
    }
    Err(LowerError::UnsupportedFeature(
        "Population must reference a model".to_string(),
    ))
}

fn extract_measure_name(pop: &PopulationDef) -> Result<String, LowerError> {
    for item in &pop.items {
        if let PopulationItem::UseMeasure(use_stmt) = item {
            return Ok(use_stmt.measure_name.clone());
        }
    }
    Err(LowerError::UnsupportedFeature(
        "Population must use a measure".to_string(),
    ))
}

/// Substitute input variables with their connected sources in IR expressions
///
/// This function walks an IR expression tree and replaces variable references
/// that match input declarations with the connected field from another submodel.
///
/// Example: If input_map contains {"QSP.C_drug" -> ("PBPK", "C_tumor")},
/// then any reference to "C_drug" in the QSP submodel's expressions will be
/// replaced with "C_tumor" (the field from PBPK submodel).
fn substitute_inputs(
    expr: &IRExpr,
    input_map: &HashMap<String, (String, String)>,
    observable_expr_map: &HashMap<String, IRExpr>,
) -> IRExpr {
    match expr {
        IRExpr::Var(name) => {
            // Check if this variable is an input that should be substituted
            for (input_key, (_from_model, from_field)) in input_map {
                // input_key format: "SubmodelName.input_field_name"
                // Extract just the field name and compare
                if let Some(field_name) = input_key.split('.').nth(1) {
                    if field_name == name {
                        // Found a match - check if the connected field is an observable
                        if let Some(obs_expr) = observable_expr_map.get(from_field) {
                            // The connected field is an observable - inline its expression
                            return obs_expr.clone();
                        } else {
                            // Regular variable - substitute with the connected field
                            return IRExpr::Var(from_field.clone());
                        }
                    }
                }
            }
            // No substitution needed
            IRExpr::Var(name.clone())
        }
        IRExpr::Literal(val) => IRExpr::Literal(*val),
        IRExpr::Unary(op, operand) => IRExpr::Unary(
            *op,
            Box::new(substitute_inputs(operand, input_map, observable_expr_map)),
        ),
        IRExpr::Binary(op, lhs, rhs) => IRExpr::Binary(
            *op,
            Box::new(substitute_inputs(lhs, input_map, observable_expr_map)),
            Box::new(substitute_inputs(rhs, input_map, observable_expr_map)),
        ),
        IRExpr::Index(array, index) => IRExpr::Index(
            Box::new(substitute_inputs(array, input_map, observable_expr_map)),
            Box::new(substitute_inputs(index, input_map, observable_expr_map)),
        ),
        IRExpr::Call(func_name, args) => {
            let substituted_args: Vec<_> = args
                .iter()
                .map(|arg| substitute_inputs(arg, input_map, observable_expr_map))
                .collect();
            IRExpr::Call(func_name.clone(), substituted_args)
        }
    }
}

/// Flatten a composite model (one with submodels) into a single unified model
fn lower_composite_model(
    ctx: &LowerContext,
    composite: &ModelDef,
    pop: &PopulationDef,
) -> Result<IRModel, LowerError> {
    // Step 1: Collect submodel declarations and connections
    let mut submodel_decls = Vec::new();
    let mut connections = Vec::new();

    for item in &composite.items {
        match item {
            ModelItem::Submodel(s) => submodel_decls.push(s.clone()),
            ModelItem::Connect(c) => connections.push(c.clone()),
            _ => {}
        }
    }

    if submodel_decls.is_empty() {
        return Err(LowerError::UnsupportedFeature(
            "Composite model has no submodels".to_string(),
        ));
    }

    // Step 2: Flatten all submodels
    let mut all_states = Vec::new();
    let mut all_params = Vec::new();
    let mut all_intermediates = Vec::new();
    let mut all_odes = Vec::new();
    let mut all_observables = Vec::new();
    let mut input_map: HashMap<String, (String, String)> = HashMap::new(); // (to_model.to_field) -> (from_model, from_field)
    let mut observable_expr_map: HashMap<String, IRExpr> = HashMap::new(); // observable_name -> its expression

    for submodel_decl in &submodel_decls {
        let submodel_def = ctx
            .models
            .get(&submodel_decl.model_type)
            .ok_or_else(|| LowerError::ModelNotFound(submodel_decl.model_type.clone()))?;

        // Process each submodel item with name prefixing
        for item in &submodel_def.items {
            match item {
                ModelItem::State(s) => {
                    all_states.push(IRStateVar {
                        name: s.name.clone(), // Keep original name (will be in separate namespace per submodel in IR)
                        dimension: type_expr_to_dimension(&s.ty),
                        initial_value: None,
                    });
                }
                ModelItem::Param(p) => {
                    all_params.push(IRParam {
                        name: p.name.clone(),
                        dimension: type_expr_to_dimension(&p.ty),
                        kind: ParamKind::Fixed,
                    });
                }
                ModelItem::Let(let_binding) => {
                    all_intermediates.push(IRIntermediate {
                        name: let_binding.name.clone(),
                        dimension: let_binding.ty.as_ref().map(type_expr_to_dimension),
                        expr: lower_expr(&let_binding.expr),
                    });
                }
                ModelItem::ODE(ode) => {
                    all_odes.push(IRODEEquation {
                        state_var: ode.state_name.clone(),
                        rhs: lower_expr(&ode.rhs),
                    });
                }
                ModelItem::Observable(obs) => {
                    let obs_expr = lower_expr(&obs.expr);
                    observable_expr_map.insert(obs.name.clone(), obs_expr.clone());
                    all_observables.push(IRObservable {
                        name: obs.name.clone(),
                        dimension: type_expr_to_dimension(&obs.ty),
                        expr: obs_expr,
                    });
                }
                ModelItem::Input(inp) => {
                    // Record input for connection resolution
                    input_map.insert(
                        format!("{}.{}", submodel_decl.name, inp.name),
                        (String::new(), String::new()),
                    );
                }
                _ => {}
            }
        }
    }

    // Step 3: Apply connections
    for conn in &connections {
        let to_key = format!("{}.{}", conn.to_model, conn.to_field);
        if let Some(entry) = input_map.get_mut(&to_key) {
            *entry = (conn.from_model.clone(), conn.from_field.clone());
        }
    }

    // Step 4: Substitute input references in ODEs and intermediates
    for ode in &mut all_odes {
        ode.rhs = substitute_inputs(&ode.rhs, &input_map, &observable_expr_map);
    }

    for intermediate in &mut all_intermediates {
        intermediate.expr = substitute_inputs(&intermediate.expr, &input_map, &observable_expr_map);
    }

    for observable in &mut all_observables {
        observable.expr = substitute_inputs(&observable.expr, &input_map, &observable_expr_map);
    }

    // Step 5: Extract population parameters
    let mut inputs = Vec::new();
    let mut random_effects = Vec::new();
    let mut individual_params = Vec::new();

    for item in &pop.items {
        match item {
            PopulationItem::Param(p) => {
                all_params.push(IRParam {
                    name: p.name.clone(),
                    dimension: type_expr_to_dimension(&p.ty),
                    kind: if p.name.starts_with("omega_") {
                        ParamKind::PopulationVariance
                    } else {
                        ParamKind::PopulationMean
                    },
                });
            }
            PopulationItem::Input(inp) => {
                inputs.push(IRInput {
                    name: inp.name.clone(),
                    dimension: type_expr_to_dimension(&inp.ty),
                });
            }
            PopulationItem::RandomEffect(re) => {
                let dist = match &re.dist {
                    DistributionExpr::Normal { mu, sigma } => IRDistribution::Normal {
                        mu: lower_expr(mu),
                        sigma: lower_expr(sigma),
                    },
                    DistributionExpr::LogNormal { mu, sigma } => IRDistribution::LogNormal {
                        mu: lower_expr(mu),
                        sigma: lower_expr(sigma),
                    },
                    DistributionExpr::Uniform { .. } => {
                        return Err(LowerError::UnsupportedFeature(
                            "Uniform distribution not supported in V0".to_string(),
                        ))
                    }
                };
                random_effects.push(IRRandomEffect {
                    name: re.name.clone(),
                    distribution: dist,
                });
            }
            PopulationItem::BindParams(bind) => {
                individual_params = extract_individual_params(bind)?;
            }
            _ => {}
        }
    }

    Ok(IRModel {
        name: composite.name.clone(),
        states: all_states,
        params: all_params,
        inputs,
        intermediates: all_intermediates,
        odes: all_odes,
        observables: all_observables,
        random_effects,
        individual_params,
    })
}

fn lower_model(
    ctx: &LowerContext,
    model: &ModelDef,
    pop: &PopulationDef,
) -> Result<IRModel, LowerError> {
    // Check if this is a composite model (has submodels)
    let has_submodels = model
        .items
        .iter()
        .any(|item| matches!(item, ModelItem::Submodel(_)));

    if has_submodels {
        // This is a composite model - need to flatten it
        return lower_composite_model(ctx, model, pop);
    }

    // Simple model - process directly
    let mut states = Vec::new();
    let mut params = Vec::new();
    let mut intermediates = Vec::new();
    let mut odes = Vec::new();
    let mut observables = Vec::new();

    // Extract structural model components
    for item in &model.items {
        match item {
            ModelItem::State(s) => {
                states.push(IRStateVar {
                    name: s.name.clone(),
                    dimension: type_expr_to_dimension(&s.ty),
                    initial_value: None,
                });
            }
            ModelItem::Param(p) => {
                params.push(IRParam {
                    name: p.name.clone(),
                    dimension: type_expr_to_dimension(&p.ty),
                    kind: ParamKind::Fixed,
                });
            }
            ModelItem::ODE(ode) => {
                odes.push(IRODEEquation {
                    state_var: ode.state_name.clone(),
                    rhs: lower_expr(&ode.rhs),
                });
            }
            ModelItem::Observable(obs) => {
                observables.push(IRObservable {
                    name: obs.name.clone(),
                    dimension: type_expr_to_dimension(&obs.ty),
                    expr: lower_expr(&obs.expr),
                });
            }
            ModelItem::Input(_) => {
                // Inputs are handled in composite model flattening
                // For now, skip them in simple models
            }
            ModelItem::Let(let_binding) => {
                intermediates.push(IRIntermediate {
                    name: let_binding.name.clone(),
                    dimension: let_binding.ty.as_ref().map(type_expr_to_dimension),
                    expr: lower_expr(&let_binding.expr),
                });
            }
            ModelItem::Submodel(_) => {
                // Should not reach here (checked above)
            }
            ModelItem::Connect(_) => {
                // Should not reach here (checked above)
            }
        }
    }

    // Extract population-level parameters and random effects
    let mut inputs = Vec::new();
    let mut random_effects = Vec::new();
    let mut individual_params = Vec::new();

    for item in &pop.items {
        match item {
            PopulationItem::Param(p) => {
                params.push(IRParam {
                    name: p.name.clone(),
                    dimension: type_expr_to_dimension(&p.ty),
                    kind: if p.name.starts_with("omega_") {
                        ParamKind::PopulationVariance
                    } else {
                        ParamKind::PopulationMean
                    },
                });
            }
            PopulationItem::Input(inp) => {
                inputs.push(IRInput {
                    name: inp.name.clone(),
                    dimension: type_expr_to_dimension(&inp.ty),
                });
            }
            PopulationItem::RandomEffect(re) => {
                let dist = match &re.dist {
                    DistributionExpr::Normal { mu, sigma } => IRDistribution::Normal {
                        mu: lower_expr(mu),
                        sigma: lower_expr(sigma),
                    },
                    DistributionExpr::LogNormal { mu, sigma } => IRDistribution::LogNormal {
                        mu: lower_expr(mu),
                        sigma: lower_expr(sigma),
                    },
                    DistributionExpr::Uniform { .. } => {
                        return Err(LowerError::UnsupportedFeature(
                            "Uniform distribution not supported in V0".to_string(),
                        ))
                    }
                };

                random_effects.push(IRRandomEffect {
                    name: re.name.clone(),
                    distribution: dist,
                });
            }
            PopulationItem::BindParams(bind) => {
                individual_params = extract_individual_params(bind)?;
            }
            _ => {}
        }
    }

    Ok(IRModel {
        name: model.name.clone(),
        states,
        params,
        inputs,
        random_effects,
        intermediates,
        odes,
        observables,
        individual_params,
    })
}

fn extract_individual_params(bind: &BindParamsBlock) -> Result<Vec<IRIndividualParam>, LowerError> {
    let mut result = Vec::new();

    for stmt in &bind.statements {
        if let Statement::Assign { target, value, .. } = stmt {
            // Extract the parameter name from qualified name (e.g., model.CL → CL)
            let param_name = if target.parts.len() == 2 && target.parts[0] == "model" {
                target.parts[1].clone()
            } else {
                target.to_string()
            };

            result.push(IRIndividualParam {
                param_name,
                expr: lower_expr(value),
            });
        }
    }

    Ok(result)
}

fn lower_measure(measure: &MeasureDef) -> Result<IRMeasure, LowerError> {
    let mut params = Vec::new();
    let mut log_likelihood = None;

    for item in &measure.items {
        match item {
            MeasureItem::Param(p) => {
                params.push(IRParam {
                    name: p.name.clone(),
                    dimension: type_expr_to_dimension(&p.ty),
                    kind: ParamKind::PopulationMean,
                });
            }
            MeasureItem::LogLikelihood(expr) => {
                log_likelihood = Some(lower_expr(expr));
            }
            _ => {}
        }
    }

    Ok(IRMeasure {
        name: measure.name.clone(),
        observable_ref: "C_plasma".to_string(), // TODO: extract from use_measure
        params,
        log_likelihood: log_likelihood.ok_or_else(|| {
            LowerError::UnsupportedFeature("Measure must have log_likelihood".to_string())
        })?,
    })
}

fn create_data_spec() -> IRDataSpec {
    let mut columns = HashMap::new();
    columns.insert("ID".to_string(), "subject_id".to_string());
    columns.insert("TIME".to_string(), "time".to_string());
    columns.insert("DV".to_string(), "observation".to_string());
    columns.insert("WT".to_string(), "weight".to_string());

    IRDataSpec {
        n_subjects: "N".to_string(),
        n_obs: "n_obs".to_string(),
        columns,
    }
}

fn type_expr_to_dimension(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Simple(name) => name.clone(),
        TypeExpr::Unit(ut) => format!("{:?}", ut),
        TypeExpr::Quantity(_, _) => "Quantity".to_string(),
    }
}

fn lower_expr(expr: &Expr) -> IRExpr {
    match &expr.kind {
        ExprKind::Literal(lit) => lower_literal(lit),
        ExprKind::Ident(name) => IRExpr::var(name),
        ExprKind::QualifiedName(qn) => {
            // For V0, convert qualified names to simple variable names
            // e.g., patient.WT → WT, model.CL → CL
            if qn.parts.len() == 2 {
                IRExpr::var(&qn.parts[1])
            } else {
                IRExpr::var(&qn.parts.join("_"))
            }
        }
        ExprKind::Unary(UnaryOp::Neg, operand) => IRExpr::neg(lower_expr(operand)),
        ExprKind::Unary(UnaryOp::Pos, operand) => lower_expr(operand),
        ExprKind::Binary(op, left, right) => {
            let ir_op = match op {
                BinaryOp::Add => IRBinaryOp::Add,
                BinaryOp::Sub => IRBinaryOp::Sub,
                BinaryOp::Mul => IRBinaryOp::Mul,
                BinaryOp::Div => IRBinaryOp::Div,
                BinaryOp::Pow => IRBinaryOp::Pow,
                _ => IRBinaryOp::Add, // TODO: handle comparisons
            };
            IRExpr::binary(ir_op, lower_expr(left), lower_expr(right))
        }
        ExprKind::Call(name, args) => {
            let ir_args: Vec<_> = args.iter().map(|arg| lower_expr(&arg.value)).collect();
            IRExpr::call(name, ir_args)
        }
    }
}

fn lower_literal(lit: &Literal) -> IRExpr {
    match lit {
        Literal::Float(val) => IRExpr::literal(*val),
        Literal::UnitFloat { value, .. } => {
            // For V0, strip units and just use the numeric value
            IRExpr::literal(*value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::parse_program;

    #[test]
    fn test_lower_simple_expr() {
        let ast_expr = Expr::binary(
            BinaryOp::Mul,
            Expr::ident("K".to_string()),
            Expr::ident("A".to_string()),
        );

        let ir_expr = lower_expr(&ast_expr);

        match ir_expr {
            IRExpr::Binary(IRBinaryOp::Mul, left, right) => {
                assert!(matches!(*left, IRExpr::Var(_)));
                assert!(matches!(*right, IRExpr::Var(_)));
            }
            _ => panic!("Expected binary multiplication"),
        }
    }

    #[test]
    fn test_lower_literal() {
        let ast_lit = Literal::Float(100.0);
        let ir_expr = lower_literal(&ast_lit);

        match ir_expr {
            IRExpr::Literal(val) => assert_eq!(val, 100.0),
            _ => panic!("Expected literal"),
        }
    }

    #[test]
    fn test_lower_unit_literal() {
        let ast_lit = Literal::UnitFloat {
            value: 70.0,
            unit: "kg".to_string(),
        };
        let ir_expr = lower_literal(&ast_lit);

        match ir_expr {
            IRExpr::Literal(val) => assert_eq!(val, 70.0),
            _ => panic!("Expected literal"),
        }
    }

    #[test]
    fn test_lower_simple_model() {
        let source = r#"
model TestModel {
    state A : DoseMass
    param K : RateConst
    dA/dt = -K * A
}

population TestPop {
    model TestModel
    param K_pop : RateConst
    rand eta_K : f64 ~ Normal(0.0, 0.3)
    bind_params(patient) {
        model.K = K_pop * exp(eta_K)
    }
    use_measure TestMeasure for model.A
}

measure TestMeasure {
    pred : DoseMass
    obs : DoseMass
    param sigma : f64
    log_likelihood = normal_lpdf(obs, pred, sigma)
}
        "#;

        let tokens = tokenize(source).unwrap();
        let program = parse_program(&tokens).unwrap();

        let ir = lower_program(&program);
        assert!(ir.is_ok(), "Lowering failed: {:?}", ir.err());

        let ir = ir.unwrap();
        assert_eq!(ir.model.name, "TestModel");
        assert_eq!(ir.model.states.len(), 1);
        assert_eq!(ir.model.odes.len(), 1);
    }
}
