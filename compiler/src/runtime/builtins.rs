// Week 29: Built-in Functions for L₀
//
// Implements runtime execution of built-in functions, particularly
// the Week 29 surrogate model functions and Week 31-32 RL built-ins.

use super::value::{RuntimeError, RuntimeValue};
use crate::ml::{BackendKind, SurrogateModelHandle, SurrogateTrainConfig};
use crate::rl::{BoxDiscretizer, RLPolicyHandle, RLTrainConfig}; // Week 31-32: RL imports
use std::collections::HashMap;

/// Enumeration of available built-in functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinFn {
    // Original built-ins
    RunEvidence,
    ExportResults,
    Print,
    RunSimulation,
    FitModel,

    // Week 29: Surrogate built-ins
    TrainSurrogate,
    RunEvidenceTyped,
    RunEvidenceWithSurrogate,

    // Week 30: Surrogate evaluation
    EvaluateSurrogate,

    // Week 31-32: RL policy training
    TrainPolicyRL,
    SimulatePolicyRL,
}

impl BuiltinFn {
    /// Get the function name as a string
    pub fn name(&self) -> &'static str {
        match self {
            BuiltinFn::RunEvidence => "run_evidence",
            BuiltinFn::ExportResults => "export_results",
            BuiltinFn::Print => "print",
            BuiltinFn::RunSimulation => "run_simulation",
            BuiltinFn::FitModel => "fit_model",
            BuiltinFn::TrainSurrogate => "train_surrogate",
            BuiltinFn::RunEvidenceTyped => "run_evidence_typed",
            BuiltinFn::RunEvidenceWithSurrogate => "run_evidence_with_surrogate",
            BuiltinFn::EvaluateSurrogate => "evaluate_surrogate",
            BuiltinFn::TrainPolicyRL => "train_policy_rl",
            BuiltinFn::SimulatePolicyRL => "simulate_policy_rl",
        }
    }

    /// Parse a function name into a BuiltinFn
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "run_evidence" => Some(BuiltinFn::RunEvidence),
            "export_results" => Some(BuiltinFn::ExportResults),
            "print" => Some(BuiltinFn::Print),
            "run_simulation" => Some(BuiltinFn::RunSimulation),
            "fit_model" => Some(BuiltinFn::FitModel),
            "train_surrogate" => Some(BuiltinFn::TrainSurrogate),
            "run_evidence_typed" => Some(BuiltinFn::RunEvidenceTyped),
            "run_evidence_with_surrogate" => Some(BuiltinFn::RunEvidenceWithSurrogate),
            "evaluate_surrogate" => Some(BuiltinFn::EvaluateSurrogate),
            "train_policy_rl" => Some(BuiltinFn::TrainPolicyRL),
            "simulate_policy_rl" => Some(BuiltinFn::SimulatePolicyRL),
            _ => None,
        }
    }

    /// Get expected argument count
    pub fn arity(&self) -> usize {
        match self {
            BuiltinFn::Print => 1,
            BuiltinFn::RunEvidence => 2,
            BuiltinFn::ExportResults => 2,
            BuiltinFn::RunSimulation => 2,
            BuiltinFn::FitModel => 2,
            BuiltinFn::TrainSurrogate => 2,
            BuiltinFn::RunEvidenceTyped => 2,
            BuiltinFn::RunEvidenceWithSurrogate => 3,
            BuiltinFn::EvaluateSurrogate => 3,
            BuiltinFn::TrainPolicyRL => 2,
            BuiltinFn::SimulatePolicyRL => 3,
        }
    }
}

/// Call a built-in function with the given arguments
pub fn call_builtin(
    func: BuiltinFn,
    args: Vec<RuntimeValue>,
) -> Result<RuntimeValue, RuntimeError> {
    // Check arity
    if args.len() != func.arity() {
        return Err(RuntimeError::ArityMismatch {
            fn_name: func.name().to_string(),
            expected: func.arity(),
            found: args.len(),
        });
    }

    match func {
        BuiltinFn::Print => builtin_print(args),
        BuiltinFn::RunEvidence => builtin_run_evidence(args),
        BuiltinFn::ExportResults => builtin_export_results(args),
        BuiltinFn::RunSimulation => builtin_run_simulation(args),
        BuiltinFn::FitModel => builtin_fit_model(args),
        BuiltinFn::TrainSurrogate => builtin_train_surrogate(args),
        BuiltinFn::RunEvidenceTyped => builtin_run_evidence_typed(args),
        BuiltinFn::RunEvidenceWithSurrogate => builtin_run_evidence_with_surrogate(args),
        BuiltinFn::EvaluateSurrogate => builtin_evaluate_surrogate(args),
        BuiltinFn::TrainPolicyRL => builtin_train_policy_rl(args),
        BuiltinFn::SimulatePolicyRL => builtin_simulate_policy_rl(args),
    }
}

// =============================================================================
// Original Built-in Functions (Stubs)
// =============================================================================

fn builtin_print(args: Vec<RuntimeValue>) -> Result<RuntimeValue, RuntimeError> {
    let msg = &args[0];
    match msg {
        RuntimeValue::String(s) => {
            println!("{}", s);
            Ok(RuntimeValue::Unit)
        }
        _ => Err(RuntimeError::TypeError {
            expected: "String".to_string(),
            found: msg.runtime_type(),
            message: "print() requires a String argument".to_string(),
        }),
    }
}

fn builtin_run_evidence(args: Vec<RuntimeValue>) -> Result<RuntimeValue, RuntimeError> {
    let _ev = &args[0];
    let backend_str = &args[1];

    match backend_str {
        RuntimeValue::String(backend) => {
            // TODO: Integrate with actual evidence runner
            // For now, return a stub result
            Ok(RuntimeValue::EvidenceResult {
                posterior_samples: vec![],
                diagnostics: HashMap::from([
                    ("backend".to_string(), backend.clone()),
                    ("status".to_string(), "stub".to_string()),
                ]),
            })
        }
        _ => Err(RuntimeError::TypeError {
            expected: "String".to_string(),
            found: backend_str.runtime_type(),
            message: "run_evidence() backend must be a String".to_string(),
        }),
    }
}

fn builtin_export_results(args: Vec<RuntimeValue>) -> Result<RuntimeValue, RuntimeError> {
    let _result = &args[0];
    let path = &args[1];

    match path {
        RuntimeValue::String(_path_str) => {
            // TODO: Implement result export
            Ok(RuntimeValue::Unit)
        }
        _ => Err(RuntimeError::TypeError {
            expected: "String".to_string(),
            found: path.runtime_type(),
            message: "export_results() path must be a String".to_string(),
        }),
    }
}

fn builtin_run_simulation(args: Vec<RuntimeValue>) -> Result<RuntimeValue, RuntimeError> {
    let _protocol = &args[0];
    let _n_subjects = &args[1];

    // TODO: Integrate with simulation engine
    Ok(RuntimeValue::SimulationResult {
        trajectories: vec![],
        summary: HashMap::new(),
    })
}

fn builtin_fit_model(args: Vec<RuntimeValue>) -> Result<RuntimeValue, RuntimeError> {
    let _model = &args[0];
    let _data_path = &args[1];

    // TODO: Integrate with Stan/Julia backend
    Ok(RuntimeValue::FitResult {
        parameters: HashMap::new(),
        diagnostics: HashMap::new(),
    })
}

// =============================================================================
// Week 29: Surrogate Built-in Functions
// =============================================================================

/// train_surrogate(ev: EvidenceProgram, cfg: Record) -> SurrogateModel
fn builtin_train_surrogate(args: Vec<RuntimeValue>) -> Result<RuntimeValue, RuntimeError> {
    let ev = &args[0];
    let cfg_record = &args[1];

    // Validate evidence program
    match ev {
        RuntimeValue::EvidenceProgram { .. } => {}
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "EvidenceProgram".to_string(),
                found: ev.runtime_type(),
                message: "train_surrogate() requires an EvidenceProgram".to_string(),
            })
        }
    }

    // Extract configuration from record
    let cfg = match cfg_record {
        RuntimeValue::Record(fields) => {
            // Extract fields
            let n_train = match fields.get("n_train") {
                Some(RuntimeValue::Int(n)) => *n,
                _ => {
                    return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "missing or wrong type".to_string(),
                        message: "n_train field must be an Int".to_string(),
                    })
                }
            };

            let backend = match fields.get("backend") {
                Some(val) => val.as_backend_kind()?,
                None => {
                    return Err(RuntimeError::FieldNotFound("backend".to_string()));
                }
            };

            let seed = match fields.get("seed") {
                Some(RuntimeValue::Int(s)) => *s,
                _ => {
                    return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "missing or wrong type".to_string(),
                        message: "seed field must be an Int".to_string(),
                    })
                }
            };

            let max_epochs = match fields.get("max_epochs") {
                Some(RuntimeValue::Int(e)) => *e,
                _ => {
                    return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "missing or wrong type".to_string(),
                        message: "max_epochs field must be an Int".to_string(),
                    })
                }
            };

            let batch_size = match fields.get("batch_size") {
                Some(RuntimeValue::Int(b)) => *b,
                _ => {
                    return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "missing or wrong type".to_string(),
                        message: "batch_size field must be an Int".to_string(),
                    })
                }
            };

            SurrogateTrainConfig {
                n_train,
                backend,
                seed,
                max_epochs,
                batch_size,
            }
        }
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "Record".to_string(),
                found: cfg_record.runtime_type(),
                message: "train_surrogate() config must be a Record".to_string(),
            })
        }
    };

    // Validate configuration
    cfg.validate()
        .map_err(|e| RuntimeError::SurrogateError(e.to_string()))?;

    // TODO: Implement actual surrogate training
    // For now, create a new handle
    let handle = SurrogateModelHandle::with_name(format!(
        "surrogate_n{}_e{}_seed{}",
        cfg.n_train, cfg.max_epochs, cfg.seed
    ));

    Ok(RuntimeValue::SurrogateModel(handle))
}

/// run_evidence_typed(ev: EvidenceProgram, backend: BackendKind) -> EvidenceResult
fn builtin_run_evidence_typed(args: Vec<RuntimeValue>) -> Result<RuntimeValue, RuntimeError> {
    let ev = &args[0];
    let backend_val = &args[1];

    // Validate evidence program
    match ev {
        RuntimeValue::EvidenceProgram { name, .. } => {
            // Extract backend
            let backend = backend_val.as_backend_kind()?;

            // Check backend requirements
            if backend.requires_surrogate() {
                return Err(RuntimeError::BackendError(
                    "Backend requires a surrogate model. Use run_evidence_with_surrogate() instead."
                        .to_string(),
                ));
            }

            // TODO: Integrate with actual evidence runner
            Ok(RuntimeValue::EvidenceResult {
                posterior_samples: vec![],
                diagnostics: HashMap::from([
                    ("evidence_program".to_string(), name.clone()),
                    ("backend".to_string(), backend.variant_name().to_string()),
                    ("status".to_string(), "stub".to_string()),
                ]),
            })
        }
        _ => Err(RuntimeError::TypeError {
            expected: "EvidenceProgram".to_string(),
            found: ev.runtime_type(),
            message: "run_evidence_typed() requires an EvidenceProgram".to_string(),
        }),
    }
}

/// run_evidence_with_surrogate(ev: EvidenceProgram, s: SurrogateModel, backend: BackendKind) -> EvidenceResult
fn builtin_run_evidence_with_surrogate(
    args: Vec<RuntimeValue>,
) -> Result<RuntimeValue, RuntimeError> {
    let ev = &args[0];
    let surrogate_val = &args[1];
    let backend_val = &args[2];

    // Validate evidence program
    match ev {
        RuntimeValue::EvidenceProgram { name, .. } => {
            // Extract surrogate model
            let surrogate = surrogate_val.as_surrogate_model()?;

            // Extract backend
            let backend = backend_val.as_backend_kind()?;

            // Check that backend can use surrogate
            if !backend.requires_surrogate() && backend != BackendKind::Mechanistic {
                return Err(RuntimeError::BackendError(format!(
                    "Backend {:?} does not support surrogate models",
                    backend
                )));
            }

            // TODO: Integrate with actual evidence runner using surrogate
            Ok(RuntimeValue::EvidenceResult {
                posterior_samples: vec![],
                diagnostics: HashMap::from([
                    ("evidence_program".to_string(), name.clone()),
                    ("backend".to_string(), backend.variant_name().to_string()),
                    ("surrogate_id".to_string(), surrogate.id.to_string()),
                    ("status".to_string(), "stub".to_string()),
                ]),
            })
        }
        _ => Err(RuntimeError::TypeError {
            expected: "EvidenceProgram".to_string(),
            found: ev.runtime_type(),
            message: "run_evidence_with_surrogate() requires an EvidenceProgram".to_string(),
        }),
    }
}

// =============================================================================
// Week 30: Surrogate Evaluation Built-in Functions
// =============================================================================

/// evaluate_surrogate(ev: EvidenceProgram, surr: SurrogateModel, cfg: SurrogateEvalConfig) -> SurrogateEvalReport
fn builtin_evaluate_surrogate(args: Vec<RuntimeValue>) -> Result<RuntimeValue, RuntimeError> {
    let ev = &args[0];
    let surr_val = &args[1];
    let cfg_record = &args[2];

    // Validate evidence program
    let ev_handle = match ev {
        RuntimeValue::EvidenceProgram { handle, .. } => handle.clone(),
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "EvidenceProgram".to_string(),
                found: ev.runtime_type(),
                message: "evaluate_surrogate() requires an EvidenceProgram".to_string(),
            })
        }
    };

    // Extract surrogate model
    let surr = surr_val.as_surrogate_model()?;

    // Extract configuration from record
    let cfg_map = match cfg_record {
        RuntimeValue::Record(fields) => fields,
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "Record".to_string(),
                found: cfg_record.runtime_type(),
                message: "evaluate_surrogate() config must be a Record".to_string(),
            })
        }
    };

    // Extract n_eval field
    let n_eval = match cfg_map.get("n_eval") {
        Some(RuntimeValue::Int(n)) => *n as usize,
        Some(other) => {
            return Err(RuntimeError::TypeError {
                expected: "Int".to_string(),
                found: other.runtime_type(),
                message: "n_eval field must be an Int".to_string(),
            })
        }
        None => return Err(RuntimeError::FieldNotFound("n_eval".to_string())),
    };

    // Extract backend_ref field
    let backend_ref = match cfg_map.get("backend_ref") {
        Some(val) => val.as_backend_kind()?,
        None => return Err(RuntimeError::FieldNotFound("backend_ref".to_string())),
    };

    // Extract seed field
    let seed = match cfg_map.get("seed") {
        Some(RuntimeValue::Int(s)) => *s as u64,
        Some(other) => {
            return Err(RuntimeError::TypeError {
                expected: "Int".to_string(),
                found: other.runtime_type(),
                message: "seed field must be an Int".to_string(),
            })
        }
        None => return Err(RuntimeError::FieldNotFound("seed".to_string())),
    };

    // Build SurrogateEvalConfig
    use crate::ml::{evaluate_surrogate, SurrogateEvalConfig};
    let cfg = SurrogateEvalConfig {
        n_eval,
        backend_ref,
        seed,
    };

    // Call evaluation engine
    let report = evaluate_surrogate(&ev_handle, surr, &cfg)
        .map_err(|e| RuntimeError::SurrogateError(e.to_string()))?;

    // Convert report to RuntimeValue::Record
    let mut report_fields = HashMap::new();
    report_fields.insert(
        "n_eval".to_string(),
        RuntimeValue::Int(report.n_eval as i64),
    );
    report_fields.insert("rmse".to_string(), RuntimeValue::Float(report.rmse));
    report_fields.insert("mae".to_string(), RuntimeValue::Float(report.mae));
    report_fields.insert(
        "max_abs_err".to_string(),
        RuntimeValue::Float(report.max_abs_err),
    );
    report_fields.insert(
        "mech_contract_violations".to_string(),
        RuntimeValue::Int(report.mech_contract_violations as i64),
    );
    report_fields.insert(
        "surr_contract_violations".to_string(),
        RuntimeValue::Int(report.surr_contract_violations as i64),
    );

    Ok(RuntimeValue::Record(report_fields))
}

// =============================================================================
// Week 31-32: RL Policy Training Built-ins
// =============================================================================

/// Built-in: train_policy_rl(env_cfg: RLEnvConfig, train_cfg: RLTrainConfig) -> (RLTrainReport, RLPolicy)
fn builtin_train_policy_rl(args: Vec<RuntimeValue>) -> Result<RuntimeValue, RuntimeError> {
    let env_cfg_record = &args[0];
    let train_cfg_record = &args[1];

    // Extract RLEnvConfig from record
    let env_cfg_map = match env_cfg_record {
        RuntimeValue::Record(fields) => fields,
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "Record".to_string(),
                found: env_cfg_record.runtime_type(),
                message: "train_policy_rl() env_cfg must be a Record".to_string(),
            })
        }
    };

    // Extract evidence program
    let ev_handle = match env_cfg_map.get("evidence_program") {
        Some(RuntimeValue::EvidenceProgram { handle, .. }) => handle.clone(),
        Some(other) => {
            return Err(RuntimeError::TypeError {
                expected: "EvidenceProgram".to_string(),
                found: other.runtime_type(),
                message: "evidence_program field must be an EvidenceProgram".to_string(),
            })
        }
        None => return Err(RuntimeError::FieldNotFound("evidence_program".to_string())),
    };

    // Extract backend
    let backend = match env_cfg_map.get("backend") {
        Some(val) => val.as_backend_kind()?,
        None => return Err(RuntimeError::FieldNotFound("backend".to_string())),
    };

    // Extract n_cycles
    let n_cycles = match env_cfg_map.get("n_cycles") {
        Some(RuntimeValue::Int(n)) => *n as usize,
        Some(other) => {
            return Err(RuntimeError::TypeError {
                expected: "Int".to_string(),
                found: other.runtime_type(),
                message: "n_cycles field must be an Int".to_string(),
            })
        }
        None => return Err(RuntimeError::FieldNotFound("n_cycles".to_string())),
    };

    // Extract reward weights (with defaults if missing for backwards compatibility)
    let w_response = match env_cfg_map.get("w_response") {
        Some(RuntimeValue::Float(w)) => *w,
        Some(RuntimeValue::Int(w)) => *w as f64,
        Some(other) => {
            return Err(RuntimeError::TypeError {
                expected: "Float".to_string(),
                found: other.runtime_type(),
                message: "w_response field must be a Float".to_string(),
            })
        }
        None => 1.0, // default
    };

    let w_tox = match env_cfg_map.get("w_tox") {
        Some(RuntimeValue::Float(w)) => *w,
        Some(RuntimeValue::Int(w)) => *w as f64,
        Some(other) => {
            return Err(RuntimeError::TypeError {
                expected: "Float".to_string(),
                found: other.runtime_type(),
                message: "w_tox field must be a Float".to_string(),
            })
        }
        None => 2.0, // default
    };

    let contract_penalty = match env_cfg_map.get("contract_penalty") {
        Some(RuntimeValue::Float(p)) => *p,
        Some(RuntimeValue::Int(p)) => *p as f64,
        Some(other) => {
            return Err(RuntimeError::TypeError {
                expected: "Float".to_string(),
                found: other.runtime_type(),
                message: "contract_penalty field must be a Float".to_string(),
            })
        }
        None => 10.0, // default
    };

    // Extract RLTrainConfig from record
    let train_cfg_map = match train_cfg_record {
        RuntimeValue::Record(fields) => fields,
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "Record".to_string(),
                found: train_cfg_record.runtime_type(),
                message: "train_policy_rl() train_cfg must be a Record".to_string(),
            })
        }
    };

    // Extract training parameters
    let n_episodes = match train_cfg_map.get("n_episodes") {
        Some(RuntimeValue::Int(n)) => *n as usize,
        Some(other) => {
            return Err(RuntimeError::TypeError {
                expected: "Int".to_string(),
                found: other.runtime_type(),
                message: "n_episodes field must be an Int".to_string(),
            })
        }
        None => return Err(RuntimeError::FieldNotFound("n_episodes".to_string())),
    };

    let max_steps = match train_cfg_map.get("max_steps_per_episode") {
        Some(RuntimeValue::Int(n)) => *n as usize,
        Some(other) => {
            return Err(RuntimeError::TypeError {
                expected: "Int".to_string(),
                found: other.runtime_type(),
                message: "max_steps_per_episode field must be an Int".to_string(),
            })
        }
        None => {
            return Err(RuntimeError::FieldNotFound(
                "max_steps_per_episode".to_string(),
            ))
        }
    };

    let gamma = match train_cfg_map.get("gamma") {
        Some(RuntimeValue::Float(g)) => *g,
        Some(other) => {
            return Err(RuntimeError::TypeError {
                expected: "Float".to_string(),
                found: other.runtime_type(),
                message: "gamma field must be a Float".to_string(),
            })
        }
        None => return Err(RuntimeError::FieldNotFound("gamma".to_string())),
    };

    let alpha = match train_cfg_map.get("alpha") {
        Some(RuntimeValue::Float(a)) => *a,
        Some(other) => {
            return Err(RuntimeError::TypeError {
                expected: "Float".to_string(),
                found: other.runtime_type(),
                message: "alpha field must be a Float".to_string(),
            })
        }
        None => return Err(RuntimeError::FieldNotFound("alpha".to_string())),
    };

    let eps_start = match train_cfg_map.get("eps_start") {
        Some(RuntimeValue::Float(e)) => *e,
        Some(other) => {
            return Err(RuntimeError::TypeError {
                expected: "Float".to_string(),
                found: other.runtime_type(),
                message: "eps_start field must be a Float".to_string(),
            })
        }
        None => return Err(RuntimeError::FieldNotFound("eps_start".to_string())),
    };

    let eps_end = match train_cfg_map.get("eps_end") {
        Some(RuntimeValue::Float(e)) => *e,
        Some(other) => {
            return Err(RuntimeError::TypeError {
                expected: "Float".to_string(),
                found: other.runtime_type(),
                message: "eps_end field must be a Float".to_string(),
            })
        }
        None => return Err(RuntimeError::FieldNotFound("eps_end".to_string())),
    };

    // Build RLTrainConfig
    let train_cfg = RLTrainConfig {
        n_episodes,
        max_steps_per_episode: max_steps,
        gamma,
        alpha,
        eps_start,
        eps_end,
    };

    // Build DoseToxEnvConfig
    use crate::rl::{DoseToxEnv, DoseToxEnvConfig};
    let env_cfg = DoseToxEnvConfig {
        ev_handle,
        backend,
        n_cycles,
        dose_levels_mg: vec![0.0, 50.0, 100.0, 200.0, 300.0], // Default dose levels
        reward_response_weight: w_response,
        reward_tox_penalty: w_tox,
        contract_penalty,
        seed: Some(42), // Fixed seed for reproducibility
    };

    // Create environment
    let mut env = DoseToxEnv::new(env_cfg);

    // Create state discretizer with reasonable defaults for dose-tox environment
    // State space: [ANC (0-2), tumour_ratio (0-2), prev_dose_norm (0-1), cycle (0-n_cycles)]
    let discretizer = BoxDiscretizer::new(
        vec![10; 4], // bins per dimension
        vec![0.0, 0.0, 0.0, 0.0],
        vec![2.0, 2.0, 1.0, n_cycles as f64],
    );

    // Create RNG for training
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(12345);

    // Train the policy
    use crate::rl::train_q_learning;
    let (train_report, policy) = train_q_learning(&mut env, &discretizer, &train_cfg, &mut rng)
        .map_err(|e| RuntimeError::Custom(format!("RL training failed: {}", e)))?;

    // Convert training report to RuntimeValue
    let mut report_fields = HashMap::new();
    report_fields.insert(
        "n_episodes".to_string(),
        RuntimeValue::Int(train_report.n_episodes as i64),
    );
    report_fields.insert(
        "avg_reward".to_string(),
        RuntimeValue::Float(train_report.avg_reward),
    );
    report_fields.insert(
        "final_epsilon".to_string(),
        RuntimeValue::Float(train_report.final_epsilon),
    );
    report_fields.insert(
        "avg_episode_length".to_string(),
        RuntimeValue::Float(train_report.avg_episode_length),
    );
    report_fields.insert(
        "total_steps".to_string(),
        RuntimeValue::Int(train_report.total_steps as i64),
    );

    // Return tuple as record with two fields
    let mut result = HashMap::new();
    result.insert("report".to_string(), RuntimeValue::Record(report_fields));
    result.insert("policy".to_string(), RuntimeValue::RLPolicy(policy));

    Ok(RuntimeValue::Record(result))
}

/// Built-in: simulate_policy_rl(env_cfg: RLEnvConfig, policy: RLPolicy, n_episodes: Int) -> PolicyEvalReport
fn builtin_simulate_policy_rl(args: Vec<RuntimeValue>) -> Result<RuntimeValue, RuntimeError> {
    let env_cfg_record = &args[0];
    let policy_val = &args[1];
    let n_episodes_val = &args[2];

    // Extract policy
    let policy = match policy_val {
        RuntimeValue::RLPolicy(p) => p,
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "RLPolicy".to_string(),
                found: policy_val.runtime_type(),
                message: "simulate_policy_rl() requires an RLPolicy".to_string(),
            })
        }
    };

    // Extract n_episodes
    let n_episodes = match n_episodes_val {
        RuntimeValue::Int(n) => *n as usize,
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "Int".to_string(),
                found: n_episodes_val.runtime_type(),
                message: "simulate_policy_rl() n_episodes must be an Int".to_string(),
            })
        }
    };

    // Extract environment config (similar to train_policy_rl)
    let env_cfg_map = match env_cfg_record {
        RuntimeValue::Record(fields) => fields,
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "Record".to_string(),
                found: env_cfg_record.runtime_type(),
                message: "simulate_policy_rl() env_cfg must be a Record".to_string(),
            })
        }
    };

    let ev_handle = match env_cfg_map.get("evidence_program") {
        Some(RuntimeValue::EvidenceProgram { handle, .. }) => handle.clone(),
        Some(other) => {
            return Err(RuntimeError::TypeError {
                expected: "EvidenceProgram".to_string(),
                found: other.runtime_type(),
                message: "evidence_program field must be an EvidenceProgram".to_string(),
            })
        }
        None => return Err(RuntimeError::FieldNotFound("evidence_program".to_string())),
    };

    let backend = match env_cfg_map.get("backend") {
        Some(val) => val.as_backend_kind()?,
        None => return Err(RuntimeError::FieldNotFound("backend".to_string())),
    };

    let n_cycles = match env_cfg_map.get("n_cycles") {
        Some(RuntimeValue::Int(n)) => *n as usize,
        Some(other) => {
            return Err(RuntimeError::TypeError {
                expected: "Int".to_string(),
                found: other.runtime_type(),
                message: "n_cycles field must be an Int".to_string(),
            })
        }
        None => return Err(RuntimeError::FieldNotFound("n_cycles".to_string())),
    };

    let w_response = match env_cfg_map.get("w_response") {
        Some(RuntimeValue::Float(w)) => *w,
        Some(RuntimeValue::Int(w)) => *w as f64,
        _ => 1.0,
    };

    let w_tox = match env_cfg_map.get("w_tox") {
        Some(RuntimeValue::Float(w)) => *w,
        Some(RuntimeValue::Int(w)) => *w as f64,
        _ => 2.0,
    };

    let contract_penalty = match env_cfg_map.get("contract_penalty") {
        Some(RuntimeValue::Float(p)) => *p,
        Some(RuntimeValue::Int(p)) => *p as f64,
        _ => 10.0,
    };

    // Build environment
    use crate::rl::{DoseToxEnv, DoseToxEnvConfig};
    let env_cfg = DoseToxEnvConfig {
        ev_handle,
        backend,
        n_cycles,
        dose_levels_mg: vec![0.0, 50.0, 100.0, 200.0, 300.0],
        reward_response_weight: w_response,
        reward_tox_penalty: w_tox,
        contract_penalty,
        seed: Some(98765), // Different seed for evaluation
    };

    let mut env = DoseToxEnv::new(env_cfg);

    // Rebuild discretizer from policy metadata
    let discretizer = BoxDiscretizer::new(
        policy.bins_per_dim.clone(),
        policy.min_vals.clone(),
        policy.max_vals.clone(),
    );

    // Create RNG for evaluation
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(54321);

    // Evaluate the policy
    use crate::rl::evaluate_policy;
    let eval_report = evaluate_policy(&mut env, policy, &discretizer, n_episodes)
        .map_err(|e| RuntimeError::Custom(format!("Policy evaluation failed: {}", e)))?;

    // Convert evaluation report to RuntimeValue
    let mut report_fields = HashMap::new();
    report_fields.insert(
        "n_episodes".to_string(),
        RuntimeValue::Int(eval_report.n_episodes as i64),
    );
    report_fields.insert(
        "avg_reward".to_string(),
        RuntimeValue::Float(eval_report.avg_reward),
    );
    report_fields.insert(
        "avg_contract_violations".to_string(),
        RuntimeValue::Float(eval_report.avg_contract_violations),
    );
    report_fields.insert(
        "avg_episode_length".to_string(),
        RuntimeValue::Float(eval_report.avg_episode_length),
    );

    Ok(RuntimeValue::Record(report_fields))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_fn_name_parsing() {
        assert_eq!(
            BuiltinFn::from_name("train_surrogate"),
            Some(BuiltinFn::TrainSurrogate)
        );
        assert_eq!(
            BuiltinFn::from_name("run_evidence_typed"),
            Some(BuiltinFn::RunEvidenceTyped)
        );
        assert_eq!(BuiltinFn::from_name("unknown_fn"), None);
    }

    #[test]
    fn test_builtin_arity() {
        assert_eq!(BuiltinFn::Print.arity(), 1);
        assert_eq!(BuiltinFn::TrainSurrogate.arity(), 2);
        assert_eq!(BuiltinFn::RunEvidenceWithSurrogate.arity(), 3);
    }

    #[test]
    fn test_print_builtin() {
        let args = vec![RuntimeValue::String("Hello, World!".to_string())];
        let result = call_builtin(BuiltinFn::Print, args);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RuntimeValue::Unit);
    }

    #[test]
    fn test_print_wrong_type() {
        let args = vec![RuntimeValue::Int(42)];
        let result = call_builtin(BuiltinFn::Print, args);
        assert!(result.is_err());
    }

    #[test]
    fn test_train_surrogate_builtin() {
        let ev = RuntimeValue::EvidenceProgram {
            name: "TestEvidence".to_string(),
            handle: "test_handle".to_string(),
        };

        let cfg = RuntimeValue::Record(HashMap::from([
            ("n_train".to_string(), RuntimeValue::Int(100)),
            (
                "backend".to_string(),
                RuntimeValue::BackendKind(BackendKind::Mechanistic),
            ),
            ("seed".to_string(), RuntimeValue::Int(42)),
            ("max_epochs".to_string(), RuntimeValue::Int(50)),
            ("batch_size".to_string(), RuntimeValue::Int(32)),
        ]));

        let result = call_builtin(BuiltinFn::TrainSurrogate, vec![ev, cfg]);
        assert!(result.is_ok());

        match result.unwrap() {
            RuntimeValue::SurrogateModel(_) => {}
            _ => panic!("Expected SurrogateModel return value"),
        }
    }

    #[test]
    fn test_train_surrogate_invalid_config() {
        let ev = RuntimeValue::EvidenceProgram {
            name: "TestEvidence".to_string(),
            handle: "test_handle".to_string(),
        };

        // Invalid: using Surrogate backend to generate training data (circular)
        let cfg = RuntimeValue::Record(HashMap::from([
            ("n_train".to_string(), RuntimeValue::Int(100)),
            (
                "backend".to_string(),
                RuntimeValue::BackendKind(BackendKind::Surrogate),
            ),
            ("seed".to_string(), RuntimeValue::Int(42)),
            ("max_epochs".to_string(), RuntimeValue::Int(50)),
            ("batch_size".to_string(), RuntimeValue::Int(32)),
        ]));

        let result = call_builtin(BuiltinFn::TrainSurrogate, vec![ev, cfg]);
        assert!(result.is_err());
        match result {
            Err(RuntimeError::SurrogateError(msg)) => {
                assert!(msg.contains("cannot use Surrogate backend"));
            }
            _ => panic!("Expected SurrogateError"),
        }
    }

    #[test]
    fn test_run_evidence_typed() {
        let ev = RuntimeValue::EvidenceProgram {
            name: "TestEvidence".to_string(),
            handle: "test_handle".to_string(),
        };
        let backend = RuntimeValue::BackendKind(BackendKind::Mechanistic);

        let result = call_builtin(BuiltinFn::RunEvidenceTyped, vec![ev, backend]);
        assert!(result.is_ok());

        match result.unwrap() {
            RuntimeValue::EvidenceResult { diagnostics, .. } => {
                assert_eq!(diagnostics.get("backend"), Some(&"Mechanistic".to_string()));
            }
            _ => panic!("Expected EvidenceResult"),
        }
    }

    #[test]
    fn test_run_evidence_with_surrogate() {
        let ev = RuntimeValue::EvidenceProgram {
            name: "TestEvidence".to_string(),
            handle: "test_handle".to_string(),
        };
        let surrogate = RuntimeValue::SurrogateModel(SurrogateModelHandle::new());
        let backend = RuntimeValue::BackendKind(BackendKind::Hybrid);

        let result = call_builtin(
            BuiltinFn::RunEvidenceWithSurrogate,
            vec![ev, surrogate, backend],
        );
        assert!(result.is_ok());

        match result.unwrap() {
            RuntimeValue::EvidenceResult { diagnostics, .. } => {
                assert_eq!(diagnostics.get("backend"), Some(&"Hybrid".to_string()));
                assert!(diagnostics.contains_key("surrogate_id"));
            }
            _ => panic!("Expected EvidenceResult"),
        }
    }

    #[test]
    fn test_arity_mismatch() {
        // print() expects 1 argument, give it 2
        let args = vec![
            RuntimeValue::String("hello".to_string()),
            RuntimeValue::String("world".to_string()),
        ];
        let result = call_builtin(BuiltinFn::Print, args);
        assert!(result.is_err());
        match result {
            Err(RuntimeError::ArityMismatch {
                fn_name,
                expected,
                found,
            }) => {
                assert_eq!(fn_name, "print");
                assert_eq!(expected, 1);
                assert_eq!(found, 2);
            }
            _ => panic!("Expected ArityMismatch error"),
        }
    }
}
