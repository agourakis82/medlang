//! External JSON-Based Policy
//!
//! Implements PolicyPlugin for policies that shell out to external processes
//! (typically Python scripts with trained RL models). Communication happens via JSON:
//! - Observation is serialized to JSON and sent to stdin
//! - Action is read as JSON from stdout
//!
//! This allows integration of policies trained with any RL framework (Stable-Baselines3,
//! RLlib, CleanRL, etc.) without requiring Rust bindings.

use crate::policy::plugin::PolicyPlugin;
use crate::rl::{RlAction, RlObservation};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::process::{Command, Stdio};

/// Configuration for an external JSON-based policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalJsonPolicyConfig {
    /// Command to run, e.g. ["python", "trained_policy.py"]
    ///
    /// The first element is the program, subsequent elements are arguments.
    pub cmd: Vec<String>,

    /// Optional: working directory for the command
    pub working_dir: Option<String>,
}

/// A policy that delegates action selection to an external process.
///
/// Protocol:
/// 1. Spawn process with configured command
/// 2. Write RlObservation as JSON to stdin
/// 3. Read RlAction as JSON from stdout
/// 4. Process exits (stateless per-call, or can be kept alive for batch)
///
/// For Week 22, we spawn a new process per action for simplicity.
/// Production version could maintain a persistent process with request/response protocol.
pub struct ExternalJsonPolicy {
    pub name: String,
    pub cfg: ExternalJsonPolicyConfig,
}

impl ExternalJsonPolicy {
    /// Create a new external JSON policy.
    pub fn new(name: String, cfg: ExternalJsonPolicyConfig) -> Self {
        assert!(
            !cfg.cmd.is_empty(),
            "ExternalJsonPolicy command cannot be empty"
        );
        Self { name, cfg }
    }

    /// Execute the external command with observation and return action.
    fn execute_policy(&self, obs: &RlObservation) -> Result<RlAction, String> {
        let cmd_prog = &self.cfg.cmd[0];
        let cmd_args = &self.cfg.cmd[1..];

        let mut command = Command::new(cmd_prog);
        command
            .args(cmd_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(ref wd) = self.cfg.working_dir {
            command.current_dir(wd);
        }

        let mut child = command
            .spawn()
            .map_err(|e| format!("Failed to spawn policy process: {}", e))?;

        // Send observation as JSON to stdin
        {
            let stdin = child
                .stdin
                .as_mut()
                .ok_or_else(|| "Failed to open stdin".to_string())?;

            let json_obs = serde_json::to_vec(obs)
                .map_err(|e| format!("Failed to serialize observation: {}", e))?;

            stdin
                .write_all(&json_obs)
                .map_err(|e| format!("Failed to write to stdin: {}", e))?;
        }

        // Read action from stdout
        let output = child
            .wait_with_output()
            .map_err(|e| format!("Failed to read policy output: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "External policy command failed with status {:?}:\n{}",
                output.status.code(),
                stderr
            ));
        }

        let action: RlAction = serde_json::from_slice(&output.stdout)
            .map_err(|e| format!("Failed to deserialize action: {}", e))?;

        Ok(action)
    }
}

impl PolicyPlugin for ExternalJsonPolicy {
    fn kind(&self) -> &'static str {
        "external_json"
    }

    fn select_action(&mut self, obs: &RlObservation) -> RlAction {
        match self.execute_policy(obs) {
            Ok(action) => action,
            Err(e) => {
                eprintln!("External policy error: {}", e);
                eprintln!("Falling back to safe default action (dose_factor = 0.5)");
                RlAction { dose_factor: 0.5 }
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_external_json_policy_config() {
        let cfg = ExternalJsonPolicyConfig {
            cmd: vec!["python".to_string(), "policy.py".to_string()],
            working_dir: Some("/tmp".to_string()),
        };

        let policy = ExternalJsonPolicy::new("test_policy".to_string(), cfg);
        assert_eq!(policy.kind(), "external_json");
        assert_eq!(policy.name(), "test_policy");
    }

    #[test]
    #[should_panic(expected = "command cannot be empty")]
    fn test_empty_command_panics() {
        let cfg = ExternalJsonPolicyConfig {
            cmd: vec![],
            working_dir: None,
        };

        ExternalJsonPolicy::new("test".to_string(), cfg);
    }

    // Integration test with actual external process would require a test script
    // For now, we test the structure and basic error handling
}
