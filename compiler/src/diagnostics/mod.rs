//! Diagnostics module for model validation and quality assessment
//!
//! This module provides tools for:
//! - MCMC quality metrics (R-hat, ESS, divergences)
//! - Posterior predictive checks (PPC)
//! - Simulation-based calibration (SBC)

pub mod mcmc;
pub mod ppc;
pub mod quantum;
pub mod quantum_trust;
pub mod sbc;

pub use mcmc::{
    summarize_cmdstan_fit, DiagnosticsError, FitMcmcSummary, FitQuality, ParamMcmcStats,
};
pub use ppc::{posterior_predictive_checks, PpcEndpoint, PpcReport, PpcTumourPerVisit};
pub use quantum::{QuantumPosteriorInfo, QuantumPriorInfo, QuantumPriorPosteriorComparison};
pub use quantum_trust::{
    build_trust_report, classify_all, classify_trust, PriorInflationPolicy, QuantumPriorConfigKind,
    QuantumPriorConfigSummary, QuantumTrustLevel, QuantumTrustReport, QuantumTrustScore,
    QuantumTrustSummary,
};
pub use sbc::{analyze_sbc_results, run_sbc, SbcConfig, SbcQuality, SbcReplication, SbcResult};
