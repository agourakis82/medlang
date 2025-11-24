//! Portfolio-Level Candidate Ranking & Cross-Ligand Design Evaluation
//!
//! Evaluates multiple QM-backed ligands under a single trial design to rank
//! candidates by probability of success (PoS) and efficacy metrics.

use crate::design::{DecisionEvalResult, DesignConfig, DesignSummary, PosteriorDraw};
use serde::{Deserialize, Serialize};

// =============================================================================
// Portfolio Data Structures
// =============================================================================

/// Per-ligand metrics combining QM properties and clinical predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LigandPortfolioMetrics {
    /// Ligand identifier (e.g., "LIG001")
    pub ligand_id: String,

    /// QM-derived binding affinity (Kd in Molar)
    pub kd_molar: Option<f64>,

    /// QM-derived tumor partition coefficient
    pub kp_tumor: Option<f64>,

    /// Mean objective response rate (ORR) across population
    pub orr_mean: Option<f64>,

    /// Median progression-free survival (days)
    pub pfs_median: Option<f64>,

    /// Probability of success for primary decision
    pub pos: Option<f64>,

    /// Full design evaluation summary
    pub design_summary: Option<DesignSummary>,
}

impl LigandPortfolioMetrics {
    /// Create metrics with QM properties only (before design evaluation)
    pub fn from_qm(ligand_id: String, kd_molar: Option<f64>, kp_tumor: Option<f64>) -> Self {
        Self {
            ligand_id,
            kd_molar,
            kp_tumor,
            orr_mean: None,
            pfs_median: None,
            pos: None,
            design_summary: None,
        }
    }

    /// Update with design evaluation results
    pub fn with_design_summary(mut self, summary: DesignSummary) -> Self {
        // Extract primary PoS (max across all decisions)
        self.pos = summary
            .decision_results
            .iter()
            .map(|d| d.pos)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        self.design_summary = Some(summary);
        self
    }

    /// Update with efficacy estimates
    pub fn with_efficacy(mut self, orr_mean: Option<f64>, pfs_median: Option<f64>) -> Self {
        self.orr_mean = orr_mean;
        self.pfs_median = pfs_median;
        self
    }
}

/// Ranked portfolio entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioEntry {
    /// Rank (1 = best)
    pub rank: usize,

    /// Ligand metrics
    pub metrics: LigandPortfolioMetrics,
}

/// Complete portfolio evaluation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSummary {
    pub protocol_name: String,
    pub population_model_name: String,
    pub design_n_per_arm: usize,
    pub backend: String,
    pub entries: Vec<PortfolioEntry>,
}

impl PortfolioSummary {
    /// Find entry by ligand ID
    pub fn find_ligand(&self, ligand_id: &str) -> Option<&PortfolioEntry> {
        self.entries
            .iter()
            .find(|e| e.metrics.ligand_id == ligand_id)
    }

    /// Get top N ligands
    pub fn top_n(&self, n: usize) -> &[PortfolioEntry] {
        &self.entries[..n.min(self.entries.len())]
    }

    /// Get PoS range across portfolio
    pub fn pos_range(&self) -> Option<(f64, f64)> {
        let pos_values: Vec<f64> = self.entries.iter().filter_map(|e| e.metrics.pos).collect();

        if pos_values.is_empty() {
            return None;
        }

        let min = pos_values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()?;
        let max = pos_values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()?;

        Some((min, max))
    }
}

// =============================================================================
// Portfolio Design Grid (N per arm sensitivity)
// =============================================================================

/// Per-ligand PoS at a specific sample size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LigandDesignPoint {
    pub ligand_id: String,
    pub pos: f64,
}

/// PoS for all ligands at a specific N per arm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignGridPoint {
    pub n_per_arm: usize,
    pub ligands: Vec<LigandDesignPoint>,
}

/// Complete design grid: PoS vs N for multiple ligands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioDesignGrid {
    pub protocol_name: String,
    pub population_model_name: String,
    pub backend: String,
    pub designs: Vec<DesignGridPoint>,
}

impl PortfolioDesignGrid {
    /// Get PoS trajectory for a specific ligand across all N values
    pub fn ligand_trajectory(&self, ligand_id: &str) -> Vec<(usize, f64)> {
        self.designs
            .iter()
            .filter_map(|design| {
                design
                    .ligands
                    .iter()
                    .find(|l| l.ligand_id == ligand_id)
                    .map(|l| (design.n_per_arm, l.pos))
            })
            .collect()
    }

    /// Find optimal N per arm for a ligand (smallest N with PoS ≥ threshold)
    pub fn optimal_n(&self, ligand_id: &str, pos_threshold: f64) -> Option<usize> {
        self.ligand_trajectory(ligand_id)
            .into_iter()
            .find(|(_, pos)| *pos >= pos_threshold)
            .map(|(n, _)| n)
    }
}

// =============================================================================
// Portfolio Evaluation Engine
// =============================================================================

/// Evaluate a single ligand candidate (scaffold for Week 20)
///
/// In a full implementation, this would:
/// 1. Build oncology PBPK-QSP-QM model with ligand's QM properties
/// 2. Run design evaluation with mechanistic/surrogate/hybrid backend
/// 3. Extract PoS and efficacy metrics
///
/// For Week 20, we use a simplified approach with synthetic evaluation.
pub fn evaluate_ligand_candidate(
    ligand_id: &str,
    kd_molar: Option<f64>,
    kp_tumor: Option<f64>,
    protocol_name: &str,
    design_cfg: &DesignConfig,
    _posterior_draws: Option<&[PosteriorDraw]>,
) -> LigandPortfolioMetrics {
    // Create base metrics from QM
    let mut metrics = LigandPortfolioMetrics::from_qm(ligand_id.to_string(), kd_molar, kp_tumor);

    // Scaffold: Synthetic design evaluation
    // In a full implementation, this would call:
    // - build_oncology_pbpk_qsp_qm(&qm_cfg)
    // - evaluate_design_pos(protocol, model, design_cfg, backend)
    //
    // For now, we create a synthetic DesignSummary based on QM properties
    let synthetic_pos = calculate_synthetic_pos(kd_molar, kp_tumor);

    let design_summary = DesignSummary {
        design_label: format!("N_per_arm={}", design_cfg.n_per_arm),
        n_per_arm: design_cfg.n_per_arm,
        decision_results: vec![DecisionEvalResult {
            decision_name: "Primary_GoNoGo".to_string(),
            endpoint_name: "ORR".to_string(),
            arm_left: "Control".to_string(),
            arm_right: "Treatment".to_string(),
            margin: 0.15,
            prob_threshold: 0.80,
            pos: synthetic_pos,
        }],
    };

    // Synthetic efficacy estimates
    let orr_mean = kd_molar.map(|kd| {
        // Better Kd → higher ORR (simplified model)
        let kd_nm = kd * 1e9; // Convert to nM
        0.2 + 0.4 / (1.0 + (kd_nm / 10.0).powf(0.7))
    });

    let pfs_median = kp_tumor.map(|kp| {
        // Higher Kp_tumor → longer PFS
        100.0 + 50.0 * kp.ln().max(0.0)
    });

    metrics
        .with_design_summary(design_summary)
        .with_efficacy(orr_mean, pfs_median)
}

/// Calculate synthetic PoS based on QM properties (Week 20 scaffold)
fn calculate_synthetic_pos(kd_molar: Option<f64>, kp_tumor: Option<f64>) -> f64 {
    // Simplified model: PoS depends on both Kd and Kp_tumor
    let kd_score = kd_molar
        .map(|kd| {
            let kd_nm = kd * 1e9;
            // Better (lower) Kd → higher score
            (-kd_nm.ln() / 20.0).max(0.0).min(1.0)
        })
        .unwrap_or(0.5);

    let kp_score = kp_tumor
        .map(|kp| {
            // Higher Kp → higher score
            (kp / 10.0).min(1.0)
        })
        .unwrap_or(0.5);

    // Combined score
    let combined = 0.6 * kd_score + 0.4 * kp_score;

    // Map to PoS range [0.4, 0.95]
    0.4 + 0.55 * combined
}

/// Evaluate portfolio of ligands under a single design
pub fn evaluate_portfolio(
    protocol_name: &str,
    population_model_name: &str,
    ligand_data: &[(String, Option<f64>, Option<f64>)], // (id, kd, kp)
    design_cfg: &DesignConfig,
    backend: &str,
    posterior_draws: Option<&[PosteriorDraw]>,
) -> PortfolioSummary {
    let mut metrics_list = Vec::new();

    for (ligand_id, kd_molar, kp_tumor) in ligand_data {
        let metrics = evaluate_ligand_candidate(
            ligand_id,
            *kd_molar,
            *kp_tumor,
            protocol_name,
            design_cfg,
            posterior_draws,
        );
        metrics_list.push(metrics);
    }

    // Rank by PoS (descending)
    metrics_list.sort_by(|a, b| {
        let pa = a.pos.unwrap_or(0.0);
        let pb = b.pos.unwrap_or(0.0);
        pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
    });

    let entries: Vec<PortfolioEntry> = metrics_list
        .into_iter()
        .enumerate()
        .map(|(i, metrics)| PortfolioEntry {
            rank: i + 1,
            metrics,
        })
        .collect();

    PortfolioSummary {
        protocol_name: protocol_name.to_string(),
        population_model_name: population_model_name.to_string(),
        design_n_per_arm: design_cfg.n_per_arm,
        backend: backend.to_string(),
        entries,
    }
}

/// Evaluate portfolio across multiple design sizes (grid search)
pub fn evaluate_portfolio_design_grid(
    protocol_name: &str,
    population_model_name: &str,
    ligand_data: &[(String, Option<f64>, Option<f64>)],
    n_per_arm_values: &[usize],
    n_draws: usize,
    backend: &str,
    posterior_draws: Option<&[PosteriorDraw]>,
) -> PortfolioDesignGrid {
    let mut designs = Vec::new();

    for &n_per_arm in n_per_arm_values {
        let design_cfg = DesignConfig { n_per_arm, n_draws };

        let mut ligand_points = Vec::new();
        for (ligand_id, kd_molar, kp_tumor) in ligand_data {
            let metrics = evaluate_ligand_candidate(
                ligand_id,
                *kd_molar,
                *kp_tumor,
                protocol_name,
                &design_cfg,
                posterior_draws,
            );

            ligand_points.push(LigandDesignPoint {
                ligand_id: ligand_id.clone(),
                pos: metrics.pos.unwrap_or(0.0),
            });
        }

        designs.push(DesignGridPoint {
            n_per_arm,
            ligands: ligand_points,
        });
    }

    PortfolioDesignGrid {
        protocol_name: protocol_name.to_string(),
        population_model_name: population_model_name.to_string(),
        backend: backend.to_string(),
        designs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ligand_metrics_creation() {
        let metrics =
            LigandPortfolioMetrics::from_qm("LIG001".to_string(), Some(3.2e-8), Some(5.1));

        assert_eq!(metrics.ligand_id, "LIG001");
        assert_eq!(metrics.kd_molar, Some(3.2e-8));
        assert_eq!(metrics.kp_tumor, Some(5.1));
        assert!(metrics.pos.is_none());
    }

    #[test]
    fn test_portfolio_ranking() {
        let ligand_data = vec![
            ("LIG001".to_string(), Some(3.2e-8), Some(5.1)),
            ("LIG002".to_string(), Some(1.0e-7), Some(3.2)),
            ("LIG003".to_string(), Some(1.5e-9), Some(7.5)),
        ];

        let design_cfg = DesignConfig {
            n_per_arm: 120,
            n_draws: 400,
        };

        let summary = evaluate_portfolio(
            "TestProtocol",
            "Oncology_PBPK_QSP_QM",
            &ligand_data,
            &design_cfg,
            "mechanistic",
            None,
        );

        assert_eq!(summary.entries.len(), 3);
        assert_eq!(summary.entries[0].rank, 1);
        assert_eq!(summary.entries[1].rank, 2);
        assert_eq!(summary.entries[2].rank, 3);

        // Best ligand should have highest PoS
        let pos_1 = summary.entries[0].metrics.pos.unwrap();
        let pos_2 = summary.entries[1].metrics.pos.unwrap();
        let pos_3 = summary.entries[2].metrics.pos.unwrap();

        assert!(pos_1 >= pos_2);
        assert!(pos_2 >= pos_3);
    }

    #[test]
    fn test_portfolio_summary_top_n() {
        let ligand_data = vec![
            ("LIG001".to_string(), Some(3.2e-8), Some(5.1)),
            ("LIG002".to_string(), Some(1.0e-7), Some(3.2)),
            ("LIG003".to_string(), Some(1.5e-9), Some(7.5)),
        ];

        let design_cfg = DesignConfig {
            n_per_arm: 120,
            n_draws: 400,
        };

        let summary = evaluate_portfolio(
            "TestProtocol",
            "Oncology_PBPK_QSP_QM",
            &ligand_data,
            &design_cfg,
            "mechanistic",
            None,
        );

        let top_2 = summary.top_n(2);
        assert_eq!(top_2.len(), 2);
        assert_eq!(top_2[0].rank, 1);
        assert_eq!(top_2[1].rank, 2);
    }

    #[test]
    fn test_portfolio_design_grid() {
        let ligand_data = vec![
            ("LIG001".to_string(), Some(3.2e-8), Some(5.1)),
            ("LIG002".to_string(), Some(1.0e-7), Some(3.2)),
        ];

        let n_values = vec![80, 120, 160];

        let grid = evaluate_portfolio_design_grid(
            "TestProtocol",
            "Oncology_PBPK_QSP_QM",
            &ligand_data,
            &n_values,
            400,
            "mechanistic",
            None,
        );

        assert_eq!(grid.designs.len(), 3);
        assert_eq!(grid.designs[0].n_per_arm, 80);
        assert_eq!(grid.designs[1].n_per_arm, 120);
        assert_eq!(grid.designs[2].n_per_arm, 160);

        // Each design point should have 2 ligands
        for design in &grid.designs {
            assert_eq!(design.ligands.len(), 2);
        }
    }

    #[test]
    fn test_ligand_trajectory() {
        let ligand_data = vec![
            ("LIG001".to_string(), Some(3.2e-8), Some(5.1)),
            ("LIG002".to_string(), Some(1.0e-7), Some(3.2)),
        ];

        let n_values = vec![80, 120, 160];

        let grid = evaluate_portfolio_design_grid(
            "TestProtocol",
            "Oncology_PBPK_QSP_QM",
            &ligand_data,
            &n_values,
            400,
            "mechanistic",
            None,
        );

        let traj = grid.ligand_trajectory("LIG001");
        assert_eq!(traj.len(), 3);
        assert_eq!(traj[0].0, 80);
        assert_eq!(traj[1].0, 120);
        assert_eq!(traj[2].0, 160);

        // PoS should be non-negative
        for (_, pos) in traj {
            assert!(pos >= 0.0 && pos <= 1.0);
        }
    }

    #[test]
    fn test_optimal_n_calculation() {
        let ligand_data = vec![("LIG001".to_string(), Some(3.2e-8), Some(5.1))];

        let n_values = vec![50, 100, 150, 200];

        let grid = evaluate_portfolio_design_grid(
            "TestProtocol",
            "Oncology_PBPK_QSP_QM",
            &ligand_data,
            &n_values,
            400,
            "mechanistic",
            None,
        );

        let optimal = grid.optimal_n("LIG001", 0.80);
        assert!(optimal.is_some());
    }

    #[test]
    fn test_pos_range() {
        let ligand_data = vec![
            ("LIG001".to_string(), Some(3.2e-8), Some(5.1)),
            ("LIG002".to_string(), Some(1.0e-7), Some(3.2)),
            ("LIG003".to_string(), Some(1.5e-9), Some(7.5)),
        ];

        let design_cfg = DesignConfig {
            n_per_arm: 120,
            n_draws: 400,
        };

        let summary = evaluate_portfolio(
            "TestProtocol",
            "Oncology_PBPK_QSP_QM",
            &ligand_data,
            &design_cfg,
            "mechanistic",
            None,
        );

        let (min, max) = summary.pos_range().unwrap();
        assert!(min <= max);
        assert!(min >= 0.0 && min <= 1.0);
        assert!(max >= 0.0 && max <= 1.0);
    }
}
