use crate::rl::core::State;
use crate::rl::dose_guideline_ir::DoseGuidelineIRHost;

/// Convert State.features into a feature map for guideline evaluation.
/// Assumes feature ordering consistent with DoseToxEnv State:
///   0: ANC (normalized)
///   1: tumour_ratio (normalized)
///   2: cycle_index (normalized by /10.0)
///   3: prev_dose (normalized by /300.0)
pub fn state_to_feature_map(s: &State) -> Vec<(String, f64)> {
    let anc = s.features.get(0).copied().unwrap_or(0.0);
    let tumour_ratio = s.features.get(1).copied().unwrap_or(1.0);
    let cycle_index = s.features.get(2).copied().unwrap_or(0.0) * 10.0;
    let prev_dose_mg = s.features.get(3).copied().unwrap_or(0.0) * 300.0;

    vec![
        ("ANC".to_string(), anc),
        ("anc".to_string(), anc),
        ("tumour_ratio".to_string(), tumour_ratio),
        ("tumor_ratio".to_string(), tumour_ratio),
        ("cycle".to_string(), cycle_index),
        ("cycle_index".to_string(), cycle_index),
        ("prev_dose".to_string(), prev_dose_mg),
        ("previous_dose".to_string(), prev_dose_mg),
    ]
}

/// Map a recommended dose (mg) to an action index in dose_levels.
/// Strategy: exact match if possible, otherwise nearest neighbour.
pub fn dose_mg_to_action_index(dose_levels_mg: &[f64], recommended_mg: f64) -> usize {
    if dose_levels_mg.is_empty() {
        return 0;
    }

    // If recommended_mg is negative or ~0, map to index of 0.0 if present,
    // else to smallest dose.
    if recommended_mg <= 0.0 {
        if let Some((idx, _)) = dose_levels_mg
            .iter()
            .enumerate()
            .find(|(_, &d)| (d - 0.0).abs() < f64::EPSILON)
        {
            return idx;
        } else {
            return 0;
        }
    }

    // Otherwise choose nearest dose level
    let mut best_idx = 0usize;
    let mut best_diff = f64::INFINITY;
    for (i, &d) in dose_levels_mg.iter().enumerate() {
        let diff = (d - recommended_mg).abs();
        if diff < best_diff {
            best_diff = diff;
            best_idx = i;
        }
    }

    best_idx
}

/// Wrapper type exposing a DoseGuideline as a policy on State.
#[derive(Debug, Clone)]
pub struct GuidelinePolicy {
    pub guideline: DoseGuidelineIRHost,
}

impl GuidelinePolicy {
    /// Given a State, returns the action index (dose level) recommended by the guideline.
    pub fn act(&self, state: &State) -> usize {
        let features = state_to_feature_map(state);
        let rec_mg = self.guideline.evaluate(&features).unwrap_or(0.0);
        dose_mg_to_action_index(&self.guideline.dose_levels_mg, rec_mg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::dose_guideline_ir::{AtomicConditionIR, ComparisonOpIR, DoseRuleIR};

    #[test]
    fn test_state_to_feature_map() {
        let s = State {
            features: vec![0.5, 1.2, 0.3, 0.5],
        };
        let fp = state_to_feature_map(&s);
        // convert to map for easy lookup
        let map: std::collections::HashMap<_, _> = fp.into_iter().collect();
        assert_eq!(map.get("ANC").copied().unwrap(), 0.5);
        assert_eq!(map.get("tumour_ratio").copied().unwrap(), 1.2);
        assert_eq!(map.get("prev_dose").copied().unwrap(), 150.0);
        assert_eq!(map.get("cycle_index").copied().unwrap(), 3.0);
    }

    #[test]
    fn test_dose_mg_to_action_index_exact() {
        let doses = vec![0.0, 50.0, 100.0, 200.0];
        assert_eq!(dose_mg_to_action_index(&doses, 100.0), 2);
        assert_eq!(dose_mg_to_action_index(&doses, 0.0), 0);
    }

    #[test]
    fn test_dose_mg_to_action_index_nearest() {
        let doses = vec![0.0, 50.0, 100.0, 200.0];
        assert_eq!(dose_mg_to_action_index(&doses, 120.0), 2); // nearest to 100
        assert_eq!(dose_mg_to_action_index(&doses, 180.0), 3); // nearest to 200
        assert_eq!(dose_mg_to_action_index(&doses, -10.0), 0); // negative maps to 0
    }

    #[test]
    fn test_guideline_policy_act() {
        // Guideline: if ANC <= 1.0 then 0.0 else 100.0
        let doses = vec![0.0, 100.0];
        let mut guideline = DoseGuidelineIRHost::new(
            "Test Guideline".to_string(),
            "Test".to_string(),
            vec!["ANC".to_string()],
            doses.clone(),
        );
        guideline.add_rule(DoseRuleIR::new(
            vec![AtomicConditionIR::new(
                "ANC".to_string(),
                ComparisonOpIR::LE,
                1.0,
            )],
            0,
            0.0,
        ));
        guideline.add_rule(DoseRuleIR::new(
            vec![AtomicConditionIR::new(
                "ANC".to_string(),
                ComparisonOpIR::GT,
                1.0,
            )],
            1,
            100.0,
        ));
        let policy = GuidelinePolicy { guideline };

        let s_low = State {
            features: vec![0.5, 1.0, 0.0, 0.0],
        };
        let s_high = State {
            features: vec![2.0, 1.0, 0.0, 0.0],
        };

        assert_eq!(policy.act(&s_low), 0); // 0.0 mg
        assert_eq!(policy.act(&s_high), 1); // 100.0 mg
    }
}
