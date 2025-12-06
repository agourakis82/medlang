// Week 36: Policy Distillation via Decision Trees
//
// Converts black-box RL policies into interpretable decision trees by:
// 1. Sampling (state, action) pairs from the policy
// 2. Training a CART-style decision tree classifier
// 3. Providing a tree representation that can be executed and analyzed

use crate::rl::core::{RLEnv, State};
use crate::rl::train::RLPolicyHandle;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

// =============================================================================
// Core Types
// =============================================================================

/// Feature metadata for interpreting state dimensions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistillFeature {
    pub name: String,
    pub index: usize,
    pub min: f64,
    pub max: f64,
}

/// Configuration for policy distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillConfig {
    /// Number of episodes to sample from the policy
    pub n_episodes: usize,

    /// Maximum steps per episode
    pub max_steps_per_episode: usize,

    /// Maximum depth of the decision tree
    pub max_depth: usize,

    /// Minimum samples required in a leaf node
    pub min_samples_leaf: usize,

    /// Feature metadata (optional, inferred if empty)
    pub features: Vec<DistillFeature>,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            n_episodes: 200,
            max_steps_per_episode: 10,
            max_depth: 3,
            min_samples_leaf: 20,
            features: Vec::new(),
        }
    }
}

/// Decision tree node
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TreeNode {
    /// Leaf node with a single action
    Leaf { action: usize },

    /// Split node
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

/// Distilled policy represented as a decision tree
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistilledPolicyTree {
    pub n_actions: usize,
    pub features: Vec<DistillFeature>,
    pub root: TreeNode,
}

/// Handle for distilled policy in runtime values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistilledPolicyHandle {
    pub tree: DistilledPolicyTree,
}

impl DistilledPolicyTree {
    /// Select action for a given state
    pub fn act(&self, state: &State) -> usize {
        fn descend(node: &TreeNode, features: &[f64]) -> usize {
            match node {
                TreeNode::Leaf { action } => *action,
                TreeNode::Split {
                    feature_index,
                    threshold,
                    left,
                    right,
                } => {
                    let x = features.get(*feature_index).copied().unwrap_or(0.0);
                    if x <= *threshold {
                        descend(left, features)
                    } else {
                        descend(right, features)
                    }
                }
            }
        }

        descend(&self.root, &state.features)
    }

    /// Get the depth of the tree
    pub fn depth(&self) -> usize {
        fn depth_node(node: &TreeNode) -> usize {
            match node {
                TreeNode::Leaf { .. } => 1,
                TreeNode::Split { left, right, .. } => {
                    1 + std::cmp::max(depth_node(left), depth_node(right))
                }
            }
        }
        depth_node(&self.root)
    }

    /// Count total nodes in the tree
    pub fn n_nodes(&self) -> usize {
        fn count(node: &TreeNode) -> usize {
            match node {
                TreeNode::Leaf { .. } => 1,
                TreeNode::Split { left, right, .. } => 1 + count(left) + count(right),
            }
        }
        count(&self.root)
    }
}

/// Distillation quality report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillReport {
    pub n_train_samples: usize,
    pub train_accuracy: f64,
    pub eval_accuracy: f64,
    pub per_action_accuracy: Vec<f64>,
    pub tree_depth: usize,
    pub n_nodes: usize,
}

/// Training sample: (state features, action label)
#[derive(Debug, Clone)]
pub struct Sample {
    pub features: Vec<f64>,
    pub action: usize,
}

// =============================================================================
// Sampling Policy Behavior
// =============================================================================

/// Sample (state, action) pairs from an environment following the greedy policy
pub fn sample_policy_behavior(
    env: &mut dyn RLEnv,
    policy: &RLPolicyHandle,
    cfg: &DistillConfig,
) -> anyhow::Result<Vec<Sample>> {
    let mut rng = ChaCha20Rng::seed_from_u64(20240601);
    let mut samples = Vec::new();

    for _ep in 0..cfg.n_episodes {
        let mut state = env.reset()?;

        for _step in 0..cfg.max_steps_per_episode {
            // Get greedy action from policy
            let action = policy.select_action_greedy(&state)?;

            // Record sample
            samples.push(Sample {
                features: state.features.clone(),
                action,
            });

            // Step environment
            let step_res = env.step(action)?;
            state = step_res.next_state;

            if step_res.done {
                break;
            }
        }

        // Prevent runaway sampling
        if samples.len() >= 50_000 {
            break;
        }
    }

    Ok(samples)
}

// =============================================================================
// Decision Tree Learning (CART)
// =============================================================================

/// Train a decision tree classifier on the samples
pub fn train_decision_tree(samples: &[Sample], n_actions: usize, cfg: &DistillConfig) -> TreeNode {
    let feature_indices: Vec<usize> = if cfg.features.is_empty() {
        // Use all features
        if let Some(first) = samples.first() {
            (0..first.features.len()).collect()
        } else {
            vec![]
        }
    } else {
        cfg.features.iter().map(|f| f.index).collect()
    };

    build_node(
        samples,
        &feature_indices,
        n_actions,
        0,
        cfg.max_depth,
        cfg.min_samples_leaf,
    )
}

/// Recursively build a decision tree node
fn build_node(
    samples: &[Sample],
    feature_indices: &[usize],
    n_actions: usize,
    depth: usize,
    max_depth: usize,
    min_samples_leaf: usize,
) -> TreeNode {
    // Stopping criteria
    if depth >= max_depth || samples.len() <= min_samples_leaf || is_pure(samples) {
        let majority_action = majority_class(samples, n_actions);
        return TreeNode::Leaf {
            action: majority_action,
        };
    }

    // Find best split
    if let Some((best_feat, best_thresh, left_indices)) =
        find_best_split(samples, feature_indices, n_actions)
    {
        // Partition samples
        let mut left = Vec::new();
        let mut right = Vec::new();

        for (i, s) in samples.iter().enumerate() {
            if left_indices.contains(&i) {
                left.push(s.clone());
            } else {
                right.push(s.clone());
            }
        }

        // Check minimum samples constraint
        if left.len() < min_samples_leaf || right.len() < min_samples_leaf {
            let majority_action = majority_class(samples, n_actions);
            return TreeNode::Leaf {
                action: majority_action,
            };
        }

        // Recursive split
        TreeNode::Split {
            feature_index: best_feat,
            threshold: best_thresh,
            left: Box::new(build_node(
                &left,
                feature_indices,
                n_actions,
                depth + 1,
                max_depth,
                min_samples_leaf,
            )),
            right: Box::new(build_node(
                &right,
                feature_indices,
                n_actions,
                depth + 1,
                max_depth,
                min_samples_leaf,
            )),
        }
    } else {
        // No valid split found
        let majority_action = majority_class(samples, n_actions);
        TreeNode::Leaf {
            action: majority_action,
        }
    }
}

/// Check if all samples have the same action (pure node)
fn is_pure(samples: &[Sample]) -> bool {
    if samples.is_empty() {
        return true;
    }
    let first = samples[0].action;
    samples.iter().all(|s| s.action == first)
}

/// Find the majority class (most common action)
fn majority_class(samples: &[Sample], n_actions: usize) -> usize {
    let mut counts = vec![0usize; n_actions];
    for s in samples {
        if s.action < n_actions {
            counts[s.action] += 1;
        }
    }
    counts
        .iter()
        .enumerate()
        .max_by_key(|(_, &c)| c)
        .map(|(a, _)| a)
        .unwrap_or(0)
}

/// Compute Gini impurity for a set of class counts
fn gini_impurity(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }

    let mut sum_sq = 0.0;
    for &c in counts {
        let p = c as f64 / total as f64;
        sum_sq += p * p;
    }
    1.0 - sum_sq
}

/// Find the best split point using Gini impurity
fn find_best_split(
    samples: &[Sample],
    feature_indices: &[usize],
    n_actions: usize,
) -> Option<(usize, f64, Vec<usize>)> {
    let mut best_feat = None;
    let mut best_thresh = 0.0;
    let mut best_gini = f64::INFINITY;
    let mut best_left_idx = Vec::new();

    if samples.is_empty() {
        return None;
    }

    for &feat in feature_indices {
        // Sort samples by feature value
        let mut indexed_vals: Vec<(usize, f64, usize)> = samples
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let val = s.features.get(feat).copied().unwrap_or(0.0);
                (i, val, s.action)
            })
            .collect();

        indexed_vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Initialize counts: all samples start on the right
        let mut left_counts = vec![0usize; n_actions];
        let mut right_counts = vec![0usize; n_actions];

        for (_, _, action) in &indexed_vals {
            if *action < n_actions {
                right_counts[*action] += 1;
            }
        }

        // Try each split point
        for i in 0..indexed_vals.len() - 1 {
            let (idx, val, action) = indexed_vals[i];

            // Move sample from right to left
            if action < n_actions {
                left_counts[action] += 1;
                right_counts[action] -= 1;
            }

            let next_val = indexed_vals[i + 1].1;

            // Skip if values are the same (no split)
            if (next_val - val).abs() < f64::EPSILON {
                continue;
            }

            // Compute weighted Gini impurity
            let n_left: usize = left_counts.iter().sum();
            let n_right: usize = right_counts.iter().sum();

            if n_left == 0 || n_right == 0 {
                continue;
            }

            let g_left = gini_impurity(&left_counts);
            let g_right = gini_impurity(&right_counts);
            let g_split =
                (n_left as f64 * g_left + n_right as f64 * g_right) / (n_left + n_right) as f64;

            // Update best split
            if g_split < best_gini {
                best_gini = g_split;
                best_feat = Some(feat);
                best_thresh = 0.5 * (val + next_val);

                // Record left indices
                let mut left_idx = Vec::new();
                for j in 0..=i {
                    left_idx.push(indexed_vals[j].0);
                }
                best_left_idx = left_idx;
            }
        }
    }

    best_feat.map(|feat| (feat, best_thresh, best_left_idx))
}

// =============================================================================
// Fidelity Metrics
// =============================================================================

/// Compute accuracy of the tree on a set of samples
pub fn compute_accuracy(tree: &TreeNode, samples: &[Sample]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }

    let mut correct = 0;
    for s in samples {
        let state = State::new(s.features.clone());
        let pred_action = act_tree(tree, &state.features);
        if pred_action == s.action {
            correct += 1;
        }
    }

    correct as f64 / samples.len() as f64
}

/// Helper to get action from tree (without needing DistilledPolicyTree)
fn act_tree(node: &TreeNode, features: &[f64]) -> usize {
    match node {
        TreeNode::Leaf { action } => *action,
        TreeNode::Split {
            feature_index,
            threshold,
            left,
            right,
        } => {
            let x = features.get(*feature_index).copied().unwrap_or(0.0);
            if x <= *threshold {
                act_tree(left, features)
            } else {
                act_tree(right, features)
            }
        }
    }
}

/// Compute per-action accuracy
pub fn compute_per_action_accuracy(
    tree: &TreeNode,
    samples: &[Sample],
    n_actions: usize,
) -> Vec<f64> {
    let mut per_action_correct = vec![0usize; n_actions];
    let mut per_action_total = vec![0usize; n_actions];

    for s in samples {
        if s.action < n_actions {
            per_action_total[s.action] += 1;

            let state = State::new(s.features.clone());
            let pred = act_tree(tree, &state.features);

            if pred == s.action {
                per_action_correct[s.action] += 1;
            }
        }
    }

    let mut accuracies = Vec::new();
    for i in 0..n_actions {
        let acc = if per_action_total[i] > 0 {
            per_action_correct[i] as f64 / per_action_total[i] as f64
        } else {
            0.0
        };
        accuracies.push(acc);
    }

    accuracies
}

/// Split samples into train and eval sets (80/20)
pub fn split_train_eval(samples: Vec<Sample>) -> (Vec<Sample>, Vec<Sample>) {
    let split_idx = (samples.len() as f64 * 0.8) as usize;
    let mut samples = samples;

    let eval_samples = samples.split_off(split_idx);
    (samples, eval_samples)
}

/// Infer feature metadata from samples (for DoseToxEnv)
pub fn infer_features(samples: &[Sample]) -> Vec<DistillFeature> {
    if samples.is_empty() {
        return vec![];
    }

    let n_features = samples[0].features.len();

    // Default names for DoseToxEnv (4 features)
    let default_names = vec!["ANC", "tumour_size", "cycle", "prev_dose"];

    let mut features = Vec::new();

    for idx in 0..n_features {
        let name = default_names
            .get(idx)
            .map(|&s| s.to_string())
            .unwrap_or_else(|| format!("feature_{}", idx));

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for s in samples {
            if let Some(&val) = s.features.get(idx) {
                min = min.min(val);
                max = max.max(val);
            }
        }

        features.push(DistillFeature {
            name,
            index: idx,
            min,
            max,
        });
    }

    features
}

// =============================================================================
// End-to-End Distillation
// =============================================================================

/// Distill a policy into a decision tree
pub fn distill_policy(
    env: &mut dyn RLEnv,
    policy: &RLPolicyHandle,
    cfg: &DistillConfig,
) -> anyhow::Result<(DistilledPolicyTree, DistillReport)> {
    // 1. Sample policy behavior
    let samples = sample_policy_behavior(env, policy, cfg)?;
    let n_actions = policy.n_actions;

    if samples.is_empty() {
        anyhow::bail!("No samples collected from policy");
    }

    // 2. Split into train and eval
    let (train_samples, eval_samples) = split_train_eval(samples);

    // 3. Train decision tree
    let tree = train_decision_tree(&train_samples, n_actions, cfg);

    // 4. Compute metrics
    let train_acc = compute_accuracy(&tree, &train_samples);
    let eval_acc = compute_accuracy(&tree, &eval_samples);
    let per_action = compute_per_action_accuracy(&tree, &eval_samples, n_actions);

    let tree_depth = tree_depth(&tree);
    let n_nodes = tree_n_nodes(&tree);

    // 5. Infer feature metadata
    let all_samples: Vec<Sample> = train_samples
        .into_iter()
        .chain(eval_samples.clone())
        .collect();
    let features = if cfg.features.is_empty() {
        infer_features(&all_samples)
    } else {
        cfg.features.clone()
    };

    let report = DistillReport {
        n_train_samples: all_samples.len(),
        train_accuracy: train_acc,
        eval_accuracy: eval_acc,
        per_action_accuracy: per_action,
        tree_depth,
        n_nodes,
    };

    let distilled = DistilledPolicyTree {
        n_actions,
        features,
        root: tree,
    };

    Ok((distilled, report))
}

/// Helper: get tree depth
fn tree_depth(node: &TreeNode) -> usize {
    match node {
        TreeNode::Leaf { .. } => 1,
        TreeNode::Split { left, right, .. } => {
            1 + std::cmp::max(tree_depth(left), tree_depth(right))
        }
    }
}

/// Helper: count tree nodes
fn tree_n_nodes(node: &TreeNode) -> usize {
    match node {
        TreeNode::Leaf { .. } => 1,
        TreeNode::Split { left, right, .. } => 1 + tree_n_nodes(left) + tree_n_nodes(right),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test_disabled)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_creation() {
        let sample = Sample {
            features: vec![0.5, 1.0, 0.3],
            action: 2,
        };

        assert_eq!(sample.features.len(), 3);
        assert_eq!(sample.action, 2);
    }

    #[test]
    fn test_is_pure() {
        let pure_samples = vec![
            Sample {
                features: vec![1.0],
                action: 1,
            },
            Sample {
                features: vec![2.0],
                action: 1,
            },
        ];

        assert!(is_pure(&pure_samples));

        let mixed_samples = vec![
            Sample {
                features: vec![1.0],
                action: 1,
            },
            Sample {
                features: vec![2.0],
                action: 2,
            },
        ];

        assert!(!is_pure(&mixed_samples));
    }

    #[test]
    fn test_majority_class() {
        let samples = vec![
            Sample {
                features: vec![1.0],
                action: 1,
            },
            Sample {
                features: vec![2.0],
                action: 2,
            },
            Sample {
                features: vec![3.0],
                action: 1,
            },
        ];

        assert_eq!(majority_class(&samples, 3), 1);
    }

    #[test]
    fn test_gini_impurity() {
        // Pure node
        let counts = vec![10, 0, 0];
        assert_eq!(gini_impurity(&counts), 0.0);

        // Maximum impurity (uniform)
        let counts = vec![5, 5];
        assert_eq!(gini_impurity(&counts), 0.5);
    }

    #[test]
    fn test_tree_learning_simple() {
        // Simple rule: if x0 > 0.5 then action 1 else action 0
        let samples = vec![
            Sample {
                features: vec![0.2],
                action: 0,
            },
            Sample {
                features: vec![0.3],
                action: 0,
            },
            Sample {
                features: vec![0.7],
                action: 1,
            },
            Sample {
                features: vec![0.8],
                action: 1,
            },
        ];

        let cfg = DistillConfig {
            n_episodes: 1,
            max_steps_per_episode: 4,
            max_depth: 2,
            min_samples_leaf: 1,
            features: vec![],
        };

        let tree = train_decision_tree(&samples, 2, &cfg);

        // Verify it learned the rule
        let acc = compute_accuracy(&tree, &samples);
        assert!(acc >= 0.9, "Accuracy {} should be >= 0.9", acc);
    }

    #[test]
    fn test_tree_act() {
        let tree = TreeNode::Split {
            feature_index: 0,
            threshold: 0.5,
            left: Box::new(TreeNode::Leaf { action: 0 }),
            right: Box::new(TreeNode::Leaf { action: 1 }),
        };

        let state1 = State::new(vec![0.3]);
        let state2 = State::new(vec![0.7]);

        assert_eq!(act_tree(&tree, &state1.features), 0);
        assert_eq!(act_tree(&tree, &state2.features), 1);
    }

    #[test]
    fn test_distilled_policy_tree_act() {
        let tree = DistilledPolicyTree {
            n_actions: 2,
            features: vec![],
            root: TreeNode::Split {
                feature_index: 0,
                threshold: 0.5,
                left: Box::new(TreeNode::Leaf { action: 0 }),
                right: Box::new(TreeNode::Leaf { action: 1 }),
            },
        };

        let state1 = State::new(vec![0.3]);
        let state2 = State::new(vec![0.7]);

        assert_eq!(tree.act(&state1), 0);
        assert_eq!(tree.act(&state2), 1);
    }

    #[test]
    fn test_tree_depth_and_nodes() {
        let tree = TreeNode::Split {
            feature_index: 0,
            threshold: 0.5,
            left: Box::new(TreeNode::Leaf { action: 0 }),
            right: Box::new(TreeNode::Split {
                feature_index: 1,
                threshold: 0.3,
                left: Box::new(TreeNode::Leaf { action: 1 }),
                right: Box::new(TreeNode::Leaf { action: 2 }),
            }),
        };

        assert_eq!(tree_depth(&tree), 3);
        assert_eq!(tree_n_nodes(&tree), 5);
    }

    #[test]
    fn test_infer_features() {
        let samples = vec![
            Sample {
                features: vec![0.1, 0.5],
                action: 0,
            },
            Sample {
                features: vec![0.9, 0.3],
                action: 1,
            },
        ];

        let features = infer_features(&samples);

        assert_eq!(features.len(), 2);
        assert_eq!(features[0].name, "ANC");
        assert_eq!(features[0].min, 0.1);
        assert_eq!(features[0].max, 0.9);
    }
}
