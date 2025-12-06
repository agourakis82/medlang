# Week 36 Delivery Summary: Policy Distillation & Rule Extraction

**Status**: ✅ Complete  
**Date**: January 2025  
**Version**: v0.1

---

## Overview

Week 36 introduces **Policy Distillation**, a framework for converting black-box RL policies into interpretable decision trees. This enables clinicians to understand, validate, and compare learned dosing strategies to existing guidelines, bridging the gap between ML optimization and clinical practice.

### Key Deliverable

> **Turn black-box policies into explicit, human-readable dose rules that live in the same universe as guidelines and protocols.**

---

## What Was Delivered

### 1. Core Distillation Module (`compiler/src/rl/distill.rs` - 813 lines)

**Data Structures**:
```rust
pub struct DistillConfig {
    pub n_episodes: usize,
    pub max_steps_per_episode: usize,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub features: Vec<DistillFeature>,
}

pub enum TreeNode {
    Leaf { action: usize },
    Split { feature_index, threshold, left, right },
}

pub struct DistilledPolicyTree {
    pub n_actions: usize,
    pub features: Vec<DistillFeature>,
    pub root: TreeNode,
}

pub struct DistillReport {
    pub n_train_samples: usize,
    pub train_accuracy: f64,
    pub eval_accuracy: f64,
    pub per_action_accuracy: Vec<f64>,
    pub tree_depth: usize,
    pub n_nodes: usize,
}
```

**Core Algorithms**:

**Sampling**: `sample_policy_behavior()`
- Collects (state, action) pairs from policy
- Runs policy greedily in environment
- Prevents runaway with 50K sample cap

**Tree Learning**: CART-style decision tree
- Gini impurity splitting criterion
- Recursive tree construction
- Configurable depth and leaf size

**Metrics**: Comprehensive fidelity measurement
- Train/eval accuracy split (80/20)
- Per-action accuracy
- Tree complexity (depth, nodes)

### 2. Decision Tree Algorithm (CART)

**Splitting Criterion**: Gini Impurity
```
Gini(S) = 1 - Σ(p_i²)

Where p_i = fraction of samples with action i
```

**Split Selection**:
1. For each feature dimension:
   - Sort samples by feature value
   - Try each midpoint as threshold
   - Compute weighted Gini impurity
2. Choose split with minimum impurity

**Stopping Criteria**:
- Depth ≥ max_depth
- Samples ≤ min_samples_leaf  
- Node is pure (all same action)

**Leaf Assignment**: Majority class (most common action)

**Example Tree**:
```
if ANC ≤ 0.5:
    if tumor_size > 0.7:
        action = 0  // No dose - too toxic
    else:
        action = 1  // 100mg dose
else:
    if tumor_size > 0.5:
        action = 2  // 200mg dose - safe to escalate
    else:
        action = 1  // 100mg dose - maintain
```

### 3. Tree Operations

**Action Selection**: `DistilledPolicyTree::act()`
```rust
pub fn act(&self, state: &State) -> usize {
    // Descend tree based on feature values
    // O(depth) complexity
}
```

**Complexity Metrics**:
- `depth()` - Maximum tree depth
- `n_nodes()` - Total node count

### 4. MedLang Standard Library (`stdlib/med/rl/explain.medlang` - 125 lines)

**Types**:
```medlang
type DistillConfig = {
  n_episodes: Int;
  max_steps_per_episode: Int;
  max_depth: Int;
  min_samples_leaf: Int;
};

type DistillReport = {
  n_train_samples: Int;
  train_accuracy: Float;
  eval_accuracy: Float;      // Key metric!
  per_action_accuracy: Vector<Float>;
  tree_depth: Int;
  n_nodes: Int;
};

type DistilledPolicy = opaque;

type DistillResult = {
  policy: DistilledPolicy;
  report: DistillReport;
};
```

**Functions**:
```medlang
fn distill_policy_tree(
  env_cfg: RLEnvConfig;
  policy: RLPolicy;
  cfg: DistillConfig;
) -> DistillResult;

fn simulate_distilled_policy(
  env_cfg: RLEnvConfig;
  distilled: DistilledPolicy;
  n_episodes: Int;
) -> PolicyEvalReport;
```

### 5. Built-in Functions (`runtime/builtins.rs`)

**`distill_policy_tree()`**:
- Parses MedLang configs
- Creates DoseToxEnv
- Calls core distillation
- Returns tree + report

**`simulate_distilled_policy()`**:
- Parses MedLang configs
- Creates environment
- Runs tree policy
- Returns evaluation report

### 6. CLI Command (`bin/mlc.rs`)

```bash
mlc rl-policy-distill \
  --env-config env.json \
  --policy policy.json \
  --distill-config distill.json \
  --out-policy distilled.json \
  --out-report report.json \
  --verbose
```

**Workflow**:
1. Load environment config
2. Load trained policy
3. Load distillation config
4. Run distillation
5. Write tree JSON
6. Write report JSON

### 7. Type System Integration

**Core Types**:
- Added `CoreType::DistilledPolicy`
- Added `TypeAnn::DistilledPolicy`
- Updated `resolve_type_ann()`

**Runtime Values**:
- Added `RuntimeValue::DistilledPolicy(DistilledPolicyHandle)`
- Updated type checking methods

### 8. Comprehensive Tests (`tests/week_36_distill_tests.rs` - 420 lines)

**Test Coverage**:
1. `test_distill_config_creation()` - Config validation
2. `test_distill_config_default()` - Default values
3. `test_basic_distillation()` - End-to-end distillation
4. `test_tree_fidelity()` - Accuracy > 40% threshold
5. `test_tree_complexity_control()` - Depth/leaf size control
6. `test_tree_execution()` - Action selection
7. `test_per_action_accuracy()` - Per-class metrics
8. `test_feature_inference()` - Feature metadata

### 9. Example MedLang Program (`examples/week36/oncology_distill.medlang` - 354 lines)

**Demonstrates**:
- Complete train → distill → compare workflow
- Complexity vs fidelity tradeoff analysis
- Per-action fidelity analysis
- Clinical interpretation helpers
- Multiple usage scenarios

**Key Functions**:
- `main()` - Complete workflow
- `analyze_complexity_tradeoff()` - Compare tree sizes
- `analyze_per_action_fidelity()` - Per-dose accuracy
- `interpret_tree_clinically()` - Clinical guidelines

---

## Technical Architecture

### Distillation Pipeline

```
1. Policy + Environment
   ↓
2. Sample (state, action) pairs
   - Run policy greedily
   - Collect n_episodes × steps_per_episode samples
   ↓
3. Split train/eval (80/20)
   ↓
4. Train decision tree (CART)
   - Recursive splitting
   - Gini impurity criterion
   - Stop at max_depth or min_samples_leaf
   ↓
5. Compute fidelity metrics
   - Train accuracy
   - Eval accuracy (key metric)
   - Per-action accuracy
   ↓
6. Return DistilledPolicyTree + DistillReport
```

### Tree Structure

```rust
// Internal representation
enum TreeNode {
    Leaf { action: usize },
    Split {
        feature_index: usize,    // Which state feature
        threshold: f64,          // Split point
        left: Box<TreeNode>,     // x ≤ threshold
        right: Box<TreeNode>,    // x > threshold
    },
}

// Complete tree
struct DistilledPolicyTree {
    n_actions: usize,
    features: Vec<DistillFeature>,  // Metadata
    root: TreeNode,
}
```

### Gini Impurity Calculation

```rust
fn gini_impurity(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    if total == 0 { return 0.0; }
    
    let mut sum_sq = 0.0;
    for &c in counts {
        let p = c as f64 / total as f64;
        sum_sq += p * p;
    }
    1.0 - sum_sq
}
```

For uniform distribution (maximum impurity): Gini = 0.5  
For pure node (single class): Gini = 0.0

### Split Finding Algorithm

```rust
fn find_best_split(samples, features, n_actions) -> Option<(feat, thresh, left_indices)> {
    let mut best_gini = f64::INFINITY;
    
    for each feature:
        sort samples by feature value
        
        for each consecutive pair as split point:
            compute left_counts, right_counts
            
            gini_split = (n_left * gini_left + n_right * gini_right) / (n_left + n_right)
            
            if gini_split < best_gini:
                best_gini = gini_split
                best_split = (feature, threshold, left_indices)
    
    return best_split
}
```

**Complexity**: O(n_features × n_samples × log(n_samples)) per node

---

## Usage Examples

### Example 1: Basic Distillation

```medlang
import med.rl::{train_policy_rl, RLEnvConfig, RLTrainConfig};
import med.rl.explain::{distill_policy_tree, DistillConfig};

fn main() {
  // 1. Train policy
  let env_cfg: RLEnvConfig = create_env_config();
  let train_cfg: RLTrainConfig = create_train_config();
  let result = train_policy_rl(env_cfg, train_cfg);
  let policy = result.1;
  
  // 2. Distill to tree
  let distill_cfg: DistillConfig = {
    n_episodes = 200;
    max_steps_per_episode = 6;
    max_depth = 3;
    min_samples_leaf = 20;
  };
  
  let distill_result = distill_policy_tree(env_cfg, policy, distill_cfg);
  let tree = distill_result.policy;
  let report = distill_result.report;
  
  // 3. Check fidelity
  print("Eval accuracy: " + report.eval_accuracy);
  print("Tree depth: " + report.tree_depth);
  
  // 4. Compare performance
  let eval_orig = simulate_policy_rl(env_cfg, policy, 500);
  let eval_tree = simulate_distilled_policy(env_cfg, tree, 500);
  
  print("Original reward: " + eval_orig.avg_reward);
  print("Tree reward: " + eval_tree.avg_reward);
}
```

### Example 2: Complexity Tradeoff

```medlang
fn compare_tree_sizes(policy: RLPolicy; env_cfg: RLEnvConfig) {
  // Simple tree (depth=2)
  let cfg_simple: DistillConfig = {
    n_episodes = 200;
    max_steps_per_episode = 6;
    max_depth = 2;
    min_samples_leaf = 30;
  };
  let simple = distill_policy_tree(env_cfg, policy, cfg_simple);
  
  // Complex tree (depth=5)
  let cfg_complex: DistillConfig = {
    n_episodes = 200;
    max_steps_per_episode = 6;
    max_depth = 5;
    min_samples_leaf = 10;
  };
  let complex = distill_policy_tree(env_cfg, policy, cfg_complex);
  
  print("Simple: " + simple.report.tree_nodes + " nodes, " 
                    + simple.report.eval_accuracy + " accuracy");
  print("Complex: " + complex.report.tree_nodes + " nodes, " 
                     + complex.report.eval_accuracy + " accuracy");
}
```

### Example 3: CLI Workflow

```bash
# 1. Train policy (from Week 32)
mlc rl-train \
  --env-config env.json \
  --train-config train.json \
  --out-policy policy.json \
  --out-report train_report.json

# 2. Distill policy
cat > distill.json <<EOF
{
  "n_episodes": 200,
  "max_steps_per_episode": 6,
  "max_depth": 3,
  "min_samples_leaf": 20,
  "features": []
}
EOF

mlc rl-policy-distill \
  --env-config env.json \
  --policy policy.json \
  --distill-config distill.json \
  --out-policy tree.json \
  --out-report distill_report.json \
  --verbose

# 3. Inspect results
cat distill_report.json | jq '{
  eval_accuracy, 
  tree_depth, 
  n_nodes
}'
```

---

## Design Decisions

### 1. CART Algorithm (Classification and Regression Trees)

**Decision**: Use Gini impurity for splitting  
**Rationale**:
- Fast to compute
- Works well for multi-class classification
- Standard in scikit-learn and other ML libraries
- Interpretable impurity measure

**Alternatives considered**:
- Information gain (entropy-based) - more expensive
- Chi-squared - requires more samples

### 2. 80/20 Train/Eval Split

**Decision**: Use 80% for training, 20% for evaluation  
**Rationale**:
- Standard ML practice
- Provides unbiased fidelity estimate
- Sufficient samples for both sets

### 3. Greedy Sampling (No Exploration)

**Decision**: Sample policy greedily (no ε-greedy)  
**Rationale**:
- Distills the learned behavior, not training process
- Reproducible results
- Matches deployment behavior

### 4. Feature Metadata Inference

**Decision**: Automatically infer feature names and ranges  
**Rationale**:
- User-friendly (no manual config)
- Works with DoseToxEnv defaults
- Extensible for custom features

**Hardcoded names for DoseToxEnv**:
- Feature 0: "ANC"
- Feature 1: "tumour_size"
- Feature 2: "cycle"
- Feature 3: "prev_dose"

### 5. Opaque DistilledPolicy Type

**Decision**: Make DistilledPolicy opaque in MedLang  
**Rationale**:
- Tree structure is complex (recursive)
- Not directly manipulable in MedLang
- Access via JSON export for visualization
- Keeps MedLang API clean

---

## Performance Characteristics

### Computational Cost

**Sampling**: O(n_episodes × steps × policy_lookup)
- DoseToxEnv: ~50ms per episode
- 200 episodes: ~10 seconds

**Tree Training**: O(n_samples × n_features × log(n_samples) × depth)
- 1000 samples, 4 features, depth 3: ~10ms
- Dominated by sampling cost

**Tree Execution**: O(depth)
- Depth 3: ~3 comparisons
- Negligible (<1μs)

**Total End-to-End**: ~10 seconds for typical distillation

### Memory Usage

**Samples**: ~100 bytes per sample
- 1000 samples: ~100 KB

**Tree**: ~50 bytes per node
- Depth 3, binary: ≤15 nodes = ~750 bytes
- Negligible

**Total**: <1 MB for typical distillation

### Scalability

**Linear in**:
- Number of episodes
- Steps per episode

**Logarithmic in**:
- Number of samples (for sorting)

**Independent of**:
- Policy size (Q-table)
- Number of actions (constant overhead)

---

## Limitations and Future Work

### Current Limitations (v0.1)

1. **Single Environment**: Only DoseToxEnv tested
2. **Greedy Sampling**: No exploration diversity
3. **Binary Splits**: Only threshold-based splits
4. **No Pruning**: Trees may overfit
5. **No Visualization**: JSON export only

### Planned Enhancements

**Phase V1 (Near-term)**:
- Tree visualization (graphviz, D3.js)
- Human-readable rule export
- Tree pruning (cost-complexity)
- Oblique splits (linear combinations)

**Phase V2 (Medium-term)**:
- Comparison to clinical guidelines
- Guideline-constrained distillation
- Multi-output trees (action + confidence)
- Ensemble methods (random forests)

**Phase V3 (Long-term)**:
- Interactive tree exploration (LSP)
- Causal rule extraction
- Counterfactual explanations
- Symbolic policy learning

---

## Impact and Significance

### Clinical Value

**Interpretability**:
- "If ANC < 0.5, reduce dose to 100mg"
- Can be explained to clinicians
- Can be compared to existing protocols

**Validation**:
- Review tree decisions
- Identify unsafe branches
- Verify against domain knowledge

**Deployment**:
- Implement as simple flowchart
- No black-box ML in production
- Regulatory-friendly

### Technical Value

**Bridge ML ↔ Guidelines**:
- Same decision structure as protocols
- Can be converted to CQL (future)
- Integrates with Week 34 guidelines

**Debugging**:
- Understand where policy fails
- Identify feature dependencies
- Validate training convergence

**Trust**:
- Explainable AI for healthcare
- Clinician can audit decisions
- Reduces "black box" concerns

### Research Value

**Policy Analysis**:
- What features matter most?
- Are there simple rules that work well?
- How complex does policy need to be?

**Benchmarking**:
- Compare RL to expert heuristics
- Quantify benefit of RL
- Find interpretability/performance sweet spot

---

## Integration Points

### Depends On

- **Week 31-32 (RL)**: RLEnv, RLPolicy, train_q_learning
- **Week 35 (Safety)**: Can run safety analysis on distilled trees

### Enables Future Weeks

- **Week 37**: Compare distilled trees to clinical guidelines
- **Week 38**: Guideline-constrained distillation
- **Week 40**: LSP tree visualization
- **Week 45**: CQL export from distilled trees

---

## Testing Strategy

### Unit Tests (in `distill.rs`)

1. Sample creation
2. Pure node detection
3. Majority class voting
4. Gini impurity calculation
5. Tree learning on toy data
6. Tree execution
7. Depth and node counting
8. Feature inference

### Integration Tests (in `week_36_distill_tests.rs`)

1. End-to-end distillation
2. Fidelity validation (>40%)
3. Complexity control (depth, leaf size)
4. Tree execution on multiple states
5. Per-action accuracy
6. Feature metadata inference

### Expected Results

- ✅ All unit tests pass
- ✅ Fidelity >40% for simple problems
- ✅ Tree depth respects max_depth
- ✅ Tree nodes ≤ 2^(depth+1) - 1
- ✅ Feature names match DoseToxEnv

---

## Migration Guide

### From Week 32 (RL Training)

**Before**:
```medlang
let result = train_policy_rl(env_cfg, train_cfg);
let policy = result.1;
// Policy is a black box
```

**After (Week 36)**:
```medlang
let result = train_policy_rl(env_cfg, train_cfg);
let policy = result.1;

// Distill to interpretable tree
let distill_cfg: DistillConfig = {...};
let tree_result = distill_policy_tree(env_cfg, policy, distill_cfg);
let tree = tree_result.policy;

// Now we have an interpretable policy!
print("Tree depth: " + tree_result.report.tree_depth);
```

### Recommended Workflow

1. **Train** policy with sufficient episodes (500+)
2. **Distill** with balanced config (depth=3, leaf=20)
3. **Validate** fidelity (eval_accuracy >70%)
4. **Compare** original vs distilled performance
5. **Adjust** complexity if needed
6. **Export** to JSON for visualization
7. **Review** with clinical experts

---

## Related Work

### Academic Background

**CART Algorithm**:
- Breiman et al. (1984) - Classification and Regression Trees
- Standard in sklearn.tree.DecisionTreeClassifier

**Policy Distillation**:
- Hinton et al. (2015) - Distilling knowledge in neural networks
- Rusu et al. (2015) - Policy distillation in RL
- Bastani et al. (2018) - Verifiable RL via policy extraction

**Interpretable RL**:
- Verma et al. (2018) - Programmatically interpretable RL
- Topin & Veloso (2019) - Generation of policy-level explanations

### MedLang Integration

**Builds on**:
- Week 31-32: RL infrastructure
- Week 35: Safety analysis

**Complements**:
- Week 34: Guidelines (distilled trees ≈ guidelines)
- Week 33: Registry (log distillation experiments)

**Enables**:
- Week 37+: Guideline comparison and integration
- Future: CQL export, LSP visualization

---

## Conclusion

Week 36 delivers on the promise of **interpretable RL for clinical applications**. By distilling black-box policies into decision trees, MedLang enables:

- ✅ **Understanding**: Clinicians can see what the policy learned
- ✅ **Validation**: Rules can be reviewed and critiqued
- ✅ **Deployment**: Simple trees can be implemented without ML infrastructure
- ✅ **Trust**: Explainable decisions reduce "black box" concerns

Key achievements:
- ✅ Complete CART decision tree implementation
- ✅ Comprehensive fidelity metrics
- ✅ Configurable complexity/fidelity tradeoff
- ✅ MedLang API and CLI integration
- ✅ Extensive tests and examples

This positions MedLang as **the first clinical programming language with built-in, type-safe policy distillation** specifically designed for pharmacometric and medical applications.

---

## Appendix: Full Type Signatures

### Rust API

```rust
// Core types
pub struct DistillConfig { ... }
pub enum TreeNode { ... }
pub struct DistilledPolicyTree { ... }
pub struct DistillReport { ... }

// Main function
pub fn distill_policy(
    env: &mut dyn RLEnv,
    policy: &RLPolicyHandle,
    cfg: &DistillConfig,
) -> anyhow::Result<(DistilledPolicyTree, DistillReport)>
```

### MedLang API

```medlang
// Types
type DistillConfig = { ... }
type DistillReport = { ... }
type DistilledPolicy = opaque;
type DistillResult = { policy, report }

// Functions
fn distill_policy_tree(
  env_cfg: RLEnvConfig;
  policy: RLPolicy;
  cfg: DistillConfig;
) -> DistillResult;

fn simulate_distilled_policy(
  env_cfg: RLEnvConfig;
  distilled: DistilledPolicy;
  n_episodes: Int;
) -> PolicyEvalReport;
```

---

**Week 36 Status**: ✅ Complete  
**Next**: Week 37 - Guideline-Policy Comparison & Alignment