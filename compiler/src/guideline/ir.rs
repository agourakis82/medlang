// Week 37: Guideline Intermediate Representation
//
// Defines the IR for clinical guidelines, supporting dose adjustment rules
// extracted from RL policies and other clinical decision logic.

use serde::{Deserialize, Serialize};

/// Reference to a clinical value (lab, vital, derived metric)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GuidelineValueRef {
    /// Absolute Neutrophil Count
    Anc,
    /// Tumour size ratio (relative to baseline)
    TumourRatio,
    /// Previous dose (mg)
    PrevDose,
    /// Current cycle index
    CycleIndex,
    /// Generic lab value by name
    Lab(String),
    /// ECOG performance status
    EcogStatus,
}

/// Comparison operators for numeric conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CmpOp {
    Lt, // <
    Le, // <=
    Gt, // >
    Ge, // >=
    Eq, // ==
    Ne, // !=
}

impl CmpOp {
    /// Get the symbol representation
    pub fn symbol(&self) -> &'static str {
        match self {
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
        }
    }

    /// Get the negation of this operator
    pub fn negate(&self) -> Self {
        match self {
            CmpOp::Lt => CmpOp::Ge,
            CmpOp::Le => CmpOp::Gt,
            CmpOp::Gt => CmpOp::Le,
            CmpOp::Ge => CmpOp::Lt,
            CmpOp::Eq => CmpOp::Ne,
            CmpOp::Ne => CmpOp::Eq,
        }
    }
}

/// Guideline condition expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GuidelineExpr {
    /// Always true
    True,

    /// Always false
    False,

    /// Logical AND of multiple conditions
    And(Vec<GuidelineExpr>),

    /// Logical OR of multiple conditions
    Or(Vec<GuidelineExpr>),

    /// Logical NOT
    Not(Box<GuidelineExpr>),

    /// Numeric comparison
    Compare {
        lhs: GuidelineValueRef,
        op: CmpOp,
        rhs: f64,
    },
}

impl GuidelineExpr {
    /// Create an AND expression, flattening nested ANDs
    pub fn and(exprs: Vec<GuidelineExpr>) -> Self {
        if exprs.is_empty() {
            return GuidelineExpr::True;
        }
        if exprs.len() == 1 {
            return exprs.into_iter().next().unwrap();
        }

        // Flatten nested ANDs
        let mut flattened = Vec::new();
        for expr in exprs {
            match expr {
                GuidelineExpr::And(inner) => flattened.extend(inner),
                other => flattened.push(other),
            }
        }

        GuidelineExpr::And(flattened)
    }

    /// Create an OR expression, flattening nested ORs
    pub fn or(exprs: Vec<GuidelineExpr>) -> Self {
        if exprs.is_empty() {
            return GuidelineExpr::False;
        }
        if exprs.len() == 1 {
            return exprs.into_iter().next().unwrap();
        }

        // Flatten nested ORs
        let mut flattened = Vec::new();
        for expr in exprs {
            match expr {
                GuidelineExpr::Or(inner) => flattened.extend(inner),
                other => flattened.push(other),
            }
        }

        GuidelineExpr::Or(flattened)
    }

    /// Simplify the expression
    pub fn simplify(&self) -> Self {
        match self {
            GuidelineExpr::True | GuidelineExpr::False => self.clone(),
            GuidelineExpr::Compare { .. } => self.clone(),

            GuidelineExpr::And(exprs) => {
                let simplified: Vec<_> = exprs.iter().map(|e| e.simplify()).collect();
                if simplified.iter().any(|e| matches!(e, GuidelineExpr::False)) {
                    GuidelineExpr::False
                } else {
                    let non_true: Vec<_> = simplified
                        .into_iter()
                        .filter(|e| !matches!(e, GuidelineExpr::True))
                        .collect();
                    GuidelineExpr::and(non_true)
                }
            }

            GuidelineExpr::Or(exprs) => {
                let simplified: Vec<_> = exprs.iter().map(|e| e.simplify()).collect();
                if simplified.iter().any(|e| matches!(e, GuidelineExpr::True)) {
                    GuidelineExpr::True
                } else {
                    let non_false: Vec<_> = simplified
                        .into_iter()
                        .filter(|e| !matches!(e, GuidelineExpr::False))
                        .collect();
                    GuidelineExpr::or(non_false)
                }
            }

            GuidelineExpr::Not(inner) => {
                let simplified = inner.simplify();
                match simplified {
                    GuidelineExpr::True => GuidelineExpr::False,
                    GuidelineExpr::False => GuidelineExpr::True,
                    GuidelineExpr::Not(inner) => *inner,
                    other => GuidelineExpr::Not(Box::new(other)),
                }
            }
        }
    }
}

/// Dose action kinds
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DoseActionKind {
    /// Set absolute dose in mg
    SetAbsoluteDoseMg(f64),

    /// Hold dose (0 mg)
    HoldDose,

    /// Reduce dose by percentage
    ReduceDosePercent(f64),

    /// Increase dose by percentage
    IncreaseDosePercent(f64),
}

/// Guideline action
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GuidelineAction {
    /// Dose adjustment action
    DoseAction(DoseActionKind),

    /// Order lab tests
    OrderLabs(Vec<String>),

    /// Recommend imaging
    RecommendImaging(String),

    /// General recommendation
    Recommend(String),
}

/// A single guideline rule: if condition then action
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuidelineRule {
    pub condition: GuidelineExpr,
    pub action: GuidelineAction,
    pub description: Option<String>,
    pub priority: Option<u32>,
}

/// Metadata for a guideline
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuidelineMeta {
    pub id: String,
    pub version: String,
    pub title: String,
    pub description: String,
    pub population: String,
    pub line_of_therapy: Option<String>,
    pub regimen_name: Option<String>,
    pub tumor_type: Option<String>,
}

/// Complete guideline artifact
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuidelineArtifact {
    pub meta: GuidelineMeta,
    pub rules: Vec<GuidelineRule>,
}

impl GuidelineArtifact {
    /// Create a new guideline artifact
    pub fn new(meta: GuidelineMeta) -> Self {
        Self {
            meta,
            rules: Vec::new(),
        }
    }

    /// Add a rule to the guideline
    pub fn add_rule(&mut self, rule: GuidelineRule) {
        self.rules.push(rule);
    }

    /// Sort rules by priority (higher priority first)
    pub fn sort_by_priority(&mut self) {
        self.rules
            .sort_by_key(|r| std::cmp::Reverse(r.priority.unwrap_or(0)));
    }
}

/// Host-side version of dose guideline metadata (from MedLang)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoseGuidelineMetaHost {
    pub id: String,
    pub version: String,
    pub title: String,
    pub description: String,
    pub population: String,
}

impl From<DoseGuidelineMetaHost> for GuidelineMeta {
    fn from(meta: DoseGuidelineMetaHost) -> Self {
        GuidelineMeta {
            id: meta.id,
            version: meta.version,
            title: meta.title,
            description: meta.description,
            population: meta.population,
            line_of_therapy: None,
            regimen_name: None,
            tumor_type: None,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmp_op_symbol() {
        assert_eq!(CmpOp::Lt.symbol(), "<");
        assert_eq!(CmpOp::Le.symbol(), "<=");
        assert_eq!(CmpOp::Gt.symbol(), ">");
        assert_eq!(CmpOp::Ge.symbol(), ">=");
    }

    #[test]
    fn test_cmp_op_negate() {
        assert_eq!(CmpOp::Lt.negate(), CmpOp::Ge);
        assert_eq!(CmpOp::Le.negate(), CmpOp::Gt);
        assert_eq!(CmpOp::Gt.negate(), CmpOp::Le);
        assert_eq!(CmpOp::Ge.negate(), CmpOp::Lt);
    }

    #[test]
    fn test_guideline_expr_and() {
        let expr = GuidelineExpr::and(vec![
            GuidelineExpr::True,
            GuidelineExpr::Compare {
                lhs: GuidelineValueRef::Anc,
                op: CmpOp::Lt,
                rhs: 0.5,
            },
        ]);

        match expr {
            GuidelineExpr::And(exprs) => assert_eq!(exprs.len(), 2),
            _ => panic!("Expected And"),
        }
    }

    #[test]
    fn test_guideline_expr_simplify() {
        // AND with False -> False
        let expr = GuidelineExpr::And(vec![GuidelineExpr::True, GuidelineExpr::False]);
        assert_eq!(expr.simplify(), GuidelineExpr::False);

        // AND with True -> remaining condition
        let cmp = GuidelineExpr::Compare {
            lhs: GuidelineValueRef::Anc,
            op: CmpOp::Lt,
            rhs: 0.5,
        };
        let expr = GuidelineExpr::And(vec![GuidelineExpr::True, cmp.clone()]);
        assert_eq!(expr.simplify(), cmp);
    }

    #[test]
    fn test_guideline_artifact_creation() {
        let meta = GuidelineMeta {
            id: "test".to_string(),
            version: "1.0".to_string(),
            title: "Test Guideline".to_string(),
            description: "Test".to_string(),
            population: "Test pop".to_string(),
            line_of_therapy: None,
            regimen_name: None,
            tumor_type: None,
        };

        let artifact = GuidelineArtifact::new(meta);
        assert_eq!(artifact.rules.len(), 0);
    }

    #[test]
    fn test_dose_action_kinds() {
        let hold = DoseActionKind::HoldDose;
        let set = DoseActionKind::SetAbsoluteDoseMg(100.0);
        let reduce = DoseActionKind::ReduceDosePercent(25.0);

        assert!(matches!(hold, DoseActionKind::HoldDose));
        assert!(matches!(set, DoseActionKind::SetAbsoluteDoseMg(_)));
        assert!(matches!(reduce, DoseActionKind::ReduceDosePercent(_)));
    }
}
