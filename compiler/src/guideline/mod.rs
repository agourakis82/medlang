// Week 37: Guideline Module
//
// Provides intermediate representation for clinical guidelines and
// translation from RL policies to guideline artifacts.

pub mod ir;

pub use ir::{
    CmpOp, DoseActionKind, DoseGuidelineMetaHost, GuidelineAction, GuidelineArtifact,
    GuidelineExpr, GuidelineMeta, GuidelineRule, GuidelineValueRef,
};
