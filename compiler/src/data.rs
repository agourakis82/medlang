/// Data module for MedLang compiler
/// 
/// This module provides data structures and utilities for handling
/// clinical trial data, including subject information, observations,
/// and time-varying covariates.

pub mod trial;
pub mod analysis;

pub use trial::{TrialDataset, TrialRow};
pub use analysis::{
    TrialAnalysisResults, EndpointAnalysisResult, EndpointComparison, EndpointResult,
    BinaryArmResult, TimeToEventArmResult, analyze_trial, compare_trials,
    TrialComparison, OverallMetrics, ArmComparison, ChiSquareResult
};

/// Placeholder for data module
/// To be expanded with data loading, validation, and transformation utilities
pub fn placeholder() {}
