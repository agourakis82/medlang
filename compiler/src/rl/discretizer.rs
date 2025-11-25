// Week 31: State Discretization for Tabular RL
//
// Provides discretization schemes to convert continuous state spaces into
// discrete state indices for tabular RL algorithms like Q-learning.

use crate::rl::core::State;

/// Trait for discretizing continuous states into discrete indices
pub trait StateDiscretizer {
    /// Get total number of discrete states
    fn num_states(&self) -> usize;

    /// Convert a continuous state to a discrete state index
    fn state_index(&self, state: &State) -> usize;

    /// Get the dimensionality of states this discretizer handles
    fn state_dim(&self) -> usize;
}

/// Box (uniform grid) discretizer
///
/// Divides each dimension of the state space into uniform bins.
/// Total number of states is the product of bins per dimension.
#[derive(Debug, Clone)]
pub struct BoxDiscretizer {
    /// Number of bins per dimension
    pub bins_per_dim: Vec<usize>,

    /// Minimum value for each dimension
    pub min_vals: Vec<f64>,

    /// Maximum value for each dimension
    pub max_vals: Vec<f64>,

    /// Total number of discrete states
    pub total_states: usize,
}

impl BoxDiscretizer {
    /// Create a new box discretizer
    ///
    /// # Arguments
    /// * `bins_per_dim` - Number of bins for each state dimension
    /// * `min_vals` - Minimum value for each dimension
    /// * `max_vals` - Maximum value for each dimension
    pub fn new(bins_per_dim: Vec<usize>, min_vals: Vec<f64>, max_vals: Vec<f64>) -> Self {
        assert_eq!(bins_per_dim.len(), min_vals.len());
        assert_eq!(bins_per_dim.len(), max_vals.len());

        let total_states = bins_per_dim.iter().product();

        Self {
            bins_per_dim,
            min_vals,
            max_vals,
            total_states,
        }
    }

    /// Create a uniform discretizer (same number of bins for all dimensions)
    pub fn uniform(state_dim: usize, bins: usize, min_val: f64, max_val: f64) -> Self {
        Self::new(
            vec![bins; state_dim],
            vec![min_val; state_dim],
            vec![max_val; state_dim],
        )
    }

    /// Discretize a single dimension value
    fn discretize_dim(&self, dim: usize, value: f64) -> usize {
        let min = self.min_vals[dim];
        let max = self.max_vals[dim];
        let bins = self.bins_per_dim[dim];

        // Clamp value to [min, max]
        let clamped = value.clamp(min, max);

        // Map to [0, bins-1]
        let normalized = (clamped - min) / (max - min);
        let bin = (normalized * bins as f64).floor() as usize;

        // Handle edge case where value == max
        bin.min(bins - 1)
    }
}

impl StateDiscretizer for BoxDiscretizer {
    fn num_states(&self) -> usize {
        self.total_states
    }

    fn state_index(&self, state: &State) -> usize {
        assert_eq!(state.features.len(), self.bins_per_dim.len());

        let mut index = 0;
        let mut multiplier = 1;

        // Convert multi-dimensional bin coordinates to flat index
        for dim in 0..state.features.len() {
            let bin = self.discretize_dim(dim, state.features[dim]);
            index += bin * multiplier;
            multiplier *= self.bins_per_dim[dim];
        }

        index
    }

    fn state_dim(&self) -> usize {
        self.bins_per_dim.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_discretizer() {
        let disc = BoxDiscretizer::uniform(2, 10, 0.0, 1.0);

        assert_eq!(disc.num_states(), 100);
        assert_eq!(disc.state_dim(), 2);
    }

    #[test]
    fn test_discretize_corners() {
        let disc = BoxDiscretizer::new(vec![10, 10], vec![0.0, 0.0], vec![1.0, 1.0]);

        // Bottom-left corner
        let s1 = State::new(vec![0.0, 0.0]);
        assert_eq!(disc.state_index(&s1), 0);

        // Top-right corner
        let s2 = State::new(vec![1.0, 1.0]);
        assert_eq!(disc.state_index(&s2), 99);

        // Middle
        let s3 = State::new(vec![0.5, 0.5]);
        assert_eq!(disc.state_index(&s3), 55);
    }

    #[test]
    fn test_discretize_clamping() {
        let disc = BoxDiscretizer::uniform(1, 5, 0.0, 1.0);

        // Values outside bounds should be clamped
        let s1 = State::new(vec![-0.5]);
        assert_eq!(disc.state_index(&s1), 0);

        let s2 = State::new(vec![1.5]);
        assert_eq!(disc.state_index(&s2), 4);
    }

    #[test]
    fn test_non_uniform_bins() {
        let disc = BoxDiscretizer::new(
            vec![5, 10], // 5 bins in dim 0, 10 bins in dim 1
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        );

        assert_eq!(disc.num_states(), 50);

        // Test specific mappings
        let s1 = State::new(vec![0.0, 0.0]);
        assert_eq!(disc.state_index(&s1), 0);

        let s2 = State::new(vec![0.2, 0.1]); // bin (1, 1)
        let expected = 1 + 1 * 5; // bin_0 + bin_1 * bins_per_dim[0]
        assert_eq!(disc.state_index(&s2), expected);
    }

    #[test]
    fn test_bin_boundaries() {
        let disc = BoxDiscretizer::uniform(1, 4, 0.0, 1.0);

        // Test boundaries: [0.0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]
        assert_eq!(disc.state_index(&State::new(vec![0.0])), 0);
        assert_eq!(disc.state_index(&State::new(vec![0.24])), 0);
        assert_eq!(disc.state_index(&State::new(vec![0.25])), 1);
        assert_eq!(disc.state_index(&State::new(vec![0.49])), 1);
        assert_eq!(disc.state_index(&State::new(vec![0.5])), 2);
        assert_eq!(disc.state_index(&State::new(vec![0.74])), 2);
        assert_eq!(disc.state_index(&State::new(vec![0.75])), 3);
        assert_eq!(disc.state_index(&State::new(vec![0.99])), 3);
        assert_eq!(disc.state_index(&State::new(vec![1.0])), 3);
    }
}
