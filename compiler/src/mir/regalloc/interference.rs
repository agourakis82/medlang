//! Interference Graph for Register Allocation
//!
//! The interference graph captures which virtual registers cannot share
//! the same physical register because their live ranges overlap.

use std::collections::{HashMap, HashSet};

use crate::mir::value::ValueId;

use super::lifetime::LivenessAnalysis;
use super::target::RegisterClass;

/// A node in the interference graph
#[derive(Debug, Clone)]
pub struct InterferenceNode {
    /// Virtual register
    pub value: ValueId,
    /// Register class this value needs
    pub reg_class: RegisterClass,
    /// Degree (number of neighbors)
    pub degree: usize,
    /// Spill cost
    pub spill_cost: f64,
    /// Is this node removed during simplification?
    pub removed: bool,
    /// Color (physical register) assigned
    pub color: Option<u8>,
}

impl InterferenceNode {
    /// Create a new interference node
    pub fn new(value: ValueId, reg_class: RegisterClass) -> Self {
        Self {
            value,
            reg_class,
            degree: 0,
            spill_cost: 1.0,
            removed: false,
            color: None,
        }
    }
}

/// Interference graph for register allocation
#[derive(Debug, Clone)]
pub struct InterferenceGraph {
    /// Nodes (indexed by node ID)
    pub nodes: Vec<InterferenceNode>,
    /// Mapping from ValueId to node index
    pub value_to_node: HashMap<ValueId, usize>,
    /// Adjacency list (edges)
    pub edges: Vec<HashSet<usize>>,
    /// Move-related edges (for coalescing)
    pub move_edges: Vec<(usize, usize)>,
}

impl InterferenceGraph {
    /// Build interference graph from liveness analysis
    pub fn build(liveness: &LivenessAnalysis) -> Self {
        let mut graph = Self {
            nodes: Vec::new(),
            value_to_node: HashMap::new(),
            edges: Vec::new(),
            move_edges: Vec::new(),
        };

        // Create nodes for all live intervals
        for (value, interval) in &liveness.intervals {
            let node_idx = graph.nodes.len();
            graph.value_to_node.insert(*value, node_idx);

            // Determine register class from spill weight heuristic
            // In practice, this would come from type information
            let reg_class = RegisterClass::GPR;

            let mut node = InterferenceNode::new(*value, reg_class);
            node.spill_cost = interval.spill_weight;
            graph.nodes.push(node);
            graph.edges.push(HashSet::new());
        }

        // Add interference edges for overlapping intervals
        let intervals: Vec<_> = liveness.intervals.values().collect();
        for i in 0..intervals.len() {
            for j in (i + 1)..intervals.len() {
                if intervals[i].overlaps(intervals[j]) {
                    let node_i = graph.value_to_node[&intervals[i].value];
                    let node_j = graph.value_to_node[&intervals[j].value];
                    graph.add_edge(node_i, node_j);
                }
            }
        }

        graph
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, a: usize, b: usize) {
        if a != b && !self.edges[a].contains(&b) {
            self.edges[a].insert(b);
            self.edges[b].insert(a);
            self.nodes[a].degree += 1;
            self.nodes[b].degree += 1;
        }
    }

    /// Add a move edge (for coalescing)
    pub fn add_move_edge(&mut self, a: usize, b: usize) {
        if a != b {
            self.move_edges.push((a.min(b), a.max(b)));
        }
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.edges[node]
            .iter()
            .copied()
            .filter(move |&n| !self.nodes[n].removed)
    }

    /// Get degree of a node (non-removed neighbors)
    pub fn degree(&self, node: usize) -> usize {
        self.neighbors(node).count()
    }

    /// Simplify the graph and return node ordering
    pub fn simplify_order(&self) -> Vec<usize> {
        let mut graph = self.clone();
        let num_colors = 14; // Typical number of allocatable registers
        let mut stack = Vec::new();

        // Simplify phase: repeatedly remove low-degree nodes
        loop {
            // Find a low-degree non-removed node
            let candidate = (0..graph.nodes.len())
                .filter(|&n| !graph.nodes[n].removed)
                .find(|&n| graph.degree(n) < num_colors);

            if let Some(node) = candidate {
                // Remove node
                graph.nodes[node].removed = true;
                stack.push(node);
            } else {
                // No low-degree nodes - need to spill
                // Pick node with lowest spill cost / degree ratio
                let spill_candidate = (0..graph.nodes.len())
                    .filter(|&n| !graph.nodes[n].removed)
                    .min_by(|&a, &b| {
                        let ratio_a = graph.nodes[a].spill_cost / (graph.degree(a) as f64 + 1.0);
                        let ratio_b = graph.nodes[b].spill_cost / (graph.degree(b) as f64 + 1.0);
                        ratio_a.partial_cmp(&ratio_b).unwrap()
                    });

                if let Some(node) = spill_candidate {
                    graph.nodes[node].removed = true;
                    stack.push(node);
                } else {
                    // All nodes removed
                    break;
                }
            }
        }

        stack
    }

    /// Try to coalesce move-related nodes
    pub fn coalesce(&mut self) -> usize {
        let mut coalesced = 0;
        let num_colors = 14;

        // Conservative coalescing: only coalesce if combined degree < k
        let moves = self.move_edges.clone();
        for (a, b) in moves {
            if self.nodes[a].removed || self.nodes[b].removed {
                continue;
            }

            // Check if nodes interfere
            if self.edges[a].contains(&b) {
                continue;
            }

            // Check if they're in the same register class
            if self.nodes[a].reg_class != self.nodes[b].reg_class {
                continue;
            }

            // Briggs criterion: coalesce if combined node has < k high-degree neighbors
            let combined_neighbors: HashSet<_> = self
                .neighbors(a)
                .chain(self.neighbors(b))
                .filter(|&n| n != a && n != b)
                .collect();

            let high_degree_neighbors = combined_neighbors
                .iter()
                .filter(|&&n| self.degree(n) >= num_colors)
                .count();

            if high_degree_neighbors < num_colors {
                // Coalesce: merge b into a
                self.merge_nodes(a, b);
                coalesced += 1;
            }
        }

        coalesced
    }

    /// Merge node b into node a
    fn merge_nodes(&mut self, a: usize, b: usize) {
        // Add all of b's edges to a
        let b_neighbors: Vec<_> = self.edges[b].iter().copied().collect();
        for neighbor in b_neighbors {
            if neighbor != a {
                self.add_edge(a, neighbor);
            }
            self.edges[neighbor].remove(&b);
        }

        // Mark b as removed
        self.nodes[b].removed = true;
        self.edges[b].clear();

        // Update spill cost
        self.nodes[a].spill_cost += self.nodes[b].spill_cost;
    }

    /// Get all nodes that need a specific register class
    pub fn nodes_for_class(&self, class: RegisterClass) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|&n| self.nodes[n].reg_class == class && !self.nodes[n].removed)
            .collect()
    }

    /// Compute chromatic number lower bound
    pub fn chromatic_lower_bound(&self) -> usize {
        // Clique number is a lower bound on chromatic number
        // We use a greedy approximation
        let mut max_clique = 0;

        for start in 0..self.nodes.len() {
            if self.nodes[start].removed {
                continue;
            }

            // Greedy clique from this node
            let mut clique = vec![start];
            let mut candidates: Vec<_> = self.neighbors(start).collect();

            while !candidates.is_empty() {
                // Find candidate connected to all clique members
                let next = candidates.iter().position(|&c| {
                    clique.iter().all(|&member| self.edges[c].contains(&member))
                });

                if let Some(idx) = next {
                    let node = candidates.remove(idx);
                    clique.push(node);
                    candidates.retain(|&c| self.edges[node].contains(&c));
                } else {
                    break;
                }
            }

            max_clique = max_clique.max(clique.len());
        }

        max_clique
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interference_graph_basics() {
        let mut graph = InterferenceGraph {
            nodes: vec![
                InterferenceNode::new(ValueId(0), RegisterClass::GPR),
                InterferenceNode::new(ValueId(1), RegisterClass::GPR),
                InterferenceNode::new(ValueId(2), RegisterClass::GPR),
            ],
            value_to_node: [(ValueId(0), 0), (ValueId(1), 1), (ValueId(2), 2)]
                .into_iter()
                .collect(),
            edges: vec![HashSet::new(), HashSet::new(), HashSet::new()],
            move_edges: vec![],
        };

        // Add edges: 0-1, 1-2
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        assert_eq!(graph.degree(0), 1);
        assert_eq!(graph.degree(1), 2);
        assert_eq!(graph.degree(2), 1);
    }

    #[test]
    fn test_simplify_order() {
        let mut graph = InterferenceGraph {
            nodes: vec![
                InterferenceNode::new(ValueId(0), RegisterClass::GPR),
                InterferenceNode::new(ValueId(1), RegisterClass::GPR),
                InterferenceNode::new(ValueId(2), RegisterClass::GPR),
            ],
            value_to_node: [(ValueId(0), 0), (ValueId(1), 1), (ValueId(2), 2)]
                .into_iter()
                .collect(),
            edges: vec![HashSet::new(), HashSet::new(), HashSet::new()],
            move_edges: vec![],
        };

        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        let order = graph.simplify_order();
        assert_eq!(order.len(), 3);
    }
}
