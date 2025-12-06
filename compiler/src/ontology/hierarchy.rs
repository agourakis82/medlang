// Week 54: Biomedical Ontology Infrastructure - Hierarchy Management
//
// Efficient hierarchy storage and subsumption queries using preorder/postorder
// numbering for O(1) ancestor testing on tree-like structures.
//
// ## Algorithms
//
// 1. **Preorder/Postorder numbering**: For trees, node v is descendant of u
//    iff u.pre < v.pre AND u.post > v.post (O(1) check)
//
// 2. **Transitive closure**: For DAGs, materialize ancestor sets or use
//    bitmap encoding for fast subsumption
//
// 3. **ELK-style reasoning**: Consequence-based classification for OWL EL

use super::core::{Concept, OntologyId, OntologySystem, Relationship, RelationshipType};
use std::collections::{HashMap, HashSet, VecDeque};

/// Preorder/postorder numbering for efficient ancestor queries
#[derive(Debug, Clone, Copy)]
pub struct TreePosition {
    /// Preorder number (visit order in DFS)
    pub pre: u32,
    /// Postorder number (finish order in DFS)
    pub post: u32,
    /// Depth in tree
    pub depth: u16,
}

impl TreePosition {
    /// Check if this position is an ancestor of another
    /// (in tree terms: this.pre < other.pre AND this.post > other.post)
    pub fn is_ancestor_of(&self, other: &TreePosition) -> bool {
        self.pre < other.pre && self.post > other.post
    }

    /// Check if this position is a descendant of another
    pub fn is_descendant_of(&self, other: &TreePosition) -> bool {
        other.is_ancestor_of(self)
    }
}

/// Ontology hierarchy with efficient subsumption queries
pub struct OntologyHierarchy {
    /// System this hierarchy is for
    pub system: OntologySystem,
    /// Concepts by ID
    concepts: HashMap<String, Concept>,
    /// Parent relationships (child -> parents)
    parents: HashMap<String, Vec<String>>,
    /// Child relationships (parent -> children)
    children: HashMap<String, Vec<String>>,
    /// Tree positions for O(1) ancestor check (valid for tree-like hierarchies)
    tree_positions: HashMap<String, TreePosition>,
    /// Root concepts (no parents)
    roots: Vec<String>,
    /// Transitive closure (for DAGs) - concept -> all ancestors
    transitive_ancestors: Option<HashMap<String, HashSet<String>>>,
}

impl OntologyHierarchy {
    pub fn new(system: OntologySystem) -> Self {
        OntologyHierarchy {
            system,
            concepts: HashMap::new(),
            parents: HashMap::new(),
            children: HashMap::new(),
            tree_positions: HashMap::new(),
            roots: Vec::new(),
            transitive_ancestors: None,
        }
    }

    /// Add a concept to the hierarchy
    pub fn add_concept(&mut self, concept: Concept) {
        let code = concept.id.code.clone();

        // Add parent relationships
        for parent in &concept.parents {
            self.parents
                .entry(code.clone())
                .or_default()
                .push(parent.code.clone());

            self.children
                .entry(parent.code.clone())
                .or_default()
                .push(code.clone());
        }

        self.concepts.insert(code, concept);

        // Invalidate computed structures
        self.tree_positions.clear();
        self.transitive_ancestors = None;
    }

    /// Add a relationship
    pub fn add_relationship(&mut self, rel: &Relationship) {
        if rel.rel_type == RelationshipType::IsA {
            self.parents
                .entry(rel.source.code.clone())
                .or_default()
                .push(rel.target.code.clone());

            self.children
                .entry(rel.target.code.clone())
                .or_default()
                .push(rel.source.code.clone());
        }

        // Invalidate computed structures
        self.tree_positions.clear();
        self.transitive_ancestors = None;
    }

    /// Get a concept by code
    pub fn get_concept(&self, code: &str) -> Option<&Concept> {
        self.concepts.get(code)
    }

    /// Get direct parents of a concept
    pub fn get_parents(&self, code: &str) -> Vec<&str> {
        self.parents
            .get(code)
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get direct children of a concept
    pub fn get_children(&self, code: &str) -> Vec<&str> {
        self.children
            .get(code)
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Find root concepts (no parents)
    pub fn find_roots(&mut self) -> &[String] {
        if self.roots.is_empty() {
            for code in self.concepts.keys() {
                if !self.parents.contains_key(code) || self.parents[code].is_empty() {
                    self.roots.push(code.clone());
                }
            }
        }
        &self.roots
    }

    /// Build tree positions using DFS (for tree-like hierarchies)
    pub fn build_tree_positions(&mut self) {
        if !self.tree_positions.is_empty() {
            return;
        }

        self.find_roots();
        let roots = self.roots.clone();

        let mut pre_counter = 0u32;
        let mut post_counter = 0u32;
        let mut visited = HashSet::new();

        for root in &roots {
            self.dfs_number(root, 0, &mut pre_counter, &mut post_counter, &mut visited);
        }
    }

    fn dfs_number(
        &mut self,
        code: &str,
        depth: u16,
        pre: &mut u32,
        post: &mut u32,
        visited: &mut HashSet<String>,
    ) {
        if visited.contains(code) {
            return; // DAG: already visited via another path
        }
        visited.insert(code.to_string());

        let pre_num = *pre;
        *pre += 1;

        // Visit children
        if let Some(children) = self.children.get(code).cloned() {
            for child in children {
                self.dfs_number(&child, depth + 1, pre, post, visited);
            }
        }

        let post_num = *post;
        *post += 1;

        self.tree_positions.insert(
            code.to_string(),
            TreePosition {
                pre: pre_num,
                post: post_num,
                depth,
            },
        );
    }

    /// Build transitive closure (all ancestors for each concept)
    pub fn build_transitive_closure(&mut self) {
        if self.transitive_ancestors.is_some() {
            return;
        }

        let mut ancestors: HashMap<String, HashSet<String>> = HashMap::new();

        // Topological sort to process parents before children
        let sorted = self.topological_sort();

        for code in sorted {
            let mut my_ancestors = HashSet::new();

            if let Some(parent_codes) = self.parents.get(&code) {
                for parent in parent_codes {
                    my_ancestors.insert(parent.clone());
                    // Add all ancestors of parent
                    if let Some(parent_ancestors) = ancestors.get(parent) {
                        my_ancestors.extend(parent_ancestors.iter().cloned());
                    }
                }
            }

            ancestors.insert(code, my_ancestors);
        }

        self.transitive_ancestors = Some(ancestors);
    }

    fn topological_sort(&self) -> Vec<String> {
        let mut result = Vec::new();
        let mut in_degree: HashMap<String, usize> = HashMap::new();

        // Initialize in-degrees
        for code in self.concepts.keys() {
            in_degree.insert(code.clone(), 0);
        }

        // Count incoming edges (children point to parents via is-a)
        for (child, parents) in &self.parents {
            if let Some(degree) = in_degree.get_mut(child) {
                *degree = parents.len();
            }
        }

        // Start with roots (no parents)
        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(k, _)| k.clone())
            .collect();

        while let Some(code) = queue.pop_front() {
            result.push(code.clone());

            if let Some(children) = self.children.get(&code) {
                for child in children {
                    if let Some(degree) = in_degree.get_mut(child) {
                        *degree = degree.saturating_sub(1);
                        if *degree == 0 {
                            queue.push_back(child.clone());
                        }
                    }
                }
            }
        }

        result
    }

    /// Check if `descendant` is subsumed by `ancestor` (descendant is-a ancestor)
    pub fn is_subsumed_by(&mut self, descendant: &str, ancestor: &str) -> bool {
        if descendant == ancestor {
            return true;
        }

        // Try tree position first (O(1) if valid)
        if !self.tree_positions.is_empty() {
            if let (Some(desc_pos), Some(anc_pos)) = (
                self.tree_positions.get(descendant),
                self.tree_positions.get(ancestor),
            ) {
                return desc_pos.is_descendant_of(anc_pos);
            }
        }

        // Fall back to transitive closure
        self.build_transitive_closure();

        if let Some(ref ancestors) = self.transitive_ancestors {
            if let Some(desc_ancestors) = ancestors.get(descendant) {
                return desc_ancestors.contains(ancestor);
            }
        }

        false
    }

    /// Get all ancestors of a concept
    pub fn get_ancestors(&mut self, code: &str) -> HashSet<String> {
        self.build_transitive_closure();

        self.transitive_ancestors
            .as_ref()
            .and_then(|tc| tc.get(code))
            .cloned()
            .unwrap_or_default()
    }

    /// Get all descendants of a concept
    pub fn get_descendants(&self, code: &str) -> HashSet<String> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(code.to_string());

        while let Some(current) = queue.pop_front() {
            if let Some(children) = self.children.get(&current) {
                for child in children {
                    if result.insert(child.clone()) {
                        queue.push_back(child.clone());
                    }
                }
            }
        }

        result
    }

    /// Find concepts matching a predicate
    pub fn find_concepts<F>(&self, predicate: F) -> Vec<&Concept>
    where
        F: Fn(&Concept) -> bool,
    {
        self.concepts.values().filter(|c| predicate(c)).collect()
    }

    /// Get concept count
    pub fn concept_count(&self) -> usize {
        self.concepts.len()
    }

    /// Check if hierarchy is a tree (no multiple inheritance)
    pub fn is_tree(&self) -> bool {
        for parents in self.parents.values() {
            if parents.len() > 1 {
                return false;
            }
        }
        true
    }
}

/// Builder for constructing hierarchies
pub struct HierarchyBuilder {
    hierarchy: OntologyHierarchy,
}

impl HierarchyBuilder {
    pub fn new(system: OntologySystem) -> Self {
        HierarchyBuilder {
            hierarchy: OntologyHierarchy::new(system),
        }
    }

    pub fn add_concept(mut self, concept: Concept) -> Self {
        self.hierarchy.add_concept(concept);
        self
    }

    pub fn add_is_a(mut self, child: &str, parent: &str) -> Self {
        let rel = Relationship::is_a(
            OntologyId::new(self.hierarchy.system, child),
            OntologyId::new(self.hierarchy.system, parent),
        );
        self.hierarchy.add_relationship(&rel);
        self
    }

    pub fn build(mut self) -> OntologyHierarchy {
        self.hierarchy.build_tree_positions();
        self.hierarchy
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::core::SemanticType;

    fn create_test_hierarchy() -> OntologyHierarchy {
        // Create a simple disease hierarchy:
        //
        //        Disease
        //       /       \
        //  MetabolicD   CardioD
        //      |           |
        //  Diabetes    HeartFailure
        //     |
        //   T2DM

        let mut hierarchy = OntologyHierarchy::new(OntologySystem::SnomedCt);

        let disease = Concept::new(
            OntologyId::snomed("64572001"),
            "Disease (disorder)",
            "Disease",
        )
        .with_semantic_type(SemanticType::Disease);

        let metabolic = Concept::new(
            OntologyId::snomed("126877002"),
            "Metabolic disease (disorder)",
            "Metabolic disease",
        )
        .with_semantic_type(SemanticType::Disease)
        .with_parent(OntologyId::snomed("64572001"));

        let cardio = Concept::new(
            OntologyId::snomed("49601007"),
            "Cardiovascular disease (disorder)",
            "Cardiovascular disease",
        )
        .with_semantic_type(SemanticType::Disease)
        .with_parent(OntologyId::snomed("64572001"));

        let diabetes = Concept::new(
            OntologyId::snomed("73211009"),
            "Diabetes mellitus (disorder)",
            "Diabetes mellitus",
        )
        .with_semantic_type(SemanticType::Disease)
        .with_parent(OntologyId::snomed("126877002"));

        let t2dm = Concept::new(
            OntologyId::snomed("44054006"),
            "Type 2 diabetes mellitus (disorder)",
            "Type 2 diabetes",
        )
        .with_semantic_type(SemanticType::Disease)
        .with_parent(OntologyId::snomed("73211009"));

        let hf = Concept::new(
            OntologyId::snomed("84114007"),
            "Heart failure (disorder)",
            "Heart failure",
        )
        .with_semantic_type(SemanticType::Disease)
        .with_parent(OntologyId::snomed("49601007"));

        hierarchy.add_concept(disease);
        hierarchy.add_concept(metabolic);
        hierarchy.add_concept(cardio);
        hierarchy.add_concept(diabetes);
        hierarchy.add_concept(t2dm);
        hierarchy.add_concept(hf);

        hierarchy
    }

    #[test]
    fn test_hierarchy_creation() {
        let hierarchy = create_test_hierarchy();
        assert_eq!(hierarchy.concept_count(), 6);
    }

    #[test]
    fn test_get_parents() {
        let hierarchy = create_test_hierarchy();

        let t2dm_parents = hierarchy.get_parents("44054006");
        assert_eq!(t2dm_parents.len(), 1);
        assert_eq!(t2dm_parents[0], "73211009");
    }

    #[test]
    fn test_get_children() {
        let hierarchy = create_test_hierarchy();

        let disease_children = hierarchy.get_children("64572001");
        assert_eq!(disease_children.len(), 2);
        assert!(disease_children.contains(&"126877002")); // Metabolic
        assert!(disease_children.contains(&"49601007")); // Cardio
    }

    #[test]
    fn test_subsumption() {
        let mut hierarchy = create_test_hierarchy();

        // T2DM is-a Diabetes
        assert!(hierarchy.is_subsumed_by("44054006", "73211009"));

        // T2DM is-a Metabolic disease
        assert!(hierarchy.is_subsumed_by("44054006", "126877002"));

        // T2DM is-a Disease
        assert!(hierarchy.is_subsumed_by("44054006", "64572001"));

        // Heart failure is NOT subsumed by Diabetes
        assert!(!hierarchy.is_subsumed_by("84114007", "73211009"));

        // Self-subsumption
        assert!(hierarchy.is_subsumed_by("73211009", "73211009"));
    }

    #[test]
    fn test_get_ancestors() {
        let mut hierarchy = create_test_hierarchy();

        let t2dm_ancestors = hierarchy.get_ancestors("44054006");
        assert_eq!(t2dm_ancestors.len(), 3);
        assert!(t2dm_ancestors.contains("73211009")); // Diabetes
        assert!(t2dm_ancestors.contains("126877002")); // Metabolic
        assert!(t2dm_ancestors.contains("64572001")); // Disease
    }

    #[test]
    fn test_get_descendants() {
        let hierarchy = create_test_hierarchy();

        let disease_descendants = hierarchy.get_descendants("64572001");
        assert_eq!(disease_descendants.len(), 5);
        assert!(disease_descendants.contains("44054006")); // T2DM
    }

    #[test]
    fn test_find_roots() {
        let mut hierarchy = create_test_hierarchy();

        let roots = hierarchy.find_roots();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], "64572001"); // Disease
    }

    #[test]
    fn test_is_tree() {
        let hierarchy = create_test_hierarchy();
        assert!(hierarchy.is_tree());
    }

    #[test]
    fn test_tree_positions() {
        let mut hierarchy = create_test_hierarchy();
        hierarchy.build_tree_positions();

        let disease_pos = hierarchy.tree_positions.get("64572001").unwrap();
        let t2dm_pos = hierarchy.tree_positions.get("44054006").unwrap();

        // Disease should be ancestor of T2DM
        assert!(disease_pos.is_ancestor_of(t2dm_pos));
        assert!(t2dm_pos.is_descendant_of(disease_pos));
    }

    #[test]
    fn test_hierarchy_builder() {
        let hierarchy = HierarchyBuilder::new(OntologySystem::SnomedCt)
            .add_concept(Concept::new(OntologyId::snomed("A"), "Concept A", "A"))
            .add_concept(Concept::new(OntologyId::snomed("B"), "Concept B", "B"))
            .add_is_a("B", "A")
            .build();

        assert_eq!(hierarchy.concept_count(), 2);
        assert_eq!(hierarchy.get_parents("B"), vec!["A"]);
    }
}
