// Week 25: Module System AST
//
// This module defines AST nodes for MedLang's module system, enabling
// multi-file programs with explicit imports and exports.

use super::{Declaration, Ident};

/// A module declaration containing imports, exports, and local declarations
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleDecl {
    pub name: ModulePath,
    pub imports: Vec<ImportDecl>,
    pub exports: Vec<ExportDecl>,
    pub declarations: Vec<Declaration>,
}

impl ModuleDecl {
    pub fn new(
        name: ModulePath,
        imports: Vec<ImportDecl>,
        exports: Vec<ExportDecl>,
        declarations: Vec<Declaration>,
    ) -> Self {
        Self {
            name,
            imports,
            exports,
            declarations,
        }
    }

    /// Find all import declarations referencing a specific module
    pub fn imports_from(&self, module_path: &str) -> Vec<&ImportDecl> {
        self.imports
            .iter()
            .filter(|imp| imp.module_path.to_string() == module_path)
            .collect()
    }

    /// Find all exported items
    pub fn exported_items(&self) -> Vec<String> {
        let mut items = Vec::new();
        for export in &self.exports {
            match export {
                ExportDecl::All => {
                    // Export all local declarations
                    for decl in &self.declarations {
                        items.push(decl.name().to_string());
                    }
                }
                ExportDecl::Item { name, .. } => {
                    items.push(name.to_string());
                }
            }
        }
        items
    }

    /// Check if a specific item is exported
    pub fn is_exported(&self, item_name: &str) -> bool {
        for export in &self.exports {
            match export {
                ExportDecl::All => return true,
                ExportDecl::Item { name, .. } if name.as_str() == item_name => return true,
                _ => {}
            }
        }
        false
    }
}

/// A dotted module path like "medlang_std.models.pkpd"
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModulePath {
    pub segments: Vec<String>,
}

impl ModulePath {
    pub fn new(segments: Vec<String>) -> Self {
        Self { segments }
    }

    pub fn from_string(s: &str) -> Self {
        Self {
            segments: s.split('.').map(|seg| seg.to_string()).collect(),
        }
    }

    pub fn to_string(&self) -> String {
        self.segments.join(".")
    }

    pub fn to_file_path(&self, base_dir: &str) -> String {
        format!("{}/{}.med", base_dir, self.segments.join("/"))
    }

    /// Return the last segment (the module name itself)
    pub fn leaf_name(&self) -> &str {
        self.segments.last().map(|s| s.as_str()).unwrap_or("")
    }
}

/// An import declaration: "import medlang_std.models.pkpd::{OneCmptIV, TwoCmptIV}"
#[derive(Debug, Clone, PartialEq)]
pub struct ImportDecl {
    pub module_path: ModulePath,
    pub items: ImportItems,
}

impl ImportDecl {
    pub fn new(module_path: ModulePath, items: ImportItems) -> Self {
        Self { module_path, items }
    }

    /// Return list of imported item names
    pub fn imported_names(&self) -> Vec<String> {
        match &self.items {
            ImportItems::All => vec!["*".to_string()],
            ImportItems::List(names) => names.iter().map(|n| n.to_string()).collect(),
        }
    }
}

/// What to import from a module
#[derive(Debug, Clone, PartialEq)]
pub enum ImportItems {
    /// Import all exported items: "import module::*"
    All,
    /// Import specific items: "import module::{Item1, Item2}"
    List(Vec<Ident>),
}

/// An export declaration: "export OneCmptIV" or "export *"
#[derive(Debug, Clone, PartialEq)]
pub enum ExportDecl {
    /// Export all local declarations
    All,
    /// Export a specific item
    Item { name: Ident, kind: ExportKind },
}

impl ExportDecl {
    pub fn item(name: Ident, kind: ExportKind) -> Self {
        Self::Item { name, kind }
    }
}

/// The kind of item being exported (for validation and resolution)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportKind {
    Model,
    Population,
    Measure,
    Timeline,
    Cohort,
    Protocol,
    Policy,
    EvidenceProgram,
}

impl ExportKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExportKind::Model => "model",
            ExportKind::Population => "population",
            ExportKind::Measure => "measure",
            ExportKind::Timeline => "timeline",
            ExportKind::Cohort => "cohort",
            ExportKind::Protocol => "protocol",
            ExportKind::Policy => "policy",
            ExportKind::EvidenceProgram => "evidence_program",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{CompartmentDef, Expr, ModelDef, RateDef};

    #[test]
    fn test_module_path_to_string() {
        let path = ModulePath::new(vec![
            "medlang_std".to_string(),
            "models".to_string(),
            "pkpd".to_string(),
        ]);
        assert_eq!(path.to_string(), "medlang_std.models.pkpd");
    }

    #[test]
    fn test_module_path_from_string() {
        let path = ModulePath::from_string("medlang_std.models.pkpd");
        assert_eq!(path.segments.len(), 3);
        assert_eq!(path.segments[0], "medlang_std");
        assert_eq!(path.segments[1], "models");
        assert_eq!(path.segments[2], "pkpd");
    }

    #[test]
    fn test_module_path_to_file_path() {
        let path = ModulePath::from_string("medlang_std.models.pkpd");
        let file_path = path.to_file_path("/base");
        assert_eq!(file_path, "/base/medlang_std/models/pkpd.med");
    }

    #[test]
    fn test_module_path_leaf_name() {
        let path = ModulePath::from_string("medlang_std.models.pkpd");
        assert_eq!(path.leaf_name(), "pkpd");
    }

    #[test]
    fn test_import_items_all() {
        let import = ImportDecl::new(
            ModulePath::from_string("medlang_std.models"),
            ImportItems::All,
        );
        assert_eq!(import.imported_names(), vec!["*"]);
    }

    #[test]
    fn test_import_items_list() {
        let import = ImportDecl::new(
            ModulePath::from_string("medlang_std.models.pkpd"),
            ImportItems::List(vec![Ident::from("OneCmptIV"), Ident::from("TwoCmptIV")]),
        );
        let names = import.imported_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"OneCmptIV".to_string()));
        assert!(names.contains(&"TwoCmptIV".to_string()));
    }

    #[test]
    fn test_export_all() {
        let module = ModuleDecl::new(
            ModulePath::from_string("my_module"),
            vec![],
            vec![ExportDecl::All],
            vec![
                Declaration::Model(ModelDef {
                    name: Ident::from("Model1"),
                    compartments: vec![],
                    rates: vec![],
                    observables: vec![],
                }),
                Declaration::Model(ModelDef {
                    name: Ident::from("Model2"),
                    compartments: vec![],
                    rates: vec![],
                    observables: vec![],
                }),
            ],
        );

        let exported = module.exported_items();
        assert_eq!(exported.len(), 2);
        assert!(exported.contains(&"Model1".to_string()));
        assert!(exported.contains(&"Model2".to_string()));
    }

    #[test]
    fn test_export_specific_item() {
        let module = ModuleDecl::new(
            ModulePath::from_string("my_module"),
            vec![],
            vec![ExportDecl::Item {
                name: Ident::from("Model1"),
                kind: ExportKind::Model,
            }],
            vec![
                Declaration::Model(ModelDef {
                    name: Ident::from("Model1"),
                    compartments: vec![],
                    rates: vec![],
                    observables: vec![],
                }),
                Declaration::Model(ModelDef {
                    name: Ident::from("Model2"),
                    compartments: vec![],
                    rates: vec![],
                    observables: vec![],
                }),
            ],
        );

        assert!(module.is_exported("Model1"));
        assert!(!module.is_exported("Model2"));
    }

    #[test]
    fn test_imports_from() {
        let module = ModuleDecl::new(
            ModulePath::from_string("my_module"),
            vec![
                ImportDecl::new(
                    ModulePath::from_string("medlang_std.models"),
                    ImportItems::All,
                ),
                ImportDecl::new(
                    ModulePath::from_string("medlang_std.protocols"),
                    ImportItems::List(vec![Ident::from("StandardDose")]),
                ),
                ImportDecl::new(
                    ModulePath::from_string("medlang_std.models"),
                    ImportItems::List(vec![Ident::from("OneCmptIV")]),
                ),
            ],
            vec![],
            vec![],
        );

        let from_models = module.imports_from("medlang_std.models");
        assert_eq!(from_models.len(), 2);

        let from_protocols = module.imports_from("medlang_std.protocols");
        assert_eq!(from_protocols.len(), 1);
    }
}
