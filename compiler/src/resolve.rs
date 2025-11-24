// Week 25: Name Resolution for Module System
//
// This module implements two-pass name resolution:
// Pass 1: Collect all exports from each module
// Pass 2: Resolve all imports against collected exports

use crate::ast::{Declaration, ExportDecl, ImportDecl, ImportItems, ModuleDecl};
use crate::ir::module::{GlobalSymbolTable, ModuleScope, ResolvedImport, Symbol, SymbolKind};
use std::collections::HashMap;

/// Errors that can occur during name resolution
#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionError {
    /// Module not found during import
    ModuleNotFound {
        module_path: String,
        importing_from: String,
    },

    /// Symbol not found in imported module
    SymbolNotFound {
        symbol_name: String,
        module_path: String,
    },

    /// Symbol not exported by module
    SymbolNotExported {
        symbol_name: String,
        module_path: String,
    },

    /// Ambiguous import (multiple modules export same symbol with wildcard import)
    AmbiguousImport {
        symbol_name: String,
        modules: Vec<String>,
    },

    /// Circular dependency detected
    CircularDependency { cycle: Vec<String> },

    /// Declaration has no name (internal error)
    UnnamedDeclaration,
}

impl std::fmt::Display for ResolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolutionError::ModuleNotFound {
                module_path,
                importing_from,
            } => {
                write!(
                    f,
                    "Module '{}' not found (imported from '{}')",
                    module_path, importing_from
                )
            }
            ResolutionError::SymbolNotFound {
                symbol_name,
                module_path,
            } => {
                write!(
                    f,
                    "Symbol '{}' not found in module '{}'",
                    symbol_name, module_path
                )
            }
            ResolutionError::SymbolNotExported {
                symbol_name,
                module_path,
            } => {
                write!(
                    f,
                    "Symbol '{}' is not exported by module '{}'",
                    symbol_name, module_path
                )
            }
            ResolutionError::AmbiguousImport {
                symbol_name,
                modules,
            } => {
                write!(
                    f,
                    "Ambiguous import of '{}' from modules: {}",
                    symbol_name,
                    modules.join(", ")
                )
            }
            ResolutionError::CircularDependency { cycle } => {
                write!(f, "Circular dependency detected: {}", cycle.join(" -> "))
            }
            ResolutionError::UnnamedDeclaration => {
                write!(f, "Internal error: declaration has no name")
            }
        }
    }
}

impl std::error::Error for ResolutionError {}

/// Name resolver for the module system
pub struct NameResolver {
    /// Global symbol table
    symbol_table: GlobalSymbolTable,

    /// Resolved imports for each module
    resolved_imports: HashMap<String, Vec<ResolvedImport>>,
}

impl NameResolver {
    pub fn new() -> Self {
        Self {
            symbol_table: GlobalSymbolTable::new(),
            resolved_imports: HashMap::new(),
        }
    }

    /// Resolve a collection of modules (two-pass algorithm)
    pub fn resolve_modules(
        &mut self,
        modules: &[ModuleDecl],
    ) -> Result<GlobalSymbolTable, ResolutionError> {
        // Pass 1: Collect all exports from each module
        for module in modules {
            self.collect_exports(module)?;
        }

        // Pass 2: Resolve all imports
        for module in modules {
            self.resolve_imports(module)?;
        }

        Ok(self.symbol_table.clone())
    }

    /// Pass 1: Collect exports from a module and register in symbol table
    fn collect_exports(&mut self, module: &ModuleDecl) -> Result<(), ResolutionError> {
        let module_path = module.name.to_string();
        let mut scope = ModuleScope::new(module_path.clone());

        // Add all declarations as symbols
        for decl in &module.declarations {
            let symbol_name = decl.name().to_string();
            let symbol_kind = self.declaration_to_symbol_kind(decl);

            let symbol = Symbol::new(symbol_name.clone(), symbol_kind, module_path.clone());
            scope.add_symbol(symbol_name, symbol);
        }

        // Process exports
        for export in &module.exports {
            match export {
                ExportDecl::All => {
                    scope.export_all();
                }
                ExportDecl::Item { name, .. } => {
                    scope.export_symbol(name.clone());
                }
            }
        }

        // Register module in symbol table
        self.symbol_table.register_module(module_path, scope);

        Ok(())
    }

    /// Pass 2: Resolve imports for a module
    fn resolve_imports(&mut self, module: &ModuleDecl) -> Result<(), ResolutionError> {
        let module_path = module.name.to_string();
        let mut resolved_imports = Vec::new();

        for import in &module.imports {
            let resolved = self.resolve_import(&module_path, import)?;
            resolved_imports.push(resolved);
        }

        self.resolved_imports.insert(module_path, resolved_imports);

        Ok(())
    }

    /// Resolve a single import declaration
    fn resolve_import(
        &self,
        importing_module: &str,
        import: &ImportDecl,
    ) -> Result<ResolvedImport, ResolutionError> {
        let target_module_path = import.module_path.to_string();

        // Check that target module exists
        let target_scope = self
            .symbol_table
            .get_module(&target_module_path)
            .ok_or_else(|| ResolutionError::ModuleNotFound {
                module_path: target_module_path.clone(),
                importing_from: importing_module.to_string(),
            })?;

        let mut resolved = ResolvedImport::new(target_module_path.clone());

        match &import.items {
            ImportItems::All => {
                // Import all exported symbols
                for (name, symbol) in target_scope.exported_symbols() {
                    resolved.add_symbol(name.to_string(), symbol.clone());
                }
            }
            ImportItems::List(names) => {
                // Import specific symbols
                for name in names {
                    let symbol = target_scope.resolve(name).ok_or_else(|| {
                        ResolutionError::SymbolNotFound {
                            symbol_name: name.clone(),
                            module_path: target_module_path.clone(),
                        }
                    })?;

                    // Check that symbol is exported
                    if !target_scope.is_exported(name) {
                        return Err(ResolutionError::SymbolNotExported {
                            symbol_name: name.clone(),
                            module_path: target_module_path.clone(),
                        });
                    }

                    resolved.add_symbol(name.clone(), symbol.clone());
                }
            }
        }

        Ok(resolved)
    }

    /// Convert a declaration to its symbol kind
    fn declaration_to_symbol_kind(&self, decl: &Declaration) -> SymbolKind {
        match decl {
            Declaration::Model(_) => SymbolKind::Model,
            Declaration::Population(_) => SymbolKind::Population,
            Declaration::Measure(_) => SymbolKind::Measure,
            Declaration::Timeline(_) => SymbolKind::Timeline,
            Declaration::Cohort(_) => SymbolKind::Cohort,
            Declaration::Protocol(_) => SymbolKind::Protocol,
            Declaration::Evidence(_) => SymbolKind::EvidenceProgram,
        }
    }

    /// Get the global symbol table
    pub fn symbol_table(&self) -> &GlobalSymbolTable {
        &self.symbol_table
    }

    /// Get resolved imports for a module
    pub fn get_resolved_imports(&self, module_path: &str) -> Option<&Vec<ResolvedImport>> {
        self.resolved_imports.get(module_path)
    }
}

impl Default for NameResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{ExportKind, ModelDef, ModulePath};

    fn make_test_module(name: &str, exports: Vec<&str>) -> ModuleDecl {
        let mut module = ModuleDecl::new(ModulePath::from_string(name), vec![], vec![], vec![]);

        // Add some test declarations
        for export_name in &exports {
            module.declarations.push(Declaration::Model(ModelDef {
                name: export_name.to_string(),
                items: vec![],
                span: None,
            }));
        }

        // Export all
        module.exports.push(ExportDecl::All);

        module
    }

    #[test]
    fn test_collect_exports() {
        let mut resolver = NameResolver::new();
        let module = make_test_module("test.module", vec!["Model1", "Model2"]);

        resolver.collect_exports(&module).unwrap();

        let scope = resolver.symbol_table().get_module("test.module").unwrap();
        assert!(scope.is_exported("Model1"));
        assert!(scope.is_exported("Model2"));
    }

    #[test]
    fn test_resolve_import_all() {
        let mut resolver = NameResolver::new();

        let provider = make_test_module("provider", vec!["ModelA", "ModelB"]);
        let mut consumer = ModuleDecl::new(
            ModulePath::from_string("consumer"),
            vec![ImportDecl::new(
                ModulePath::from_string("provider"),
                ImportItems::All,
            )],
            vec![],
            vec![],
        );

        resolver.collect_exports(&provider).unwrap();
        resolver.collect_exports(&consumer).unwrap();
        resolver.resolve_imports(&consumer).unwrap();

        let imports = resolver.get_resolved_imports("consumer").unwrap();
        assert_eq!(imports.len(), 1);
        assert!(imports[0].resolve("ModelA").is_some());
        assert!(imports[0].resolve("ModelB").is_some());
    }

    #[test]
    fn test_resolve_import_specific() {
        let mut resolver = NameResolver::new();

        let provider = make_test_module("provider", vec!["ModelA", "ModelB", "ModelC"]);
        let mut consumer = ModuleDecl::new(
            ModulePath::from_string("consumer"),
            vec![ImportDecl::new(
                ModulePath::from_string("provider"),
                ImportItems::List(vec!["ModelA".to_string(), "ModelB".to_string()]),
            )],
            vec![],
            vec![],
        );

        resolver.collect_exports(&provider).unwrap();
        resolver.collect_exports(&consumer).unwrap();
        resolver.resolve_imports(&consumer).unwrap();

        let imports = resolver.get_resolved_imports("consumer").unwrap();
        assert_eq!(imports.len(), 1);
        assert!(imports[0].resolve("ModelA").is_some());
        assert!(imports[0].resolve("ModelB").is_some());
        assert!(imports[0].resolve("ModelC").is_none());
    }

    #[test]
    fn test_module_not_found() {
        let mut resolver = NameResolver::new();

        let consumer = ModuleDecl::new(
            ModulePath::from_string("consumer"),
            vec![ImportDecl::new(
                ModulePath::from_string("nonexistent"),
                ImportItems::All,
            )],
            vec![],
            vec![],
        );

        resolver.collect_exports(&consumer).unwrap();
        let result = resolver.resolve_imports(&consumer);

        assert!(result.is_err());
        match result.unwrap_err() {
            ResolutionError::ModuleNotFound { module_path, .. } => {
                assert_eq!(module_path, "nonexistent");
            }
            _ => panic!("Expected ModuleNotFound error"),
        }
    }

    #[test]
    fn test_symbol_not_found() {
        let mut resolver = NameResolver::new();

        let provider = make_test_module("provider", vec!["ModelA"]);
        let consumer = ModuleDecl::new(
            ModulePath::from_string("consumer"),
            vec![ImportDecl::new(
                ModulePath::from_string("provider"),
                ImportItems::List(vec!["NonExistent".to_string()]),
            )],
            vec![],
            vec![],
        );

        resolver.collect_exports(&provider).unwrap();
        resolver.collect_exports(&consumer).unwrap();
        let result = resolver.resolve_imports(&consumer);

        assert!(result.is_err());
        match result.unwrap_err() {
            ResolutionError::SymbolNotFound { symbol_name, .. } => {
                assert_eq!(symbol_name, "NonExistent");
            }
            _ => panic!("Expected SymbolNotFound error"),
        }
    }

    #[test]
    fn test_symbol_not_exported() {
        let mut resolver = NameResolver::new();

        // Create a module with a private symbol
        let mut provider = ModuleDecl::new(
            ModulePath::from_string("provider"),
            vec![],
            vec![ExportDecl::Item {
                name: "PublicModel".to_string(),
                kind: ExportKind::Model,
            }],
            vec![
                Declaration::Model(ModelDef {
                    name: "PublicModel".to_string(),
                    items: vec![],
                    span: None,
                }),
                Declaration::Model(ModelDef {
                    name: "PrivateModel".to_string(),
                    items: vec![],
                    span: None,
                }),
            ],
        );

        let consumer = ModuleDecl::new(
            ModulePath::from_string("consumer"),
            vec![ImportDecl::new(
                ModulePath::from_string("provider"),
                ImportItems::List(vec!["PrivateModel".to_string()]),
            )],
            vec![],
            vec![],
        );

        resolver.collect_exports(&provider).unwrap();
        resolver.collect_exports(&consumer).unwrap();
        let result = resolver.resolve_imports(&consumer);

        assert!(result.is_err());
        match result.unwrap_err() {
            ResolutionError::SymbolNotExported { symbol_name, .. } => {
                assert_eq!(symbol_name, "PrivateModel");
            }
            _ => panic!("Expected SymbolNotExported error"),
        }
    }

    #[test]
    fn test_resolve_modules() {
        let mut resolver = NameResolver::new();

        let provider = make_test_module("provider", vec!["SharedModel"]);
        let consumer = ModuleDecl::new(
            ModulePath::from_string("consumer"),
            vec![ImportDecl::new(
                ModulePath::from_string("provider"),
                ImportItems::List(vec!["SharedModel".to_string()]),
            )],
            vec![],
            vec![],
        );

        let modules = vec![provider, consumer];
        let symbol_table = resolver.resolve_modules(&modules).unwrap();

        assert!(symbol_table.has_module("provider"));
        assert!(symbol_table.has_module("consumer"));
        assert!(symbol_table
            .resolve_from_module("provider", "SharedModel")
            .is_some());
    }
}
