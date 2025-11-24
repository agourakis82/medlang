// Week 25: Module System IR
//
// This module defines the intermediate representation for MedLang's module system,
// including symbol tables and resolved references.

use std::collections::HashMap;

/// Global symbol table containing all loaded modules and their exports
#[derive(Debug, Clone)]
pub struct GlobalSymbolTable {
    /// Map from module path to module scope
    modules: HashMap<String, ModuleScope>,
}

impl GlobalSymbolTable {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    /// Register a new module in the symbol table
    pub fn register_module(&mut self, path: String, scope: ModuleScope) {
        self.modules.insert(path, scope);
    }

    /// Look up a module by path
    pub fn get_module(&self, path: &str) -> Option<&ModuleScope> {
        self.modules.get(path)
    }

    /// Resolve a symbol from a specific module
    pub fn resolve_from_module(&self, module_path: &str, symbol_name: &str) -> Option<&Symbol> {
        self.modules
            .get(module_path)
            .and_then(|scope| scope.resolve(symbol_name))
    }

    /// Get all registered module paths
    pub fn module_paths(&self) -> Vec<&str> {
        self.modules.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a module exists
    pub fn has_module(&self, path: &str) -> bool {
        self.modules.contains_key(path)
    }
}

impl Default for GlobalSymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Module scope containing all symbols defined and exported by a module
#[derive(Debug, Clone)]
pub struct ModuleScope {
    /// Module path (e.g., "medlang_std.models.pkpd")
    pub path: String,

    /// Symbols defined in this module
    symbols: HashMap<String, Symbol>,

    /// Exported symbol names
    exports: Vec<String>,
}

impl ModuleScope {
    pub fn new(path: String) -> Self {
        Self {
            path,
            symbols: HashMap::new(),
            exports: Vec::new(),
        }
    }

    /// Add a symbol to this module's scope
    pub fn add_symbol(&mut self, name: String, symbol: Symbol) {
        self.symbols.insert(name, symbol);
    }

    /// Mark a symbol as exported
    pub fn export_symbol(&mut self, name: String) {
        if !self.exports.contains(&name) {
            self.exports.push(name);
        }
    }

    /// Export all symbols
    pub fn export_all(&mut self) {
        self.exports = self.symbols.keys().cloned().collect();
    }

    /// Resolve a symbol by name
    pub fn resolve(&self, name: &str) -> Option<&Symbol> {
        self.symbols.get(name)
    }

    /// Check if a symbol is exported
    pub fn is_exported(&self, name: &str) -> bool {
        self.exports.contains(&name.to_string())
    }

    /// Get all exported symbols
    pub fn exported_symbols(&self) -> Vec<(&str, &Symbol)> {
        self.exports
            .iter()
            .filter_map(|name| self.symbols.get(name).map(|sym| (name.as_str(), sym)))
            .collect()
    }

    /// Get all symbol names
    pub fn symbol_names(&self) -> Vec<&str> {
        self.symbols.keys().map(|s| s.as_str()).collect()
    }

    /// Get exported symbol names
    pub fn exported_names(&self) -> Vec<&str> {
        self.exports.iter().map(|s| s.as_str()).collect()
    }
}

/// A symbol in the module system (resolved declaration)
#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    /// Symbol name
    pub name: String,

    /// Symbol kind
    pub kind: SymbolKind,

    /// Module path where this symbol is defined
    pub defining_module: String,
}

impl Symbol {
    pub fn new(name: String, kind: SymbolKind, defining_module: String) -> Self {
        Self {
            name,
            kind,
            defining_module,
        }
    }
}

/// Kind of symbol
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Model,
    Population,
    Measure,
    Timeline,
    Cohort,
    Protocol,
    Policy,
    EvidenceProgram,
}

impl SymbolKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            SymbolKind::Model => "model",
            SymbolKind::Population => "population",
            SymbolKind::Measure => "measure",
            SymbolKind::Timeline => "timeline",
            SymbolKind::Cohort => "cohort",
            SymbolKind::Protocol => "protocol",
            SymbolKind::Policy => "policy",
            SymbolKind::EvidenceProgram => "evidence_program",
        }
    }

    pub fn from_export_kind(kind: crate::ast::ExportKind) -> Self {
        match kind {
            crate::ast::ExportKind::Model => SymbolKind::Model,
            crate::ast::ExportKind::Population => SymbolKind::Population,
            crate::ast::ExportKind::Measure => SymbolKind::Measure,
            crate::ast::ExportKind::Timeline => SymbolKind::Timeline,
            crate::ast::ExportKind::Cohort => SymbolKind::Cohort,
            crate::ast::ExportKind::Protocol => SymbolKind::Protocol,
            crate::ast::ExportKind::Policy => SymbolKind::Policy,
            crate::ast::ExportKind::EvidenceProgram => SymbolKind::EvidenceProgram,
        }
    }
}

/// Module import resolution information
#[derive(Debug, Clone)]
pub struct ResolvedImport {
    /// Module path being imported from
    pub module_path: String,

    /// Imported symbols (name in current module -> symbol)
    pub symbols: HashMap<String, Symbol>,
}

impl ResolvedImport {
    pub fn new(module_path: String) -> Self {
        Self {
            module_path,
            symbols: HashMap::new(),
        }
    }

    /// Add a resolved symbol
    pub fn add_symbol(&mut self, local_name: String, symbol: Symbol) {
        self.symbols.insert(local_name, symbol);
    }

    /// Resolve a local name to its symbol
    pub fn resolve(&self, local_name: &str) -> Option<&Symbol> {
        self.symbols.get(local_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_symbol_table() {
        let mut gst = GlobalSymbolTable::new();
        let mut scope = ModuleScope::new("test.module".to_string());

        scope.add_symbol(
            "TestModel".to_string(),
            Symbol::new(
                "TestModel".to_string(),
                SymbolKind::Model,
                "test.module".to_string(),
            ),
        );
        scope.export_symbol("TestModel".to_string());

        gst.register_module("test.module".to_string(), scope);

        assert!(gst.has_module("test.module"));
        assert_eq!(gst.module_paths(), vec!["test.module"]);
    }

    #[test]
    fn test_module_scope_exports() {
        let mut scope = ModuleScope::new("test".to_string());

        scope.add_symbol(
            "Public".to_string(),
            Symbol::new("Public".to_string(), SymbolKind::Model, "test".to_string()),
        );
        scope.add_symbol(
            "Private".to_string(),
            Symbol::new("Private".to_string(), SymbolKind::Model, "test".to_string()),
        );

        scope.export_symbol("Public".to_string());

        assert!(scope.is_exported("Public"));
        assert!(!scope.is_exported("Private"));
        assert_eq!(scope.exported_names(), vec!["Public"]);
    }

    #[test]
    fn test_module_scope_export_all() {
        let mut scope = ModuleScope::new("test".to_string());

        scope.add_symbol(
            "Model1".to_string(),
            Symbol::new("Model1".to_string(), SymbolKind::Model, "test".to_string()),
        );
        scope.add_symbol(
            "Model2".to_string(),
            Symbol::new("Model2".to_string(), SymbolKind::Model, "test".to_string()),
        );

        scope.export_all();

        assert!(scope.is_exported("Model1"));
        assert!(scope.is_exported("Model2"));
        assert_eq!(scope.exported_names().len(), 2);
    }

    #[test]
    fn test_symbol_resolution() {
        let mut gst = GlobalSymbolTable::new();
        let mut scope = ModuleScope::new("mymodule".to_string());

        let symbol = Symbol::new(
            "MyModel".to_string(),
            SymbolKind::Model,
            "mymodule".to_string(),
        );

        scope.add_symbol("MyModel".to_string(), symbol.clone());
        scope.export_symbol("MyModel".to_string());
        gst.register_module("mymodule".to_string(), scope);

        let resolved = gst.resolve_from_module("mymodule", "MyModel");
        assert!(resolved.is_some());
        assert_eq!(resolved.unwrap().name, "MyModel");
        assert_eq!(resolved.unwrap().kind, SymbolKind::Model);
    }

    #[test]
    fn test_resolved_import() {
        let mut import = ResolvedImport::new("other.module".to_string());

        let symbol = Symbol::new(
            "ExternalModel".to_string(),
            SymbolKind::Model,
            "other.module".to_string(),
        );

        import.add_symbol("LocalName".to_string(), symbol.clone());

        let resolved = import.resolve("LocalName");
        assert!(resolved.is_some());
        assert_eq!(resolved.unwrap().name, "ExternalModel");
    }

    #[test]
    fn test_symbol_kind_conversion() {
        use crate::ast::ExportKind;

        assert_eq!(
            SymbolKind::from_export_kind(ExportKind::Model),
            SymbolKind::Model
        );
        assert_eq!(
            SymbolKind::from_export_kind(ExportKind::Protocol),
            SymbolKind::Protocol
        );
        assert_eq!(
            SymbolKind::from_export_kind(ExportKind::EvidenceProgram),
            SymbolKind::EvidenceProgram
        );
    }
}
