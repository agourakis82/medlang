// Week 25: Module Loader
//
// This module handles file system mapping and loading of MedLang modules.
// It resolves module paths to file paths and loads module source code.

use crate::ast::{ModuleDecl, ModulePath};
use std::fs;
use std::path::{Path, PathBuf};

/// Errors that can occur during module loading
#[derive(Debug, Clone, PartialEq)]
pub enum LoaderError {
    /// Module file not found
    FileNotFound {
        module_path: String,
        searched_paths: Vec<String>,
    },

    /// I/O error reading module file
    IoError { module_path: String, error: String },

    /// Parse error in module file
    ParseError { module_path: String, error: String },

    /// Invalid module path
    InvalidPath { module_path: String },
}

impl std::fmt::Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoaderError::FileNotFound {
                module_path,
                searched_paths,
            } => {
                write!(
                    f,
                    "Module '{}' not found. Searched in: {}",
                    module_path,
                    searched_paths.join(", ")
                )
            }
            LoaderError::IoError { module_path, error } => {
                write!(f, "I/O error loading module '{}': {}", module_path, error)
            }
            LoaderError::ParseError { module_path, error } => {
                write!(f, "Parse error in module '{}': {}", module_path, error)
            }
            LoaderError::InvalidPath { module_path } => {
                write!(f, "Invalid module path: '{}'", module_path)
            }
        }
    }
}

impl std::error::Error for LoaderError {}

/// Module loader with configurable search paths
pub struct ModuleLoader {
    /// Search paths for modules (in order of preference)
    search_paths: Vec<PathBuf>,
}

impl ModuleLoader {
    /// Create a new module loader with default search paths
    pub fn new() -> Self {
        Self {
            search_paths: vec![
                PathBuf::from("."),             // Current directory
                PathBuf::from("./medlang_std"), // Standard library
            ],
        }
    }

    /// Create a module loader with custom search paths
    pub fn with_search_paths(paths: Vec<PathBuf>) -> Self {
        Self {
            search_paths: paths,
        }
    }

    /// Add a search path
    pub fn add_search_path(&mut self, path: PathBuf) {
        self.search_paths.push(path);
    }

    /// Resolve a module path to a file path
    ///
    /// Example: "medlang_std.models.pkpd" -> "./medlang_std/models/pkpd.med"
    pub fn resolve_file_path(&self, module_path: &ModulePath) -> Result<PathBuf, LoaderError> {
        let relative_path = self.module_path_to_file_path(module_path);

        // Try each search path
        for search_path in &self.search_paths {
            let full_path = search_path.join(&relative_path);
            if full_path.exists() {
                return Ok(full_path);
            }
        }

        // Not found in any search path
        Err(LoaderError::FileNotFound {
            module_path: module_path.to_string(),
            searched_paths: self
                .search_paths
                .iter()
                .map(|p| p.join(&relative_path).display().to_string())
                .collect(),
        })
    }

    /// Load a module from the file system
    ///
    /// This reads the file and returns the source code.
    /// Parsing is done separately by the caller.
    pub fn load_module_source(&self, module_path: &ModulePath) -> Result<String, LoaderError> {
        let file_path = self.resolve_file_path(module_path)?;

        fs::read_to_string(&file_path).map_err(|e| LoaderError::IoError {
            module_path: module_path.to_string(),
            error: e.to_string(),
        })
    }

    /// Convert module path to relative file path
    ///
    /// Example: "medlang_std.models.pkpd" -> "medlang_std/models/pkpd.med"
    fn module_path_to_file_path(&self, module_path: &ModulePath) -> PathBuf {
        let mut path = PathBuf::new();
        for segment in &module_path.segments {
            path.push(segment);
        }
        path.set_extension("med");
        path
    }

    /// Get all search paths
    pub fn search_paths(&self) -> &[PathBuf] {
        &self.search_paths
    }

    /// Check if a module exists in any search path
    pub fn module_exists(&self, module_path: &ModulePath) -> bool {
        self.resolve_file_path(module_path).is_ok()
    }

    /// List all .med files in a directory (non-recursive)
    pub fn list_modules_in_dir(&self, dir: &Path) -> Result<Vec<String>, LoaderError> {
        let entries = fs::read_dir(dir).map_err(|e| LoaderError::IoError {
            module_path: dir.display().to_string(),
            error: e.to_string(),
        })?;

        let mut modules = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| LoaderError::IoError {
                module_path: dir.display().to_string(),
                error: e.to_string(),
            })?;

            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("med") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    modules.push(stem.to_string());
                }
            }
        }

        Ok(modules)
    }
}

impl Default for ModuleLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Load and parse a module with all its dependencies
pub struct ModuleResolver {
    loader: ModuleLoader,
    loaded_modules: std::collections::HashMap<String, ModuleDecl>,
}

impl ModuleResolver {
    pub fn new(loader: ModuleLoader) -> Self {
        Self {
            loader,
            loaded_modules: std::collections::HashMap::new(),
        }
    }

    /// Check if a module has already been loaded
    pub fn is_loaded(&self, module_path: &str) -> bool {
        self.loaded_modules.contains_key(module_path)
    }

    /// Get a loaded module
    pub fn get_loaded(&self, module_path: &str) -> Option<&ModuleDecl> {
        self.loaded_modules.get(module_path)
    }

    /// Register a parsed module
    pub fn register_module(&mut self, module_path: String, module: ModuleDecl) {
        self.loaded_modules.insert(module_path, module);
    }

    /// Get all loaded modules
    pub fn all_modules(&self) -> Vec<&ModuleDecl> {
        self.loaded_modules.values().collect()
    }

    /// Get the module loader
    pub fn loader(&self) -> &ModuleLoader {
        &self.loader
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_path_to_file_path() {
        let loader = ModuleLoader::new();
        let module_path = ModulePath::from_string("medlang_std.models.pkpd");
        let file_path = loader.module_path_to_file_path(&module_path);

        assert_eq!(file_path.to_str().unwrap(), "medlang_std/models/pkpd.med");
    }

    #[test]
    fn test_single_segment_path() {
        let loader = ModuleLoader::new();
        let module_path = ModulePath::from_string("mymodule");
        let file_path = loader.module_path_to_file_path(&module_path);

        assert_eq!(file_path.to_str().unwrap(), "mymodule.med");
    }

    #[test]
    fn test_default_search_paths() {
        let loader = ModuleLoader::new();
        assert!(loader.search_paths().len() >= 2);
        assert!(loader.search_paths().contains(&PathBuf::from(".")));
        assert!(loader
            .search_paths()
            .contains(&PathBuf::from("./medlang_std")));
    }

    #[test]
    fn test_custom_search_paths() {
        let paths = vec![PathBuf::from("/custom/path")];
        let loader = ModuleLoader::with_search_paths(paths.clone());
        assert_eq!(loader.search_paths(), &paths);
    }

    #[test]
    fn test_add_search_path() {
        let mut loader = ModuleLoader::new();
        let initial_count = loader.search_paths().len();

        loader.add_search_path(PathBuf::from("/new/path"));
        assert_eq!(loader.search_paths().len(), initial_count + 1);
        assert!(loader.search_paths().contains(&PathBuf::from("/new/path")));
    }

    #[test]
    fn test_module_resolver_tracking() {
        let loader = ModuleLoader::new();
        let mut resolver = ModuleResolver::new(loader);

        assert!(!resolver.is_loaded("test.module"));

        let module = ModuleDecl::new(
            ModulePath::from_string("test.module"),
            vec![],
            vec![],
            vec![],
        );

        resolver.register_module("test.module".to_string(), module);

        assert!(resolver.is_loaded("test.module"));
        assert!(resolver.get_loaded("test.module").is_some());
    }

    #[test]
    fn test_module_resolver_all_modules() {
        let loader = ModuleLoader::new();
        let mut resolver = ModuleResolver::new(loader);

        let module1 = ModuleDecl::new(ModulePath::from_string("module1"), vec![], vec![], vec![]);
        let module2 = ModuleDecl::new(ModulePath::from_string("module2"), vec![], vec![], vec![]);

        resolver.register_module("module1".to_string(), module1);
        resolver.register_module("module2".to_string(), module2);

        assert_eq!(resolver.all_modules().len(), 2);
    }
}
