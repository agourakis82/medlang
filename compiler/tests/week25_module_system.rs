// Week 25 Integration Tests: Module System
//
// These tests verify the module system infrastructure:
// - AST for modules, imports, exports
// - Symbol table and name resolution
// - Module loader and file system mapping
// - CLI commands (check, build)

use medlangc::ast::{
    Declaration, ExportDecl, ExportKind, ImportDecl, ImportItems, ModelDef, ModuleDecl, ModulePath,
};
use medlangc::ir::module::{GlobalSymbolTable, ModuleScope, Symbol, SymbolKind};
use medlangc::loader::{ModuleLoader, ModuleResolver};
use medlangc::resolve::{NameResolver, ResolutionError};
use std::path::PathBuf;

#[test]
fn test_module_path_conversion() {
    let path = ModulePath::from_string("medlang_std.models.pkpd");
    assert_eq!(path.to_string(), "medlang_std.models.pkpd");
    assert_eq!(path.segments.len(), 3);
    assert_eq!(path.leaf_name(), "pkpd");
}

#[test]
fn test_module_path_to_file_path() {
    let path = ModulePath::from_string("medlang_std.models.pkpd");
    let file_path = path.to_file_path("/base");
    assert_eq!(file_path, "/base/medlang_std/models/pkpd.med");
}

#[test]
fn test_module_decl_creation() {
    let module = ModuleDecl::new(
        ModulePath::from_string("test.module"),
        vec![],
        vec![ExportDecl::All],
        vec![Declaration::Model(ModelDef {
            name: "TestModel".to_string(),
            items: vec![],
            span: None,
        })],
    );

    assert_eq!(module.name.to_string(), "test.module");
    assert_eq!(module.declarations.len(), 1);
    assert!(module.is_exported("TestModel"));
}

#[test]
fn test_import_export_tracking() {
    let mut module = ModuleDecl::new(
        ModulePath::from_string("consumer"),
        vec![ImportDecl::new(
            ModulePath::from_string("provider"),
            ImportItems::List(vec!["Model1".to_string(), "Model2".to_string()]),
        )],
        vec![ExportDecl::Item {
            name: "ConsumerModel".to_string(),
            kind: ExportKind::Model,
        }],
        vec![],
    );

    let imports = module.imports_from("provider");
    assert_eq!(imports.len(), 1);

    let exported = module.exported_items();
    assert_eq!(exported.len(), 0); // No declarations yet
}

#[test]
fn test_symbol_table_operations() {
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
    let resolved = gst.resolve_from_module("test.module", "TestModel");
    assert!(resolved.is_some());
    assert_eq!(resolved.unwrap().kind, SymbolKind::Model);
}

#[test]
fn test_name_resolution_two_pass() {
    let mut resolver = NameResolver::new();

    // Create provider module
    let mut provider = ModuleDecl::new(
        ModulePath::from_string("provider"),
        vec![],
        vec![ExportDecl::All],
        vec![Declaration::Model(ModelDef {
            name: "SharedModel".to_string(),
            items: vec![],
            span: None,
        })],
    );

    // Create consumer module that imports from provider
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

#[test]
fn test_name_resolution_module_not_found() {
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

    let result = resolver.resolve_modules(&vec![consumer]);
    assert!(result.is_err());

    match result.unwrap_err() {
        ResolutionError::ModuleNotFound { module_path, .. } => {
            assert_eq!(module_path, "nonexistent");
        }
        _ => panic!("Expected ModuleNotFound error"),
    }
}

#[test]
fn test_name_resolution_symbol_not_exported() {
    let mut resolver = NameResolver::new();

    // Provider with selective exports
    let provider = ModuleDecl::new(
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

    // Consumer trying to import private symbol
    let consumer = ModuleDecl::new(
        ModulePath::from_string("consumer"),
        vec![ImportDecl::new(
            ModulePath::from_string("provider"),
            ImportItems::List(vec!["PrivateModel".to_string()]),
        )],
        vec![],
        vec![],
    );

    let result = resolver.resolve_modules(&vec![provider, consumer]);
    assert!(result.is_err());

    match result.unwrap_err() {
        ResolutionError::SymbolNotExported { symbol_name, .. } => {
            assert_eq!(symbol_name, "PrivateModel");
        }
        _ => panic!("Expected SymbolNotExported error"),
    }
}

#[test]
fn test_module_loader_search_paths() {
    let loader = ModuleLoader::new();
    assert!(loader.search_paths().len() >= 2);
    assert!(loader.search_paths().contains(&PathBuf::from(".")));
}

#[test]
fn test_module_loader_custom_paths() {
    let custom_paths = vec![PathBuf::from("/custom1"), PathBuf::from("/custom2")];
    let loader = ModuleLoader::with_search_paths(custom_paths.clone());
    assert_eq!(loader.search_paths(), &custom_paths);
}

#[test]
fn test_module_loader_add_search_path() {
    let mut loader = ModuleLoader::new();
    let initial_count = loader.search_paths().len();

    loader.add_search_path(PathBuf::from("/new/path"));
    assert_eq!(loader.search_paths().len(), initial_count + 1);
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

    let module1 = ModuleDecl::new(ModulePath::from_string("mod1"), vec![], vec![], vec![]);
    let module2 = ModuleDecl::new(ModulePath::from_string("mod2"), vec![], vec![], vec![]);

    resolver.register_module("mod1".to_string(), module1);
    resolver.register_module("mod2".to_string(), module2);

    assert_eq!(resolver.all_modules().len(), 2);
}

#[test]
fn test_export_kind_to_symbol_kind() {
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

#[test]
fn test_symbol_kind_string_representation() {
    assert_eq!(SymbolKind::Model.as_str(), "model");
    assert_eq!(SymbolKind::Protocol.as_str(), "protocol");
    assert_eq!(SymbolKind::Policy.as_str(), "policy");
    assert_eq!(SymbolKind::EvidenceProgram.as_str(), "evidence_program");
}

#[test]
fn test_module_scope_export_all() {
    let mut scope = ModuleScope::new("test".to_string());

    scope.add_symbol(
        "Symbol1".to_string(),
        Symbol::new("Symbol1".to_string(), SymbolKind::Model, "test".to_string()),
    );
    scope.add_symbol(
        "Symbol2".to_string(),
        Symbol::new("Symbol2".to_string(), SymbolKind::Model, "test".to_string()),
    );

    scope.export_all();

    assert!(scope.is_exported("Symbol1"));
    assert!(scope.is_exported("Symbol2"));
    assert_eq!(scope.exported_names().len(), 2);
}

#[test]
fn test_resolution_error_display() {
    let err = ResolutionError::ModuleNotFound {
        module_path: "missing".to_string(),
        importing_from: "consumer".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("missing"));
    assert!(msg.contains("consumer"));

    let err2 = ResolutionError::SymbolNotFound {
        symbol_name: "Unknown".to_string(),
        module_path: "provider".to_string(),
    };
    let msg2 = format!("{}", err2);
    assert!(msg2.contains("Unknown"));
    assert!(msg2.contains("provider"));
}

#[test]
fn test_multi_module_resolution() {
    let mut resolver = NameResolver::new();

    // Create three modules: lib1, lib2, app
    let lib1 = ModuleDecl::new(
        ModulePath::from_string("lib1"),
        vec![],
        vec![ExportDecl::All],
        vec![Declaration::Model(ModelDef {
            name: "Model1".to_string(),
            items: vec![],
            span: None,
        })],
    );

    let lib2 = ModuleDecl::new(
        ModulePath::from_string("lib2"),
        vec![],
        vec![ExportDecl::All],
        vec![Declaration::Model(ModelDef {
            name: "Model2".to_string(),
            items: vec![],
            span: None,
        })],
    );

    let app = ModuleDecl::new(
        ModulePath::from_string("app"),
        vec![
            ImportDecl::new(ModulePath::from_string("lib1"), ImportItems::All),
            ImportDecl::new(ModulePath::from_string("lib2"), ImportItems::All),
        ],
        vec![],
        vec![],
    );

    let modules = vec![lib1, lib2, app];
    let symbol_table = resolver.resolve_modules(&modules).unwrap();

    assert!(symbol_table.has_module("lib1"));
    assert!(symbol_table.has_module("lib2"));
    assert!(symbol_table.has_module("app"));
}
