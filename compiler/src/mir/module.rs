//! MIR Module
//!
//! A MIR module contains all the functions, globals, and type definitions
//! for a compilation unit.

use std::collections::HashMap;

use super::function::*;
use super::types::*;
use super::value::*;

/// A MIR module (compilation unit)
#[derive(Clone, Debug)]
pub struct MirModule {
    /// Module name
    pub name: String,
    /// Functions
    pub functions: Vec<MirFunction>,
    /// Global variables
    pub globals: Vec<GlobalDecl>,
    /// Type definitions
    pub types: Vec<TypeDef>,
    /// Constants
    pub constants: Vec<ConstantDef>,
    /// External function declarations
    pub external_functions: Vec<ExternalFunction>,
    /// Module attributes
    pub attributes: ModuleAttributes,
}

impl MirModule {
    /// Create a new module
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            functions: Vec::new(),
            globals: Vec::new(),
            types: Vec::new(),
            constants: Vec::new(),
            external_functions: Vec::new(),
            attributes: ModuleAttributes::default(),
        }
    }

    /// Add a function
    pub fn add_function(&mut self, func: MirFunction) {
        self.functions.push(func);
    }

    /// Add a global variable
    pub fn add_global(&mut self, global: GlobalDecl) {
        self.globals.push(global);
    }

    /// Add a type definition
    pub fn add_type(&mut self, type_def: TypeDef) {
        self.types.push(type_def);
    }

    /// Add a constant
    pub fn add_constant(&mut self, constant: ConstantDef) {
        self.constants.push(constant);
    }

    /// Add an external function declaration
    pub fn add_external_function(&mut self, ext: ExternalFunction) {
        self.external_functions.push(ext);
    }

    /// Get a function by name
    pub fn get_function(&self, name: &str) -> Option<&MirFunction> {
        self.functions.iter().find(|f| f.name == name)
    }

    /// Get a function mutably by name
    pub fn get_function_mut(&mut self, name: &str) -> Option<&mut MirFunction> {
        self.functions.iter_mut().find(|f| f.name == name)
    }

    /// Get a global by name
    pub fn get_global(&self, name: &str) -> Option<&GlobalDecl> {
        self.globals.iter().find(|g| g.name == name)
    }

    /// Get a type by name
    pub fn get_type(&self, name: &str) -> Option<&TypeDef> {
        self.types.iter().find(|t| t.name == name)
    }

    /// Check if an external function exists
    pub fn has_external(&self, name: &str) -> bool {
        self.external_functions.iter().any(|e| e.name == name)
    }

    /// Number of functions
    pub fn num_functions(&self) -> usize {
        self.functions.len()
    }

    /// Number of globals
    pub fn num_globals(&self) -> usize {
        self.globals.len()
    }

    /// Validate the entire module
    pub fn validate(&self) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        for func in &self.functions {
            if let Err(e) = func.validate() {
                errors.push(e);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get all function names
    pub fn function_names(&self) -> Vec<&str> {
        self.functions.iter().map(|f| f.name.as_str()).collect()
    }

    /// Get all global names
    pub fn global_names(&self) -> Vec<&str> {
        self.globals.iter().map(|g| g.name.as_str()).collect()
    }

    /// Build a symbol table
    pub fn symbol_table(&self) -> SymbolTable {
        let mut table = SymbolTable::new();

        for (i, func) in self.functions.iter().enumerate() {
            table.functions.insert(func.name.clone(), i);
        }

        for (i, global) in self.globals.iter().enumerate() {
            table.globals.insert(global.name.clone(), i);
        }

        for (i, type_def) in self.types.iter().enumerate() {
            table.types.insert(type_def.name.clone(), i);
        }

        for (i, ext) in self.external_functions.iter().enumerate() {
            table.externals.insert(ext.name.clone(), i);
        }

        table
    }
}

/// Symbol table for looking up definitions
#[derive(Debug, Default)]
pub struct SymbolTable {
    pub functions: HashMap<String, usize>,
    pub globals: HashMap<String, usize>,
    pub types: HashMap<String, usize>,
    pub externals: HashMap<String, usize>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Global variable declaration
#[derive(Clone, Debug)]
pub struct GlobalDecl {
    /// Name
    pub name: String,
    /// Type
    pub ty: MirType,
    /// Initial value (if any)
    pub initializer: Option<ConstValue>,
    /// Is this mutable?
    pub mutable: bool,
    /// Linkage
    pub linkage: Linkage,
    /// Section
    pub section: Option<String>,
    /// Alignment override
    pub alignment: Option<usize>,
    /// Is thread-local?
    pub thread_local: bool,
}

impl GlobalDecl {
    pub fn new(name: &str, ty: MirType) -> Self {
        Self {
            name: name.to_string(),
            ty,
            initializer: None,
            mutable: true,
            linkage: Linkage::Internal,
            section: None,
            alignment: None,
            thread_local: false,
        }
    }

    pub fn with_initializer(mut self, init: ConstValue) -> Self {
        self.initializer = Some(init);
        self
    }

    pub fn immutable(mut self) -> Self {
        self.mutable = false;
        self
    }

    pub fn with_linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }

    pub fn with_section(mut self, section: &str) -> Self {
        self.section = Some(section.to_string());
        self
    }

    pub fn thread_local(mut self) -> Self {
        self.thread_local = true;
        self
    }
}

/// Type definition (for structs, enums)
#[derive(Clone, Debug)]
pub struct TypeDef {
    /// Name
    pub name: String,
    /// The type
    pub ty: MirType,
}

impl TypeDef {
    pub fn new(name: &str, ty: MirType) -> Self {
        Self {
            name: name.to_string(),
            ty,
        }
    }
}

/// Constant definition
#[derive(Clone, Debug)]
pub struct ConstantDef {
    /// Name
    pub name: String,
    /// Type
    pub ty: MirType,
    /// Value
    pub value: ConstValue,
}

impl ConstantDef {
    pub fn new(name: &str, ty: MirType, value: ConstValue) -> Self {
        Self {
            name: name.to_string(),
            ty,
            value,
        }
    }
}

/// Constant value
#[derive(Clone, Debug)]
pub enum ConstValue {
    /// Integer constant
    Int(i128),
    /// Float constant
    Float(f64),
    /// Boolean constant
    Bool(bool),
    /// String constant
    String(String),
    /// Array constant
    Array(Vec<ConstValue>),
    /// Struct constant
    Struct(Vec<(String, ConstValue)>),
    /// Null pointer
    Null,
    /// Zero initialized
    Zero,
    /// Undefined
    Undef,
}

impl ConstValue {
    /// Get integer value
    pub fn as_int(&self) -> Option<i128> {
        match self {
            ConstValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Get float value
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ConstValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Get bool value
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConstValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// External function declaration
#[derive(Clone, Debug)]
pub struct ExternalFunction {
    /// Name
    pub name: String,
    /// Signature
    pub signature: FunctionSignature,
    /// Library name (for linking)
    pub library: Option<String>,
}

impl ExternalFunction {
    pub fn new(name: &str, signature: FunctionSignature) -> Self {
        Self {
            name: name.to_string(),
            signature,
            library: None,
        }
    }

    pub fn from_library(name: &str, signature: FunctionSignature, library: &str) -> Self {
        Self {
            name: name.to_string(),
            signature,
            library: Some(library.to_string()),
        }
    }
}

/// Linkage type
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Linkage {
    /// Internal (not visible outside module)
    #[default]
    Internal,
    /// External (visible, can be linked)
    External,
    /// Weak (can be overridden)
    Weak,
    /// Link-once ODR (one definition rule)
    LinkOnceODR,
    /// Common (like C common symbols)
    Common,
    /// Private (not visible to linker)
    Private,
}

/// Module attributes
#[derive(Clone, Debug, Default)]
pub struct ModuleAttributes {
    /// Source file
    pub source_file: Option<String>,
    /// Target triple (e.g., "x86_64-unknown-linux-gnu")
    pub target_triple: Option<String>,
    /// Data layout string
    pub data_layout: Option<String>,
    /// Module flags
    pub flags: Vec<ModuleFlag>,
}

/// Module flag
#[derive(Clone, Debug)]
pub struct ModuleFlag {
    pub name: String,
    pub value: String,
}

/// Builder for constructing modules
pub struct ModuleBuilder {
    module: MirModule,
}

impl ModuleBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            module: MirModule::new(name),
        }
    }

    /// Set target triple
    pub fn target(&mut self, triple: &str) -> &mut Self {
        self.module.attributes.target_triple = Some(triple.to_string());
        self
    }

    /// Set source file
    pub fn source_file(&mut self, file: &str) -> &mut Self {
        self.module.attributes.source_file = Some(file.to_string());
        self
    }

    /// Add a function using a builder
    pub fn function(
        &mut self,
        name: &str,
        signature: FunctionSignature,
        build_fn: impl FnOnce(&mut FunctionBuilder),
    ) -> &mut Self {
        let mut builder = FunctionBuilder::new(name, signature);
        build_fn(&mut builder);
        self.module.add_function(builder.build());
        self
    }

    /// Add a global variable
    pub fn global(&mut self, global: GlobalDecl) -> &mut Self {
        self.module.add_global(global);
        self
    }

    /// Add a type definition
    pub fn type_def(&mut self, name: &str, ty: MirType) -> &mut Self {
        self.module.add_type(TypeDef::new(name, ty));
        self
    }

    /// Add a constant
    pub fn constant(&mut self, name: &str, ty: MirType, value: ConstValue) -> &mut Self {
        self.module.add_constant(ConstantDef::new(name, ty, value));
        self
    }

    /// Add an external function
    pub fn external(&mut self, name: &str, signature: FunctionSignature) -> &mut Self {
        self.module
            .add_external_function(ExternalFunction::new(name, signature));
        self
    }

    /// Add an external function from a library
    pub fn external_from_lib(
        &mut self,
        name: &str,
        signature: FunctionSignature,
        library: &str,
    ) -> &mut Self {
        self.module
            .add_external_function(ExternalFunction::from_library(name, signature, library));
        self
    }

    /// Build the module
    pub fn build(self) -> MirModule {
        self.module
    }

    /// Build with validation
    pub fn build_validated(self) -> Result<MirModule, Vec<ValidationError>> {
        self.module.validate()?;
        Ok(self.module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::block::Terminator;
    use crate::mir::inst::Operation;

    #[test]
    fn test_module_creation() {
        let module = MirModule::new("test_module");
        assert_eq!(module.name, "test_module");
        assert!(module.functions.is_empty());
        assert!(module.globals.is_empty());
    }

    #[test]
    fn test_module_builder() {
        let mut builder = ModuleBuilder::new("example");

        builder
            .target("x86_64-unknown-linux-gnu")
            .source_file("example.med");

        builder.function(
            "add",
            FunctionSignature::new(vec![MirType::F64, MirType::F64], MirType::F64),
            |f| {
                let x = f.param(0).unwrap();
                let y = f.param(1).unwrap();
                let sum = f.push_op(Operation::FAdd { lhs: x, rhs: y }, MirType::F64);
                f.terminate(Terminator::Return { value: Some(sum) });
            },
        );

        builder.global(
            GlobalDecl::new("PI", MirType::F64)
                .with_initializer(ConstValue::Float(std::f64::consts::PI))
                .immutable()
                .with_linkage(Linkage::External),
        );

        let module = builder.build();

        assert_eq!(module.num_functions(), 1);
        assert_eq!(module.num_globals(), 1);
        assert!(module.get_function("add").is_some());
        assert!(module.get_global("PI").is_some());
    }

    #[test]
    fn test_external_functions() {
        let mut module = MirModule::new("test");

        module.add_external_function(ExternalFunction::from_library(
            "printf",
            FunctionSignature::new(vec![MirType::ptr(MirType::I8, false)], MirType::I32).variadic(),
            "libc",
        ));

        assert!(module.has_external("printf"));
        assert!(!module.has_external("scanf"));
    }

    #[test]
    fn test_symbol_table() {
        let mut module = MirModule::new("test");

        module.add_function(MirFunction::new(
            "foo",
            FunctionSignature::new(vec![], MirType::Void),
        ));
        module.add_function(MirFunction::new(
            "bar",
            FunctionSignature::new(vec![], MirType::Void),
        ));
        module.add_global(GlobalDecl::new("g1", MirType::I32));

        let table = module.symbol_table();

        assert_eq!(table.functions.get("foo"), Some(&0));
        assert_eq!(table.functions.get("bar"), Some(&1));
        assert_eq!(table.globals.get("g1"), Some(&0));
    }

    #[test]
    fn test_type_definitions() {
        let mut module = MirModule::new("test");

        let point_type = MirType::structure(
            "Point",
            vec![
                ("x".to_string(), MirType::F64),
                ("y".to_string(), MirType::F64),
            ],
        );
        module.add_type(TypeDef::new("Point", point_type));

        assert!(module.get_type("Point").is_some());
    }

    #[test]
    fn test_constants() {
        let mut module = MirModule::new("test");

        module.add_constant(ConstantDef::new(
            "GRAVITY",
            MirType::F64,
            ConstValue::Float(9.81),
        ));

        assert_eq!(module.constants.len(), 1);
        assert_eq!(module.constants[0].value.as_float(), Some(9.81));
    }

    #[test]
    fn test_global_attributes() {
        let global = GlobalDecl::new("counter", MirType::I64)
            .with_initializer(ConstValue::Int(0))
            .with_linkage(Linkage::External)
            .with_section(".data")
            .thread_local();

        assert!(global.thread_local);
        assert_eq!(global.linkage, Linkage::External);
        assert_eq!(global.section, Some(".data".to_string()));
    }
}
