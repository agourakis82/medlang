//! MIR Functions
//!
//! MIR functions contain the CFG of basic blocks representing the function body.

use std::collections::HashMap;

use super::block::*;
use super::inst::{Instruction, Operation};
use super::types::*;
use super::value::*;

/// A MIR function
#[derive(Clone, Debug)]
pub struct MirFunction {
    /// Function name
    pub name: String,
    /// Function signature
    pub signature: FunctionSignature,
    /// Basic blocks (CFG)
    pub blocks: Vec<BasicBlock>,
    /// Local variable declarations
    pub locals: Vec<LocalDecl>,
    /// Debug info
    pub debug_info: Option<FunctionDebugInfo>,
    /// Attributes
    pub attributes: FunctionAttributes,
}

impl MirFunction {
    /// Create a new function
    pub fn new(name: &str, signature: FunctionSignature) -> Self {
        Self {
            name: name.to_string(),
            signature,
            blocks: Vec::new(),
            locals: Vec::new(),
            debug_info: None,
            attributes: FunctionAttributes::default(),
        }
    }

    /// Add a basic block
    pub fn add_block(&mut self, block: BasicBlock) -> BlockId {
        let id = block.id;
        self.blocks.push(block);
        id
    }

    /// Add a local variable
    pub fn add_local(&mut self, decl: LocalDecl) -> usize {
        let idx = self.locals.len();
        self.locals.push(decl);
        idx
    }

    /// Get the entry block
    pub fn entry_block(&self) -> Option<&BasicBlock> {
        self.blocks.first()
    }

    /// Get the entry block mutably
    pub fn entry_block_mut(&mut self) -> Option<&mut BasicBlock> {
        self.blocks.first_mut()
    }

    /// Get a block by ID
    pub fn block(&self, id: BlockId) -> Option<&BasicBlock> {
        self.blocks.iter().find(|b| b.id == id)
    }

    /// Get a block mutably by ID
    pub fn block_mut(&mut self, id: BlockId) -> Option<&mut BasicBlock> {
        self.blocks.iter_mut().find(|b| b.id == id)
    }

    /// Number of blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Number of locals
    pub fn num_locals(&self) -> usize {
        self.locals.len()
    }

    /// Is this function a declaration only (no body)?
    pub fn is_declaration(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Get all return types
    pub fn return_type(&self) -> &MirType {
        &self.signature.return_type
    }

    /// Get parameter types
    pub fn param_types(&self) -> &[MirType] {
        &self.signature.params
    }

    /// Compute predecessor map
    pub fn predecessors(&self) -> HashMap<BlockId, Vec<BlockId>> {
        let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();

        // Initialize all blocks
        for block in &self.blocks {
            preds.entry(block.id).or_default();
        }

        // Build predecessor lists
        for block in &self.blocks {
            for succ in block.successors() {
                preds.entry(succ).or_default().push(block.id);
            }
        }

        preds
    }

    /// Check if function is well-formed (basic validation)
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Must have at least one block
        if self.blocks.is_empty() && !self.attributes.is_external {
            return Err(ValidationError::EmptyFunction);
        }

        // Entry block must be first with ID 0
        if let Some(entry) = self.blocks.first() {
            if entry.id != BlockId::ENTRY {
                return Err(ValidationError::InvalidEntryBlock);
            }
        }

        // All blocks must be terminated
        for block in &self.blocks {
            if !block.is_terminated() {
                return Err(ValidationError::UnterminatedBlock(block.id));
            }
        }

        // Check that all referenced blocks exist
        let block_ids: std::collections::HashSet<BlockId> =
            self.blocks.iter().map(|b| b.id).collect();
        for block in &self.blocks {
            for succ in block.successors() {
                if !block_ids.contains(&succ) {
                    return Err(ValidationError::InvalidBlockReference(succ));
                }
            }
        }

        Ok(())
    }

    /// Get CFG as DOT graph for visualization
    pub fn to_dot(&self) -> String {
        let mut dot = String::new();
        dot.push_str(&format!("digraph {} {{\n", self.name));
        dot.push_str("  node [shape=box];\n");

        for block in &self.blocks {
            let default_label = format!("bb{}", block.id.0);
            let label = block.name.as_deref().unwrap_or(&default_label);
            dot.push_str(&format!("  {} [label=\"{}\"];\n", block.id, label));

            for succ in block.successors() {
                dot.push_str(&format!("  {} -> {};\n", block.id, succ));
            }
        }

        dot.push_str("}\n");
        dot
    }
}

/// Function signature
#[derive(Clone, Debug)]
pub struct FunctionSignature {
    /// Parameter types
    pub params: Vec<MirType>,
    /// Parameter names (for debug info)
    pub param_names: Vec<String>,
    /// Return type
    pub return_type: MirType,
    /// Calling convention
    pub calling_convention: CallingConvention,
    /// Is variadic?
    pub variadic: bool,
}

impl FunctionSignature {
    pub fn new(params: Vec<MirType>, return_type: MirType) -> Self {
        let param_names = (0..params.len()).map(|i| format!("arg{}", i)).collect();
        Self {
            params,
            param_names,
            return_type,
            calling_convention: CallingConvention::MedLang,
            variadic: false,
        }
    }

    pub fn with_names(mut self, names: Vec<String>) -> Self {
        self.param_names = names;
        self
    }

    pub fn with_convention(mut self, cc: CallingConvention) -> Self {
        self.calling_convention = cc;
        self
    }

    pub fn variadic(mut self) -> Self {
        self.variadic = true;
        self
    }

    /// Get arity (number of parameters)
    pub fn arity(&self) -> usize {
        self.params.len()
    }
}

/// Local variable declaration
#[derive(Clone, Debug)]
pub struct LocalDecl {
    /// Variable name (for debug info)
    pub name: String,
    /// Type
    pub ty: MirType,
    /// Is this a mutable local?
    pub mutable: bool,
    /// Source location
    pub span: Option<Span>,
    /// Scope where this local is visible
    pub scope: ScopeId,
}

impl LocalDecl {
    pub fn new(name: &str, ty: MirType) -> Self {
        Self {
            name: name.to_string(),
            ty,
            mutable: true,
            span: None,
            scope: ScopeId::ROOT,
        }
    }

    pub fn immutable(mut self) -> Self {
        self.mutable = false;
        self
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_scope(mut self, scope: ScopeId) -> Self {
        self.scope = scope;
        self
    }
}

/// Debug info for a function
#[derive(Clone, Debug)]
pub struct FunctionDebugInfo {
    /// Source file path
    pub file: String,
    /// Line number
    pub line: u32,
    /// Column number
    pub column: u32,
    /// Scope tree
    pub scopes: Vec<ScopeInfo>,
}

impl FunctionDebugInfo {
    pub fn new(file: &str, line: u32, column: u32) -> Self {
        Self {
            file: file.to_string(),
            line,
            column,
            scopes: vec![ScopeInfo {
                id: ScopeId::ROOT,
                parent: None,
                span: None,
            }],
        }
    }

    pub fn add_scope(&mut self, parent: ScopeId, span: Option<Span>) -> ScopeId {
        let id = ScopeId((self.scopes.len()) as u32);
        self.scopes.push(ScopeInfo {
            id,
            parent: Some(parent),
            span,
        });
        id
    }
}

/// Scope info for debug
#[derive(Clone, Debug)]
pub struct ScopeInfo {
    pub id: ScopeId,
    pub parent: Option<ScopeId>,
    pub span: Option<Span>,
}

/// Function attributes
#[derive(Clone, Debug, Default)]
pub struct FunctionAttributes {
    /// Is this an external function?
    pub is_external: bool,
    /// Is this function pure (no side effects)?
    pub is_pure: bool,
    /// Should this function be inlined?
    pub inline: InlineHint,
    /// Is this function unsafe?
    pub is_unsafe: bool,
    /// Is this a const function?
    pub is_const: bool,
    /// Target CPU features required
    pub target_features: Vec<String>,
    /// Link section
    pub section: Option<String>,
    /// Export name (for FFI)
    pub export_name: Option<String>,
    /// Optimization level
    pub opt_level: OptLevel,
}

/// Inline hint
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InlineHint {
    #[default]
    None,
    Always,
    Never,
    Hint,
}

/// Optimization level
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptLevel {
    None,
    #[default]
    Default,
    Size,
    Speed,
}

/// Validation error
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationError {
    EmptyFunction,
    InvalidEntryBlock,
    UnterminatedBlock(BlockId),
    InvalidBlockReference(BlockId),
    TypeMismatch { expected: MirType, found: MirType },
    UndefinedValue(ValueId),
    InvalidOperandCount { expected: usize, found: usize },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::EmptyFunction => write!(f, "function has no basic blocks"),
            ValidationError::InvalidEntryBlock => {
                write!(f, "entry block must have ID 0")
            }
            ValidationError::UnterminatedBlock(id) => {
                write!(f, "block {} is not terminated", id)
            }
            ValidationError::InvalidBlockReference(id) => {
                write!(f, "reference to undefined block {}", id)
            }
            ValidationError::TypeMismatch { expected, found } => {
                write!(f, "type mismatch: expected {}, found {}", expected, found)
            }
            ValidationError::UndefinedValue(id) => {
                write!(f, "reference to undefined value {}", id)
            }
            ValidationError::InvalidOperandCount { expected, found } => {
                write!(
                    f,
                    "invalid operand count: expected {}, found {}",
                    expected, found
                )
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Builder for constructing MIR functions
pub struct FunctionBuilder {
    function: MirFunction,
    value_gen: ValueIdGen,
    block_gen: BlockIdGen,
    current_block: Option<BlockId>,
}

impl FunctionBuilder {
    pub fn new(name: &str, signature: FunctionSignature) -> Self {
        let mut function = MirFunction::new(name, signature);
        let mut block_gen = BlockIdGen::new();
        let mut value_gen = ValueIdGen::new();

        // Create entry block with parameters
        let entry_id = block_gen.next();
        let mut entry = BasicBlock::new(entry_id);

        // Add function parameters as block parameters
        for ty in &function.signature.params {
            let param_id = value_gen.next();
            entry.add_param(param_id, ty.clone());
        }

        function.add_block(entry);

        Self {
            function,
            value_gen,
            block_gen,
            current_block: Some(entry_id),
        }
    }

    /// Create a new basic block
    pub fn create_block(&mut self) -> BlockId {
        let id = self.block_gen.next();
        self.function.add_block(BasicBlock::new(id));
        id
    }

    /// Create a new basic block with a name
    pub fn create_named_block(&mut self, name: &str) -> BlockId {
        let id = self.block_gen.next();
        self.function.add_block(BasicBlock::with_name(id, name));
        id
    }

    /// Switch to a different block
    pub fn switch_to(&mut self, block: BlockId) {
        self.current_block = Some(block);
    }

    /// Get current block
    pub fn current_block(&self) -> Option<BlockId> {
        self.current_block
    }

    /// Add a block parameter
    pub fn block_param(&mut self, ty: MirType) -> ValueId {
        let value = self.value_gen.next();
        if let Some(block_id) = self.current_block {
            if let Some(block) = self.function.block_mut(block_id) {
                block.add_param(value, ty);
            }
        }
        value
    }

    /// Add an instruction to current block
    pub fn push_op(&mut self, op: Operation, ty: MirType) -> ValueId {
        let result = self.value_gen.next();
        let inst = Instruction::new(op, ty).with_result(result);

        if let Some(block_id) = self.current_block {
            if let Some(block) = self.function.block_mut(block_id) {
                block.push(inst);
            }
        }

        result
    }

    /// Add a void instruction
    pub fn push_void(&mut self, op: Operation) {
        let inst = Instruction::new(op, MirType::Void);

        if let Some(block_id) = self.current_block {
            if let Some(block) = self.function.block_mut(block_id) {
                block.push(inst);
            }
        }
    }

    /// Terminate current block
    pub fn terminate(&mut self, terminator: Terminator) {
        if let Some(block_id) = self.current_block {
            if let Some(block) = self.function.block_mut(block_id) {
                block.terminate(terminator);
            }
        }
    }

    /// Add a local variable
    pub fn add_local(&mut self, name: &str, ty: MirType) -> usize {
        self.function.add_local(LocalDecl::new(name, ty))
    }

    /// Get function parameter value
    pub fn param(&self, idx: usize) -> Option<ValueId> {
        self.function
            .entry_block()
            .and_then(|b| b.params.get(idx))
            .map(|p| p.value)
    }

    /// Finish building and return the function
    pub fn build(self) -> MirFunction {
        self.function
    }

    /// Build with validation
    pub fn build_validated(self) -> Result<MirFunction, ValidationError> {
        self.function.validate()?;
        Ok(self.function)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::inst::Operation;

    #[test]
    fn test_function_creation() {
        let sig = FunctionSignature::new(vec![MirType::F64, MirType::F64], MirType::F64);
        let func = MirFunction::new("add", sig);

        assert_eq!(func.name, "add");
        assert!(func.is_declaration());
    }

    #[test]
    fn test_function_builder() {
        let sig = FunctionSignature::new(vec![MirType::F64, MirType::F64], MirType::F64);
        let mut builder = FunctionBuilder::new("add", sig);

        // Get parameters
        let x = builder.param(0).unwrap();
        let y = builder.param(1).unwrap();

        // Add instruction
        let sum = builder.push_op(Operation::FAdd { lhs: x, rhs: y }, MirType::F64);

        // Terminate
        builder.terminate(Terminator::Return { value: Some(sum) });

        let func = builder.build();
        assert_eq!(func.num_blocks(), 1);
        assert_eq!(func.entry_block().unwrap().instructions.len(), 1);
    }

    #[test]
    fn test_function_validation() {
        let sig = FunctionSignature::new(vec![], MirType::Void);
        let mut builder = FunctionBuilder::new("empty", sig);
        builder.terminate(Terminator::Return { value: None });

        let func = builder.build_validated();
        assert!(func.is_ok());
    }

    #[test]
    fn test_function_with_branches() {
        let sig = FunctionSignature::new(vec![MirType::Bool], MirType::I32);
        let mut builder = FunctionBuilder::new("conditional", sig);

        let cond = builder.param(0).unwrap();

        // Create then/else blocks
        let then_block = builder.create_named_block("then");
        let else_block = builder.create_named_block("else");
        let merge_block = builder.create_named_block("merge");

        // Branch in entry
        builder.terminate(Terminator::Branch {
            cond,
            then_block,
            then_args: vec![],
            else_block,
            else_args: vec![],
        });

        // Then block
        builder.switch_to(then_block);
        let then_val = builder.push_op(
            Operation::ConstInt {
                value: 1,
                ty: MirType::I32,
            },
            MirType::I32,
        );
        builder.terminate(Terminator::Goto {
            target: merge_block,
            args: vec![then_val],
        });

        // Else block
        builder.switch_to(else_block);
        let else_val = builder.push_op(
            Operation::ConstInt {
                value: 0,
                ty: MirType::I32,
            },
            MirType::I32,
        );
        builder.terminate(Terminator::Goto {
            target: merge_block,
            args: vec![else_val],
        });

        // Merge block with phi via block param
        builder.switch_to(merge_block);
        let result = builder.block_param(MirType::I32);
        builder.terminate(Terminator::Return {
            value: Some(result),
        });

        let func = builder.build_validated().unwrap();
        assert_eq!(func.num_blocks(), 4);
    }

    #[test]
    fn test_predecessors() {
        let sig = FunctionSignature::new(vec![], MirType::Void);
        let mut builder = FunctionBuilder::new("test", sig);

        let bb1 = builder.create_block();
        let bb2 = builder.create_block();

        // Entry -> bb1, bb2
        let cond = builder.push_op(Operation::ConstBool { value: true }, MirType::Bool);
        builder.terminate(Terminator::Branch {
            cond,
            then_block: bb1,
            then_args: vec![],
            else_block: bb2,
            else_args: vec![],
        });

        builder.switch_to(bb1);
        builder.terminate(Terminator::Return { value: None });

        builder.switch_to(bb2);
        builder.terminate(Terminator::Return { value: None });

        let func = builder.build();
        let preds = func.predecessors();

        assert!(preds[&BlockId::ENTRY].is_empty());
        assert_eq!(preds[&bb1], vec![BlockId::ENTRY]);
        assert_eq!(preds[&bb2], vec![BlockId::ENTRY]);
    }

    #[test]
    fn test_to_dot() {
        let sig = FunctionSignature::new(vec![], MirType::Void);
        let mut builder = FunctionBuilder::new("graph_test", sig);
        builder.terminate(Terminator::Return { value: None });

        let func = builder.build();
        let dot = func.to_dot();

        assert!(dot.contains("digraph graph_test"));
        assert!(dot.contains("bb0"));
    }

    #[test]
    fn test_function_signature() {
        let sig = FunctionSignature::new(vec![MirType::F64, MirType::I32], MirType::Bool)
            .with_names(vec!["x".to_string(), "n".to_string()])
            .with_convention(CallingConvention::C);

        assert_eq!(sig.arity(), 2);
        assert_eq!(sig.param_names[0], "x");
        assert_eq!(sig.calling_convention, CallingConvention::C);
    }
}
