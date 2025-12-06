# Phase V2 Architecture: Advanced Verification & Performance

**Version**: 0.6.0 (Phase V2)  
**Date**: December 6, 2024  
**Status**: Design Phase

---

## Executive Summary

Phase V2 extends MedLang with advanced verification and performance capabilities:

1. **Z3 SMT Solver Integration** - Mathematical proof of refinement constraints
2. **LSP (Language Server Protocol)** - Professional IDE integration
3. **Cranelift JIT Compilation** - Interactive REPL with hot-reloading
4. **GPU Code Generation** - CUDA/SPIR-V for population simulations

**Goals**:
- Mathematical certainty for clinical safety properties
- Professional developer experience (VS Code, IntelliJ, etc.)
- 100x faster iteration with JIT-compiled REPL
- 1000x speedup for 10,000+ subject simulations on GPU

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MedLang Compiler v0.6.0                      │
│                  (Phase V2: Verification & Perf)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Phase V1     │    │  Phase V2     │    │  Phase V2     │
│  Pipeline     │    │  Verification │    │  Performance  │
│  (Existing)   │    │               │    │               │
├───────────────┤    ├───────────────┤    ├───────────────┤
│ Lexer         │    │ Z3 Solver     │    │ LSP Server    │
│ Parser        │    │ SMT-LIB Gen   │    │ JIT Compiler  │
│ Type Checker  │    │ VC Generation │    │ GPU Codegen   │
│ V1 Extensions │    │ Counterexample│    │ REPL          │
│ IR Lowering   │    │               │    │               │
│ Codegen       │    │               │    │               │
│ (Stan/Julia)  │    │               │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Output Files  │
                    ├─────────────────┤
                    │ • Stan/Julia    │
                    │ • Proof logs    │
                    │ • LSP responses │
                    │ • JIT binary    │
                    │ • GPU kernels   │
                    └─────────────────┘
```

---

## Module 1: Z3 SMT Solver Integration

### Goals

- Mathematically prove refinement constraints at compile time
- Generate counterexamples when constraints cannot be proven
- Ensure division-by-zero safety, range violations detected
- Integration with existing Phase V1 refinement type checker

### Design

#### 1.1 SMT-LIB Translation

**New Module**: `compiler/src/smt/mod.rs`

**Key Components**:
```rust
pub mod translator;  // MedLang → SMT-LIB
pub mod solver;      // Z3 solver interface
pub mod vc_gen;      // Verification condition generation

pub struct SMTContext {
    pub solver: Z3Solver,
    pub assertions: Vec<SMTFormula>,
    pub variables: HashMap<String, SMTSort>,
}

pub enum SMTSort {
    Real,
    Int,
    Bool,
    Custom(String),  // For unit types
}

pub enum SMTFormula {
    Comparison { lhs: SMTExpr, op: ComparisonOp, rhs: SMTExpr },
    Logical { op: LogicalOp, operands: Vec<SMTFormula> },
    Quantified { vars: Vec<(String, SMTSort)>, body: Box<SMTFormula> },
}
```

**Translation Examples**:

```medlang
// MedLang refinement
param CL : Clearance where CL > 0.0_L_per_h
```

↓ Translates to SMT-LIB ↓

```smt2
(declare-const CL Real)
(assert (> CL 0.0))
```

```medlang
// Complex constraint
param WT : Mass where WT in 30.0_kg..200.0_kg
```

↓

```smt2
(declare-const WT Real)
(assert (and (>= WT 30.0) (<= WT 200.0)))
```

#### 1.2 Verification Condition Generation

**Purpose**: Generate proof obligations from MedLang code

**Example**:
```medlang
param CL : Clearance where CL > 0.0_L_per_h
param V : Volume where V > 0.0_L

obs C : ConcMass = DOSE / V
```

**Generated VC**:
```
Assume: V > 0.0
Prove: V ≠ 0.0  (division safety)
```

**Implementation**:
```rust
pub struct VerificationCondition {
    pub assumptions: Vec<SMTFormula>,
    pub goal: SMTFormula,
    pub location: SourceLocation,
    pub description: String,
}

impl VCGenerator {
    pub fn generate_division_safety(
        &self,
        numerator: &IRExpr,
        denominator: &IRExpr,
        location: SourceLocation,
    ) -> VerificationCondition {
        // Extract constraint from denominator variable
        // Generate: denominator ≠ 0
        // ...
    }
}
```

#### 1.3 Z3 Solver Interface

**Dependencies**: Add to `Cargo.toml`:
```toml
z3 = { version = "0.12", features = ["static-link-z3"] }
```

**Implementation**:
```rust
use z3::{Config, Context, Solver};

pub struct Z3Solver {
    ctx: Context,
    solver: Solver<'static>,
}

impl Z3Solver {
    pub fn check_vc(&mut self, vc: &VerificationCondition) -> Z3Result {
        // Add assumptions
        for assumption in &vc.assumptions {
            self.solver.assert(&self.translate(assumption));
        }
        
        // Negate goal (to find counterexample)
        let negated_goal = self.negate(&self.translate(&vc.goal));
        self.solver.assert(&negated_goal);
        
        // Check satisfiability
        match self.solver.check() {
            SatResult::Unsat => Z3Result::Proven,
            SatResult::Sat => {
                let model = self.solver.get_model().unwrap();
                Z3Result::Counterexample(model)
            }
            SatResult::Unknown => Z3Result::Unknown,
        }
    }
}

pub enum Z3Result {
    Proven,
    Counterexample(Z3Model),
    Unknown,
}
```

#### 1.4 Integration with Type Checker

**Modified**: `compiler/src/typeck_v1.rs`

```rust
impl V1TypeChecker {
    pub fn verify_constraints_with_smt(&mut self) -> Result<(), TypeError> {
        let mut smt_ctx = SMTContext::new();
        
        // Translate all refinement constraints to SMT
        for (var_name, constraints) in &self.refinement_constraints {
            for constraint in constraints {
                let smt_formula = translate_constraint(constraint);
                smt_ctx.add_assertion(var_name, smt_formula);
            }
        }
        
        // Generate and check verification conditions
        let vcs = self.generate_verification_conditions();
        for vc in vcs {
            match smt_ctx.check(vc) {
                Z3Result::Proven => {
                    println!("✓ Verified: {}", vc.description);
                }
                Z3Result::Counterexample(model) => {
                    return Err(TypeError::ConstraintViolation {
                        message: format!("Cannot prove: {}", vc.description),
                        counterexample: Some(model),
                    });
                }
                Z3Result::Unknown => {
                    // Warn but don't fail
                    eprintln!("⚠ Could not verify: {}", vc.description);
                }
            }
        }
        
        Ok(())
    }
}
```

#### 1.5 User Experience

**CLI Flag**: `--verify` or `--smt`

```bash
$ mlc compile model.medlang --verify
Stage 1: Tokenization...
  ✓ 288 tokens generated
Stage 2: Parsing...
  ✓ AST constructed
Stage 3: Type checking...
  ✓ Type checking passed
Stage 4: SMT Verification...
  ✓ Verified: CL > 0.0 (parameter constraint)
  ✓ Verified: V > 0.0 (parameter constraint)
  ✓ Verified: V ≠ 0.0 (division safety in obs C)
  ✓ Verified: WT in [30.0, 200.0] (range constraint)
  ✓ All constraints proven mathematically
Stage 5: Code generation...
  ✓ 107 lines of Stan code generated
```

**Error with Counterexample**:
```bash
$ mlc compile bad_model.medlang --verify
...
Stage 4: SMT Verification...
  ✗ Cannot prove: denominator ≠ 0 in expression DOSE / CL
  
  Counterexample found:
    CL = 0.0
    
  Hint: Add constraint "where CL > 0.0" to param declaration
  
✗ Compilation failed
```

---

## Module 2: Language Server Protocol (LSP)

### Goals

- Professional IDE integration (VS Code, IntelliJ, Emacs, Vim)
- Real-time diagnostics as user types
- Hover information (types, effects, confidence, constraints)
- Code completion with effect-aware suggestions
- Go-to-definition, find-references
- Refactoring (rename, extract function)

### Design

#### 2.1 LSP Server Architecture

**New Module**: `compiler/src/lsp/mod.rs`

**Dependencies**: Add to `Cargo.toml`:
```toml
tower-lsp = "0.20"
tokio = { version = "1", features = ["full"] }
serde_json = "1.0"
```

**Core Structure**:
```rust
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

pub struct MedLangLspBackend {
    client: Client,
    document_cache: Arc<RwLock<HashMap<Url, DocumentState>>>,
}

pub struct DocumentState {
    pub source: String,
    pub ast: Option<Program>,
    pub ir: Option<IRProgram>,
    pub diagnostics: Vec<Diagnostic>,
    pub type_info: HashMap<Range, InferredType>,
    pub effect_info: HashMap<Range, EffectSet>,
    pub epistemic_info: HashMap<Range, EpistemicMetadata>,
}

#[tower_lsp::async_trait]
impl LanguageServer for MedLangLspBackend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions::default()),
                definition_provider: Some(OneOf::Left(true)),
                references_provider: Some(OneOf::Left(true)),
                rename_provider: Some(OneOf::Left(true)),
                diagnostic_provider: Some(DiagnosticServerCapabilities::Options(
                    DiagnosticOptions {
                        identifier: Some("medlang".to_string()),
                        inter_file_dependencies: false,
                        workspace_diagnostics: false,
                        work_done_progress_options: WorkDoneProgressOptions::default(),
                    },
                )),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "MedLang Language Server".to_string(),
                version: Some("0.6.0".to_string()),
            }),
        })
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.on_change(params.text_document.uri, params.text_document.text).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let text = params.content_changes[0].text.clone();
        self.on_change(params.text_document.uri, text).await;
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        // Return type, effects, confidence info
        // ...
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        // Effect-aware completion suggestions
        // ...
    }

    // ... other LSP methods
}
```

#### 2.2 Real-Time Diagnostics

**Implementation**:
```rust
impl MedLangLspBackend {
    async fn on_change(&self, uri: Url, text: String) {
        let diagnostics = self.compile_and_diagnose(&text).await;
        
        self.client.publish_diagnostics(uri.clone(), diagnostics, None).await;
        
        // Cache document state
        let mut cache = self.document_cache.write().await;
        cache.insert(uri, DocumentState {
            source: text,
            ast: /* parsed AST */,
            ir: /* lowered IR */,
            diagnostics,
            // ...
        });
    }
    
    async fn compile_and_diagnose(&self, source: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        
        // Tokenization errors
        match tokenize(source) {
            Ok(tokens) => {
                // Parsing errors
                match parse(&tokens) {
                    Ok(ast) => {
                        // Type checking errors
                        match type_check(&ast) {
                            Ok(_) => {},
                            Err(e) => diagnostics.push(type_error_to_diagnostic(e)),
                        }
                    }
                    Err(e) => diagnostics.push(parse_error_to_diagnostic(e)),
                }
            }
            Err(e) => diagnostics.push(lex_error_to_diagnostic(e)),
        }
        
        diagnostics
    }
}
```

#### 2.3 Hover Information

**Example** (user hovers over `C_plasma`):

```medlang
obs C_plasma : Knowledge<ConcMass> = A_central / V
//  ^^^^^^^^ <-- hover here
```

**Hover response**:
```markdown
### `C_plasma`

**Type**: `Knowledge<ConcMass>`  
**Base Type**: `ConcMass` (Mass/Length³)  
**Confidence**: `0.87` (propagated)  
**Effects**: `Pure`  

**Provenance**:
- Computed { operation: "div", inputs: ["A_central", "V"] }

**Constraints**:
- `V > 0.0` (proven by refinement type)

**Dependencies**:
- `A_central` : DoseMass
- `V` : Volume where V > 0.0_L
```

#### 2.4 VS Code Extension

**New Directory**: `vscode-extension/`

**Structure**:
```
vscode-extension/
├── package.json
├── src/
│   └── extension.ts
├── syntaxes/
│   └── medlang.tmLanguage.json
└── language-configuration.json
```

**package.json**:
```json
{
  "name": "medlang",
  "displayName": "MedLang",
  "description": "MedLang language support",
  "version": "0.6.0",
  "engines": { "vscode": "^1.75.0" },
  "categories": ["Programming Languages"],
  "activationEvents": ["onLanguage:medlang"],
  "main": "./out/extension.js",
  "contributes": {
    "languages": [{
      "id": "medlang",
      "aliases": ["MedLang", "medlang"],
      "extensions": [".medlang"],
      "configuration": "./language-configuration.json"
    }],
    "grammars": [{
      "language": "medlang",
      "scopeName": "source.medlang",
      "path": "./syntaxes/medlang.tmLanguage.json"
    }]
  }
}
```

**extension.ts**:
```typescript
import * as path from 'path';
import { workspace, ExtensionContext } from 'vscode';
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: ExtensionContext) {
  const serverExecutable = path.join(context.extensionPath, 'server', 'mlc-lsp');
  
  const serverOptions: ServerOptions = {
    run: { command: serverExecutable },
    debug: { command: serverExecutable },
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: 'file', language: 'medlang' }],
  };

  client = new LanguageClient(
    'medlang',
    'MedLang Language Server',
    serverOptions,
    clientOptions
  );

  client.start();
}

export function deactivate(): Thenable<void> | undefined {
  if (!client) {
    return undefined;
  }
  return client.stop();
}
```

---

## Module 3: Cranelift JIT Compilation

### Goals

- Interactive REPL for rapid model development
- Hot-reloading for instant feedback
- 100x faster than Stan compilation for single-subject simulations
- Visualization hooks for plotting

### Design

#### 3.1 Cranelift Integration

**New Module**: `compiler/src/jit/mod.rs`

**Dependencies**: Add to `Cargo.toml`:
```toml
cranelift = "0.103"
cranelift-module = "0.103"
cranelift-jit = "0.103"
cranelift-codegen = "0.103"
```

**Architecture**:
```rust
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

pub struct JITCompiler {
    module: JITModule,
    builder_context: FunctionBuilderContext,
    ctx: codegen::Context,
}

impl JITCompiler {
    pub fn compile_model(&mut self, ir: &IRProgram) -> Result<CompiledModel> {
        // Translate IR to Cranelift IR
        let func = self.translate_ode_system(&ir.model.odes)?;
        
        // Define function
        let id = self.module.declare_function(
            "solve_ode",
            Linkage::Export,
            &func.signature
        )?;
        
        self.module.define_function(id, &mut self.ctx)?;
        self.module.finalize_definitions();
        
        // Get function pointer
        let code_ptr = self.module.get_finalized_function(id);
        
        Ok(CompiledModel {
            solve_ode: unsafe { std::mem::transmute(code_ptr) },
            state_count: ir.model.states.len(),
            param_count: ir.model.params.len(),
        })
    }
    
    fn translate_ode_system(&mut self, odes: &[ODE]) -> Result<Function> {
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
        
        // Function signature: fn(t: f64, y: *const f64, params: *const f64, dydt: *mut f64)
        // ...
        
        for (i, ode) in odes.enumerate() {
            let dydt_expr = self.translate_expr(&ode.rhs, &builder)?;
            // Store to dydt[i]
        }
        
        builder.finalize();
        Ok(self.ctx.func.clone())
    }
}

pub struct CompiledModel {
    solve_ode: extern "C" fn(f64, *const f64, *const f64, *mut f64),
    state_count: usize,
    param_count: usize,
}
```

#### 3.2 REPL Implementation

**New Binary**: `compiler/src/bin/mlc-repl.rs`

**Implementation**:
```rust
use rustyline::Editor;
use rustyline::error::ReadlineError;

struct ReplState {
    jit: JITCompiler,
    loaded_models: HashMap<String, CompiledModel>,
    variables: HashMap<String, f64>,
}

fn main() {
    let mut rl = Editor::<()>::new();
    let mut state = ReplState::new();
    
    println!("MedLang REPL v0.6.0 (JIT mode)");
    println!("Type 'help' for commands, 'exit' to quit");
    
    loop {
        let readline = rl.readline(">>> ");
        match readline {
            Ok(line) => {
                if line.trim() == "exit" {
                    break;
                }
                
                match execute_command(&line, &mut state) {
                    Ok(result) => println!("{}", result),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }
}

fn execute_command(cmd: &str, state: &mut ReplState) -> Result<String> {
    if cmd.starts_with("load ") {
        let path = cmd.strip_prefix("load ").unwrap();
        let source = std::fs::read_to_string(path)?;
        let ir = compile_to_ir(&source)?;
        let model = state.jit.compile_model(&ir)?;
        state.loaded_models.insert("current".to_string(), model);
        Ok(format!("✓ Loaded model from {}", path))
    } else if cmd.starts_with("simulate ") {
        // Parse parameters: simulate dose=100.0_mg weight=70.0_kg
        let params = parse_params(&cmd)?;
        let result = state.loaded_models["current"].simulate(params)?;
        Ok(format_simulation_result(result))
    } else if cmd == "plot" {
        // Generate plot using plotters
        plot_current_simulation(state)?;
        Ok("✓ Plot generated".to_string())
    } else {
        Err(anyhow!("Unknown command"))
    }
}
```

#### 3.3 REPL Example Session

```bash
$ mlc repl --jit
MedLang REPL v0.6.0 (JIT mode)
Type 'help' for commands, 'exit' to quit

>>> load examples/one_comp_oral_pk.medlang
✓ Loaded model from examples/one_comp_oral_pk.medlang
Compiled in 15ms (JIT)

>>> simulate dose=150.0_mg weight=80.0_kg
Running simulation...
✓ Completed in 2ms

Results:
  C_max: 8.3 mg/L
  T_max: 2.1 h
  AUC_0-24: 98.5 mg*h/L
  
  Time points:
    0.0h: 0.0 mg/L
    1.0h: 5.2 mg/L
    2.0h: 8.1 mg/L
    4.0h: 7.3 mg/L
    8.0h: 4.1 mg/L
    24.0h: 0.3 mg/L

>>> plot
✓ Plot saved to simulation.png

>>> set CL = 15.0_L_per_h
✓ CL updated

>>> simulate dose=150.0_mg weight=80.0_kg
Running simulation...
✓ Completed in 2ms

Results:
  C_max: 6.1 mg/L (decreased due to higher CL)
  ...

>>> exit
Goodbye!
```

---

## Module 4: GPU Code Generation

### Goals

- Generate CUDA kernels for population simulations
- SPIR-V backend for portable GPU code (OpenCL, Vulkan)
- 1000x speedup for 10,000+ subject simulations
- Automatic memory management for device arrays

### Design

#### 4.1 CUDA Code Generator

**New Module**: `compiler/src/codegen/cuda.rs`

**Dependencies**: Add to `Cargo.toml`:
```toml
cuda-sys = "0.3"  # Low-level CUDA bindings
```

**Implementation**:
```rust
pub struct CudaCodegen {
    kernel_count: usize,
}

impl CudaCodegen {
    pub fn generate(&mut self, ir: &IRProgram) -> Result<String> {
        let mut cuda_code = String::new();
        
        // Headers
        cuda_code.push_str("#include <cuda_runtime.h>\n");
        cuda_code.push_str("#include <math.h>\n\n");
        
        // ODE system kernel
        cuda_code.push_str(&self.generate_ode_kernel(&ir.model)?);
        
        // Population simulation kernel
        cuda_code.push_str(&self.generate_population_kernel(&ir)?);
        
        // Host wrapper functions
        cuda_code.push_str(&self.generate_host_wrapper()?);
        
        Ok(cuda_code)
    }
    
    fn generate_ode_kernel(&mut self, model: &IRModel) -> Result<String> {
        let mut code = String::new();
        
        code.push_str("__device__ void ode_system(\n");
        code.push_str("    float t,\n");
        code.push_str("    float* y,\n");
        code.push_str("    float* params,\n");
        code.push_str("    float* dydt\n");
        code.push_str(") {\n");
        
        // Generate ODE right-hand sides
        for (i, ode) in model.odes.iter().enumerate() {
            code.push_str(&format!("    dydt[{}] = ", i));
            code.push_str(&self.translate_expr_cuda(&ode.rhs)?);
            code.push_str(";\n");
        }
        
        code.push_str("}\n\n");
        Ok(code)
    }
    
    fn generate_population_kernel(&mut self, ir: &IRProgram) -> Result<String> {
        Ok(format!(r#"
__global__ void simulate_population(
    float* params_array,     // [n_subjects * n_params]
    float* initial_states,   // [n_subjects * n_states]
    float* results,          // [n_subjects * n_timepoints]
    int n_subjects,
    int n_timepoints,
    float* timepoints
) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n_subjects) {{
        // Extract subject parameters
        float* subject_params = &params_array[tid * {}];
        float* subject_y0 = &initial_states[tid * {}];
        
        // Allocate shared memory for state
        float y[{}];
        float dydt[{}];
        
        // Initialize
        for (int i = 0; i < {}; i++) {{
            y[i] = subject_y0[i];
        }}
        
        // Solve ODE using RK4
        for (int t_idx = 0; t_idx < n_timepoints; t_idx++) {{
            float t = timepoints[t_idx];
            
            // RK4 step
            // ... (RK4 implementation)
            
            // Store result
            results[tid * n_timepoints + t_idx] = y[0] / subject_params[V_INDEX];
        }}
    }}
}}
"#, 
        ir.model.params.len(),
        ir.model.states.len(),
        ir.model.states.len(),
        ir.model.states.len(),
        ir.model.states.len()
        ))
    }
}
```

#### 4.2 SPIR-V Backend

**New Module**: `compiler/src/codegen/spirv.rs`

**Dependencies**:
```toml
rspirv = "0.11"
spirv-tools = "0.10"
```

**Purpose**: Generate portable GPU code that works on:
- OpenCL (Intel, AMD, NVIDIA)
- Vulkan (cross-platform)
- Metal (via translation)

#### 4.3 MedLang GPU Syntax

**New Keyword**: `with GPU`

```medlang
// Mark function for GPU execution
fn simulate_subject_gpu(
    params: PKParams,
    initial: StateVector
) : Concentration with GPU {
    // Automatically compiled to GPU kernel
    solve_ode(params, initial)
}

population LargeCohort {
    model OneCompOral
    
    cohort size = 10000
    
    // Parallel GPU execution
    simulate_subjects(n = 10000) with GPU
}
```

**Generated CUDA**:
```cuda
__global__ void simulate_subject_gpu(
    PKParams* params,
    StateVector* initial,
    Concentration* results
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // ...
}
```

---

## Integration Plan

### Phase V2.1: Z3 SMT Integration (2 weeks)

**Week 1**:
- [ ] Add Z3 dependency
- [ ] Implement SMT-LIB translator
- [ ] Create SMT context and solver interface
- [ ] Unit tests for translation

**Week 2**:
- [ ] Implement VC generator
- [ ] Integrate with type checker
- [ ] Add CLI flag `--verify`
- [ ] Integration tests with counterexamples

### Phase V2.2: LSP Implementation (3 weeks)

**Week 1**:
- [ ] Add tower-lsp dependency
- [ ] Implement basic LSP server
- [ ] Text document sync
- [ ] Real-time diagnostics

**Week 2**:
- [ ] Hover information
- [ ] Code completion
- [ ] Go-to-definition
- [ ] Find references

**Week 3**:
- [ ] VS Code extension scaffolding
- [ ] Syntax highlighting grammar
- [ ] Extension packaging
- [ ] Documentation

### Phase V2.3: JIT Compilation (2 weeks)

**Week 1**:
- [ ] Add Cranelift dependency
- [ ] Implement IR → Cranelift translation
- [ ] JIT compilation for ODE system
- [ ] Basic REPL

**Week 2**:
- [ ] Full REPL commands (load, simulate, plot)
- [ ] Hot-reloading
- [ ] Plotting integration
- [ ] Performance benchmarks

### Phase V2.4: GPU Code Generation (2 weeks)

**Week 1**:
- [ ] CUDA kernel generation
- [ ] RK4 solver on GPU
- [ ] Memory management

**Week 2**:
- [ ] SPIR-V backend
- [ ] MedLang GPU syntax (`with GPU`)
- [ ] Population simulation example
- [ ] Benchmarks (1000+ subjects)

---

## Testing Strategy

### Unit Tests

**Per Module**:
- `smt/translator.rs` - 20 tests (constraint translation)
- `smt/solver.rs` - 15 tests (Z3 integration)
- `lsp/server.rs` - 30 tests (LSP protocol)
- `jit/compiler.rs` - 25 tests (Cranelift codegen)
- `codegen/cuda.rs` - 20 tests (CUDA generation)

**Total**: ~110 new unit tests

### Integration Tests

**Test Suites**:
- `tests/smt_integration.rs` - End-to-end SMT verification
- `tests/lsp_integration.rs` - LSP request/response cycles
- `tests/jit_integration.rs` - JIT compilation and execution
- `tests/gpu_integration.rs` - GPU kernel execution (requires CUDA)

### Performance Benchmarks

**Criterion Benchmarks**:
- SMT verification time vs constraint complexity
- JIT compilation time vs model size
- JIT execution time vs Stan/Julia
- GPU simulation throughput (subjects/second)

---

## Documentation Plan

### User Documentation

- **`docs/VERIFICATION_GUIDE.md`** - Using SMT verification
- **`docs/LSP_SETUP.md`** - IDE setup instructions
- **`docs/REPL_TUTORIAL.md`** - Interactive REPL usage
- **`docs/GPU_PROGRAMMING.md`** - GPU-accelerated simulations

### Developer Documentation

- **`docs/SMT_ARCHITECTURE.md`** - SMT translation internals
- **`docs/LSP_PROTOCOL.md`** - LSP implementation details
- **`docs/JIT_INTERNALS.md`** - Cranelift integration
- **`docs/GPU_CODEGEN.md`** - CUDA/SPIR-V generation

---

## Success Metrics

### Technical Metrics

- **SMT Verification**: 95%+ of refinement constraints provable
- **LSP Latency**: <50ms for hover, <100ms for completion
- **JIT Performance**: 100x faster than Stan for single-subject
- **GPU Throughput**: 10,000+ subjects/second on RTX 4090

### Quality Metrics

- **Test Coverage**: 90%+ for new modules
- **Build Time**: <30s for full rebuild with all features
- **Documentation**: 100% of public APIs documented

---

## Dependencies Summary

### New Production Dependencies

```toml
# SMT Verification
z3 = { version = "0.12", features = ["static-link-z3"] }

# LSP Server
tower-lsp = "0.20"
tokio = { version = "1", features = ["full"] }

# JIT Compilation
cranelift = "0.103"
cranelift-module = "0.103"
cranelift-jit = "0.103"
cranelift-codegen = "0.103"

# REPL
rustyline = "13.0"

# GPU
cuda-sys = "0.3"
rspirv = "0.11"
spirv-tools = "0.10"

# Plotting (for REPL)
plotters = "0.3"
```

---

## Risks and Mitigations

### Risk 1: Z3 Complexity

**Risk**: SMT solving may be slow for complex constraints  
**Mitigation**: Timeout after 5s, fallback to syntactic checking

### Risk 2: LSP Scalability

**Risk**: Large files may cause LSP lag  
**Mitigation**: Incremental parsing, async processing

### Risk 3: JIT Compilation Bugs

**Risk**: Cranelift may have edge cases  
**Mitigation**: Extensive testing, fallback to interpreter

### Risk 4: GPU Availability

**Risk**: Not all users have CUDA GPUs  
**Mitigation**: SPIR-V for CPU fallback, clear error messages

---

## Conclusion

Phase V2 transforms MedLang from a research compiler into a production-grade tool with:

1. **Mathematical rigor** (Z3 verification)
2. **Professional UX** (LSP, IDE integration)
3. **Interactive development** (JIT REPL)
4. **Extreme performance** (GPU acceleration)

**Estimated Timeline**: 9 weeks (2 months)  
**Estimated LOC**: +5,000 lines of production code  
**Test Growth**: +110 unit tests, +4 integration suites

**Next Steps**: Begin with Z3 integration (highest impact, foundational for safety).

---

**End of Phase V2 Architecture Document**
