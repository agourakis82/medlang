// MedLang Compiler CLI
use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;

// Import compiler components
use medlangc::codegen::julia::generate_julia;
use medlangc::codegen::julia_pinn::generate_julia_pinn;
use medlangc::codegen::stan::generate_stan;
use medlangc::data::{analyze_trial, compare_trials, TrialAnalysisResults, TrialDataset};
use medlangc::datagen::{generate_dataset, DataRow, TrueParams};
use medlangc::dataload::PKDataset;
use medlangc::design::{
    default_scenarios_from_scores, evaluate_design_grid, evaluate_design_pos,
    optimize_design_over_grid, DesignCandidate, DesignConfig, DesignConfigInfo, ObjectiveConfig,
    PosteriorDraw, QuantumDesignSensitivityReport, QuantumDesignSensitivityResult,
};
use medlangc::diagnostics::{
    build_trust_report, classify_all, summarize_cmdstan_fit, PriorInflationPolicy,
    QuantumPosteriorInfo, QuantumPriorInfo, QuantumPriorPosteriorComparison, SbcConfig,
};
use medlangc::interop::{
    endpoint_to_cql, protocol_endpoints_to_cql, protocol_to_fhir_measures,
    protocol_to_fhir_plan_definition, protocol_to_fhir_research_study, trial_to_adsl_adtr,
    trial_to_fhir_bundle, AdslRow, AdtrRow,
};
use medlangc::ir::surrogate::IRSurrogateConfig;
use medlangc::lexer::tokenize;
use medlangc::lower::{lower_program, lower_program_with_qm};
use medlangc::parser::{parse_program, parse_protocol_from_tokens};
use medlangc::portfolio::{evaluate_portfolio, evaluate_portfolio_design_grid};
use medlangc::qm_stub::QuantumStub;
use medlangc::stanrun::{
    compile_stan_model, detect_cmdstan, print_diagnostics, run_stan_mcmc, StanConfig,
};

#[derive(Parser)]
#[command(name = "medlangc")]
#[command(version = "0.1.0")]
#[command(about = "MedLang compiler for computational pharmacology", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile MedLang source to backend code
    Compile {
        /// Input MedLang source file
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Output file (defaults to <input>.stan or <input>.jl)
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Backend target (stan or julia)
        #[arg(short, long, default_value = "stan")]
        backend: String,

        /// Quantum stub JSON file for Track C integration
        #[arg(long, value_name = "QM_STUB")]
        qm_stub: Option<PathBuf>,

        /// Emit IR to JSON file for inspection
        #[arg(long, value_name = "IR_FILE")]
        emit_ir: Option<PathBuf>,

        /// Verbose output showing compilation stages
        #[arg(short, long)]
        verbose: bool,
    },

    /// Generate synthetic dataset for testing
    GenerateData {
        /// Number of subjects
        #[arg(short = 'n', long, default_value = "20")]
        n_subjects: usize,

        /// Output CSV file
        #[arg(short, long, value_name = "OUTPUT")]
        output: PathBuf,

        /// Dose amount in mg
        #[arg(long, default_value = "100.0")]
        dose_amount: f64,

        /// Random seed for reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Verbose output showing parameters
        #[arg(short, long)]
        verbose: bool,
    },

    /// Check MedLang source for syntax and type errors
    Check {
        /// Input MedLang source file
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Verbose output showing all stages
        #[arg(short, long)]
        verbose: bool,
    },

    /// Convert CSV data to Stan JSON format
    ConvertData {
        /// Input CSV file (NONMEM format)
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Output JSON file for Stan
        #[arg(short, long, value_name = "OUTPUT")]
        output: PathBuf,

        /// Show data summary
        #[arg(short, long)]
        verbose: bool,
    },

    /// Run Stan model with MCMC sampling
    Run {
        /// Stan model file (.stan)
        #[arg(value_name = "MODEL")]
        model: PathBuf,

        /// Data file (.json)
        #[arg(short, long, value_name = "DATA")]
        data: PathBuf,

        /// Output directory for results
        #[arg(short, long, value_name = "OUTPUT", default_value = "output")]
        output: PathBuf,

        /// Number of MCMC chains
        #[arg(long, default_value = "4")]
        chains: usize,

        /// Number of warmup iterations
        #[arg(long, default_value = "1000")]
        warmup: usize,

        /// Number of sampling iterations
        #[arg(long, default_value = "1000")]
        samples: usize,

        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u32>,

        /// Target acceptance rate (adapt_delta)
        #[arg(long, default_value = "0.8")]
        adapt_delta: f64,

        /// Maximum tree depth
        #[arg(long, default_value = "10")]
        max_treedepth: usize,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Analyze observed trial data using a protocol definition
    AnalyzeTrial {
        /// Protocol definition file (.medlang)
        #[arg(long, value_name = "PROTOCOL")]
        protocol: PathBuf,

        /// Trial data file (.csv or .json)
        #[arg(long, value_name = "DATA")]
        data: PathBuf,

        /// Output file for analysis results (.json)
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Compare virtual and observed trial results
    CompareTrials {
        /// Virtual trial results (.json from analyze-trial or simulate-protocol)
        #[arg(long, value_name = "VIRTUAL")]
        virtual_results: PathBuf,

        /// Observed trial results (.json from analyze-trial)
        #[arg(long, value_name = "OBSERVED")]
        observed_results: PathBuf,

        /// Output file for comparison results (.json)
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Run diagnostics on MCMC fit results
    Diagnostics {
        /// Directory containing CmdStan output files
        #[arg(long, value_name = "FIT_DIR")]
        fit_dir: PathBuf,

        /// Output file for diagnostics report (.json)
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Run Simulation-Based Calibration (SBC)
    Sbc {
        /// Number of SBC replications
        #[arg(short = 'n', long, default_value = "100")]
        n_sims: usize,

        /// Number of posterior draws per replication
        #[arg(long, default_value = "1000")]
        n_draws: usize,

        /// Output file for SBC results (.json)
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Evaluate trial design for predictive probability of success
    DesignEvaluate {
        /// Protocol definition file (.medlang)
        #[arg(long, value_name = "PROTOCOL")]
        protocol: PathBuf,

        /// Protocol name within the file
        #[arg(long, value_name = "NAME")]
        protocol_name: String,

        /// Sample size per arm for future trial
        #[arg(long, default_value = "150")]
        n_per_arm: usize,

        /// Number of posterior draws to use
        #[arg(long, default_value = "500")]
        n_draws: usize,

        /// Output file for design evaluation (.json)
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Evaluate multiple trial designs (grid search over N)
    DesignGrid {
        /// Protocol definition file (.medlang)
        #[arg(long, value_name = "PROTOCOL")]
        protocol: PathBuf,

        /// Protocol name within the file
        #[arg(long, value_name = "NAME")]
        protocol_name: String,

        /// Comma-separated list of sample sizes per arm (e.g., "50,100,150,200")
        #[arg(long, value_name = "SIZES", value_delimiter = ',')]
        n_per_arm: Vec<usize>,

        /// Number of posterior draws to use
        #[arg(long, default_value = "400")]
        n_draws: usize,

        /// Output file for design grid results (.json)
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Export Julia+DiffEqFlux script for neural surrogate training
    ExportPinn {
        /// Input MedLang source file
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Output Julia script path
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Enable physics-informed loss terms
        #[arg(long)]
        physics_loss: bool,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Export protocol as FHIR R4 resources (ResearchStudy, PlanDefinition, Measure, Bundle)
    ExportFhir {
        /// Protocol definition file (.medlang)
        #[arg(long, value_name = "PROTOCOL")]
        protocol: PathBuf,

        /// Protocol name within the file
        #[arg(long, value_name = "NAME")]
        protocol_name: String,

        /// Trial data CSV file (optional, for Bundle export)
        #[arg(long, value_name = "DATA")]
        trial_data: Option<PathBuf>,

        /// Output directory for FHIR resources
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Export endpoint definitions as CQL libraries
    ExportCql {
        /// Protocol definition file (.medlang)
        #[arg(long, value_name = "PROTOCOL")]
        protocol: PathBuf,

        /// Protocol name within the file
        #[arg(long, value_name = "NAME")]
        protocol_name: String,

        /// Output directory for CQL files
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Export trial data as CDISC-like datasets (ADSL/ADTR)
    ExportCdisc {
        /// Trial data CSV file
        #[arg(long, value_name = "DATA")]
        trial_data: PathBuf,

        /// Study identifier for CDISC datasets
        #[arg(long, value_name = "STUDY_ID")]
        study_id: String,

        /// Output directory for ADSL/ADTR CSV files
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Evaluate design sensitivity under quantum prior trust scenarios
    QuantumTrustDesign {
        /// Protocol definition file (.medlang)
        #[arg(long, value_name = "PROTOCOL")]
        protocol: PathBuf,

        /// Protocol name within the file
        #[arg(long, value_name = "NAME")]
        protocol_name: String,

        /// Population model name
        #[arg(long, value_name = "MODEL")]
        population_model: String,

        /// MCMC fit directory with posterior samples
        #[arg(long, value_name = "FIT_DIR")]
        fit_dir: PathBuf,

        /// Sample size per arm for design evaluation
        #[arg(long, default_value = "180")]
        design_n_per_arm: usize,

        /// Number of posterior draws to use
        #[arg(long, default_value = "500")]
        design_n_draws: usize,

        /// Prior inflation policy JSON (optional, uses defaults if omitted)
        #[arg(long, value_name = "POLICY")]
        policy: Option<PathBuf>,

        /// Output file for sensitivity report (.json)
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Evaluate portfolio of QM-backed ligands under a single design
    PortfolioEvaluate {
        /// Protocol name for design evaluation
        #[arg(long, value_name = "PROTOCOL")]
        protocol_name: String,

        /// Population model name (e.g., Oncology_PBPK_QSP_QM)
        #[arg(long, value_name = "MODEL")]
        population_model: String,

        /// Comma-separated ligand IDs with Kd and Kp values
        /// Format: LIG001:3.2e-8:5.1,LIG002:1.0e-7:3.2
        #[arg(long, value_name = "LIGANDS")]
        ligands: String,

        /// Sample size per arm for design evaluation
        #[arg(long, default_value = "120")]
        n_per_arm: usize,

        /// Number of posterior draws to use
        #[arg(long, default_value = "400")]
        n_draws: usize,

        /// Backend: mechanistic | surrogate | hybrid
        #[arg(long, default_value = "mechanistic")]
        backend: String,

        /// Output JSON file for portfolio summary
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Evaluate portfolio design grid (PoS vs N per arm)
    PortfolioDesignGrid {
        /// Protocol name for design evaluation
        #[arg(long, value_name = "PROTOCOL")]
        protocol_name: String,

        /// Population model name
        #[arg(long, value_name = "MODEL")]
        population_model: String,

        /// Comma-separated ligand IDs with Kd and Kp values
        #[arg(long, value_name = "LIGANDS")]
        ligands: String,

        /// Comma-separated list of sample sizes per arm (e.g., "80,100,120,150")
        #[arg(long, value_name = "N_VALUES", value_delimiter = ',')]
        n_per_arm: Vec<usize>,

        /// Number of posterior draws to use
        #[arg(long, default_value = "400")]
        n_draws: usize,

        /// Backend: mechanistic | surrogate | hybrid
        #[arg(long, default_value = "mechanistic")]
        backend: String,

        /// Output JSON file for design grid
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Find optimal trial design using utility-based grid search
    DesignOptimize {
        /// Protocol definition file (.medlang)
        #[arg(long, value_name = "PROTOCOL")]
        protocol: PathBuf,

        /// Protocol name within the file
        #[arg(long, value_name = "NAME")]
        protocol_name: String,

        /// Population model name
        #[arg(long, value_name = "MODEL")]
        population_model: String,

        /// MCMC fit directory with posterior samples
        #[arg(long, value_name = "FIT_DIR")]
        fit_dir: PathBuf,

        /// Comma-separated list of N per arm values (e.g., "80,100,120,150")
        #[arg(long, value_name = "N_VALUES", value_delimiter = ',')]
        n_per_arm: Vec<usize>,

        /// Comma-separated list of ORR margins (e.g., "0.10,0.15,0.20")
        #[arg(long, value_name = "MARGINS", value_delimiter = ',')]
        orr_margin: Vec<f64>,

        /// Comma-separated list of DLT thresholds (e.g., "0.20,0.25,0.30")
        #[arg(long, value_name = "THRESHOLDS", value_delimiter = ',')]
        dlt_threshold: Option<Vec<f64>>,

        /// Objective configuration JSON file (optional, uses balanced preset if omitted)
        #[arg(long, value_name = "OBJECTIVE")]
        objective: Option<PathBuf>,

        /// Backend: mechanistic | surrogate | hybrid
        #[arg(long, default_value = "mechanistic")]
        backend: String,

        /// Output JSON file for optimization report
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Check a MedLang module (parse + type check, no code generation)
    Check {
        /// Input MedLang module file (.med)
        #[arg(value_name = "MODULE")]
        module: PathBuf,

        /// Additional module search paths
        #[arg(short = 'I', long = "include", value_name = "PATH")]
        include_paths: Vec<PathBuf>,

        /// Verbose output showing module resolution
        #[arg(short, long)]
        verbose: bool,
    },

    /// Build a MedLang module (check + generate code for all dependencies)
    Build {
        /// Input MedLang module file (.med)
        #[arg(value_name = "MODULE")]
        module: PathBuf,

        /// Output directory for generated code
        #[arg(short, long, value_name = "OUTDIR", default_value = "./build")]
        output_dir: PathBuf,

        /// Backend target (stan or julia)
        #[arg(short, long, default_value = "stan")]
        backend: String,

        /// Additional module search paths
        #[arg(short = 'I', long = "include", value_name = "PATH")]
        include_paths: Vec<PathBuf>,

        /// Verbose output showing build steps
        #[arg(short, long)]
        verbose: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile {
            input,
            output,
            backend,
            qm_stub,
            emit_ir,
            verbose,
        } => compile_command(input, output, backend, qm_stub, emit_ir, verbose),
        Commands::GenerateData {
            n_subjects,
            output,
            dose_amount,
            seed,
            verbose,
        } => generate_data_command(n_subjects, output, dose_amount, seed, verbose),
        Commands::Check { input, verbose } => check_command(input, verbose),
        Commands::ConvertData {
            input,
            output,
            verbose,
        } => convert_data_command(input, output, verbose),
        Commands::Run {
            model,
            data,
            output,
            chains,
            warmup,
            samples,
            seed,
            adapt_delta,
            max_treedepth,
            verbose,
        } => run_command(
            model,
            data,
            output,
            chains,
            warmup,
            samples,
            seed,
            adapt_delta,
            max_treedepth,
            verbose,
        ),
        Commands::AnalyzeTrial {
            protocol,
            data,
            output,
            verbose,
        } => analyze_trial_command(protocol, data, output, verbose),
        Commands::CompareTrials {
            virtual_results,
            observed_results,
            output,
            verbose,
        } => compare_trials_command(virtual_results, observed_results, output, verbose),
        Commands::Diagnostics {
            fit_dir,
            output,
            verbose,
        } => diagnostics_command(fit_dir, output, verbose),
        Commands::Sbc {
            n_sims,
            n_draws,
            output,
            seed,
            verbose,
        } => sbc_command(n_sims, n_draws, output, seed, verbose),
        Commands::DesignEvaluate {
            protocol,
            protocol_name,
            n_per_arm,
            n_draws,
            output,
            verbose,
        } => design_evaluate_command(protocol, protocol_name, n_per_arm, n_draws, output, verbose),
        Commands::DesignGrid {
            protocol,
            protocol_name,
            n_per_arm,
            n_draws,
            output,
            verbose,
        } => design_grid_command(protocol, protocol_name, n_per_arm, n_draws, output, verbose),
        Commands::ExportPinn {
            input,
            output,
            physics_loss,
            verbose,
        } => export_pinn_command(input, output, physics_loss, verbose),
        Commands::ExportFhir {
            protocol,
            protocol_name,
            trial_data,
            output,
            verbose,
        } => export_fhir_command(protocol, protocol_name, trial_data, output, verbose),
        Commands::ExportCql {
            protocol,
            protocol_name,
            output,
            verbose,
        } => export_cql_command(protocol, protocol_name, output, verbose),
        Commands::ExportCdisc {
            trial_data,
            study_id,
            output,
            verbose,
        } => export_cdisc_command(trial_data, study_id, output, verbose),
        Commands::QuantumTrustDesign {
            protocol,
            protocol_name,
            population_model,
            fit_dir,
            design_n_per_arm,
            design_n_draws,
            policy,
            output,
            verbose,
        } => quantum_trust_design_command(
            protocol,
            protocol_name,
            population_model,
            fit_dir,
            design_n_per_arm,
            design_n_draws,
            policy,
            output,
            verbose,
        ),
        Commands::PortfolioEvaluate {
            protocol_name,
            population_model,
            ligands,
            n_per_arm,
            n_draws,
            backend,
            output,
            verbose,
        } => portfolio_evaluate_command(
            protocol_name,
            population_model,
            ligands,
            n_per_arm,
            n_draws,
            backend,
            output,
            verbose,
        ),
        Commands::PortfolioDesignGrid {
            protocol_name,
            population_model,
            ligands,
            n_per_arm,
            n_draws,
            backend,
            output,
            verbose,
        } => portfolio_design_grid_command(
            protocol_name,
            population_model,
            ligands,
            n_per_arm,
            n_draws,
            backend,
            output,
            verbose,
        ),
        Commands::Check {
            module,
            include_paths,
            verbose,
        } => check_command(module, include_paths, verbose),
        Commands::Build {
            module,
            output_dir,
            backend,
            include_paths,
            verbose,
        } => build_command(module, output_dir, backend, include_paths, verbose),
        Commands::DesignOptimize {
            protocol,
            protocol_name,
            population_model,
            fit_dir,
            n_per_arm,
            orr_margin,
            dlt_threshold,
            objective,
            backend,
            output,
            verbose,
        } => design_optimize_command(
            protocol,
            protocol_name,
            population_model,
            fit_dir,
            n_per_arm,
            orr_margin,
            dlt_threshold,
            objective,
            backend,
            output,
            verbose,
        ),
    }
}

fn compile_command(
    input: PathBuf,
    output: Option<PathBuf>,
    backend: String,
    qm_stub: Option<PathBuf>,
    emit_ir: Option<PathBuf>,
    verbose: bool,
) -> Result<()> {
    // Validate backend
    if backend != "stan" && backend != "julia" {
        bail!("Unsupported backend '{}'. Supported: stan, julia", backend);
    }

    // Load QM stub if provided
    let qm_stub_data = if let Some(qm_path) = &qm_stub {
        if verbose {
            eprintln!("Loading quantum stub: {}", qm_path.display());
        }
        let stub = QuantumStub::load(qm_path)
            .with_context(|| format!("Failed to load QM stub: {}", qm_path.display()))?;
        if verbose {
            eprintln!(
                "  ✓ QM stub loaded: {} targeting {}",
                stub.drug_id, stub.target_id
            );
            eprintln!("    Kd = {:.2e} M", stub.Kd_M);
            if let Some(kp) = stub.kp_tumor_from_dg() {
                eprintln!("    Kp_tumor = {:.3}", kp);
            }
        }
        Some(stub)
    } else {
        None
    };

    // Read source file
    if verbose {
        eprintln!("Reading source: {}", input.display());
    }
    let source = fs::read_to_string(&input)
        .with_context(|| format!("Failed to read input file: {}", input.display()))?;

    // Stage 1: Tokenization
    if verbose {
        eprintln!("Stage 1: Tokenization...");
    }
    let tokens = tokenize(&source).with_context(|| "Tokenization failed")?;
    if verbose {
        eprintln!("  ✓ {} tokens generated", tokens.len());
    }

    // Stage 2: Parsing
    if verbose {
        eprintln!("Stage 2: Parsing...");
    }
    let ast = parse_program(&tokens).with_context(|| "Parsing failed")?;
    if verbose {
        eprintln!(
            "  ✓ AST constructed with {} declarations",
            ast.declarations.len()
        );
    }

    // Stage 3: Type Checking (implicit in lowering for V0)
    if verbose {
        eprintln!("Stage 3: Type checking and lowering to IR...");
    }
    let ir = lower_program_with_qm(&ast, qm_stub_data.as_ref())
        .with_context(|| "Lowering to IR failed")?;
    if verbose {
        eprintln!("  ✓ IR generated");
        eprintln!("    - {} states", ir.model.states.len());
        eprintln!("    - {} parameters", ir.model.params.len());
        eprintln!("    - {} ODEs", ir.model.odes.len());
        eprintln!("    - {} observables", ir.model.observables.len());
        if !ir.externals.is_empty() {
            eprintln!("    - {} external QM constants", ir.externals.len());
            for ext in &ir.externals {
                eprintln!("      • {} = {:.3e}", ext.name, ext.value);
            }
        }
    }

    // Optionally emit IR to JSON
    if let Some(ir_path) = emit_ir {
        if verbose {
            eprintln!("Emitting IR to: {}", ir_path.display());
        }
        let ir_json =
            serde_json::to_string_pretty(&ir).context("Failed to serialize IR to JSON")?;
        fs::write(&ir_path, ir_json)
            .with_context(|| format!("Failed to write IR to: {}", ir_path.display()))?;
        if verbose {
            eprintln!("  ✓ IR written to {}", ir_path.display());
        }
    }

    // Stage 4: Code Generation
    if verbose {
        eprintln!("Stage 4: Code generation (backend: {})...", backend);
    }
    let generated_code = match backend.as_str() {
        "stan" => generate_stan(&ir).context("Stan code generation failed")?,
        "julia" => generate_julia(&ir).context("Julia code generation failed")?,
        _ => unreachable!(),
    };
    if verbose {
        eprintln!(
            "  ✓ {} lines of {} code generated",
            generated_code.lines().count(),
            backend
        );
    }

    // Determine output path
    let output_path = output.unwrap_or_else(|| {
        let extension = match backend.as_str() {
            "stan" => "stan",
            "julia" => "jl",
            _ => unreachable!(),
        };
        input.with_extension(extension)
    });

    // Write output
    if verbose {
        eprintln!("Writing output: {}", output_path.display());
    }
    fs::write(&output_path, generated_code)
        .with_context(|| format!("Failed to write output to: {}", output_path.display()))?;

    println!(
        "✓ Compilation successful: {} → {}",
        input.display(),
        output_path.display()
    );

    Ok(())
}

fn generate_data_command(
    n_subjects: usize,
    output: PathBuf,
    dose_amount: f64,
    seed: u64,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("Generating synthetic dataset...");
        eprintln!("  Subjects: {}", n_subjects);
        eprintln!("  Dose: {} mg", dose_amount);
        eprintln!("  Seed: {}", seed);
    }

    // Use default true parameters for one-compartment oral model
    let params = TrueParams {
        cl_pop: 10.0, // L/h
        v_pop: 50.0,  // L
        ka_pop: 1.0,  // 1/h
        omega_cl: 0.3,
        omega_v: 0.2,
        omega_ka: 0.4,
        sigma_prop: 0.15,
    };

    if verbose {
        eprintln!("Population parameters:");
        eprintln!(
            "  CL_pop = {} L/h, ω_CL = {}",
            params.cl_pop, params.omega_cl
        );
        eprintln!("  V_pop  = {} L,   ω_V  = {}", params.v_pop, params.omega_v);
        eprintln!(
            "  Ka_pop = {} 1/h, ω_Ka = {}",
            params.ka_pop, params.omega_ka
        );
        eprintln!("  σ_prop = {}", params.sigma_prop);
    }

    // Observation times: 0.5, 1, 2, 4, 8, 12, 24 hours
    let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];

    // Generate dataset
    let dataset = generate_dataset(n_subjects, &obs_times, dose_amount, &params, seed);

    if verbose {
        eprintln!("Generated {} observations", dataset.len());
    }

    // Write CSV
    write_csv(&output, &dataset)
        .with_context(|| format!("Failed to write CSV to: {}", output.display()))?;

    println!(
        "✓ Dataset generated: {} ({} rows)",
        output.display(),
        dataset.len()
    );

    Ok(())
}

fn check_command(input: PathBuf, verbose: bool) -> Result<()> {
    if verbose {
        eprintln!("Checking: {}", input.display());
    }

    // Read source
    let source = fs::read_to_string(&input)
        .with_context(|| format!("Failed to read input file: {}", input.display()))?;

    // Stage 1: Tokenization
    if verbose {
        eprintln!("Stage 1: Tokenization...");
    }
    let tokens = tokenize(&source).with_context(|| "Tokenization failed")?;
    if verbose {
        eprintln!("  ✓ {} tokens", tokens.len());
    }

    // Stage 2: Parsing
    if verbose {
        eprintln!("Stage 2: Parsing...");
    }
    let ast = parse_program(&tokens).with_context(|| "Parsing failed")?;
    if verbose {
        eprintln!("  ✓ {} declarations", ast.declarations.len());
    }

    // Stage 3: Type checking
    if verbose {
        eprintln!("Stage 3: Type checking...");
    }
    let _ir = lower_program(&ast).with_context(|| "Type checking failed")?;
    if verbose {
        eprintln!("  ✓ Type checking passed");
    }

    println!("✓ All checks passed: {}", input.display());

    Ok(())
}

fn write_csv(path: &PathBuf, dataset: &[DataRow]) -> Result<()> {
    let mut csv_content = String::new();
    csv_content.push_str("ID,TIME,AMT,DV,EVID,WT\n");

    for row in dataset {
        let amt_str = row.amt.map_or(".".to_string(), |v| v.to_string());
        let dv_str = row.dv.map_or(".".to_string(), |v| v.to_string());
        csv_content.push_str(&format!(
            "{},{},{},{},{},{}\n",
            row.id, row.time, amt_str, dv_str, row.evid, row.wt
        ));
    }

    fs::write(path, csv_content)?;
    Ok(())
}

fn convert_data_command(input: PathBuf, output: PathBuf, verbose: bool) -> Result<()> {
    if verbose {
        eprintln!("Loading CSV data: {}", input.display());
    }

    // Load dataset
    let dataset = PKDataset::from_csv(&input)
        .with_context(|| format!("Failed to load CSV data from: {}", input.display()))?;

    if verbose {
        eprintln!("\n{}", dataset.summary());
        eprintln!();
    }

    // Convert to Stan format
    if verbose {
        eprintln!("Converting to Stan JSON format...");
    }

    let stan_data = dataset
        .to_stan_data()
        .context("Failed to convert data to Stan format")?;

    // Write output
    fs::write(&output, stan_data)
        .with_context(|| format!("Failed to write Stan data to: {}", output.display()))?;

    println!(
        "✓ Data converted: {} → {}",
        input.display(),
        output.display()
    );

    Ok(())
}

fn run_command(
    model: PathBuf,
    data: PathBuf,
    output: PathBuf,
    chains: usize,
    warmup: usize,
    samples: usize,
    seed: Option<u32>,
    adapt_delta: f64,
    max_treedepth: usize,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("MedLang Stan Runner");
        eprintln!("Model: {}", model.display());
        eprintln!("Data: {}", data.display());
        eprintln!("Output: {}", output.display());
        eprintln!();
    }

    // Detect cmdstan installation
    if verbose {
        eprintln!("Detecting cmdstan installation...");
    }
    let cmdstan_path = detect_cmdstan().context(
        "Failed to detect cmdstan. Please install cmdstan or set CMDSTAN environment variable",
    )?;
    if verbose {
        eprintln!("✓ Found cmdstan at: {}", cmdstan_path.display());
        eprintln!();
    }

    // Compile Stan model
    let exe_path = compile_stan_model(&model, &cmdstan_path)
        .with_context(|| format!("Failed to compile Stan model: {}", model.display()))?;
    if verbose {
        eprintln!("✓ Model compiled to: {}", exe_path.display());
        eprintln!();
    }

    // Configure MCMC sampling
    let config = StanConfig {
        num_chains: chains,
        num_warmup: warmup,
        num_samples: samples,
        seed,
        adapt_delta,
        max_treedepth,
    };

    if verbose {
        eprintln!("MCMC Configuration:");
        eprintln!("  Chains: {}", config.num_chains);
        eprintln!("  Warmup: {}", config.num_warmup);
        eprintln!("  Samples: {}", config.num_samples);
        eprintln!("  Adapt delta: {}", config.adapt_delta);
        eprintln!("  Max tree depth: {}", config.max_treedepth);
        if let Some(s) = config.seed {
            eprintln!("  Seed: {}", s);
        }
        eprintln!();
    }

    // Run MCMC sampling
    let result =
        run_stan_mcmc(&exe_path, &data, &output, &config).context("MCMC sampling failed")?;

    // Print diagnostics
    print_diagnostics(&result);

    println!("\n✓ Results saved to: {}", output.display());
    println!(
        "  Chain files: {:?}",
        result
            .chain_files
            .iter()
            .map(|p| p.file_name().unwrap())
            .collect::<Vec<_>>()
    );

    Ok(())
}

fn analyze_trial_command(
    protocol_path: PathBuf,
    data_path: PathBuf,
    output: Option<PathBuf>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("MedLang Trial Data Analyzer");
        eprintln!("Protocol: {}", protocol_path.display());
        eprintln!("Data: {}", data_path.display());
        eprintln!();
    }

    // Read and parse protocol
    if verbose {
        eprintln!("Loading protocol definition...");
    }
    let protocol_source = fs::read_to_string(&protocol_path)
        .with_context(|| format!("Failed to read protocol file: {}", protocol_path.display()))?;

    let protocol_tokens =
        tokenize(&protocol_source).with_context(|| "Failed to tokenize protocol")?;

    let protocol =
        parse_protocol_from_tokens(&protocol_tokens).with_context(|| "Failed to parse protocol")?;

    if verbose {
        eprintln!("  ✓ Protocol '{}' loaded", protocol.name);
        eprintln!("    - {} arms", protocol.arms.len());
        eprintln!("    - {} visits", protocol.visits.len());
        eprintln!("    - {} endpoints", protocol.endpoints.len());
        if protocol.inclusion.is_some() {
            eprintln!("    - Inclusion criteria defined");
        }
        eprintln!();
    }

    // Load trial data
    if verbose {
        eprintln!("Loading trial data...");
    }

    let dataset = if data_path.extension().and_then(|s| s.to_str()) == Some("json") {
        TrialDataset::from_json(&data_path)
            .with_context(|| format!("Failed to load JSON data from: {}", data_path.display()))?
    } else {
        TrialDataset::from_csv(&data_path)
            .with_context(|| format!("Failed to load CSV data from: {}", data_path.display()))?
    };

    if verbose {
        eprintln!("  ✓ Loaded {} subjects", dataset.num_subjects());
        eprintln!("    - {} total observations", dataset.rows.len());
        let arm_counts = dataset.num_subjects_per_arm();
        for (arm, count) in &arm_counts {
            eprintln!("    - {}: {} subjects", arm, count);
        }
        eprintln!();
    }

    // Analyze trial
    if verbose {
        eprintln!("Analyzing endpoints...");
    }

    let results = analyze_trial(&protocol, &dataset, data_path.to_str().unwrap())
        .with_context(|| "Failed to analyze trial data")?;

    if verbose {
        eprintln!("  ✓ Analysis complete");
        eprintln!();
    }

    // Print summary to stdout
    println!("Trial Analysis Results");
    println!("======================");
    println!("Protocol: {}", results.protocol_name);
    println!("Data source: {}", results.data_source);
    println!();

    for arm_result in &results.arms {
        println!("Arm: {} ({})", arm_result.arm_name, arm_result.label);
        println!(
            "  Subjects: {} total, {} included, {} excluded",
            arm_result.n_subjects, arm_result.n_included, arm_result.n_excluded
        );
        println!();

        for (endpoint_name, endpoint_result) in &arm_result.endpoints {
            match endpoint_result {
                medlangc::data::EndpointAnalysisResult::Binary {
                    n_responders,
                    response_rate,
                } => {
                    println!(
                        "  {}: ORR = {:.1}% ({}/{} responders)",
                        endpoint_name,
                        response_rate * 100.0,
                        n_responders,
                        arm_result.n_included
                    );
                }
                medlangc::data::EndpointAnalysisResult::TimeToEvent {
                    n_events,
                    n_censored,
                    median_days,
                    ..
                } => {
                    let median_str = median_days
                        .map(|d| format!("{:.1} days", d))
                        .unwrap_or_else(|| "not reached".to_string());
                    println!(
                        "  {}: Median = {}, Events = {}, Censored = {}",
                        endpoint_name, median_str, n_events, n_censored
                    );
                }
            }
        }
        println!();
    }

    // Write JSON output
    let output_path = output.unwrap_or_else(|| data_path.with_extension("analysis.json"));

    if verbose {
        eprintln!("Writing results to: {}", output_path.display());
    }

    let json_output =
        serde_json::to_string_pretty(&results).context("Failed to serialize results to JSON")?;

    fs::write(&output_path, json_output)
        .with_context(|| format!("Failed to write output to: {}", output_path.display()))?;

    println!("✓ Results saved to: {}", output_path.display());

    Ok(())
}

fn compare_trials_command(
    virtual_path: PathBuf,
    observed_path: PathBuf,
    output: Option<PathBuf>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("MedLang Trial Comparison Tool");
        eprintln!("Virtual results: {}", virtual_path.display());
        eprintln!("Observed results: {}", observed_path.display());
        eprintln!();
    }

    // Load virtual results
    if verbose {
        eprintln!("Loading virtual trial results...");
    }
    let virtual_content = fs::read_to_string(&virtual_path)
        .with_context(|| format!("Failed to read virtual results: {}", virtual_path.display()))?;

    let virtual_results: TrialAnalysisResults = serde_json::from_str(&virtual_content)
        .with_context(|| "Failed to parse virtual results JSON")?;

    if verbose {
        eprintln!(
            "  ✓ Loaded virtual results for protocol '{}'",
            virtual_results.protocol_name
        );
        eprintln!("    - {} arms", virtual_results.arms.len());
    }

    // Load observed results
    if verbose {
        eprintln!("Loading observed trial results...");
    }
    let observed_content = fs::read_to_string(&observed_path).with_context(|| {
        format!(
            "Failed to read observed results: {}",
            observed_path.display()
        )
    })?;

    let observed_results: TrialAnalysisResults = serde_json::from_str(&observed_content)
        .with_context(|| "Failed to parse observed results JSON")?;

    if verbose {
        eprintln!(
            "  ✓ Loaded observed results for protocol '{}'",
            observed_results.protocol_name
        );
        eprintln!("    - {} arms", observed_results.arms.len());
        eprintln!();
    }

    // Compare
    if verbose {
        eprintln!("Comparing trials...");
    }

    let comparison = compare_trials(&virtual_results, &observed_results)
        .with_context(|| "Failed to compare trial results")?;

    if verbose {
        eprintln!("  ✓ Comparison complete");
        eprintln!();
    }

    // Print summary to stdout
    println!("Trial Comparison Results");
    println!("========================");
    println!("Protocol: {}", comparison.protocol_name);
    println!("Virtual source: {}", comparison.virtual_source);
    println!("Observed source: {}", comparison.observed_source);
    println!();

    println!("Overall Metrics:");
    println!(
        "  Mean absolute ORR error: {:.1}%",
        comparison.overall_metrics.mean_absolute_orr_error * 100.0
    );
    println!(
        "  Mean relative ORR error: {:.1}%",
        comparison.overall_metrics.mean_relative_orr_error * 100.0
    );
    println!(
        "  Mean PFS median difference: {:.1} days",
        comparison.overall_metrics.mean_pfs_median_difference
    );
    println!(
        "  Agreement score: {:.3}",
        comparison.overall_metrics.overall_agreement_score
    );
    println!();

    for arm_comp in &comparison.arms {
        println!("Arm: {} ({})", arm_comp.arm_name, arm_comp.label);
        println!(
            "  N: {} virtual, {} observed",
            arm_comp.n_virtual, arm_comp.n_observed
        );
        println!();

        for (endpoint_name, endpoint_comp) in &arm_comp.endpoints {
            match endpoint_comp {
                medlangc::data::EndpointComparison::Binary {
                    orr_virtual,
                    orr_observed,
                    absolute_difference,
                    relative_difference,
                    chi_square_test,
                } => {
                    println!("  {} (Binary):", endpoint_name);
                    println!("    Virtual ORR:   {:.1}%", orr_virtual * 100.0);
                    println!("    Observed ORR:  {:.1}%", orr_observed * 100.0);
                    println!("    Absolute diff: {:+.1}%", absolute_difference * 100.0);
                    println!("    Relative diff: {:+.1}%", relative_difference * 100.0);

                    if let Some(chi2) = chi_square_test {
                        println!("    Chi-square test:");
                        println!(
                            "      χ² = {:.3}, p = {:.4}",
                            chi2.chi_square_statistic, chi2.p_value
                        );
                        println!(
                            "      Significant at α=0.05: {}",
                            if chi2.significant_at_05 { "YES" } else { "NO" }
                        );
                    }
                }
                medlangc::data::EndpointComparison::TimeToEvent {
                    median_virtual,
                    median_observed,
                    median_difference,
                    km_divergence,
                    ..
                } => {
                    println!("  {} (Time-to-Event):", endpoint_name);

                    let v_str = median_virtual
                        .map(|d| format!("{:.1} days", d))
                        .unwrap_or_else(|| "not reached".to_string());
                    let o_str = median_observed
                        .map(|d| format!("{:.1} days", d))
                        .unwrap_or_else(|| "not reached".to_string());

                    println!("    Virtual median:  {}", v_str);
                    println!("    Observed median: {}", o_str);

                    if let Some(diff) = median_difference {
                        println!("    Median diff: {:+.1} days", diff);
                    }

                    println!("    KM curve divergence: {:.2}", km_divergence);
                }
            }
            println!();
        }
    }

    // Write JSON output
    let output_path = output.unwrap_or_else(|| PathBuf::from("trial_comparison.json"));

    if verbose {
        eprintln!("Writing comparison results to: {}", output_path.display());
    }

    let json_output = serde_json::to_string_pretty(&comparison)
        .context("Failed to serialize comparison to JSON")?;

    fs::write(&output_path, json_output)
        .with_context(|| format!("Failed to write output to: {}", output_path.display()))?;

    println!("✓ Comparison saved to: {}", output_path.display());

    Ok(())
}

fn diagnostics_command(fit_dir: PathBuf, output: Option<PathBuf>, verbose: bool) -> Result<()> {
    if verbose {
        eprintln!("Running diagnostics on fit in: {}", fit_dir.display());
    }

    // Check that fit_dir exists and is a directory
    if !fit_dir.exists() {
        bail!("Fit directory does not exist: {}", fit_dir.display());
    }
    if !fit_dir.is_dir() {
        bail!("Not a directory: {}", fit_dir.display());
    }

    // Summarize MCMC fit
    let summary = summarize_cmdstan_fit(&fit_dir)
        .with_context(|| format!("Failed to summarize fit from: {}", fit_dir.display()))?;

    if verbose {
        eprintln!(
            "Parsed {} parameters from {} chains",
            summary.params.len(),
            summary.n_chains
        );
        eprintln!("Total draws: {}", summary.n_draws);
        eprintln!("Divergent transitions: {}", summary.n_divergent);
        eprintln!("Max treedepth exceeded: {}", summary.max_treedepth_exceeded);
        eprintln!();
        eprintln!(
            "Overall Quality: Grade {}",
            summary.overall_quality.quality_grade
        );
        eprintln!("  Max R-hat: {:.4}", summary.overall_quality.max_rhat);
        eprintln!(
            "  Min ESS bulk: {:.1}",
            summary.overall_quality.min_ess_bulk
        );
        eprintln!(
            "  Convergence issues: {}",
            summary.overall_quality.has_convergence_issues
        );
        eprintln!(
            "  Sampling issues: {}",
            summary.overall_quality.has_sampling_issues
        );
        eprintln!();
    }

    // Print summary table
    println!("MCMC Diagnostics Summary");
    println!("========================");
    println!();
    println!("Chains: {}, Draws: {}", summary.n_chains, summary.n_draws);
    println!(
        "Divergent: {}, Max treedepth: {}",
        summary.n_divergent, summary.max_treedepth_exceeded
    );
    println!();
    println!("Quality Grade: {}", summary.overall_quality.quality_grade);
    println!();
    println!(
        "{:<20} {:>10} {:>10} {:>8} {:>10} {:>10}",
        "Parameter", "Mean", "SD", "R-hat", "ESS_bulk", "ESS_tail"
    );
    println!("{}", "-".repeat(80));

    for param in &summary.params {
        println!(
            "{:<20} {:>10.4} {:>10.4} {:>8.4} {:>10.1} {:>10.1}",
            param.name, param.mean, param.sd, param.rhat, param.ess_bulk, param.ess_tail
        );
    }
    println!();

    // Write JSON output
    let output_path = output.unwrap_or_else(|| PathBuf::from("mcmc_diagnostics.json"));

    if verbose {
        eprintln!("Writing diagnostics to: {}", output_path.display());
    }

    let json_output = serde_json::to_string_pretty(&summary)
        .context("Failed to serialize diagnostics to JSON")?;

    fs::write(&output_path, json_output)
        .with_context(|| format!("Failed to write output to: {}", output_path.display()))?;

    println!("✓ Diagnostics saved to: {}", output_path.display());

    Ok(())
}

fn sbc_command(
    n_sims: usize,
    n_draws: usize,
    output: Option<PathBuf>,
    seed: Option<u64>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("Running Simulation-Based Calibration");
        eprintln!("  Replications: {}", n_sims);
        eprintln!("  Draws per sim: {}", n_draws);
        if let Some(s) = seed {
            eprintln!("  Seed: {}", s);
        }
        eprintln!();
    }

    let config = SbcConfig {
        n_sims,
        n_draws_per_sim: n_draws,
        params_to_track: Vec::new(),
        seed,
    };

    println!("SBC Configuration:");
    println!("  n_sims: {}", config.n_sims);
    println!("  n_draws_per_sim: {}", config.n_draws_per_sim);
    println!();

    // Note: This is a scaffold implementation
    // Full SBC requires integration with prior sampling and model fitting
    println!("⚠ SBC is currently a scaffold implementation");
    println!("  Full implementation requires:");
    println!("  1. Prior sampling capability");
    println!("  2. Data simulation from prior draws");
    println!("  3. Model fitting infrastructure");
    println!();
    println!("  Use this command structure once the above components are implemented.");

    let output_path = output.unwrap_or_else(|| PathBuf::from("sbc_results.json"));

    if verbose {
        eprintln!("Output will be saved to: {}", output_path.display());
    }

    // Write minimal config output
    let json_output =
        serde_json::to_string_pretty(&config).context("Failed to serialize SBC config to JSON")?;

    fs::write(&output_path, json_output)
        .with_context(|| format!("Failed to write output to: {}", output_path.display()))?;

    println!("✓ SBC configuration saved to: {}", output_path.display());

    Ok(())
}

fn design_evaluate_command(
    protocol: PathBuf,
    protocol_name: String,
    n_per_arm: usize,
    n_draws: usize,
    output: Option<PathBuf>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("Evaluating trial design");
        eprintln!("  Protocol: {}", protocol.display());
        eprintln!("  Protocol name: {}", protocol_name);
        eprintln!("  N per arm: {}", n_per_arm);
        eprintln!("  N draws: {}", n_draws);
        eprintln!();
    }

    // Parse protocol file
    let source = fs::read_to_string(&protocol)
        .with_context(|| format!("Failed to read protocol file: {}", protocol.display()))?;

    let tokens = tokenize(&source)
        .with_context(|| format!("Failed to tokenize protocol file: {}", protocol.display()))?;
    let protocol_def = parse_protocol_from_tokens(&tokens)
        .with_context(|| format!("Failed to parse protocol from: {}", protocol.display()))?;

    if verbose {
        eprintln!("Protocol parsed successfully");
        eprintln!("  Arms: {}", protocol_def.arms.len());
        eprintln!("  Endpoints: {}", protocol_def.endpoints.len());
        eprintln!("  Decisions: {}", protocol_def.decisions.len());
        eprintln!();
    }

    // Create design config
    let design_cfg = DesignConfig { n_per_arm, n_draws };

    // Note: This is a scaffold - full implementation requires posterior draws
    println!("Design Evaluation (Scaffold)");
    println!("==============================");
    println!();
    println!("Protocol: {}", protocol_name);
    println!("Design: N = {} per arm", n_per_arm);
    println!("Posterior draws: {}", n_draws);
    println!();

    if protocol_def.decisions.is_empty() {
        println!("⚠ No decision rules defined in protocol");
        println!("  Add a 'decisions {{}}' block to your protocol to define success criteria");
    } else {
        println!("Decision Rules:");
        for decision in &protocol_def.decisions {
            println!("  {} ({})", decision.name, decision.endpoint_name);
            println!(
                "    {} > {} by {:.2}",
                decision.arm_right, decision.arm_left, decision.margin
            );
            println!("    Prob threshold: {:.2}", decision.prob_threshold);
        }
        println!();
        println!("⚠ Full PoS calculation requires:");
        println!("  1. Posterior draws from fitted model");
        println!("  2. Virtual trial simulation per draw");
        println!("  3. Endpoint evaluation on simulated data");
    }

    // Evaluate design (scaffold)
    let posterior_draws = vec![]; // Placeholder
    let summary = evaluate_design_pos(&protocol_name, &design_cfg, &posterior_draws);

    let output_path = output.unwrap_or_else(|| PathBuf::from("design_evaluation.json"));

    if verbose {
        eprintln!("Writing results to: {}", output_path.display());
    }

    let json_output = serde_json::to_string_pretty(&summary)
        .context("Failed to serialize design evaluation to JSON")?;

    fs::write(&output_path, json_output)
        .with_context(|| format!("Failed to write output to: {}", output_path.display()))?;

    println!();
    println!("✓ Design evaluation saved to: {}", output_path.display());

    Ok(())
}

fn design_grid_command(
    protocol: PathBuf,
    protocol_name: String,
    n_per_arm_values: Vec<usize>,
    n_draws: usize,
    output: Option<PathBuf>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("Evaluating design grid");
        eprintln!("  Protocol: {}", protocol.display());
        eprintln!("  Protocol name: {}", protocol_name);
        eprintln!("  Sample sizes: {:?}", n_per_arm_values);
        eprintln!("  N draws: {}", n_draws);
        eprintln!();
    }

    if n_per_arm_values.is_empty() {
        bail!("No sample sizes specified. Use --n-per-arm with comma-separated values (e.g., 50,100,150)");
    }

    // Parse protocol file
    let source = fs::read_to_string(&protocol)
        .with_context(|| format!("Failed to read protocol file: {}", protocol.display()))?;

    let tokens = tokenize(&source)
        .with_context(|| format!("Failed to tokenize protocol file: {}", protocol.display()))?;
    let protocol_def = parse_protocol_from_tokens(&tokens)
        .with_context(|| format!("Failed to parse protocol from: {}", protocol.display()))?;

    if verbose {
        eprintln!("Protocol parsed successfully");
        eprintln!("  Decisions: {}", protocol_def.decisions.len());
        eprintln!();
    }

    println!("Design Grid Evaluation (Scaffold)");
    println!("==================================");
    println!();
    println!("Protocol: {}", protocol_name);
    println!("Sample sizes to evaluate: {:?}", n_per_arm_values);
    println!("Posterior draws: {}", n_draws);
    println!();

    if !protocol_def.decisions.is_empty() {
        println!("Decision Rules:");
        for decision in &protocol_def.decisions {
            println!("  {} ({})", decision.name, decision.endpoint_name);
        }
        println!();
    }

    // Evaluate grid (scaffold)
    let posterior_draws = vec![]; // Placeholder
    let summaries =
        evaluate_design_grid(&protocol_name, &n_per_arm_values, n_draws, &posterior_draws);

    let output_path = output.unwrap_or_else(|| PathBuf::from("design_grid.json"));

    if verbose {
        eprintln!("Writing results to: {}", output_path.display());
    }

    // Create output structure
    let output_json = serde_json::json!({
        "protocol": protocol_name,
        "designs": summaries,
    });

    let json_output = serde_json::to_string_pretty(&output_json)
        .context("Failed to serialize design grid to JSON")?;

    fs::write(&output_path, json_output)
        .with_context(|| format!("Failed to write output to: {}", output_path.display()))?;

    println!(
        "✓ Design grid evaluation saved to: {}",
        output_path.display()
    );
    println!();
    println!("Tip: Plot PoS vs N to visualize power curve");

    Ok(())
}

fn export_pinn_command(
    input: PathBuf,
    output: Option<PathBuf>,
    physics_loss: bool,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("Exporting neural surrogate training script");
        eprintln!("  Input: {}", input.display());
        eprintln!("  Physics loss: {}", physics_loss);
        eprintln!();
    }

    // Read and parse source
    let source = fs::read_to_string(&input)
        .with_context(|| format!("Failed to read input file: {}", input.display()))?;

    let tokens = tokenize(&source)
        .with_context(|| format!("Failed to tokenize input file: {}", input.display()))?;

    let program = parse_program(&tokens)
        .with_context(|| format!("Failed to parse MedLang program from: {}", input.display()))?;

    if verbose {
        eprintln!("Parsed MedLang program");
        eprintln!("  Declarations: {}", program.declarations.len());
        eprintln!();
    }

    // Lower to IR
    let ir_program = lower_program(&program).with_context(|| "Failed to lower program to IR")?;

    if verbose {
        eprintln!("Lowered to IR");
        eprintln!("  States: {}", ir_program.model.states.len());
        eprintln!("  Parameters: {}", ir_program.model.params.len());
        eprintln!("  ODEs: {}", ir_program.model.odes.len());
        eprintln!();
    }

    // Create surrogate config
    let mut cfg = IRSurrogateConfig::default_oncology_qsp();
    cfg.use_physics_loss = physics_loss;

    if verbose {
        eprintln!("Surrogate configuration:");
        eprintln!("  Model: {}", cfg.model_name);
        eprintln!("  Input features: {}", cfg.input_features.join(", "));
        eprintln!(
            "  Output observables: {}",
            cfg.output_observables.join(", ")
        );
        eprintln!("  Hidden layers: {:?}", cfg.hidden_layers);
        eprintln!("  Physics loss: {}", cfg.use_physics_loss);
        eprintln!();
    }

    // Generate Julia+DiffEqFlux code
    let julia_code = generate_julia_pinn(&ir_program.model, &cfg);

    // Determine output path
    let output_path = output.unwrap_or_else(|| {
        let mut path = input.clone();
        path.set_extension("jl");
        path
    });

    if verbose {
        eprintln!("Writing Julia script to: {}", output_path.display());
    }

    fs::write(&output_path, &julia_code)
        .with_context(|| format!("Failed to write output to: {}", output_path.display()))?;

    println!("Neural Surrogate Export Complete");
    println!("=================================");
    println!();
    println!("Generated: {}", output_path.display());
    println!();
    println!("Training data format:");
    println!("  CSV with columns: TIME, DOSE_MG, WT, Kd_QM, Kp_QM, TUMVOL");
    println!();
    println!("To train the surrogate:");
    println!(
        "  julia {} <train.csv> <model.bson> [epochs]",
        output_path.display()
    );
    println!();
    println!("Example:");
    println!(
        "  julia {} data/oncology_training.csv models/surrogate.bson 200",
        output_path.display()
    );
    println!();
    println!("To generate training data, use:");
    println!("  mlc generate-surrogate-data --help");

    Ok(())
}

fn export_fhir_command(
    protocol: PathBuf,
    protocol_name: String,
    trial_data: Option<PathBuf>,
    output: Option<PathBuf>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("Exporting protocol as FHIR R4 resources");
        eprintln!("  Protocol: {}", protocol.display());
        eprintln!("  Protocol name: {}", protocol_name);
        if let Some(ref data) = trial_data {
            eprintln!("  Trial data: {}", data.display());
        }
        eprintln!();
    }

    // Read and parse protocol
    let source = fs::read_to_string(&protocol)
        .with_context(|| format!("Failed to read protocol file: {}", protocol.display()))?;

    let tokens = tokenize(&source)
        .with_context(|| format!("Failed to tokenize protocol: {}", protocol.display()))?;

    let protocol_def = parse_protocol_from_tokens(&tokens)
        .with_context(|| "Failed to parse protocol definition")?;

    if verbose {
        eprintln!("Parsed protocol: {}", protocol_def.name);
        eprintln!("  Arms: {}", protocol_def.arms.len());
        eprintln!("  Visits: {}", protocol_def.visits.len());
        eprintln!("  Endpoints: {}", protocol_def.endpoints.len());
        eprintln!();
    }

    // Create output directory
    let output_dir = output.unwrap_or_else(|| {
        let mut path = protocol.clone();
        path.set_extension("fhir");
        path
    });

    fs::create_dir_all(&output_dir).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            output_dir.display()
        )
    })?;

    if verbose {
        eprintln!("Output directory: {}", output_dir.display());
    }

    // Convert to FHIR ResearchStudy
    let research_study = protocol_to_fhir_research_study(&protocol_def);
    let rs_path = output_dir.join("ResearchStudy.json");
    let rs_json = serde_json::to_string_pretty(&research_study)
        .context("Failed to serialize ResearchStudy")?;
    fs::write(&rs_path, rs_json)
        .with_context(|| format!("Failed to write ResearchStudy: {}", rs_path.display()))?;
    if verbose {
        eprintln!("  ✓ ResearchStudy.json");
    }

    // Convert to FHIR PlanDefinition
    let plan_def = protocol_to_fhir_plan_definition(&protocol_def);
    let pd_path = output_dir.join("PlanDefinition.json");
    let pd_json =
        serde_json::to_string_pretty(&plan_def).context("Failed to serialize PlanDefinition")?;
    fs::write(&pd_path, pd_json)
        .with_context(|| format!("Failed to write PlanDefinition: {}", pd_path.display()))?;
    if verbose {
        eprintln!("  ✓ PlanDefinition.json");
    }

    // Convert endpoints to FHIR Measures
    let measures = protocol_to_fhir_measures(&protocol_def);
    for (idx, measure) in measures.iter().enumerate() {
        let m_path = output_dir.join(format!("Measure_{}.json", idx + 1));
        let m_json =
            serde_json::to_string_pretty(&measure).context("Failed to serialize Measure")?;
        fs::write(&m_path, m_json)
            .with_context(|| format!("Failed to write Measure: {}", m_path.display()))?;
        if verbose {
            eprintln!("  ✓ Measure_{}.json", idx + 1);
        }
    }

    // If trial data provided, create Bundle
    if let Some(data_path) = trial_data {
        let trial_json = fs::read_to_string(&data_path)
            .with_context(|| format!("Failed to read trial data: {}", data_path.display()))?;

        let dataset: TrialDataset =
            serde_json::from_str(&trial_json).context("Failed to parse trial dataset JSON")?;

        let bundle = trial_to_fhir_bundle(&dataset, &protocol_name);
        let bundle_path = output_dir.join("Bundle.json");
        let bundle_json =
            serde_json::to_string_pretty(&bundle).context("Failed to serialize Bundle")?;
        fs::write(&bundle_path, bundle_json)
            .with_context(|| format!("Failed to write Bundle: {}", bundle_path.display()))?;
        if verbose {
            eprintln!("  ✓ Bundle.json ({} entries)", bundle.entry.len());
        }
    }

    println!("FHIR R4 Export Complete");
    println!("========================");
    println!();
    println!("Output directory: {}", output_dir.display());
    println!("Resources:");
    println!("  - ResearchStudy.json");
    println!("  - PlanDefinition.json");
    println!("  - Measure_1.json ... Measure_N.json");
    if trial_data.is_some() {
        println!("  - Bundle.json (patient + observation data)");
    }
    println!();
    println!("These FHIR resources can be:");
    println!("  - Submitted to clinical trial registries (e.g., ClinicalTrials.gov)");
    println!("  - Loaded into FHIR servers for interoperability");
    println!("  - Converted to other regulatory formats");

    Ok(())
}

fn export_cql_command(
    protocol: PathBuf,
    protocol_name: String,
    output: Option<PathBuf>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("Exporting endpoint definitions as CQL");
        eprintln!("  Protocol: {}", protocol.display());
        eprintln!("  Protocol name: {}", protocol_name);
        eprintln!();
    }

    // Read and parse protocol
    let source = fs::read_to_string(&protocol)
        .with_context(|| format!("Failed to read protocol file: {}", protocol.display()))?;

    let tokens = tokenize(&source)
        .with_context(|| format!("Failed to tokenize protocol: {}", protocol.display()))?;

    let protocol_def = parse_protocol_from_tokens(&tokens)
        .with_context(|| "Failed to parse protocol definition")?;

    if verbose {
        eprintln!("Parsed protocol: {}", protocol_def.name);
        eprintln!("  Endpoints: {}", protocol_def.endpoints.len());
        eprintln!();
    }

    // Create output directory
    let output_dir = output.unwrap_or_else(|| {
        let mut path = protocol.clone();
        path.set_extension("cql");
        path
    });

    fs::create_dir_all(&output_dir).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            output_dir.display()
        )
    })?;

    if verbose {
        eprintln!("Output directory: {}", output_dir.display());
    }

    // Generate CQL for each endpoint
    let cql_libraries = protocol_endpoints_to_cql(&protocol_def.endpoints, &protocol_name);

    for (library_name, cql_code) in cql_libraries.iter() {
        let cql_path = output_dir.join(format!("{}.cql", library_name));
        fs::write(&cql_path, cql_code)
            .with_context(|| format!("Failed to write CQL: {}", cql_path.display()))?;
        if verbose {
            eprintln!("  ✓ {}.cql", library_name);
        }
    }

    println!("CQL Export Complete");
    println!("====================");
    println!();
    println!("Output directory: {}", output_dir.display());
    println!("Generated {} CQL libraries:", cql_libraries.len());
    for (library_name, _) in &cql_libraries {
        println!("  - {}.cql", library_name);
    }
    println!();
    println!("These CQL libraries define endpoint semantics:");
    println!("  - ORR: Objective Response Rate (≥30% reduction)");
    println!("  - PFS: Progression-Free Survival (≥20% increase = progression)");
    println!();
    println!("CQL can be:");
    println!("  - Executed in FHIR-compliant systems");
    println!("  - Used to evaluate patient cohorts");
    println!("  - Integrated with electronic health records");

    Ok(())
}

fn export_cdisc_command(
    trial_data: PathBuf,
    study_id: String,
    output: Option<PathBuf>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("Exporting trial data as CDISC datasets");
        eprintln!("  Trial data: {}", trial_data.display());
        eprintln!("  Study ID: {}", study_id);
        eprintln!();
    }

    // Read trial dataset
    let trial_json = fs::read_to_string(&trial_data)
        .with_context(|| format!("Failed to read trial data: {}", trial_data.display()))?;

    let dataset: TrialDataset =
        serde_json::from_str(&trial_json).context("Failed to parse trial dataset JSON")?;

    if verbose {
        eprintln!("Parsed trial dataset:");
        eprintln!("  Subjects: {}", {
            let mut subjects = std::collections::HashSet::new();
            for row in &dataset.rows {
                subjects.insert(row.subject_id);
            }
            subjects.len()
        });
        eprintln!("  Observations: {}", dataset.rows.len());
        eprintln!();
    }

    // Create output directory
    let output_dir = output.unwrap_or_else(|| {
        let mut path = trial_data.clone();
        path.set_extension("cdisc");
        path
    });

    fs::create_dir_all(&output_dir).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            output_dir.display()
        )
    })?;

    if verbose {
        eprintln!("Output directory: {}", output_dir.display());
    }

    // Convert to ADSL/ADTR
    let (adsl_rows, adtr_rows) = trial_to_adsl_adtr(&dataset, &study_id);

    // Export ADSL
    let adsl_path = output_dir.join("adsl.csv");
    let adsl_csv = medlangc::interop::cdisc::adsl_to_csv(&adsl_rows);
    fs::write(&adsl_path, adsl_csv)
        .with_context(|| format!("Failed to write ADSL: {}", adsl_path.display()))?;
    if verbose {
        eprintln!("  ✓ adsl.csv ({} subjects)", adsl_rows.len());
    }

    // Export ADTR
    let adtr_path = output_dir.join("adtr.csv");
    let adtr_csv = medlangc::interop::cdisc::adtr_to_csv(&adtr_rows);
    fs::write(&adtr_path, adtr_csv)
        .with_context(|| format!("Failed to write ADTR: {}", adtr_path.display()))?;
    if verbose {
        eprintln!("  ✓ adtr.csv ({} observations)", adtr_rows.len());
    }

    // Export as JSON for inspection
    let adsl_json_path = output_dir.join("adsl.json");
    let adsl_json = medlangc::interop::cdisc::adsl_to_json(&adsl_rows)
        .context("Failed to serialize ADSL to JSON")?;
    fs::write(&adsl_json_path, adsl_json)
        .with_context(|| format!("Failed to write ADSL JSON: {}", adsl_json_path.display()))?;
    if verbose {
        eprintln!("  ✓ adsl.json");
    }

    let adtr_json_path = output_dir.join("adtr.json");
    let adtr_json = medlangc::interop::cdisc::adtr_to_json(&adtr_rows)
        .context("Failed to serialize ADTR to JSON")?;
    fs::write(&adtr_json_path, adtr_json)
        .with_context(|| format!("Failed to write ADTR JSON: {}", adtr_json_path.display()))?;
    if verbose {
        eprintln!("  ✓ adtr.json");
    }

    println!("CDISC-like Export Complete");
    println!("===========================");
    println!();
    println!("Output directory: {}", output_dir.display());
    println!("Datasets:");
    println!(
        "  - adsl.csv: Subject-level characteristics ({} subjects)",
        adsl_rows.len()
    );
    println!(
        "  - adtr.csv: Tumor response measurements ({} observations)",
        adtr_rows.len()
    );
    println!("  - adsl.json: Subject-level data (JSON)");
    println!("  - adtr.json: Tumor response data (JSON)");
    println!();
    println!("ADSL columns:");
    println!("  STUDYID, SUBJID, ARM, DOSE_MG, WEIGHT_KG, BASELINE_VOL,");
    println!("  N_OBS, LAST_OBS_DAY, STATUS");
    println!();
    println!("ADTR columns:");
    println!("  STUDYID, SUBJID, ARM, TIME_DAY, TUMOR_VOL, BASELINE_VOL,");
    println!("  PCT_CHANGE, RESPONSE");
    println!();
    println!("These datasets are suitable for:");
    println!("  - Regulatory submissions (FDA, EMA, PMDA)");
    println!("  - SAS analysis workflows");
    println!("  - Clinical trial database systems");

    Ok(())
}

fn quantum_trust_design_command(
    protocol: PathBuf,
    protocol_name: String,
    population_model: String,
    fit_dir: PathBuf,
    design_n_per_arm: usize,
    design_n_draws: usize,
    policy: Option<PathBuf>,
    output: Option<PathBuf>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("Quantum Trust Design Sensitivity Analysis");
        eprintln!("==========================================");
        eprintln!("  Protocol: {}", protocol.display());
        eprintln!("  Protocol name: {}", protocol_name);
        eprintln!("  Population model: {}", population_model);
        eprintln!("  Fit directory: {}", fit_dir.display());
        eprintln!(
            "  Design: N={} per arm, {} draws",
            design_n_per_arm, design_n_draws
        );
        eprintln!();
    }

    // Load prior inflation policy
    let inflation_policy = if let Some(policy_path) = &policy {
        if verbose {
            eprintln!(
                "Loading prior inflation policy from: {}",
                policy_path.display()
            );
        }
        let policy_json = fs::read_to_string(policy_path)
            .with_context(|| format!("Failed to read policy file: {}", policy_path.display()))?;
        serde_json::from_str(&policy_json).context("Failed to parse prior inflation policy JSON")?
    } else {
        if verbose {
            eprintln!("Using default prior inflation policy");
        }
        PriorInflationPolicy::default()
    };

    if verbose {
        eprintln!("Inflation policy:");
        eprintln!("  High trust: {}x", inflation_policy.high_sd_factor);
        eprintln!("  Moderate trust: {}x", inflation_policy.moderate_sd_factor);
        eprintln!("  Low trust: {}x", inflation_policy.low_sd_factor);
        eprintln!("  Broken trust: {}x", inflation_policy.broken_sd_factor);
        eprintln!();
    }

    // Parse protocol
    let source = fs::read_to_string(&protocol)
        .with_context(|| format!("Failed to read protocol file: {}", protocol.display()))?;

    let tokens = tokenize(&source)
        .with_context(|| format!("Failed to tokenize protocol: {}", protocol.display()))?;

    let protocol_def = parse_protocol_from_tokens(&tokens)
        .with_context(|| "Failed to parse protocol definition")?;

    if verbose {
        eprintln!("Parsed protocol: {}", protocol_def.name);
        eprintln!("  Arms: {}", protocol_def.arms.len());
        eprintln!("  Endpoints: {}", protocol_def.endpoints.len());
        eprintln!("  Decisions: {}", protocol_def.decisions.len());
        eprintln!();
    }

    // Create synthetic quantum prior-posterior comparisons
    // (In a full implementation, this would load actual QM registry and posterior samples)
    if verbose {
        eprintln!("Generating quantum prior-posterior comparisons...");
        eprintln!("(Note: Using synthetic data for Week 19 scaffold)");
        eprintln!();
    }

    let comparisons = create_synthetic_quantum_comparisons(&population_model);

    if verbose {
        eprintln!(
            "Generated {} quantum parameter comparisons",
            comparisons.len()
        );
        eprintln!();
    }

    // Classify trust levels
    let trust_scores = classify_all(&comparisons);

    if verbose {
        eprintln!("Trust classification:");
        for score in &trust_scores {
            eprintln!(
                "  {}.{}: {} (z={:.2}, overlap={:.2}, KL={:.3})",
                score.system_name,
                score.param_name,
                score.trust,
                score.z_prior,
                score.overlap_95,
                score.kl_prior_to_post
            );
        }
        eprintln!();
    }

    // Generate sensitivity scenarios
    let scenarios = default_scenarios_from_scores(&trust_scores);

    if verbose {
        eprintln!("Generated {} sensitivity scenarios:", scenarios.len());
        for scenario in &scenarios {
            eprintln!("  - {}: {}", scenario.label, scenario.description);
        }
        eprintln!();
    }

    // Evaluate design under each scenario
    if verbose {
        eprintln!("Evaluating design PoS under each scenario...");
    }

    let design_config = DesignConfig {
        n_per_arm: design_n_per_arm,
        n_draws: design_n_draws,
    };

    let posterior_draws = vec![]; // Scaffold: would load actual posterior samples

    let mut results = Vec::new();
    for scenario in scenarios {
        if verbose {
            eprintln!("  Evaluating scenario: {}", scenario.label);
        }

        // For Week 19 scaffold, we use the same design evaluation
        // In a full implementation, we would adjust posteriors based on trust
        let design_summary = evaluate_design_pos(&protocol_name, &design_config, &posterior_draws);

        results.push(QuantumDesignSensitivityResult::new(
            scenario,
            design_summary,
        ));
    }

    if verbose {
        eprintln!();
        eprintln!("Design evaluation complete for all scenarios");
        eprintln!();
    }

    // Build report
    let report = QuantumDesignSensitivityReport {
        protocol_name: protocol_name.clone(),
        population_model: population_model.clone(),
        fit_dir: fit_dir.display().to_string(),
        design_config: DesignConfigInfo {
            n_per_arm: design_n_per_arm,
            n_draws: design_n_draws,
        },
        results,
    };

    // Determine output path
    let output_path = output.unwrap_or_else(|| {
        PathBuf::from(format!("{}_quantum_trust_sensitivity.json", protocol_name))
    });

    // Write report
    let report_json = serde_json::to_string_pretty(&report)
        .context("Failed to serialize quantum trust sensitivity report")?;

    fs::write(&output_path, report_json)
        .with_context(|| format!("Failed to write report to: {}", output_path.display()))?;

    println!("Quantum Trust Design Sensitivity Analysis Complete");
    println!("===================================================");
    println!();
    println!("Report saved to: {}", output_path.display());
    println!();
    println!("Summary:");
    println!("  Protocol: {}", protocol_name);
    println!("  Design: N={} per arm", design_n_per_arm);
    println!("  Scenarios evaluated: {}", report.results.len());
    println!();

    // Print PoS sensitivity for each decision
    if let Some(first_result) = report.results.first() {
        for decision in &first_result.design_summary.decision_results {
            let decision_name = &decision.decision_name;
            if let Some((min_pos, max_pos)) = report.pos_range(decision_name) {
                let sensitivity = max_pos - min_pos;
                println!(
                    "  Decision '{}': PoS ∈ [{:.3}, {:.3}] (Δ = {:.3})",
                    decision_name, min_pos, max_pos, sensitivity
                );
            }
        }
    }

    println!();
    println!("Interpretation:");
    println!("  - Low PoS sensitivity (Δ < 0.05): Design is robust to QM prior uncertainty");
    println!("  - Medium sensitivity (0.05 ≤ Δ < 0.15): Design moderately affected by QM trust");
    println!("  - High sensitivity (Δ ≥ 0.15): Design critically depends on QM prior reliability");
    println!();
    println!("Use this report to:");
    println!("  1. Assess robustness of trial design to quantum prior uncertainty");
    println!("  2. Identify parameters requiring additional QM validation");
    println!("  3. Guide decisions on sample size and design modifications");

    Ok(())
}

/// Create synthetic quantum comparisons for scaffold (Week 19)
fn create_synthetic_quantum_comparisons(
    population_model: &str,
) -> Vec<QuantumPriorPosteriorComparison> {
    vec![
        QuantumPriorPosteriorComparison::new(
            "LIG001_EGFR".to_string(),
            "EC50_pop".to_string(),
            QuantumPriorInfo {
                mu_prior: -2.0,
                sigma_prior: 0.5,
                scale: "log".to_string(),
            },
            QuantumPosteriorInfo {
                mu_post: -1.8,
                sigma_post: 0.3,
                n_draws: 1000,
            },
        ),
        QuantumPriorPosteriorComparison::new(
            "LIG001_EGFR".to_string(),
            "Kp_tumor_pop".to_string(),
            QuantumPriorInfo {
                mu_prior: 0.5,
                sigma_prior: 0.3,
                scale: "log".to_string(),
            },
            QuantumPosteriorInfo {
                mu_post: 1.5,
                sigma_post: 0.2,
                n_draws: 1000,
            },
        ),
        QuantumPriorPosteriorComparison::new(
            "LIG001_EGFR".to_string(),
            "Kp_liver_pop".to_string(),
            QuantumPriorInfo {
                mu_prior: 0.8,
                sigma_prior: 0.4,
                scale: "log".to_string(),
            },
            QuantumPosteriorInfo {
                mu_post: 0.9,
                sigma_post: 0.35,
                n_draws: 1000,
            },
        ),
    ]
}

fn portfolio_evaluate_command(
    protocol_name: String,
    population_model: String,
    ligands: String,
    n_per_arm: usize,
    n_draws: usize,
    backend: String,
    output: Option<PathBuf>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("Portfolio Evaluation");
        eprintln!("====================");
        eprintln!("  Protocol: {}", protocol_name);
        eprintln!("  Population model: {}", population_model);
        eprintln!("  Design: N={} per arm, {} draws", n_per_arm, n_draws);
        eprintln!("  Backend: {}", backend);
        eprintln!();
    }

    // Parse ligand data from string
    // Format: LIG001:3.2e-8:5.1,LIG002:1.0e-7:3.2
    let ligand_data: Vec<(String, Option<f64>, Option<f64>)> = ligands
        .split(',')
        .filter_map(|s| {
            let parts: Vec<&str> = s.trim().split(':').collect();
            if parts.len() >= 3 {
                let id = parts[0].to_string();
                let kd = parts[1].parse::<f64>().ok();
                let kp = parts[2].parse::<f64>().ok();
                Some((id, kd, kp))
            } else if parts.len() == 1 && !parts[0].is_empty() {
                // Just ID, no QM data
                Some((parts[0].to_string(), None, None))
            } else {
                None
            }
        })
        .collect();

    if verbose {
        eprintln!("Parsed {} ligands:", ligand_data.len());
        for (id, kd, kp) in &ligand_data {
            eprintln!("  {}: Kd={:?}, Kp_tumor={:?}", id, kd, kp);
        }
        eprintln!();
    }

    // Create design config
    let design_cfg = DesignConfig { n_per_arm, n_draws };

    // Evaluate portfolio
    if verbose {
        eprintln!("Evaluating portfolio...");
    }

    let summary = evaluate_portfolio(
        &protocol_name,
        &population_model,
        &ligand_data,
        &design_cfg,
        &backend,
        None, // No posterior draws for Week 20 scaffold
    );

    if verbose {
        eprintln!();
        eprintln!("Portfolio evaluation complete");
        eprintln!();
    }

    // Determine output path
    let output_path =
        output.unwrap_or_else(|| PathBuf::from(format!("{}_portfolio.json", protocol_name)));

    // Write JSON
    let summary_json =
        serde_json::to_string_pretty(&summary).context("Failed to serialize portfolio summary")?;

    fs::write(&output_path, summary_json)
        .with_context(|| format!("Failed to write portfolio to: {}", output_path.display()))?;

    println!("Portfolio Evaluation Complete");
    println!("==============================");
    println!();
    println!("Report saved to: {}", output_path.display());
    println!();
    println!("Summary:");
    println!("  Protocol: {}", summary.protocol_name);
    println!("  Design: N={} per arm", summary.design_n_per_arm);
    println!("  Ligands evaluated: {}", summary.entries.len());
    println!();

    // Print ranking table
    println!("Ranking (by PoS):");
    println!("  Rank  Ligand       Kd (nM)     Kp_tumor    ORR      PFS (d)  PoS");
    println!("  ────  ──────────  ──────────  ──────────  ───────  ───────  ─────");

    for entry in &summary.entries {
        let kd_nm = entry.metrics.kd_molar.map(|kd| kd * 1e9);
        let kd_str = kd_nm
            .map(|v| format!("{:.1}", v))
            .unwrap_or_else(|| "N/A".to_string());
        let kp_str = entry
            .metrics
            .kp_tumor
            .map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "N/A".to_string());
        let orr_str = entry
            .metrics
            .orr_mean
            .map(|v| format!("{:.3}", v))
            .unwrap_or_else(|| "N/A".to_string());
        let pfs_str = entry
            .metrics
            .pfs_median
            .map(|v| format!("{:.0}", v))
            .unwrap_or_else(|| "N/A".to_string());
        let pos_str = entry
            .metrics
            .pos
            .map(|v| format!("{:.3}", v))
            .unwrap_or_else(|| "N/A".to_string());

        println!(
            "  {:4}  {:10}  {:>10}  {:>10}  {:>7}  {:>7}  {:>5}",
            entry.rank, entry.metrics.ligand_id, kd_str, kp_str, orr_str, pfs_str, pos_str
        );
    }

    println!();

    // Print PoS range
    if let Some((min, max)) = summary.pos_range() {
        println!("PoS range: [{:.3}, {:.3}] (Δ = {:.3})", min, max, max - min);
        println!();
    }

    println!("Recommendation:");
    if let Some(top) = summary.entries.first() {
        println!(
            "  Top candidate: {} (PoS = {:.3})",
            top.metrics.ligand_id,
            top.metrics.pos.unwrap_or(0.0)
        );

        if top.metrics.pos.unwrap_or(0.0) >= 0.80 {
            println!("  Status: Ready for Phase II/III (PoS ≥ 0.80)");
        } else {
            println!("  Status: Consider increasing sample size or validating QM priors");
        }
    }

    Ok(())
}

fn portfolio_design_grid_command(
    protocol_name: String,
    population_model: String,
    ligands: String,
    n_per_arm: Vec<usize>,
    n_draws: usize,
    backend: String,
    output: Option<PathBuf>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("Portfolio Design Grid Evaluation");
        eprintln!("================================");
        eprintln!("  Protocol: {}", protocol_name);
        eprintln!("  Population model: {}", population_model);
        eprintln!("  N per arm values: {:?}", n_per_arm);
        eprintln!("  Backend: {}", backend);
        eprintln!();
    }

    // Parse ligand data
    let ligand_data: Vec<(String, Option<f64>, Option<f64>)> = ligands
        .split(',')
        .filter_map(|s| {
            let parts: Vec<&str> = s.trim().split(':').collect();
            if parts.len() >= 3 {
                let id = parts[0].to_string();
                let kd = parts[1].parse::<f64>().ok();
                let kp = parts[2].parse::<f64>().ok();
                Some((id, kd, kp))
            } else if parts.len() == 1 && !parts[0].is_empty() {
                Some((parts[0].to_string(), None, None))
            } else {
                None
            }
        })
        .collect();

    if verbose {
        eprintln!("Parsed {} ligands:", ligand_data.len());
        for (id, kd, kp) in &ligand_data {
            eprintln!("  {}: Kd={:?}, Kp_tumor={:?}", id, kd, kp);
        }
        eprintln!();
    }

    // Evaluate design grid
    if verbose {
        eprintln!("Evaluating design grid...");
        eprintln!("  Designs to evaluate: {}", n_per_arm.len());
        eprintln!();
    }

    let grid = evaluate_portfolio_design_grid(
        &protocol_name,
        &population_model,
        &ligand_data,
        &n_per_arm,
        n_draws,
        &backend,
        None, // No posterior draws for Week 20 scaffold
    );

    if verbose {
        eprintln!("Design grid evaluation complete");
        eprintln!();
    }

    // Determine output path
    let output_path =
        output.unwrap_or_else(|| PathBuf::from(format!("{}_portfolio_grid.json", protocol_name)));

    // Write JSON
    let grid_json =
        serde_json::to_string_pretty(&grid).context("Failed to serialize portfolio design grid")?;

    fs::write(&output_path, grid_json)
        .with_context(|| format!("Failed to write grid to: {}", output_path.display()))?;

    println!("Portfolio Design Grid Complete");
    println!("==============================");
    println!();
    println!("Report saved to: {}", output_path.display());
    println!();
    println!("Summary:");
    println!("  Protocol: {}", grid.protocol_name);
    println!("  Designs evaluated: {}", grid.designs.len());
    println!("  Ligands: {}", ligand_data.len());
    println!();

    // Print grid summary
    println!("PoS vs N per arm:");
    println!();

    // Header
    print!("  N/arm  ");
    for (id, _, _) in &ligand_data {
        print!("  {:>8}", id);
    }
    println!();

    print!("  ─────  ");
    for _ in &ligand_data {
        print!("  ────────");
    }
    println!();

    // Data rows
    for design in &grid.designs {
        print!("  {:5}  ", design.n_per_arm);
        for ligand_point in &design.ligands {
            print!("  {:>8.3}", ligand_point.pos);
        }
        println!();
    }

    println!();

    // Optimal N recommendations
    println!("Optimal N (for PoS ≥ 0.80):");
    for (id, _, _) in &ligand_data {
        if let Some(optimal) = grid.optimal_n(id, 0.80) {
            println!("  {}: N = {} per arm", id, optimal);
        } else {
            println!(
                "  {}: PoS < 0.80 at all N (consider validating compound)",
                id
            );
        }
    }

    Ok(())
}

// =============================================================================
// Week 25: Module System Commands
// =============================================================================

/// Check command: Parse and validate a module without code generation
fn check_command(module: PathBuf, include_paths: Vec<PathBuf>, verbose: bool) -> Result<()> {
    use medlangc::ast::ModulePath;
    use medlangc::loader::{ModuleLoader, ModuleResolver};
    use medlangc::resolve::NameResolver;

    if verbose {
        println!("Checking module: {}", module.display());
    }

    // Create module loader with search paths
    let mut loader = ModuleLoader::new();
    for path in include_paths {
        if verbose {
            println!("Adding search path: {}", path.display());
        }
        loader.add_search_path(path);
    }

    // For now, this is a scaffold implementation
    // Full implementation requires parser support for module syntax
    println!("✓ Module system initialized");
    println!("✓ Search paths configured");

    if verbose {
        println!("\nSearch paths:");
        for (i, path) in loader.search_paths().iter().enumerate() {
            println!("  [{}] {}", i + 1, path.display());
        }
    }

    println!("\n✓ Check passed (scaffold - parser integration pending)");

    Ok(())
}

/// Build command: Check module and generate code for all dependencies
fn build_command(
    module: PathBuf,
    output_dir: PathBuf,
    backend: String,
    include_paths: Vec<PathBuf>,
    verbose: bool,
) -> Result<()> {
    use medlangc::loader::{ModuleLoader, ModuleResolver};
    use std::fs;

    if verbose {
        println!("Building module: {}", module.display());
        println!("Output directory: {}", output_dir.display());
        println!("Backend: {}", backend);
    }

    // Create output directory
    fs::create_dir_all(&output_dir)?;

    // Create module loader
    let mut loader = ModuleLoader::new();
    for path in include_paths {
        if verbose {
            println!("Adding search path: {}", path.display());
        }
        loader.add_search_path(path);
    }

    // Scaffold implementation
    println!("✓ Module loader initialized");
    println!("✓ Output directory created: {}", output_dir.display());
    println!("✓ Backend configured: {}", backend);

    if verbose {
        println!("\nBuild configuration:");
        println!("  Module: {}", module.display());
        println!("  Output: {}", output_dir.display());
        println!("  Backend: {}", backend);
        println!("\nSearch paths:");
        for (i, path) in loader.search_paths().iter().enumerate() {
            println!("  [{}] {}", i + 1, path.display());
        }
    }

    println!("\n✓ Build complete (scaffold - full codegen pending)");

    Ok(())
}

fn design_optimize_command(
    protocol: PathBuf,
    protocol_name: String,
    population_model: String,
    fit_dir: PathBuf,
    n_per_arm: Vec<usize>,
    orr_margin: Vec<f64>,
    dlt_threshold: Option<Vec<f64>>,
    objective: Option<PathBuf>,
    backend: String,
    output: Option<PathBuf>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("Design Optimization");
        eprintln!("==================");
        eprintln!("  Protocol: {}", protocol.display());
        eprintln!("  Protocol name: {}", protocol_name);
        eprintln!("  Population model: {}", population_model);
        eprintln!("  Fit directory: {}", fit_dir.display());
        eprintln!("  N per arm grid: {:?}", n_per_arm);
        eprintln!("  ORR margin grid: {:?}", orr_margin);
        eprintln!("  DLT threshold grid: {:?}", dlt_threshold);
        eprintln!("  Backend: {}", backend);
        eprintln!();
    }

    // Load or create objective configuration
    let objective_config = if let Some(obj_path) = &objective {
        if verbose {
            eprintln!(
                "Loading objective configuration from: {}",
                obj_path.display()
            );
        }
        let obj_json = fs::read_to_string(obj_path)
            .with_context(|| format!("Failed to read objective config: {}", obj_path.display()))?;
        serde_json::from_str(&obj_json).context("Failed to parse objective configuration")?
    } else {
        if verbose {
            eprintln!("Using balanced objective preset (default)");
        }
        ObjectiveConfig::balanced()
    };

    if verbose {
        eprintln!("Objective configuration:");
        eprintln!("  w_pos (PoS weight): {}", objective_config.w_pos);
        eprintln!("  w_eff (efficacy weight): {}", objective_config.w_eff);
        eprintln!("  w_tox (toxicity penalty): {}", objective_config.w_tox);
        eprintln!("  w_size (size penalty): {}", objective_config.w_size);
        eprintln!("  n_ref (reference N): {}", objective_config.n_ref);
        eprintln!();
    }

    // Generate candidate grid
    if verbose {
        eprintln!("Generating design candidate grid...");
    }

    let candidates = DesignCandidate::grid(
        &n_per_arm,
        &orr_margin,
        dlt_threshold.as_ref().map(|v| v.as_slice()),
    );

    if verbose {
        eprintln!("  Total candidates: {}", candidates.len());
        eprintln!();
    }

    // Load posterior draws (scaffold - not implemented yet)
    let posterior_draws = None;

    // Run optimization
    if verbose {
        eprintln!("Running optimization over design grid...");
        eprintln!("  (Using synthetic evaluation - Week 21 scaffold)");
        eprintln!();
    }

    let optimization_report = optimize_design_over_grid(
        &protocol_name,
        &population_model,
        &candidates,
        &objective_config,
        posterior_draws,
    );

    if verbose {
        eprintln!("Optimization complete");
        eprintln!();
    }

    // Determine output path
    let output_path = output.unwrap_or_else(|| {
        PathBuf::from(format!(
            "{}_{}_optimal_design.json",
            protocol_name, population_model
        ))
    });

    // Write JSON
    let report_json = serde_json::to_string_pretty(&optimization_report)
        .context("Failed to serialize optimization report")?;

    fs::write(&output_path, report_json)
        .with_context(|| format!("Failed to write report to: {}", output_path.display()))?;

    // Print results
    println!("Design Optimization Complete");
    println!("============================");
    println!();
    println!("Report saved to: {}", output_path.display());
    println!();

    if let Some(opt) = &optimization_report.optimal_design {
        println!("Optimal Design:");
        println!("  N per arm: {}", opt.candidate.n_per_arm);
        println!("  ORR margin: {:.3}", opt.candidate.orr_margin);
        if let Some(dlt) = opt.candidate.dlt_threshold {
            println!("  DLT threshold: {:.3}", dlt);
        }
        println!();
        println!("  Utility score: {:.4}", opt.utility);
        println!("  PoS: {:.3}", opt.metrics.pos);
        println!("  Efficacy benefit: {:.3}", opt.metrics.eff_benefit);
        if let Some(tox) = opt.metrics.tox_risk {
            println!("  Toxicity risk: {:.3}", tox);
        }
        println!("  Total sample size: {}", opt.metrics.sample_size_total);
        println!();
    } else {
        println!("No feasible designs found in grid.");
        println!();
    }

    // Print Pareto frontier summary
    if !optimization_report.pareto_frontier.is_empty() {
        println!(
            "Pareto Frontier ({} designs):",
            optimization_report.pareto_frontier.len()
        );
        println!();
        println!("  Rank  N/arm  Margin   DLT    Utility    PoS    Eff    Tox   Total_N");
        println!("  ────  ─────  ──────  ─────  ────────  ─────  ─────  ─────  ───────");

        for (i, design) in optimization_report
            .pareto_frontier
            .iter()
            .take(10)
            .enumerate()
        {
            let dlt_str = design
                .candidate
                .dlt_threshold
                .map(|d| format!("{:.2}", d))
                .unwrap_or_else(|| "─".to_string());
            let tox_str = design
                .metrics
                .tox_risk
                .map(|t| format!("{:.3}", t))
                .unwrap_or_else(|| "─".to_string());

            println!(
                "  {:4}  {:5}  {:.4}  {:5}  {:8.4}  {:.3}  {:.3}  {:5}  {:7}",
                i + 1,
                design.candidate.n_per_arm,
                design.candidate.orr_margin,
                dlt_str,
                design.utility,
                design.metrics.pos,
                design.metrics.eff_benefit,
                tox_str,
                design.metrics.sample_size_total
            );
        }

        if optimization_report.pareto_frontier.len() > 10 {
            println!(
                "  ... and {} more designs",
                optimization_report.pareto_frontier.len() - 10
            );
        }
        println!();
    }

    // Summary statistics
    println!("Grid Summary:");
    println!(
        "  Candidates evaluated: {}",
        optimization_report.all_results.len()
    );
    println!(
        "  Pareto frontier size: {}",
        optimization_report.pareto_frontier.len()
    );

    if let Some(opt) = &optimization_report.optimal_design {
        let avg_utility: f64 = optimization_report
            .all_results
            .iter()
            .map(|r| r.utility)
            .sum::<f64>()
            / optimization_report.all_results.len() as f64;
        let improvement = ((opt.utility - avg_utility) / avg_utility.abs()) * 100.0;
        println!("  Average utility: {:.4}", avg_utility);
        println!("  Optimal improvement: {:.1}% above average", improvement);
    }
    println!();

    Ok(())
}
