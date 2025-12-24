// ============================================================================
// MedLang FFI Crate — libmedlang.rs
// ============================================================================
//
// C ABI exports for external language integration (C, Python, Julia, etc.)
// Build: cargo build --release (produces libmedlang.so / libmedlang.dylib)
//
// Location in MedLang repo: medlang/ffi/src/lib.rs
//
// Usage:
//   - C: #include "medlang.h" and link against libmedlang
//   - Python: ctypes.CDLL("libmedlang.so")
//   - Julia: ccall((:medlang_init, "libmedlang"), ...)
//
// ============================================================================

use std::ffi::{c_char, c_double, c_int, c_void, CStr, CString};
use std::ptr;
use std::slice;
use std::sync::Mutex;

// Import from medlang-core (when available as library)
// use medlang_core::{parse, compile, CIR, NIR};
// use medlang_runtime::{execute, ODESolver, QMBackend};

// ============================================================================
// OPAQUE TYPES
// ============================================================================

/// Opaque MedLang context handle
#[repr(C)]
pub struct MedLangContext {
    // Internal state
    _private: [u8; 0],
}

/// Opaque model handle
#[repr(C)]
pub struct MedLangModel {
    _private: [u8; 0],
}

/// Opaque fit result handle
#[repr(C)]
pub struct MedLangFitResult {
    _private: [u8; 0],
}

/// Opaque simulation result handle
#[repr(C)]
pub struct MedLangSimResult {
    _private: [u8; 0],
}

// ============================================================================
// RESULT STRUCTURES (C-compatible)
// ============================================================================

/// Parameter estimate with uncertainty
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ParamEstimate {
    pub name: [c_char; 64],
    pub value: c_double,
    pub se: c_double,
    pub cv_percent: c_double,
    pub ci_lower: c_double,
    pub ci_upper: c_double,
    pub omega: c_double,
    pub omega_se: c_double,
}

impl Default for ParamEstimate {
    fn default() -> Self {
        ParamEstimate {
            name: [0; 64],
            value: 0.0,
            se: 0.0,
            cv_percent: 0.0,
            ci_lower: 0.0,
            ci_upper: 0.0,
            omega: 0.0,
            omega_se: 0.0,
        }
    }
}

/// Fit result structure (C-compatible)
#[repr(C)]
#[derive(Clone)]
pub struct FitResultData {
    pub model_name: [c_char; 64],
    pub n_params: c_int,
    pub params: [ParamEstimate; 30],
    
    // Named accessors
    pub cl: ParamEstimate,
    pub v: ParamEstimate,
    pub vc: ParamEstimate,
    pub vp: ParamEstimate,
    pub q: ParamEstimate,
    pub ka: ParamEstimate,
    pub f_bio: ParamEstimate,
    
    // Fit statistics
    pub objective: c_double,
    pub aic: c_double,
    pub bic: c_double,
    pub r_squared: c_double,
    pub rmse: c_double,
    
    // Convergence
    pub n_iterations: c_int,
    pub n_function_evals: c_int,
    pub converged: c_int,  // 0 = false, 1 = true
    pub convergence_code: c_int,
    
    // Data info
    pub n_observations: c_int,
    pub n_subjects: c_int,
}

/// Simulation result structure (C-compatible)
#[repr(C)]
#[derive(Clone)]
pub struct SimResultData {
    pub model_name: [c_char; 64],
    pub n_times: c_int,
    pub times: [c_double; 1000],
    pub ipred: [c_double; 1000],
    pub pred: [c_double; 1000],
    pub median: [c_double; 1000],
    pub ci_lower: [c_double; 1000],
    pub ci_upper: [c_double; 1000],
    pub n_subjects: c_int,
    pub seed: c_int,
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

/// Error codes
#[repr(C)]
pub enum MedLangError {
    Success = 0,
    ParseError = 1,
    CompileError = 2,
    RuntimeError = 3,
    IOError = 4,
    InvalidArgument = 5,
    OutOfMemory = 6,
    NotFound = 7,
    NotImplemented = 8,
}

/// Thread-local error message
thread_local! {
    static LAST_ERROR: Mutex<String> = Mutex::new(String::new());
}

fn set_error(msg: &str) {
    LAST_ERROR.with(|e| {
        *e.lock().unwrap() = msg.to_string();
    });
}

/// Get last error message
#[no_mangle]
pub extern "C" fn medlang_get_error(buffer: *mut c_char, buffer_len: c_int) -> c_int {
    LAST_ERROR.with(|e| {
        let msg = e.lock().unwrap();
        if buffer.is_null() || buffer_len <= 0 {
            return msg.len() as c_int;
        }
        
        let bytes = msg.as_bytes();
        let copy_len = std::cmp::min(bytes.len(), (buffer_len - 1) as usize);
        
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), buffer as *mut u8, copy_len);
            *buffer.add(copy_len) = 0;
        }
        
        copy_len as c_int
    })
}

// ============================================================================
// CONTEXT MANAGEMENT
// ============================================================================

/// Internal context state
struct ContextInternal {
    // Parsed programs
    // programs: Vec<medlang_core::Program>,
    
    // Compiled models
    // models: HashMap<String, CompiledModel>,
    
    // Fit results
    fit_results: Vec<FitResultData>,
    
    // Simulation results
    sim_results: Vec<SimResultData>,
    
    // Configuration
    gpu_enabled: bool,
    num_threads: usize,
}

impl Default for ContextInternal {
    fn default() -> Self {
        ContextInternal {
            fit_results: Vec::new(),
            sim_results: Vec::new(),
            gpu_enabled: false,
            num_threads: num_cpus::get(),
        }
    }
}

/// Initialize MedLang context
#[no_mangle]
pub extern "C" fn medlang_init() -> *mut MedLangContext {
    let ctx = Box::new(ContextInternal::default());
    Box::into_raw(ctx) as *mut MedLangContext
}

/// Free MedLang context
#[no_mangle]
pub extern "C" fn medlang_free(ctx: *mut MedLangContext) {
    if !ctx.is_null() {
        unsafe {
            drop(Box::from_raw(ctx as *mut ContextInternal));
        }
    }
}

/// Configure context
#[no_mangle]
pub extern "C" fn medlang_configure(
    ctx: *mut MedLangContext,
    key: *const c_char,
    value: *const c_char,
) -> c_int {
    if ctx.is_null() || key.is_null() {
        return MedLangError::InvalidArgument as c_int;
    }
    
    let ctx = unsafe { &mut *(ctx as *mut ContextInternal) };
    let key_str = unsafe { CStr::from_ptr(key).to_string_lossy() };
    
    match key_str.as_ref() {
        "gpu" => {
            if !value.is_null() {
                let val = unsafe { CStr::from_ptr(value).to_string_lossy() };
                ctx.gpu_enabled = val == "true" || val == "1";
            }
        }
        "threads" => {
            if !value.is_null() {
                let val = unsafe { CStr::from_ptr(value).to_string_lossy() };
                if let Ok(n) = val.parse::<usize>() {
                    ctx.num_threads = n;
                }
            }
        }
        _ => {
            set_error(&format!("Unknown configuration key: {}", key_str));
            return MedLangError::InvalidArgument as c_int;
        }
    }
    
    MedLangError::Success as c_int
}

// ============================================================================
// PARSING AND COMPILATION
// ============================================================================

/// Register MedLang source code
#[no_mangle]
pub extern "C" fn medlang_register(
    ctx: *mut MedLangContext,
    source: *const c_char,
    source_len: c_int,
) -> c_int {
    if ctx.is_null() || source.is_null() {
        return MedLangError::InvalidArgument as c_int;
    }
    
    let _ctx = unsafe { &mut *(ctx as *mut ContextInternal) };
    
    let source_slice = unsafe {
        slice::from_raw_parts(source as *const u8, source_len as usize)
    };
    
    let source_str = match std::str::from_utf8(source_slice) {
        Ok(s) => s,
        Err(e) => {
            set_error(&format!("Invalid UTF-8 in source: {}", e));
            return MedLangError::ParseError as c_int;
        }
    };
    
    // Parse source
    // match medlang_core::parse(source_str) {
    //     Ok(program) => {
    //         ctx.programs.push(program);
    //     }
    //     Err(e) => {
    //         set_error(&format!("Parse error: {}", e));
    //         return MedLangError::ParseError as c_int;
    //     }
    // }
    
    // Placeholder implementation
    eprintln!("[MedLang] Registered {} bytes of source", source_str.len());
    
    MedLangError::Success as c_int
}

/// Compile registered models
#[no_mangle]
pub extern "C" fn medlang_compile(ctx: *mut MedLangContext) -> c_int {
    if ctx.is_null() {
        return MedLangError::InvalidArgument as c_int;
    }
    
    let _ctx = unsafe { &mut *(ctx as *mut ContextInternal) };
    
    // Compile all programs to NIR
    // for program in &ctx.programs {
    //     match medlang_core::compile(program) {
    //         Ok(model) => {
    //             ctx.models.insert(model.name.clone(), model);
    //         }
    //         Err(e) => {
    //             set_error(&format!("Compile error: {}", e));
    //             return MedLangError::CompileError as c_int;
    //         }
    //     }
    // }
    
    MedLangError::Success as c_int
}

// ============================================================================
// MODEL FITTING
// ============================================================================

/// Execute all fit specifications
#[no_mangle]
pub extern "C" fn medlang_execute_fits(ctx: *mut MedLangContext) -> c_int {
    if ctx.is_null() {
        return MedLangError::InvalidArgument as c_int;
    }
    
    let ctx = unsafe { &mut *(ctx as *mut ContextInternal) };
    
    // Execute fits
    // for fit_spec in &ctx.programs.iter().flat_map(|p| &p.fit_specs) {
    //     let result = execute_fit(&ctx.models[&fit_spec.model_name], fit_spec);
    //     ctx.fit_results.push(result);
    // }
    
    // Placeholder: create dummy result
    let mut result = FitResultData {
        model_name: [0; 64],
        n_params: 2,
        params: [ParamEstimate::default(); 30],
        cl: ParamEstimate::default(),
        v: ParamEstimate::default(),
        vc: ParamEstimate::default(),
        vp: ParamEstimate::default(),
        q: ParamEstimate::default(),
        ka: ParamEstimate::default(),
        f_bio: ParamEstimate::default(),
        objective: 125.3,
        aic: 133.3,
        bic: 138.5,
        r_squared: 0.95,
        rmse: 0.12,
        n_iterations: 42,
        n_function_evals: 168,
        converged: 1,
        convergence_code: 0,
        n_observations: 96,
        n_subjects: 12,
    };
    
    // Set CL
    result.cl.value = 10.0;
    result.cl.se = 1.2;
    result.cl.cv_percent = 12.0;
    result.cl.omega = 0.30;
    
    // Set V
    result.v.value = 50.0;
    result.v.se = 5.0;
    result.v.cv_percent = 10.0;
    result.v.omega = 0.25;
    
    ctx.fit_results.push(result);
    
    ctx.fit_results.len() as c_int
}

/// Get fit result by model name
#[no_mangle]
pub extern "C" fn medlang_get_fit_result(
    ctx: *mut MedLangContext,
    model_name: *const c_char,
    result: *mut FitResultData,
) -> c_int {
    if ctx.is_null() || result.is_null() {
        return MedLangError::InvalidArgument as c_int;
    }
    
    let ctx = unsafe { &*(ctx as *const ContextInternal) };
    
    // If model_name is null, return first result
    if model_name.is_null() {
        if !ctx.fit_results.is_empty() {
            unsafe { *result = ctx.fit_results[0].clone(); }
            return MedLangError::Success as c_int;
        }
        set_error("No fit results available");
        return MedLangError::NotFound as c_int;
    }
    
    let name = unsafe { CStr::from_ptr(model_name).to_string_lossy() };
    
    // Find result
    for r in &ctx.fit_results {
        let r_name = unsafe { CStr::from_ptr(r.model_name.as_ptr() as *const c_char).to_string_lossy() };
        if r_name == name || ctx.fit_results.len() == 1 {
            unsafe { *result = r.clone(); }
            return MedLangError::Success as c_int;
        }
    }
    
    // Return first result if only one
    if !ctx.fit_results.is_empty() {
        unsafe { *result = ctx.fit_results[0].clone(); }
        return MedLangError::Success as c_int;
    }
    
    set_error(&format!("Model not found: {}", name));
    MedLangError::NotFound as c_int
}

// ============================================================================
// SIMULATION
// ============================================================================

/// Execute simulation
#[no_mangle]
pub extern "C" fn medlang_simulate(
    ctx: *mut MedLangContext,
    model_name: *const c_char,
    t_start: c_double,
    t_end: c_double,
    dt: c_double,
    n_subjects: c_int,
    seed: c_int,
) -> c_int {
    if ctx.is_null() {
        return MedLangError::InvalidArgument as c_int;
    }
    
    let ctx = unsafe { &mut *(ctx as *mut ContextInternal) };
    
    // Create simulation result
    let mut result = SimResultData {
        model_name: [0; 64],
        n_times: 0,
        times: [0.0; 1000],
        ipred: [0.0; 1000],
        pred: [0.0; 1000],
        median: [0.0; 1000],
        ci_lower: [0.0; 1000],
        ci_upper: [0.0; 1000],
        n_subjects: n_subjects,
        seed: seed,
    };
    
    // Copy model name if provided
    if !model_name.is_null() {
        let name = unsafe { CStr::from_ptr(model_name) };
        let name_bytes = name.to_bytes();
        let copy_len = std::cmp::min(name_bytes.len(), 63);
        unsafe {
            ptr::copy_nonoverlapping(name_bytes.as_ptr(), result.model_name.as_mut_ptr() as *mut u8, copy_len);
        }
    }
    
    // Generate time grid
    let mut t = t_start;
    let mut idx = 0;
    while t <= t_end && idx < 1000 {
        result.times[idx] = t;
        
        // Placeholder: one-compartment decay
        // C(t) = (Dose/V) * exp(-k*t), k = CL/V
        let dose = 100.0;
        let v = 50.0;
        let k = 10.0 / 50.0;  // CL/V
        let c = (dose / v) * (-k * t).exp();
        
        result.ipred[idx] = c;
        result.pred[idx] = c;
        result.median[idx] = c;
        result.ci_lower[idx] = c * 0.8;
        result.ci_upper[idx] = c * 1.2;
        
        idx += 1;
        t += dt;
    }
    result.n_times = idx as c_int;
    
    ctx.sim_results.push(result);
    
    MedLangError::Success as c_int
}

/// Get simulation result
#[no_mangle]
pub extern "C" fn medlang_get_sim_result(
    ctx: *mut MedLangContext,
    model_name: *const c_char,
    result: *mut SimResultData,
) -> c_int {
    if ctx.is_null() || result.is_null() {
        return MedLangError::InvalidArgument as c_int;
    }
    
    let ctx = unsafe { &*(ctx as *const ContextInternal) };
    
    if !ctx.sim_results.is_empty() {
        unsafe { *result = ctx.sim_results[ctx.sim_results.len() - 1].clone(); }
        return MedLangError::Success as c_int;
    }
    
    MedLangError::NotFound as c_int
}

// ============================================================================
// ODE SOLVER (Direct access)
// ============================================================================

/// ODE derivative function type
pub type ODEDerivFn = extern "C" fn(
    t: c_double,
    y: *const c_double,
    dydt: *mut c_double,
    n: c_int,
    params: *const c_double,
    n_params: c_int,
);

/// Solve ODE system using RK4
#[no_mangle]
pub extern "C" fn medlang_solve_ode(
    deriv_fn: ODEDerivFn,
    y0: *const c_double,
    n_states: c_int,
    params: *const c_double,
    n_params: c_int,
    t_start: c_double,
    t_end: c_double,
    dt: c_double,
    t_out: *mut c_double,
    y_out: *mut c_double,
    max_steps: c_int,
) -> c_int {
    if y0.is_null() || t_out.is_null() || y_out.is_null() {
        return -1;
    }
    
    let n = n_states as usize;
    
    // Initialize state
    let mut y: Vec<f64> = vec![0.0; n];
    unsafe {
        ptr::copy_nonoverlapping(y0, y.as_mut_ptr(), n);
    }
    
    let mut dydt = vec![0.0; n];
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut y_temp = vec![0.0; n];
    
    let mut t = t_start;
    let mut step = 0;
    
    while t < t_end && step < max_steps as usize {
        // Record output
        unsafe {
            *t_out.add(step) = t;
            ptr::copy_nonoverlapping(y.as_ptr(), y_out.add(step * n), n);
        }
        
        // RK4 step
        deriv_fn(t, y.as_ptr(), k1.as_mut_ptr(), n_states, params, n_params);
        
        for i in 0..n {
            y_temp[i] = y[i] + 0.5 * dt * k1[i];
        }
        deriv_fn(t + 0.5 * dt, y_temp.as_ptr(), k2.as_mut_ptr(), n_states, params, n_params);
        
        for i in 0..n {
            y_temp[i] = y[i] + 0.5 * dt * k2[i];
        }
        deriv_fn(t + 0.5 * dt, y_temp.as_ptr(), k3.as_mut_ptr(), n_states, params, n_params);
        
        for i in 0..n {
            y_temp[i] = y[i] + dt * k3[i];
        }
        deriv_fn(t + dt, y_temp.as_ptr(), k4.as_mut_ptr(), n_states, params, n_params);
        
        for i in 0..n {
            y[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        
        t += dt;
        step += 1;
    }
    
    step as c_int
}

// ============================================================================
// QUANTUM CHEMISTRY
// ============================================================================

/// Compute Hartree-Fock energy
#[no_mangle]
pub extern "C" fn medlang_quantum_hf(
    ctx: *mut MedLangContext,
    xyz_path: *const c_char,
    basis_set: *const c_char,
    energy: *mut c_double,
) -> c_int {
    if ctx.is_null() || xyz_path.is_null() || basis_set.is_null() || energy.is_null() {
        return MedLangError::InvalidArgument as c_int;
    }
    
    let _xyz = unsafe { CStr::from_ptr(xyz_path).to_string_lossy() };
    let _basis_set = unsafe { CStr::from_ptr(basis_set).to_string_lossy() };
    
    // Placeholder: would call QM backend
    // let qm = medlang_runtime::QMBackend::new();
    // *energy = qm.hartree_fock(&xyz, &basis)?;
    
    unsafe { *energy = -76.0267; }  // Water HF/6-31G* energy
    
    MedLangError::Success as c_int
}

/// Compute DFT energy
#[no_mangle]
pub extern "C" fn medlang_quantum_dft(
    ctx: *mut MedLangContext,
    xyz_path: *const c_char,
    functional: *const c_char,
    basis_set: *const c_char,
    energy: *mut c_double,
) -> c_int {
    if energy.is_null() {
        return MedLangError::InvalidArgument as c_int;
    }
    
    // Placeholder
    unsafe { *energy = -76.4178; }  // Water B3LYP/6-31G* energy
    
    MedLangError::Success as c_int
}

// ============================================================================
// FRACTAL ANALYSIS
// ============================================================================

/// Compute Higuchi Fractal Dimension (CPU)
#[no_mangle]
pub extern "C" fn medlang_fractal_higuchi(
    signal: *const c_double,
    n: c_int,
    k_max: c_int,
    hfd: *mut c_double,
    r_squared: *mut c_double,
) -> c_int {
    if signal.is_null() || hfd.is_null() {
        return MedLangError::InvalidArgument as c_int;
    }
    
    let signal_slice = unsafe { slice::from_raw_parts(signal, n as usize) };
    let k_max = k_max as usize;
    
    // Compute Higuchi FD
    let mut ln_k = Vec::with_capacity(k_max);
    let mut ln_l = Vec::with_capacity(k_max);
    
    for k in 1..=k_max {
        let mut l_k = 0.0;
        
        for m in 1..=k {
            let mut l_mk = 0.0;
            let n_m = ((n as usize - m) / k) as f64;
            
            for i in 1..=(((n as usize - m) / k) as usize) {
                let idx1 = m + i * k - 1;
                let idx2 = m + (i - 1) * k - 1;
                if idx1 < signal_slice.len() && idx2 < signal_slice.len() {
                    l_mk += (signal_slice[idx1] - signal_slice[idx2]).abs();
                }
            }
            
            if n_m > 0.0 {
                l_mk *= ((n as usize - 1) as f64) / (k as f64 * n_m * k as f64);
            }
            l_k += l_mk;
        }
        
        l_k /= k as f64;
        
        if l_k > 0.0 {
            ln_k.push((k as f64).ln());
            ln_l.push(l_k.ln());
        }
    }
    
    // Linear regression: ln(L) = -D * ln(k) + c
    if ln_k.len() >= 2 {
        let n = ln_k.len() as f64;
        let sum_x: f64 = ln_k.iter().sum();
        let sum_y: f64 = ln_l.iter().sum();
        let sum_xy: f64 = ln_k.iter().zip(ln_l.iter()).map(|(x, y)| x * y).sum();
        let sum_xx: f64 = ln_k.iter().map(|x| x * x).sum();
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        
        unsafe { *hfd = -slope; }
        
        // R²
        if !r_squared.is_null() {
            let mean_y = sum_y / n;
            let ss_tot: f64 = ln_l.iter().map(|y| (y - mean_y).powi(2)).sum();
            let intercept = (sum_y - slope * sum_x) / n;
            let ss_res: f64 = ln_k.iter().zip(ln_l.iter())
                .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
                .sum();
            
            if ss_tot > 0.0 {
                unsafe { *r_squared = 1.0 - ss_res / ss_tot; }
            }
        }
    }
    
    MedLangError::Success as c_int
}

/// Compute Higuchi Fractal Dimension (GPU accelerated)
#[no_mangle]
pub extern "C" fn medlang_fractal_higuchi_gpu(
    signal: *const c_double,
    n: c_int,
    k_max: c_int,
    hfd: *mut c_double,
) -> c_int {
    // For now, fall back to CPU
    // In full implementation: use rust-cuda or wgpu
    
    let mut r_sq: f64 = 0.0;
    medlang_fractal_higuchi(signal, n, k_max, hfd, &mut r_sq)
}

/// Compute Detrended Fluctuation Analysis
#[no_mangle]
pub extern "C" fn medlang_fractal_dfa(
    _signal: *const c_double,
    _n: c_int,
    _order: c_int,
    alpha: *mut c_double,
) -> c_int {
    if _signal.is_null() || alpha.is_null() {
        return MedLangError::InvalidArgument as c_int;
    }
    
    // Placeholder: DFA implementation
    unsafe { *alpha = 0.65; }  // Typical long-range correlation
    
    MedLangError::Success as c_int
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Get MedLang version
#[no_mangle]
pub extern "C" fn medlang_version() -> *const c_char {
    static VERSION: &[u8] = b"0.5.0\0";
    VERSION.as_ptr() as *const c_char
}

/// Check if GPU is available
#[no_mangle]
pub extern "C" fn medlang_gpu_available() -> c_int {
    // Check for CUDA/OpenCL
    // For now, return 0 (not available)
    0
}

/// Get number of CPU cores
#[no_mangle]
pub extern "C" fn medlang_cpu_cores() -> c_int {
    num_cpus::get() as c_int
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

/// Allocate array
#[no_mangle]
pub extern "C" fn medlang_alloc_f64(n: c_int) -> *mut c_double {
    let mut v = vec![0.0f64; n as usize];
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

/// Free array
#[no_mangle]
pub extern "C" fn medlang_free_f64(ptr: *mut c_double, n: c_int) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, n as usize, n as usize);
        }
    }
}

