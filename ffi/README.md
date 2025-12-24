# MedLang FFI Crate

C ABI exports for external language integration (C, Python, Julia, R, etc.).

## Overview

This crate provides a C-compatible foreign function interface (FFI) for MedLang, allowing the compiler and runtime to be called from other languages.

## Building

```bash
cd ffi
cargo build --release
```

This produces:
- **Linux/macOS**: `target/release/libmedlang.so` or `libmedlang.dylib`
- **Windows**: `target/release/medlang.dll`

## C Header

The C header file (`include/medlang.h`) is **manually maintained** with comprehensive documentation. It contains:

- Complete API documentation with Doxygen-style comments
- Version macros (`MEDLANG_VERSION_MAJOR`, etc.)
- Detailed parameter descriptions
- Usage examples

To regenerate from Rust code (optional, will overwrite manual documentation):

```bash
cbindgen --config cbindgen.toml --output include/medlang.h
```

**Note**: Manual header is preferred for better documentation quality.

## Usage

### C Example

```c
#include "include/medlang.h"
#include <stdio.h>

int main() {
    // Initialize context
    medlang_ctx_t* ctx = medlang_init();
    
    // Configure
    medlang_configure(ctx, "threads", "4");
    
    // Register source code
    const char* source = "model MyModel { ... }";
    medlang_register(ctx, source, strlen(source));
    
    // Compile
    medlang_compile(ctx);
    
    // Execute fits
    medlang_execute_fits(ctx);
    
    // Get results
    FitResultData result;
    medlang_get_fit_result(ctx, NULL, &result);
    
    printf("CL = %.2f ± %.2f\n", result.cl.value, result.cl.se);
    
    // Cleanup
    medlang_free(ctx);
    return 0;
}
```

### Python Example (ctypes)

```python
import ctypes
import os

# Load library
lib = ctypes.CDLL("./target/release/libmedlang.so")

# Define types
lib.medlang_init.restype = ctypes.POINTER(ctypes.c_void_p)
lib.medlang_version.restype = ctypes.c_char_p

# Initialize
ctx = lib.medlang_init()

# Get version
version = lib.medlang_version()
print(f"MedLang version: {version.decode()}")

# Cleanup
lib.medlang_free(ctx)
```

### Julia Example

```julia
using Libdl

# Load library
lib = Libdl.dlopen("./target/release/libmedlang.so")

# Initialize
init_fn = Libdl.dlsym(lib, :medlang_init)
ctx = ccall(init_fn, Ptr{Cvoid}, ())

# Get version
version_fn = Libdl.dlsym(lib, :medlang_version)
version = unsafe_string(ccall(version_fn, Ptr{Cchar}, ()))

println("MedLang version: $version")

# Cleanup
free_fn = Libdl.dlsym(lib, :medlang_free)
ccall(free_fn, Cvoid, (Ptr{Cvoid},), ctx)
```

## API Overview

### Context Management
- `medlang_init()` - Create new context
- `medlang_free(ctx)` - Free context
- `medlang_configure(ctx, key, value)` - Configure context

### Parsing and Compilation
- `medlang_register(ctx, source, len)` - Register source code
- `medlang_compile(ctx)` - Compile registered models

### Model Fitting
- `medlang_execute_fits(ctx)` - Execute all fits
- `medlang_get_fit_result(ctx, model_name, result)` - Get fit results

### Simulation
- `medlang_simulate(ctx, model_name, t_start, t_end, dt, n_subjects, seed)` - Run simulation
- `medlang_get_sim_result(ctx, model_name, result)` - Get simulation results

### ODE Solver
- `medlang_solve_ode(deriv_fn, y0, n_states, params, n_params, t_start, t_end, dt, t_out, y_out, max_steps)` - Solve ODE system

### Quantum Chemistry
- `medlang_quantum_hf(ctx, xyz_path, basis_set, energy)` - Hartree-Fock energy
- `medlang_quantum_dft(ctx, xyz_path, functional, basis_set, energy)` - DFT energy

### Fractal Analysis
- `medlang_fractal_higuchi(signal, n, k_max, hfd, r_squared)` - Higuchi fractal dimension
- `medlang_fractal_dfa(signal, n, order, alpha)` - Detrended fluctuation analysis

### Utilities
- `medlang_version()` - Get version string
- `medlang_gpu_available()` - Check GPU availability
- `medlang_cpu_cores()` - Get CPU core count
- `medlang_get_error(buffer, len)` - Get last error message

## Error Handling

All functions return an integer error code (0 = success). Use `medlang_get_error()` to retrieve error messages.

Error codes:
- `0` - Success
- `1` - ParseError
- `2` - CompileError
- `3` - RuntimeError
- `4` - IOError
- `5` - InvalidArgument
- `6` - OutOfMemory
- `7` - NotFound
- `8` - NotImplemented

## Memory Management

The FFI uses Rust's ownership system internally. Contexts and results are managed by the library. Use `medlang_free()` to release contexts.

For arrays, use:
- `medlang_alloc_f64(n)` - Allocate double array
- `medlang_free_f64(ptr, n)` - Free double array

## Thread Safety

The FFI is **not** thread-safe. Each thread should create its own context using `medlang_init()`.

## Integration with MedLang Core

Currently, the FFI uses placeholder implementations. To integrate with the actual compiler:

1. Uncomment imports in `src/lib.rs`:
   ```rust
   use medlang_core::{parse, compile, CIR, NIR};
   use medlang_runtime::{execute, ODESolver, QMBackend};
   ```

2. Update `Cargo.toml` to include dependencies:
   ```toml
   medlang-core = { path = "../compiler" }
   medlang-runtime = { path = "../runtime" }
   ```

3. Implement actual parsing/compilation in the FFI functions.

## License

MIT OR Apache-2.0 (same as main MedLang project)

