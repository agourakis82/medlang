# MedLang FFI Examples

## C Example

```c
#include "include/medlang.h"
#include <stdio.h>
#include <string.h>

int main() {
    // Initialize context
    medlang_ctx_t* ctx = medlang_init();
    if (!ctx) {
        fprintf(stderr, "Failed to initialize MedLang context\n");
        return 1;
    }
    
    // Configure
    medlang_configure(ctx, "threads", "4");
    
    // Register source code
    const char* source = 
        "model OneComp {\n"
        "    state A_central : DoseMass\n"
        "    param CL : Clearance\n"
        "    param V : Volume\n"
        "    dA_central/dt = -(CL / V) * A_central\n"
        "    obs C_plasma : ConcMass = A_central / V\n"
        "}\n";
    
    int err = medlang_register(ctx, source, strlen(source));
    if (err != MEDLANG_SUCCESS) {
        char error_msg[256];
        medlang_get_error(error_msg, sizeof(error_msg));
        fprintf(stderr, "Registration error: %s\n", error_msg);
        medlang_free(ctx);
        return 1;
    }
    
    // Compile
    err = medlang_compile(ctx);
    if (err != MEDLANG_SUCCESS) {
        char error_msg[256];
        medlang_get_error(error_msg, sizeof(error_msg));
        fprintf(stderr, "Compilation error: %s\n", error_msg);
        medlang_free(ctx);
        return 1;
    }
    
    // Execute fits
    int n_fits = medlang_execute_fits(ctx);
    printf("Completed %d fits\n", n_fits);
    
    // Get results
    FitResultData result;
    err = medlang_get_fit_result(ctx, "OneComp", &result);
    if (err == MEDLANG_SUCCESS) {
        printf("Model: %s\n", result.model_name);
        printf("CL = %.2f ± %.2f L/h (CV: %.1f%%)\n", 
               result.cl.value, result.cl.se, result.cl.cv_percent);
        printf("V  = %.2f ± %.2f L (CV: %.1f%%)\n",
               result.v.value, result.v.se, result.v.cv_percent);
        printf("AIC = %.2f, BIC = %.2f\n", result.aic, result.bic);
        printf("R² = %.3f, RMSE = %.3f\n", result.r_squared, result.rmse);
    }
    
    // Run simulation
    err = medlang_simulate(ctx, "OneComp", 0.0, 24.0, 0.1, 100, 42);
    if (err == MEDLANG_SUCCESS) {
        SimResultData sim;
        medlang_get_sim_result(ctx, "OneComp", &sim);
        printf("Simulation: %d time points, %d subjects\n", 
               sim.n_times, sim.n_subjects);
    }
    
    // Cleanup
    medlang_free(ctx);
    return 0;
}
```

## Python Example (ctypes)

```python
import ctypes
import os

# Load library
if os.name == 'nt':  # Windows
    lib = ctypes.CDLL("./target/release/medlang.dll")
else:  # Linux/macOS
    lib = ctypes.CDLL("./target/release/libmedlang.so")

# Define types
lib.medlang_init.restype = ctypes.POINTER(ctypes.c_void_p)
lib.medlang_version.restype = ctypes.c_char_p
lib.medlang_cpu_cores.restype = ctypes.c_int

# Initialize
ctx = lib.medlang_init()
if not ctx:
    raise RuntimeError("Failed to initialize MedLang")

# Get version
version = lib.medlang_version()
print(f"MedLang version: {version.decode()}")

# Get CPU cores
cores = lib.medlang_cpu_cores()
print(f"CPU cores: {cores}")

# Cleanup
lib.medlang_free(ctx)
```

## Julia Example

```julia
using Libdl

# Load library
lib_path = "./target/release/libmedlang.so"
lib = Libdl.dlopen(lib_path)

# Initialize
init_fn = Libdl.dlsym(lib, :medlang_init)
ctx = ccall(init_fn, Ptr{Cvoid}, ())

if ctx == C_NULL
    error("Failed to initialize MedLang")
end

# Get version
version_fn = Libdl.dlsym(lib, :medlang_version)
version_ptr = ccall(version_fn, Ptr{Cchar}, ())
version = unsafe_string(version_ptr)
println("MedLang version: $version")

# Get CPU cores
cores_fn = Libdl.dlsym(lib, :medlang_cpu_cores)
cores = ccall(cores_fn, Cint, ())
println("CPU cores: $cores")

# Cleanup
free_fn = Libdl.dlsym(lib, :medlang_free)
ccall(free_fn, Cvoid, (Ptr{Cvoid},), ctx)
```

## R Example (via .Call)

```r
# Load library
dyn.load("target/release/libmedlang.so")

# Initialize
.Call("medlang_init")

# Get version
version <- .Call("medlang_version")
cat("MedLang version:", version, "\n")

# Get CPU cores
cores <- .Call("medlang_cpu_cores")
cat("CPU cores:", cores, "\n")
```

## Compiling C Example

```bash
# Compile with library
gcc example.c -o example -L./target/release -lmedlang -lm

# Run (may need to set LD_LIBRARY_PATH on Linux)
export LD_LIBRARY_PATH=./target/release:$LD_LIBRARY_PATH
./example
```

## Building FFI

```bash
cd ffi
cargo build --release

# Output:
# - target/release/libmedlang.so (Linux)
# - target/release/libmedlang.dylib (macOS)
# - target/release/medlang.dll (Windows)
```

