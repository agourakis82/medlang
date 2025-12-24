# MedLang FFI Integration Guide

## Current Status

The FFI crate is set up and ready, but currently uses placeholder implementations. To integrate with the actual MedLang compiler:

## Step 1: Enable Compiler Dependency

Edit `ffi/Cargo.toml`:

```toml
[dependencies]
medlangc = { path = "../compiler" }  # Uncomment this line
```

## Step 2: Update lib.rs Imports

Edit `ffi/src/lib.rs` and uncomment:

```rust
use medlangc::{parse, compile, /* other exports */};
```

## Step 3: Implement Actual Functions

Replace placeholder implementations in:
- `medlang_register()` - Use `medlangc::parse()`
- `medlang_compile()` - Use `medlangc::compile()`
- `medlang_execute_fits()` - Use actual fitting routines
- `medlang_simulate()` - Use actual ODE solvers

## Step 4: Runtime Integration

When runtime is ready, enable it:

```toml
[dependencies]
medlang-runtime = { path = "../runtime" }
```

Then use runtime functions for:
- ODE solving
- Quantum chemistry
- GPU acceleration
- Fractal analysis

## Building

```bash
cd ffi
cargo build --release
```

This produces:
- `target/release/libmedlang.so` (Linux)
- `target/release/libmedlang.dylib` (macOS)
- `target/release/libmedlang.dll` (Windows)

## Generating C Header

The header is manually maintained in `include/medlang.h`. To regenerate:

```bash
cbindgen --config cbindgen.toml --output include/medlang.h
```

Or it will be auto-generated during build if `cbindgen` is available.

