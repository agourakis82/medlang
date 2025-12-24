// Build script to generate C header using cbindgen

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_file = PathBuf::from(&crate_dir)
        .join("include")
        .join("medlang.h");

    // Only generate header if cbindgen is available
    // This allows building without cbindgen if header is pre-generated
    if let Ok(_) = std::process::Command::new("cbindgen")
        .arg("--config")
        .arg("cbindgen.toml")
        .arg("--crate")
        .arg(&crate_dir)
        .arg("--output")
        .arg(&output_file)
        .output()
    {
        println!("cargo:rerun-if-changed=cbindgen.toml");
        println!("cargo:rerun-if-changed=src/lib.rs");
    } else {
        println!("cargo:warning=cbindgen not found, skipping header generation");
        println!("cargo:warning=Run 'cargo install cbindgen' to generate C header");
    }
}

