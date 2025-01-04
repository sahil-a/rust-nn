use std::process::Command;

fn main() {
    // Instruct Cargo to re-run this script if the shader file changes.
    println!("cargo:rerun-if-changed=compute-kernel.metal");

    // 1. Compile compute-kernel.metal into compute-kernel.air
    let status = Command::new("xcrun")
        .args([
            "metal",
            "-c",
            "compute-kernel.metal",
            "-o",
            "compute-kernel.air",
        ])
        .status()
        .expect("Failed to run xcrun metal command.");
    if !status.success() {
        panic!("Metal shader compilation (compute-kernel.metal -> compute-kernel.air) failed!");
    }

    // 2. Link compute-kernel.air into compute-kernel.metallib
    let status = Command::new("xcrun")
        .args([
            "metallib",
            "compute-kernel.air",
            "-o",
            "compute-kernel.metallib",
        ])
        .status()
        .expect("Failed to run xcrun metallib command.");
    if !status.success() {
        panic!("Linking (compute-kernel.air -> compute-kernel.metallib) failed!");
    }
}
