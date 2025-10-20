// Main entry point for the Rust AI/ML tutorial
// This file orchestrates all the learning modules

// Module declarations
mod fundamentals;
mod ownership;
mod structs_enums;
mod performance;
mod concurrency;
mod ml_examples;

fn main() {
    println!("🦀 Welcome to Rust for AI/ML!");
    println!("{}", "=".repeat(60));
    println!();

    // Chapter 1: Fundamentals
    println!("📖 Chapter 1: Rust Fundamentals");
    println!("{}", "-".repeat(60));
    fundamentals::run_examples();
    println!();

    // Chapter 2: Ownership & Borrowing
    println!("📖 Chapter 2: Ownership & Memory Management");
    println!("{}", "-".repeat(60));
    ownership::run_examples();
    println!();

    // Chapter 3: Structs & Enums
    println!("📖 Chapter 3: Data Structures");
    println!("{}", "-".repeat(60));
    structs_enums::run_examples();
    println!();

    // Chapter 4: Performance Patterns
    println!("📖 Chapter 4: Performance Optimization");
    println!("{}", "-".repeat(60));
    performance::run_examples();
    println!();

    // Chapter 5: Concurrency
    println!("📖 Chapter 5: Concurrent & Parallel Programming");
    println!("{}", "-".repeat(60));
    concurrency::run_examples();
    println!();

    // Chapter 6: ML Examples
    println!("📖 Chapter 6: AI/ML Applications");
    println!("{}", "-".repeat(60));
    ml_examples::run_examples();
    println!();

    println!("{}", "=".repeat(60));
    println!("✅ Tutorial complete! Try running specific examples:");
    println!("   cargo run --example quick_start");
    println!("   cargo run --example tensor_operations");
    println!();
    println!("🚀 For production performance, use:");
    println!("   cargo run --release");
}
