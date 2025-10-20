// Quick Start Example
// Your first Rust program for ML

use ndarray::{Array1, Array2};

fn main() {
    println!("🚀 Rust for AI/ML - Quick Start");
    println!("{}", "=".repeat(60));
    
    // 1. Basic Vector Operations
    vector_operations();
    
    // 2. Matrix Operations
    matrix_operations();
    
    // 3. Simple Model
    simple_model();
}

fn vector_operations() {
    println!("\n📊 Vector Operations");
    
    // Create vectors
    let v1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let v2 = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
    
    println!("   v1: {}", v1);
    println!("   v2: {}", v2);
    
    // Vector addition
    let sum = &v1 + &v2;
    println!("   v1 + v2: {}", sum);
    
    // Dot product
    let dot = v1.dot(&v2);
    println!("   v1 · v2: {}", dot);
    
    // Norm (magnitude)
    let norm: f64 = v1.mapv(|x| x * x).sum();
    let norm = norm.sqrt();
    println!("   ||v1||: {:.2}", norm);
}

fn matrix_operations() {
    println!("\n🔢 Matrix Operations");
    
    // Create matrix (2x3)
    let matrix = Array2::from_shape_vec(
        (2, 3),
        vec![1.0, 2.0, 3.0, 
             4.0, 5.0, 6.0]
    ).unwrap();
    
    println!("   Matrix:\n{}", matrix);
    
    // Matrix transpose
    let transposed = matrix.t();
    println!("   Transposed:\n{}", transposed);
    
    // Matrix multiplication
    let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Array2::from_shape_vec((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let product = a.dot(&b);
    
    println!("   A × B:\n{}", product);
}

fn simple_model() {
    println!("\n🤖 Simple Linear Model");
    
    // Model: y = w * x + b
    let w = 2.5;
    let b = 1.0;
    
    println!("   Model: y = {:.1}x + {:.1}", w, b);
    
    // Make predictions
    let inputs = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let predictions = inputs.mapv(|x| w * x + b);
    
    println!("\n   Predictions:");
    for (x, y) in inputs.iter().zip(predictions.iter()) {
        println!("   x={:.1} -> y={:.1}", x, y);
    }
    
    // Compute loss (MSE)
    let targets = Array1::from_vec(vec![3.5, 6.0, 8.5, 11.0, 13.5]);
    let errors = &predictions - &targets;
    let mse = errors.mapv(|e| e * e).sum() / inputs.len() as f64;
    
    println!("\n   Mean Squared Error: {:.4}", mse);
}

// Next Steps:
// 
// 1. Run the main tutorial: `cargo run`
// 2. Explore more examples: `cargo run --example tensor_operations`
// 3. Read the documentation: Check README.md
// 4. Experiment with the code: Modify and learn!
// 
// Key Resources:
// - ndarray docs: https://docs.rs/ndarray
// - Rust book: https://doc.rust-lang.org/book/
// - Are We Learning Yet?: http://www.arewelearningyet.com/
