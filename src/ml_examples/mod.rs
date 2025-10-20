// AI/ML Examples Module
// Practical implementations of ML algorithms in Rust

pub mod linear_regression;
pub mod matrix_ops;
pub mod neural_network;

pub fn run_examples() {
    println!("Running AI/ML Examples...\n");
    
    linear_regression::demo();
    println!();
    
    matrix_ops::demo();
    println!();
    
    neural_network::demo();
}
