// Matrix Operations for ML
// Foundation for deep learning computations

use ndarray::{Array1, Array2};

pub fn demo() {
    println!("🔢 Matrix Operations Demo");
    println!("{}", "=".repeat(60));
    
    basic_operations();
    matrix_multiplication();
    broadcasting();
    activation_functions();
}

fn basic_operations() {
    println!("\n1️⃣  Basic Matrix Operations");
    
    // Create matrices
    let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Array2::from_shape_vec((2, 3), vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
    
    println!("   Matrix A:\n{}", a);
    println!("   Matrix B:\n{}", b);
    
    // Element-wise addition
    let sum = &a + &b;
    println!("   A + B:\n{}", sum);
    
    // Element-wise multiplication (Hadamard product)
    let product = &a * &b;
    println!("   A ⊙ B (element-wise):\n{}", product);
    
    // Scalar multiplication
    let scaled = &a * 2.0;
    println!("   A * 2:\n{}", scaled);
    
    // Transpose
    let transposed = a.t();
    println!("   A^T:\n{}", transposed);
}

fn matrix_multiplication() {
    println!("\n2️⃣  Matrix Multiplication");
    
    // Matrix multiplication (dot product)
    let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Array2::from_shape_vec((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    
    println!("   A (2x3):\n{}", a);
    println!("   B (3x2):\n{}", b);
    
    // Matrix multiply: (2x3) × (3x2) = (2x2)
    let result = a.dot(&b);
    println!("   A × B (2x2):\n{}", result);
    
    // Vector-matrix multiplication
    let vec = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let mat = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    
    println!("\n   Vector (3):\n{}", vec);
    println!("   Matrix (3x2):\n{}", mat);
    
    let result = vec.dot(&mat);
    println!("   vec × mat (2):\n{}", result);
}

fn broadcasting() {
    println!("\n3️⃣  Broadcasting");
    
    // Broadcasting: automatic expansion of dimensions
    let mat = Array2::from_shape_vec((3, 4), 
        vec![1.0, 2.0, 3.0, 4.0, 
             5.0, 6.0, 7.0, 8.0,
             9.0, 10.0, 11.0, 12.0]).unwrap();
    
    let bias = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    
    println!("   Matrix (3x4):\n{}", mat);
    println!("   Bias (4):\n{}", bias);
    
    // Add bias to each row (broadcasting)
    let result = &mat + &bias;
    println!("   Matrix + Bias (broadcasted):\n{}", result);
}

fn activation_functions() {
    println!("\n4️⃣  Activation Functions");
    
    let x = Array1::from_vec(vec![-2.0_f64, -1.0, 0.0, 1.0, 2.0]);
    
    println!("   Input: {}", x);
    
    // ReLU: max(0, x)
    let relu = x.mapv(|v: f64| v.max(0.0));
    println!("   ReLU: {}", relu);
    
    // Sigmoid: 1 / (1 + e^-x)
    let sigmoid = x.mapv(|v: f64| 1.0 / (1.0 + (-v).exp()));
    println!("   Sigmoid: {}", sigmoid);
    
    // Tanh: (e^x - e^-x) / (e^x + e^-x)
    let tanh = x.mapv(|v: f64| v.tanh());
    println!("   Tanh: {}", tanh);
    
    // Softmax for classification
    let logits = Array1::from_vec(vec![2.0, 1.0, 0.1]);
    let softmax = softmax(&logits);
    println!("\n   Logits: {}", logits);
    println!("   Softmax: {}", softmax);
    println!("   Sum: {:.6} (should be 1.0)", softmax.sum());
}

// Softmax implementation
fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp = x.mapv(|v| (v - max).exp());
    let sum = exp.sum();
    exp / sum
}

// Key Matrix Operations for ML:
// 
// 1. **Element-wise ops**: Fast, SIMD-friendly
// 2. **Matrix multiplication**: Core of neural networks
// 3. **Broadcasting**: Efficient batch operations
// 4. **Activation functions**: Non-linearity
// 5. **Transpose**: Layer computations
// 
// ndarray provides:
// - NumPy-like API
// - Zero-copy operations when possible
// - BLAS integration for fast linear algebra
// - Memory-efficient views and slices
