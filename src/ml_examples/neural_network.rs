// Simple Neural Network Implementation
// 2-layer network for binary classification

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub fn demo() {
    println!("🧠 Neural Network Demo");
    println!("{}", "=".repeat(60));
    
    // XOR problem (classic non-linearly separable problem)
    let x_train = Array2::from_shape_vec((4, 2), vec![
        0.0, 0.0,  // XOR(0, 0) = 0
        0.0, 1.0,  // XOR(0, 1) = 1
        1.0, 0.0,  // XOR(1, 0) = 1
        1.0, 1.0,  // XOR(1, 1) = 0
    ]).unwrap();
    
    let y_train = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);
    
    println!("   Training XOR problem (non-linearly separable)");
    println!("   Network architecture: 2 -> 4 -> 1");
    
    // Create and train network
    let mut nn = NeuralNetwork::new(2, 4, 1);
    nn.train(&x_train, &y_train, 5000, 0.5);
    
    // Test predictions
    println!("\n   Test Results:");
    for i in 0..4 {
        let input = x_train.row(i);
        let prediction = nn.predict(&input.to_owned());
        let target = y_train[i];
        println!("   {:?} -> pred={:.4}, target={:.0}", 
                 input.to_vec(), prediction[0], target);
    }
}

struct NeuralNetwork {
    // Layer 1: input -> hidden
    w1: Array2<f64>,
    b1: Array1<f64>,
    
    // Layer 2: hidden -> output
    w2: Array2<f64>,
    b2: Array1<f64>,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        // Xavier initialization for better convergence
        let scale1 = (2.0 / input_size as f64).sqrt();
        let scale2 = (2.0 / hidden_size as f64).sqrt();
        
        NeuralNetwork {
            w1: Array2::random((input_size, hidden_size), Uniform::new(-scale1, scale1)),
            b1: Array1::zeros(hidden_size),
            w2: Array2::random((hidden_size, output_size), Uniform::new(-scale2, scale2)),
            b2: Array1::zeros(output_size),
        }
    }
    
    /// Forward pass
    fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        // Layer 1: input -> hidden (with ReLU activation)
        let z1 = x.dot(&self.w1) + &self.b1;
        let a1 = z1.mapv(|v| v.max(0.0)); // ReLU
        
        // Layer 2: hidden -> output (with sigmoid activation)
        let z2 = a1.dot(&self.w2) + &self.b2;
        let a2 = z2.mapv(|v| sigmoid(v));
        
        (a1, z2, a2)
    }
    
    /// Backward pass (backpropagation)
    fn backward(&self, x: &Array1<f64>, y: f64, a1: &Array1<f64>, a2: &Array1<f64>) 
        -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) 
    {
        // Output layer gradient
        let error = a2[0] - y;
        let delta2 = error * sigmoid_derivative(a2[0]);
        
        // Gradients for layer 2
        let grad_w2 = outer_product(a1, &Array1::from_vec(vec![delta2]));
        let grad_b2 = Array1::from_vec(vec![delta2]);
        
        // Hidden layer gradient (backpropagate through ReLU)
        let delta1 = self.w2.column(0).to_owned() * delta2;
        let delta1 = &delta1 * &a1.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }); // ReLU derivative
        
        // Gradients for layer 1
        let grad_w1 = outer_product(x, &delta1);
        let grad_b1 = delta1;
        
        (grad_w1, grad_b1, grad_w2, grad_b2)
    }
    
    /// Train using gradient descent with backpropagation
    fn train(&mut self, x: &Array2<f64>, y: &Array1<f64>, epochs: usize, learning_rate: f64) {
        println!("\n   Training for {} epochs...", epochs);
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            // Process each training example
            for i in 0..x.nrows() {
                let xi = x.row(i).to_owned();
                let yi = y[i];
                
                // Forward pass
                let (a1, _z2, a2) = self.forward(&xi);
                
                // Compute loss (binary cross-entropy)
                let loss = -yi * a2[0].ln() - (1.0 - yi) * (1.0 - a2[0]).ln();
                total_loss += loss;
                
                // Backward pass
                let (grad_w1, grad_b1, grad_w2, grad_b2) = self.backward(&xi, yi, &a1, &a2);
                
                // Update weights
                self.w1 = &self.w1 - &(grad_w1 * learning_rate);
                self.b1 = &self.b1 - &(grad_b1 * learning_rate);
                self.w2 = &self.w2 - &(grad_w2 * learning_rate);
                self.b2 = &self.b2 - &(grad_b2 * learning_rate);
            }
            
            // Print progress
            if epoch % 1000 == 0 || epoch == epochs - 1 {
                let avg_loss = total_loss / x.nrows() as f64;
                println!("   Epoch {}: loss={:.4}", epoch, avg_loss);
            }
        }
    }
    
    /// Predict on new input
    fn predict(&self, x: &Array1<f64>) -> Array1<f64> {
        let (_a1, _z2, a2) = self.forward(x);
        a2
    }
}

// Activation functions
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(y: f64) -> f64 {
    y * (1.0 - y)
}

// Helper: outer product for gradient computation
fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let mut result = Array2::zeros((a.len(), b.len()));
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

// Neural Network Concepts:
// 
// 1. **Forward Pass**: 
//    - Input -> Hidden: z1 = W1·x + b1, a1 = ReLU(z1)
//    - Hidden -> Output: z2 = W2·a1 + b2, a2 = sigmoid(z2)
// 
// 2. **Loss Function**:
//    - Binary cross-entropy: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
// 
// 3. **Backpropagation**:
//    - Compute gradients using chain rule
//    - Propagate errors backward through network
//    - Update weights: W -= lr * ∂L/∂W
// 
// 4. **Activation Functions**:
//    - ReLU: max(0, x) - for hidden layers
//    - Sigmoid: 1/(1+e^-x) - for binary classification
// 
// Key Points:
// - Non-linearity enables learning complex patterns
// - Backpropagation efficiently computes gradients
// - Xavier initialization helps convergence
// - Learning rate controls update step size
