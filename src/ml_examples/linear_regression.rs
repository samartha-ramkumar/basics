// Linear Regression Implementation
// y = wx + b (simple linear model)

use ndarray::Array1;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub fn demo() {
    println!("📈 Linear Regression Demo");
    println!("{}", "=".repeat(60));
    
    // Generate synthetic data: y = 3x + 7 + noise
    let (x_train, y_train) = generate_data(100);
    
    println!("   Generated {} training samples", x_train.len());
    println!("   True parameters: w=3.0, b=7.0");
    
    // Train model
    let mut model = LinearRegression::new();
    model.train(&x_train, &y_train, 1000, 0.01);
    
    // Test predictions
    let x_test = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let predictions = model.predict(&x_test);
    
    println!("\n   Test predictions:");
    for (x, pred) in x_test.iter().zip(predictions.iter()) {
        let actual = 3.0 * x + 7.0;
        println!("   x={:.1} -> pred={:.2}, actual={:.2}, error={:.2}", 
                 x, pred, actual, (pred - actual).abs());
    }
}

struct LinearRegression {
    weight: f64,
    bias: f64,
}

impl LinearRegression {
    fn new() -> Self {
        LinearRegression {
            weight: 0.0,
            bias: 0.0,
        }
    }
    
    /// Train using gradient descent
    fn train(&mut self, x: &Array1<f64>, y: &Array1<f64>, epochs: usize, learning_rate: f64) {
        let n = x.len() as f64;
        
        println!("\n   Training for {} epochs...", epochs);
        
        for epoch in 0..epochs {
            // Forward pass: predictions
            let predictions = x.mapv(|xi| self.weight * xi + self.bias);
            
            // Compute loss (MSE)
            let errors = &predictions - y;
            let loss = errors.mapv(|e| e.powi(2)).sum() / n;
            
            // Compute gradients
            let grad_w = (2.0 / n) * (&errors * x).sum();
            let grad_b = (2.0 / n) * errors.sum();
            
            // Update parameters
            self.weight -= learning_rate * grad_w;
            self.bias -= learning_rate * grad_b;
            
            // Print progress
            if epoch % 100 == 0 || epoch == epochs - 1 {
                println!("   Epoch {}: loss={:.4}, w={:.4}, b={:.4}", 
                         epoch, loss, self.weight, self.bias);
            }
        }
    }
    
    /// Make predictions
    fn predict(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|xi| self.weight * xi + self.bias)
    }
}

/// Generate synthetic data: y = 3x + 7 + noise
fn generate_data(n: usize) -> (Array1<f64>, Array1<f64>) {
    let x = Array1::random(n, Uniform::new(0.0, 10.0));
    let noise = Array1::random(n, Uniform::new(-1.0, 1.0));
    let y = x.mapv(|xi| 3.0 * xi + 7.0) + noise;
    (x, y)
}

// Gradient Descent Algorithm:
// 
// 1. Initialize parameters (w, b) randomly or to zero
// 2. For each epoch:
//    a. Forward pass: compute predictions
//    b. Compute loss: MSE = mean((y_pred - y_true)²)
//    c. Compute gradients: ∂L/∂w and ∂L/∂b
//    d. Update parameters: w -= lr * ∂L/∂w, b -= lr * ∂L/∂b
// 3. Repeat until convergence
// 
// Key Concepts:
// - **Loss function**: Measures prediction error
// - **Gradient**: Direction of steepest ascent
// - **Learning rate**: Step size for updates
// - **Epoch**: One pass through entire dataset
