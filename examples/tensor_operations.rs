// Advanced Tensor Operations
// Common operations for deep learning

use ndarray::{Array1, Array2, Array3, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn main() {
    println!("🔥 Tensor Operations for Deep Learning");
    println!("{}", "=".repeat(60));
    
    batch_operations();
    convolution_demo();
    pooling_demo();
    normalization();
}

fn batch_operations() {
    println!("\n1️⃣  Batch Operations");
    
    // Batch of samples (batch_size, features)
    let batch = Array2::from_shape_vec(
        (4, 3),  // 4 samples, 3 features each
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ]
    ).unwrap();
    
    println!("   Batch (4 samples × 3 features):\n{}", batch);
    
    // Batch normalization: normalize each feature across batch
    let mean = batch.mean_axis(Axis(0)).unwrap();
    let std = batch.std_axis(Axis(0), 0.0);
    
    println!("   Mean per feature: {}", mean);
    println!("   Std per feature: {}", std);
    
    // Normalize: (x - mean) / std
    let normalized = (&batch - &mean) / &std;
    println!("   Normalized batch:\n{}", normalized);
}

fn convolution_demo() {
    println!("\n2️⃣  1D Convolution (simplified)");
    
    // Input signal (e.g., time series)
    let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    
    // Kernel (filter)
    let kernel = Array1::from_vec(vec![0.25, 0.5, 0.25]);
    
    println!("   Signal: {}", signal);
    println!("   Kernel: {}", kernel);
    
    // Apply convolution
    let output = convolve_1d(&signal, &kernel);
    println!("   Convolution output: {}", output);
}

fn convolve_1d(signal: &Array1<f64>, kernel: &Array1<f64>) -> Array1<f64> {
    let output_size = signal.len() - kernel.len() + 1;
    let mut output = Array1::zeros(output_size);
    
    for i in 0..output_size {
        let mut sum = 0.0;
        for j in 0..kernel.len() {
            sum += signal[i + j] * kernel[j];
        }
        output[i] = sum;
    }
    
    output
}

fn pooling_demo() {
    println!("\n3️⃣  Pooling Operations");
    
    // 2D feature map
    let feature_map = Array2::from_shape_vec(
        (4, 4),
        vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]
    ).unwrap();
    
    println!("   Feature map (4×4):\n{}", feature_map);
    
    // Max pooling (2×2)
    let pooled = max_pool_2x2(&feature_map);
    println!("   Max pooling (2×2):\n{}", pooled);
    
    // Average pooling (2×2)
    let avg_pooled = avg_pool_2x2(&feature_map);
    println!("   Average pooling (2×2):\n{}", avg_pooled);
}

fn max_pool_2x2(input: &Array2<f64>) -> Array2<f64> {
    let (h, w) = input.dim();
    let mut output = Array2::zeros((h / 2, w / 2));
    
    for i in 0..(h / 2) {
        for j in 0..(w / 2) {
            let max_val = input[[i * 2, j * 2]]
                .max(input[[i * 2, j * 2 + 1]])
                .max(input[[i * 2 + 1, j * 2]])
                .max(input[[i * 2 + 1, j * 2 + 1]]);
            output[[i, j]] = max_val;
        }
    }
    
    output
}

fn avg_pool_2x2(input: &Array2<f64>) -> Array2<f64> {
    let (h, w) = input.dim();
    let mut output = Array2::zeros((h / 2, w / 2));
    
    for i in 0..(h / 2) {
        for j in 0..(w / 2) {
            let avg = (input[[i * 2, j * 2]] 
                     + input[[i * 2, j * 2 + 1]]
                     + input[[i * 2 + 1, j * 2]]
                     + input[[i * 2 + 1, j * 2 + 1]]) / 4.0;
            output[[i, j]] = avg;
        }
    }
    
    output
}

fn normalization() {
    println!("\n4️⃣  Normalization Techniques");
    
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    println!("   Original data: {}", data);
    
    // Min-Max normalization: (x - min) / (max - min)
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let minmax = data.mapv(|x| (x - min) / (max - min));
    println!("   Min-Max (0-1): {}", minmax);
    
    // Z-score normalization: (x - mean) / std
    let mean = data.mean().unwrap();
    let std = data.std(0.0);
    let zscore = data.mapv(|x| (x - mean) / std);
    println!("   Z-score: {}", zscore);
    
    // L2 normalization: x / ||x||
    let norm = data.mapv(|x| x * x).sum().sqrt();
    let l2_norm = data.mapv(|x| x / norm);
    println!("   L2 normalized: {}", l2_norm);
    println!("   L2 norm check: {:.6}", l2_norm.mapv(|x| x * x).sum().sqrt());
}

// Key Tensor Operations for Deep Learning:
// 
// 1. **Batch Processing**: Process multiple samples simultaneously
// 2. **Convolution**: Extract spatial features (CNNs)
// 3. **Pooling**: Reduce spatial dimensions
// 4. **Normalization**: Stabilize training
// 
// Real-world frameworks provide:
// - Automatic differentiation (autograd)
// - GPU acceleration
// - Optimized implementations
// - Dynamic computation graphs
// 
// Popular Rust ML frameworks:
// - burn: Modern deep learning framework
// - candle: Hugging Face's ML framework
// - tch-rs: PyTorch bindings for Rust
