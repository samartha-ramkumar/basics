# Rust for AI/ML - Detailed Walkthrough

## 📖 Introduction

Welcome to your journey of learning Rust for AI/ML applications! This guide will take you step-by-step through Rust's syntax, concepts, and how they apply to building high-performance, low-latency AI systems.

## Why Rust for AI/ML?

### Performance Advantages
- **No Garbage Collection**: Predictable latency, crucial for real-time inference
- **Zero-Cost Abstractions**: High-level code compiles to optimal machine code
- **Memory Safety**: No segfaults, no data races - guaranteed at compile time
- **Explicit Control**: You decide what goes on stack vs heap
- **SIMD Support**: Automatic vectorization for parallel operations

### Real-World Use Cases
- **Model Inference**: Low-latency serving in production
- **Data Processing**: ETL pipelines with high throughput
- **Edge AI**: Embedded devices with limited resources
- **Training Acceleration**: GPU kernels and custom ops
- **System Integration**: Rust interfaces with C/C++ ML libraries

---

## 🎯 Chapter 1: Rust Fundamentals

### Variables and Mutability

```rust
// Immutable by default (can't change)
let x = 5;

// Mutable variables need 'mut'
let mut y = 10;
y = 15; // OK

// Constants (always immutable, known at compile time)
const MAX_ITERATIONS: u32 = 1000;
```

**Why this matters for ML:**
- Immutability prevents accidental model parameter changes
- Compiler catches bugs before runtime
- Easier to reason about data flow in pipelines

### Data Types

```rust
// Integers: i8, i16, i32, i64, i128 (signed)
//          u8, u16, u32, u64, u128 (unsigned)
let samples: usize = 1000;  // usize for array indices

// Floating point: f32 (single precision), f64 (double precision)
let learning_rate: f64 = 0.001;

// Arrays (stack allocated, fixed size)
let features: [f64; 4] = [1.0, 2.0, 3.0, 4.0];

// Vectors (heap allocated, growable)
let mut dataset: Vec<f64> = Vec::new();
dataset.push(1.5);
```

**ML Application:**
- Use `f32` for GPU computations (faster)
- Use `f64` for numerical stability in training
- Arrays for small, fixed features
- Vectors for dynamic datasets

### Control Flow

```rust
// Pattern matching (exhaustive!)
match prediction {
    pred if pred < 0.3 => ClassLabel::Negative,
    pred if pred > 0.7 => ClassLabel::Positive,
    _ => ClassLabel::Uncertain,
}

// Iterators (zero-cost abstraction)
let sum: f64 = features.iter().sum();
let normalized: Vec<f64> = features
    .iter()
    .map(|x| x / sum)
    .collect();
```

---

## 🔑 Chapter 2: Ownership & Memory Management

This is Rust's **superpower** - memory safety without garbage collection!

### The Three Rules

1. **Each value has an owner**
2. **Only one owner at a time**
3. **When owner goes out of scope, value is dropped**

```rust
let s1 = String::from("hello");
let s2 = s1;  // s1 is MOVED to s2
// println!("{}", s1); // ERROR! s1 no longer valid
```

**Why this matters:**
```rust
// Python equivalent (simplified)
model = load_model()  # Reference, GC will collect later
predictions = model.predict(data)
# When does memory free? Unknown! (GC decides)

// Rust
let model = load_model();  // Owner: model
let predictions = model.predict(&data);  // Borrow model
// Model dropped here - deterministic!
```

### Borrowing

```rust
fn process(data: &[f64]) -> f64 {  // Immutable borrow
    data.iter().sum()
}

fn normalize(data: &mut [f64]) {  // Mutable borrow
    let sum: f64 = data.iter().sum();
    for x in data.iter_mut() {
        *x /= sum;
    }
}
```

**Rules:**
- Multiple immutable borrows: ✅ OK
- One mutable borrow: ✅ OK
- Mutable + immutable: ❌ Compile error (prevents data races!)

### Lifetimes

```rust
// Tells compiler: returned reference is valid as long as both inputs
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

**ML Example:**
```rust
struct Model<'a> {
    config: &'a Config,  // Model references config
}
// Compiler ensures config outlives model!
```

---

## 🏗️ Chapter 3: Data Structures for ML

### Structs (Your ML Models)

```rust
#[derive(Debug, Clone)]
struct LinearModel {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

impl LinearModel {
    fn new(input_dim: usize, lr: f64) -> Self {
        LinearModel {
            weights: vec![0.0; input_dim],
            bias: 0.0,
            learning_rate: lr,
        }
    }
    
    fn predict(&self, input: &[f64]) -> f64 {
        let mut result = self.bias;
        for (w, x) in self.weights.iter().zip(input) {
            result += w * x;
        }
        result
    }
    
    fn update(&mut self, grad_w: &[f64], grad_b: f64) {
        for (w, g) in self.weights.iter_mut().zip(grad_w) {
            *w -= self.learning_rate * g;
        }
        self.bias -= self.learning_rate * grad_b;
    }
}
```

### Enums (Model Architectures)

```rust
enum Layer {
    Dense { units: usize, activation: Activation },
    Conv2D { filters: usize, kernel_size: usize },
    Dropout { rate: f64 },
}

enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
}
```

**Pattern matching on architectures:**
```rust
fn forward(&self, layer: &Layer, input: &Tensor) -> Tensor {
    match layer {
        Layer::Dense { units, activation } => {
            let output = self.dense(input, *units);
            self.activate(output, activation)
        }
        Layer::Dropout { rate } => {
            self.dropout(input, *rate)
        }
        // Compiler ensures all variants handled!
        _ => panic!("Layer not implemented"),
    }
}
```

### Option and Result (No NULL!)

```rust
// Option<T> - might have a value, might not
fn find_best_model(models: &[Model]) -> Option<&Model> {
    models.iter().max_by(|a, b| 
        a.accuracy.partial_cmp(&b.accuracy).unwrap()
    )
}

// Result<T, E> - operation might fail
fn load_dataset(path: &str) -> Result<Dataset, DataError> {
    let file = File::open(path)?;  // ? propagates errors
    parse_csv(file)
}

// Usage
match load_dataset("data.csv") {
    Ok(data) => train_model(&data),
    Err(e) => println!("Error: {}", e),
}
```

---

## ⚡ Chapter 4: Performance Optimization

### Stack vs Heap

```rust
// Stack (fast, limited size)
let small_array: [f32; 100] = [0.0; 100];  // ~400 bytes

// Heap (flexible, slower)
let large_vector: Vec<f32> = vec![0.0; 1_000_000];

// Box for large objects
let huge_matrix = Box::new([[0.0; 1000]; 1000]);  // 8MB on heap
```

**Guideline:**
- Features: Stack if < 4KB, otherwise heap
- Models: Always heap (can be MBs/GBs)
- Temporary buffers: Consider arena allocation

### Zero-Cost Abstractions

```rust
// High-level iterator chain
let result: Vec<f64> = data
    .iter()
    .filter(|&&x| x > 0.0)
    .map(|&x| x.sqrt())
    .collect();

// Compiles to same code as manual loop!
let mut result = Vec::new();
for &x in &data {
    if x > 0.0 {
        result.push(x.sqrt());
    }
}
```

**Compiler optimizations:**
- Iterator fusion (single pass)
- Bounds check elimination
- SIMD auto-vectorization
- Inline small functions

### Memory Layout for SIMD

```rust
// BAD: Array of Structs (cache unfriendly)
struct Point { x: f32, y: f32, z: f32 }
let aos: Vec<Point> = vec![/*...*/];

// GOOD: Struct of Arrays (SIMD friendly)
struct Points {
    x: Vec<f32>,  // All x values contiguous
    y: Vec<f32>,  // All y values contiguous
    z: Vec<f32>,
}

// Process all X coordinates at once (vectorized!)
let sum_x: f32 = points.x.iter().sum();
```

### Profiling

```bash
# Install cargo-flamegraph
cargo install flamegraph

# Profile your code
cargo flamegraph --bin your_app

# Opens SVG showing hotspots
```

---

## 🚀 Chapter 5: Concurrency for ML

### Data Parallelism (Easy with Rayon)

```rust
use rayon::prelude::*;

// Sequential
let results: Vec<_> = data.iter()
    .map(|sample| expensive_computation(sample))
    .collect();

// Parallel (just add par_)
let results: Vec<_> = data.par_iter()
    .map(|sample| expensive_computation(sample))
    .collect();
```

**Batch processing:**
```rust
let predictions: Vec<f64> = test_data
    .par_chunks(batch_size)
    .flat_map(|batch| model.predict_batch(batch))
    .collect();
```

### Safe Shared State

```rust
use std::sync::{Arc, Mutex};

// Share model across threads
let model = Arc::new(Mutex::new(Model::new()));

let handles: Vec<_> = (0..4).map(|i| {
    let model = Arc::clone(&model);
    thread::spawn(move || {
        let mut m = model.lock().unwrap();
        m.train_step(batch_i);
    })
}).collect();

for h in handles {
    h.join().unwrap();
}
```

**No data races!** Mutex ensures exclusive access, Arc manages lifetime.

### Async I/O (Data Loading)

```rust
use tokio::fs::File;
use tokio::io::AsyncReadExt;

async fn load_batch(path: &str) -> Result<Vec<f64>, Error> {
    let mut file = File::open(path).await?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).await?;
    parse_features(&buffer)
}

// Load multiple batches concurrently
let batches = tokio::join!(
    load_batch("batch1.bin"),
    load_batch("batch2.bin"),
    load_batch("batch3.bin"),
);
```

---

## 🧠 Chapter 6: ML Implementations

### Linear Regression

**Algorithm:**
1. Initialize: w = 0, b = 0
2. For each epoch:
   - Forward: ŷ = wx + b
   - Loss: L = mean((ŷ - y)²)
   - Gradients: ∂L/∂w, ∂L/∂b
   - Update: w -= lr·∂L/∂w, b -= lr·∂L/∂b

**Rust implementation highlights:**
```rust
// Efficient using ndarray
let predictions = x.mapv(|xi| self.weight * xi + self.bias);
let errors = &predictions - y;
let grad_w = (2.0 / n) * (&errors * x).sum();

// Update in-place (no allocation)
self.weight -= learning_rate * grad_w;
```

### Neural Network

**Architecture: 2 → 4 → 1 (XOR problem)**

```rust
// Forward pass
z1 = x · W1 + b1
a1 = ReLU(z1)
z2 = a1 · W2 + b2
a2 = sigmoid(z2)

// Backward pass (chain rule)
δ2 = (a2 - y) * sigmoid'(a2)
δ1 = (W2^T · δ2) ⊙ ReLU'(a1)

// Update weights
W2 -= lr * (a1^T · δ2)
W1 -= lr * (x^T · δ1)
```

**Key optimizations:**
- Pre-allocate matrices
- In-place operations where possible
- BLAS for matrix multiplication
- Batch processing

---

## 📊 Using ndarray (NumPy for Rust)

```rust
use ndarray::{Array1, Array2, s!};

// Create arrays
let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
let b = Array2::zeros((3, 4));

// Operations (similar to NumPy)
let sum = &a + &b.row(0);
let prod = a.dot(&b);

// Slicing
let slice = arr.slice(s![1..3, ..]);

// Broadcasting
let matrix = Array2::ones((5, 3));
let bias = Array1::from_vec(vec![1.0, 2.0, 3.0]);
let result = &matrix + &bias;  // Broadcasts bias to each row

// Efficient mapping
let relu = matrix.mapv(|x| x.max(0.0));
```

---

## 🎓 Best Practices for Low-Latency AI

### 1. Pre-allocate Buffers

```rust
// BAD
let mut results = Vec::new();
for _ in 0..n {
    results.push(compute());  // Reallocates!
}

// GOOD
let mut results = Vec::with_capacity(n);
for _ in 0..n {
    results.push(compute());  // No reallocation
}
```

### 2. Avoid Clones

```rust
// BAD
fn process(data: Vec<f64>) -> Vec<f64> {  // Takes ownership
    data.iter().map(|x| x * 2.0).collect()
}

// GOOD
fn process(data: &[f64]) -> Vec<f64> {  // Borrows
    data.iter().map(|x| x * 2.0).collect()
}
```

### 3. Use Slices

```rust
fn predict(&self, input: &[f64]) -> f64 {  // Works with any contiguous data
    // Can pass &Vec, &Array, or &[f64; N]
}
```

### 4. Release Mode

```bash
# Development (debug symbols, no optimization)
cargo run

# Production (full optimization)
cargo run --release  # 10-100x faster!
```

### 5. Profile-Guided Optimization

```toml
[profile.release]
opt-level = 3
lto = true           # Link-time optimization
codegen-units = 1    # Better optimization
```

---

## 🛠️ Essential ML Crates

### Core Numeric
- `ndarray` - N-dimensional arrays
- `nalgebra` - Linear algebra
- `num-traits` - Numeric traits

### Machine Learning
- `smartcore` - ML algorithms
- `linfa` - Comprehensive ML toolkit
- `burn` - Deep learning framework
- `candle` - Hugging Face ML framework

### Performance
- `rayon` - Data parallelism
- `packed_simd` - SIMD operations
- `mimalloc` - Fast allocator

### Data Processing
- `polars` - Fast DataFrame library
- `csv` - CSV parsing
- `serde` - Serialization

---

## 🎯 Next Steps

1. **Build the project:**
   ```bash
   cd basics
   cargo build --release
   ```

2. **Run tutorials:**
   ```bash
   cargo run --release
   ```

3. **Explore examples:**
   ```bash
   cargo run --example quick_start
   cargo run --example tensor_operations
   ```

4. **Experiment:**
   - Modify hyperparameters
   - Add new layers
   - Implement new algorithms
   - Profile and optimize

5. **Go deeper:**
   - Read "The Rust Book"
   - Explore burn/candle frameworks
   - Build a real project
   - Contribute to ML libraries

---

## 📚 Additional Resources

- [The Rust Programming Language](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Are We Learning Yet?](http://www.arewelearningyet.com/)
- [ndarray documentation](https://docs.rs/ndarray/)
- [Burn Book](https://burn-rs.github.io/)

Happy coding! 🦀
