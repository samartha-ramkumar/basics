// Chapter 4: Performance Optimization
// Critical for low-latency AI/ML applications

pub fn run_examples() {
    stack_vs_heap();
    zero_cost_abstractions();
    simd_example();
    memory_layout();
    compiler_optimizations();
}

/// Stack vs Heap Allocation
/// Stack is much faster but limited in size
fn stack_vs_heap() {
    println!("1️⃣  Stack vs Heap Allocation");
    
    // Stack allocation - fast, fixed size
    let arr: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    println!("   Stack array: {:?}", arr);
    
    // Heap allocation - flexible, slower
    let vec: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    println!("   Heap vector: {:?}", vec);
    
    // For ML: Use stack for small, fixed-size data
    // Use heap for large, dynamic data (datasets, models)
    
    // Box for large objects to avoid stack overflow
    let large_matrix = Box::new([[0.0; 100]; 100]);
    println!("   Large matrix allocated on heap: {}x{}", 
             large_matrix.len(), large_matrix[0].len());
}

/// Zero-Cost Abstractions
/// High-level code compiles to same performance as hand-written low-level code
fn zero_cost_abstractions() {
    println!("\n2️⃣  Zero-Cost Abstractions");
    
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    
    // Iterator-based (high-level)
    let sum: i32 = data.iter().map(|x| x * 2).filter(|x| x % 3 == 0).sum();
    println!("   Iterator result: {}", sum);
    
    // Manual loop (low-level)
    let mut manual_sum = 0;
    for &x in &data {
        let doubled = x * 2;
        if doubled % 3 == 0 {
            manual_sum += doubled;
        }
    }
    println!("   Manual loop result: {}", manual_sum);
    println!("   Both compile to same machine code! (zero-cost)");
    
    // Iterator example for ML preprocessing
    let features: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    // Normalize features (mean=0, std=1)
    let mean: f64 = features.iter().sum::<f64>() / features.len() as f64;
    let variance: f64 = features.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / features.len() as f64;
    let std = variance.sqrt();
    
    let normalized: Vec<f64> = features.iter()
        .map(|x| (x - mean) / std)
        .collect();
    
    println!("   Original: {:?}", features);
    println!("   Normalized: {:?}", normalized);
}

/// SIMD (Single Instruction Multiple Data)
/// Process multiple data points in parallel (CPU vectorization)
fn simd_example() {
    println!("\n3️⃣  SIMD Operations");
    
    let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![8.0_f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    
    // Simple element-wise multiplication
    let result: Vec<f32> = a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .collect();
    
    println!("   Element-wise multiply: {:?}", result);
    println!("   Compiler auto-vectorizes this for SIMD!");
    
    // For explicit SIMD, use crates like 'packed_simd' or 'wide'
    // But often the compiler does it automatically with optimizations
}

/// Memory Layout
/// Understanding data layout for cache efficiency
fn memory_layout() {
    println!("\n4️⃣  Memory Layout & Cache Efficiency");
    
    // Array of Structs (AoS) - bad for cache
    #[derive(Clone)]
    struct Point3D {
        x: f32,
        y: f32,
        z: f32,
    }
    
    let aos = vec![
        Point3D { x: 1.0, y: 2.0, z: 3.0 },
        Point3D { x: 4.0, y: 5.0, z: 6.0 },
        Point3D { x: 7.0, y: 8.0, z: 9.0 },
    ];
    println!("   AoS (Array of Structs): {} points", aos.len());
    
    // Struct of Arrays (SoA) - good for cache and SIMD
    struct Points3D {
        x: Vec<f32>,
        y: Vec<f32>,
        z: Vec<f32>,
    }
    
    let soa = Points3D {
        x: vec![1.0, 4.0, 7.0],
        y: vec![2.0, 5.0, 8.0],
        z: vec![3.0, 6.0, 9.0],
    };
    println!("   SoA (Struct of Arrays): {} points", soa.x.len());
    println!("   SoA is better for vectorized operations!");
    
    // Processing all X coordinates together = better cache locality
    let sum_x: f32 = soa.x.iter().sum();
    println!("   Sum of X coordinates: {}", sum_x);
}

/// Compiler Optimizations
/// Tips for helping the compiler generate fast code
fn compiler_optimizations() {
    println!("\n5️⃣  Compiler Optimization Tips");
    
    println!("   ✅ Use iterators (auto-vectorized)");
    println!("   ✅ Use slices instead of Vec when possible");
    println!("   ✅ Inline small functions with #[inline]");
    println!("   ✅ Use const and const fn for compile-time computation");
    println!("   ✅ Profile with 'cargo flamegraph'");
    println!("   ✅ Benchmark with 'criterion'");
    println!("   ✅ Compile with --release flag");
    println!("   ✅ Use LTO (Link Time Optimization)");
    
    // Example: const evaluation
    const BUFFER_SIZE: usize = compute_buffer_size(128);
    println!("\n   Buffer size computed at compile time: {}", BUFFER_SIZE);
    
    // Inline example
    let result = fast_multiply(10, 20);
    println!("   Inlined multiply: {}", result);
}

// Compile-time computation
const fn compute_buffer_size(base: usize) -> usize {
    base * 1024
}

// Inline hint for small, frequently called functions
#[inline]
fn fast_multiply(a: i32, b: i32) -> i32 {
    a * b
}

// Performance Best Practices for AI/ML:
// 
// 1. **Stack Allocation**: Use arrays for small, fixed-size data
// 2. **Avoid Copies**: Use references (&) and slices (&[])
// 3. **Iterator Chains**: Compiler optimizes these heavily
// 4. **SoA Layout**: Better for SIMD and cache (tensors, matrices)
// 5. **Pre-allocate**: Use Vec::with_capacity() to avoid reallocations
// 6. **Release Mode**: Always benchmark with --release
// 7. **Profile First**: Use profilers to find actual bottlenecks
// 8. **Unsafe Code**: Only when proven necessary (rarely needed)
// 
// Typical latency targets:
// - Real-time inference: < 10ms
// - Interactive: < 100ms  
// - Batch processing: < 1s per batch
