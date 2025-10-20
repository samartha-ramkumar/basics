// Chapter 5: Concurrency & Parallelism
// Essential for high-performance ML training and inference

use std::thread;
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub fn run_examples() {
    threads_demo();
    shared_state();
    message_passing();
    parallel_processing();
}

/// Threading Basics
/// Rust's ownership system prevents data races at compile time!
fn threads_demo() {
    println!("1️⃣  Threading Basics");
    
    // Spawn a thread
    let handle = thread::spawn(|| {
        for i in 1..5 {
            println!("   Thread: count {}", i);
            thread::sleep(Duration::from_millis(10));
        }
    });
    
    // Main thread continues
    for i in 1..3 {
        println!("   Main: count {}", i);
        thread::sleep(Duration::from_millis(10));
    }
    
    // Wait for thread to finish
    handle.join().unwrap();
    println!("   Thread completed!");
    
    // Moving data into threads
    let data = vec![1, 2, 3, 4, 5];
    let handle = thread::spawn(move || {
        let sum: i32 = data.iter().sum();
        println!("   Thread computed sum: {}", sum);
        sum
    });
    
    let result = handle.join().unwrap();
    println!("   Result from thread: {}", result);
}

/// Shared State with Arc and Mutex
/// Arc = Atomic Reference Counting (thread-safe shared ownership)
/// Mutex = Mutual exclusion (thread-safe mutable access)
fn shared_state() {
    println!("\n2️⃣  Shared State (Arc + Mutex)");
    
    // Counter shared between threads
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    
    for i in 0..5 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
            println!("   Thread {} incremented counter", i);
        });
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("   Final counter: {}", *counter.lock().unwrap());
    
    // ML Example: Shared gradient accumulation
    let gradients = Arc::new(Mutex::new(vec![0.0; 3]));
    let mut handles = vec![];
    
    for i in 0..3 {
        let gradients = Arc::clone(&gradients);
        let handle = thread::spawn(move || {
            // Simulate computing gradients
            let local_grad = vec![i as f64 + 0.1, i as f64 + 0.2, i as f64 + 0.3];
            
            // Accumulate into shared gradients
            let mut shared = gradients.lock().unwrap();
            for (j, grad) in local_grad.iter().enumerate() {
                shared[j] += grad;
            }
            println!("   Worker {} accumulated gradients", i);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let final_grads = gradients.lock().unwrap();
    println!("   Accumulated gradients: {:?}", *final_grads);
}

/// Message Passing with Channels
/// "Do not communicate by sharing memory; share memory by communicating"
fn message_passing() {
    println!("\n3️⃣  Message Passing (Channels)");
    
    use std::sync::mpsc;
    
    // Create channel
    let (tx, rx) = mpsc::channel();
    
    // Producer thread
    thread::spawn(move || {
        let messages = vec![
            "Training epoch 1",
            "Training epoch 2",
            "Training epoch 3",
            "Training complete",
        ];
        
        for msg in messages {
            tx.send(msg).unwrap();
            thread::sleep(Duration::from_millis(100));
        }
    });
    
    // Consumer (main thread)
    for received in rx {
        println!("   Received: {}", received);
    }
    
    // Multiple producers
    let (tx, rx) = mpsc::channel();
    
    for id in 0..3 {
        let tx = tx.clone();
        thread::spawn(move || {
            let msg = format!("Worker {} finished batch", id);
            tx.send(msg).unwrap();
        });
    }
    drop(tx); // Drop original sender
    
    println!("\n   Messages from workers:");
    for received in rx {
        println!("   {}", received);
    }
}

/// Parallel Processing for ML
/// Data parallelism - process different data in parallel
fn parallel_processing() {
    println!("\n4️⃣  Parallel Data Processing");
    
    // Simulate dataset
    let dataset: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![i as f64, (i * 2) as f64, (i * 3) as f64])
        .collect();
    
    println!("   Processing {} samples in parallel...", dataset.len());
    
    // Split dataset for parallel processing
    let chunk_size = dataset.len() / 4;
    let mut handles = vec![];
    
    // Share dataset across threads (Arc = read-only sharing)
    let dataset = Arc::new(dataset);
    
    for i in 0..4 {
        let dataset = Arc::clone(&dataset);
        let handle = thread::spawn(move || {
            let start = i * chunk_size;
            let end = if i == 3 { dataset.len() } else { (i + 1) * chunk_size };
            
            let mut local_sum = 0.0;
            for sample in &dataset[start..end] {
                local_sum += sample.iter().sum::<f64>();
            }
            
            println!("   Thread {} processed samples {}..{}", i, start, end);
            local_sum
        });
        handles.push(handle);
    }
    
    // Collect results
    let mut total_sum = 0.0;
    for handle in handles {
        total_sum += handle.join().unwrap();
    }
    
    println!("   Total sum from all threads: {}", total_sum);
    
    // Note: For real ML workloads, use rayon for easier parallelism!
    println!("\n   💡 Pro tip: Use 'rayon' crate for easy data parallelism");
    println!("      Example: data.par_iter().map(|x| process(x)).collect()");
}

// Concurrency Patterns for AI/ML:
// 
// 1. **Data Parallelism**: 
//    - Split batches across threads
//    - Each thread processes different data
//    - Combine results at the end
// 
// 2. **Model Parallelism**:
//    - Split model across threads
//    - Each thread computes different layers
//    - Pipeline processing
// 
// 3. **Async I/O**:
//    - Load data while training
//    - Non-blocking file operations
//    - Concurrent API requests
// 
// 4. **Producer-Consumer**:
//    - Data loading thread (producer)
//    - Training thread (consumer)
//    - Buffered pipeline
// 
// Key Benefits:
// - ✅ No data races (compile-time guarantee)
// - ✅ No need for GIL (Python's Global Interpreter Lock)
// - ✅ True parallelism on multiple cores
// - ✅ Predictable performance
// 
// Best Practices:
// - Use rayon for data parallelism (easier than manual threads)
// - Use channels for pipeline parallelism
// - Use Arc<Mutex<T>> for shared mutable state
// - Profile to ensure speedup (Amdahl's law)
