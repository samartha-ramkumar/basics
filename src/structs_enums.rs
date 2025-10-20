// Chapter 3: Structs and Enums
// Building blocks for AI/ML data structures

pub fn run_examples() {
    structs_demo();
    methods_demo();
    enums_demo();
    option_and_result();
}

/// Structs - Custom data types
fn structs_demo() {
    println!("1️⃣  Structs");
    
    // Define and create a struct
    let point = Point { x: 10.0, y: 20.0 };
    println!("   Point: ({}, {})", point.x, point.y);
    
    // Tuple struct (unnamed fields)
    let color = Color(255, 128, 0);
    println!("   Color: RGB({}, {}, {})", color.0, color.1, color.2);
    
    // Update syntax (create new struct from existing)
    let point2 = Point { x: 15.0, ..point };
    println!("   Point2: ({}, {})", point2.x, point2.y);
    
    // Struct for ML: Training data point
    let data_point = DataPoint {
        features: vec![1.5, 2.3, 0.8, 1.2],
        label: 1,
        weight: 1.0,
    };
    println!("   DataPoint features: {:?}, label: {}", 
             data_point.features, data_point.label);
}

// Classic struct
#[derive(Debug, Clone)]
struct Point {
    x: f64,
    y: f64,
}

// Tuple struct
struct Color(u8, u8, u8);

// ML-oriented struct
#[derive(Debug, Clone)]
struct DataPoint {
    features: Vec<f64>,
    label: i32,
    weight: f64,
}

/// Methods and Associated Functions
fn methods_demo() {
    println!("\n2️⃣  Methods & Associated Functions");
    
    // Create using associated function (like static method)
    let mut vector = Vector2D::new(3.0, 4.0);
    println!("   Vector: {:?}", vector);
    
    // Call methods
    let magnitude = vector.magnitude();
    println!("   Magnitude: {:.2}", magnitude);
    
    vector.normalize();
    println!("   Normalized: {:?}", vector);
    
    // ML Model example
    let mut model = LinearModel::new(2);
    model.set_weights(vec![0.5, -0.3]);
    model.set_bias(0.1);
    
    let input = vec![2.0, 3.0];
    let prediction = model.predict(&input);
    println!("   Model prediction: {:.2}", prediction);
}

#[derive(Debug)]
struct Vector2D {
    x: f64,
    y: f64,
}

impl Vector2D {
    // Associated function (constructor)
    fn new(x: f64, y: f64) -> Self {
        Vector2D { x, y }
    }
    
    // Method (takes &self)
    fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    
    // Mutable method (takes &mut self)
    fn normalize(&mut self) {
        let mag = self.magnitude();
        if mag != 0.0 {
            self.x /= mag;
            self.y /= mag;
        }
    }
}

// Simple linear model for ML
#[derive(Debug)]
struct LinearModel {
    weights: Vec<f64>,
    bias: f64,
}

impl LinearModel {
    fn new(input_size: usize) -> Self {
        LinearModel {
            weights: vec![0.0; input_size],
            bias: 0.0,
        }
    }
    
    fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights;
    }
    
    fn set_bias(&mut self, bias: f64) {
        self.bias = bias;
    }
    
    fn predict(&self, input: &[f64]) -> f64 {
        let mut result = self.bias;
        for (w, x) in self.weights.iter().zip(input.iter()) {
            result += w * x;
        }
        result
    }
}

/// Enums - Types with variants
fn enums_demo() {
    println!("\n3️⃣  Enums");
    
    // Simple enum
    let status = TrainingStatus::Training;
    println!("   Status: {:?}", status);
    
    // Enum with data
    let layer1 = Layer::Dense { units: 128, activation: Activation::ReLU };
    let layer2 = Layer::Dropout { rate: 0.5 };
    let layer3 = Layer::Conv2D { filters: 32, kernel_size: 3 };
    
    println!("   Layer 1: {:?}", layer1);
    println!("   Layer 2: {:?}", layer2);
    println!("   Layer 3: {:?}", layer3);
    
    // Pattern matching on enums
    describe_layer(&layer1);
    describe_layer(&layer2);
    describe_layer(&layer3);
}

#[derive(Debug)]
enum TrainingStatus {
    NotStarted,
    Training,
    Converged,
    Failed,
}

#[derive(Debug)]
enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
}

// Enum with data in variants (great for ML architectures)
#[derive(Debug)]
enum Layer {
    Dense { 
        units: usize, 
        activation: Activation 
    },
    Conv2D { 
        filters: usize, 
        kernel_size: usize 
    },
    Dropout { 
        rate: f64 
    },
    BatchNorm,
}

fn describe_layer(layer: &Layer) {
    match layer {
        Layer::Dense { units, activation } => {
            println!("   Dense layer: {} units, {:?} activation", units, activation);
        }
        Layer::Conv2D { filters, kernel_size } => {
            println!("   Conv2D layer: {} filters, {}x{} kernel", 
                     filters, kernel_size, kernel_size);
        }
        Layer::Dropout { rate } => {
            println!("   Dropout layer: {:.1}% dropout", rate * 100.0);
        }
        Layer::BatchNorm => {
            println!("   Batch normalization layer");
        }
    }
}

/// Option and Result - Rust's way of handling nullability and errors
fn option_and_result() {
    println!("\n4️⃣  Option<T> and Result<T, E>");
    
    // Option<T> - represents optional values (no null!)
    let numbers = vec![1, 2, 3, 4, 5];
    
    let found = find_number(&numbers, 3);
    match found {
        Some(index) => println!("   Found at index: {}", index),
        None => println!("   Not found"),
    }
    
    let not_found = find_number(&numbers, 10);
    if let Some(index) = not_found {
        println!("   Found at index: {}", index);
    } else {
        println!("   Number 10 not found");
    }
    
    // Result<T, E> - for operations that can fail
    let valid_result = parse_feature("3.14");
    match valid_result {
        Ok(value) => println!("   Parsed value: {}", value),
        Err(e) => println!("   Parse error: {}", e),
    }
    
    let invalid_result = parse_feature("not_a_number");
    match invalid_result {
        Ok(value) => println!("   Parsed value: {}", value),
        Err(e) => println!("   Parse error: {}", e),
    }
    
    // Chaining operations with ? operator
    match process_training_data() {
        Ok(msg) => println!("   {}", msg),
        Err(e) => println!("   Error: {}", e),
    }
}

fn find_number(nums: &[i32], target: i32) -> Option<usize> {
    for (i, &num) in nums.iter().enumerate() {
        if num == target {
            return Some(i);
        }
    }
    None
}

fn parse_feature(s: &str) -> Result<f64, String> {
    s.parse::<f64>()
        .map_err(|_| format!("Failed to parse '{}' as f64", s))
}

fn process_training_data() -> Result<String, String> {
    let feature1 = parse_feature("1.5")?; // ? operator propagates errors
    let feature2 = parse_feature("2.3")?;
    let feature3 = parse_feature("0.8")?;
    
    Ok(format!("Processed features: {}, {}, {}", feature1, feature2, feature3))
}

// Key Takeaways for AI/ML:
// 
// 1. **Structs**: Perfect for models, data points, hyperparameters
// 2. **Enums**: Great for architectures, activation functions, optimizer types
// 3. **Option<T>**: Safe handling of optional values (no null pointer errors)
// 4. **Result<T, E>**: Explicit error handling for loading data, training, etc.
// 5. **Pattern Matching**: Clean, exhaustive handling of all cases
