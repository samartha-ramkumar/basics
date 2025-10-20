// Chapter 1: Rust Fundamentals
// Learn the basics of Rust syntax, types, and control flow

pub fn run_examples() {
    variables_and_mutability();
    data_types();
    functions_demo();
    control_flow();
    collections_demo();
}

/// Variables and Mutability
/// Rust is immutable by default - you must explicitly mark mutable variables
fn variables_and_mutability() {
    println!("1️⃣  Variables and Mutability");
    
    // Immutable by default (cannot be changed)
    let x = 5;
    println!("   Immutable x: {}", x);
    
    // Mutable variables need 'mut' keyword
    let mut y = 10;
    println!("   Initial y: {}", y);
    y = 15;
    println!("   Modified y: {}", y);
    
    // Constants - always immutable, type must be annotated
    const MAX_POINTS: u32 = 100_000;
    println!("   Constant MAX_POINTS: {}", MAX_POINTS);
    
    // Shadowing - redeclare a variable with same name
    let z = 5;
    let z = z + 1; // Creates a new variable, original is shadowed
    println!("   Shadowed z: {}", z);
}

/// Data Types
/// Rust is statically typed - all types known at compile time
fn data_types() {
    println!("\n2️⃣  Data Types");
    
    // Integers: i8, i16, i32, i64, i128, isize (signed)
    //          u8, u16, u32, u64, u128, usize (unsigned)
    let integer: i32 = -42;
    let unsigned: u32 = 42;
    println!("   Integer: {}, Unsigned: {}", integer, unsigned);
    
    // Floating point: f32, f64 (default)
    let float: f64 = 3.14159;
    println!("   Float: {}", float);
    
    // Boolean
    let is_learning: bool = true;
    println!("   Boolean: {}", is_learning);
    
    // Character (4 bytes, Unicode)
    let emoji: char = '🦀';
    println!("   Character: {}", emoji);
    
    // Tuple - fixed size, mixed types
    let tuple: (i32, f64, char) = (500, 6.4, 'R');
    let (x, y, z) = tuple; // Destructuring
    println!("   Tuple: ({}, {}, {})", x, y, z);
    
    // Array - fixed size, same type
    let array: [i32; 5] = [1, 2, 3, 4, 5];
    println!("   Array: {:?}", array);
    println!("   Array length: {}", array.len());
}

/// Functions
/// Functions are declared with 'fn' keyword
fn functions_demo() {
    println!("\n3️⃣  Functions");
    
    let result = add(5, 3);
    println!("   5 + 3 = {}", result);
    
    let product = multiply(4, 7);
    println!("   4 * 7 = {}", product);
    
    // Functions with no return value return ()
    print_message("Hello from Rust!");
}

// Function with parameters and return value
// Last expression is returned (no semicolon)
fn add(a: i32, b: i32) -> i32 {
    a + b // No semicolon = return value
}

// Alternative explicit return
fn multiply(a: i32, b: i32) -> i32 {
    return a * b; // Explicit return
}

// Function with no return value (returns unit type ())
fn print_message(msg: &str) {
    println!("   Message: {}", msg);
}

/// Control Flow
/// if/else, loops, pattern matching
fn control_flow() {
    println!("\n4️⃣  Control Flow");
    
    // if/else expressions
    let number = 7;
    if number < 5 {
        println!("   Number is less than 5");
    } else if number == 5 {
        println!("   Number is 5");
    } else {
        println!("   Number is greater than 5");
    }
    
    // if is an expression, can assign result
    let condition = true;
    let value = if condition { 5 } else { 10 };
    println!("   Conditional value: {}", value);
    
    // loop - infinite loop (use 'break' to exit)
    let mut counter = 0;
    let result = loop {
        counter += 1;
        if counter == 10 {
            break counter * 2; // Return value from loop
        }
    };
    println!("   Loop result: {}", result);
    
    // while loop
    let mut countdown = 3;
    print!("   Countdown: ");
    while countdown > 0 {
        print!("{} ", countdown);
        countdown -= 1;
    }
    println!("🚀");
    
    // for loop - iterate over collections
    print!("   For loop: ");
    for i in 1..=5 {
        print!("{} ", i);
    }
    println!();
    
    // Match - powerful pattern matching (like switch++)
    let number = 3;
    match number {
        1 => println!("   Match: One"),
        2 | 3 => println!("   Match: Two or Three"),
        4..=10 => println!("   Match: Between 4 and 10"),
        _ => println!("   Match: Something else"),
    }
}

/// Collections
/// Vec, HashMap, and other common collections
fn collections_demo() {
    println!("\n5️⃣  Collections");
    
    // Vector - growable array (heap allocated)
    let mut vec = Vec::new();
    vec.push(1);
    vec.push(2);
    vec.push(3);
    println!("   Vector: {:?}", vec);
    
    // Vec macro
    let vec2 = vec![10, 20, 30, 40, 50];
    println!("   Vector from macro: {:?}", vec2);
    
    // Iterating over vector
    print!("   Iterating: ");
    for item in &vec2 {
        print!("{} ", item);
    }
    println!();
    
    // String - growable, UTF-8 encoded text
    let mut s = String::from("Hello");
    s.push_str(", Rust!");
    println!("   String: {}", s);
    
    // String slices (&str)
    let slice = &s[0..5];
    println!("   Slice: {}", slice);
}

// Key Takeaways for AI/ML:
// 
// 1. Immutability by default - prevents bugs in ML pipelines
// 2. Strong typing - catches errors at compile time
// 3. No null - use Option<T> for optional values
// 4. Pattern matching - clean error handling
// 5. Zero-cost abstractions - high-level code, low-level performance
