// Chapter 2: Ownership & Memory Management
// Rust's most unique feature - guarantees memory safety without garbage collection

pub fn run_examples() {
    ownership_basics();
    borrowing_demo();
    lifetimes_demo();
    smart_pointers();
}

/// Ownership Rules:
/// 1. Each value has an owner
/// 2. There can only be one owner at a time
/// 3. When owner goes out of scope, value is dropped
fn ownership_basics() {
    println!("1️⃣  Ownership Basics");
    
    // s1 owns the String
    let s1 = String::from("hello");
    println!("   s1: {}", s1);
    
    // Move: ownership transfers from s1 to s2
    let s2 = s1;
    println!("   s2: {}", s2);
    // println!("   s1: {}", s1); // ❌ Error! s1 no longer valid
    
    // Clone: creates a deep copy
    let s3 = s2.clone();
    println!("   s2: {}, s3: {}", s2, s3); // Both valid
    
    // Copy types (stored on stack) don't move
    let x = 5;
    let y = x; // Copy, not move
    println!("   x: {}, y: {}", x, y); // Both valid
    
    // Function calls transfer ownership
    let message = String::from("Rust");
    takes_ownership(message);
    // println!("{}", message); // ❌ Error! message moved
    
    let num = 10;
    makes_copy(num);
    println!("   num still valid: {}", num); // ✅ OK, i32 is Copy
}

fn takes_ownership(s: String) {
    println!("   Inside function: {}", s);
} // s goes out of scope and is dropped

fn makes_copy(n: i32) {
    println!("   Inside function: {}", n);
} // n goes out of scope, nothing special happens

/// Borrowing: Access data without taking ownership
/// - Immutable references: &T (multiple allowed)
/// - Mutable references: &mut T (only one at a time)
fn borrowing_demo() {
    println!("\n2️⃣  Borrowing & References");
    
    let s = String::from("hello");
    
    // Immutable borrow - can have multiple
    let len = calculate_length(&s);
    println!("   Length of '{}' is {}", s, len); // s still valid!
    
    // Multiple immutable borrows are OK
    let r1 = &s;
    let r2 = &s;
    println!("   r1: {}, r2: {}", r1, r2);
    
    // Mutable borrow - only one at a time
    let mut s2 = String::from("hello");
    change(&mut s2);
    println!("   Modified: {}", s2);
    
    // Rules prevent data races at compile time!
    let mut s3 = String::from("test");
    let r3 = &s3; // ✅ Immutable borrow
    let r4 = &s3; // ✅ Another immutable borrow
    println!("   r3: {}, r4: {}", r3, r4);
    // let r5 = &mut s3; // ❌ Error! Can't have mutable while immutable exists
    
    // After last use of immutable refs, we can have mutable ref
    let r5 = &mut s3; // ✅ OK, r3 and r4 no longer used
    r5.push_str("!");
    println!("   r5: {}", r5);
}

fn calculate_length(s: &String) -> usize {
    s.len()
} // s goes out of scope, but doesn't drop (doesn't own the data)

fn change(s: &mut String) {
    s.push_str(", world");
}

/// Lifetimes: Ensure references are valid
/// Prevent dangling references at compile time
fn lifetimes_demo() {
    println!("\n3️⃣  Lifetimes");
    
    let string1 = String::from("long string");
    let string2 = String::from("short");
    
    let result = longest(&string1, &string2);
    println!("   Longest: {}", result);
    
    // Struct with lifetime
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence = novel.split('.').next().expect("No '.'");
    let excerpt = ImportantExcerpt { part: first_sentence };
    println!("   Excerpt: {}", excerpt.part);
}

// Lifetime annotation: tells compiler how lifetimes of parameters relate
// 'a means both parameters and return value live at least as long
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

// Struct with lifetime parameter
struct ImportantExcerpt<'a> {
    part: &'a str, // This reference must be valid as long as struct exists
}

/// Smart Pointers
/// Box<T>, Rc<T>, RefCell<T> for advanced memory management
fn smart_pointers() {
    println!("\n4️⃣  Smart Pointers");
    
    // Box<T> - heap allocation
    let b = Box::new(5);
    println!("   Box value: {}", b);
    
    // Useful for recursive types
    let list = List::Cons(1,
        Box::new(List::Cons(2,
            Box::new(List::Cons(3,
                Box::new(List::Nil))))));
    print!("   Linked list: ");
    print_list(&list);
    println!();
    
    // Box is great for large data (ML models, matrices)
    // Avoids stack overflow and enables easy transfer
}

// Recursive type using Box
enum List {
    Cons(i32, Box<List>),
    Nil,
}

fn print_list(list: &List) {
    match list {
        List::Cons(value, next) => {
            print!("{} -> ", value);
            print_list(next);
        }
        List::Nil => print!("Nil"),
    }
}

// Why Ownership Matters for AI/ML:
// 
// 1. **No Garbage Collection**: Predictable performance, no GC pauses
// 2. **Memory Safety**: No use-after-free, double-free, or data races
// 3. **Zero-Cost**: Compile-time checks, no runtime overhead
// 4. **Explicit Control**: Know exactly when allocations/deallocations happen
// 5. **Thread Safety**: Ownership rules prevent data races at compile time
// 
// For ML inference:
// - Predictable latency (no GC pauses)
// - Efficient memory usage (no runtime tracking)
// - Safe parallelism (fearless concurrency)
