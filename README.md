# Rhai - embedded scripting for Rust

Rhai is an embedded scripting language for Rust that gives you a safe and easy way to add scripting to your applications.

Rhai's current feature set:

* Easy integration with Rust functions and data types
* Fairly efficient (1 mil iterations in 0.75 sec on my 5 year old laptop)
* Low compile-time overhead (~0.6 sec debug/~3 sec release for script runner app)
* Easy-to-use language based on JS+Rust
* Support for overloaded functions
* No additional dependencies
* No unsafe code

**Note:** Currently, the version is 0.8.1, so the language and APIs may change before they stabilize.*

## Installation

You can install Rhai using crates by adding this line to your dependences:

```toml
[dependencies]
rhai = "0.8.1"
```

## Related

Other cool projects to check out:
* [ChaiScript](http://chaiscript.com/) - A strong inspiration for Rhai.  An embedded scripting language for C++ that I helped created many moons ago, now being lead by my cousin.
* You can also check out the list of [scripting languages for Rust](https://github.com/rust-unofficial/awesome-rust#scripting) on [awesome-rust](https://github.com/rust-unofficial/awesome-rust)

## Examples
The repository contains several examples in the `examples` folder:
- `arrays_and_structs` demonstrates registering a new type to Rhai and the usage of arrays on it
- `custom_types_and_methods` shows how to register a type and methods for it
- `hello` simple example that evaluates an expression and prints the result
- `reuse_scope` evaluates two pieces of code in separate runs, but using a common scope
- `rhai_runner` runs each filename passed to it as a Rhai script
- `simple_fn` shows how to register a Rust function to a Rhai engine
- `repl` a simple REPL, see source code for what it can do at the moment

Examples can be run with the following command:
```bash
cargo run --example name
```

## Example Scripts
We also have a few examples scripts that showcase Rhai's features, all stored in the `scripts` folder:
- `array.rhai` - arrays in Rhai
- `assignment.rhai` - variable declarations
- `comments.rhai` - just comments
- `function_decl1.rhai` - a function without parameters
- `function_decl2.rhai` - a function with two parameters
- `function_decl3.rhai` - a function with many parameters
- `if1.rhai` - if example
- `loop.rhai` - endless loop in Rhai, this example emulates a do..while cycle
- `op1.rhai` - just a simple addition
- `op2.rhai` - simple addition and multiplication
- `op3.rhai` - change evaluation order with parenthesis
- `speed_test.rhai` - a simple program to measure the speed of Rhai's interpreter
- `string.rhai`- string operations
- `while.rhai` - while loop

To run the scripts, you can either make your own tiny program, or make use of the `rhai_runner`
example program:
```bash
cargo run --example rhai_runner scripts/any_script.rhai
```

# Hello world

To get going with Rhai, you create an instance of the scripting engine and then run eval.

```rust
extern crate rhai;
use rhai::Engine;

fn main() {
    let mut engine = Engine::new();

    if let Ok(result) = engine.eval::<i64>("40 + 2") {
        println!("Answer: {}", result);  // prints 42
    }
}
```

You can also evaluate a script file:

```rust
if let Ok(result) = engine.eval_file::<i64>("hello_world.rhai") { ... }
```

# Working with functions

Rhai's scripting engine is very lightweight.  It gets its ability from the functions in your program.  To call these functions, you need to register them with the scripting engine.

```rust
extern crate rhai;
use rhai::{Engine, RegisterFn};

fn add(x: i64, y: i64) -> i64 {
    x + y
}

fn main() {
    let mut engine = Engine::new();

    engine.register_fn("add", add);

    if let Ok(result) = engine.eval::<i64>("add(40, 2)") {
       println!("Answer: {}", result);  // prints 42
    }
}
```

# Working with generic functions

Generic functions can be used in Rhai, but you'll need to register separate instances for each concrete type:

```rust
use std::fmt::Display;

extern crate rhai;
use rhai::{Engine, RegisterFn};

fn showit<T: Display>(x: &mut T) -> () {
    println!("{}", x)
}

fn main() {
    let mut engine = Engine::new();

    engine.register_fn("print", showit as fn(x: &mut i64)->());
    engine.register_fn("print", showit as fn(x: &mut bool)->());
    engine.register_fn("print", showit as fn(x: &mut String)->());
}
```

You can also see in this example how you can register multiple functions (or in this case multiple instances of the same function) to the same name in script.  This gives you a way to overload functions and call the correct one, based on the types of the arguments, from your script.

# Custom types and methods

Here's an more complete example of working with Rust.  First the example, then we'll break it into parts:

```rust
#[macro_use]
extern crate rhai;
use rhai::{Engine, Type, RegisterTypeFn};

#[derive(Clone)]
struct TestStruct {
    x: i64
}

impl TestStruct {
    fn update(&mut self) {
        self.x += 1000;
    }

    fn new() -> TestStruct {
        TestStruct { x: 1 }
    }
}

fn main() {
    let mut engine = Engine::new();

    register_type!(engine, TestStruct,
        functions: new, update
    );

    if let Ok(result) = engine.eval::<TestStruct>("let x = TestStruct::new(); x.update(); x") {
        println!("result: {}", result.x); // prints 1001
    }
}
```

First, for each type we use with the engine, we need to be able to Clone.  This allows the engine to pass by value and still keep its own state.

```rust
#[derive(Clone)]
struct TestStruct {
    x: i64
}
```

Next, we create a few methods that we'll later use in our scripts.  Notice that we register our custom type with the engine.
```rust
impl TestStruct {
    fn update(&mut self) {
        self.x += 1000;
    }

    fn new() -> TestStruct {
        TestStruct { x: 1 }
    }
}

let mut engine = Engine::new();

```

To use methods and functions of a type with the engine, we need to register them. There is a macro that wil help us to do that.

*Note: the engine follows the convention that methods use a &mut first parameter so that invoking methods can update the value in memory. (For now it doesn't work for double-colon invocation like TestStruct::update(x) because x is being copied and then modified)*

```rust
register_type!(engine, TestStruct,
    functions: new, update
);
```

Finally, we call our script.  The script can see the function and type we registered earlier.  We need to get the result back out from script land just as before, this time casting to our custom struct type.
```rust
if let Ok(result) = engine.eval::<TestStruct>("let x = TestStruct::new(); x.update(); x") {
    println!("result: {}", result.x); // prints 1001
}
```

# Getters and setters

Similarly, you can work with fields of your custom types.

For example:

```rust
#[derive(Clone)]
struct TestStruct {
    x: i64
}

impl TestStruct {
    fn new() -> TestStruct {
        TestStruct { x: 1 }
    }
}

let mut engine = Engine::new();

register_type!(engine, TestStruct,
    fields: x;
    functions: new
);

if let Ok(result) = engine.eval::<i64>("let a = TestStruct::new(); a.x = 500; a.x") {
    println!("result: {}", result);
}
```

# Maintaining state

By default, Rhai treats each engine invocation as a fresh one, persisting only the functions that have been defined but no top-level state.  This gives each one a fairly clean starting place.  Sometimes, though, you want to continue using the same top-level state from one invocation to the next.

In this example, we thread the same state through multiple invocations:

```rust
extern crate rhai;
use rhai::{Engine, Scope};

fn main() {
    let mut engine = Engine::new();
    let mut scope: Scope = Vec::new();

    if let Ok(_) = engine.eval_with_scope::<()>(&mut scope, "let x = 4 + 5") { } else { assert!(false); }

    if let Ok(result) = engine.eval_with_scope::<i64>(&mut scope, "x") {
       println!("result: {}", result);
    }
}
```

# Rhai Language guide

## Variables

```rust
let x = 3;
```

## Operators

```rust
let x = (1 + 2) * (6 - 4) / 2;
```

## If
```rust
if true {
    print("it's true!");
}
else {
    print("It's false!");
}
```

## While
```rust
let x = 10;
while x > 0 {
    print(x);
    if x == 5 {
        break;
    }
    x = x - 1;
}
```

## Loop
```rust
let x = 10;

loop {
    print(x);
    x = x - 1;
	if x == 0 { break; }
}
```

## Functions

Rhai supports defining functions in script:

```rust
fn add(x, y) {
    return x + y;
}

print(add(2, 3))
```

Just like in Rust, you can also use an implicit return.

```rust
fn add(x, y) {
    x + y
}

print(add(2, 3))
```
## Arrays

You can create arrays of values, and then access them with numeric indices.

```rust
let y = [1, 2, 3];
y[1] = 5;

print(y[1]);
```

## Members and methods

```rust
let a = TestStruct::new();
a.x = 500;
a.update();
```

## Strings and Chars

```rust
let name = "Bob";
let middle_initial = 'C';
```

## Comments

```rust
let /* intruder comment */ name = "Bob";
// This is a very important comment
/* This comment spans
   multiple lines, so it
   only makes sense that
   it is even more important */

/* Fear not, Rhai satisfies all your nesting
   needs with nested comments:
   /*/*/*/*/**/*/*/*/*/
*/
```

## Unary operators

```rust
let number = -5;
number = -5 - +5;
let booly = !true;
```

## Compound assignment operators

```rust
let number = 5;
number += 4;
number -= 3;
number *= 2;
number /= 1;
number %= 3;
number <<= 2;
number >>= 1;
```

The `+=` operator can also be used to build strings:

```rust
let my_str = "abc";
my_str += "ABC";

my_str == "abcABC"
```
