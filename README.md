# rusty-parser

A self-learning project for making a new language using Rust and nom.

## Requirements

* rust 1.44

## Interpreter

Currently it only works as an iterpreter. It can read a text file, parser it to a
AST and run it.

## TODOs

In ascending order of difficulty.

* [x] Functions, recursive calls
* [x] Loops
* [x] Proper expression statements (brace expressions)
* [ ] Definition initialization
* [ ] Type declaration
* [ ] Primitive types (i32, u32, f32, f64)
* [ ] String types? (Optional?)
* [ ] Array types
* [ ] Tuple types
* [ ] Mutability qualifiers
* [ ] Array slice syntax
* [ ] Array shape constraints
* [ ] Broadcasting operators
* [ ] Custom operators
* [ ] Run on VM (not directly on AST)
* [ ] Compile to bytecode (via serde?)

## Ideas

I want to make it a complementary DSL for data manipulation, such as numpy.

* First-class array and matrix operations - from Matlab
* Statically typed array shapes - from Futhark
* Organized broadcasting operators (dot prefix) - from Julia
* Adapt to LLVM backend to make it a native compiler
