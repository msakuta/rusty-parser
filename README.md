# rusty-parser

A self-learning project for making a new language using Rust and nom.

Try it now on your browser! https://msakuta.github.io/rusty-parser/

## Requirements

* rust 1.44
* npm 7.0.2 (for WebAssembly browser, any web server should work)


# How to build

There are 2 ways to build this project.

* Command line interpreter
* WebAssembly browser application

## Command line interpreter

One is for native command line application.
It can read a text file, parse it to an AST and run it.

    cd cli
    cargo run --release <script-file>.dragon


## WebAssembly browser application

You can also build a wasm package and run the interpreter on the browser.

    cd wasm
    wasm-pack build --target web

To launch the application, you can use `npx`

    npx serve

and browse http://localhost:5000.

## TODOs

In ascending order of difficulty.

* [x] Arithmetic operators (+, -, *, /)
* [x] Functions, recursive calls
* [x] Loops (for, while, loop)
* [x] Proper expression statements (brace expressions)
* [x] Variable definition initializer
* [x] Type declaration
* [x] Primitive types (i32, i64, f32, f64)
* [x] String type
* [x] Logical operators (||, &&, !)
* [ ] Bitwise operators (|, &, ^, ~)
* [ ] String manipulations
* [x] Array types
* [x] WebAssembly build target
* [x] Function return types
* [ ] Proper error handling
* [ ] Tuple types
* [ ] Multi-dimensional arrays
* [ ] Function types (first class function variables)
* [ ] Lambda expressions
* [ ] Static type checking (instead of runtime coercion)
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
