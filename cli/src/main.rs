use parser::*;
use std::fs::File;
use std::io::prelude::*;
use std::{
    cell::RefCell,
    collections::HashMap,
    env,
    rc::Rc,
};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut contents = String::new();
    let code = if 1 < args.len() {
        if let Ok(mut file) = File::open(&args[1]) {
            file.read_to_string(&mut contents)?;
            &contents
        } else {
            &args[1]
        }
    } else {
        r"var x;
  /* This is a block comment. */
  var y;
  123;
  123 + 456;
  "
    };
    if let Ok(result) = source(code) {
        println!("Match: {:?}", result.1);
        run(&result.1, &mut EvalContext::new()).expect("Error in run()");
    } else {
        println!("failed");
    }
    Ok(())
}
