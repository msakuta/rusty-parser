use clap::Parser;
use parser::*;

use std::fs::File;
use std::io::prelude::*;

#[derive(Parser, Debug)]
#[clap(author, version, about = "A CLI interpreter of dragon language")]
struct Args {
    #[clap(
        default_value = "Go_Logo.png",
        help = "Input source file name or one-linear program"
    )]
    input: String,
    #[clap(short, long, help = "Evaluate one line program")]
    eval: bool,
    #[clap(short, long, help = "Show AST")]
    ast: bool,
    #[clap(short = 'A', long, help = "Show AST in pretty format")]
    ast_pretty: bool,
    #[clap(short, help = "Compile to bytecode")]
    compile: bool,
    #[clap(short = 'R', help = "Compile and run")]
    compile_and_run: bool,
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    let code = {
        if args.eval {
            args.input.clone()
        } else if let Ok(mut file) = File::open(&args.input) {
            let mut contents = String::new();
            file.read_to_string(&mut contents)
                .map_err(|e| e.to_string())?;
            contents
        } else {
            return Err("Error: can't open file".to_string());
        }
    };
    let result = source(&code).map_err(|e| e.to_string())?;
    if !result.0.is_empty() {
        return Err(format!("Input has terminated unexpectedly: {:?}", result.0));
    }
    if args.ast_pretty {
        println!("Match: {:#?}", result.1);
    } else if args.ast {
        println!("Match: {:?}", result.1);
    }
    if args.compile {
        let bytecode = compile(&result.1, &mut EvalContext::new())
            .map_err(|e| format!("Error in compile(): {:?}", e))?;
        println!("bytecode: {:#?}", bytecode);
        if args.compile_and_run {
            interpret(&bytecode)?;
        }
    } else {
        run(&result.1, &mut EvalContext::new()).map_err(|e| format!("Error in run(): {:?}", e))?;
    }
    Ok(())
}
