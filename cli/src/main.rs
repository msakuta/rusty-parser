use clap::Parser;
use parser::*;

use std::fs::File;
use std::io::{prelude::*, BufReader, BufWriter};

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
    #[clap(short, help = "Read from bytecode")]
    bytecode: bool,
}

fn print_fn(values: &[Value]) -> Value {
    println!("Print: {values:?}");
    Value::I64(0)
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    let parse_source = |code| -> Result<(), String> {
        let result = source(code).map_err(|e| e.to_string())?;
        if !result.0.is_empty() {
            return Err(format!("Input has terminated unexpectedly: {:?}", result.0));
        }
        if args.ast_pretty {
            println!("Match: {:#?}", result.1);
        } else if args.ast {
            println!("Match: {:?}", result.1);
        }

        if args.compile {
            let mut bytecode =
                compile(&result.1).map_err(|e| format!("Error in compile(): {:?}", e))?;
            // println!("bytecode: {:#?}", bytecode);
            if let Ok(writer) = std::fs::File::create("out.cdragon") {
                bytecode
                    .write(&mut BufWriter::new(writer))
                    .map_err(|s| s.to_string())?;
            }
            if args.compile_and_run {
                bytecode.add_ext_fn("print".to_string(), Box::new(print_fn));
                bytecode.add_ext_fn(
                    "len".to_string(),
                    Box::new(|val| {
                        Value::I64(if let Value::Array(_, ref arr) = val[0] {
                            arr.len() as i64
                        } else {
                            0
                        })
                    }),
                );
                interpret(&bytecode)?;
            }
        } else {
            run(&result.1, &mut EvalContext::new())
                .map_err(|e| format!("Error in run(): {:?}", e))?;
        }

        Ok(())
    };

    if args.eval {
        parse_source(&args.input)?;
    } else if let Ok(mut file) = File::open(&args.input) {
        if args.bytecode {
            let mut bytecode = Bytecode::read(&mut BufReader::new(file))?;
            // println!("bytecode: {:#?}", bytecode);
            bytecode.add_ext_fn("print".to_string(), Box::new(print_fn));
            interpret(&bytecode)?;
        } else {
            let mut contents = String::new();
            file.read_to_string(&mut contents)
                .map_err(|e| e.to_string())?;
            parse_source(&contents)?;
        }
    } else {
        return Err("Error: can't open file".to_string());
    }

    Ok(())
}
