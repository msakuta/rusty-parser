use std::{cell::RefCell, collections::HashMap, io::Write, rc::Rc};

use crate::{
    bytecode::{
        std_functions, Bytecode, BytecodeArg, FnBytecode, FnProto, Instruction, NativeFn, OpCode,
    },
    interpreter::{eval, EvalContext, RunResult},
    parser::{ExprEnum, Expression, Statement},
    value::ArrayInt,
    EvalError, TypeDecl, Value,
};

#[derive(Debug, Clone, Copy)]
enum Target {
    None,
    /// Literal value with literal index.
    /// Right now whether the target is literal is not utilized, but we may use it for optimization.
    #[allow(unused)]
    Literal(usize),
    /// If it is an allocated stack slot for a local variable, it will contain the index
    /// into locals array. We use it to keep track of which is local variable so that
    /// we won't destroy it by accident.
    /// **NOTE** that it is not a stack index.
    #[allow(unused)]
    Local(usize),
}

impl Default for Target {
    fn default() -> Target {
        Self::None
    }
}

#[derive(Debug)]
struct LocalVar {
    name: String,
    stack_idx: usize,
}

struct CompilerEnv {
    functions: HashMap<String, FnProto>,
}

impl CompilerEnv {
    fn new(mut functions: HashMap<String, FnProto>) -> Self {
        std_functions(&mut |name, f| {
            functions.insert(name, FnProto::Native(f));
        });
        Self { functions }
    }
}

struct Compiler<'a> {
    env: &'a mut CompilerEnv,
    bytecode: FnBytecode,
    target_stack: Vec<Target>,
    locals: Vec<Vec<LocalVar>>,
    break_ips: Vec<usize>,
}

impl<'a> Compiler<'a> {
    fn new(args: Vec<LocalVar>, fn_args: Vec<BytecodeArg>, env: &'a mut CompilerEnv) -> Self {
        Self {
            env,
            bytecode: FnBytecode {
                literals: vec![],
                args: fn_args,
                instructions: vec![],
                stack_size: 0,
            },
            target_stack: (0..args.len() + 1)
                .map(|i| {
                    if i == 0 {
                        Target::None
                    } else {
                        Target::Local(i - 1)
                    }
                })
                .collect(),
            locals: vec![args],
            break_ips: vec![],
        }
    }

    /// Fixup the jump address for the break statements in the previous loop to current instruction pointer.
    /// Call it just after leaving loop body.
    fn fixup_breaks(&mut self) {
        let break_jmp_addr = self.bytecode.instructions.len();
        for ip in &self.break_ips {
            self.bytecode.instructions[*ip].arg1 = break_jmp_addr as u16;
        }
        self.break_ips.clear();
    }

    /// Returns a stack index, removing potential duplicate values
    fn find_or_create_literal(&mut self, value: &Value) -> usize {
        let bytecode = &mut self.bytecode;
        let literal = if let Some((i, _lit)) = bytecode
            .literals
            .iter()
            .enumerate()
            .find(|(_, lit)| *lit == value)
        {
            i
        } else {
            let literal = bytecode.literals.len();
            bytecode.literals.push(value.clone());
            literal
        };

        let stk_target = self.target_stack.len();
        self.target_stack.push(Target::Literal(literal));
        bytecode.instructions.push(Instruction::new(
            OpCode::LoadLiteral,
            literal as u8,
            stk_target as u16,
        ));
        stk_target
    }
}

type CompileResult<T> = Result<T, CompileError>;

#[non_exhaustive]
#[derive(Debug)]
pub enum CompileError {
    LocalsStackUnderflow,
    BreakInArrayLiteral,
    DisallowedBreak,
    EvalError(EvalError),
    VarNotFound(String),
    FnNotFound(String),
    InsufficientNamedArgs,
    FromUtf8Error(std::string::FromUtf8Error),
    IoError(std::io::Error),
}

impl std::error::Error for CompileError {}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LocalsStackUnderflow => write!(f, "Local variables stack underflow"),
            Self::BreakInArrayLiteral => write!(f, "Break in array literal not supported"),
            Self::DisallowedBreak => write!(f, "Break in function default arg is not allowed"),
            Self::EvalError(e) => write!(f, "Evaluation error: {e}"),
            Self::VarNotFound(name) => write!(f, "Variable {name} not found in scope"),
            Self::FnNotFound(name) => write!(f, "Function {name} is not defined"),
            Self::InsufficientNamedArgs => {
                write!(f, "Named arguments does not cover all required args")
            }
            Self::FromUtf8Error(e) => e.fmt(f),
            Self::IoError(e) => e.fmt(f),
        }
    }
}

impl From<std::string::FromUtf8Error> for CompileError {
    fn from(value: std::string::FromUtf8Error) -> Self {
        Self::FromUtf8Error(value)
    }
}

impl From<std::io::Error> for CompileError {
    fn from(value: std::io::Error) -> Self {
        Self::IoError(value)
    }
}

pub fn compile<'src, 'ast>(
    stmts: &'ast [Statement<'src>],
    functions: HashMap<String, NativeFn>,
) -> CompileResult<Bytecode> {
    compile_int(stmts, functions, &mut std::io::sink())
}

pub fn disasm<'src, 'ast>(
    stmts: &'ast [Statement<'src>],
    functions: HashMap<String, NativeFn>,
) -> CompileResult<String> {
    let mut disasm = Vec::<u8>::new();
    let mut cursor = std::io::Cursor::new(&mut disasm);

    compile_int(stmts, functions, &mut cursor)?;

    Ok(String::from_utf8(disasm)?)
}

fn compile_int<'src, 'ast>(
    stmts: &'ast [Statement<'src>],
    functions: HashMap<String, NativeFn>,
    disasm: &mut impl Write,
) -> CompileResult<Bytecode> {
    let functions = functions
        .into_iter()
        .map(|(k, v)| (k, FnProto::Native(v)))
        .collect();

    let mut env = CompilerEnv::new(functions);

    retrieve_fn_signatures(stmts, &mut env);

    let mut compiler = Compiler::new(vec![], vec![], &mut env);
    if let Some(last_target) = emit_stmts(stmts, &mut compiler)? {
        compiler
            .bytecode
            .instructions
            .push(Instruction::new(OpCode::Ret, 0, last_target as u16));
    }
    compiler.bytecode.stack_size = compiler.target_stack.len();

    let bytecode = FnProto::Code(compiler.bytecode);

    let mut functions = env.functions;
    functions.insert("".to_string(), bytecode);

    for (fname, fnproto) in &functions {
        if let FnProto::Code(bytecode) = fnproto {
            if fname.is_empty() {
                writeln!(disasm, "\nFunction <toplevel> disassembly:")?;
            } else {
                writeln!(disasm, "\nFunction {fname} disassembly:")?;
            }
            bytecode.disasm(disasm)?;
        }
    }

    Ok(Bytecode { functions })
}

fn compile_fn<'src, 'ast>(
    env: &mut CompilerEnv,
    stmts: &'ast [Statement<'src>],
    args: Vec<LocalVar>,
    fn_args: Vec<BytecodeArg>,
) -> CompileResult<FnProto> {
    let mut compiler = Compiler::new(args, fn_args, env);
    if let Some(last_target) = emit_stmts(stmts, &mut compiler)? {
        compiler
            .bytecode
            .instructions
            .push(Instruction::new(OpCode::Ret, 0, last_target as u16));
    }
    compiler.bytecode.stack_size = compiler.target_stack.len();

    Ok(FnProto::Code(compiler.bytecode))
}

fn retrieve_fn_signatures(stmts: &[Statement], env: &mut CompilerEnv) {
    for stmt in stmts {
        match stmt {
            Statement::FnDecl {
                name, args, stmts, ..
            } => {
                let args = args.iter().map(|arg| arg.name.to_string()).collect();
                let bytecode = FnBytecode::proto(args);
                env.functions
                    .insert(name.to_string(), FnProto::Code(bytecode));
                retrieve_fn_signatures(stmts, env);
            }
            _ => {}
        }
    }
}

fn emit_stmts(stmts: &[Statement], compiler: &mut Compiler) -> CompileResult<Option<usize>> {
    let mut last_target = None;
    for stmt in stmts {
        match stmt {
            Statement::VarDecl(var, _type, initializer) => {
                let locals = compiler
                    .locals
                    .last()
                    .ok_or_else(|| CompileError::LocalsStackUnderflow)?
                    .len();
                let init_val = if let Some(init_expr) = initializer {
                    let stk_var = emit_expr(init_expr, compiler)?;
                    compiler.target_stack[stk_var] = Target::Local(locals);
                    stk_var
                } else {
                    let stk_var = compiler.target_stack.len();
                    compiler.target_stack.push(Target::Local(locals));
                    stk_var
                };
                let locals = compiler
                    .locals
                    .last_mut()
                    .ok_or_else(|| CompileError::LocalsStackUnderflow)?;
                locals.push(LocalVar {
                    name: var.to_string(),
                    stack_idx: init_val,
                });
                dbg_println!("Locals: {:?}", compiler.locals);
            }
            Statement::FnDecl {
                name, args, stmts, ..
            } => {
                dbg_println!("FnDecl: Args: {:?}", args);
                let a_args = args
                    .iter()
                    .enumerate()
                    .map(|(idx, arg)| {
                        // The 0th index is used for function name / return value, so the args start with 1.
                        let target = idx + 1;
                        let local = LocalVar {
                            name: arg.name.to_owned(),
                            stack_idx: target,
                        };
                        compiler.target_stack.push(Target::Local(target));
                        local
                    })
                    .collect();
                let fn_args = args
                    .iter()
                    .map(|arg| {
                        let init = if let Some(ref init) = arg.init {
                            // Run the interpreter to fold the constant expression into a value.
                            // Note that the interpreter has an empty context, so it cannot access any
                            // global variables or user defined functions.
                            match eval(init, &mut EvalContext::new())
                                .map_err(CompileError::EvalError)?
                            {
                                RunResult::Yield(val) => Some(val),
                                _ => return Err(CompileError::DisallowedBreak),
                            }
                        } else {
                            None
                        };
                        Ok(BytecodeArg::new(arg.name.to_owned(), init))
                    })
                    .collect::<Result<_, _>>()?;
                dbg_println!("FnDecl actual args: {:?} fn_args: {:?}", a_args, fn_args);
                let fun = compile_fn(&mut compiler.env, stmts, a_args, fn_args)?;
                compiler.env.functions.insert(name.to_string(), fun);
            }
            Statement::Expression(ref ex) => {
                last_target = Some(emit_expr(ex, compiler)?);
            }
            Statement::Loop(stmts) => {
                let inst_loop_start = compiler.bytecode.instructions.len();
                last_target = emit_stmts(stmts, compiler)?;
                compiler
                    .bytecode
                    .push_inst(OpCode::Jmp, 0, inst_loop_start as u16);
                compiler.fixup_breaks();
            }
            Statement::While(cond, stmts) => {
                let inst_loop_start = compiler.bytecode.instructions.len();
                let stk_cond = emit_expr(cond, compiler)?;
                let inst_break = compiler.bytecode.push_inst(OpCode::Jf, stk_cond as u8, 0);
                last_target = emit_stmts(stmts, compiler)?;
                compiler
                    .bytecode
                    .push_inst(OpCode::Jmp, 0, inst_loop_start as u16);
                compiler.bytecode.instructions[inst_break].arg1 =
                    compiler.bytecode.instructions.len() as u16;
                compiler.fixup_breaks();
            }
            Statement::For(iter, from, to, stmts) => {
                let stk_from = emit_expr(from, compiler)?;
                let stk_to = emit_expr(to, compiler)?;
                let local_iter = compiler
                    .locals
                    .last()
                    .ok_or_else(|| CompileError::LocalsStackUnderflow)?
                    .len();
                let stk_check = compiler.target_stack.len();

                // stack: [stk_from, stk_to, stk_check]
                //   where stk_from is the starting variable being incremented until stk_to,
                //         stk_to is the end value
                //     and stk_check is the value to store the result of comparison

                let inst_loop_start = compiler.bytecode.instructions.len();
                compiler
                    .locals
                    .last_mut()
                    .ok_or_else(|| CompileError::LocalsStackUnderflow)?
                    .push(LocalVar {
                        name: iter.to_string(),
                        stack_idx: stk_from,
                    });
                compiler.target_stack[stk_from] = Target::Local(local_iter);
                compiler.target_stack.push(Target::None);
                compiler
                    .bytecode
                    .push_inst(OpCode::Move, stk_from as u8, stk_check as u16);
                compiler
                    .bytecode
                    .push_inst(OpCode::Lt, stk_check as u8, stk_to as u16);
                let inst_break = compiler.bytecode.push_inst(OpCode::Jf, stk_check as u8, 0);
                last_target = emit_stmts(stmts, compiler)?;
                compiler.bytecode.push_inst(OpCode::Incr, stk_from as u8, 0);
                compiler
                    .bytecode
                    .push_inst(OpCode::Jmp, 0, inst_loop_start as u16);
                compiler.bytecode.instructions[inst_break].arg1 =
                    compiler.bytecode.instructions.len() as u16;
                compiler.fixup_breaks();
            }
            Statement::Break => {
                let break_ip = compiler.bytecode.instructions.len();
                compiler.bytecode.push_inst(OpCode::Jmp, 0, 0);
                compiler.break_ips.push(break_ip);
            }
            Statement::Comment(_) => (),
        }
    }
    Ok(last_target)
}

fn emit_expr(expr: &Expression, compiler: &mut Compiler) -> CompileResult<usize> {
    match &expr.expr {
        ExprEnum::NumLiteral(val) => Ok(compiler.find_or_create_literal(val)),
        ExprEnum::StrLiteral(val) => Ok(compiler.find_or_create_literal(&Value::Str(val.clone()))),
        ExprEnum::ArrLiteral(val) => {
            let mut ctx = EvalContext::new();
            let val = Value::Array(Rc::new(RefCell::new(ArrayInt {
                type_decl: TypeDecl::Any,
                values: val
                    .iter()
                    .map(|v| {
                        if let RunResult::Yield(y) =
                            eval(v, &mut ctx).map_err(CompileError::EvalError)?
                        {
                            Ok(y)
                        } else {
                            Err(CompileError::BreakInArrayLiteral)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            })));
            Ok(compiler.find_or_create_literal(&val))
        }
        ExprEnum::Variable(str) => {
            let local = compiler.locals.iter().rev().fold(None, |acc, rhs| {
                if acc.is_some() {
                    acc
                } else {
                    rhs.iter().rev().find(|lo| lo.name == **str)
                }
            });

            if let Some(local) = local {
                return Ok(local.stack_idx);
            } else {
                return Err(CompileError::VarNotFound(str.to_string()));
            }
        }
        ExprEnum::Cast(ex, decl) => {
            let val = emit_expr(ex, compiler)?;
            // Make a copy of the value to avoid overwriting original variable
            let val_copy = compiler.target_stack.len();
            compiler
                .bytecode
                .push_inst(OpCode::Move, val as u8, val_copy as u16);
            compiler.target_stack.push(Target::None);
            let mut decl_buf = [0u8; std::mem::size_of::<i64>()];
            decl.serialize(&mut std::io::Cursor::new(&mut decl_buf[..]))?;
            let decl_stk =
                compiler.find_or_create_literal(&Value::I64(i64::from_le_bytes(decl_buf)));
            compiler
                .bytecode
                .push_inst(OpCode::Cast, val_copy as u8, decl_stk as u16);
            Ok(val_copy)
        }
        ExprEnum::Not(val) => {
            let val = emit_expr(val, compiler)?;
            compiler.bytecode.push_inst(OpCode::Not, val as u8, 0);
            Ok(val)
        }
        ExprEnum::BitNot(val) => {
            let val = emit_expr(val, compiler)?;
            compiler.bytecode.push_inst(OpCode::BitNot, val as u8, 0);
            Ok(val)
        }
        ExprEnum::Add(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Add, lhs, rhs)?),
        ExprEnum::Sub(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Sub, lhs, rhs)?),
        ExprEnum::Mult(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Mul, lhs, rhs)?),
        ExprEnum::Div(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Div, lhs, rhs)?),
        ExprEnum::VarAssign(lhs, rhs) => {
            let lhs_result = emit_expr(lhs, compiler)?;
            let rhs_result = emit_expr(rhs, compiler)?;
            compiler.bytecode.instructions.push(Instruction {
                op: OpCode::Move,
                arg0: rhs_result as u8,
                arg1: lhs_result as u16,
            });
            Ok(lhs_result)
        }
        ExprEnum::FnInvoke(fname, argss) => {
            let default_args = {
                let fun = compiler
                    .env
                    .functions
                    .get(*fname)
                    .ok_or_else(|| CompileError::FnNotFound(fname.to_string()))?;

                let fn_args = fun.args();

                if argss.len() <= fn_args.len() {
                    let fn_args = fn_args[argss.len()..].to_vec();

                    fn_args
                        .into_iter()
                        .filter_map(|arg| arg.init.as_ref().map(|init| init.clone()))
                        .collect()
                } else {
                    vec![]
                }
            };

            // Function arguments have value semantics, even if it was an array element.
            // Unless we emit `Deref` here, we might leave a reference in the stack that can be
            // accidentally overwritten. I'm not sure this is the best way to avoid it.
            let mut unnamed_args = argss
                .iter()
                .filter(|v| v.name.is_none())
                .map(|v| emit_rvalue(&v.expr, compiler))
                .collect::<Result<Vec<_>, _>>()?;
            unnamed_args.extend(
                default_args
                    .into_iter()
                    .map(|v| compiler.find_or_create_literal(&v)),
            );
            let named_args = argss
                .iter()
                .filter_map(|v| {
                    if let Some(name) = v.name {
                        match emit_rvalue(&v.expr, compiler) {
                            Ok(res) => Some(Ok((name, res))),
                            Err(e) => Some(Err(e)),
                        }
                    } else {
                        None
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let num_args = unnamed_args.len() + named_args.len();

            let stk_fname = compiler.find_or_create_literal(&Value::Str(fname.to_string()));

            let Some(fun) = compiler.env.functions.get(*fname) else {
                return Err(CompileError::FnNotFound(fname.to_string()));
            };

            let fn_args = fun.args();
            dbg_println!("FnProto found for: {fname}, args: {:?}", fn_args);

            // Prepare a buffer for actual arguments. It could be a mix of unnamed and named arguments.
            // Unnamed arguments are indexed from 0, while named arguments can appear at any index.
            let mut args = vec![None; fn_args.len().max(num_args)];

            // First, fill the buffer with unnamed arguments. Technically it could be more optimized by
            // allocating and initializing at the same time, but we do not pursue performance that much yet.
            for (arg, un_arg) in args.iter_mut().zip(unnamed_args.iter()) {
                *arg = Some(*un_arg);
            }

            // Second, fill the buffer with named arguments.
            for (_, arg) in named_args.iter().enumerate() {
                if let Some((f_idx, _)) = fn_args
                    .iter()
                    .enumerate()
                    .find(|(_, fn_arg)| fn_arg.name == *arg.0)
                {
                    args[f_idx] = Some(arg.1);
                }
            }

            // If a named argument is duplicate, you would have a hole in actual args.
            // Until we have the default parameter value, it would be a compile error.
            let args = args
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| CompileError::InsufficientNamedArgs)?;

            // Align arguments to the stack to prepare a call.
            for arg in args {
                let arg_target = compiler.target_stack.len();
                compiler.target_stack.push(Target::None);
                compiler
                    .bytecode
                    .push_inst(OpCode::Move, arg as u8, arg_target as u16);
            }

            // let func = compiler
            //     .functions
            //     .get(*str)
            //     .ok_or_else(|| format!("function {} is not defined.", str))?;

            compiler
                .bytecode
                .push_inst(OpCode::Call, num_args as u8, stk_fname as u16);
            compiler.target_stack.push(Target::None);
            Ok(stk_fname)
        }
        ExprEnum::ArrIndex(ex, args) => {
            let stk_ex = emit_expr(ex, compiler)?;
            let args = args
                .iter()
                .map(|v| emit_expr(v, compiler))
                .collect::<Result<Vec<_>, _>>()?;
            let arg = args[0];
            let arg = if matches!(compiler.target_stack[arg], Target::Local(_)) {
                // We move the local variable to another slot because our instructions are destructive
                let top = compiler.target_stack.len();
                compiler
                    .bytecode
                    .push_inst(OpCode::Move, arg as u8, top as u16);
                compiler.target_stack.push(Target::None);
                top
            } else {
                arg
            };
            compiler
                .bytecode
                .push_inst(OpCode::Get, stk_ex as u8, arg as u16);
            Ok(arg)
        }
        ExprEnum::LT(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Lt, lhs, rhs)?),
        ExprEnum::GT(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Gt, lhs, rhs)?),
        ExprEnum::BitAnd(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::BitAnd, lhs, rhs)?),
        ExprEnum::BitXor(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::BitXor, lhs, rhs)?),
        ExprEnum::BitOr(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::BitOr, lhs, rhs)?),
        ExprEnum::And(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::And, lhs, rhs)?),
        ExprEnum::Or(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Or, lhs, rhs)?),
        ExprEnum::Conditional(cond, true_branch, false_branch) => {
            let cond = emit_expr(cond, compiler)?;
            let cond_inst_idx = compiler.bytecode.instructions.len();
            compiler.bytecode.push_inst(OpCode::Jf, cond as u8, 0);
            let true_branch = emit_stmts(true_branch, compiler)?;
            if let Some(false_branch) = false_branch {
                let true_inst_idx = compiler.bytecode.instructions.len();
                compiler.bytecode.push_inst(OpCode::Jmp, 0, 0);
                compiler.bytecode.instructions[cond_inst_idx].arg1 =
                    compiler.bytecode.instructions.len() as u16;
                if let Some((false_branch, true_branch)) =
                    emit_stmts(false_branch, compiler)?.zip(true_branch)
                {
                    compiler.bytecode.push_inst(
                        OpCode::Move,
                        false_branch as u8,
                        true_branch as u16,
                    );
                }
                compiler.bytecode.instructions[true_inst_idx].arg1 =
                    compiler.bytecode.instructions.len() as u16;
            } else {
                compiler.bytecode.instructions[cond_inst_idx].arg1 =
                    compiler.bytecode.instructions.len() as u16;
            }
            Ok(true_branch.unwrap_or(0))
        }
        ExprEnum::Brace(stmts) => {
            compiler.locals.push(vec![]);
            let res = emit_stmts(stmts, compiler)?.unwrap_or(0);
            compiler.locals.pop();
            Ok(res)
        }
    }
}

fn emit_rvalue(ex: &Expression, compiler: &mut Compiler) -> CompileResult<usize> {
    let ret = emit_expr(ex, compiler)?;
    compiler.bytecode.push_inst(OpCode::Deref, ret as u8, 0);
    Ok(ret)
}

fn emit_binary_op(
    compiler: &mut Compiler,
    op: OpCode,
    lhs: &Expression,
    rhs: &Expression,
) -> CompileResult<usize> {
    let lhs = emit_expr(&lhs, compiler)?;
    let rhs = emit_expr(&rhs, compiler)?;
    let lhs = if matches!(compiler.target_stack[lhs], Target::Local(_)) {
        // We move the local variable to another slot because our instructions are destructive
        let top = compiler.target_stack.len();
        compiler
            .bytecode
            .push_inst(OpCode::Move, lhs as u8, top as u16);
        compiler.target_stack.push(Target::None);
        top
    } else {
        lhs
    };
    compiler.bytecode.instructions.push(Instruction {
        op,
        arg0: lhs as u8,
        arg1: rhs as u16,
    });
    Ok(lhs)
}
