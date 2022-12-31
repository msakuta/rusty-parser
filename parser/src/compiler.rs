use std::{
    cell::RefCell,
    collections::HashMap,
    io::{Read, Write},
    rc::Rc,
};

use crate::{
    interpreter::{
        eval, s_hex_string, s_len, s_print, s_push, s_type, EvalContext, EvalError, RunResult,
    },
    parser::{ArrayInt, ExprEnum, Expression, ReadError, Statement, TypeDecl, Value},
};

macro_rules! dbg_println {
    ($($rest:tt)*) => {{
        #[cfg(debug_assertions)]
        std::println!($($rest)*)
    }}
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum OpCode {
    LoadLiteral,
    /// Move values between stack elements, from arg0 to arg1.
    Move,
    /// Increment the operand arg0
    Incr,
    Add,
    Sub,
    Mul,
    Div,
    /// Bitwise and (&)
    BitAnd,
    /// Bitwise xor (^)
    BitXor,
    /// Bitwise or (|)
    BitOr,
    /// Logical and (&&)
    And,
    /// Logical or (||)
    Or,
    /// Logical not (!)
    Not,
    /// Bitwise not (~). Interestingly, Rust does not have dedicated bitwise not operator, because
    /// it has bool type. It can distinguish logical or bitwise operation by the operand type.
    /// However, we do not have bool type (yet), so we need a dedicated operator for bitwise not, like C.
    BitNot,
    /// Get an element of an array (or a table in the future) at arg0 with the key at arg1, and make a copy at arg1.
    /// Array elements are always Rc wrapped, so the user can assign into it.
    Get,
    /// If a value specified with arg0 in the stack is a reference (pointer), dereference it.
    Deref,
    /// Compare arg0 and arg1, sets result -1, 0 or 1 to arg0, meaning less, equal and more, respectively
    // Cmp,
    Lt,
    Gt,
    /// Unconditional jump to arg1.
    Jmp,
    /// Conditional jump. If arg0 is truthy, jump to arg1.
    Jt,
    /// Conditional jump. If arg0 is falthy, jump to arg1.
    Jf,
    /// Call a function with arg0 aruguments on the stack with index arg1.
    Call,
    /// Returns from current call stack.
    Ret,
}

macro_rules! impl_op_from {
    ($($op:ident),*) => {
        impl From<u8> for OpCode {
            #[allow(non_upper_case_globals)]
            fn from(o: u8) -> Self {
                $(const $op: u8 = OpCode::$op as u8;)*

                match o {
                    $($op => Self::$op,)*
                    _ => panic!("Opcode unrecognized!"),
                }
            }
        }
    }
}

impl_op_from!(
    LoadLiteral,
    Move,
    Incr,
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Not,
    Get,
    Deref,
    Lt,
    Gt,
    Jmp,
    Jt,
    Jf,
    Call,
    Ret
);

#[derive(Debug, Clone, Copy)]
pub struct Instruction {
    pub(crate) op: OpCode,
    pub(crate) arg0: u8,
    pub(crate) arg1: u16,
}

impl Instruction {
    fn new(op: OpCode, arg0: u8, arg1: u16) -> Self {
        Self { op, arg0, arg1 }
    }
    fn serialize(&self, writer: &mut impl Write) -> std::io::Result<()> {
        writer.write_all(&(self.op as u8).to_le_bytes())?;
        writer.write_all(&self.arg0.to_le_bytes())?;
        writer.write_all(&self.arg1.to_le_bytes())?;
        Ok(())
    }

    fn deserialize(reader: &mut impl Read) -> Result<Self, ReadError> {
        let mut op = [0u8; std::mem::size_of::<u8>()];
        reader.read_exact(&mut op)?;
        let mut arg0 = [0u8; std::mem::size_of::<u8>()];
        reader.read_exact(&mut arg0)?;
        let mut arg1 = [0u8; std::mem::size_of::<u16>()];
        reader.read_exact(&mut arg1)?;
        Ok(Self {
            op: u8::from_le_bytes(op).into(),
            arg0: u8::from_le_bytes(arg0),
            arg1: u16::from_le_bytes(arg1),
        })
    }
}

fn write_str(s: &str, writer: &mut impl Write) -> std::io::Result<()> {
    writer.write_all(&s.len().to_le_bytes())?;
    writer.write_all(&s.as_bytes())?;
    Ok(())
}

fn read_str(reader: &mut impl Read) -> Result<String, ReadError> {
    let mut len = [0u8; std::mem::size_of::<usize>()];
    reader.read_exact(&mut len)?;
    let len = usize::from_le_bytes(len);
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

pub(crate) enum FnProto {
    Code(FnBytecode),
    Native(Box<dyn Fn(&[Value]) -> Result<Value, EvalError>>),
}

pub struct Bytecode {
    pub(crate) functions: HashMap<String, FnProto>,
}

impl Bytecode {
    /// Add a user-application provided native function to this bytecode.
    pub fn add_ext_fn(
        &mut self,
        name: String,
        f: Box<dyn Fn(&[Value]) -> Result<Value, EvalError>>,
    ) {
        self.functions.insert(name, FnProto::Native(f));
    }

    /// Add standard common functions, such as `print`, `len` and `push`, to this bytecode.
    pub fn add_std_fn(&mut self) {
        self.add_ext_fn("print".to_string(), Box::new(s_print));
        self.add_ext_fn(
            "puts".to_string(),
            Box::new(|values: &[Value]| -> Result<Value, EvalError> {
                print!(
                    "{}",
                    values.iter().fold("".to_string(), |acc, cur: &Value| {
                        if acc.is_empty() {
                            cur.to_string()
                        } else {
                            acc + &cur.to_string()
                        }
                    })
                );
                Ok(Value::I64(0))
            }),
        );
        self.add_ext_fn("type".to_string(), Box::new(&s_type));
        self.add_ext_fn("len".to_string(), Box::new(s_len));
        self.add_ext_fn("push".to_string(), Box::new(s_push));
        self.add_ext_fn("hex_string".to_string(), Box::new(s_hex_string));
    }

    pub fn write(&self, writer: &mut impl Write) -> std::io::Result<()> {
        writer.write_all(&self.functions.len().to_le_bytes())?;
        for (fname, func) in self.functions.iter() {
            if let FnProto::Code(func) = func {
                write_str(fname, writer)?;
                func.write(writer)?;
            }
        }
        Ok(())
    }

    pub fn read(reader: &mut impl Read) -> Result<Self, ReadError> {
        let mut len = [0u8; std::mem::size_of::<usize>()];
        reader.read_exact(&mut len)?;
        let len = usize::from_le_bytes(len);
        let ret = Bytecode {
            functions: (0..len)
                .map(|_| -> Result<(String, FnProto), ReadError> {
                    Ok((read_str(reader)?, FnProto::Code(FnBytecode::read(reader)?)))
                })
                .collect::<Result<HashMap<_, _>, ReadError>>()?,
        };
        dbg_println!("loaded {} functions", ret.functions.len());
        let loaded_fn = ret.functions.iter().find(|(name, _)| *name == "").unwrap();
        if let FnProto::Code(ref _code) = loaded_fn.1 {
            dbg_println!("instructions: {:#?}", _code.instructions);
        }
        Ok(ret)
    }
}

#[derive(Debug, Clone)]
pub struct FnBytecode {
    pub(crate) literals: Vec<Value>,
    pub(crate) args: Vec<String>,
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) stack_size: usize,
}

impl FnBytecode {
    fn push_inst(&mut self, op: OpCode, arg0: u8, arg1: u16) -> usize {
        let ret = self.instructions.len();
        self.instructions.push(Instruction::new(op, arg0, arg1));
        ret
    }

    pub fn write(&self, writer: &mut impl Write) -> std::io::Result<()> {
        writer.write_all(&self.stack_size.to_le_bytes())?;
        writer.write_all(&self.literals.len().to_le_bytes())?;
        for literal in &self.literals {
            literal.serialize(writer)?;
        }
        writer.write_all(&self.args.len().to_le_bytes())?;
        for literal in &self.args {
            write_str(literal, writer)?;
        }
        writer.write_all(&self.instructions.len().to_le_bytes())?;
        for inst in &self.instructions {
            inst.serialize(writer)?;
        }
        Ok(())
    }

    pub fn read(reader: &mut impl Read) -> Result<Self, ReadError> {
        let mut stack_size = [0u8; std::mem::size_of::<usize>()];
        reader.read_exact(&mut stack_size)?;

        let mut literals = [0u8; std::mem::size_of::<usize>()];
        reader.read_exact(&mut literals)?;
        let literals = usize::from_le_bytes(literals);
        let literals = (0..literals)
            .map(|_| Value::deserialize(reader))
            .collect::<Result<Vec<_>, _>>()?;

        let mut args = [0u8; std::mem::size_of::<usize>()];
        reader.read_exact(&mut args)?;
        let args = usize::from_le_bytes(args);
        let args = (0..args)
            .map(|_| read_str(reader))
            .collect::<Result<Vec<_>, _>>()?;

        let mut instructions = [0u8; std::mem::size_of::<usize>()];
        reader.read_exact(&mut instructions)?;
        let instructions = usize::from_le_bytes(instructions);
        let instructions = (0..instructions)
            .map(|_| Instruction::deserialize(reader))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            literals,
            args,
            instructions,
            stack_size: usize::from_le_bytes(stack_size),
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum Target {
    None,
    /// Literal value with literal index.
    /// Right now whether the target is literal is not utilized, but we may use it for optimization.
    Literal(usize),
    /// If it is an allocated stack slot for a local variable, it will contain the index
    /// into locals array. We use it to keep track of which is local variable so that
    /// we won't destroy it by accident.
    /// **NOTE** that it is not a stack index.
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

struct Compiler {
    functions: HashMap<String, FnProto>,
    bytecode: FnBytecode,
    target_stack: Vec<Target>,
    locals: Vec<Vec<LocalVar>>,
    break_ips: Vec<usize>,
}

impl Compiler {
    fn new(args: Vec<LocalVar>) -> Self {
        Self {
            functions: HashMap::new(),
            bytecode: FnBytecode {
                literals: vec![],
                args: args.iter().map(|arg| arg.name.to_owned()).collect(),
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

pub fn compile<'src, 'ast>(stmts: &'ast [Statement<'src>]) -> Result<Bytecode, String> {
    let mut compiler = Compiler::new(vec![]);
    if let Some(last_target) = emit_stmts(stmts, &mut compiler)? {
        compiler
            .bytecode
            .instructions
            .push(Instruction::new(OpCode::Ret, 0, last_target as u16));
    }
    compiler.bytecode.stack_size = compiler.target_stack.len();

    dbg_println!("compile stack: {:#?}", compiler.bytecode);

    let mut functions = compiler.functions;
    functions.insert("".to_string(), FnProto::Code(compiler.bytecode));

    #[cfg(debug_assertions)]
    for fun in &functions {
        match fun.1 {
            FnProto::Code(code) => dbg_println!("fn {} -> {:?}", fun.0, code.instructions),
            _ => dbg_println!("fn {} -> <Native>", fun.0),
        }
    }
    Ok(Bytecode { functions })
}

fn compile_fn<'src, 'ast>(
    stmts: &'ast [Statement<'src>],
    args: Vec<LocalVar>,
) -> Result<Bytecode, String> {
    let mut compiler = Compiler::new(args);
    if let Some(last_target) = emit_stmts(stmts, &mut compiler)? {
        compiler
            .bytecode
            .instructions
            .push(Instruction::new(OpCode::Ret, 0, last_target as u16));
    }
    compiler.bytecode.stack_size = compiler.target_stack.len();

    dbg_println!("compile_fn stack: {:#?}", compiler.bytecode);

    let mut functions = compiler.functions;
    functions.insert("".to_string(), FnProto::Code(compiler.bytecode));
    Ok(Bytecode { functions })
}

fn emit_stmts(stmts: &[Statement], compiler: &mut Compiler) -> Result<Option<usize>, String> {
    let mut last_target = None;
    for stmt in stmts {
        match stmt {
            Statement::VarDecl(var, _type, initializer) => {
                let locals = compiler.locals.last().unwrap().len();
                let init_val = if let Some(init_expr) = initializer {
                    let stk_var = emit_expr(init_expr, compiler)?;
                    compiler.target_stack[stk_var] = Target::Local(locals);
                    stk_var
                } else {
                    let stk_var = compiler.target_stack.len();
                    compiler.target_stack.push(Target::Local(locals));
                    stk_var
                };
                let locals = compiler.locals.last_mut().unwrap();
                locals.push(LocalVar {
                    name: var.to_string(),
                    stack_idx: init_val,
                });
                dbg_println!("Locals: {:?}", compiler.locals);
            }
            Statement::FnDecl {
                name, args, stmts, ..
            } => {
                dbg_println!("Args: {:?}", args);
                let args = args
                    .iter()
                    .enumerate()
                    .map(|(idx, arg)| {
                        // The 0th index is used for function name / return value, so the args start with 1.
                        let target = idx + 1;
                        let local = LocalVar {
                            name: arg.0.to_owned(),
                            stack_idx: target,
                        };
                        compiler.target_stack.push(Target::Local(target));
                        local
                    })
                    .collect();
                dbg_println!("Locals: {:?}", args);
                let fun = compile_fn(stmts, args)?;
                compiler.functions.insert(
                    name.to_string(),
                    fun.functions
                        .into_iter()
                        .find(|(fname, _)| fname.is_empty())
                        .unwrap()
                        .1,
                );
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
                let local_iter = compiler.locals.last().unwrap().len();
                let stk_check = compiler.target_stack.len();

                // stack: [stk_from, stk_to, stk_check]
                //   where stk_from is the starting variable being incremented until stk_to,
                //         stk_to is the end value
                //     and stk_check is the value to store the result of comparison

                let inst_loop_start = compiler.bytecode.instructions.len();
                compiler.locals.last_mut().unwrap().push(LocalVar {
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

fn emit_expr(expr: &Expression, compiler: &mut Compiler) -> Result<usize, String> {
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
                        if let RunResult::Yield(y) = eval(v, &mut ctx)? {
                            Ok(y)
                        } else {
                            Err("Break in array literal not supported".to_string())
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
                return Err(format!("Variable {} not found in scope", str));
            }
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
        ExprEnum::Add(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Add, lhs, rhs)),
        ExprEnum::Sub(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Sub, lhs, rhs)),
        ExprEnum::Mult(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Mul, lhs, rhs)),
        ExprEnum::Div(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Div, lhs, rhs)),
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
            // Function arguments have value semantics, even if it was an array element.
            // Unless we emit `Deref` here, we might leave a reference in the stack that can be
            // accidentally overwritten. I'm not sure this is the best way to avoid it.
            let args = argss
                .iter()
                .map(|v| emit_rvalue(v, compiler))
                .collect::<Result<Vec<_>, _>>()?;
            let num_args = args.len();

            let stk_fname = compiler.find_or_create_literal(&Value::Str(fname.to_string()));

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
        ExprEnum::LT(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Lt, lhs, rhs)),
        ExprEnum::GT(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Gt, lhs, rhs)),
        ExprEnum::BitAnd(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::BitAnd, lhs, rhs)),
        ExprEnum::BitXor(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::BitXor, lhs, rhs)),
        ExprEnum::BitOr(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::BitOr, lhs, rhs)),
        ExprEnum::And(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::And, lhs, rhs)),
        ExprEnum::Or(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Or, lhs, rhs)),
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

fn emit_rvalue(ex: &Expression, compiler: &mut Compiler) -> Result<usize, String> {
    let ret = emit_expr(ex, compiler)?;
    compiler.bytecode.push_inst(OpCode::Deref, ret as u8, 0);
    Ok(ret)
}

fn emit_binary_op(
    compiler: &mut Compiler,
    op: OpCode,
    lhs: &Expression,
    rhs: &Expression,
) -> usize {
    let lhs = emit_expr(&lhs, compiler).unwrap();
    let rhs = emit_expr(&rhs, compiler).unwrap();
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
    lhs
}
