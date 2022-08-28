use std::{
    collections::HashMap,
    io::{Read, Write},
};

use crate::{Expression, ReadError, Statement, Value};

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum OpCode {
    LoadLiteral,
    /// Move values between stack elements, from arg0 to arg1.
    Move,
    Add,
    Sub,
    Mul,
    Div,
    /// Logical and (&&)
    And,
    /// Logical or (||)
    Or,
    /// Logical not (!)
    Not,
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

impl From<u8> for OpCode {
    #[allow(non_upper_case_globals)]
    fn from(o: u8) -> Self {
        const LoadLiteral: u8 = OpCode::LoadLiteral as u8;
        const Move: u8 = OpCode::Move as u8;
        const Add: u8 = OpCode::Add as u8;
        const Sub: u8 = OpCode::Sub as u8;
        const Mul: u8 = OpCode::Mul as u8;
        const Div: u8 = OpCode::Div as u8;
        match o {
            LoadLiteral => Self::LoadLiteral,
            Move => Self::Move,
            Add => Self::Add,
            Sub => Self::Sub,
            Mul => Self::Mul,
            Div => Self::Div,
            _ => panic!("Opcode unrecognized!"),
        }
    }
}

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
    Native(Box<dyn Fn(&[Value])>),
}

pub struct Bytecode {
    pub(crate) functions: HashMap<String, FnProto>,
}

impl Bytecode {
    pub fn add_ext_fn(&mut self, name: String, f: Box<dyn Fn(&[Value])>) {
        self.functions.insert(name, FnProto::Native(f));
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
        Ok(Bytecode {
            functions: (0..len)
                .map(|i| -> Result<(String, FnProto), ReadError> {
                    Ok((read_str(reader)?, FnProto::Code(FnBytecode::read(reader)?)))
                })
                .collect::<Result<HashMap<_, _>, ReadError>>()?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct FnBytecode {
    pub(crate) literals: Vec<Value>,
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) stack_size: usize,
}

impl FnBytecode {
    pub fn write(&self, writer: &mut impl Write) -> std::io::Result<()> {
        writer.write_all(&self.stack_size.to_le_bytes())?;
        writer.write_all(&self.literals.len().to_le_bytes())?;
        for literal in &self.literals {
            literal.serialize(writer)?;
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
        let mut instructions = [0u8; std::mem::size_of::<usize>()];
        reader.read_exact(&mut instructions)?;
        let instructions = usize::from_le_bytes(instructions);
        let instructions = (0..instructions)
            .map(|_| Instruction::deserialize(reader))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            literals,
            instructions,
            stack_size: usize::from_le_bytes(stack_size),
        })
    }
}

struct Target {
    literal: Option<usize>,
    local: Option<usize>,
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
    stack_base: usize,
    locals: Vec<Vec<LocalVar>>,
}

impl Compiler {
    fn new() -> Self {
        Self {
            functions: HashMap::new(),
            bytecode: FnBytecode {
                literals: vec![],
                instructions: vec![],
                stack_size: 0,
            },
            target_stack: vec![],
            stack_base: 0,
            locals: vec![vec![]],
        }
    }
}

pub fn compile<'src, 'ast>(stmts: &'ast [Statement<'src>]) -> Result<Bytecode, String> {
    let mut compiler = Compiler::new();
    if let Some(last_target) = emit_stmts(stmts, &mut compiler)? {
        compiler
            .bytecode
            .instructions
            .push(Instruction::new(OpCode::Ret, 0, last_target as u16));
    }
    compiler.bytecode.stack_size = compiler.target_stack.len();
    let mut functions = compiler.functions;
    functions.insert("".to_string(), FnProto::Code(compiler.bytecode));
    Ok(Bytecode { functions })
}

fn emit_stmts(stmts: &[Statement], compiler: &mut Compiler) -> Result<Option<usize>, String> {
    let mut last_target = None;
    for stmt in stmts {
        match stmt {
            Statement::VarDecl(var, _type, initializer) => {
                let init_val = if let Some(init_expr) = initializer {
                    emit_expr(init_expr, compiler)?
                } else {
                    0
                };
                compiler.locals.last_mut().unwrap().push(LocalVar {
                    name: var.to_string(),
                    stack_idx: init_val,
                });
            }
            Statement::Expression(ref ex) => {
                last_target = Some(emit_expr(ex, compiler)?);
            }
            _ => todo!(),
        }
    }
    Ok(last_target)
}

fn add_literal(val: Value, compiler: &mut Compiler) -> usize {
    let bytecode = &mut compiler.bytecode;
    let literal = bytecode.literals.len();
    let target_idx = compiler.target_stack.len();
    bytecode.literals.push(val.clone());
    bytecode.instructions.push(Instruction::new(
        OpCode::LoadLiteral,
        literal as u8,
        target_idx as u16,
    ));
    compiler.target_stack.push(Target {
        literal: Some(literal),
        local: None,
    });
    target_idx
}

fn emit_expr(expr: &Expression, compiler: &mut Compiler) -> Result<usize, String> {
    match expr {
        Expression::NumLiteral(val) => Ok(add_literal(val.clone(), compiler)),
        Expression::StrLiteral(val) => Ok(add_literal(Value::Str(val.clone()), compiler)),
        Expression::Variable(str) => {
            // if compiler.locals.len() >= 1 {
            //     println!("locals: {:?}", compiler.locals);
            // }
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
        Expression::Add(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Add, lhs, rhs)),
        Expression::Sub(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Sub, lhs, rhs)),
        Expression::Mult(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Mul, lhs, rhs)),
        Expression::Div(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Div, lhs, rhs)),
        Expression::VarAssign(lhs, rhs) => {
            let lhs_result = emit_expr(lhs, compiler)?;
            let rhs_result = emit_expr(rhs, compiler)?;
            compiler.bytecode.instructions.push(Instruction {
                op: OpCode::Move,
                arg0: rhs_result as u8,
                arg1: lhs_result as u16,
            });
            Ok(lhs_result)
        }
        Expression::FnInvoke(fname, args) => {
            let args = args
                .iter()
                .map(|v| emit_expr(v, compiler))
                .collect::<Result<Vec<_>, _>>()?;

            let num_args = args.len();

            let fname_target = compiler.target_stack.len();
            let fname_literal = compiler.bytecode.literals.len();
            compiler.target_stack.push(Target {
                literal: Some(fname_literal),
                local: None,
            });
            compiler
                .bytecode
                .literals
                .push(Value::Str(fname.to_string()));
            compiler.bytecode.instructions.push(Instruction::new(
                OpCode::LoadLiteral,
                fname_literal as u8,
                fname_target as u16,
            ));

            // Align arguments to the stack to prepare a call.
            for arg in args {
                let arg_target = compiler.target_stack.len();
                compiler.target_stack.push(Target {
                    literal: None,
                    local: None,
                });
                compiler.bytecode.instructions.push(Instruction::new(
                    OpCode::Move,
                    arg as u8,
                    arg_target as u16,
                ));
            }

            // let func = compiler
            //     .functions
            //     .get(*str)
            //     .ok_or_else(|| format!("function {} is not defined.", str))?;

            compiler.bytecode.instructions.push(Instruction::new(
                OpCode::Call,
                num_args as u8,
                fname_target as u16,
            ));
            compiler.target_stack.push(Target {
                literal: None,
                local: None,
            });
            Ok(fname_target + num_args + 1)
        }
        Expression::LT(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Lt, lhs, rhs)),
        Expression::GT(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Gt, lhs, rhs)),
        Expression::Not(val) => {
            let val = emit_expr(val, compiler)?;
            compiler
                .bytecode
                .instructions
                .push(Instruction::new(OpCode::Not, val as u8, 0));
            Ok(val)
        }
        Expression::And(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::And, lhs, rhs)),
        Expression::Or(lhs, rhs) => Ok(emit_binary_op(compiler, OpCode::Or, lhs, rhs)),
        Expression::Conditional(cond, true_branch, false_branch) => {
            let cond = emit_expr(cond, compiler)?;
            let cond_inst_idx = compiler.bytecode.instructions.len();
            compiler
                .bytecode
                .instructions
                .push(Instruction::new(OpCode::Jf, cond as u8, 0));
            let true_branch = emit_stmts(true_branch, compiler)?;
            if let Some(false_branch) = false_branch {
                let true_inst_idx = compiler.bytecode.instructions.len();
                compiler
                    .bytecode
                    .instructions
                    .push(Instruction::new(OpCode::Jmp, 0, 0));
                compiler.bytecode.instructions[cond_inst_idx].arg1 =
                    compiler.bytecode.instructions.len() as u16;
                if let Some((false_branch, true_branch)) =
                    emit_stmts(false_branch, compiler)?.zip(true_branch)
                {
                    compiler.bytecode.instructions.push(Instruction::new(
                        OpCode::Move,
                        true_branch as u8,
                        false_branch as u16,
                    ));
                }
                compiler.bytecode.instructions[true_inst_idx].arg1 =
                    compiler.bytecode.instructions.len() as u16;
            } else {
                compiler.bytecode.instructions[cond_inst_idx].arg1 =
                    compiler.bytecode.instructions.len() as u16;
            }
            Ok(true_branch.unwrap_or(0))
        }
        Expression::Brace(stmts) => {
            compiler.locals.push(vec![]);
            let res = emit_stmts(stmts, compiler)?.unwrap_or(0);
            compiler.locals.pop();
            Ok(res)
        }
        _ => todo!(),
    }
}

fn emit_binary_op(
    compiler: &mut Compiler,
    op: OpCode,
    lhs: &Expression,
    rhs: &Expression,
) -> usize {
    let lhs = emit_expr(&lhs, compiler).unwrap();
    let rhs = emit_expr(&rhs, compiler).unwrap();
    compiler.bytecode.instructions.push(Instruction {
        op,
        arg0: lhs as u8,
        arg1: rhs as u16,
    });
    compiler.target_stack[lhs] = Target {
        literal: None,
        local: None,
    };
    lhs
}
