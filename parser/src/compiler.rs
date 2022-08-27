use std::io::{Write, Read};

use crate::{Statement, Expression, Value, ReadError};

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum OpCode {
    LoadLiteral,
    Add,
    Sub,
    Mul,
    Div,
}

impl From<u8> for OpCode {
    #[allow(non_upper_case_globals)]
    fn from(o: u8) -> Self {
        const LoadLiteral: u8 = OpCode::LoadLiteral as u8;
        const Add: u8 = OpCode::Add as u8;
        const Sub: u8 = OpCode::Sub as u8;
        const Mul: u8 = OpCode::Mul as u8;
        const Div: u8 = OpCode::Div as u8;
        match o {
            LoadLiteral => Self::LoadLiteral,
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

#[derive(Debug, Clone)]
pub struct Bytecode {
    pub(crate) literals: Vec<Value>,
    pub(crate) instructions: Vec<Instruction>,
}

impl Bytecode {
    pub fn write(&self, writer: &mut impl Write) -> std::io::Result<()> {
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
        let mut literals = [0u8; std::mem::size_of::<usize>()];
        reader.read_exact(&mut literals)?;
        let literals = usize::from_le_bytes(literals);
        let literals = (0..literals).map(|_| {
            Value::deserialize(reader)
        }).collect::<Result<Vec<_>, _>>()?;
        let mut instructions = [0u8; std::mem::size_of::<usize>()];
        reader.read_exact(&mut instructions)?;
        let instructions = usize::from_le_bytes(instructions);
        let instructions = (0..instructions).map(|_| {
            Instruction::deserialize(reader)
        }).collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            literals,
            instructions,
        })
    }
}

struct Target {
    literal: Option<usize>,
    local: Option<usize>,
}

struct LocalVar {
    name: String,
    stack_idx: usize,
}

pub fn compile<'src, 'ast>(
    stmts: &'ast Vec<Statement<'src>>,
) -> Result<Bytecode, String> {
    let mut ret = Bytecode {
        literals: vec![],
        instructions: vec![],
    };
    let mut target_stack = vec![];
    let mut locals = vec![];
    for stmt in stmts {
        match stmt {
            Statement::VarDecl(var, _type, initializer) => {
                let init_val = if let Some(init_expr) = initializer {
                    emit_expr(init_expr, &mut ret, &mut target_stack, &mut locals)?
                } else {
                    0
                };
                locals.push(LocalVar {
                    name: var.to_string(),
                    stack_idx: init_val,
                });
            }
            Statement::Expression(ref ex) => {
                emit_expr(ex, &mut ret, &mut target_stack, &mut locals)?;
            }
            _ => todo!(),
        }
    }
    Ok(ret)
}

fn add_literal(val: Value, bytecode: &mut Bytecode, target_stack: &mut Vec<Target>) -> usize {
    let literal = bytecode.literals.len();
    let target_idx = target_stack.len();
    bytecode.literals.push(val.clone());
    bytecode.instructions.push(Instruction {
        op: OpCode::LoadLiteral,
        arg0: literal as u8,
        arg1: target_idx as u16,
    });
    target_stack.push(Target { literal: Some(literal), local: None });
    target_idx
}

fn emit_expr(expr: &Expression, bytecode: &mut Bytecode, target_stack: &mut Vec<Target>, locals: &mut Vec<LocalVar>) -> Result<usize, String> {
    match expr {
        Expression::NumLiteral(val) => Ok(add_literal(val.clone(), bytecode, target_stack)),
        Expression::StrLiteral(val) => Ok(add_literal(Value::Str(val.clone()), bytecode, target_stack)),
        Expression::Variable(str) => {
            if let Some(local) = locals.iter().find(|lo| lo.name == **str) {
                return Ok(local.stack_idx);
            } else {
                return Err(format!("Variable {} not found in scope", str));
            }
        }
        Expression::Add(lhs, rhs) => {
            Ok(emit_binary_op(bytecode, target_stack, OpCode::Add, lhs, rhs, locals))
        }
        Expression::Sub(lhs, rhs) => {
            Ok(emit_binary_op(bytecode, target_stack, OpCode::Sub, lhs, rhs, locals))
        }
        Expression::Mult(lhs, rhs) => {
            Ok(emit_binary_op(bytecode, target_stack, OpCode::Mul, lhs, rhs, locals))
        }
        Expression::Div(lhs, rhs) => {
            Ok(emit_binary_op(bytecode, target_stack, OpCode::Div, lhs, rhs, locals))
        }
        _ => todo!(),
    }
}

fn emit_binary_op(bytecode: &mut Bytecode, target_stack: &mut Vec<Target>, op: OpCode, lhs: &Expression, rhs: &Expression, locals: &mut Vec<LocalVar>) -> usize {
    let lhs = emit_expr(&lhs, bytecode, target_stack, locals).unwrap();
    let rhs = emit_expr(&rhs, bytecode, target_stack, locals).unwrap();
    bytecode.instructions.push(Instruction {
        op,
        arg0: lhs as u8,
        arg1: rhs as u16,
    });
    let target_idx = target_stack.len();
    target_stack.push(Target { literal: None, local: None });
    target_idx
}