use crate::{Statement, EvalContext, RunResult, EvalError, Expression, Value};

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum OpCode {
    LoadLiteral,
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy)]
pub struct Instruction {
    pub(crate) op: OpCode,
    pub(crate) arg0: u8,
    pub(crate) arg1: u16,
}

#[derive(Debug, Clone)]
pub struct Bytecode {
    pub(crate) literals: Vec<Value>,
    pub(crate) instructions: Vec<Instruction>,
}

struct Target {
    literal: Option<usize>,
}

pub fn compile<'src, 'ast>(
    stmts: &'ast Vec<Statement<'src>>,
    ctx: &mut EvalContext<'src, 'ast, '_, '_>,
) -> Result<Bytecode, EvalError> {
    let mut ret = Bytecode {
        literals: vec![],
        instructions: vec![],
    };
    let mut target_stack = vec![];
    for stmt in stmts {
        match stmt {
            Statement::Expression(ref ex) => {
                emit_expr(ex, &mut ret, &mut target_stack);
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
    target_stack.push(Target { literal: Some(literal) });
    target_idx
}

fn emit_expr(expr: &Expression, bytecode: &mut Bytecode, target_stack: &mut Vec<Target>) -> Option<usize> {
    match expr {
        Expression::NumLiteral(val) => Some(add_literal(val.clone(), bytecode, target_stack)),
        Expression::StrLiteral(val) => Some(add_literal(Value::Str(val.clone()), bytecode, target_stack)),
        Expression::Add(lhs, rhs) => {
            Some(emit_binary_op(bytecode, target_stack, OpCode::Add, lhs, rhs))
        }
        Expression::Sub(lhs, rhs) => {
            Some(emit_binary_op(bytecode, target_stack, OpCode::Sub, lhs, rhs))
        }
        Expression::Mult(lhs, rhs) => {
            Some(emit_binary_op(bytecode, target_stack, OpCode::Mul, lhs, rhs))
        }
        Expression::Div(lhs, rhs) => {
            Some(emit_binary_op(bytecode, target_stack, OpCode::Div, lhs, rhs))
        }
        _ => todo!(),
    }
}

fn emit_binary_op(bytecode: &mut Bytecode, target_stack: &mut Vec<Target>, op: OpCode, lhs: &Expression, rhs: &Expression) -> usize {
    let lhs = emit_expr(&lhs, bytecode, target_stack).unwrap();
    let rhs = emit_expr(&rhs, bytecode, target_stack).unwrap();
    bytecode.instructions.push(Instruction {
        op,
        arg0: lhs as u8,
        arg1: rhs as u16,
    });
    let target_idx = target_stack.len();
    target_stack.push(Target { literal: None });
    target_idx
}