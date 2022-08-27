use crate::{Bytecode, OpCode, binary_op_str, EvalError, binary_op};


pub fn interpret(bytecode: &Bytecode) -> Result<(), EvalError> {

    let mut stack = vec![];
    let mut ip = 0;

    while ip < bytecode.instructions.len() {
        let inst = bytecode.instructions[ip];
        match inst.op {
            OpCode::LoadLiteral => {
                stack.push(bytecode.literals[inst.arg0 as usize].clone());
            }
            OpCode::Add => {
                let result = binary_op_str(&stack[inst.arg0 as usize], &stack[inst.arg1 as usize],
                    |lhs, rhs| lhs + rhs,
                    |lhs, rhs| lhs + rhs,
                    |lhs: &str, rhs: &str| Ok(lhs.to_string() + rhs),
                )?;
                stack.push(result);
            }
            OpCode::Sub => {
                let result = binary_op(&stack[inst.arg0 as usize], &stack[inst.arg1 as usize],
                    |lhs, rhs| lhs - rhs,
                    |lhs, rhs| lhs - rhs,
                )?;
                stack.push(result);
            }
            OpCode::Mul => {
                let result = binary_op(&stack[inst.arg0 as usize], &stack[inst.arg1 as usize],
                    |lhs, rhs| lhs * rhs,
                    |lhs, rhs| lhs * rhs,
                )?;
                stack.push(result);
            }
            OpCode::Div => {
                let result = binary_op(&stack[inst.arg0 as usize], &stack[inst.arg1 as usize],
                    |lhs, rhs| lhs / rhs,
                    |lhs, rhs| lhs / rhs,
                )?;
                stack.push(result);
            }
        }
        ip += 1;
    }

    println!("Final stack: {:?}", stack);
    Ok(())
}