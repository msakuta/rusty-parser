use std::cmp::Ordering;

use crate::{binary_op, binary_op_str, coerce_f64, truthy, Bytecode, EvalError, OpCode, Value};

pub fn interpret(bytecode: &Bytecode) -> Result<Value, EvalError> {
    println!("size inst: {}", std::mem::size_of::<crate::Instruction>());
    println!("size value: {}", std::mem::size_of::<Value>());
    let mut stack = vec![Value::I64(0); bytecode.stack_size];
    let mut ip = 0;

    let dump_stack = |stack: &[Value]| {
        println!(
            "stack[{}]: {}",
            stack.len(),
            stack.iter().fold("".to_string(), |acc, cur: &Value| {
                if acc.is_empty() {
                    cur.to_string()
                } else {
                    acc + ", " + &cur.to_string()
                }
            })
        );
    };

    while ip < bytecode.instructions.len() {
        let inst = bytecode.instructions[ip];
        match inst.op {
            OpCode::LoadLiteral => {
                stack[inst.arg1 as usize] = bytecode.literals[inst.arg0 as usize].clone();
            }
            OpCode::Move => {
                stack[inst.arg1 as usize] = stack[inst.arg0 as usize].clone();
            }
            OpCode::Add => {
                let result = binary_op_str(
                    &stack[inst.arg0 as usize],
                    &stack[inst.arg1 as usize],
                    |lhs, rhs| lhs + rhs,
                    |lhs, rhs| lhs + rhs,
                    |lhs: &str, rhs: &str| Ok(lhs.to_string() + rhs),
                )?;
                stack[inst.arg0 as usize] = result;
            }
            OpCode::Sub => {
                let result = binary_op(
                    &stack[inst.arg0 as usize],
                    &stack[inst.arg1 as usize],
                    |lhs, rhs| lhs - rhs,
                    |lhs, rhs| lhs - rhs,
                )?;
                stack[inst.arg0 as usize] = result;
            }
            OpCode::Mul => {
                let result = binary_op(
                    &stack[inst.arg0 as usize],
                    &stack[inst.arg1 as usize],
                    |lhs, rhs| lhs * rhs,
                    |lhs, rhs| lhs * rhs,
                )?;
                stack[inst.arg0 as usize] = result;
            }
            OpCode::Div => {
                let result = binary_op(
                    &stack[inst.arg0 as usize],
                    &stack[inst.arg1 as usize],
                    |lhs, rhs| lhs / rhs,
                    |lhs, rhs| lhs / rhs,
                )?;
                stack[inst.arg0 as usize] = result;
            }
            OpCode::And => {
                let result =
                    truthy(&stack[inst.arg0 as usize]) && truthy(&stack[inst.arg1 as usize]);
                stack[inst.arg0 as usize] = Value::I32(result as i32);
            }
            OpCode::Or => {
                let result =
                    truthy(&stack[inst.arg0 as usize]) || truthy(&stack[inst.arg1 as usize]);
                stack[inst.arg0 as usize] = Value::I32(result as i32);
            }
            OpCode::Not => {
                let result = !truthy(&stack[inst.arg0 as usize]);
                stack[inst.arg0 as usize] = Value::I32(result as i32);
            }
            OpCode::Lt => {
                let result = compare_op(
                    &stack[inst.arg0 as usize],
                    &stack[inst.arg1 as usize],
                    |lhs, rhs| lhs.lt(&rhs),
                    |lhs, rhs| lhs.lt(&rhs),
                )?;
                stack[inst.arg0 as usize] = Value::I64(result as i64);
            }
            OpCode::Gt => {
                let result = compare_op(
                    &stack[inst.arg0 as usize],
                    &stack[inst.arg1 as usize],
                    |lhs, rhs| lhs.gt(&rhs),
                    |lhs, rhs| lhs.gt(&rhs),
                )?;
                stack[inst.arg0 as usize] = Value::I64(result as i64);
            }
            OpCode::Jmp => {
                println!("[{ip}] Jumping by Jmp to {}", inst.arg1);
                ip = inst.arg1 as usize;
                continue;
            }
            OpCode::Jt => {
                if truthy(&stack[inst.arg0 as usize]) {
                    println!("[{ip}] Jumping by Jt to {}", inst.arg1);
                    ip = inst.arg1 as usize;
                    continue;
                }
            }
            OpCode::Jf => {
                if !truthy(&stack[inst.arg0 as usize]) {
                    println!("[{ip}] Jumping by Jf to {}", inst.arg1);
                    ip = inst.arg1 as usize;
                    continue;
                }
            }
            OpCode::Ret => {
                return Ok(stack[inst.arg1 as usize].clone());
            }
        }

        dump_stack(&stack);

        ip += 1;
    }

    println!("Final stack: {:?}", stack);
    Ok(Value::I64(0))
}

fn compare_op(
    lhs: &Value,
    rhs: &Value,
    d: impl Fn(f64, f64) -> bool,
    i: impl Fn(i64, i64) -> bool,
) -> Result<bool, EvalError> {
    Ok(match (lhs.clone(), rhs.clone()) {
        (Value::F64(lhs), rhs) => d(lhs, coerce_f64(&rhs)?),
        (lhs, Value::F64(rhs)) => d(coerce_f64(&lhs)?, rhs),
        (Value::F32(lhs), rhs) => d(lhs as f64, coerce_f64(&rhs)?),
        (lhs, Value::F32(rhs)) => d(coerce_f64(&lhs)?, rhs as f64),
        (Value::I64(lhs), Value::I64(rhs)) => i(lhs, rhs),
        (Value::I64(lhs), Value::I32(rhs)) => i(lhs, rhs as i64),
        (Value::I32(lhs), Value::I64(rhs)) => i(lhs as i64, rhs),
        (Value::I32(lhs), Value::I32(rhs)) => i(lhs as i64, rhs as i64),
        _ => {
            return Err(format!(
                "Unsupported comparison between {:?} and {:?}",
                lhs, rhs
            ))
        }
    })
}

#[cfg(test)]
mod test;
