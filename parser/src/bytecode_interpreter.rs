use std::collections::HashMap;

use crate::{
    binary_op, binary_op_str, coerce_f64, truthy, Bytecode, EvalError, FnBytecode, FnProto, OpCode,
    Value,
};

pub fn interpret(bytecode: &Bytecode) -> Result<Value, EvalError> {
    if let Some(FnProto::Code(main)) = bytecode.functions.get("") {
        interpret_fn(main, &bytecode.functions)
    } else {
        Err("Main function not found".to_string())
    }
}

struct CallInfo<'a> {
    fun: &'a FnBytecode,
    ip: usize,
}

struct Vm {
    stack: Vec<Value>,
    stack_base: usize,
}

impl Vm {
    fn get(&self, idx: impl Into<usize>) -> &Value {
        &self.stack[self.stack_base + idx.into()]
    }

    fn set(&mut self, idx: impl Into<usize>, val: Value) {
        self.stack[self.stack_base + idx.into()] = val;
    }

    fn slice(&self, from: usize, to: usize) -> &[Value] {
        &self.stack[self.stack_base + from..self.stack_base + to]
    }
}

fn interpret_fn(
    bytecode: &FnBytecode,
    functions: &HashMap<String, FnProto>,
) -> Result<Value, EvalError> {
    println!("size inst: {}", std::mem::size_of::<crate::Instruction>());
    println!("size value: {}", std::mem::size_of::<Value>());
    println!("literals: {:?}", bytecode.literals);
    let mut vm = Vm {
        stack: vec![Value::I64(0); bytecode.stack_size],
        stack_base: 0,
    };
    let mut call_stack = vec![CallInfo {
        fun: &bytecode,
        ip: 0,
    }];

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

    while call_stack.last().unwrap().ip < bytecode.instructions.len() {
        let ip = call_stack.last().unwrap().ip;
        let inst = bytecode.instructions[ip];

        println!("inst[{ip}]: {inst:?}");

        match inst.op {
            OpCode::LoadLiteral => {
                vm.set(inst.arg1, bytecode.literals[inst.arg0 as usize].clone());
            }
            OpCode::Move => {
                let val = vm.get(inst.arg0);
                vm.set(inst.arg1, val.clone());
            }
            OpCode::Add => {
                let result = binary_op_str(
                    &vm.get(inst.arg0),
                    &vm.get(inst.arg1),
                    |lhs, rhs| lhs + rhs,
                    |lhs, rhs| lhs + rhs,
                    |lhs: &str, rhs: &str| Ok(lhs.to_string() + rhs),
                )?;
                vm.set(inst.arg0, result);
            }
            OpCode::Sub => {
                let result = binary_op(
                    &vm.get(inst.arg0),
                    &vm.get(inst.arg1),
                    |lhs, rhs| lhs - rhs,
                    |lhs, rhs| lhs - rhs,
                )?;
                vm.set(inst.arg0, result);
            }
            OpCode::Mul => {
                let result = binary_op(
                    &vm.get(inst.arg0),
                    &vm.get(inst.arg1),
                    |lhs, rhs| lhs * rhs,
                    |lhs, rhs| lhs * rhs,
                )?;
                vm.set(inst.arg0, result);
            }
            OpCode::Div => {
                let result = binary_op(
                    &vm.get(inst.arg0),
                    &vm.get(inst.arg1),
                    |lhs, rhs| lhs / rhs,
                    |lhs, rhs| lhs / rhs,
                )?;
                vm.set(inst.arg0, result);
            }
            OpCode::And => {
                let result = truthy(&vm.get(inst.arg0)) && truthy(&vm.get(inst.arg1));
                vm.set(inst.arg0, Value::I32(result as i32));
            }
            OpCode::Or => {
                let result = truthy(&vm.get(inst.arg0)) || truthy(&vm.get(inst.arg1));
                vm.set(inst.arg0, Value::I32(result as i32));
            }
            OpCode::Not => {
                let result = !truthy(&vm.get(inst.arg0));
                vm.set(inst.arg0, Value::I32(result as i32));
            }
            OpCode::Lt => {
                let result = compare_op(
                    &vm.get(inst.arg0),
                    &vm.get(inst.arg1),
                    |lhs, rhs| lhs.lt(&rhs),
                    |lhs, rhs| lhs.lt(&rhs),
                )?;
                vm.set(inst.arg0, Value::I64(result as i64));
            }
            OpCode::Gt => {
                let result = compare_op(
                    &vm.get(inst.arg0),
                    &vm.get(inst.arg1),
                    |lhs, rhs| lhs.gt(&rhs),
                    |lhs, rhs| lhs.gt(&rhs),
                )?;
                vm.set(inst.arg0, Value::I64(result as i64));
            }
            OpCode::Jmp => {
                println!("[{ip}] Jumping by Jmp to {}", inst.arg1);
                call_stack.last_mut().unwrap().ip = inst.arg1 as usize;
                continue;
            }
            OpCode::Jt => {
                if truthy(&vm.get(inst.arg0)) {
                    println!("[{ip}] Jumping by Jt to {}", inst.arg1);
                    call_stack.last_mut().unwrap().ip = inst.arg1 as usize;
                    continue;
                }
            }
            OpCode::Jf => {
                if !truthy(&vm.get(inst.arg0)) {
                    println!("[{ip}] Jumping by Jf to {}", inst.arg1);
                    call_stack.last_mut().unwrap().ip = inst.arg1 as usize;
                    continue;
                }
            }
            OpCode::Call => {
                let arg_name = vm.get(inst.arg1);
                let arg_name = if let Value::Str(s) = arg_name {
                    s
                } else {
                    return Err("Function can be only specified by a name (yet)".to_string());
                };
                let fun = functions.iter().find(|(fname, _)| *fname == arg_name);
                if let Some((_, fun)) = fun {
                    match fun {
                        FnProto::Code(fun) => {
                            vm.stack_base += bytecode.stack_size;
                            vm.stack
                                .resize(vm.stack_base + fun.stack_size, Value::default());
                            call_stack.push(CallInfo { fun, ip: 0 });
                        }
                        FnProto::Native(nat) => {
                            nat(&vm.slice(
                                inst.arg1 as usize + 1,
                                inst.arg1 as usize + 1 + inst.arg0 as usize,
                            ));
                        }
                    }
                } else {
                    return Err("Unknown function called".to_string());
                }
            }
            OpCode::Ret => {
                if let Some(ci) = call_stack.pop() {
                    if call_stack.is_empty() {
                        return Ok(vm.get(inst.arg1).clone());
                    } else {
                    }
                } else {
                    return Err("Call stack underflow!".to_string());
                }
            }
        }

        dump_stack(&vm.stack);

        call_stack.last_mut().unwrap().ip += 1;
    }

    println!("Final stack: {:?}", vm.stack);
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
