//! Bytecode interpreter, aka a Virtual Machine.

use std::collections::HashMap;

use crate::{
    bytecode::{Bytecode, FnBytecode, FnProto, OpCode},
    interpreter::{
        binary_op, binary_op_int, binary_op_str, coerce_f64, coerce_i64, coerce_type, truthy,
        EvalError,
    },
    type_decl::TypeDecl,
    Value,
};

macro_rules! dbg_println {
    ($($rest:tt)*) => {
        #[cfg(debug_assertions)]
        std::println!($($rest)*)
    }
}

pub fn interpret(bytecode: &Bytecode) -> Result<Value, EvalError> {
    if let Some(FnProto::Code(main)) = bytecode.functions.get("") {
        interpret_fn(main, &bytecode.functions)
    } else {
        Err(EvalError::NoMainFound)
    }
}

struct CallInfo<'a> {
    fun: &'a FnBytecode,
    ip: usize,
    stack_size: usize,
    stack_base: usize,
}

impl<'a> CallInfo<'a> {
    fn has_next_inst(&self) -> bool {
        self.ip < self.fun.instructions.len()
    }
}

struct Vm {
    stack: Vec<Value>,
    stack_base: usize,
}

impl Vm {
    fn get(&self, idx: impl Into<usize>) -> &Value {
        &self.stack[self.stack_base + idx.into()]
    }

    fn get_mut(&mut self, idx: impl Into<usize>) -> &mut Value {
        &mut self.stack[self.stack_base + idx.into()]
    }

    fn set(&mut self, idx: impl Into<usize>, val: Value) {
        self.stack[self.stack_base + idx.into()] = val;
    }

    fn slice(&self, from: usize, to: usize) -> &[Value] {
        &self.stack[self.stack_base + from..self.stack_base + to]
    }

    fn dump_stack(&self) {
        dbg_println!(
            "stack[{}..{}]: {}",
            self.stack_base,
            self.stack.len(),
            self.stack[self.stack_base..]
                .iter()
                .fold("".to_string(), |acc, cur: &Value| {
                    if acc.is_empty() {
                        cur.to_string()
                    } else {
                        acc + ", " + &cur.to_string()
                    }
                })
        );
    }
}

fn interpret_fn(
    bytecode: &FnBytecode,
    functions: &HashMap<String, FnProto>,
) -> Result<Value, EvalError> {
    dbg_println!("size inst: {}", std::mem::size_of::<crate::Instruction>());
    dbg_println!("size value: {}", std::mem::size_of::<Value>());
    dbg_println!(
        "size RefCell<Value>: {}",
        std::mem::size_of::<std::cell::RefCell<Value>>()
    );
    dbg_println!("size callInfo: {}", std::mem::size_of::<CallInfo>());
    dbg_println!("literals: {:?}", bytecode.literals);
    let mut vm = Vm {
        stack: vec![Value::I64(0); bytecode.stack_size],
        stack_base: 0,
    };
    let mut call_stack = vec![CallInfo {
        fun: &bytecode,
        ip: 0,
        stack_size: vm.stack.len(),
        stack_base: vm.stack_base,
    }];

    while call_stack.last().unwrap().has_next_inst() {
        let ci = call_stack.last().unwrap();
        let ip = ci.ip;
        let inst = ci.fun.instructions[ip];

        dbg_println!("inst[{ip}]: {inst:?}");

        match inst.op {
            OpCode::LoadLiteral => {
                vm.set(inst.arg1, ci.fun.literals[inst.arg0 as usize].clone());
            }
            OpCode::Move => {
                if let (Value::Array(lhs), Value::Array(rhs)) =
                    (vm.get(inst.arg0), vm.get(inst.arg1))
                {
                    if lhs as *const _ == rhs as *const _ {
                        println!("Self-assignment!");
                        call_stack.last_mut().unwrap().ip += 1;
                        continue;
                    }
                }
                let val = match vm.get(inst.arg0) {
                    Value::Ref(aref) => (*aref.borrow()).clone(),
                    Value::ArrayRef(aref, idx) => (*aref.borrow()).values[*idx].clone(),
                    v => v.clone(),
                };
                let target = vm.get_mut(inst.arg1);
                match target {
                    Value::Ref(vref) => {
                        vref.replace(val);
                    }
                    Value::ArrayRef(vref, idx) => vref.borrow_mut().values[*idx] = val,
                    _ => vm.set(inst.arg1, val),
                }
            }
            OpCode::Incr => {
                let val = vm.get_mut(inst.arg0);
                fn incr(val: &mut Value) -> Result<(), String> {
                    match val {
                        Value::I64(i) => *i += 1,
                        Value::I32(i) => *i += 1,
                        Value::F64(i) => *i += 1.,
                        Value::F32(i) => *i += 1.,
                        Value::Ref(r) => incr(&mut r.borrow_mut())?,
                        _ => {
                            return Err(format!(
                                "Attempt to increment non-numerical value {:?}",
                                val
                            ))
                        }
                    }
                    Ok(())
                }
                incr(val)?;
            }
            OpCode::Add => {
                let result = binary_op_str(
                    &vm.get(inst.arg0),
                    &vm.get(inst.arg1),
                    |lhs, rhs| Ok(lhs + rhs),
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
            OpCode::BitAnd => {
                let result =
                    binary_op_int(&vm.get(inst.arg0), &vm.get(inst.arg1), |lhs, rhs| lhs & rhs)?;
                vm.set(inst.arg0, result);
            }
            OpCode::BitXor => {
                let result =
                    binary_op_int(&vm.get(inst.arg0), &vm.get(inst.arg1), |lhs, rhs| lhs ^ rhs)?;
                vm.set(inst.arg0, result);
            }
            OpCode::BitOr => {
                let result =
                    binary_op_int(&vm.get(inst.arg0), &vm.get(inst.arg1), |lhs, rhs| lhs | rhs)?;
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
            OpCode::BitNot => {
                let val = vm.get(inst.arg0);
                let result = match val {
                    Value::I32(i) => Value::I32(!i),
                    Value::I64(i) => Value::I64(!i),
                    _ => return Err(EvalError::NonIntegerBitwise(format!("{val:?}"))),
                };
                vm.set(inst.arg0, result);
            }
            OpCode::Get => {
                let target_array = &vm.get(inst.arg0);
                let target_index = &vm.get(inst.arg1);
                let new_val = target_array.array_get_ref(coerce_i64(target_index)? as u64).map_err(|e| {
                    format!("Get instruction failed with {target_array:?} and {target_index:?}: {e:?}")
                })?;
                vm.set(inst.arg1, new_val);
            }
            OpCode::Deref => {
                let target = vm.get_mut(inst.arg0);
                match target {
                    Value::Ref(v) => {
                        let cloned = v.borrow().clone();
                        *target = cloned;
                    }
                    Value::ArrayRef(a, idx) => {
                        let a = a.borrow();
                        let cloned = a
                            .values
                            .get(*idx)
                            .ok_or_else(|| EvalError::ArrayOutOfBounds(*idx, a.values.len()))?
                            .clone();
                        drop(a);
                        *target = cloned;
                    }
                    _ => (),
                }
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
                dbg_println!("[{ip}] Jumping by Jmp to {}", inst.arg1);
                call_stack.last_mut().unwrap().ip = inst.arg1 as usize;
                continue;
            }
            OpCode::Jt => {
                if truthy(&vm.get(inst.arg0)) {
                    dbg_println!("[{ip}] Jumping by Jt to {}", inst.arg1);
                    call_stack.last_mut().unwrap().ip = inst.arg1 as usize;
                    continue;
                }
            }
            OpCode::Jf => {
                if !truthy(&vm.get(inst.arg0)) {
                    dbg_println!("[{ip}] Jumping by Jf to {}", inst.arg1);
                    call_stack.last_mut().unwrap().ip = inst.arg1 as usize;
                    continue;
                }
            }
            OpCode::Call => {
                let arg_name = vm.get(inst.arg1);
                let arg_name = if let Value::Str(s) = arg_name {
                    s
                } else {
                    return Err(EvalError::NonNameFnRef(format!("{arg_name:?}")));
                };
                let fun = functions.iter().find(|(fname, _)| *fname == arg_name);
                if let Some((_, fun)) = fun {
                    match fun {
                        FnProto::Code(fun) => {
                            dbg_println!("Calling code function with stack size (base:{}) + (fn: 1) + (params: {}) + (cur stack:{})", inst.arg1, inst.arg0, fun.stack_size);
                            // +1 for function name and return slot
                            vm.stack_base += inst.arg1 as usize;
                            vm.stack.resize(
                                vm.stack_base + inst.arg0 as usize + fun.stack_size + 1,
                                Value::default(),
                            );
                            call_stack.push(CallInfo {
                                fun,
                                ip: 0,
                                stack_size: vm.stack.len(),
                                stack_base: vm.stack_base,
                            });
                            continue;
                        }
                        FnProto::Native(nat) => {
                            let ret = nat(&vm.slice(
                                inst.arg1 as usize + 1,
                                inst.arg1 as usize + 1 + inst.arg0 as usize,
                            ));
                            vm.set(inst.arg1, ret?);
                        }
                    }
                } else {
                    return Err(EvalError::FnNotFound(arg_name.clone()));
                }
            }
            OpCode::Ret => {
                let retval = vm.stack_base + inst.arg1 as usize;
                if let Some(prev_ci) = call_stack.pop() {
                    if call_stack.is_empty() {
                        return Ok(vm.get(inst.arg1).clone());
                    } else {
                        let ci = call_stack.last().unwrap();
                        vm.stack_base = ci.stack_base;
                        vm.stack[prev_ci.stack_base] = vm.stack[retval].clone();
                        vm.stack.resize(ci.stack_size, Value::default());
                        vm.dump_stack();
                    }
                } else {
                    return Err(EvalError::CallStackUndeflow);
                }
            }
            OpCode::Cast => {
                let target_var = &vm.get(inst.arg0);
                let target_type = coerce_i64(vm.get(inst.arg1))
                    .map_err(|e| format!("arg1 of Cast was not a number: {e:?}"))?;
                let tt_buf = target_type.to_le_bytes();
                let tt = TypeDecl::deserialize(&mut &tt_buf[..])
                    .map_err(|e| format!("arg1 of Cast was not a TypeDecl: {e:?}"))?;
                let new_val = coerce_type(target_var, &tt)?;
                vm.set(inst.arg0, new_val);
            }
        }

        vm.dump_stack();

        call_stack.last_mut().unwrap().ip += 1;
    }

    dbg_println!("Final stack: {:?}", vm.stack);
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
        _ => return Err(EvalError::OpError(lhs.to_string(), rhs.to_string())),
    })
}

#[cfg(test)]
mod test;
