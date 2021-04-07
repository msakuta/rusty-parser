use crate::parser::*;
use std::{cell::RefCell, collections::HashMap, rc::Rc};

#[derive(Debug, PartialEq, Clone)]
pub enum RunResult {
    Yield(Value),
    Break,
}

type EvalError = String;

fn unwrap_deref(e: RunResult) -> RunResult {
    match &e {
        RunResult::Yield(Value::Ref(vref)) => {
            let r = vref.borrow();
            return RunResult::Yield(r.clone());
        }
        RunResult::Break => return RunResult::Break,
        _ => (),
    }
    e
}

macro_rules! unwrap_run {
    ($e:expr) => {
        match unwrap_deref($e) {
            RunResult::Yield(v) => v,
            RunResult::Break => return Ok(RunResult::Break),
        }
    };
}

fn binary_op_str(
    lhs: Value,
    rhs: Value,
    d: impl Fn(f64, f64) -> f64,
    i: impl Fn(i64, i64) -> i64,
    s: impl Fn(&str, &str) -> Result<String, EvalError>,
) -> Result<Value, EvalError> {
    Ok(match (lhs.clone(), rhs.clone()) {
        (Value::F64(lhs), rhs) => Value::F64(d(lhs, coerce_f64(&rhs))),
        (lhs, Value::F64(rhs)) => Value::F64(d(coerce_f64(&lhs), rhs)),
        (Value::F32(lhs), rhs) => Value::F32(d(lhs as f64, coerce_f64(&rhs)) as f32),
        (lhs, Value::F32(rhs)) => Value::F32(d(coerce_f64(&lhs), rhs as f64) as f32),
        (Value::I64(lhs), Value::I64(rhs)) => Value::I64(i(lhs, rhs)),
        (Value::I64(lhs), Value::I32(rhs)) => Value::I64(i(lhs, rhs as i64)),
        (Value::I32(lhs), Value::I64(rhs)) => Value::I64(i(lhs as i64, rhs)),
        (Value::I32(lhs), Value::I32(rhs)) => Value::I32(i(lhs as i64, rhs as i64) as i32),
        (Value::Str(lhs), Value::Str(rhs)) => Value::Str(s(&lhs, &rhs)?),
        _ => {
            return Err(format!(
                "Unsupported addition between {:?} and {:?}",
                lhs, rhs
            ))
        }
    })
}

fn binary_op(
    lhs: Value,
    rhs: Value,
    d: impl Fn(f64, f64) -> f64,
    i: impl Fn(i64, i64) -> i64,
) -> Result<Value, EvalError> {
    binary_op_str(lhs, rhs, d, i, |_lhs, _rhs| {
        Err("This operator is not supported for strings".to_string())
    })
}

fn truthy(a: &Value) -> bool {
    match a {
        Value::F64(v) => *v != 0.,
        Value::F32(v) => *v != 0.,
        Value::I64(v) => *v != 0,
        Value::I32(v) => *v != 0,
        Value::Ref(r) => truthy(&r.borrow()),
        _ => false,
    }
}

fn coerce_f64(a: &Value) -> f64 {
    match a {
        Value::F64(v) => *v as f64,
        Value::F32(v) => *v as f64,
        Value::I64(v) => *v as f64,
        Value::I32(v) => *v as f64,
        Value::Ref(r) => coerce_f64(&r.borrow()),
        _ => 0.,
    }
}

fn coerce_i64(a: &Value) -> Result<i64, EvalError> {
    Ok(match a {
        Value::F64(v) => *v as i64,
        Value::F32(v) => *v as i64,
        Value::I64(v) => *v as i64,
        Value::I32(v) => *v as i64,
        Value::Ref(r) => coerce_i64(&r.borrow())?,
        _ => 0,
    })
}

fn coerce_str(a: &Value) -> String {
    match a {
        Value::F64(v) => v.to_string(),
        Value::F32(v) => v.to_string(),
        Value::I64(v) => v.to_string(),
        Value::I32(v) => v.to_string(),
        Value::Str(v) => v.clone(),
        _ => panic!("Can't convert array to str"),
    }
}

fn _coerce_var(value: &Value, target: &Value) -> Result<Value, EvalError> {
    Ok(match target {
        Value::F64(_) => Value::F64(coerce_f64(value)),
        Value::F32(_) => Value::F32(coerce_f64(value) as f32),
        Value::I64(_) => Value::I64(coerce_i64(value)?),
        Value::I32(_) => Value::I32(coerce_i64(value)? as i32),
        Value::Str(_) => Value::Str(coerce_str(value)),
        Value::Array(inner_type, inner) => {
            if inner.len() == 0 {
                if let Value::Array(_, value_inner) = value {
                    if value_inner.len() == 0 {
                        return Ok(value.clone());
                    }
                }
                panic!("Cannot coerce type to empty array");
            } else {
                if let Value::Array(_, value_inner) = value {
                    Value::Array(
                        inner_type.clone(),
                        value_inner
                            .iter()
                            .map(|val| -> Result<_, String> {
                                Ok(Rc::new(RefCell::new(coerce_type(
                                    &val.borrow(),
                                    inner_type,
                                )?)))
                            })
                            .collect::<Result<_, _>>()?,
                    )
                } else {
                    panic!("Cannot coerce scalar to array");
                }
            }
        }
        // We usually don't care about coercion
        Value::Ref(_) => value.clone(),
    })
}

pub fn coerce_type(value: &Value, target: &TypeDecl) -> Result<Value, EvalError> {
    Ok(match target {
        TypeDecl::Any => value.clone(),
        TypeDecl::F64 => Value::F64(coerce_f64(value)),
        TypeDecl::F32 => Value::F32(coerce_f64(value) as f32),
        TypeDecl::I64 => Value::I64(coerce_i64(value)?),
        TypeDecl::I32 => Value::I32(coerce_i64(value)? as i32),
        TypeDecl::Str => Value::Str(coerce_str(value)),
        TypeDecl::Array(inner) => {
            if let Value::Array(_, value_inner) = value {
                Value::Array(
                    (**inner).clone(),
                    value_inner
                        .iter()
                        .map(|value_elem| -> Result<_, EvalError> {
                            Ok(Rc::new(RefCell::new(coerce_type(
                                &value_elem.borrow(),
                                inner,
                            )?)))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            } else {
                panic!(format!("Incompatible type to array! {:?}", value));
            }
        }
    })
}

fn eval<'a, 'b>(
    e: &'b Expression<'a>,
    ctx: &mut EvalContext<'a, 'b, '_, '_>,
) -> Result<RunResult, EvalError> {
    Ok(match e {
        Expression::NumLiteral(val) => RunResult::Yield(val.clone()),
        Expression::StrLiteral(val) => RunResult::Yield(Value::Str(val.clone())),
        Expression::ArrLiteral(val) => RunResult::Yield(Value::Array(
            TypeDecl::Any,
            val.iter()
                .map(|v| {
                    if let RunResult::Yield(y) = eval(v, ctx)? {
                        Ok(Rc::new(RefCell::new(y)))
                    } else {
                        Err("Break in array literal not supported".to_string())
                    }
                })
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Expression::Variable(str) => RunResult::Yield(Value::Ref(
            ctx.get_var_rc(str)
                .ok_or_else(|| format!("Variable {} not found in scope", str))?,
        )),
        Expression::VarAssign(lhs, rhs) => {
            let lhs_result = eval(lhs, ctx)?;
            let lhs_value = if let RunResult::Yield(Value::Ref(rc)) = lhs_result {
                rc
            } else {
                return Err(format!(
                    "We need variable reference on lhs to assign. Actually we got {:?}",
                    lhs_result
                ));
            };
            let rhs_value = unwrap_run!(eval(rhs, ctx)?);
            *lhs_value.borrow_mut() = rhs_value.clone();
            RunResult::Yield(rhs_value)
        }
        Expression::FnInvoke(str, args) => {
            let args = args
                .iter()
                .map(|v| eval(v, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            let mut subctx = EvalContext::push_stack(ctx);
            let func = ctx
                .get_fn(*str)
                .ok_or_else(|| format!("function {} is not defined.", str))?;
            match func {
                FuncDef::Code(func) => {
                    for (k, v) in func.args.iter().zip(&args) {
                        subctx.variables.borrow_mut().insert(
                            k.0,
                            Rc::new(RefCell::new(coerce_type(&unwrap_run!(v.clone()), &k.1)?)),
                        );
                    }
                    let run_result = run(func.stmts, &mut subctx)?;
                    match unwrap_deref(run_result) {
                        RunResult::Yield(v) => match &func.ret_type {
                            Some(ty) => RunResult::Yield(coerce_type(&v, ty)?),
                            None => RunResult::Yield(v),
                        },
                        RunResult::Break => panic!("break in function toplevel"),
                    }
                }
                FuncDef::Native(native) => RunResult::Yield(native(
                    &args
                        .into_iter()
                        .map(|e| match e {
                            RunResult::Yield(v) => v.clone(),
                            RunResult::Break => {
                                panic!("Break in function argument is not supported yet!")
                            }
                        })
                        .collect::<Vec<_>>(),
                )),
            }
        }
        Expression::ArrIndex(ex, args) => {
            let args = args
                .iter()
                .map(|v| eval(v, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            let arg0 = match unwrap_deref(args[0].clone()) {
                RunResult::Yield(v) => {
                    if let Value::I64(idx) = coerce_type(&v, &TypeDecl::I64)? {
                        idx as u64
                    } else {
                        panic!("Subscript type should be integer types");
                    }
                }
                RunResult::Break => {
                    return Ok(RunResult::Break);
                }
            };
            let result = unwrap_run!(eval(ex, ctx)?);
            RunResult::Yield(result.array_get_ref(arg0))
        }
        Expression::Not(val) => {
            RunResult::Yield(Value::I32(if truthy(&unwrap_run!(eval(val, ctx)?)) {
                0
            } else {
                1
            }))
        }
        Expression::Add(lhs, rhs) => {
            let res = RunResult::Yield(binary_op_str(
                unwrap_run!(eval(lhs, ctx)?),
                unwrap_run!(eval(rhs, ctx)?),
                |lhs, rhs| lhs + rhs,
                |lhs, rhs| lhs + rhs,
                |lhs: &str, rhs: &str| Ok(lhs.to_string() + rhs),
            )?);
            res
        }
        Expression::Sub(lhs, rhs) => RunResult::Yield(binary_op(
            unwrap_run!(eval(lhs, ctx)?),
            unwrap_run!(eval(rhs, ctx)?),
            |lhs, rhs| lhs - rhs,
            |lhs, rhs| lhs - rhs,
        )?),
        Expression::Mult(lhs, rhs) => RunResult::Yield(binary_op(
            unwrap_run!(eval(lhs, ctx)?),
            unwrap_run!(eval(rhs, ctx)?),
            |lhs, rhs| lhs * rhs,
            |lhs, rhs| lhs * rhs,
        )?),
        Expression::Div(lhs, rhs) => RunResult::Yield(binary_op(
            unwrap_run!(eval(lhs, ctx)?),
            unwrap_run!(eval(rhs, ctx)?),
            |lhs, rhs| lhs / rhs,
            |lhs, rhs| lhs / rhs,
        )?),
        Expression::LT(lhs, rhs) => RunResult::Yield(binary_op(
            unwrap_run!(eval(lhs, ctx)?),
            unwrap_run!(eval(rhs, ctx)?),
            |lhs, rhs| if lhs < rhs { 1. } else { 0. },
            |lhs, rhs| if lhs < rhs { 1 } else { 0 },
        )?),
        Expression::GT(lhs, rhs) => RunResult::Yield(binary_op(
            unwrap_run!(eval(lhs, ctx)?),
            unwrap_run!(eval(rhs, ctx)?),
            |lhs, rhs| if lhs > rhs { 1. } else { 0. },
            |lhs, rhs| if lhs > rhs { 1 } else { 0 },
        )?),
        Expression::And(lhs, rhs) => RunResult::Yield(Value::I32(
            if truthy(&unwrap_run!(eval(lhs, ctx)?)) && truthy(&unwrap_run!(eval(rhs, ctx)?)) {
                1
            } else {
                0
            },
        )),
        Expression::Or(lhs, rhs) => RunResult::Yield(Value::I32(
            if truthy(&unwrap_run!(eval(lhs, ctx)?)) || truthy(&unwrap_run!(eval(rhs, ctx)?)) {
                1
            } else {
                0
            },
        )),
        Expression::Conditional(cond, true_branch, false_branch) => {
            if truthy(&unwrap_run!(eval(cond, ctx)?)) {
                run(true_branch, ctx)?
            } else if let Some(ast) = false_branch {
                run(ast, ctx)?
            } else {
                RunResult::Yield(Value::I32(0))
            }
        }
        Expression::Brace(stmts) => {
            let mut subctx = EvalContext::push_stack(ctx);
            run(stmts, &mut subctx)?
        }
    })
}

fn s_print(vals: &[Value]) -> Value {
    println!("print:");
    fn print_inner(vals: &[Value]) {
        for val in vals {
            match val {
                Value::F64(val) => println!(" {}", val),
                Value::F32(val) => println!(" {}", val),
                Value::I64(val) => println!(" {}", val),
                Value::I32(val) => println!(" {}", val),
                Value::Str(val) => println!(" {}", val),
                Value::Array(_, val) => {
                    print!("[");
                    print_inner(&val.iter().map(|v| v.borrow().clone()).collect::<Vec<_>>());
                    print!("]");
                }
                Value::Ref(r) => {
                    print!("ref(");
                    print_inner(&[r.borrow().clone()]);
                    print!(")");
                }
            }
        }
    }
    print_inner(vals);
    print!("\n");
    Value::I32(0)
}

fn s_puts(vals: &[Value]) -> Value {
    fn puts_inner(vals: &[Value]) {
        for val in vals {
            match val {
                Value::F64(val) => print!("{}", val),
                Value::F32(val) => print!("{}", val),
                Value::I64(val) => print!("{}", val),
                Value::I32(val) => print!("{}", val),
                Value::Str(val) => print!("{}", val),
                Value::Array(_, val) => {
                    puts_inner(&val.iter().map(|v| v.borrow().clone()).collect::<Vec<_>>())
                }
                Value::Ref(r) => puts_inner(&[r.borrow().clone()]),
            }
        }
    }
    puts_inner(vals);
    Value::I32(0)
}

fn type_decl_to_str(t: &TypeDecl) -> String {
    match t {
        TypeDecl::Any => "any".to_string(),
        TypeDecl::F64 => "f64".to_string(),
        TypeDecl::F32 => "f32".to_string(),
        TypeDecl::I64 => "i64".to_string(),
        TypeDecl::I32 => "i32".to_string(),
        TypeDecl::Str => "str".to_string(),
        TypeDecl::Array(inner) => format!("[{}]", type_decl_to_str(inner)),
    }
}

fn s_type(vals: &[Value]) -> Value {
    fn type_str(val: &Value) -> String {
        match val {
            Value::F64(_) => "f64".to_string(),
            Value::F32(_) => "f32".to_string(),
            Value::I64(_) => "i64".to_string(),
            Value::I32(_) => "i32".to_string(),
            Value::Str(_) => "str".to_string(),
            Value::Array(inner, _) => format!("[{}]", type_decl_to_str(inner)),
            Value::Ref(r) => format!("ref[{}]", type_str(&r.borrow())),
        }
    }
    if let [val, ..] = vals {
        Value::Str(type_str(val))
    } else {
        Value::I32(0)
    }
}

fn s_len(vals: &[Value]) -> Value {
    if let [val, ..] = vals {
        Value::I64(val.array_len() as i64)
    } else {
        Value::I32(0)
    }
}

fn s_push(vals: &[Value]) -> Value {
    if let [arr, val, ..] = vals {
        match arr {
            Value::Ref(rc) => {
                rc.borrow_mut().array_push(val.clone());
                Value::I32(0)
            }
            _ => panic!("len() not supported other than arrays"),
        }
    } else {
        Value::I32(0)
    }
}

fn s_hex_string(vals: &[Value]) -> Value {
    if let [val, ..] = vals {
        match coerce_type(val, &TypeDecl::I64).unwrap() {
            Value::I64(i) => Value::Str(format!("{:02x}", i)),
            _ => panic!("hex_string() could not convert argument to i64"),
        }
    } else {
        Value::Str("".to_string())
    }
}

#[derive(Clone)]
pub struct FuncCode<'src, 'ast> {
    args: &'ast Vec<ArgDecl<'src>>,
    ret_type: Option<TypeDecl>,
    stmts: &'ast Vec<Statement<'src>>,
}

#[derive(Clone)]
pub enum FuncDef<'src, 'ast, 'native> {
    Code(FuncCode<'src, 'ast>),
    Native(&'native dyn Fn(&[Value]) -> Value),
}

/// A context stat for evaluating a script.
///
/// It has 4 lifetime arguments:
///  * the source code ('src)
///  * the AST ('ast),
///  * the native function code ('native) and
///  * the parent eval context ('ctx)
///
/// In general, they all can have different lifetimes. For example,
/// usually AST is created after the source.
#[derive(Clone)]
pub struct EvalContext<'src, 'ast, 'native, 'ctx> {
    /// RefCell to allow mutation in super context.
    /// Also, the inner values must be Rc of RefCell because a reference could be returned from
    /// a function so that the variable scope may have been ended.
    variables: RefCell<HashMap<&'src str, Rc<RefCell<Value>>>>,
    /// Function names are owned strings because it can be either from source or native.
    /// Unlike variables, functions cannot be overwritten in the outer scope, so it does not
    /// need to be wrapped in a RefCell.
    functions: HashMap<String, FuncDef<'src, 'ast, 'native>>,
    super_context: Option<&'ctx EvalContext<'src, 'ast, 'native, 'ctx>>,
}

impl<'src, 'ast, 'native, 'ctx> EvalContext<'src, 'ast, 'native, 'ctx> {
    pub fn new() -> Self {
        let mut functions = HashMap::new();
        functions.insert("print".to_string(), FuncDef::Native(&s_print));
        functions.insert("puts".to_string(), FuncDef::Native(&s_puts));
        functions.insert("type".to_string(), FuncDef::Native(&s_type));
        functions.insert("len".to_string(), FuncDef::Native(&s_len));
        functions.insert("push".to_string(), FuncDef::Native(&s_push));
        functions.insert("hex_string".to_string(), FuncDef::Native(&s_hex_string));
        Self {
            variables: RefCell::new(HashMap::new()),
            functions,
            super_context: None,
        }
    }

    pub fn set_fn(&mut self, name: &str, fun: FuncDef<'src, 'ast, 'native>) {
        self.functions.insert(name.to_string(), fun);
    }

    fn push_stack(super_ctx: &'ctx Self) -> Self {
        Self {
            variables: RefCell::new(HashMap::new()),
            functions: HashMap::new(),
            super_context: Some(super_ctx),
        }
    }

    fn _get_var(&self, name: &str) -> Option<Value> {
        if let Some(val) = self.variables.borrow().get(name) {
            Some(val.borrow().clone())
        } else if let Some(super_ctx) = self.super_context {
            super_ctx._get_var(name)
        } else {
            None
        }
    }

    fn get_var_rc(&self, name: &str) -> Option<Rc<RefCell<Value>>> {
        if let Some(val) = self.variables.borrow().get(name) {
            Some(val.clone())
        } else if let Some(super_ctx) = self.super_context {
            super_ctx.get_var_rc(name)
        } else {
            None
        }
    }

    fn get_fn(&self, name: &str) -> Option<&FuncDef<'src, 'ast, 'native>> {
        if let Some(val) = self.functions.get(name) {
            Some(val)
        } else if let Some(super_ctx) = self.super_context {
            super_ctx.get_fn(name)
        } else {
            None
        }
    }
}

macro_rules! unwrap_break {
    ($e:expr) => {
        match unwrap_deref($e) {
            RunResult::Yield(v) => v,
            RunResult::Break => break,
        }
    };
}

pub fn run<'src, 'ast>(
    stmts: &'ast Vec<Statement<'src>>,
    ctx: &mut EvalContext<'src, 'ast, '_, '_>,
) -> Result<RunResult, EvalError> {
    let mut res = RunResult::Yield(Value::I32(0));
    for stmt in stmts {
        match stmt {
            Statement::VarDecl(var, type_, initializer) => {
                let init_val = if let Some(init_expr) = initializer {
                    unwrap_break!(eval(init_expr, ctx)?)
                } else {
                    Value::I32(0)
                };
                let init_val = coerce_type(&init_val, type_)?;
                ctx.variables
                    .borrow_mut()
                    .insert(*var, Rc::new(RefCell::new(init_val)));
            }
            Statement::FnDecl {
                name,
                args,
                ret_type,
                stmts,
            } => {
                ctx.functions.insert(
                    name.to_string(),
                    FuncDef::Code(FuncCode {
                        args,
                        ret_type: ret_type.clone(),
                        stmts,
                    }),
                );
            }
            Statement::Expression(e) => {
                res = eval(&e, ctx)?;
                if let RunResult::Break = res {
                    return Ok(res);
                }
                // println!("Expression evaluates to: {:?}", res);
            }
            Statement::Loop(e) => loop {
                res = RunResult::Yield(unwrap_break!(run(e, ctx)?));
            },
            Statement::While(cond, e) => loop {
                match unwrap_deref(eval(cond, ctx)?) {
                    RunResult::Yield(v) => {
                        if !truthy(&v) {
                            break;
                        }
                    }
                    RunResult::Break => break,
                }
                res = match unwrap_deref(run(e, ctx)?) {
                    RunResult::Yield(v) => RunResult::Yield(v),
                    RunResult::Break => break,
                };
            },
            Statement::For(iter, from, to, e) => {
                let from_res = coerce_i64(&unwrap_break!(eval(from, ctx)?))? as i64;
                let to_res = coerce_i64(&unwrap_break!(eval(to, ctx)?))? as i64;
                for i in from_res..to_res {
                    ctx.variables
                        .borrow_mut()
                        .insert(iter, Rc::new(RefCell::new(Value::I64(i))));
                    res = RunResult::Yield(unwrap_break!(run(e, ctx)?));
                }
            }
            Statement::Break => {
                return Ok(RunResult::Break);
            }
            _ => {}
        }
    }
    Ok(res)
}

mod test;
