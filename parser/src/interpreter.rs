use crate::{parser::*, value::ArrayInt, TypeDecl, Value};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

#[derive(Debug, PartialEq, Clone)]
pub enum RunResult {
    Yield(Value),
    Break,
}

pub type EvalError = String;

fn unwrap_deref(e: RunResult) -> RunResult {
    match &e {
        RunResult::Yield(Value::Ref(vref)) => {
            let r = vref.borrow();
            return RunResult::Yield(r.clone());
        }
        RunResult::Yield(Value::ArrayRef(a, idx)) => {
            return RunResult::Yield(a.borrow().values.get(*idx).unwrap().clone());
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

pub(crate) fn binary_op_str(
    lhs: &Value,
    rhs: &Value,
    d: impl Fn(f64, f64) -> Result<f64, EvalError>,
    i: impl Fn(i64, i64) -> i64,
    s: impl Fn(&str, &str) -> Result<String, EvalError>,
) -> Result<Value, EvalError> {
    Ok(match (lhs, rhs) {
        // "Deref" the references before binary
        (Value::Ref(lhs), ref rhs) => binary_op_str(&lhs.borrow(), rhs, d, i, s)?,
        (ref lhs, Value::Ref(rhs)) => binary_op_str(lhs, &rhs.borrow(), d, i, s)?,
        (Value::ArrayRef(lhs, idx), ref rhs) => {
            binary_op_str(&lhs.borrow().values.get(*idx).unwrap(), rhs, d, i, s)?
        }
        (ref lhs, Value::ArrayRef(rhs, idx)) => {
            binary_op_str(lhs, &rhs.borrow().values.get(*idx).unwrap(), d, i, s)?
        }
        (Value::F64(lhs), rhs) => Value::F64(d(*lhs, coerce_f64(&rhs)?)?),
        (lhs, Value::F64(rhs)) => Value::F64(d(coerce_f64(&lhs)?, *rhs)?),
        (Value::F32(lhs), rhs) => Value::F32(d(*lhs as f64, coerce_f64(&rhs)?)? as f32),
        (lhs, Value::F32(rhs)) => Value::F32(d(coerce_f64(&lhs)?, *rhs as f64)? as f32),
        (Value::I64(lhs), Value::I64(rhs)) => Value::I64(i(*lhs, *rhs)),
        (Value::I64(lhs), Value::I32(rhs)) => Value::I64(i(*lhs, *rhs as i64)),
        (Value::I32(lhs), Value::I64(rhs)) => Value::I64(i(*lhs as i64, *rhs)),
        (Value::I32(lhs), Value::I32(rhs)) => Value::I32(i(*lhs as i64, *rhs as i64) as i32),
        (Value::Str(lhs), Value::Str(rhs)) => Value::Str(s(lhs, rhs)?),
        _ => {
            return Err(format!(
                "Unsupported addition between {:?} and {:?}",
                lhs, rhs
            ))
        }
    })
}

pub(crate) fn binary_op(
    lhs: &Value,
    rhs: &Value,
    d: impl Fn(f64, f64) -> f64,
    i: impl Fn(i64, i64) -> i64,
) -> Result<Value, EvalError> {
    binary_op_str(
        lhs,
        rhs,
        |lhs, rhs| Ok(d(lhs, rhs)),
        i,
        |_lhs, _rhs| Err("This operator is not supported for strings".to_string()),
    )
}

pub(crate) fn binary_op_int(
    lhs: &Value,
    rhs: &Value,
    i: impl Fn(i64, i64) -> i64,
) -> Result<Value, EvalError> {
    binary_op_str(
        lhs,
        rhs,
        |_lhs, _rhs| Err("This operator is not supported for double".to_string()),
        i,
        |_lhs, _rhs| Err("This operator is not supported for strings".to_string()),
    )
}

pub(crate) fn truthy(a: &Value) -> bool {
    match a {
        Value::F64(v) => *v != 0.,
        Value::F32(v) => *v != 0.,
        Value::I64(v) => *v != 0,
        Value::I32(v) => *v != 0,
        Value::Ref(r) => truthy(&r.borrow()),
        _ => false,
    }
}

pub(crate) fn coerce_f64(a: &Value) -> Result<f64, EvalError> {
    Ok(match a {
        Value::F64(v) => *v as f64,
        Value::F32(v) => *v as f64,
        Value::I64(v) => *v as f64,
        Value::I32(v) => *v as f64,
        Value::Ref(r) => coerce_f64(&r.borrow())?,
        _ => 0.,
    })
}

pub(crate) fn coerce_i64(a: &Value) -> Result<i64, EvalError> {
    Ok(match a {
        Value::F64(v) => *v as i64,
        Value::F32(v) => *v as i64,
        Value::I64(v) => *v as i64,
        Value::I32(v) => *v as i64,
        Value::Ref(r) => coerce_i64(&r.borrow())?,
        _ => 0,
    })
}

fn coerce_str(a: &Value) -> Result<String, EvalError> {
    Ok(match a {
        Value::F64(v) => v.to_string(),
        Value::F32(v) => v.to_string(),
        Value::I64(v) => v.to_string(),
        Value::I32(v) => v.to_string(),
        Value::Str(v) => v.clone(),
        _ => return Err("Can't convert array to str".to_string()),
    })
}

fn _coerce_var(value: &Value, target: &Value) -> Result<Value, EvalError> {
    Ok(match target {
        Value::F64(_) => Value::F64(coerce_f64(value)?),
        Value::F32(_) => Value::F32(coerce_f64(value)? as f32),
        Value::I64(_) => Value::I64(coerce_i64(value)?),
        Value::I32(_) => Value::I32(coerce_i64(value)? as i32),
        Value::Str(_) => Value::Str(coerce_str(value)?),
        Value::Array(array) => {
            let ArrayInt {
                type_decl: inner_type,
                values: inner,
            } = &array.borrow() as &ArrayInt;
            if inner.len() == 0 {
                if let Value::Array(array) = value {
                    if array.borrow().values.len() == 0 {
                        return Ok(value.clone());
                    }
                }
                return Err("Cannot coerce type to empty array".to_string());
            } else {
                if let Value::Array(array) = value {
                    Value::Array(ArrayInt::new(
                        inner_type.clone(),
                        array
                            .borrow()
                            .values
                            .iter()
                            .map(|val| -> Result<_, String> { Ok(coerce_type(val, inner_type)?) })
                            .collect::<Result<_, _>>()?,
                    ))
                } else {
                    return Err("Cannot coerce scalar to array".to_string());
                }
            }
        }
        // We usually don't care about coercion
        Value::Ref(_) => value.clone(),
        Value::ArrayRef(_, _) => value.clone(),
    })
}

pub fn coerce_type(value: &Value, target: &TypeDecl) -> Result<Value, EvalError> {
    Ok(match target {
        TypeDecl::Any => value.clone(),
        TypeDecl::F64 => Value::F64(coerce_f64(value)?),
        TypeDecl::F32 => Value::F32(coerce_f64(value)? as f32),
        TypeDecl::I64 => Value::I64(coerce_i64(value)?),
        TypeDecl::I32 => Value::I32(coerce_i64(value)? as i32),
        TypeDecl::Str => Value::Str(coerce_str(value)?),
        TypeDecl::Array(inner) => {
            if let Value::Array(array) = value {
                Value::Array(ArrayInt::new(
                    (**inner).clone(),
                    array
                        .borrow()
                        .values
                        .iter()
                        .map(|value_elem| -> Result<_, EvalError> {
                            Ok(coerce_type(value_elem, inner)?)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ))
            } else {
                return Err(format!("Incompatible type to array! {:?}", value));
            }
        }
        TypeDecl::Float => Value::F64(coerce_f64(value)?),
        TypeDecl::Integer => Value::I64(coerce_i64(value)?),
    })
}

pub(crate) fn eval<'src, 'native>(
    e: &Expression<'src>,
    ctx: &mut EvalContext<'src, 'native, '_>,
) -> Result<RunResult, EvalError>
where
    'native: 'src,
{
    Ok(match &e.expr {
        ExprEnum::NumLiteral(val) => RunResult::Yield(val.clone()),
        ExprEnum::StrLiteral(val) => RunResult::Yield(Value::Str(val.clone())),
        ExprEnum::ArrLiteral(val) => RunResult::Yield(Value::Array(ArrayInt::new(
            TypeDecl::Any,
            val.iter()
                .map(|v| {
                    if let RunResult::Yield(y) = eval(v, ctx)? {
                        Ok(y)
                    } else {
                        Err("Break in array literal not supported".to_string())
                    }
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))),
        ExprEnum::Variable(str) => RunResult::Yield(Value::Ref(
            ctx.get_var_rc(str)
                .ok_or_else(|| format!("Variable {} not found in scope", str))?,
        )),
        ExprEnum::Cast(ex, decl) => {
            RunResult::Yield(coerce_type(&unwrap_run!(eval(ex, ctx)?), decl)?)
        }
        ExprEnum::VarAssign(lhs, rhs) => {
            let lhs_result = eval(lhs, ctx)?;
            let result = match lhs_result {
                RunResult::Yield(Value::Ref(rc)) => {
                    let rhs_value = unwrap_run!(eval(rhs, ctx)?);
                    *rc.borrow_mut() = rhs_value.clone();
                    rhs_value
                }
                RunResult::Yield(Value::ArrayRef(rc, idx)) => {
                    if let Some(mref) = rc.borrow_mut().values.get_mut(idx) {
                        let rhs_value = unwrap_run!(eval(rhs, ctx)?);
                        *mref = rhs_value.clone();
                        rhs_value
                    } else {
                        return Err(format!("ArrayRef index out of range",));
                    }
                }
                _ => {
                    return Err(format!(
                        "We need variable reference on lhs to assign. Actually we got {:?}",
                        lhs_result
                    ));
                }
            };
            RunResult::Yield(result)
        }
        ExprEnum::FnInvoke(str, args) => {
            let default_args = {
                let fn_args = ctx
                    .get_fn(*str)
                    .ok_or_else(|| format!("function {} is not defined.", str))?
                    .args();

                if args.len() <= fn_args.len() {
                    let fn_args = fn_args[args.len()..].to_vec();

                    fn_args
                        .into_iter()
                        .filter_map(|arg| {
                            arg.init
                                .as_ref()
                                .map(|init| eval(init, &mut EvalContext::new()))
                        })
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    vec![]
                }
            };

            // Collect unordered args first
            let mut eval_args = args
                .iter()
                .filter(|v| v.name.is_none())
                .map(|v| eval(&v.expr, ctx))
                .chain(default_args.into_iter().map(Ok))
                .collect::<Result<Vec<_>, _>>()?;
            let named_args: Vec<_> = args
                .into_iter()
                .filter_map(|arg| {
                    if let Some(ref name) = arg.name {
                        Some((name, &arg.expr))
                    } else {
                        None
                    }
                })
                .map(|(name, expr)| Ok::<_, String>((name, eval(expr, ctx)?)))
                .collect::<Result<Vec<_>, _>>()?;

            let func = ctx
                .get_fn(*str)
                .ok_or_else(|| format!("function {} is not defined.", str))?;

            let mut subctx = EvalContext::push_stack(ctx);
            match func {
                FuncDef::Code(func) => {
                    for (name, val) in named_args.into_iter() {
                        if let Some((i, _decl_arg)) =
                            func.args.iter().enumerate().find(|f| f.1.name == **name)
                        {
                            if eval_args.len() <= i {
                                eval_args.resize(i + 1, RunResult::Yield(Value::I32(0)));
                            }
                            eval_args[i] = val;
                        } else {
                            return Err(format!("No matching named parameter \"{name}\" is found in function \"{str}\""));
                        }
                    }

                    for (k, v) in func.args.iter().zip(&eval_args) {
                        subctx.variables.borrow_mut().insert(
                            k.name,
                            Rc::new(RefCell::new(coerce_type(&unwrap_run!(v.clone()), &k.ty)?)),
                        );
                    }
                    let run_result = run(&func.stmts, &mut subctx)?;
                    match unwrap_deref(run_result) {
                        RunResult::Yield(v) => match &func.ret_type {
                            Some(ty) => RunResult::Yield(coerce_type(&v, ty)?),
                            None => RunResult::Yield(v),
                        },
                        RunResult::Break => return Err("break in function toplevel".to_string()),
                    }
                }
                FuncDef::Native(native) => RunResult::Yield((native.code)(
                    &eval_args
                        .into_iter()
                        .map(|e| {
                            Ok(match e {
                                RunResult::Yield(v) => v.clone(),
                                RunResult::Break => {
                                    return Err("Break in function argument is not supported yet!"
                                        .to_string())
                                }
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )?),
            }
        }
        ExprEnum::ArrIndex(ex, args) => {
            let args = args
                .iter()
                .map(|v| eval(v, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            let arg0 = match unwrap_deref(args[0].clone()) {
                RunResult::Yield(v) => {
                    if let Value::I64(idx) = coerce_type(&v, &TypeDecl::I64)? {
                        idx as u64
                    } else {
                        return Err("Subscript type should be integer types".to_string());
                    }
                }
                RunResult::Break => {
                    return Ok(RunResult::Break);
                }
            };
            let result = unwrap_run!(eval(ex, ctx)?);
            RunResult::Yield(result.array_get_ref(arg0)?)
        }
        ExprEnum::Not(val) => {
            RunResult::Yield(Value::I32(if truthy(&unwrap_run!(eval(val, ctx)?)) {
                0
            } else {
                1
            }))
        }
        ExprEnum::BitNot(val) => {
            let val = unwrap_run!(eval(val, ctx)?);
            RunResult::Yield(match val {
                Value::I32(i) => Value::I32(!i),
                Value::I64(i) => Value::I64(!i),
                _ => return Err(format!("Bitwise not is not supported for {:?}", val)),
            })
        }
        ExprEnum::Add(lhs, rhs) => {
            let res = RunResult::Yield(binary_op_str(
                &unwrap_run!(eval(lhs, ctx)?),
                &unwrap_run!(eval(rhs, ctx)?),
                |lhs, rhs| Ok(lhs + rhs),
                |lhs, rhs| lhs + rhs,
                |lhs: &str, rhs: &str| Ok(lhs.to_string() + rhs),
            )?);
            res
        }
        ExprEnum::Sub(lhs, rhs) => RunResult::Yield(binary_op(
            &unwrap_run!(eval(lhs, ctx)?),
            &unwrap_run!(eval(rhs, ctx)?),
            |lhs, rhs| lhs - rhs,
            |lhs, rhs| lhs - rhs,
        )?),
        ExprEnum::Mult(lhs, rhs) => RunResult::Yield(binary_op(
            &unwrap_run!(eval(lhs, ctx)?),
            &unwrap_run!(eval(rhs, ctx)?),
            |lhs, rhs| lhs * rhs,
            |lhs, rhs| lhs * rhs,
        )?),
        ExprEnum::Div(lhs, rhs) => RunResult::Yield(binary_op(
            &unwrap_run!(eval(lhs, ctx)?),
            &unwrap_run!(eval(rhs, ctx)?),
            |lhs, rhs| lhs / rhs,
            |lhs, rhs| lhs / rhs,
        )?),
        ExprEnum::LT(lhs, rhs) => RunResult::Yield(binary_op(
            &unwrap_run!(eval(lhs, ctx)?),
            &unwrap_run!(eval(rhs, ctx)?),
            |lhs, rhs| if lhs < rhs { 1. } else { 0. },
            |lhs, rhs| if lhs < rhs { 1 } else { 0 },
        )?),
        ExprEnum::GT(lhs, rhs) => RunResult::Yield(binary_op(
            &unwrap_run!(eval(lhs, ctx)?),
            &unwrap_run!(eval(rhs, ctx)?),
            |lhs, rhs| if lhs > rhs { 1. } else { 0. },
            |lhs, rhs| if lhs > rhs { 1 } else { 0 },
        )?),
        ExprEnum::BitAnd(lhs, rhs) => RunResult::Yield(binary_op_int(
            &unwrap_run!(eval(lhs, ctx)?),
            &unwrap_run!(eval(rhs, ctx)?),
            |lhs, rhs| lhs & rhs,
        )?),
        ExprEnum::BitXor(lhs, rhs) => RunResult::Yield(binary_op_int(
            &unwrap_run!(eval(lhs, ctx)?),
            &unwrap_run!(eval(rhs, ctx)?),
            |lhs, rhs| lhs ^ rhs,
        )?),
        ExprEnum::BitOr(lhs, rhs) => RunResult::Yield(binary_op_int(
            &unwrap_run!(eval(lhs, ctx)?),
            &unwrap_run!(eval(rhs, ctx)?),
            |lhs, rhs| lhs | rhs,
        )?),
        ExprEnum::And(lhs, rhs) => RunResult::Yield(Value::I32(
            if truthy(&unwrap_run!(eval(lhs, ctx)?)) && truthy(&unwrap_run!(eval(rhs, ctx)?)) {
                1
            } else {
                0
            },
        )),
        ExprEnum::Or(lhs, rhs) => RunResult::Yield(Value::I32(
            if truthy(&unwrap_run!(eval(lhs, ctx)?)) || truthy(&unwrap_run!(eval(rhs, ctx)?)) {
                1
            } else {
                0
            },
        )),
        ExprEnum::Conditional(cond, true_branch, false_branch) => {
            if truthy(&unwrap_run!(eval(cond, ctx)?)) {
                run(true_branch, ctx)?
            } else if let Some(ast) = false_branch {
                run(ast, ctx)?
            } else {
                RunResult::Yield(Value::I32(0))
            }
        }
        ExprEnum::Brace(stmts) => {
            let mut subctx = EvalContext::push_stack(ctx);
            let res = run(stmts, &mut subctx)?;
            if let RunResult::Yield(Value::Ref(res)) = res {
                // "Dereference" if the result was a reference
                RunResult::Yield(std::mem::take(&mut res.borrow_mut()))
            } else {
                res
            }
        }
    })
}

pub(crate) fn s_print(vals: &[Value]) -> Result<Value, EvalError> {
    println!("print:");
    fn print_inner(val: &Value) {
        match val {
            Value::F64(val) => print!(" {}", val),
            Value::F32(val) => print!(" {}", val),
            Value::I64(val) => print!(" {}", val),
            Value::I32(val) => print!(" {}", val),
            Value::Str(val) => print!(" {}", val),
            Value::Array(val) => {
                print!("[");
                for val in val.borrow().values.iter() {
                    print_inner(val);
                }
                print!("]");
            }
            Value::Ref(r) => {
                print!("ref(");
                print_inner(&r.borrow());
                print!(")");
            }
            Value::ArrayRef(r, idx) => {
                print!("arrayref(");
                print_inner((*r.borrow()).values.get(*idx).unwrap());
                print!(")");
            }
        }
    }
    for val in vals {
        print_inner(val);
    }
    print!("\n");
    Ok(Value::I32(0))
}

fn s_puts(vals: &[Value]) -> Result<Value, EvalError> {
    fn puts_inner(vals: &[Value]) {
        for val in vals {
            match val {
                Value::F64(val) => print!("{}", val),
                Value::F32(val) => print!("{}", val),
                Value::I64(val) => print!("{}", val),
                Value::I32(val) => print!("{}", val),
                Value::Str(val) => print!("{}", val),
                Value::Array(val) => puts_inner(
                    &val.borrow()
                        .values
                        .iter()
                        .map(|v| v.clone())
                        .collect::<Vec<_>>(),
                ),
                Value::Ref(r) => puts_inner(&[r.borrow().clone()]),
                Value::ArrayRef(r, idx) => {
                    if let Some(r) = r.borrow().values.get(*idx) {
                        puts_inner(&[r.clone()])
                    }
                }
            }
        }
    }
    puts_inner(vals);
    Ok(Value::I32(0))
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
        TypeDecl::Float => "<Float>".to_string(),
        TypeDecl::Integer => "<Integer>".to_string(),
    }
}

pub(crate) fn s_type(vals: &[Value]) -> Result<Value, EvalError> {
    fn type_str(val: &Value) -> String {
        match val {
            Value::F64(_) => "f64".to_string(),
            Value::F32(_) => "f32".to_string(),
            Value::I64(_) => "i64".to_string(),
            Value::I32(_) => "i32".to_string(),
            Value::Str(_) => "str".to_string(),
            Value::Array(inner) => format!("[{}]", type_decl_to_str(&inner.borrow().type_decl)),
            Value::Ref(r) => format!("ref[{}]", type_str(&r.borrow())),
            Value::ArrayRef(r, _) => format!("aref[{}]", type_decl_to_str(&r.borrow().type_decl)),
        }
    }
    if let [val, ..] = vals {
        Ok(Value::Str(type_str(val)))
    } else {
        Ok(Value::I32(0))
    }
}

pub(crate) fn s_len(vals: &[Value]) -> Result<Value, EvalError> {
    if let [val, ..] = vals {
        Ok(Value::I64(val.array_len() as i64))
    } else {
        Ok(Value::I32(0))
    }
}

pub(crate) fn s_push(vals: &[Value]) -> Result<Value, EvalError> {
    if let [arr, val, ..] = vals {
        let val = val.clone().deref();
        arr.array_push(val).map(|_| Value::I32(0))
    } else {
        Ok(Value::I32(0))
    }
}

pub(crate) fn s_hex_string(vals: &[Value]) -> Result<Value, EvalError> {
    if let [val, ..] = vals {
        match coerce_type(val, &TypeDecl::I64).unwrap() {
            Value::I64(i) => Ok(Value::Str(format!("{:02x}", i))),
            _ => Err("hex_string() could not convert argument to i64".to_string()),
        }
    } else {
        Ok(Value::Str("".to_string()))
    }
}

#[derive(Clone)]
pub struct FuncCode<'src> {
    args: Vec<ArgDecl<'src>>,
    pub(crate) ret_type: Option<TypeDecl>,
    /// Owning a clone of AST of statements is not quite efficient, but we could not get
    /// around the borrow checker.
    stmts: Vec<Statement<'src>>,
}

impl<'src> FuncCode<'src> {
    pub(crate) fn new(
        stmts: Vec<Statement<'src>>,
        args: Vec<ArgDecl<'src>>,
        ret_type: Option<TypeDecl>,
    ) -> Self {
        Self {
            args,
            ret_type,
            stmts,
        }
    }
}

#[derive(Clone)]
pub struct NativeCode<'native> {
    args: Vec<ArgDecl<'native>>,
    pub(crate) ret_type: Option<TypeDecl>,
    code: &'native dyn Fn(&[Value]) -> Result<Value, EvalError>,
}

impl<'native> NativeCode<'native> {
    pub(crate) fn new(
        code: &'native dyn Fn(&[Value]) -> Result<Value, EvalError>,
        args: Vec<ArgDecl<'native>>,
        ret_type: Option<TypeDecl>,
    ) -> Self {
        Self {
            args,
            ret_type,
            code,
        }
    }
}

#[derive(Clone)]
pub enum FuncDef<'src, 'native> {
    Code(FuncCode<'src>),
    Native(NativeCode<'native>),
}

impl<'src, 'ast, 'native> FuncDef<'src, 'native> {
    pub fn new_native(
        code: &'native dyn Fn(&[Value]) -> Result<Value, EvalError>,
        args: Vec<ArgDecl<'native>>,
        ret_type: Option<TypeDecl>,
    ) -> Self {
        Self::Native(NativeCode::new(code, args, ret_type))
    }

    pub(crate) fn args(&self) -> &Vec<ArgDecl<'src>>
    where
        'native: 'src,
    {
        match self {
            FuncDef::Code(FuncCode { args, .. }) => args,
            FuncDef::Native(NativeCode { args, .. }) => args,
        }
    }
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
pub struct EvalContext<'src, 'native, 'ctx> {
    /// RefCell to allow mutation in super context.
    /// Also, the inner values must be Rc of RefCell because a reference could be returned from
    /// a function so that the variable scope may have been ended.
    variables: RefCell<HashMap<&'src str, Rc<RefCell<Value>>>>,
    /// Function names are owned strings because it can be either from source or native.
    /// Unlike variables, functions cannot be overwritten in the outer scope, so it does not
    /// need to be wrapped in a RefCell.
    functions: HashMap<String, FuncDef<'src, 'native>>,
    super_context: Option<&'ctx EvalContext<'src, 'native, 'ctx>>,
}

impl<'src, 'ast, 'native, 'ctx> EvalContext<'src, 'native, 'ctx> {
    pub fn new() -> Self {
        Self {
            variables: RefCell::new(HashMap::new()),
            functions: std_functions(),
            super_context: None,
        }
    }

    pub fn set_fn(&mut self, name: &str, fun: FuncDef<'src, 'native>) {
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

    fn get_fn(&self, name: &str) -> Option<&FuncDef<'src, 'native>> {
        if let Some(val) = self.functions.get(name) {
            Some(val)
        } else if let Some(super_ctx) = self.super_context {
            super_ctx.get_fn(name)
        } else {
            None
        }
    }
}

pub(crate) fn std_functions<'src, 'native>() -> HashMap<String, FuncDef<'src, 'native>> {
    let mut functions = HashMap::new();
    functions.insert(
        "print".to_string(),
        FuncDef::new_native(&s_print, vec![], None),
    );
    functions.insert(
        "puts".to_string(),
        FuncDef::new_native(&s_puts, vec![ArgDecl::new("val", TypeDecl::Any)], None),
    );
    functions.insert(
        "type".to_string(),
        FuncDef::new_native(
            &s_type,
            vec![ArgDecl::new("value", TypeDecl::Any)],
            Some(TypeDecl::Str),
        ),
    );
    functions.insert(
        "len".to_string(),
        FuncDef::new_native(
            &s_len,
            vec![ArgDecl::new(
                "array",
                TypeDecl::Array(Box::new(TypeDecl::Any)),
            )],
            Some(TypeDecl::I64),
        ),
    );
    functions.insert(
        "push".to_string(),
        FuncDef::new_native(
            &s_push,
            vec![
                ArgDecl::new("array", TypeDecl::Array(Box::new(TypeDecl::Any))),
                ArgDecl::new("value", TypeDecl::Any),
            ],
            None,
        ),
    );
    functions.insert(
        "hex_string".to_string(),
        FuncDef::new_native(
            &s_hex_string,
            vec![ArgDecl::new("value", TypeDecl::I64)],
            Some(TypeDecl::Str),
        ),
    );
    functions
}

macro_rules! unwrap_break {
    ($e:expr) => {
        match unwrap_deref($e) {
            RunResult::Yield(v) => v,
            RunResult::Break => break,
        }
    };
}

pub fn run<'src, 'native>(
    stmts: &Vec<Statement<'src>>,
    ctx: &mut EvalContext<'src, 'native, '_>,
) -> Result<RunResult, EvalError>
where
    'native: 'src,
{
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
                    FuncDef::Code(FuncCode::new(stmts.clone(), args.clone(), ret_type.clone())),
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

#[cfg(test)]
mod test;
