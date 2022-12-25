use std::{cell::RefCell, collections::HashMap};

use crate::{
    interpreter::{std_functions, FuncCode},
    parser::{Expression, Statement},
    FuncDef, TypeDecl, Value,
};

#[derive(Debug, PartialEq, Clone)]
pub enum TypeCheckResult {
    Yield(TypeDecl),
    Break,
}

pub type TypeCheckError = String;

macro_rules! unwrap_tc {
    ($e:expr) => {
        match $e {
            TypeCheckResult::Yield(v) => v,
            TypeCheckResult::Break => return Ok(TypeCheckResult::Break),
        }
    };
}

#[derive(Clone)]
pub struct TypeCheckContext<'src, 'ast, 'native, 'ctx> {
    /// RefCell to allow mutation in super context.
    /// Also, the inner values must be Rc of RefCell because a reference could be returned from
    /// a function so that the variable scope may have been ended.
    variables: RefCell<HashMap<&'src str, TypeDecl>>,
    /// Function names are owned strings because it can be either from source or native.
    /// Unlike variables, functions cannot be overwritten in the outer scope, so it does not
    /// need to be wrapped in a RefCell.
    functions: HashMap<String, FuncDef<'src, 'ast, 'native>>,
    super_context: Option<&'ctx TypeCheckContext<'src, 'ast, 'native, 'ctx>>,
}

impl<'src, 'ast, 'native, 'ctx> TypeCheckContext<'src, 'ast, 'native, 'ctx> {
    pub fn new() -> Self {
        Self {
            variables: RefCell::new(HashMap::new()),
            functions: std_functions(),
            super_context: None,
        }
    }

    fn get_var(&self, name: &str) -> Option<TypeDecl> {
        if let Some(val) = self.variables.borrow().get(name) {
            Some(val.clone())
        } else if let Some(super_ctx) = self.super_context {
            super_ctx.get_var(name)
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

    fn push_stack(super_ctx: &'ctx Self) -> Self {
        Self {
            variables: RefCell::new(HashMap::new()),
            functions: HashMap::new(),
            super_context: Some(super_ctx),
        }
    }
}

fn tc_expr<'a, 'b>(
    e: &'b Expression<'a>,
    ctx: &mut TypeCheckContext<'a, 'b, '_, '_>,
) -> Result<TypeCheckResult, TypeCheckError> {
    Ok(match e {
        Expression::NumLiteral(val) => TypeCheckResult::Yield(match val {
            Value::F64(_) | Value::F32(_) => TypeDecl::Float,
            Value::I64(_) | Value::I32(_) => TypeDecl::Integer,
            _ => return Err("Numeric literal has a non-number value".to_string()),
        }),
        Expression::StrLiteral(_val) => TypeCheckResult::Yield(TypeDecl::Str),
        Expression::ArrLiteral(val) => {
            if !val.is_empty() {
                for (ex1, ex2) in val[..val.len() - 1].iter().zip(val[1..].iter()) {
                    let el1 = tc_expr(ex1, ctx)?;
                    let el2 = tc_expr(ex2, ctx)?;
                    if el1 != el2 {
                        return Err(format!(
                            "Types in an array is not homogeneous: {el1:?} and {el2:?}"
                        ));
                    }
                }
            }
            let ty = if let TypeCheckResult::Yield(ty) = val
                .first()
                .map(|e| tc_expr(e, ctx))
                .unwrap_or(Ok(TypeCheckResult::Yield(TypeDecl::Any)))?
            {
                ty
            } else {
                return Err("Should not yield".to_string());
            };
            TypeCheckResult::Yield(TypeDecl::Array(Box::new(ty)))
        }
        Expression::Variable(str) => TypeCheckResult::Yield(
            ctx.get_var(str)
                .ok_or_else(|| format!("Variable {} not found in scope", str))?,
        ),
        Expression::Add(lhs, rhs) => binary_op(&lhs, &rhs, ctx, "Add")?,
        Expression::Sub(lhs, rhs) => binary_op(&lhs, &rhs, ctx, "Sub")?,
        Expression::Mult(lhs, rhs) => binary_op(&lhs, &rhs, ctx, "Mult")?,
        Expression::Div(lhs, rhs) => binary_op(&lhs, &rhs, ctx, "Div")?,
        Expression::FnInvoke(str, args) => {
            let args = args
                .iter()
                .map(|v| tc_expr(v, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            let func = ctx
                .get_fn(*str)
                .ok_or_else(|| format!("function {} is not defined.", str))?;
            let args_decl = func.args();
            for (arg, decl) in args.iter().zip(args_decl.iter()) {
                let arg = unwrap_tc!(arg);
                tc_coerce_type(&arg, &decl.1)?;
            }
            match func {
                FuncDef::Code(code) => {
                    TypeCheckResult::Yield(code.ret_type.clone().unwrap_or(TypeDecl::Any))
                }
                FuncDef::Native(native) => {
                    TypeCheckResult::Yield(native.ret_type.clone().unwrap_or(TypeDecl::Any))
                }
            }
        }
        Expression::ArrIndex(ex, args) => {
            let args = args
                .iter()
                .map(|v| tc_expr(v, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            let _arg0 = match args[0].clone() {
                TypeCheckResult::Yield(v) => {
                    if let TypeDecl::I64 = tc_coerce_type(&v, &TypeDecl::I64)? {
                        v
                    } else {
                        return Err("Subscript type should be integer types".to_string());
                    }
                }
                TypeCheckResult::Break => {
                    return Ok(TypeCheckResult::Break);
                }
            };
            if let TypeDecl::Array(inner) = unwrap_tc!(tc_expr(ex, ctx)?) {
                TypeCheckResult::Yield(*inner.clone())
            } else {
                return Err("Subscript operator's first operand is not an array".to_string());
            }
        }
        _ => todo!(),
    })
}

fn tc_coerce_type(value: &TypeDecl, target: &TypeDecl) -> Result<TypeDecl, TypeCheckError> {
    use TypeDecl::*;
    Ok(match (value, target) {
        (_, Any) => value.clone(),
        (Any, _) => target.clone(),
        (F64 | Float, F64) => F64,
        (F32 | Float, F32) => F32,
        (I64 | Integer, I64) => I64,
        (I32 | Integer, I32) => I32,
        (Str, Str) => Str,
        (Array(v_inner), Array(t_inner)) => Array(Box::new(tc_coerce_type(v_inner, t_inner)?)),
        _ => {
            return Err(format!(
                "Type check error! {:?} cannot be assigned to {:?}",
                value, target
            ))
        }
    })
}

pub fn type_check<'src, 'ast>(
    stmts: &'ast Vec<Statement<'src>>,
    ctx: &mut TypeCheckContext<'src, 'ast, '_, '_>,
) -> Result<TypeCheckResult, TypeCheckError> {
    let mut res = TypeCheckResult::Yield(TypeDecl::Any);
    for stmt in stmts {
        match stmt {
            Statement::VarDecl(var, type_, initializer) => {
                let init_val = if let Some(init_expr) = initializer {
                    match tc_expr(init_expr, ctx)? {
                        TypeCheckResult::Yield(ty) => ty,
                        _ => break,
                    }
                } else {
                    TypeDecl::I32
                };
                let init_val = tc_coerce_type(&init_val, type_)?;
                ctx.variables.borrow_mut().insert(*var, init_val);
            }
            Statement::FnDecl {
                name,
                args,
                ret_type,
                stmts,
            } => {
                let mut subctx = TypeCheckContext::push_stack(ctx);
                for arg in args.iter() {
                    subctx
                        .variables
                        .borrow_mut()
                        .insert(arg.0.clone(), arg.1.clone());
                }
                type_check(stmts, &mut subctx)?;
                ctx.functions.insert(
                    name.to_string(),
                    FuncDef::Code(FuncCode::new(stmts, args, ret_type.clone())),
                );
            }
            Statement::Expression(e) => {
                res = tc_expr(&e, ctx)?;
                if let TypeCheckResult::Break = res {
                    return Ok(res);
                }
            }
            Statement::For(iter, from, to, e) => {
                tc_coerce_type(&unwrap_tc!(tc_expr(from, ctx)?), &TypeDecl::I64)?;
                tc_coerce_type(&unwrap_tc!(tc_expr(to, ctx)?), &TypeDecl::I64)?;
                ctx.variables.borrow_mut().insert(iter, TypeDecl::I64);
                res = TypeCheckResult::Yield(unwrap_tc!(type_check(e, ctx)?));
            }
            Statement::Comment(_) => (),
            _ => todo!(),
        }
    }
    Ok(res)
}

fn binary_op<'src, 'ast>(
    lhs: &'ast Expression<'src>,
    rhs: &'ast Expression<'src>,
    ctx: &mut TypeCheckContext<'src, 'ast, '_, '_>,
    op: &str,
) -> Result<TypeCheckResult, TypeCheckError> {
    let lhs = unwrap_tc!(tc_expr(lhs, ctx)?);
    let rhs = unwrap_tc!(tc_expr(rhs, ctx)?);
    let res = TypeCheckResult::Yield(match (&lhs, &rhs) {
        (TypeDecl::F64 | TypeDecl::F32, TypeDecl::F64 | TypeDecl::F32) => TypeDecl::Float,
        (TypeDecl::I64 | TypeDecl::I32, TypeDecl::I64 | TypeDecl::I32) => TypeDecl::Integer,
        (TypeDecl::Str, TypeDecl::Str) => TypeDecl::Str,
        (TypeDecl::Float, TypeDecl::Float) => TypeDecl::Float,
        (TypeDecl::Integer, TypeDecl::Integer) => TypeDecl::Integer,
        _ => {
            return Err(format!(
                "{op} between incompatible type: {:?} and {:?}",
                lhs, rhs
            ))
        }
    });
    Ok(res)
}
