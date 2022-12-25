use std::{cell::RefCell, collections::HashMap};

use crate::{
    interpreter::std_functions,
    parser::{Expression, Statement},
    FuncDef, TypeDecl, Value,
};

#[derive(Debug, PartialEq, Clone)]
pub enum TypeCheckResult {
    Yield(TypeDecl),
    Break,
}

pub type TypeCheckError = String;

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
        // Expression::ArrLiteral(val) => TypeCheckResult::
        Expression::Add(lhs, rhs) => {
            let lhs = if let TypeCheckResult::Yield(lhs) = tc_expr(lhs, ctx)? {
                lhs
            } else {
                return Ok(TypeCheckResult::Break);
            };
            let rhs = if let TypeCheckResult::Yield(rhs) = tc_expr(rhs, ctx)? {
                rhs
            } else {
                return Ok(TypeCheckResult::Break);
            };
            let res = TypeCheckResult::Yield(match (&lhs, &rhs) {
                (TypeDecl::F64 | TypeDecl::F32, TypeDecl::F64 | TypeDecl::F32) => TypeDecl::Float,
                (TypeDecl::I64 | TypeDecl::I32, TypeDecl::F64 | TypeDecl::F32) => TypeDecl::Integer,
                (TypeDecl::Str, TypeDecl::Str) => TypeDecl::Str,
                (TypeDecl::Float, TypeDecl::Float) => TypeDecl::Float,
                (TypeDecl::Integer, TypeDecl::Integer) => TypeDecl::Integer,
                _ => {
                    return Err(format!(
                        "Add between incompatible type: {:?} and {:?}",
                        lhs, rhs
                    ))
                }
            });
            res
        }
        Expression::Variable(str) => TypeCheckResult::Yield(
            ctx.get_var(str)
                .ok_or_else(|| format!("Variable {} not found in scope", str))?,
        ),
        Expression::FnInvoke(str, args) => {
            let args = args
                .iter()
                .map(|v| tc_expr(v, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            let func = ctx
                .get_fn(*str)
                .ok_or_else(|| format!("function {} is not defined.", str))?;
            match func {
                FuncDef::Code(code) => {
                    TypeCheckResult::Yield(code.ret_type.clone().unwrap_or(TypeDecl::Any))
                }
                FuncDef::Native(_) => TypeCheckResult::Yield(TypeDecl::Any),
            }
        }
        _ => todo!(),
    })
}

fn tc_coerce_type(value: &TypeDecl, target: &TypeDecl) -> Result<TypeDecl, TypeCheckError> {
    use TypeDecl::*;
    Ok(match (value, target) {
        (_, Any) => value.clone(),
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
    let res = TypeCheckResult::Yield(TypeDecl::Any);
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
                dbg!(&init_val);
                let init_val = tc_coerce_type(&init_val, type_)?;
                ctx.variables.borrow_mut().insert(*var, init_val);
            }
            Statement::Expression(e) => {
                let res = tc_expr(&e, ctx)?;
                if let TypeCheckResult::Break = res {
                    return Ok(res);
                }
            }
            Statement::Comment(_) => (),
            _ => todo!(),
        }
    }
    Ok(res)
}
