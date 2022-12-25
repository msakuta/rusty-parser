use std::{cell::RefCell, collections::HashMap};

use crate::{
    interpreter::{std_functions, FuncCode},
    parser::{Expression, Statement},
    FuncDef, TypeDecl, Value,
};

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

    pub fn set_fn(&mut self, name: &str, fun: FuncDef<'src, 'ast, 'native>) {
        self.functions.insert(name.to_string(), fun);
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
) -> Result<TypeDecl, TypeCheckError> {
    Ok(match e {
        Expression::NumLiteral(val) => match val {
            Value::F64(_) | Value::F32(_) => TypeDecl::Float,
            Value::I64(_) | Value::I32(_) => TypeDecl::Integer,
            _ => return Err("Numeric literal has a non-number value".to_string()),
        },
        Expression::StrLiteral(_val) => TypeDecl::Str,
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
            let ty = val
                .first()
                .map(|e| tc_expr(e, ctx))
                .unwrap_or(Ok(TypeDecl::Any))?;
            TypeDecl::Array(Box::new(ty))
        }
        Expression::Variable(str) => ctx
            .get_var(str)
            .ok_or_else(|| format!("Variable {} not found in scope", str))?,
        Expression::VarAssign(lhs, rhs) => binary_op(&lhs, &rhs, ctx, "Assignment")?,
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
                let arg = arg;
                tc_coerce_type(&arg, &decl.1)?;
            }
            match func {
                FuncDef::Code(code) => code.ret_type.clone().unwrap_or(TypeDecl::Any),
                FuncDef::Native(native) => native.ret_type.clone().unwrap_or(TypeDecl::Any),
            }
        }
        Expression::ArrIndex(ex, args) => {
            let args = args
                .iter()
                .map(|v| tc_expr(v, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            let _arg0 = {
                let v = args[0].clone();
                if let TypeDecl::I64 = tc_coerce_type(&v, &TypeDecl::I64)? {
                    v
                } else {
                    return Err("Subscript type should be integer types".to_string());
                }
            };
            if let TypeDecl::Array(inner) = tc_expr(ex, ctx)? {
                *inner.clone()
            } else {
                return Err("Subscript operator's first operand is not an array".to_string());
            }
        }
        Expression::Not(val) => tc_expr(val, ctx)?,
        Expression::Add(lhs, rhs) => binary_op(&lhs, &rhs, ctx, "Add")?,
        Expression::Sub(lhs, rhs) => binary_op(&lhs, &rhs, ctx, "Sub")?,
        Expression::Mult(lhs, rhs) => binary_op(&lhs, &rhs, ctx, "Mult")?,
        Expression::Div(lhs, rhs) => binary_op(&lhs, &rhs, ctx, "Div")?,
        Expression::LT(lhs, rhs) => binary_cmp(&lhs, &rhs, ctx, "LT")?,
        Expression::GT(lhs, rhs) => binary_cmp(&lhs, &rhs, ctx, "GT")?,
        Expression::And(lhs, rhs) => binary_op(&lhs, &rhs, ctx, "And")?,
        Expression::Or(lhs, rhs) => binary_op(&lhs, &rhs, ctx, "Or")?,
        Expression::Conditional(cond, true_branch, false_branch) => {
            tc_coerce_type(&tc_expr(cond, ctx)?, &TypeDecl::I32)?;
            let true_type = type_check(true_branch, ctx)?;
            if let Some(false_type) = false_branch {
                let false_type = type_check(false_type, ctx)?;
                binary_op_type(&true_type, &false_type).map_err(|_| {
                    format!("Conditional expression doesn't have the compatible types in true and false branch: {:?} and {:?}", true_type, false_type)
                })?
            } else {
                true_type
            }
        }
        Expression::Brace(stmts) => {
            let mut subctx = TypeCheckContext::push_stack(ctx);
            type_check(stmts, &mut subctx)?
        }
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
        (Float, Float) => Float,
        (Integer, Integer) => Integer,
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
) -> Result<TypeDecl, TypeCheckError> {
    let mut res = TypeDecl::Any;
    for stmt in stmts {
        match stmt {
            Statement::VarDecl(var, type_, initializer) => {
                let init_val = if let Some(init_expr) = initializer {
                    tc_expr(init_expr, ctx)?
                } else {
                    TypeDecl::Any
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
                // Function declaration needs to be added first to allow recursive calls
                ctx.functions.insert(
                    name.to_string(),
                    FuncDef::Code(FuncCode::new(stmts, args, ret_type.clone())),
                );
                let mut subctx = TypeCheckContext::push_stack(ctx);
                for arg in args.iter() {
                    subctx
                        .variables
                        .borrow_mut()
                        .insert(arg.0.clone(), arg.1.clone());
                }
                let last_stmt = type_check(stmts, &mut subctx)?;
                if let Some(ret_type) = ret_type {
                    tc_coerce_type(&last_stmt, ret_type)?;
                }
            }
            Statement::Expression(e) => {
                res = tc_expr(&e, ctx)?;
            }
            Statement::Loop(e) => {
                res = type_check(e, ctx)?;
            }
            Statement::While(cond, e) => {
                tc_coerce_type(&tc_expr(cond, ctx)?, &TypeDecl::I32)
                    .map_err(|e| format!("Type error in condition: {e}"))?;
                res = type_check(e, ctx)?;
            }
            Statement::For(iter, from, to, e) => {
                tc_coerce_type(&tc_expr(from, ctx)?, &TypeDecl::I64)?;
                tc_coerce_type(&tc_expr(to, ctx)?, &TypeDecl::I64)?;
                ctx.variables.borrow_mut().insert(iter, TypeDecl::I64);
                res = type_check(e, ctx)?;
            }
            Statement::Break => {
                // TODO: check types in break out site. For now we disallow break with values like Rust.
            }
            Statement::Comment(_) => (),
        }
    }
    Ok(res)
}

fn binary_op_gen<'src, 'ast>(
    lhs: &'ast Expression<'src>,
    rhs: &'ast Expression<'src>,
    ctx: &mut TypeCheckContext<'src, 'ast, '_, '_>,
    op: &str,
    mut f: impl FnMut(&TypeDecl, &TypeDecl) -> Result<TypeDecl, ()>,
) -> Result<TypeDecl, TypeCheckError> {
    let lhs = tc_expr(lhs, ctx)?;
    let rhs = tc_expr(rhs, ctx)?;
    f(&lhs, &rhs).map_err(|()| {
        format!(
            "Operation {op} between incompatible type: {:?} and {:?}",
            lhs, rhs
        )
    })
}

fn binary_op<'src, 'ast>(
    lhs: &'ast Expression<'src>,
    rhs: &'ast Expression<'src>,
    ctx: &mut TypeCheckContext<'src, 'ast, '_, '_>,
    op: &str,
) -> Result<TypeDecl, TypeCheckError> {
    binary_op_gen(lhs, rhs, ctx, op, binary_op_type)
}

fn binary_op_type(lhs: &TypeDecl, rhs: &TypeDecl) -> Result<TypeDecl, ()> {
    let res = match (&lhs, &rhs) {
        // Any type spreads contamination in the source code.
        (TypeDecl::Any, _) => TypeDecl::Any,
        (_, TypeDecl::Any) => TypeDecl::Any,
        (TypeDecl::F64, TypeDecl::F64) => TypeDecl::F64,
        (TypeDecl::F32, TypeDecl::F32) => TypeDecl::F32,
        (TypeDecl::I64, TypeDecl::I64) => TypeDecl::I64,
        (TypeDecl::I32, TypeDecl::I32) => TypeDecl::I32,
        (TypeDecl::Str, TypeDecl::Str) => TypeDecl::Str,
        (TypeDecl::Float, TypeDecl::Float) => TypeDecl::Float,
        (TypeDecl::Integer, TypeDecl::Integer) => TypeDecl::Integer,
        (TypeDecl::Float, TypeDecl::F64) | (TypeDecl::F64, TypeDecl::Float) => TypeDecl::F64,
        (TypeDecl::Float, TypeDecl::F32) | (TypeDecl::F32, TypeDecl::Float) => TypeDecl::F32,
        (TypeDecl::Integer, TypeDecl::I64) | (TypeDecl::I64, TypeDecl::Integer) => TypeDecl::I64,
        (TypeDecl::Integer, TypeDecl::I32) | (TypeDecl::I32, TypeDecl::Integer) => TypeDecl::I32,
        (TypeDecl::Array(lhs), TypeDecl::Array(rhs)) => {
            return binary_op_type(lhs, rhs);
        }
        _ => return Err(()),
    };
    Ok(res)
}

fn binary_cmp<'src, 'ast>(
    lhs: &'ast Expression<'src>,
    rhs: &'ast Expression<'src>,
    ctx: &mut TypeCheckContext<'src, 'ast, '_, '_>,
    op: &str,
) -> Result<TypeDecl, TypeCheckError> {
    binary_op_gen(lhs, rhs, ctx, op, binary_cmp_type)
}

/// Binary comparison operator type check. It will always return i32, which is used as a bool in this language.
fn binary_cmp_type(lhs: &TypeDecl, rhs: &TypeDecl) -> Result<TypeDecl, ()> {
    let res = match (&lhs, &rhs) {
        // Any type spreads contamination in the source code.
        (TypeDecl::Any, _) => TypeDecl::I32,
        (_, TypeDecl::Any) => TypeDecl::I32,
        (TypeDecl::F64, TypeDecl::F64) => TypeDecl::I32,
        (TypeDecl::F32, TypeDecl::F32) => TypeDecl::I32,
        (TypeDecl::I64, TypeDecl::I64) => TypeDecl::I32,
        (TypeDecl::I32, TypeDecl::I32) => TypeDecl::I32,
        (TypeDecl::Str, TypeDecl::Str) => TypeDecl::I32,
        (TypeDecl::Float, TypeDecl::Float) => TypeDecl::I32,
        (TypeDecl::Integer, TypeDecl::Integer) => TypeDecl::I32,
        (TypeDecl::Float, TypeDecl::F64 | TypeDecl::F32)
        | (TypeDecl::F64 | TypeDecl::F32, TypeDecl::Float) => TypeDecl::I32,
        (TypeDecl::Integer, TypeDecl::I64 | TypeDecl::I32)
        | (TypeDecl::I64 | TypeDecl::I32, TypeDecl::Integer) => TypeDecl::I32,
        (TypeDecl::Array(lhs), TypeDecl::Array(rhs)) => {
            return binary_op_type(lhs, rhs);
        }
        _ => return Err(()),
    };
    Ok(res)
}
