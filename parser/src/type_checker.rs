use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{
    parser::{Expression, Statement},
    EvalError, FuncDef, TypeDecl, Value,
};

#[derive(Debug, PartialEq, Clone)]
pub enum TypeCheckResult {
    Yield(TypeDecl),
    Break,
}

#[derive(Clone)]
pub struct TypeCheckContext<'src, 'ast, 'native, 'ctx> {
    /// RefCell to allow mutation in super context.
    /// Also, the inner values must be Rc of RefCell because a reference could be returned from
    /// a function so that the variable scope may have been ended.
    variables: RefCell<HashMap<&'src str, Rc<RefCell<TypeDecl>>>>,
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
            functions: HashMap::new(),
            super_context: None,
        }
    }
}

fn tc_expr<'a, 'b>(
    e: &'b Expression<'a>,
    _ctx: &mut TypeCheckContext<'a, 'b, '_, '_>,
) -> Result<TypeCheckResult, EvalError> {
    Ok(match e {
        Expression::NumLiteral(val) => TypeCheckResult::Yield(match val {
            Value::F64(_) | Value::F32(_) => TypeDecl::Float,
            Value::I64(_) | Value::I32(_) => TypeDecl::Integer,
            _ => return Err("Numeric literal has a non-number value".to_string()),
        }),
        _ => todo!(),
    })
}

fn tc_coerce_type(value: &TypeDecl, target: &TypeDecl) -> Result<TypeDecl, EvalError> {
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
) -> Result<TypeCheckResult, EvalError> {
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
                ctx.variables
                    .borrow_mut()
                    .insert(*var, Rc::new(RefCell::new(init_val)));
            }
            _ => todo!(),
        }
    }
    Ok(res)
}
