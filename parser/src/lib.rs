use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alpha1, alphanumeric1, char, multispace0, multispace1, none_of},
    combinator::{map_res, opt, recognize},
    multi::{fold_many0, many0, many1},
    number::complete::recognize_float,
    sequence::{delimited, pair, preceded, terminated, tuple},
    IResult,
};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

#[derive(Debug, PartialEq, Clone)]
pub enum TypeDecl {
    Any,
    F64,
    F32,
    I64,
    I32,
    Str,
    Array(Box<TypeDecl>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    F64(f64),
    F32(f32),
    I64(i64),
    I32(i32),
    Str(String),
    Array(TypeDecl, Vec<Rc<RefCell<Value>>>),
    Ref(Rc<RefCell<Value>>),
}

impl Value {
    /// We don't really need assignment operation for an array (yet), because
    /// array index will return a reference.
    fn _array_assign(&mut self, idx: usize, value: Value) {
        if let Value::Array(_, array) = self {
            array[idx] = Rc::new(RefCell::new(value.deref()));
        } else {
            panic!("assign_array must be called for an array")
        }
    }

    fn _array_get(&self, idx: u64) -> Value {
        match self {
            Value::Ref(rc) => rc.borrow()._array_get(idx),
            Value::Array(_, array) => array[idx as usize].borrow().clone(),
            _ => panic!("array index must be called for an array"),
        }
    }

    fn array_get_ref(&self, idx: u64) -> Value {
        match self {
            Value::Ref(rc) => rc.borrow().array_get_ref(idx),
            Value::Array(_, array) => Value::Ref(array[idx as usize].clone()),
            _ => panic!("array index must be called for an array"),
        }
    }

    fn array_push(&mut self, value: Value) {
        if let Value::Array(_, array) = self {
            array.push(Rc::new(RefCell::new(value.deref())));
        } else {
            panic!("push() must be called for an array")
        }
    }

    /// Returns the length of an array, dereferencing recursively if the value was a reference.
    fn array_len(&self) -> usize {
        match self {
            Value::Ref(rc) => rc.borrow().array_len(),
            Value::Array(_, array) => array.len(),
            _ => panic!("len() must be called for an array"),
        }
    }

    /// Recursively peels off references
    fn deref(self) -> Self {
        if let Value::Ref(r) = self {
            r.borrow().clone().deref()
        } else {
            self
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ArgDecl<'a>(&'a str, TypeDecl);

#[derive(Debug, PartialEq, Clone)]
pub enum Statement<'a> {
    Comment(&'a str),
    VarDecl(&'a str, TypeDecl, Option<Expression<'a>>),
    FnDecl(&'a str, Vec<ArgDecl<'a>>, Vec<Statement<'a>>),
    Expression(Expression<'a>),
    Loop(Vec<Statement<'a>>),
    While(Expression<'a>, Vec<Statement<'a>>),
    For(&'a str, Expression<'a>, Expression<'a>, Vec<Statement<'a>>),
    Break,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expression<'a> {
    NumLiteral(Value),
    StrLiteral(String),
    ArrLiteral(Vec<Expression<'a>>),
    Variable(&'a str),
    VarAssign(Box<Expression<'a>>, Box<Expression<'a>>),
    FnInvoke(&'a str, Vec<Expression<'a>>),
    ArrIndex(Box<Expression<'a>>, Vec<Expression<'a>>),
    Not(Box<Expression<'a>>),
    Add(Box<Expression<'a>>, Box<Expression<'a>>),
    Sub(Box<Expression<'a>>, Box<Expression<'a>>),
    Mult(Box<Expression<'a>>, Box<Expression<'a>>),
    Div(Box<Expression<'a>>, Box<Expression<'a>>),
    LT(Box<Expression<'a>>, Box<Expression<'a>>),
    GT(Box<Expression<'a>>, Box<Expression<'a>>),
    And(Box<Expression<'a>>, Box<Expression<'a>>),
    Or(Box<Expression<'a>>, Box<Expression<'a>>),
    Conditional(
        Box<Expression<'a>>,
        Vec<Statement<'a>>,
        Option<Vec<Statement<'a>>>,
    ),
    Brace(Vec<Statement<'a>>),
}

fn comment(input: &str) -> IResult<&str, Statement> {
    let (r, _) = multispace0(input)?;
    delimited(tag("/*"), take_until("*/"), tag("*/"))(r).map(|(r, s)| (r, Statement::Comment(s)))
}

pub fn identifier(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))(input)
}

fn ident_space(input: &str) -> IResult<&str, &str> {
    delimited(multispace0, identifier, multispace0)(input)
}

fn var_ref(input: &str) -> IResult<&str, Expression> {
    let (r, res) = ident_space(input)?;
    Ok((r, Expression::Variable(res)))
}

fn type_scalar(input: &str) -> IResult<&str, TypeDecl> {
    let (r, type_) = opt(delimited(
        multispace0,
        alt((tag("f64"), tag("f32"), tag("i64"), tag("i32"), tag("str"))),
        multispace0,
    ))(input)?;
    Ok((
        r,
        match type_ {
            Some("f64") | None => TypeDecl::F64,
            Some("f32") => TypeDecl::F32,
            Some("i32") => TypeDecl::I32,
            Some("i64") => TypeDecl::I64,
            Some("str") => TypeDecl::Str,
            Some(unknown) => panic!(format!("Unknown type: \"{}\"", unknown)),
        },
    ))
}

fn type_array(input: &str) -> IResult<&str, TypeDecl> {
    let (r, arr) = delimited(
        delimited(multispace0, tag("["), multispace0),
        alt((type_array, type_scalar)),
        delimited(multispace0, tag("]"), multispace0),
    )(input)?;
    Ok((r, TypeDecl::Array(Box::new(arr))))
}

fn type_spec(input: &str) -> IResult<&str, TypeDecl> {
    let (r, type_) = opt(delimited(
        delimited(multispace0, tag(":"), multispace0),
        alt((type_array, type_scalar)),
        multispace0,
    ))(input)?;
    Ok((
        r,
        if let Some(a) = type_ {
            a
        } else {
            TypeDecl::Any
        },
    ))
}

fn var_decl(input: &str) -> IResult<&str, Statement> {
    let (r, _) = multispace1(tag("var")(multispace0(input)?.0)?.0)?;
    let (r, ident) = identifier(r)?;
    let (r, ts) = type_spec(r)?;
    let (r, initializer) = opt(delimited(
        delimited(multispace0, tag("="), multispace0),
        full_expression,
        multispace0,
    ))(r)?;
    let (r, _) = char(';')(multispace0(r)?.0)?;
    Ok((r, Statement::VarDecl(ident, ts, initializer)))
}

fn double_expr(input: &str) -> IResult<&str, Expression> {
    let (r, v) = recognize_float(input)?;
    // For now we have very simple conditinon to decide if it is a floating point literal
    // by a presense of a period.
    Ok((
        r,
        Expression::NumLiteral(if v.contains('.') {
            let parsed = v.parse().map_err(|_| {
                nom::Err::Error(nom::error::Error {
                    input,
                    code: nom::error::ErrorKind::Digit,
                })
            })?;
            Value::F64(parsed)
        } else {
            Value::I64(v.parse().map_err(|_| {
                nom::Err::Error(nom::error::Error {
                    input,
                    code: nom::error::ErrorKind::Digit,
                })
            })?)
        }),
    ))
}

fn numeric_literal_expression(input: &str) -> IResult<&str, Expression> {
    delimited(multispace0, double_expr, multispace0)(input)
}

fn str_literal(input: &str) -> IResult<&str, Expression> {
    let (r, val) = delimited(
        preceded(multispace0, char('\"')),
        many0(none_of("\"")),
        terminated(char('"'), multispace0),
    )(input)?;
    Ok((
        r,
        Expression::StrLiteral(
            val.iter()
                .collect::<String>()
                .replace("\\\\", "\\")
                .replace("\\n", "\n"),
        ),
    ))
}

fn array_literal(input: &str) -> IResult<&str, Expression> {
    let (r, (mut val, last)) = delimited(
        multispace0,
        delimited(
            tag("["),
            pair(
                many0(terminated(full_expression, tag(","))),
                opt(full_expression),
            ),
            tag("]"),
        ),
        multispace0,
    )(input)?;
    if let Some(last) = last {
        val.push(last);
    }
    Ok((r, Expression::ArrLiteral(val)))
}

// We parse any expr surrounded by parens, ignoring all whitespaces around those
fn parens(i: &str) -> IResult<&str, Expression> {
    delimited(
        multispace0,
        delimited(tag("("), conditional_expr, tag(")")),
        multispace0,
    )(i)
}

fn func_invoke(i: &str) -> IResult<&str, Expression> {
    let (r, ident) = delimited(multispace0, identifier, multispace0)(i)?;
    // println!("func_invoke ident: {}", ident);
    let (r, args) = delimited(
        multispace0,
        delimited(
            tag("("),
            many0(delimited(
                multispace0,
                expr,
                delimited(multispace0, opt(tag(",")), multispace0),
            )),
            tag(")"),
        ),
        multispace0,
    )(r)?;
    Ok((r, Expression::FnInvoke(ident, args)))
}

fn array_index(i: &str) -> IResult<&str, Expression> {
    let (r, (prim, indices)) = pair(
        primary_expression,
        many1(delimited(
            multispace0,
            delimited(
                tag("["),
                many0(delimited(
                    multispace0,
                    full_expression,
                    delimited(multispace0, opt(tag(",")), multispace0),
                )),
                tag("]"),
            ),
            multispace0,
        )),
    )(i)?;
    Ok((
        r,
        indices
            .into_iter()
            .fold(prim, |acc, v| Expression::ArrIndex(Box::new(acc), v)),
    ))
}

fn primary_expression(i: &str) -> IResult<&str, Expression> {
    alt((
        numeric_literal_expression,
        str_literal,
        array_literal,
        var_ref,
        parens,
        brace_expr,
    ))(i)
}

fn postfix_expression(i: &str) -> IResult<&str, Expression> {
    alt((func_invoke, array_index, primary_expression))(i)
}

fn not(i: &str) -> IResult<&str, Expression> {
    let (r, v) = preceded(delimited(multispace0, tag("!"), multispace0), not_factor)(i)?;
    Ok((r, Expression::Not(Box::new(v))))
}

fn not_factor(i: &str) -> IResult<&str, Expression> {
    alt((not, postfix_expression))(i)
}

// We read an initial factor and for each time we find
// a * or / operator followed by another factor, we do
// the math by folding everything
fn term(i: &str) -> IResult<&str, Expression> {
    let (i, init) = not_factor(i)?;

    fold_many0(
        pair(alt((char('*'), char('/'))), not_factor),
        init,
        |acc, (op, val): (char, Expression)| {
            if op == '*' {
                Expression::Mult(Box::new(acc), Box::new(val))
            } else {
                Expression::Div(Box::new(acc), Box::new(val))
            }
        },
    )(i)
}

fn expr(i: &str) -> IResult<&str, Expression> {
    let (i, init) = term(i)?;

    fold_many0(
        pair(alt((char('+'), char('-'))), term),
        init,
        |acc, (op, val): (char, Expression)| {
            if op == '+' {
                Expression::Add(Box::new(acc), Box::new(val))
            } else {
                Expression::Sub(Box::new(acc), Box::new(val))
            }
        },
    )(i)
}

fn cmp(i: &str) -> IResult<&str, Expression> {
    let (i, lhs) = expr(i)?;

    let (i, (op, val)) = pair(alt((char('<'), char('>'))), expr)(i)?;
    Ok((
        i,
        if op == '<' {
            Expression::LT(Box::new(lhs), Box::new(val))
        } else {
            Expression::GT(Box::new(lhs), Box::new(val))
        },
    ))
}

fn conditional(i: &str) -> IResult<&str, Expression> {
    let (r, _) = delimited(multispace0, tag("if"), multispace0)(i)?;
    let (r, cond) = or_expr(r)?;
    let (r, true_branch) = delimited(
        delimited(multispace0, tag("{"), multispace0),
        source,
        delimited(multispace0, tag("}"), multispace0),
    )(r)?;
    let (r, false_branch) = opt(preceded(
        delimited(multispace0, tag("else"), multispace0),
        alt((
            delimited(
                delimited(multispace0, tag("{"), multispace0),
                source,
                delimited(multispace0, tag("}"), multispace0),
            ),
            map_res(
                conditional,
                |v| -> Result<Vec<Statement>, nom::error::Error<&str>> {
                    Ok(vec![Statement::Expression(v)])
                },
            ),
        )),
    ))(r)?;
    Ok((
        r,
        Expression::Conditional(Box::new(cond), true_branch, false_branch),
    ))
}

fn var_assign(input: &str) -> IResult<&str, Expression> {
    let (r, res) = tuple((cmp_expr, char('='), cmp_expr))(input)?;
    Ok((r, Expression::VarAssign(Box::new(res.0), Box::new(res.2))))
}

fn cmp_expr(i: &str) -> IResult<&str, Expression> {
    alt((cmp, expr))(i)
}

fn and(i: &str) -> IResult<&str, Expression> {
    let (r, first) = cmp_expr(i)?;
    let (r, _) = delimited(multispace0, tag("&&"), multispace0)(r)?;
    let (r, second) = cmp_expr(r)?;
    Ok((r, Expression::And(Box::new(first), Box::new(second))))
}

fn and_expr(i: &str) -> IResult<&str, Expression> {
    alt((and, cmp_expr))(i)
}

fn or(i: &str) -> IResult<&str, Expression> {
    let (r, first) = and_expr(i)?;
    let (r, _) = delimited(multispace0, tag("||"), multispace0)(r)?;
    let (r, second) = and_expr(r)?;
    Ok((r, Expression::Or(Box::new(first), Box::new(second))))
}

fn or_expr(i: &str) -> IResult<&str, Expression> {
    alt((or, and_expr))(i)
}

fn assign_expr(i: &str) -> IResult<&str, Expression> {
    alt((var_assign, or_expr))(i)
}

fn conditional_expr(i: &str) -> IResult<&str, Expression> {
    alt((conditional, assign_expr))(i)
}

fn brace_expr(input: &str) -> IResult<&str, Expression> {
    let (r, v) = delimited(
        delimited(multispace0, tag("{"), multispace0),
        source,
        delimited(multispace0, tag("}"), multispace0),
    )(input)?;
    Ok((r, Expression::Brace(v)))
}

fn full_expression(input: &str) -> IResult<&str, Expression> {
    conditional_expr(input)
}

fn expression_statement(input: &str) -> IResult<&str, Statement> {
    let (r, val) = full_expression(input)?;
    Ok((r, Statement::Expression(val)))
}

fn func_arg(input: &str) -> IResult<&str, ArgDecl> {
    let (r, v) = pair(
        identifier,
        opt(delimited(multispace0, type_spec, multispace0)),
    )(input)?;
    Ok((r, ArgDecl(v.0, v.1.unwrap_or(TypeDecl::F64))))
}

fn func_decl(input: &str) -> IResult<&str, Statement> {
    let (r, _) = multispace1(tag("fn")(multispace0(input)?.0)?.0)?;
    let (r, ident) = identifier(r)?;
    let (r, args) = delimited(
        multispace0,
        delimited(
            tag("("),
            many0(delimited(
                multispace0,
                func_arg,
                delimited(multispace0, opt(tag(",")), multispace0),
            )),
            tag(")"),
        ),
        multispace0,
    )(r)?;
    let (r, stmts) = delimited(
        delimited(multispace0, tag("{"), multispace0),
        source,
        delimited(multispace0, tag("}"), multispace0),
    )(r)?;
    Ok((r, Statement::FnDecl(ident, args, stmts)))
}

fn loop_stmt(input: &str) -> IResult<&str, Statement> {
    let (r, _) = multispace0(tag("loop")(multispace0(input)?.0)?.0)?;
    let (r, stmts) = delimited(
        delimited(multispace0, tag("{"), multispace0),
        source,
        delimited(multispace0, tag("}"), multispace0),
    )(r)?;
    Ok((r, Statement::Loop(stmts)))
}

fn while_stmt(input: &str) -> IResult<&str, Statement> {
    let (r, _) = multispace0(tag("while")(multispace0(input)?.0)?.0)?;
    let (r, cond) = cmp_expr(r)?;
    let (r, stmts) = delimited(
        delimited(multispace0, tag("{"), multispace0),
        source,
        delimited(multispace0, tag("}"), multispace0),
    )(r)?;
    Ok((r, Statement::While(cond, stmts)))
}

fn for_stmt(input: &str) -> IResult<&str, Statement> {
    let (r, _) = delimited(multispace0, tag("for"), multispace1)(input)?;
    let (r, iter) = identifier(r)?;
    let (r, _) = delimited(multispace0, tag("in"), multispace0)(r)?;
    let (r, from) = expr(r)?;
    let (r, _) = delimited(multispace0, tag(".."), multispace0)(r)?;
    let (r, to) = expr(r)?;
    let (r, stmts) = delimited(
        delimited(multispace0, tag("{"), multispace0),
        source,
        delimited(multispace0, tag("}"), multispace0),
    )(r)?;
    Ok((r, Statement::For(iter, from, to, stmts)))
}

fn break_stmt(input: &str) -> IResult<&str, Statement> {
    let (r, _) = delimited(multispace0, tag("break"), multispace0)(input)?;
    Ok((r, Statement::Break))
}

fn general_statement<'a>(last: bool) -> impl Fn(&'a str) -> IResult<&'a str, Statement> {
    let terminator = move |i| -> IResult<&str, ()> {
        let mut semicolon = pair(tag(";"), multispace0);
        if last {
            Ok((opt(semicolon)(i)?.0, ()))
        } else {
            Ok((semicolon(i)?.0, ()))
        }
    };
    move |input: &str| {
        alt((
            var_decl,
            func_decl,
            loop_stmt,
            while_stmt,
            for_stmt,
            terminated(break_stmt, terminator),
            terminated(expression_statement, terminator),
            comment,
        ))(input)
    }
}

fn last_statement(input: &str) -> IResult<&str, Statement> {
    general_statement(true)(input)
}

fn statement(input: &str) -> IResult<&str, Statement> {
    general_statement(false)(input)
}

pub fn source(input: &str) -> IResult<&str, Vec<Statement>> {
    let (r, mut v) = many0(statement)(input)?;
    let (r, last) = opt(last_statement)(r)?;
    if let Some(last) = last {
        v.push(last);
    }
    Ok((r, v))
}

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
            RunResult::Break => return RunResult::Break,
        }
    };
}

fn binary_op(
    lhs: Value,
    rhs: Value,
    d: impl Fn(f64, f64) -> f64,
    i: impl Fn(i64, i64) -> i64,
) -> Value {
    match (lhs.clone(), rhs.clone()) {
        (Value::F64(lhs), rhs) => Value::F64(d(lhs, coerce_f64(&rhs))),
        (lhs, Value::F64(rhs)) => Value::F64(d(coerce_f64(&lhs), rhs)),
        (Value::F32(lhs), rhs) => Value::F32(d(lhs as f64, coerce_f64(&rhs)) as f32),
        (lhs, Value::F32(rhs)) => Value::F32(d(coerce_f64(&lhs), rhs as f64) as f32),
        (Value::I64(lhs), Value::I64(rhs)) => Value::I64(i(lhs, rhs)),
        (Value::I64(lhs), Value::I32(rhs)) => Value::I64(i(lhs, rhs as i64)),
        (Value::I32(lhs), Value::I64(rhs)) => Value::I64(i(lhs as i64, rhs)),
        (Value::I32(lhs), Value::I32(rhs)) => Value::I32(i(lhs as i64, rhs as i64) as i32),
        _ => panic!(format!(
            "Unsupported addition between {:?} and {:?}",
            lhs, rhs
        )),
    }
}

fn truthy(a: &Value) -> bool {
    match *a {
        Value::F64(v) => v != 0.,
        Value::F32(v) => v != 0.,
        Value::I64(v) => v != 0,
        Value::I32(v) => v != 0,
        _ => false,
    }
}

fn coerce_f64(a: &Value) -> f64 {
    match *a {
        Value::F64(v) => v as f64,
        Value::F32(v) => v as f64,
        Value::I64(v) => v as f64,
        Value::I32(v) => v as f64,
        _ => 0.,
    }
}

fn coerce_i64(a: &Value) -> i64 {
    match *a {
        Value::F64(v) => v as i64,
        Value::F32(v) => v as i64,
        Value::I64(v) => v as i64,
        Value::I32(v) => v as i64,
        _ => 0,
    }
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

fn _coerce_var(value: &Value, target: &Value) -> Value {
    match target {
        Value::F64(_) => Value::F64(coerce_f64(value)),
        Value::F32(_) => Value::F32(coerce_f64(value) as f32),
        Value::I64(_) => Value::I64(coerce_i64(value)),
        Value::I32(_) => Value::I32(coerce_i64(value) as i32),
        Value::Str(_) => Value::Str(coerce_str(value)),
        Value::Array(inner_type, inner) => {
            if inner.len() == 0 {
                if let Value::Array(_, value_inner) = value {
                    if value_inner.len() == 0 {
                        return value.clone();
                    }
                }
                panic!("Cannot coerce type to empty array");
            } else {
                if let Value::Array(_, value_inner) = value {
                    Value::Array(
                        inner_type.clone(),
                        value_inner
                            .iter()
                            .map(|val| {
                                Rc::new(RefCell::new(coerce_type(&val.borrow(), inner_type)))
                            })
                            .collect(),
                    )
                } else {
                    panic!("Cannot coerce scalar to array");
                }
            }
        }
        // We usually don't care about coercion
        Value::Ref(_) => value.clone(),
    }
}

fn coerce_type(value: &Value, target: &TypeDecl) -> Value {
    match target {
        TypeDecl::Any => value.clone(),
        TypeDecl::F64 => Value::F64(coerce_f64(value)),
        TypeDecl::F32 => Value::F32(coerce_f64(value) as f32),
        TypeDecl::I64 => Value::I64(coerce_i64(value)),
        TypeDecl::I32 => Value::I32(coerce_i64(value) as i32),
        TypeDecl::Str => Value::Str(coerce_str(value)),
        TypeDecl::Array(inner) => {
            if let Value::Array(_, value_inner) = value {
                Value::Array(
                    (**inner).clone(),
                    value_inner
                        .iter()
                        .map(|value_elem| {
                            Rc::new(RefCell::new(coerce_type(&value_elem.borrow(), inner)))
                        })
                        .collect(),
                )
            } else {
                panic!(format!("Incompatible type to array! {:?}", value));
            }
        }
    }
}

fn eval<'a, 'b>(e: &'b Expression<'a>, ctx: &mut EvalContext<'a, 'b, '_, '_>) -> RunResult {
    match e {
        Expression::NumLiteral(val) => RunResult::Yield(val.clone()),
        Expression::StrLiteral(val) => RunResult::Yield(Value::Str(val.clone())),
        Expression::ArrLiteral(val) => RunResult::Yield(Value::Array(
            TypeDecl::Any,
            val.iter()
                .map(|v| {
                    if let RunResult::Yield(y) = eval(v, ctx) {
                        Rc::new(RefCell::new(y))
                    } else {
                        panic!("Break in array literal not supported");
                    }
                })
                .collect(),
        )),
        Expression::Variable(str) => RunResult::Yield(Value::Ref(
            ctx.get_var_rc(str)
                .expect(&format!("Variable {} not found in scope", str)),
        )),
        Expression::VarAssign(lhs, rhs) => {
            let lhs_value = eval(lhs, ctx);
            let lhs_value = if let RunResult::Yield(Value::Ref(rc)) = lhs_value {
                rc
            } else {
                panic!(format!(
                    "We need variable reference on lhs to assign. Actually we got {:?}",
                    lhs_value
                ))
            };
            let rhs_value = unwrap_run!(eval(rhs, ctx));
            *lhs_value.borrow_mut() = rhs_value.clone();
            RunResult::Yield(rhs_value)
        }
        Expression::FnInvoke(str, args) => {
            let args = args.iter().map(|v| eval(v, ctx)).collect::<Vec<_>>();
            let mut subctx = EvalContext::push_stack(ctx);
            let func = ctx
                .get_fn(*str)
                .expect(&format!("function {} is not defined.", str));
            match func {
                FuncDef::Code(func) => {
                    for (k, v) in func.args.iter().zip(&args) {
                        subctx.variables.borrow_mut().insert(
                            k.0,
                            Rc::new(RefCell::new(coerce_type(&unwrap_run!(v.clone()), &k.1))),
                        );
                    }
                    let run_result = run(func.stmts, &mut subctx).unwrap();
                    match unwrap_deref(run_result) {
                        RunResult::Yield(v) => RunResult::Yield(v),
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
            let args = args.iter().map(|v| eval(v, ctx)).collect::<Vec<_>>();
            let arg0 = match unwrap_deref(args[0].clone()) {
                RunResult::Yield(v) => {
                    if let Value::I64(idx) = coerce_type(&v, &TypeDecl::I64) {
                        idx as u64
                    } else {
                        panic!("Subscript type should be integer types");
                    }
                }
                RunResult::Break => {
                    return RunResult::Break;
                }
            };
            let result = unwrap_run!(eval(ex, ctx));
            RunResult::Yield(result.array_get_ref(arg0))
        }
        Expression::Not(val) => {
            RunResult::Yield(Value::I32(if truthy(&unwrap_run!(eval(val, ctx))) {
                0
            } else {
                1
            }))
        }
        Expression::Add(lhs, rhs) => {
            let res = RunResult::Yield(binary_op(
                unwrap_run!(eval(lhs, ctx)),
                unwrap_run!(eval(rhs, ctx)),
                |lhs, rhs| lhs + rhs,
                |lhs, rhs| lhs + rhs,
            ));
            res
        }
        Expression::Sub(lhs, rhs) => RunResult::Yield(binary_op(
            unwrap_run!(eval(lhs, ctx)),
            unwrap_run!(eval(rhs, ctx)),
            |lhs, rhs| lhs - rhs,
            |lhs, rhs| lhs - rhs,
        )),
        Expression::Mult(lhs, rhs) => RunResult::Yield(binary_op(
            unwrap_run!(eval(lhs, ctx)),
            unwrap_run!(eval(rhs, ctx)),
            |lhs, rhs| lhs * rhs,
            |lhs, rhs| lhs * rhs,
        )),
        Expression::Div(lhs, rhs) => RunResult::Yield(binary_op(
            unwrap_run!(eval(lhs, ctx)),
            unwrap_run!(eval(rhs, ctx)),
            |lhs, rhs| lhs / rhs,
            |lhs, rhs| lhs / rhs,
        )),
        Expression::LT(lhs, rhs) => RunResult::Yield(binary_op(
            unwrap_run!(eval(lhs, ctx)),
            unwrap_run!(eval(rhs, ctx)),
            |lhs, rhs| if lhs < rhs { 1. } else { 0. },
            |lhs, rhs| if lhs < rhs { 1 } else { 0 },
        )),
        Expression::GT(lhs, rhs) => RunResult::Yield(binary_op(
            unwrap_run!(eval(lhs, ctx)),
            unwrap_run!(eval(rhs, ctx)),
            |lhs, rhs| if lhs > rhs { 1. } else { 0. },
            |lhs, rhs| if lhs > rhs { 1 } else { 0 },
        )),
        Expression::And(lhs, rhs) => RunResult::Yield(Value::I32(
            if truthy(&unwrap_run!(eval(lhs, ctx))) && truthy(&unwrap_run!(eval(rhs, ctx))) {
                1
            } else {
                0
            },
        )),
        Expression::Or(lhs, rhs) => RunResult::Yield(Value::I32(
            if truthy(&unwrap_run!(eval(lhs, ctx))) || truthy(&unwrap_run!(eval(rhs, ctx))) {
                1
            } else {
                0
            },
        )),
        Expression::Conditional(cond, true_branch, false_branch) => {
            if truthy(&unwrap_run!(eval(cond, ctx))) {
                run(true_branch, ctx).unwrap()
            } else if let Some(ast) = false_branch {
                run(ast, ctx).unwrap()
            } else {
                RunResult::Yield(Value::I32(0))
            }
        }
        Expression::Brace(stmts) => {
            let mut subctx = EvalContext::push_stack(ctx);
            run(stmts, &mut subctx).unwrap()
        }
    }
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

#[derive(Clone)]
pub struct FuncCode<'src, 'ast> {
    args: &'ast Vec<ArgDecl<'src>>,
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

#[derive(Debug, PartialEq, Clone)]
pub enum RunResult {
    Yield(Value),
    Break,
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
) -> Result<RunResult, ()> {
    let mut res = RunResult::Yield(Value::I32(0));
    for stmt in stmts {
        match stmt {
            Statement::VarDecl(var, type_, initializer) => {
                let init_val = if let Some(init_expr) = initializer {
                    unwrap_break!(eval(init_expr, ctx))
                } else {
                    Value::I32(0)
                };
                let init_val = coerce_type(&init_val, type_);
                ctx.variables
                    .borrow_mut()
                    .insert(*var, Rc::new(RefCell::new(init_val)));
            }
            Statement::FnDecl(var, args, stmts) => {
                ctx.functions
                    .insert(var.to_string(), FuncDef::Code(FuncCode { args, stmts }));
            }
            Statement::Expression(e) => {
                res = eval(&e, ctx);
                if let RunResult::Break = res {
                    return Ok(res);
                }
                // println!("Expression evaluates to: {:?}", res);
            }
            Statement::Loop(e) => loop {
                res = RunResult::Yield(unwrap_break!(run(e, ctx)?));
            },
            Statement::While(cond, e) => loop {
                match unwrap_deref(eval(cond, ctx)) {
                    RunResult::Yield(v) => {
                        if truthy(&v) {
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
                let from_res = coerce_i64(&unwrap_break!(eval(from, ctx))) as i64;
                let to_res = coerce_i64(&unwrap_break!(eval(to, ctx))) as i64;
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
