use super::interpreter::EvalError;
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
use std::{
    cell::RefCell,
    io::{Read, Write},
    rc::Rc,
    string::FromUtf8Error,
};

#[derive(Debug, PartialEq, Clone)]
#[repr(u8)]
pub enum TypeDecl {
    Any,
    F64,
    F32,
    I64,
    I32,
    Str,
    Array(Box<TypeDecl>),
}

impl TypeDecl {
    fn serialize(&self, writer: &mut impl Write) -> std::io::Result<()> {
        let tag = match self {
            Self::Any => 0xff,
            Self::F64 => F64_TAG,
            Self::F32 => F32_TAG,
            Self::I64 => I64_TAG,
            Self::I32 => I32_TAG,
            Self::Str => STR_TAG,
            Self::Array(inner) => {
                writer.write_all(&ARRAY_TAG.to_le_bytes())?;
                inner.serialize(writer)?;
                return Ok(());
            }
        };
        writer.write_all(&tag.to_le_bytes())?;
        Ok(())
    }

    fn deserialize(reader: &mut impl Read) -> std::io::Result<Self> {
        macro_rules! read {
            ($ty:ty) => {{
                let mut buf = [0u8; std::mem::size_of::<$ty>()];
                reader.read_exact(&mut buf)?;
                <$ty>::from_le_bytes(buf)
            }};
        }

        let tag = read!(u8);
        Ok(match tag {
            0xff => Self::Any,
            F64_TAG => Self::F64,
            F32_TAG => Self::F32,
            I64_TAG => Self::I64,
            I32_TAG => Self::I32,
            STR_TAG => Self::Str,
            ARRAY_TAG => Self::Array(Box::new(Self::deserialize(reader)?)),
            REF_TAG => todo!(),
            _ => unreachable!(),
        })
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ArrayInt {
    pub(crate) type_decl: TypeDecl,
    pub(crate) values: Vec<Rc<RefCell<Value>>>,
}

impl ArrayInt {
    pub(crate) fn new(type_decl: TypeDecl, values: Vec<Rc<RefCell<Value>>>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self { type_decl, values }))
    }

    pub fn values(&self) -> &[Rc<RefCell<Value>>] {
        &self.values
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    F64(f64),
    F32(f32),
    I64(i64),
    I32(i32),
    Str(String),
    Array(Rc<RefCell<ArrayInt>>),
    Ref(Rc<RefCell<Value>>),
}

impl Default for Value {
    fn default() -> Self {
        Self::I64(0)
    }
}

const F64_TAG: u8 = 0;
const F32_TAG: u8 = 1;
const I64_TAG: u8 = 2;
const I32_TAG: u8 = 3;
const STR_TAG: u8 = 4;
const ARRAY_TAG: u8 = 5;
const REF_TAG: u8 = 6;

#[derive(Debug)]
pub enum ReadError {
    IO(std::io::Error),
    FromUtf8(FromUtf8Error),
}

impl From<std::io::Error> for ReadError {
    fn from(e: std::io::Error) -> Self {
        ReadError::IO(e)
    }
}

impl From<FromUtf8Error> for ReadError {
    fn from(e: FromUtf8Error) -> Self {
        ReadError::FromUtf8(e)
    }
}

impl From<ReadError> for String {
    fn from(e: ReadError) -> Self {
        match e {
            ReadError::IO(e) => e.to_string(),
            ReadError::FromUtf8(e) => e.to_string(),
        }
    }
}

impl ToString for Value {
    fn to_string(&self) -> String {
        match self {
            Self::F64(v) => v.to_string(),
            Self::F32(v) => v.to_string(),
            Self::I64(v) => v.to_string(),
            Self::I32(v) => v.to_string(),
            Self::Str(v) => v.clone(),
            Self::Array(v) => format!(
                "[{}]",
                &v.borrow().values.iter().fold("".to_string(), |acc, cur| {
                    if acc.is_empty() {
                        cur.borrow().to_string()
                    } else {
                        acc + ", " + &cur.borrow().to_string()
                    }
                })
            ),
            Self::Ref(v) => "&".to_string() + &v.borrow().to_string(),
        }
    }
}

impl Value {
    pub(crate) fn serialize(&self, writer: &mut impl Write) -> std::io::Result<()> {
        macro_rules! serialize_with_tag {
            ($tag:ident, $val:expr) => {{
                writer.write_all(&$tag.to_le_bytes())?;
                writer.write_all(&$val.to_le_bytes())?;
                Ok(())
            }};
        }

        match self {
            Self::F64(val) => serialize_with_tag!(F64_TAG, val),
            Self::F32(val) => serialize_with_tag!(F32_TAG, val),
            Self::I64(val) => serialize_with_tag!(I64_TAG, val),
            Self::I32(val) => serialize_with_tag!(I32_TAG, val),
            Self::Str(val) => {
                writer.write_all(&STR_TAG.to_le_bytes())?;
                writer.write_all(&(val.len() as u32).to_le_bytes())?;
                writer.write_all(val.as_bytes())?;
                Ok(())
            }
            Self::Array(rc) => {
                let ArrayInt {
                    type_decl: decl,
                    values,
                } = &rc.borrow() as &ArrayInt;
                writer.write_all(&ARRAY_TAG.to_le_bytes())?;
                writer.write_all(&values.len().to_le_bytes())?;
                decl.serialize(writer)?;
                for value in values {
                    value.borrow().serialize(writer)?;
                }
                Ok(())
            }
            Self::Ref(val) => {
                writer.write_all(&REF_TAG.to_le_bytes())?;
                val.borrow().serialize(writer)?;
                Ok(())
            }
        }
    }

    pub(crate) fn deserialize(reader: &mut impl Read) -> Result<Self, ReadError> {
        let mut tag = [0u8; 1];
        reader.read_exact(&mut tag)?;

        macro_rules! parse {
            ($typ:ty) => {{
                let mut buf = [0u8; std::mem::size_of::<$typ>()];
                reader.read_exact(&mut buf)?;
                <$typ>::from_le_bytes(buf)
            }};
        }

        Ok(match tag[0] {
            F64_TAG => Value::F64(parse!(f64)),
            F32_TAG => Value::F32(parse!(f32)),
            I64_TAG => Value::I64(parse!(i64)),
            I32_TAG => Value::I32(parse!(i32)),
            STR_TAG => Value::Str({
                let len = parse!(u32);
                let mut buf = vec![0u8; len as usize];
                reader.read_exact(&mut buf)?;
                String::from_utf8(buf)?
            }),
            ARRAY_TAG => {
                let value_count = parse!(usize);
                let decl = TypeDecl::deserialize(reader)?;
                let values = (0..value_count)
                    .map(|_| Value::deserialize(reader).map(RefCell::new).map(Rc::new))
                    .collect::<Result<_, _>>()?;
                Self::Array(ArrayInt::new(decl, values))
            }
            _ => todo!(),
        })
    }

    /// We don't really need assignment operation for an array (yet), because
    /// array index will return a reference.
    fn _array_assign(&mut self, idx: usize, value: Value) {
        if let Value::Array(array) = self {
            array.borrow_mut().values[idx] = Rc::new(RefCell::new(value.deref()));
        } else {
            panic!("assign_array must be called for an array")
        }
    }

    fn _array_get(&self, idx: u64) -> Value {
        match self {
            Value::Ref(rc) => rc.borrow()._array_get(idx),
            Value::Array(array) => array.borrow_mut().values[idx as usize].borrow().clone(),
            _ => panic!("array index must be called for an array"),
        }
    }

    pub fn array_get_ref(&self, idx: u64) -> Result<Value, EvalError> {
        Ok(match self {
            Value::Ref(rc) => rc.borrow().array_get_ref(idx)?,
            Value::Array(array) => Value::Ref({
                let array = array.borrow();
                array.values.get(idx as usize)
                    .ok_or_else(|| format!("array index out of range: {idx} is larger than array length {}", array.values.len()))?
                    .clone()
            }),
            _ => return Err("array index must be called for an array".to_string()),
        })
    }

    pub fn array_push(&self, value: Value) -> Result<(), EvalError> {
        match self {
            Value::Ref(r) => r.borrow_mut().array_push(value),
            Value::Array(array) => {
                array
                    .borrow_mut()
                    .values
                    .push(Rc::new(RefCell::new(value.deref())));
                Ok(())
            }
            _ => Err("push() must be called for an array".to_string()),
        }
    }

    /// Returns the length of an array, dereferencing recursively if the value was a reference.
    pub fn array_len(&self) -> usize {
        match self {
            Value::Ref(rc) => rc.borrow().array_len(),
            Value::Array(array) => array.borrow().values.len(),
            _ => panic!("len() must be called for an array"),
        }
    }

    /// Recursively peels off references
    pub fn deref(self) -> Self {
        if let Value::Ref(r) = self {
            r.borrow().clone().deref()
        } else {
            self
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ArgDecl<'a>(pub &'a str, pub TypeDecl);

#[derive(Debug, PartialEq, Clone)]
pub enum Statement<'a> {
    Comment(&'a str),
    VarDecl(&'a str, TypeDecl, Option<Expression<'a>>),
    FnDecl {
        name: &'a str,
        args: Vec<ArgDecl<'a>>,
        ret_type: Option<TypeDecl>,
        stmts: Vec<Statement<'a>>,
    },
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

pub(crate) fn var_ref(input: &str) -> IResult<&str, Expression> {
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
            Some(unknown) => panic!("Unknown type: \"{}\"", unknown),
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

pub(crate) fn type_decl(input: &str) -> IResult<&str, TypeDecl> {
    alt((type_array, type_scalar))(input)
}

pub(crate) fn type_spec(input: &str) -> IResult<&str, TypeDecl> {
    let (r, type_) = opt(delimited(
        delimited(multispace0, tag(":"), multispace0),
        type_decl,
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

pub(crate) fn array_literal(input: &str) -> IResult<&str, Expression> {
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

pub(crate) fn func_invoke(i: &str) -> IResult<&str, Expression> {
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

pub(crate) fn array_index(i: &str) -> IResult<&str, Expression> {
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

pub(crate) fn primary_expression(i: &str) -> IResult<&str, Expression> {
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

pub(crate) fn expr(i: &str) -> IResult<&str, Expression> {
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

pub(crate) fn conditional(i: &str) -> IResult<&str, Expression> {
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

pub(crate) fn var_assign(input: &str) -> IResult<&str, Expression> {
    let (r, res) = tuple((cmp_expr, char('='), assign_expr))(input)?;
    Ok((r, Expression::VarAssign(Box::new(res.0), Box::new(res.2))))
}

pub(crate) fn cmp_expr(i: &str) -> IResult<&str, Expression> {
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

pub(crate) fn conditional_expr(i: &str) -> IResult<&str, Expression> {
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

pub(crate) fn full_expression(input: &str) -> IResult<&str, Expression> {
    conditional_expr(input)
}

fn expression_statement(input: &str) -> IResult<&str, Statement> {
    let (r, val) = full_expression(input)?;
    Ok((r, Statement::Expression(val)))
}

pub(crate) fn func_arg(input: &str) -> IResult<&str, ArgDecl> {
    let (r, v) = pair(
        identifier,
        opt(delimited(multispace0, type_spec, multispace0)),
    )(input)?;
    Ok((r, ArgDecl(v.0, v.1.unwrap_or(TypeDecl::F64))))
}

pub(crate) fn func_decl(input: &str) -> IResult<&str, Statement> {
    let (r, _) = multispace1(tag("fn")(multispace0(input)?.0)?.0)?;
    let (r, name) = identifier(r)?;
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
    let (r, ret_type) = opt(preceded(
        delimited(multispace0, tag("->"), multispace0),
        type_decl,
    ))(r)?;
    let (r, stmts) = delimited(
        delimited(multispace0, tag("{"), multispace0),
        source,
        delimited(multispace0, tag("}"), multispace0),
    )(r)?;
    Ok((
        r,
        Statement::FnDecl {
            name,
            args,
            ret_type,
            stmts,
        },
    ))
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

pub(crate) fn last_statement(input: &str) -> IResult<&str, Statement> {
    general_statement(true)(input)
}

pub(crate) fn statement(input: &str) -> IResult<&str, Statement> {
    general_statement(false)(input)
}

pub fn source(input: &str) -> IResult<&str, Vec<Statement>> {
    let (r, mut v) = many0(statement)(input)?;
    let (r, last) = opt(last_statement)(r)?;
    let (r, _) = opt(multispace0)(r)?;
    if let Some(last) = last {
        v.push(last);
    }
    Ok((r, v))
}

mod test;
