use super::interpreter::EvalError;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alpha1, alphanumeric1, char, multispace0, multispace1, none_of},
    combinator::{map_res, opt, recognize},
    multi::{fold_many0, many0, many1},
    number::complete::recognize_float,
    sequence::{delimited, pair, preceded, terminated, tuple},
    IResult, InputTake, Offset,
};
use nom_locate::LocatedSpan;
use std::{
    cell::RefCell,
    io::{Read, Write},
    rc::Rc,
    string::FromUtf8Error,
};

pub type Span<'a> = LocatedSpan<&'a str>;

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
    /// An abstract type that can match F64 or F32
    Float,
    /// An abstract type that can match I64 or I32
    Integer,
}

impl TypeDecl {
    pub(crate) fn _from_value(value: &Value) -> Self {
        match value {
            Value::F64(_) => Self::F64,
            Value::F32(_) => Self::F32,
            Value::I32(_) => Self::I32,
            Value::I64(_) => Self::I64,
            Value::Str(_) => Self::Str,
            Value::Array(a) => Self::Array(Box::new(a.borrow().type_decl.clone())),
            Value::Ref(a) => Self::_from_value(&*a.borrow()),
            Value::ArrayRef(a, _) => a.borrow().type_decl.clone(),
        }
    }

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
            Self::Float => FLOAT_TAG,
            Self::Integer => INTEGER_TAG,
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
            FLOAT_TAG => Self::Float,
            INTEGER_TAG => Self::Integer,
            _ => unreachable!(),
        })
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ArrayInt {
    pub(crate) type_decl: TypeDecl,
    pub(crate) values: Vec<Value>,
}

impl ArrayInt {
    pub(crate) fn new(type_decl: TypeDecl, values: Vec<Value>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self { type_decl, values }))
    }

    pub fn values(&self) -> &[Value] {
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
    ArrayRef(Rc<RefCell<ArrayInt>>, usize),
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
const FLOAT_TAG: u8 = 7;
const INTEGER_TAG: u8 = 8;

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
                        cur.to_string()
                    } else {
                        acc + ", " + &cur.to_string()
                    }
                })
            ),
            Self::Ref(v) => "&".to_string() + &v.borrow().to_string(),
            Self::ArrayRef(v, idx) => {
                if let Some(v) = (*v.borrow()).values.get(*idx) {
                    v.to_string()
                } else {
                    "Array index out of range".to_string()
                }
            }
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
                    value.serialize(writer)?;
                }
                Ok(())
            }
            Self::Ref(val) => {
                writer.write_all(&REF_TAG.to_le_bytes())?;
                val.borrow().serialize(writer)?;
                Ok(())
            }
            Self::ArrayRef(val, idx) => {
                if let Some(v) = (*val.borrow()).values.get(*idx) {
                    writer.write_all(&REF_TAG.to_le_bytes())?;
                    v.serialize(writer)?;
                    Ok(())
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "ArrayRef out of range".to_string(),
                    ))
                }
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
                    .map(|_| Value::deserialize(reader))
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
            array.borrow_mut().values[idx] = value.deref();
        } else {
            panic!("assign_array must be called for an array")
        }
    }

    fn _array_get(&self, idx: u64) -> Value {
        match self {
            Value::Ref(rc) => rc.borrow()._array_get(idx),
            Value::Array(array) => array.borrow_mut().values[idx as usize].clone(),
            _ => panic!("array index must be called for an array"),
        }
    }

    pub fn array_get_ref(&self, idx: u64) -> Result<Value, EvalError> {
        Ok(match self {
            Value::Ref(rc) => rc.borrow().array_get_ref(idx)?,
            Value::Array(array) => {
                let array_int = array.borrow();
                if (idx as usize) < array_int.values.len() {
                    Value::ArrayRef(array.clone(), idx as usize)
                } else {
                    return Err(format!(
                        "array index out of range: {idx} is larger than array length {}",
                        array_int.values.len()
                    ));
                }
            }
            Value::ArrayRef(rc, idx2) => {
                let array_int = rc.borrow();
                array_int
                    .values
                    .get(*idx2)
                    .ok_or_else(|| {
                        format!(
                            "array index out of range: {idx2} is larger than array length {}",
                            array_int.values.len()
                        )
                    })?
                    .array_get_ref(idx)?
            }
            _ => return Err("array index must be called for an array".to_string()),
        })
    }

    pub fn array_push(&self, value: Value) -> Result<(), EvalError> {
        match self {
            Value::Ref(r) => r.borrow_mut().array_push(value),
            Value::Array(array) => {
                array.borrow_mut().values.push(value.deref());
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
        match self {
            Value::Ref(r) => r.borrow().clone().deref(),
            Value::ArrayRef(r, idx) => (*r.borrow()).values.get(idx).cloned().unwrap(),
            _ => self,
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
pub(crate) enum ExprEnum<'a> {
    NumLiteral(Value),
    StrLiteral(String),
    ArrLiteral(Vec<Expression<'a>>),
    Variable(&'a str),
    VarAssign(Box<Expression<'a>>, Box<Expression<'a>>),
    FnInvoke(&'a str, Vec<Expression<'a>>),
    ArrIndex(Box<Expression<'a>>, Vec<Expression<'a>>),
    Not(Box<Expression<'a>>),
    BitNot(Box<Expression<'a>>),
    Add(Box<Expression<'a>>, Box<Expression<'a>>),
    Sub(Box<Expression<'a>>, Box<Expression<'a>>),
    Mult(Box<Expression<'a>>, Box<Expression<'a>>),
    Div(Box<Expression<'a>>, Box<Expression<'a>>),
    LT(Box<Expression<'a>>, Box<Expression<'a>>),
    GT(Box<Expression<'a>>, Box<Expression<'a>>),
    BitAnd(Box<Expression<'a>>, Box<Expression<'a>>),
    BitXor(Box<Expression<'a>>, Box<Expression<'a>>),
    BitOr(Box<Expression<'a>>, Box<Expression<'a>>),
    And(Box<Expression<'a>>, Box<Expression<'a>>),
    Or(Box<Expression<'a>>, Box<Expression<'a>>),
    Conditional(
        Box<Expression<'a>>,
        Vec<Statement<'a>>,
        Option<Vec<Statement<'a>>>,
    ),
    Brace(Vec<Statement<'a>>),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Expression<'a> {
    pub(crate) expr: ExprEnum<'a>,
    pub(crate) span: Span<'a>,
}

impl<'a> Expression<'a> {
    pub(crate) fn new(expr: ExprEnum<'a>, span: Span<'a>) -> Self {
        Self { expr, span }
    }
}

/// Calculate offset between the start positions of the input spans and return a span between them.
///
/// Note: `i` shall start earlier than `r`, otherwise wrapping would occur.
fn calc_offset<'a>(i: Span<'a>, r: Span<'a>) -> Span<'a> {
    let rp = r.fragment().as_ptr() as usize;
    let ip = i.fragment().as_ptr() as usize;
    assert!(ip < rp);
    let offset = rp - ip;
    i.take(offset)
}

/// An extension trait for writing subslice concisely
pub(super) trait Subslice {
    fn subslice(&self, start: usize, length: usize) -> Self;
}

impl<'a> Subslice for Span<'a> {
    fn subslice(&self, start: usize, length: usize) -> Self {
        self.take_split(start).0.take(length)
    }
}

fn comment(input: Span) -> IResult<Span, Statement> {
    let (r, _) = multispace0(input)?;
    delimited(tag("/*"), take_until("*/"), tag("*/"))(r)
        .map(|(r, s)| (r, Statement::Comment(s.fragment())))
}

pub fn identifier(input: Span) -> IResult<Span, Span> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))(input)
}

fn ident_space(input: Span) -> IResult<Span, Span> {
    delimited(multispace0, identifier, multispace0)(input)
}

pub(crate) fn var_ref(input: Span) -> IResult<Span, Expression> {
    let (r, res) = ident_space(input)?;
    Ok((r, Expression::new(ExprEnum::Variable(res.fragment()), res)))
}

fn type_scalar(input: Span) -> IResult<Span, TypeDecl> {
    let (r, type_) = opt(delimited(
        multispace0,
        alt((tag("f64"), tag("f32"), tag("i64"), tag("i32"), tag("str"))),
        multispace0,
    ))(input)?;
    Ok((
        r,
        match type_.map(|ty| *ty) {
            Some("f64") | None => TypeDecl::F64,
            Some("f32") => TypeDecl::F32,
            Some("i32") => TypeDecl::I32,
            Some("i64") => TypeDecl::I64,
            Some("str") => TypeDecl::Str,
            Some(unknown) => panic!("Unknown type: \"{}\"", unknown),
        },
    ))
}

fn type_array(input: Span) -> IResult<Span, TypeDecl> {
    let (r, arr) = delimited(
        delimited(multispace0, tag("["), multispace0),
        alt((type_array, type_scalar)),
        delimited(multispace0, tag("]"), multispace0),
    )(input)?;
    Ok((r, TypeDecl::Array(Box::new(arr))))
}

pub(crate) fn type_decl(input: Span) -> IResult<Span, TypeDecl> {
    alt((type_array, type_scalar))(input)
}

pub(crate) fn type_spec(input: Span) -> IResult<Span, TypeDecl> {
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

fn var_decl(input: Span) -> IResult<Span, Statement> {
    let (r, _) = multispace1(tag("var")(multispace0(input)?.0)?.0)?;
    let (r, ident) = identifier(r)?;
    let (r, ts) = type_spec(r)?;
    let (r, initializer) = opt(delimited(
        delimited(multispace0, tag("="), multispace0),
        full_expression,
        multispace0,
    ))(r)?;
    let (r, _) = char(';')(multispace0(r)?.0)?;
    Ok((r, Statement::VarDecl(*ident, ts, initializer)))
}

fn double_expr(input: Span) -> IResult<Span, Expression> {
    let (r, v) = recognize_float(input)?;
    // For now we have very simple conditinon to decide if it is a floating point literal
    // by a presense of a period.
    Ok((
        r,
        Expression::new(
            ExprEnum::NumLiteral(if v.contains('.') {
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
            v,
        ),
    ))
}

fn numeric_literal_expression(input: Span) -> IResult<Span, Expression> {
    delimited(multispace0, double_expr, multispace0)(input)
}

fn str_literal(i: Span) -> IResult<Span, Expression> {
    let (r0, _) = preceded(multispace0, char('\"'))(i)?;
    let (r, val) = many0(none_of("\""))(r0)?;
    let (r, _) = terminated(char('"'), multispace0)(r)?;
    Ok((
        r,
        Expression::new(
            ExprEnum::StrLiteral(
                val.iter()
                    .collect::<String>()
                    .replace("\\\\", "\\")
                    .replace("\\n", "\n"),
            ),
            i,
        ),
    ))
}

pub(crate) fn array_literal(i: Span) -> IResult<Span, Expression> {
    let (r, _) = multispace0(i)?;
    let (r, open_br) = tag("[")(r)?;
    let (r, (mut val, last)) = pair(
        many0(terminated(full_expression, tag(","))),
        opt(full_expression),
    )(r)?;
    let (r, close_br) = tag("]")(r)?;
    if let Some(last) = last {
        val.push(last);
    }
    let span = i.subslice(
        i.offset(&open_br),
        open_br.offset(&close_br) + close_br.len(),
    );
    Ok((r, Expression::new(ExprEnum::ArrLiteral(val), span)))
}

// We parse any expr surrounded by parens, ignoring all whitespaces around those
fn parens(i: Span) -> IResult<Span, Expression> {
    let (r0, _) = multispace0(i)?;
    let (r, res) = delimited(tag("("), conditional_expr, tag(")"))(r0)?;
    let (r, _) = multispace0(r)?;
    Ok((r, Expression::new(res.expr, r0.take(r0.offset(&r)))))
}

pub(crate) fn func_invoke(i: Span) -> IResult<Span, Expression> {
    let (r, ident) = delimited(multispace0, identifier, multispace0)(i)?;
    // println!("func_invoke ident: {}", ident);
    let (r, args) = delimited(
        multispace0,
        delimited(
            tag("("),
            many0(delimited(
                multispace0,
                full_expression,
                delimited(multispace0, opt(tag(",")), multispace0),
            )),
            tag(")"),
        ),
        multispace0,
    )(r)?;
    Ok((
        r,
        Expression::new(
            ExprEnum::FnInvoke(*ident, args),
            i.subslice(i.offset(&ident), ident.offset(&r)),
        ),
    ))
}

pub(crate) fn array_index(i: Span) -> IResult<Span, Expression> {
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
    let offset = r.fragment().as_ptr() as usize - i.fragment().as_ptr() as usize;
    Ok((
        r,
        indices.into_iter().fold(prim, |acc, v| {
            Expression::new(ExprEnum::ArrIndex(Box::new(acc), v), i.take(offset))
        }),
    ))
}

pub(crate) fn primary_expression(i: Span) -> IResult<Span, Expression> {
    alt((
        numeric_literal_expression,
        str_literal,
        array_literal,
        var_ref,
        parens,
        brace_expr,
    ))(i)
}

fn postfix_expression(i: Span) -> IResult<Span, Expression> {
    alt((func_invoke, array_index, primary_expression))(i)
}

fn not(i: Span) -> IResult<Span, Expression> {
    let (r, op) = delimited(multispace0, alt((char('!'), char('~'))), multispace0)(i)?;
    let (r, v) = not_factor(r)?;
    Ok((
        r,
        match op {
            '!' => Expression::new(ExprEnum::Not(Box::new(v)), calc_offset(i, r)),
            '~' => Expression::new(ExprEnum::BitNot(Box::new(v)), calc_offset(i, r)),
            _ => unreachable!("not operator should be ! or ~"),
        },
    ))
}

fn not_factor(i: Span) -> IResult<Span, Expression> {
    alt((not, postfix_expression))(i)
}

// We read an initial factor and for each time we find
// a * or / operator followed by another factor, we do
// the math by folding everything
fn term(i: Span) -> IResult<Span, Expression> {
    let (r, init) = not_factor(i)?;

    fold_many0(
        pair(alt((char('*'), char('/'))), not_factor),
        move || init.clone(),
        move |acc, (op, val): (char, Expression)| {
            let span = i.subslice(
                i.offset(&acc.span),
                acc.span.offset(&val.span) + val.span.len(),
            );
            if op == '*' {
                Expression::new(ExprEnum::Mult(Box::new(acc), Box::new(val)), span)
            } else {
                Expression::new(ExprEnum::Div(Box::new(acc), Box::new(val)), span)
            }
        },
    )(r)
}

pub(crate) fn expr(i: Span) -> IResult<Span, Expression> {
    let (r, init) = term(i)?;

    fold_many0(
        pair(alt((char('+'), char('-'))), term),
        move || init.clone(),
        move |acc, (op, val): (char, Expression)| {
            let span = i.subslice(
                i.offset(&acc.span),
                acc.span.offset(&val.span) + val.span.len(),
            );
            if op == '+' {
                Expression::new(ExprEnum::Add(Box::new(acc), Box::new(val)), span)
            } else {
                Expression::new(ExprEnum::Sub(Box::new(acc), Box::new(val)), span)
            }
        },
    )(r)
}

fn cmp(i: Span) -> IResult<Span, Expression> {
    let (r, lhs) = expr(i)?;

    let (r, (op, val)) = pair(alt((char('<'), char('>'))), expr)(r)?;
    let span = calc_offset(i, r);
    Ok((
        r,
        if op == '<' {
            Expression::new(ExprEnum::LT(Box::new(lhs), Box::new(val)), span)
        } else {
            Expression::new(ExprEnum::GT(Box::new(lhs), Box::new(val)), span)
        },
    ))
}

pub(crate) fn conditional(i: Span) -> IResult<Span, Expression> {
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
        Expression::new(
            ExprEnum::Conditional(Box::new(cond), true_branch, false_branch),
            calc_offset(i, r),
        ),
    ))
}

pub(crate) fn var_assign(i: Span) -> IResult<Span, Expression> {
    let (r, res) = tuple((cmp_expr, char('='), assign_expr))(i)?;
    let span = calc_offset(i, r);
    Ok((
        r,
        Expression::new(ExprEnum::VarAssign(Box::new(res.0), Box::new(res.2)), span),
    ))
}

pub(crate) fn cmp_expr(i: Span) -> IResult<Span, Expression> {
    alt((cmp, expr))(i)
}

/// A functor to create a function for a binary operator
fn bin_op<'src>(
    t: &'static str,
    sub: impl Fn(Span<'src>) -> IResult<Span<'src>, Expression<'src>>,
    cons: impl Fn(Box<Expression<'src>>, Box<Expression<'src>>) -> ExprEnum<'src>,
) -> impl Fn(Span<'src>) -> IResult<Span<'src>, Expression<'src>> {
    move |i| {
        let sub = &sub;
        let cons = &cons;
        let (r, init) = sub.clone()(i)?;

        fold_many0(
            pair(delimited(multispace0, tag(t), multispace0), sub),
            move || init.clone(),
            move |acc: Expression, (_, val): (Span, Expression)| {
                let span = i.subslice(
                    i.offset(&acc.span),
                    acc.span.offset(&val.span) + val.span.len(),
                );
                Expression::new(cons(Box::new(acc), Box::new(val)), span)
            },
        )(r)
    }
}

fn bit_and(i: Span) -> IResult<Span, Expression> {
    bin_op("&", cmp_expr, |lhs, rhs| ExprEnum::BitAnd(lhs, rhs))(i)
}

fn bit_xor(i: Span) -> IResult<Span, Expression> {
    bin_op("^", bit_and, |lhs, rhs| ExprEnum::BitXor(lhs, rhs))(i)
}

fn bit_or(i: Span) -> IResult<Span, Expression> {
    bin_op("|", bit_xor, |lhs, rhs| ExprEnum::BitOr(lhs, rhs))(i)
}

fn and(i: Span) -> IResult<Span, Expression> {
    let (r, first) = bit_or(i)?;
    let (r, _) = delimited(multispace0, tag("&&"), multispace0)(r)?;
    let (r, second) = bit_or(r)?;
    Ok((
        r,
        Expression::new(
            ExprEnum::And(Box::new(first), Box::new(second)),
            calc_offset(i, r),
        ),
    ))
}

fn and_expr(i: Span) -> IResult<Span, Expression> {
    alt((and, bit_or))(i)
}

fn or(i: Span) -> IResult<Span, Expression> {
    let (r, first) = and_expr(i)?;
    let (r, _) = delimited(multispace0, tag("||"), multispace0)(r)?;
    let (r, second) = and_expr(r)?;
    Ok((
        r,
        Expression::new(
            ExprEnum::Or(Box::new(first), Box::new(second)),
            calc_offset(i, r),
        ),
    ))
}

fn or_expr(i: Span) -> IResult<Span, Expression> {
    alt((or, and_expr))(i)
}

fn assign_expr(i: Span) -> IResult<Span, Expression> {
    alt((var_assign, or_expr))(i)
}

pub(crate) fn conditional_expr(i: Span) -> IResult<Span, Expression> {
    alt((conditional, assign_expr))(i)
}

fn brace_expr(i: Span) -> IResult<Span, Expression> {
    let (r, open_br) = delimited(multispace0, tag("{"), multispace0)(i)?;
    let (r, v) = source(r)?;
    let (r, close_br) = delimited(multispace0, tag("}"), multispace0)(r)?;
    let span = i.subslice(
        i.offset(&open_br),
        open_br.offset(&close_br) + close_br.len(),
    );
    Ok((r, Expression::new(ExprEnum::Brace(v), span)))
}

pub(crate) fn full_expression(input: Span) -> IResult<Span, Expression> {
    conditional_expr(input)
}

fn expression_statement(input: Span) -> IResult<Span, Statement> {
    let (r, val) = full_expression(input)?;
    Ok((r, Statement::Expression(val)))
}

pub(crate) fn func_arg(input: Span) -> IResult<Span, ArgDecl> {
    let (r, v) = pair(
        identifier,
        opt(delimited(multispace0, type_spec, multispace0)),
    )(input)?;
    Ok((r, ArgDecl(*v.0, v.1.unwrap_or(TypeDecl::F64))))
}

pub(crate) fn func_decl(input: Span) -> IResult<Span, Statement> {
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
            name: *name,
            args,
            ret_type,
            stmts,
        },
    ))
}

fn loop_stmt(input: Span) -> IResult<Span, Statement> {
    let (r, _) = multispace0(tag("loop")(multispace0(input)?.0)?.0)?;
    let (r, stmts) = delimited(
        delimited(multispace0, tag("{"), multispace0),
        source,
        delimited(multispace0, tag("}"), multispace0),
    )(r)?;
    Ok((r, Statement::Loop(stmts)))
}

fn while_stmt(input: Span) -> IResult<Span, Statement> {
    let (r, _) = multispace0(tag("while")(multispace0(input)?.0)?.0)?;
    let (r, cond) = cmp_expr(r)?;
    let (r, stmts) = delimited(
        delimited(multispace0, tag("{"), multispace0),
        source,
        delimited(multispace0, tag("}"), multispace0),
    )(r)?;
    Ok((r, Statement::While(cond, stmts)))
}

fn for_stmt(input: Span) -> IResult<Span, Statement> {
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
    Ok((r, Statement::For(*iter, from, to, stmts)))
}

fn break_stmt(input: Span) -> IResult<Span, Statement> {
    let (r, _) = delimited(multispace0, tag("break"), multispace0)(input)?;
    Ok((r, Statement::Break))
}

fn general_statement<'a>(last: bool) -> impl Fn(Span<'a>) -> IResult<Span<'a>, Statement> {
    let terminator = move |i| -> IResult<Span, ()> {
        let mut semicolon = pair(tag(";"), multispace0);
        if last {
            Ok((opt(semicolon)(i)?.0, ()))
        } else {
            Ok((semicolon(i)?.0, ()))
        }
    };
    move |input: Span| {
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

pub(crate) fn last_statement(input: Span) -> IResult<Span, Statement> {
    general_statement(true)(input)
}

pub(crate) fn statement(input: Span) -> IResult<Span, Statement> {
    general_statement(false)(input)
}

pub fn source(input: Span) -> IResult<Span, Vec<Statement>> {
    let (r, mut v) = many0(statement)(input)?;
    let (r, last) = opt(last_statement)(r)?;
    let (r, _) = opt(multispace0)(r)?;
    if let Some(last) = last {
        v.push(last);
    }
    Ok((r, v))
}

pub fn span_source(input: &str) -> IResult<Span, Vec<Statement>> {
    source(Span::new(input))
}

#[cfg(test)]
mod test;
