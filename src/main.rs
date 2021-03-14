use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alpha1, alphanumeric1, char, multispace0, multispace1},
    combinator::recognize,
    multi::{fold_many0, many0},
    number::complete::double,
    sequence::{delimited, pair},
    IResult,
};
use std::{collections::HashMap, env};

#[derive(Debug, PartialEq, Clone)]
enum Statement<'a> {
    Comment(&'a str),
    VarDecl(&'a str),
    Expression(Expression<'a>),
}

#[derive(Debug, PartialEq, Clone)]
enum Expression<'a> {
    NumLiteral(f64),
    Variable(&'a str),
    Add(Box<Expression<'a>>, Box<Expression<'a>>),
    Sub(Box<Expression<'a>>, Box<Expression<'a>>),
    Mult(Box<Expression<'a>>, Box<Expression<'a>>),
    Div(Box<Expression<'a>>, Box<Expression<'a>>),
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

fn var_ref(input: &str) -> IResult<&str, Expression> {
    let (r, res) = identifier(multispace0(input)?.0)?;
    Ok((multispace0(r)?.0, Expression::Variable(res)))
}

fn var_decl(input: &str) -> IResult<&str, Statement> {
    let (r, _) = multispace1(tag("var")(multispace0(input)?.0)?.0)?;
    let (r, ident) = identifier(r)?;
    let (r, _) = char(';')(multispace0(r)?.0)?;
    Ok((r, Statement::VarDecl(ident)))
}

fn numeric_literal_expression(input: &str) -> IResult<&str, Expression> {
    let (r, val) = double(multispace0(input)?.0)?;
    Ok((multispace0(r)?.0, Expression::NumLiteral(val)))
}

// We parse any expr surrounded by parens, ignoring all whitespaces around those
fn parens(i: &str) -> IResult<&str, Expression> {
    delimited(
        multispace0,
        delimited(tag("("), expr, tag(")")),
        multispace0,
    )(i)
}

// We transform an double string into a Expression::NumLiteral
// on failure, we fallback to the parens parser defined above
fn factor(i: &str) -> IResult<&str, Expression> {
    alt((numeric_literal_expression, var_ref, parens))(i)
}

// We read an initial factor and for each time we find
// a * or / operator followed by another factor, we do
// the math by folding everything
fn term(i: &str) -> IResult<&str, Expression> {
    let (i, init) = factor(i)?;

    fold_many0(
        pair(alt((char('*'), char('/'))), factor),
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

fn expression_statement(input: &str) -> IResult<&str, Statement> {
    let (r, val) = expr(input)?;
    Ok((char(';')(r)?.0, Statement::Expression(val)))
}

fn source(input: &str) -> IResult<&str, Vec<Statement>> {
    many0(alt((var_decl, expression_statement, comment)))(input)
}

fn eval(e: &Expression, ctx: &HashMap<&str, f64>) -> f64 {
    match e {
        Expression::NumLiteral(val) => *val,
        Expression::Variable(str) => *ctx.get(str).unwrap(),
        Expression::Add(lhs, rhs) => eval(lhs, ctx) + eval(rhs, ctx),
        Expression::Sub(lhs, rhs) => eval(lhs, ctx) - eval(rhs, ctx),
        Expression::Mult(lhs, rhs) => eval(lhs, ctx) * eval(rhs, ctx),
        Expression::Div(lhs, rhs) => eval(lhs, ctx) / eval(rhs, ctx),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let code = if 1 < args.len() {
        &args[1]
    } else {
        r"var x;
  /* This is a block comment. */
  var y;
  123;
  123 + 456;
  "
    };
    if let Ok(result) = source(code) {
        println!("Match: {:?}", result.1);
        let mut variables = HashMap::new();
        for stmt in result.1 {
            match stmt {
                Statement::VarDecl(var) => {
                    variables.insert(var, 0.);
                }
                Statement::Expression(e) => {
                    println!("Expression evaluates to: {}", eval(&e, &variables))
                }
                _ => {}
            }
        }
    } else {
        println!("failed");
    }
}

mod test;
