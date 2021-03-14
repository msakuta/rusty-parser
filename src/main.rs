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
use std::env;

#[derive(Debug, PartialEq, Clone)]
enum Statement<'a> {
    Comment(&'a str),
    VarDecl(&'a str),
    Expression(Expression),
}

#[derive(Debug, PartialEq, Clone)]
enum Expression {
    NumLiteral(f64),
    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Mult(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
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
    alt((numeric_literal_expression, parens))(i)
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

fn eval(e: &Expression) -> f64 {
    match e {
        Expression::NumLiteral(val) => *val,
        Expression::Add(lhs, rhs) => eval(lhs) + eval(rhs),
        Expression::Sub(lhs, rhs) => eval(lhs) - eval(rhs),
        Expression::Mult(lhs, rhs) => eval(lhs) * eval(rhs),
        Expression::Div(lhs, rhs) => eval(lhs) / eval(rhs),
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
        for stmt in result.1 {
            match stmt {
                Statement::VarDecl(_) => println!("Variable declaration is not implemented yet!"),
                Statement::Expression(e) => println!("Expression evaluates to: {}", eval(&e)),
                _ => {}
            }
        }
    } else {
        println!("failed");
    }
}

#[test]
fn test_comments() {
    assert_eq!(
        Ok(("", Statement::Comment(" x * y "))),
        comment("/* x * y */")
    );
}

#[test]
fn test_add() {
    assert_eq!(
        Ok((
            "",
            Statement::Expression(Expression::Add(
                Box::new(Expression::NumLiteral(123.4)),
                Box::new(Expression::NumLiteral(456.0))
            ))
        )),
        expression_statement("123.4 + 456;")
    );
}

#[test]
fn test_add_paren() {
    assert_eq!(
        Ok((
            "",
            Statement::Expression(Expression::Add(
                Box::new(Expression::NumLiteral(123.4)),
                Box::new(Expression::Add(
                    Box::new(Expression::NumLiteral(456.0)),
                    Box::new(Expression::NumLiteral(789.5)),
                ))
            ))
        )),
        expression_statement("123.4 + (456 + 789.5);")
    );
}

#[test]
fn expr_test() {
    assert_eq!(
        expr(" 1 +  2 "),
        Ok((
            "",
            Expression::Add(
                Box::new(Expression::NumLiteral(1.)),
                Box::new(Expression::NumLiteral(2.))
            )
        ))
    );
    assert_eq!(
        expr(" 12 + 6 - 4+  3"),
        Ok((
            "",
            Expression::Add(
                Box::new(Expression::Sub(
                    Box::new(Expression::Add(
                        Box::new(Expression::NumLiteral(12.)),
                        Box::new(Expression::NumLiteral(6.)),
                    )),
                    Box::new(Expression::NumLiteral(4.)),
                )),
                Box::new(Expression::NumLiteral(3.))
            )
        ))
    );
    assert_eq!(
        expr(" 1 + 2*3 + 4"),
        Ok((
            "",
            Expression::Add(
                Box::new(Expression::Add(
                    Box::new(Expression::NumLiteral(1.)),
                    Box::new(Expression::Mult(
                        Box::new(Expression::NumLiteral(2.)),
                        Box::new(Expression::NumLiteral(3.)),
                    ))
                )),
                Box::new(Expression::NumLiteral(4.))
            )
        ))
    );
}

#[test]
fn parens_test() {
    assert_eq!(expr(" (  2 )"), Ok(("", Expression::NumLiteral(2.))));
    assert_eq!(
        expr(" 2* (  3 + 4 ) "),
        Ok((
            "",
            Expression::Mult(
                Box::new(Expression::NumLiteral(2.)),
                Box::new(Expression::Add(
                    Box::new(Expression::NumLiteral(3.)),
                    Box::new(Expression::NumLiteral(4.)),
                ))
            )
        ))
    );
    assert_eq!(
        expr("  2*2 / ( 5 - 1) + 3"),
        Ok((
            "",
            Expression::Add(
                Box::new(Expression::Div(
                    Box::new(Expression::Mult(
                        Box::new(Expression::NumLiteral(2.)),
                        Box::new(Expression::NumLiteral(2.)),
                    )),
                    Box::new(Expression::Sub(
                        Box::new(Expression::NumLiteral(5.)),
                        Box::new(Expression::NumLiteral(1.)),
                    )),
                )),
                Box::new(Expression::NumLiteral(3.)),
            )
        ))
    );
}

#[test]
fn eval_test() {
    assert_eq!(eval(&expr(" 1 +  2 ").unwrap().1), 3.);
    assert_eq!(eval(&expr(" 12 + 6 - 4+  3").unwrap().1), 17.);
    assert_eq!(eval(&expr(" 1 + 2*3 + 4").unwrap().1), 11.);
}

#[test]
fn parens_eval_test() {
    assert_eq!(eval(&expr(" (  2 )").unwrap().1), 2.);
    assert_eq!(eval(&expr(" 2* (  3 + 4 ) ").unwrap().1), 14.);
    assert_eq!(eval(&expr("  2*2 / ( 5 - 1) + 3").unwrap().1), 4.);
}
