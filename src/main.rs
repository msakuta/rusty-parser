use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alpha1, alphanumeric1, char, digit1, multispace0, multispace1},
    combinator::recognize,
    multi::many0,
    sequence::{delimited, pair},
    IResult,
};

#[derive(Debug, PartialEq)]
enum Statement<'a> {
    Comment(&'a str),
    VarDecl(&'a str),
    Expression(Expression<'a>),
}

#[derive(Debug, PartialEq)]
enum Expression<'a> {
    Empty,
    NumLiteral(&'a str),
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

fn expression(input: &str) -> IResult<&str, Expression> {
    let (r, val) = digit1(multispace0(input)?.0)?;
    Ok((multispace0(r)?.0, Expression::NumLiteral(val)))
}

fn empty_expression(input: &str) -> IResult<&str, Expression> {
    Ok((multispace0(input)?.0, Expression::Empty))
}

fn expression_statement(input: &str) -> IResult<&str, Statement> {
    let (r, val) = alt((expression, empty_expression))(input)?;
    Ok((char(';')(r)?.0, Statement::Expression(val)))
}

fn source(input: &str) -> IResult<&str, Vec<Statement>> {
    many0(alt((var_decl, expression_statement, comment)))(input)
}

fn main() {
    let code = r"var x;
  /* This is a block comment. */
  var y;
  ;
  123;
  ";
    if let Ok(result) = source(code) {
        println!("Match: {:?}", result.1);
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