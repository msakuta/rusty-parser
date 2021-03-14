use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alpha1, alphanumeric1, char, multispace0, multispace1},
    combinator::recognize,
    multi::many0,
    sequence::{delimited, pair},
    IResult,
};

#[derive(Debug, PartialEq)]
enum Statement<'a> {
    Comment(&'a str),
    VarDecl(&'a str),
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
    let (r, _) = multispace0(input)?;
    let (r, _) = tag("var")(r)?;
    let (r, _) = multispace1(r)?;
    let (r, ident) = identifier(r)?;
    let (r, _) = multispace0(r)?;
    let (r, _) = char(';')(r)?;
    Ok((r, Statement::VarDecl(ident)))
}

fn source(input: &str) -> IResult<&str, Vec<Statement>> {
    many0(alt((var_decl, comment)))(input)
}

fn main() {
    let code = r"var x;
  /* This is a block comment. */
  var y;";
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