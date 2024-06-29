//! A demonstration of precedence climbing method
//!
//! It is a more stack-efficient way to parse binary operator expressions.
//! It is also eaiser to adapt to custom operators with different precedence
//! and associativity at runtime.

use std::convert::TryFrom;

fn main() {
    let input = "123";
    println!("source: {:?}, parsed: {:?}", input, expr(input));

    let input = "Hello + world";
    println!("source: {:?}, parsed: {:?}", input, expr(input));

    let input = "1 * 3";
    println!("source: {:?}, parsed: {:?}", input, expr(input));

    let input = "1 + 2 + 3";
    println!("source: {:?}, parsed: {:?}", input, expr(input));

    let input = "1 - 2 + 3";
    println!("source: {:?}, parsed: {:?}", input, expr(input));

    let input = "10 + 1 * 3";
    println!("source: {:?}, parsed: {:?}", input, expr(input));

    let input = "10 * 1 + 3";
    println!("source: {:?}, parsed: {:?}", input, expr(input));

    let input = "10 + 1 * 3 + 100";
    println!("source: {:?}, parsed: {:?}", input, expr(input));

    let input = "(123 + 456 ) + world";
    println!("source: {:?}, parsed: {:?}", input, expr(input));

    let input = "car + cdr + cdr";
    println!("source: {:?}, parsed: {:?}", input, expr(input));

    let input = "((1 + 2) + (3 + 4)) + 5 + 6";
    println!("source: {:?}, parsed: {:?}", input, expr(input));
}

fn advance_char(input: &str) -> &str {
    let mut chars = input.chars();
    chars.next();
    chars.as_str()
}

fn peek_char(input: &str) -> Option<char> {
    input.chars().next()
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum OpCode {
    Add,
    Sub,
    Mul,
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum Token<'src> {
    Ident(&'src str),
    NumLiteral(f64),
    Op(OpCode),
}

#[derive(Debug, PartialEq)]
enum Expression<'src> {
    Ident(&'src str),
    NumLiteral(f64),
    BinOp {
        op: OpCode,
        lhs: Box<Expression<'src>>,
        rhs: Box<Expression<'src>>,
    },
}

impl<'src> TryFrom<Token<'src>> for Expression<'src> {
    type Error = ();
    fn try_from(value: Token<'src>) -> Result<Self, Self::Error> {
        match value {
            Token::Ident(id) => Ok(Expression::Ident(id)),
            Token::NumLiteral(num) => Ok(Expression::NumLiteral(num)),
            _ => Err(()),
        }
    }
}

impl<'src> Expression<'src> {
    fn bin_op(op: OpCode, lhs: Self, rhs: Self) -> Self {
        Self::BinOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

fn precedence(op: &OpCode) -> usize {
    match op {
        OpCode::Add => 1,
        OpCode::Sub => 1,
        OpCode::Mul => 2,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Associativity {
    Left,
    Right,
}

fn associativity(_op: &OpCode) -> Associativity {
    Associativity::Left
}

fn expr(input: &str) -> Option<(&str, Expression)> {
    if let Some(res) = bin_op(0)(input) {
        return Some(res);
    }

    if let Some(res) = term(input) {
        return Some(res);
    }

    None
}

fn paren(input: &str) -> Option<(&str, Expression)> {
    let next_input = lparen(whitespace(input))?;

    let (next_input, expr) = expr(next_input)?;

    let next_input = rparen(whitespace(next_input))?;

    Some((next_input, expr))
}

fn bin_op(prec: usize) -> impl Fn(&str) -> Option<(&str, Expression)> {
    use std::convert::TryInto;
    move |input: &str| {
        let (r, lhs) = token(input)?;
        println!("[{prec}] First token: {lhs:?}");
        let mut ret: Expression = lhs.try_into().ok()?;
        let (mut next, mut lookahead) = token(r)?;
        println!("[{prec}] First op: {lookahead:?}");
        while let Token::Op(op) = lookahead {
            if precedence(&op) < prec {
                break;
            }
            let (r, rhs) = token(next)?;
            println!("[{prec}] Next token: {rhs:?}");
            let mut rhs: Expression = rhs.try_into().ok()?;
            let Some((p_next, p_lookahead)) = token(r) else {
                println!("[{prec}] Exhausted input, returning {ret:?} and {rhs:?}");
                return Some((r, Expression::bin_op(op, ret, rhs)));
            };
            println!("[{prec}] Inner lookahead: {p_lookahead:?}");
            (next, lookahead) = (p_next, p_lookahead);
            while let Token::Op(next_op) = lookahead {
                if precedence(&next_op) <= precedence(&op)
                    && (precedence(&next_op) != precedence(&op)
                        || associativity(&op) != Associativity::Right)
                {
                    break;
                }
                let next_prec = precedence(&op)
                    + if precedence(&op) < precedence(&next_op) {
                        1
                    } else {
                        0
                    };
                println!("[{prec}] next_prec: {:?}", next_prec);
                (next, rhs) = bin_op(next_prec)(next)?;
                let Some((p_next, p_lookahead)) = token(next) else {
                    println!("[{prec}] Inner Exhausted input, returning {ret:?} and {rhs:?}");
                    return Some((next, Expression::bin_op(op, ret, rhs)));
                };
                (next, lookahead) = (p_next, p_lookahead);
            }
            println!("[{prec}] Combining bin_op outer: {ret:?}, {rhs:?}, next: {lookahead:?}");
            ret = Expression::bin_op(op, ret, rhs);
        }

        Some((next, ret))
    }
}

fn term(input: &str) -> Option<(&str, Expression)> {
    if let Some(res) = paren(input) {
        return Some(res);
    }

    if let Some(res) = token(input) {
        let ex = match res.1 {
            Token::Ident(id) => Expression::Ident(id),
            Token::NumLiteral(num) => Expression::NumLiteral(num),
            _ => return None,
        };
        return Some((res.0, ex));
    }

    None
}

fn token(input: &str) -> Option<(&str, Token)> {
    if let Some(res) = ident(whitespace(input)) {
        return Some(res);
    }
    if let Some(res) = number(whitespace(input)) {
        return Some(res);
    }
    if let Some(res) = operator(whitespace(input)) {
        return Some(res);
    }
    None
}

fn whitespace(mut input: &str) -> &str {
    while matches!(peek_char(input), Some(' ')) {
        input = advance_char(input);
    }
    input
}

fn ident(mut input: &str) -> Option<(&str, Token)> {
    let start = input;
    if matches!(peek_char(input), Some(_x @ ('a'..='z' | 'A'..='Z'))) {
        input = advance_char(input);
        while matches!(
            peek_char(input),
            Some(_x @ ('a'..='z' | 'A'..='Z' | '0'..='9'))
        ) {
            input = advance_char(input);
        }
    }
    if start.len() == input.len() {
        None
    } else {
        Some((input, Token::Ident(&start[..(start.len() - input.len())])))
    }
}

fn number(mut input: &str) -> Option<(&str, Token)> {
    let start = input;
    if matches!(peek_char(input), Some(_x @ ('-' | '+' | '.' | '0'..='9'))) {
        input = advance_char(input);
        while matches!(peek_char(input), Some(_x @ ('.' | '0'..='9'))) {
            input = advance_char(input);
        }
    }
    if let Ok(num) = start[..(start.len() - input.len())].parse::<f64>() {
        Some((input, Token::NumLiteral(num)))
    } else {
        None
    }
}

fn operator(input: &str) -> Option<(&str, Token)> {
    match peek_char(input) {
        Some('+') => Some((advance_char(input), Token::Op(OpCode::Add))),
        Some('-') => Some((advance_char(input), Token::Op(OpCode::Sub))),
        Some('*') => Some((advance_char(input), Token::Op(OpCode::Mul))),
        _ => None,
    }
}

fn lparen(input: &str) -> Option<&str> {
    if matches!(peek_char(input), Some('(')) {
        Some(advance_char(input))
    } else {
        None
    }
}

fn rparen(input: &str) -> Option<&str> {
    if matches!(peek_char(input), Some(')')) {
        Some(advance_char(input))
    } else {
        None
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_whitespace() {
        assert_eq!(whitespace("    "), "");
    }

    #[test]
    fn test_ident() {
        assert_eq!(ident("Adam"), Some(("", Token::Ident("Adam"))));
    }

    #[test]
    fn test_number() {
        assert_eq!(number("123.45 "), Some((" ", Token::NumLiteral(123.45))));
    }

    #[test]
    fn test_expr() {
        assert_eq!(
            expr("1 + 2 * 3 "),
            Some((
                " ",
                Expression::BinOp {
                    op: OpCode::Add,
                    lhs: Box::new(Expression::NumLiteral(1.)),
                    rhs: Box::new(Expression::BinOp {
                        op: OpCode::Mul,
                        lhs: Box::new(Expression::NumLiteral(2.)),
                        rhs: Box::new(Expression::NumLiteral(3.))
                    })
                }
            ))
        );
    }

    #[test]
    fn test_associativity() {
        assert_eq!(
            expr("1 - 2 + 3 "),
            Some((
                " ",
                Expression::bin_op(
                    OpCode::Add,
                    Expression::bin_op(OpCode::Sub, Expression::NumLiteral(1.), Expression::NumLiteral(2.)),
                    Expression::NumLiteral(3.),
                )
            ))
        );
    }
}
