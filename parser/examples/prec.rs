//! A demonstration of precedence climbing method
//!
//! It is a more stack-efficient way to parse binary operator expressions.
//! It is also eaiser to adapt to custom operators with different precedence
//! and associativity at runtime.

use std::convert::TryFrom;

fn main() {
    test_case("123");
    test_case("Hello + world");
    test_case("1 * 3");
    test_case("1 + 2 + 3");
    test_case("1 - 2 + 3");
    test_case("10 + 1 * 3");
    test_case("10 * 1 + 3");
    test_case("10 + 1 * 3 + 100");
    test_case("(123 + 456 ) + world");
    test_case("car + cdr + cdr");
    test_case("((1 + 2) + (3 + 4)) + 5 + 6");
}

fn test_case(input: &str) {
    match expr(input) {
        Some((_, res)) => {
            println!("source: {:?}, parsed: {}", input, res);
        }
        _ => {
            println!("source: {input:?}, failed");
        }
    }
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

impl std::fmt::Display for OpCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Add => "+",
                Self::Sub => "-",
                Self::Mul => "*",
            }
        )
    }
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

impl<'src> std::fmt::Display for Expression<'src> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ident(id) => write!(f, "{id}"),
            Self::NumLiteral(num) => write!(f, "{num}"),
            Self::BinOp { op, lhs, rhs } => {
                write!(f, "(")?;
                lhs.fmt(f)?;
                write!(f, " {op} ")?;
                rhs.fmt(f)?;
                write!(f, ")")
            }
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
        let (mut outer_next, lhs) = token(input)?;
        println!("[{prec}] First token: {lhs:?}");
        let mut ret: Expression = lhs.try_into().ok()?;
        println!("[{prec}] First expression: {ret:?} next: {outer_next:?}");
        let Some((_peek_next, mut lookahead)) = token(outer_next) else {
            return Some((outer_next, ret));
        };
        println!("[{prec}] First op: {lookahead:?}");
        while let Token::Op(op) = lookahead {
            if precedence(&op) < prec {
                break;
            }
            let (op_next, _) = token(outer_next)?;
            let mut inner_next = op_next;
            let (rhs_next, rhs) = token(op_next)?;
            println!("[{prec}] Outer loop Next token: {rhs:?}");
            let mut rhs: Expression = rhs.try_into().ok()?;
            let Some((p_next, p_lookahead)) = token(rhs_next) else {
                println!("[{prec}] Exhausted input, returning {ret:?} and {rhs:?}");
                return Some((rhs_next, Expression::bin_op(op, ret, rhs)));
            };
            println!("[{prec}] Outer lookahead: {p_lookahead:?} outer_next: {p_next:?} next: {rhs_next:?}");
            (outer_next, lookahead) = (rhs_next, p_lookahead);

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
                println!("[{prec}] Inner next_prec: {:?} inner_next: {inner_next:}, outer_next: {outer_next:?}", next_prec);
                (inner_next, rhs) = bin_op(next_prec)(op_next)?;
                let Some((_p_next, p_lookahead)) = token(inner_next) else {
                    println!("[{prec}] Inner Exhausted input, returning {ret:?} and {rhs:?}");
                    return Some((inner_next, Expression::bin_op(op, ret, rhs)));
                };
                lookahead = p_lookahead;
            }
            println!("[{prec}] Combining bin_op outer: {ret:?}, {rhs:?}, next: {lookahead:?}");
            ret = Expression::bin_op(op, ret, rhs);
        }

        println!("[{prec}] Exiting normally with {ret:?}");

        Some((outer_next, ret))
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

    use Expression::NumLiteral as Lit;
    use OpCode::*;

    fn e_bin_op<'src>(
        op: OpCode,
        lhs: Expression<'src>,
        rhs: Expression<'src>,
    ) -> Expression<'src> {
        Expression::bin_op(op, lhs, rhs)
    }

    #[test]
    fn test_expr() {
        assert_eq!(
            expr("1 + 2 * 3"),
            Some(("", e_bin_op(Add, Lit(1.), e_bin_op(Mul, Lit(2.), Lit(3.)))))
        );
    }

    #[test]
    fn test_associativity() {
        assert_eq!(
            expr("1 - 2 + 3 "),
            Some((
                " ",
                e_bin_op(Add, e_bin_op(Sub, Lit(1.), Lit(2.)), Lit(3.),)
            ))
        );
    }
}
