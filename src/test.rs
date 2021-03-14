#![cfg(test)]
use crate::*;

#[test]
fn test_comments() {
    assert_eq!(
        Ok(("", Statement::Comment(" x * y "))),
        comment("/* x * y */")
    );
}

#[test]
fn test_ident() {
    assert_eq!(identifier("x123"), Ok(("", "x123")));
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

fn eval0(s: &Expression) -> f64 {
    eval(s, &HashMap::new())
}

#[test]
fn eval_test() {
    assert_eq!(eval0(&expr(" 1 +  2 ").unwrap().1), 3.);
    assert_eq!(eval0(&expr(" 12 + 6 - 4+  3").unwrap().1), 17.);
    assert_eq!(eval0(&expr(" 1 + 2*3 + 4").unwrap().1), 11.);
}

#[test]
fn parens_eval_test() {
    assert_eq!(eval0(&expr(" (  2 )").unwrap().1), 2.);
    assert_eq!(eval0(&expr(" 2* (  3 + 4 ) ").unwrap().1), 14.);
    assert_eq!(eval0(&expr("  2*2 / ( 5 - 1) + 3").unwrap().1), 4.);
}

#[test]
fn var_ident_test() {
    let mut vars = HashMap::new();
    vars.insert("x", 42.);
    assert_eq!(var_ref(" x123 "), Ok(("", Expression::Variable("x123"))));
}

#[test]
fn var_test() {
    let mut vars = HashMap::new();
    vars.insert("x", 42.);
    assert_eq!(eval(&expr(" x +  2 ").unwrap().1, &vars), 44.);
}
