#![cfg(test)]

use super::*;
use nom::Finish;

#[test]
fn test_comments() {
    let res = comment(Span::new("/* x * y */")).unwrap();
    assert_eq!(res.0.fragment(), &"");
    assert_eq!(res.1, Statement::Comment(" x * y "));
}

#[test]
fn test_comments_error() {
    if let Err(e) = comment(Span::new("/* x * y")).finish() {
        assert_eq!(e.input.fragment(), &" x * y");
        assert_eq!(e.code, nom::error::ErrorKind::TakeUntil);
    } else {
        panic!();
    };
}

#[test]
fn test_ident() {
    let res = identifier(Span::new("x123")).unwrap();
    assert_eq!(res.0.fragment(), &"");
    assert_eq!(res.1.fragment(), &"x123");
}

#[test]
fn test_add() {
    let res = expression_statement(Span::new("123.4 + 456")).unwrap();
    assert_eq!(res.0.fragment(), &"");
    assert_eq!(
        res.1,
        Statement::Expression(Expression::Add(
            Box::new(Expression::NumLiteral(Value::F64(123.4))),
            Box::new(Expression::NumLiteral(Value::I64(456)))
        ))
    );
}

#[test]
fn test_add_paren() {
    let res = expression_statement(Span::new("123.4 + (456 + 789.5)"))
        .finish()
        .unwrap();
    assert_eq!(
        res.1,
        Statement::Expression(Expression::Add(
            Box::new(Expression::NumLiteral(Value::F64(123.4))),
            Box::new(Expression::Add(
                Box::new(Expression::NumLiteral(Value::I64(456))),
                Box::new(Expression::NumLiteral(Value::F64(789.5))),
            ))
        ))
    );
}

#[test]
fn str_test() {
    assert_eq!(
        expr(Span::new("\"hello\"")).finish().unwrap().1,
        Expression::StrLiteral("hello".to_string())
    );
    assert_eq!(
        expr(Span::new("\"sl\\\\ash\"")).finish().unwrap().1,
        Expression::StrLiteral("sl\\ash".to_string())
    );
    assert_eq!(
        expr(Span::new("\"new\\nline\"")).finish().unwrap().1,
        Expression::StrLiteral("new\nline".to_string())
    );
}

#[test]
fn expr_test() {
    assert_eq!(
        expr(Span::new(" 1 +  2 ")).finish().unwrap().1,
        Expression::Add(
            Box::new(Expression::NumLiteral(Value::I64(1))),
            Box::new(Expression::NumLiteral(Value::I64(2)))
        )
    );
    assert_eq!(
        expr(Span::new(" 12 + 6 - 4+  3")).finish().unwrap().1,
        Expression::Add(
            Box::new(Expression::Sub(
                Box::new(Expression::Add(
                    Box::new(Expression::NumLiteral(Value::I64(12))),
                    Box::new(Expression::NumLiteral(Value::I64(6))),
                )),
                Box::new(Expression::NumLiteral(Value::I64(4))),
            )),
            Box::new(Expression::NumLiteral(Value::I64(3)))
        )
    );
    assert_eq!(
        expr(Span::new(" 1 + 2*3 + 4")).finish().unwrap().1,
        Expression::Add(
            Box::new(Expression::Add(
                Box::new(Expression::NumLiteral(Value::I64(1))),
                Box::new(Expression::Mult(
                    Box::new(Expression::NumLiteral(Value::I64(2))),
                    Box::new(Expression::NumLiteral(Value::I64(3))),
                ))
            )),
            Box::new(Expression::NumLiteral(Value::I64(4)))
        )
    );
}

#[test]
fn parens_test() {
    assert_eq!(
        expr(Span::new(" (  2 )")).finish().unwrap().1,
        Expression::NumLiteral(Value::I64(2))
    );
    assert_eq!(
        expr(Span::new(" 2* (  3 + 4 ) ")).finish().unwrap().1,
        Expression::Mult(
            Box::new(Expression::NumLiteral(Value::I64(2))),
            Box::new(Expression::Add(
                Box::new(Expression::NumLiteral(Value::I64(3))),
                Box::new(Expression::NumLiteral(Value::I64(4))),
            ))
        )
    );
    assert_eq!(
        expr(Span::new("  2*2 / ( 5 - 1) + 3")).finish().unwrap().1,
        Expression::Add(
            Box::new(Expression::Div(
                Box::new(Expression::Mult(
                    Box::new(Expression::NumLiteral(Value::I64(2))),
                    Box::new(Expression::NumLiteral(Value::I64(2))),
                )),
                Box::new(Expression::Sub(
                    Box::new(Expression::NumLiteral(Value::I64(5))),
                    Box::new(Expression::NumLiteral(Value::I64(1))),
                )),
            )),
            Box::new(Expression::NumLiteral(Value::I64(3))),
        )
    );
}

fn var_r(name: &str) -> Box<Expression> {
    Box::new(Expression::Variable(name))
}

#[test]
fn fn_decl_test() {
    assert_eq!(
        func_decl(Span::new(
            "fn f(a) {
        x = 123;
        x * a;
    }"
        ))
        .finish()
        .unwrap()
        .1,
        Statement::FnDecl {
            name: "f",
            args: vec![ArgDecl("a", TypeDecl::Any)],
            ret_type: None,
            stmts: vec![
                Statement::Expression(Expression::VarAssign(
                    var_r("x"),
                    Box::new(Expression::NumLiteral(Value::I64(123)))
                )),
                Statement::Expression(Expression::Mult(
                    Box::new(Expression::Variable("x")),
                    Box::new(Expression::Variable("a"))
                ))
            ]
        }
    );
    assert_eq!(
        func_arg(Span::new("a: i32")).finish().unwrap().1,
        ArgDecl("a", TypeDecl::I32)
    );
    assert_eq!(
        func_decl(Span::new("fn f(a: i32) { a * 2 }"))
            .finish()
            .unwrap()
            .1,
        Statement::FnDecl {
            name: "f",
            args: vec![ArgDecl("a", TypeDecl::I32)],
            ret_type: None,
            stmts: vec![Statement::Expression(Expression::Mult(
                Box::new(Expression::Variable("a")),
                Box::new(Expression::NumLiteral(Value::I64(2)))
            ))]
        }
    );
    assert_eq!(
        func_decl(Span::new("fn f(a: i32) -> f64 { a * 2 }"))
            .finish()
            .unwrap()
            .1,
        Statement::FnDecl {
            name: "f",
            args: vec![ArgDecl("a", TypeDecl::I32)],
            ret_type: Some(TypeDecl::F64),
            stmts: vec![Statement::Expression(Expression::Mult(
                Box::new(Expression::Variable("a")),
                Box::new(Expression::NumLiteral(Value::I64(2)))
            ))]
        }
    );
}
