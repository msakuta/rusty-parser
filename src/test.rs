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
        expression_statement("123.4 + 456")
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
        expression_statement("123.4 + (456 + 789.5)")
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

fn eval0(s: &Expression) -> RunResult {
    let mut ctx = EvalContext::new();
    eval(s, &mut ctx)
}

#[test]
fn eval_test() {
    assert_eq!(eval0(&expr(" 1 +  2 ").unwrap().1), RunResult::Yield(3.));
    assert_eq!(
        eval0(&expr(" 12 + 6 - 4+  3").unwrap().1),
        RunResult::Yield(17.)
    );
    assert_eq!(
        eval0(&expr(" 1 + 2*3 + 4").unwrap().1),
        RunResult::Yield(11.)
    );
}

#[test]
fn parens_eval_test() {
    assert_eq!(eval0(&expr(" (  2 )").unwrap().1), RunResult::Yield(2.));
    assert_eq!(
        eval0(&expr(" 2* (  3 + 4 ) ").unwrap().1),
        RunResult::Yield(14.)
    );
    assert_eq!(
        eval0(&expr("  2*2 / ( 5 - 1) + 3").unwrap().1),
        RunResult::Yield(4.)
    );
}

#[test]
fn var_ident_test() {
    let mut vars = HashMap::new();
    vars.insert("x", 42.);
    assert_eq!(var_ref(" x123 "), Ok(("", Expression::Variable("x123"))));
}

#[test]
fn var_test() {
    let mut ctx = EvalContext::new();
    ctx.variables.insert("x", 42.);
    assert_eq!(
        eval(&expr(" x +  2 ").unwrap().1, &mut ctx),
        RunResult::Yield(44.)
    );
}

#[test]
fn var_assign_test() {
    let mut ctx = EvalContext::new();
    ctx.variables.insert("x", 42.);
    assert_eq!(
        var_assign("x=12"),
        Ok((
            "",
            Expression::VarAssign("x", Box::new(Expression::NumLiteral(12.)))
        ))
    );
    assert_eq!(
        eval(&var_assign("x=12").unwrap().1, &mut ctx),
        RunResult::Yield(12.)
    );
}

#[test]
fn fn_decl_test() {
    assert_eq!(
        func_decl(
            "fn f(a) {
        x = 123;
        x * a;
    }"
        ),
        Ok((
            "",
            Statement::FnDecl(
                "f",
                vec!["a"],
                vec![
                    Statement::Expression(Expression::VarAssign(
                        "x",
                        Box::new(Expression::NumLiteral(123.))
                    )),
                    Statement::Expression(Expression::Mult(
                        Box::new(Expression::Variable("x")),
                        Box::new(Expression::Variable("a"))
                    ))
                ]
            )
        ))
    );
}

#[test]
fn fn_invoke_test() {
    assert_eq!(
        source("f();"),
        Ok((
            "",
            vec![Statement::Expression(Expression::FnInvoke("f", vec![]))]
        ))
    );
    assert_eq!(
        source("f(1);"),
        Ok((
            "",
            vec![Statement::Expression(Expression::FnInvoke(
                "f",
                vec![Expression::NumLiteral(1.)]
            ))]
        ))
    );
}

#[test]
fn cond_test() {
    assert_eq!(
        conditional("if 0 { 1; }"),
        Ok((
            "",
            Expression::Conditional(
                Box::new(Expression::NumLiteral(0.)),
                vec![Statement::Expression(Expression::NumLiteral(1.))],
                None,
            )
        ))
    );
    assert_eq!(
        conditional("if (1) { 2; } else { 3; }"),
        Ok((
            "",
            Expression::Conditional(
                Box::new(Expression::NumLiteral(1.)),
                vec![Statement::Expression(Expression::NumLiteral(2.))],
                Some(vec![Statement::Expression(Expression::NumLiteral(3.))]),
            )
        ))
    );
}

#[test]
fn cond_eval_test() {
    assert_eq!(
        eval0(&conditional("if 0 { 1; }").unwrap().1),
        RunResult::Yield(0.)
    );
    assert_eq!(
        eval0(&conditional("if (1) { 2; } else { 3;}").unwrap().1),
        RunResult::Yield(2.)
    );
    assert_eq!(
        eval0(&conditional("if (0) { 2; } else { 3;}").unwrap().1),
        RunResult::Yield(3.)
    );
}

#[test]
fn cmp_test() {
    assert_eq!(
        conditional_expr(" 1 <  2 "),
        Ok((
            "",
            Expression::LT(
                Box::new(Expression::NumLiteral(1.)),
                Box::new(Expression::NumLiteral(2.))
            )
        ))
    );
    assert_eq!(
        conditional_expr(" 1 > 2"),
        Ok((
            "",
            Expression::GT(
                Box::new(Expression::NumLiteral(1.)),
                Box::new(Expression::NumLiteral(2.))
            )
        ))
    );
}

#[test]
fn cmp_eval_test() {
    assert_eq!(
        eval0(&cmp_expr(" 1 <  2 ").unwrap().1),
        RunResult::Yield(1.)
    );
    assert_eq!(eval0(&cmp_expr(" 1 > 2").unwrap().1), RunResult::Yield(0.));
    assert_eq!(eval0(&cmp_expr(" 2 < 1").unwrap().1), RunResult::Yield(0.));
    assert_eq!(eval0(&cmp_expr(" 2 > 1").unwrap().1), RunResult::Yield(1.));
}

#[test]
fn brace_expr_test() {
    use Expression::NumLiteral as NL;
    use Statement::Expression as Expr;
    assert_eq!(
        full_expression(" { 1; }"),
        Ok(("", Expression::Brace(vec![Expr(NL(1.))])))
    );
    assert_eq!(
        full_expression(" { 1; 2; }"),
        Ok(("", Expression::Brace(vec![Expr(NL(1.)), Expr(NL(2.)),])))
    );
    assert_eq!(
        full_expression(" { 1; 2 }"),
        Ok(("", Expression::Brace(vec![Expr(NL(1.)), Expr(NL(2.)),])))
    );
    assert_eq!(
        statement(" { x = 1; x }; "),
        Ok((
            "",
            Expr(Expression::Brace(vec![
                Expr(Expression::VarAssign("x", Box::new(NL(1.)))),
                Expr(Expression::Variable("x")),
            ]))
        ))
    );
}

#[test]
fn brace_expr_eval_test() {
    assert_eq!(
        eval0(&full_expression(" { 1; } ").unwrap().1),
        RunResult::Yield(1.)
    );
    assert_eq!(
        eval0(&full_expression(" { 1; 2 }").unwrap().1),
        RunResult::Yield(2.)
    );
    assert_eq!(
        eval0(&full_expression(" {1; 2;} ").unwrap().1),
        RunResult::Yield(2.)
    );
    assert_eq!(
        eval0(&full_expression("  { var x; x = 1; x } ").unwrap().1),
        RunResult::Yield(1.)
    );
}

#[test]
fn stmt_test() {
    assert_eq!(
        statement(" 1;"),
        Ok(("", Statement::Expression(Expression::NumLiteral(1.)),))
    );
    assert_eq!(
        last_statement(" 1 "),
        Ok(("", Statement::Expression(Expression::NumLiteral(1.)),))
    );
    assert_eq!(
        last_statement(" 1; "),
        Ok(("", Statement::Expression(Expression::NumLiteral(1.)),))
    );
}

#[test]
fn stmts_test() {
    assert_eq!(
        source(" 1; 2 "),
        Ok((
            "",
            vec![
                Statement::Expression(Expression::NumLiteral(1.)),
                Statement::Expression(Expression::NumLiteral(2.)),
            ]
        ))
    );
    assert_eq!(
        source(" 1; 2; "),
        Ok((
            "",
            vec![
                Statement::Expression(Expression::NumLiteral(1.)),
                Statement::Expression(Expression::NumLiteral(2.)),
            ]
        ))
    );
}

#[test]
fn loop_test() {
    assert_eq!(
        source(" var i; i = 0; loop { i = i + 1; }"),
        Ok((
            "",
            vec![
                Statement::VarDecl("i"),
                Statement::Expression(Expression::VarAssign(
                    "i",
                    Box::new(Expression::NumLiteral(0.))
                )),
                Statement::Loop(vec![Statement::Expression(Expression::VarAssign(
                    "i",
                    Box::new(Expression::Add(
                        Box::new(Expression::Variable("i")),
                        Box::new(Expression::NumLiteral(1.)),
                    ))
                )),])
            ]
        ))
    );
    assert_eq!(
        source("if i < 10 { break };"),
        Ok((
            "",
            vec![Statement::Expression(Expression::Conditional(
                Box::new(Expression::LT(
                    Box::new(Expression::Variable("i")),
                    Box::new(Expression::NumLiteral(10.)),
                )),
                vec![Statement::Break],
                None,
            ))]
        ))
    );
    assert_eq!(
        source(" var i; i = 0; loop { i = i + 1; if i < 10 { break }; }"),
        Ok((
            "",
            vec![
                Statement::VarDecl("i"),
                Statement::Expression(Expression::VarAssign(
                    "i",
                    Box::new(Expression::NumLiteral(0.))
                )),
                Statement::Loop(vec![
                    Statement::Expression(Expression::VarAssign(
                        "i",
                        Box::new(Expression::Add(
                            Box::new(Expression::Variable("i")),
                            Box::new(Expression::NumLiteral(1.)),
                        ))
                    )),
                    Statement::Expression(Expression::Conditional(
                        Box::new(Expression::LT(
                            Box::new(Expression::Variable("i")),
                            Box::new(Expression::NumLiteral(10.)),
                        )),
                        vec![Statement::Break],
                        None
                    ))
                ])
            ]
        ))
    );
}

#[test]
fn while_test() {
    assert_eq!(
        source(" var i; i = 0; while i < 10 { i = i + 1; }"),
        Ok((
            "",
            vec![
                Statement::VarDecl("i"),
                Statement::Expression(Expression::VarAssign(
                    "i",
                    Box::new(Expression::NumLiteral(0.))
                )),
                Statement::While(
                    Expression::LT(
                        Box::new(Expression::Variable("i")),
                        Box::new(Expression::NumLiteral(10.)),
                    ),
                    vec![Statement::Expression(Expression::VarAssign(
                        "i",
                        Box::new(Expression::Add(
                            Box::new(Expression::Variable("i")),
                            Box::new(Expression::NumLiteral(1.)),
                        ))
                    )),]
                )
            ]
        ))
    );
}

#[test]
fn for_test() {
    assert_eq!(
        source(" for i in 0 .. 10 { print(i); }"),
        Ok((
            "",
            vec![Statement::For(
                "i",
                Expression::NumLiteral(0.),
                Expression::NumLiteral(10.),
                vec![Statement::Expression(Expression::FnInvoke(
                    "print",
                    vec![Expression::Variable("i")],
                ))]
            )]
        ))
    );
}
