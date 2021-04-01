use super::*;

fn eval0(s: &Expression) -> RunResult {
    let mut ctx = EvalContext::new();
    eval(s, &mut ctx)
}

fn run0(s: &Vec<Statement>) -> Result<RunResult, ()> {
    let mut ctx = EvalContext::new();
    run(s, &mut ctx)
}

#[test]
fn eval_test() {
    assert_eq!(
        eval0(&expr(" 1 +  2 ").unwrap().1),
        RunResult::Yield(Value::I64(3))
    );
    assert_eq!(
        eval0(&expr(" 12 + 6 - 4+  3").unwrap().1),
        RunResult::Yield(Value::I64(17))
    );
    assert_eq!(
        eval0(&expr(" 1 + 2*3 + 4").unwrap().1),
        RunResult::Yield(Value::I64(11))
    );
    assert_eq!(
        eval0(&expr(" 1 +  2.5 ").unwrap().1),
        RunResult::Yield(Value::F64(3.5))
    );
}

#[test]
fn parens_eval_test() {
    assert_eq!(
        eval0(&expr(" (  2 )").unwrap().1),
        RunResult::Yield(Value::I64(2))
    );
    assert_eq!(
        eval0(&expr(" 2* (  3 + 4 ) ").unwrap().1),
        RunResult::Yield(Value::I64(14))
    );
    assert_eq!(
        eval0(&expr("  2*2 / ( 5 - 1) + 3").unwrap().1),
        RunResult::Yield(Value::I64(4))
    );
}

#[test]
fn var_ident_test() {
    assert_eq!(var_ref(" x123 "), Ok(("", Expression::Variable("x123"))));
}

#[test]
fn var_test() {
    let mut ctx = EvalContext::new();
    ctx.variables
        .borrow_mut()
        .insert("x", Rc::new(RefCell::new(Value::F64(42.))));
    assert_eq!(
        eval(&expr(" x +  2 ").unwrap().1, &mut ctx),
        RunResult::Yield(Value::F64(44.))
    );
}

fn var_r(name: &str) -> Box<Expression> {
    Box::new(Expression::Variable(name))
}

#[test]
fn var_assign_test() {
    let mut ctx = EvalContext::new();
    ctx.variables
        .borrow_mut()
        .insert("x", Rc::new(RefCell::new(Value::F64(42.))));
    assert_eq!(
        var_assign("x=12"),
        Ok((
            "",
            Expression::VarAssign(var_r("x"), Box::new(Expression::NumLiteral(Value::I64(12))))
        ))
    );
    assert_eq!(
        eval(&var_assign("x=12").unwrap().1, &mut ctx),
        RunResult::Yield(Value::I64(12))
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
                vec![ArgDecl("a", TypeDecl::Any)],
                vec![
                    Statement::Expression(Expression::VarAssign(
                        var_r("x"),
                        Box::new(Expression::NumLiteral(Value::I64(123)))
                    )),
                    Statement::Expression(Expression::Mult(
                        Box::new(Expression::Variable("x")),
                        Box::new(Expression::Variable("a"))
                    ))
                ]
            )
        ))
    );
    assert_eq!(func_arg("a: i32"), Ok(("", ArgDecl("a", TypeDecl::I32))),);
    assert_eq!(
        func_decl("fn f(a: i32) { a * 2 }"),
        Ok((
            "",
            Statement::FnDecl(
                "f",
                vec![ArgDecl("a", TypeDecl::I32)],
                vec![Statement::Expression(Expression::Mult(
                    Box::new(Expression::Variable("a")),
                    Box::new(Expression::NumLiteral(Value::I64(2)))
                ))]
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
                vec![Expression::NumLiteral(Value::I64(1))]
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
                Box::new(Expression::NumLiteral(Value::I64(0))),
                vec![Statement::Expression(Expression::NumLiteral(Value::I64(1)))],
                None,
            )
        ))
    );
    assert_eq!(
        conditional("if (1) { 2; } else { 3; }"),
        Ok((
            "",
            Expression::Conditional(
                Box::new(Expression::NumLiteral(Value::I64(1))),
                vec![Statement::Expression(Expression::NumLiteral(Value::I64(2)))],
                Some(vec![Statement::Expression(Expression::NumLiteral(
                    Value::I64(3)
                ))]),
            )
        ))
    );
    assert_eq!(
        conditional("if 1 && 2 { 2; } else { 3; }"),
        Ok((
            "",
            Expression::Conditional(
                Box::new(Expression::And(
                    Box::new(Expression::NumLiteral(Value::I64(1))),
                    Box::new(Expression::NumLiteral(Value::I64(2))),
                )),
                vec![Statement::Expression(Expression::NumLiteral(Value::I64(2)))],
                Some(vec![Statement::Expression(Expression::NumLiteral(
                    Value::I64(3)
                ))]),
            )
        ))
    );
}

#[test]
fn cond_eval_test() {
    assert_eq!(
        eval0(&conditional("if 0 { 1; }").unwrap().1),
        RunResult::Yield(Value::I32(0))
    );
    assert_eq!(
        eval0(&conditional("if (1) { 2; } else { 3;}").unwrap().1),
        RunResult::Yield(Value::I64(2))
    );
    assert_eq!(
        eval0(&conditional("if (0) { 2; } else { 3;}").unwrap().1),
        RunResult::Yield(Value::I64(3))
    );
}

#[test]
fn cmp_test() {
    assert_eq!(
        conditional_expr(" 1 <  2 "),
        Ok((
            "",
            Expression::LT(
                Box::new(Expression::NumLiteral(Value::I64(1))),
                Box::new(Expression::NumLiteral(Value::I64(2)))
            )
        ))
    );
    assert_eq!(
        conditional_expr(" 1 > 2"),
        Ok((
            "",
            Expression::GT(
                Box::new(Expression::NumLiteral(Value::I64(1))),
                Box::new(Expression::NumLiteral(Value::I64(2)))
            )
        ))
    );
}

#[test]
fn cmp_eval_test() {
    assert_eq!(
        eval0(&cmp_expr(" 1 <  2 ").unwrap().1),
        RunResult::Yield(Value::I64(1))
    );
    assert_eq!(
        eval0(&cmp_expr(" 1 > 2").unwrap().1),
        RunResult::Yield(Value::I64(0))
    );
    assert_eq!(
        eval0(&cmp_expr(" 2 < 1").unwrap().1),
        RunResult::Yield(Value::I64(0))
    );
    assert_eq!(
        eval0(&cmp_expr(" 2 > 1").unwrap().1),
        RunResult::Yield(Value::I64(1))
    );
}

#[test]
fn logic_test() {
    assert_eq!(
        conditional_expr(" 0 && 1 "),
        Ok((
            "",
            Expression::And(
                Box::new(Expression::NumLiteral(Value::I64(0))),
                Box::new(Expression::NumLiteral(Value::I64(1)))
            )
        ))
    );
    assert_eq!(
        conditional_expr(" 1 || 2"),
        Ok((
            "",
            Expression::Or(
                Box::new(Expression::NumLiteral(Value::I64(1))),
                Box::new(Expression::NumLiteral(Value::I64(2)))
            )
        ))
    );
    assert_eq!(
        conditional_expr(" 1 && 2 || 3 && 4"),
        Ok((
            "",
            Expression::Or(
                Box::new(Expression::And(
                    Box::new(Expression::NumLiteral(Value::I64(1))),
                    Box::new(Expression::NumLiteral(Value::I64(2)))
                )),
                Box::new(Expression::And(
                    Box::new(Expression::NumLiteral(Value::I64(3))),
                    Box::new(Expression::NumLiteral(Value::I64(4)))
                )),
            )
        ))
    );
    assert_eq!(
        conditional_expr(" 1 || !1"),
        Ok((
            "",
            Expression::Or(
                Box::new(Expression::NumLiteral(Value::I64(1))),
                Box::new(Expression::Not(Box::new(Expression::NumLiteral(
                    Value::I64(1)
                )),)),
            )
        ))
    );
    assert_eq!(
        conditional_expr(" !!1"),
        Ok((
            "",
            Expression::Not(Box::new(Expression::Not(Box::new(Expression::NumLiteral(
                Value::I64(1)
            )),)))
        ))
    );
}

#[test]
fn logic_eval_test() {
    assert_eq!(
        eval0(&full_expression(" 0 && 1 ").unwrap().1),
        RunResult::Yield(Value::I32(0))
    );
    assert_eq!(
        eval0(&full_expression(" 0 || 1 ").unwrap().1),
        RunResult::Yield(Value::I32(1))
    );
    assert_eq!(
        eval0(&full_expression(" 1 && 0 || 1 ").unwrap().1),
        RunResult::Yield(Value::I32(1))
    );
    assert_eq!(
        eval0(&full_expression(" 1 && 0 || 0 ").unwrap().1),
        RunResult::Yield(Value::I32(0))
    );
    assert_eq!(
        eval0(&full_expression(" 1 && !0 ").unwrap().1),
        RunResult::Yield(Value::I32(1))
    );
}

#[test]
fn brace_expr_test() {
    use Expression::NumLiteral as NL;
    use Statement::Expression as Expr;
    assert_eq!(
        full_expression(" { 1; }"),
        Ok(("", Expression::Brace(vec![Expr(NL(Value::I64(1)))])))
    );
    assert_eq!(
        full_expression(" { 1; 2; }"),
        Ok((
            "",
            Expression::Brace(vec![Expr(NL(Value::I64(1))), Expr(NL(Value::I64(2))),])
        ))
    );
    assert_eq!(
        full_expression(" { 1; 2 }"),
        Ok((
            "",
            Expression::Brace(vec![Expr(NL(Value::I64(1))), Expr(NL(Value::I64(2))),])
        ))
    );
    assert_eq!(
        statement(" { x = 1; x }; "),
        Ok((
            "",
            Expr(Expression::Brace(vec![
                Expr(Expression::VarAssign(
                    var_r("x"),
                    Box::new(NL(Value::I64(1)))
                )),
                Expr(Expression::Variable("x")),
            ]))
        ))
    );
}

#[test]
fn brace_expr_eval_test() {
    assert_eq!(
        eval0(&full_expression(" { 1; } ").unwrap().1),
        RunResult::Yield(Value::I64(1))
    );
    assert_eq!(
        eval0(&full_expression(" { 1; 2 }").unwrap().1),
        RunResult::Yield(Value::I64(2))
    );
    assert_eq!(
        eval0(&full_expression(" {1; 2;} ").unwrap().1),
        RunResult::Yield(Value::I64(2))
    );
    assert_eq!(
        eval0(
            &full_expression("  { var x: i64 = 0; x = 1; x } ")
                .unwrap()
                .1
        ),
        RunResult::Yield(Value::Ref(Rc::new(RefCell::new(Value::I64(1)))))
    );
}

#[test]
fn stmt_test() {
    assert_eq!(
        statement(" 1;"),
        Ok((
            "",
            Statement::Expression(Expression::NumLiteral(Value::I64(1))),
        ))
    );
    assert_eq!(
        last_statement(" 1 "),
        Ok((
            "",
            Statement::Expression(Expression::NumLiteral(Value::I64(1))),
        ))
    );
    assert_eq!(
        last_statement(" 1; "),
        Ok((
            "",
            Statement::Expression(Expression::NumLiteral(Value::I64(1))),
        ))
    );
}

#[test]
fn stmts_test() {
    assert_eq!(
        source(" 1; 2 "),
        Ok((
            "",
            vec![
                Statement::Expression(Expression::NumLiteral(Value::I64(1))),
                Statement::Expression(Expression::NumLiteral(Value::I64(2))),
            ]
        ))
    );
    assert_eq!(
        source(" 1; 2; "),
        Ok((
            "",
            vec![
                Statement::Expression(Expression::NumLiteral(Value::I64(1))),
                Statement::Expression(Expression::NumLiteral(Value::I64(2))),
            ]
        ))
    );
}

#[test]
fn array_decl_test() {
    assert_eq!(type_spec(": i32"), Ok(("", TypeDecl::I32)));
    assert_eq!(
        type_spec(": [i32]"),
        Ok(("", TypeDecl::Array(Box::new(TypeDecl::I32))))
    );
    assert_eq!(
        type_spec(": [[f32]]"),
        Ok((
            "",
            TypeDecl::Array(Box::new(TypeDecl::Array(Box::new(TypeDecl::F32))))
        ))
    );
}

#[test]
fn array_literal_test() {
    use Expression::NumLiteral as NL;
    use Value::*;
    assert_eq!(
        array_literal("[1,3,5]"),
        Ok((
            "",
            Expression::ArrLiteral(vec![NL(I64(1)), NL(I64(3)), NL(I64(5))])
        ))
    );
    assert_eq!(
        full_expression("[1,3,5]"),
        Ok((
            "",
            Expression::ArrLiteral(vec![NL(I64(1)), NL(I64(3)), NL(I64(5))])
        ))
    );
    assert_eq!(
        full_expression("[[1,3,5],[7,8,9]]"),
        Ok((
            "",
            Expression::ArrLiteral(vec![
                Expression::ArrLiteral(vec![NL(I64(1)), NL(I64(3)), NL(I64(5))]),
                Expression::ArrLiteral(vec![NL(I64(7)), NL(I64(8)), NL(I64(9))]),
            ])
        ))
    );
}

#[test]
fn array_literal_eval_test() {
    use Value::*;
    fn i64(i: i64) -> Rc<RefCell<Value>> {
        Rc::new(RefCell::new(I64(i)))
    }
    fn f64(i: f64) -> Rc<RefCell<Value>> {
        Rc::new(RefCell::new(F64(i)))
    }
    assert_eq!(
        eval0(&full_expression("[1,3,5]").unwrap().1),
        // Right now array literals have "Any" internal type, but it should be decided somehow.
        RunResult::Yield(Value::Array(TypeDecl::Any, vec![i64(1), i64(3), i64(5)]))
    );

    // Type coarsion through variable declaration
    assert_eq!(
        run0(&source("var v: [f64] = [1,3,5]; v").unwrap().1),
        Ok(RunResult::Yield(Value::Ref(Rc::new(RefCell::new(
            Value::Array(TypeDecl::F64, vec![f64(1.), f64(3.), f64(5.)])
        )))))
    );
}

#[test]
fn fn_array_decl_test() {
    assert_eq!(
        func_decl("fn f(a: [i32]) { x = 123; }"),
        Ok((
            "",
            Statement::FnDecl(
                "f",
                vec![ArgDecl("a", TypeDecl::Array(Box::new(TypeDecl::I32)))],
                vec![Statement::Expression(Expression::VarAssign(
                    var_r("x"),
                    Box::new(Expression::NumLiteral(Value::I64(123)))
                ))]
            )
        ))
    );
}

#[test]
fn array_index_test() {
    use Expression::{NumLiteral as NL, Variable as Var};
    use Value::*;
    assert_eq!(
        array_index("a[1]"),
        Ok((
            "",
            Expression::ArrIndex(Box::new(Var("a")), vec![NL(I64(1))])
        ))
    );
    assert_eq!(
        full_expression("b[1,3,5]"),
        Ok((
            "",
            Expression::ArrIndex(Box::new(Var("b")), vec![NL(I64(1)), NL(I64(3)), NL(I64(5))])
        ))
    );
}

#[test]
fn array_index_eval_test() {
    use Value::*;
    let mut ctx = EvalContext::new();

    // This is very verbose, but necessary to match against a variable in ctx.variables.
    let ast = source("var a: [i32] = [1,3,5]; a[1]").unwrap().1;
    let run_result = run(&ast, &mut ctx);
    let a_ref = ctx.get_var_rc("a").unwrap();
    let mut a_rc = None;
    // Very ugly idiom to extract a clone of a variant in a RefCell
    std::cell::Ref::map(a_ref.borrow(), |v| match v {
        Value::Array(_, a) => {
            a_rc = Some(a[1].clone());
            &()
        }
        _ => panic!("a must be an array"),
    });

    assert_eq!(run_result, Ok(RunResult::Yield(Value::Ref(a_rc.unwrap()))));

    // Technically, this test will return a reference to an element in a temporary array,
    // but we wouldn't care and just unwrap_deref.
    assert_eq!(
        run0(&source("[1,3,5][1]").unwrap().1).map(unwrap_deref),
        Ok(RunResult::Yield(I64(3)))
    );
    assert_eq!(
        run0(&source("len([1,3,5])").unwrap().1),
        Ok(RunResult::Yield(I64(3)))
    );
}

#[test]
fn array_index_assign_test() {
    use Expression::{NumLiteral as NL, Variable as Var};
    use Value::*;
    assert_eq!(
        full_expression("a[0] = b[0]"),
        Ok((
            "",
            Expression::VarAssign(
                Box::new(Expression::ArrIndex(Box::new(Var("a")), vec![NL(I64(0))])),
                Box::new(Expression::ArrIndex(Box::new(Var("b")), vec![NL(I64(0))])),
            )
        ))
    );
    assert_eq!(
        run0(&source("var a: [i32] = [1,3,5]; a[1] = 123").unwrap().1),
        Ok(RunResult::Yield(I64(123)))
    );
}

#[test]
fn var_decl_test() {
    use Expression::NumLiteral as NL;
    use Statement::VarDecl as VD;
    assert_eq!(
        source(" var x; x = 0;"),
        Ok((
            "",
            vec![
                VD("x", TypeDecl::Any, None),
                Statement::Expression(Expression::VarAssign(
                    var_r("x"),
                    Box::new(NL(Value::I64(0)))
                )),
            ]
        ))
    );
    assert_eq!(
        source(" var x = 0;"),
        Ok(("", vec![VD("x", TypeDecl::Any, Some(NL(Value::I64(0))))]))
    );
    assert_eq!(
        source(" var x: f64 = 0;"),
        Ok(("", vec![VD("x", TypeDecl::F64, Some(NL(Value::I64(0))))]))
    );
    assert_eq!(
        source(" var x: f32 = 0;"),
        Ok(("", vec![VD("x", TypeDecl::F32, Some(NL(Value::I64(0))))]))
    );
    assert_eq!(
        source(" var x: i64 = 0;"),
        Ok(("", vec![VD("x", TypeDecl::I64, Some(NL(Value::I64(0))))]))
    );
    assert_eq!(
        source(" var x: i32 = 0;"),
        Ok(("", vec![VD("x", TypeDecl::I32, Some(NL(Value::I64(0))))]))
    );
}

#[test]
fn loop_test() {
    assert_eq!(
        source(" var i; i = 0; loop { i = i + 1; }"),
        Ok((
            "",
            vec![
                Statement::VarDecl("i", TypeDecl::Any, None),
                Statement::Expression(Expression::VarAssign(
                    var_r("i"),
                    Box::new(Expression::NumLiteral(Value::I64(0)))
                )),
                Statement::Loop(vec![Statement::Expression(Expression::VarAssign(
                    var_r("i"),
                    Box::new(Expression::Add(
                        var_r("i"),
                        Box::new(Expression::NumLiteral(Value::I64(1))),
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
                    Box::new(Expression::NumLiteral(Value::I64(10))),
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
                Statement::VarDecl("i", TypeDecl::Any, None),
                Statement::Expression(Expression::VarAssign(
                    var_r("i"),
                    Box::new(Expression::NumLiteral(Value::I64(0)))
                )),
                Statement::Loop(vec![
                    Statement::Expression(Expression::VarAssign(
                        var_r("i"),
                        Box::new(Expression::Add(
                            Box::new(Expression::Variable("i")),
                            Box::new(Expression::NumLiteral(Value::I64(1))),
                        ))
                    )),
                    Statement::Expression(Expression::Conditional(
                        Box::new(Expression::LT(
                            Box::new(Expression::Variable("i")),
                            Box::new(Expression::NumLiteral(Value::I64(10))),
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
        source(" var i: i64; i = 0; while i < 10 { i = i + 1; }"),
        Ok((
            "",
            vec![
                Statement::VarDecl("i", TypeDecl::I64, None),
                Statement::Expression(Expression::VarAssign(
                    var_r("i"),
                    Box::new(Expression::NumLiteral(Value::I64(0)))
                )),
                Statement::While(
                    Expression::LT(
                        Box::new(Expression::Variable("i")),
                        Box::new(Expression::NumLiteral(Value::I64(10))),
                    ),
                    vec![Statement::Expression(Expression::VarAssign(
                        var_r("i"),
                        Box::new(Expression::Add(
                            Box::new(Expression::Variable("i")),
                            Box::new(Expression::NumLiteral(Value::I64(1))),
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
                Expression::NumLiteral(Value::I64(0)),
                Expression::NumLiteral(Value::I64(10)),
                vec![Statement::Expression(Expression::FnInvoke(
                    "print",
                    vec![Expression::Variable("i")],
                ))]
            )]
        ))
    );
}
