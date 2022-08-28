use std::{cell::RefCell, rc::Rc};

use super::*;
use crate::{compile, expr, source, Statement};

fn compile_expr(s: &str) -> Bytecode {
    let bytecode = compile(&[Statement::Expression(expr(s).unwrap().1)]).unwrap();
    println!("bytecode: {:#?}", bytecode);
    bytecode
}

#[test]
fn eval_test() {
    assert_eq!(interpret(&compile_expr(" 1 +  2 ")), Ok(Value::I64(3)));
    assert_eq!(
        interpret(&compile_expr(" 12 + 6 - 4+  3")),
        Ok(Value::I64(17))
    );
    assert_eq!(interpret(&compile_expr(" 1 + 2*3 + 4")), Ok(Value::I64(11)));
    assert_eq!(interpret(&compile_expr(" 1 +  2.5 ")), Ok(Value::F64(3.5)));
}

#[test]
fn parens_eval_test() {
    assert_eq!(interpret(&compile_expr(" (  2 )")), Ok(Value::I64(2)));
    assert_eq!(
        interpret(&compile_expr(" 2* (  3 + 4 ) ")),
        Ok(Value::I64(14))
    );
    assert_eq!(
        interpret(&compile_expr("  2*2 / ( 5 - 1) + 3")),
        Ok(Value::I64(4))
    );
}

fn compile_and_run(src: &str) -> Result<Value, EvalError> {
    interpret(&compile(&source(src).unwrap().1).unwrap())
}

#[test]
fn var_test() {
    assert_eq!(
        compile_and_run("var x = 42.; x +  2; "),
        Ok(Value::F64(44.))
    );
}

#[test]
fn var_assign_test() {
    assert_eq!(compile_and_run("var x = 42.; x=12"), Ok(Value::I64(12)));
}

#[test]
fn cond_test() {
    assert_eq!(compile_and_run("if 0 { 1; }"), Ok(Value::I64(0)));
    assert_eq!(
        compile_and_run("if (1) { 2; } else { 3; }"),
        Ok(Value::I64(2))
    );
    assert_eq!(
        compile_and_run("if 1 && 2 { 2; } else { 3; }"),
        Ok(Value::I64(2))
    );
}

#[test]
fn cmp_eval_test() {
    assert_eq!(compile_and_run(" 1 <  2 "), Ok(Value::I64(1)));
    assert_eq!(compile_and_run(" 1 > 2"), Ok(Value::I64(0)));
    assert_eq!(compile_and_run(" 2 < 1"), Ok(Value::I64(0)));
    assert_eq!(compile_and_run(" 2 > 1"), Ok(Value::I64(1)));
}

#[test]
fn logic_eval_test() {
    assert_eq!(compile_and_run(" 0 && 1 "), Ok(Value::I32(0)));
    assert_eq!(compile_and_run(" 0 || 1 "), Ok(Value::I32(1)));
    assert_eq!(compile_and_run(" 1 && 0 || 1 "), Ok(Value::I32(1)));
    assert_eq!(compile_and_run(" 1 && 0 || 0 "), Ok(Value::I32(0)));
    assert_eq!(compile_and_run(" 1 && !0 "), Ok(Value::I32(1)));
}

#[test]
fn brace_expr_eval_test() {
    assert_eq!(compile_and_run(" { 1; } "), Ok(Value::I64(1)));
    assert_eq!(compile_and_run(" { 1; 2 }"), Ok(Value::I64(2)));
    assert_eq!(compile_and_run(" {1; 2;} "), Ok(Value::I64(2)));
    assert_eq!(
        compile_and_run("  { var x: i64 = 0; x = 1; x } "),
        Ok(Value::I64(1))
    );
}

#[test]
fn brace_shadowing_test() {
    assert_eq!(
        compile_and_run(" var x = 0; { var x = 1; }; x;"),
        Ok(Value::I64(0))
    );
    assert_eq!(
        compile_and_run(" var x = 0; { var x = 1; x; };"),
        Ok(Value::I64(1))
    );
    assert_eq!(
        compile_and_run(" var x = 0; { var x = 1; x = 2; }; x;"),
        Ok(Value::I64(0))
    );
}
