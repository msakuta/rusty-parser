use super::*;
use crate::{compile, expr, Statement, source};

fn compile_expr(s: &str) -> Bytecode {
    let bytecode = compile(&[Statement::Expression(expr(s).unwrap().1)]).unwrap();
    println!("bytecode: {:#?}", bytecode);
    bytecode
}

#[test]
fn eval_test() {
    assert_eq!(
        interpret(&compile_expr(" 1 +  2 ")),
        Ok(Value::I64(3))
    );
    assert_eq!(
        interpret(&compile_expr(" 12 + 6 - 4+  3")),
        Ok(Value::I64(17))
    );
    assert_eq!(
        interpret(&compile_expr(" 1 + 2*3 + 4")),
        Ok(Value::I64(11))
    );
    assert_eq!(
        interpret(&compile_expr(" 1 +  2.5 ")),
        Ok(Value::F64(3.5))
    );
}

#[test]
fn parens_eval_test() {
    assert_eq!(
        interpret(&compile_expr(" (  2 )")),
        Ok(Value::I64(2))
    );
    assert_eq!(
        interpret(&compile_expr(" 2* (  3 + 4 ) ")),
        Ok(Value::I64(14))
    );
    assert_eq!(
        interpret(&compile_expr("  2*2 / ( 5 - 1) + 3")),
        Ok(Value::I64(4))
    );
}

#[test]
fn var_test() {
    assert_eq!(
        interpret(&compile(&source("var x = 42.; x +  2; ").unwrap().1).unwrap()),
        Ok(Value::F64(44.))
    );
}

#[test]
fn var_assign_test() {
    assert_eq!(
        interpret(&compile(&source("var x = 42.; x=12").unwrap().1).unwrap()),
        Ok(Value::I64(12))
    );
}
