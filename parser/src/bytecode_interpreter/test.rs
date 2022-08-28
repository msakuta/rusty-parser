use super::*;
use crate::{compile, expr, Statement};

fn compile_expr(s: &str) -> Bytecode {
    compile(&[Statement::Expression(expr(s).unwrap().1)]).unwrap()
}

#[test]
fn eval_test() {
    assert_eq!(
        interpret(&compile_expr(" 1 +  2 ")),
        Ok(())
    );
    assert_eq!(
        interpret(&compile_expr(" 12 + 6 - 4+  3")),
        Ok(())
    );
    assert_eq!(
        interpret(&compile_expr(" 1 + 2*3 + 4")),
        Ok(())
    );
    assert_eq!(
        interpret(&compile_expr(" 1 +  2.5 ")),
        Ok(())
    );
}

#[test]
fn parens_eval_test() {
    assert_eq!(
        interpret(&compile_expr(" (  2 )")),
        Ok(())
    );
    assert_eq!(
        interpret(&compile_expr(" 2* (  3 + 4 ) ")),
        Ok(())
    );
    assert_eq!(
        interpret(&compile_expr("  2*2 / ( 5 - 1) + 3")),
        Ok(())
    );
}
