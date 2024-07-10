use parser::*;
use std::{cell::RefCell, collections::HashMap, io::Write, rc::Rc};

#[test]
fn test_default_arg() {
    let src = r#"
fn add(a: i32 = 123, b: i32 = 456) {
    a + b;
}

output(add());
"#;
    let (_, ast) = crate::source(src).unwrap();
    let buf = Rc::new(RefCell::new(vec![]));
    let output = s_writer(buf.clone());
    let mut funcs = HashMap::new();
    funcs.insert("output".to_string(), output);
    let bytecode = compile(&ast, funcs).unwrap();
    crate::interpret(&bytecode).unwrap();
    assert_eq!(*buf.borrow(), b"579");
}

pub(crate) fn s_writer(
    out: Rc<RefCell<Vec<u8>>>,
) -> Box<dyn Fn(&[Value]) -> Result<Value, EvalError>> {
    Box::new(move |vals| {
        let mut writer = out.borrow_mut();
        for val in vals {
            write!(writer, "{}", val).unwrap();
        }
        Ok(Value::I32(0))
    })
}
