use crate::parser::Subslice;

use super::*;

#[test]
fn test_cast() {
    let span = Span::new("var a: i64; a as i32");
    let (_, ast) = crate::parser::source(span).unwrap();
    assert_eq!(
        type_check(&ast, &mut TypeCheckContext::new(None)).unwrap(),
        TypeDecl::I32
    );
}

#[test]
fn test_named_arg() {
    let span = Span::new("fn f(a: i32, b: str) -> i32 { a }\n f(b: \"hey\", a: 42)");
    let (_, ast) = crate::parser::source(span).unwrap();
    assert_eq!(
        type_check(&ast, &mut TypeCheckContext::new(None)).unwrap(),
        TypeDecl::I32
    );
}

#[test]
fn test_tuple_index() {
    let span = Span::new(r#"var a: (i32, str, f64) = (42, "a", 3.14); a.0"#);
    let (_, ast) = crate::parser::source(span).unwrap();
    assert_eq!(
        type_check(&ast, &mut TypeCheckContext::new(None)).unwrap(),
        TypeDecl::I32
    );
}

#[test]
fn test_tuple_index_err() {
    let span = Span::new(r#"var a: (i32, str, f64) = (42, "a", 3.14); a.3"#);
    let (_, ast) = crate::parser::source(span).unwrap();
    let res = type_check(&ast, &mut TypeCheckContext::new(None));
    assert_eq!(
        res,
        Err(TypeCheckError::new(
            "Tuple index out of range".to_string(),
            span.subslice(42, 1),
            None
        ))
    );
}
