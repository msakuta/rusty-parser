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
