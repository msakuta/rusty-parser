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
