use nom::Finish;
use parser::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub(crate) fn log(s: &str);
}

#[wasm_bindgen(module = "/wasm_api.js")]
extern "C" {
    pub(crate) fn wasm_print(s: &str);
    pub(crate) fn wasm_rectangle(x0: i32, y0: i32, x1: i32, y1: i32);
    pub(crate) fn wasm_set_fill_style(s: &str);
}

fn s_print(vals: &[Value]) -> Value {
    wasm_print("print:");
    fn print_inner(vals: &[Value]) {
        for val in vals {
            match val {
                Value::F64(val) => wasm_print(&format!(" {}", val)),
                Value::F32(val) => wasm_print(&format!(" {}", val)),
                Value::I64(val) => wasm_print(&format!(" {}", val)),
                Value::I32(val) => wasm_print(&format!(" {}", val)),
                Value::Str(val) => wasm_print(&format!(" {}", val)),
                Value::Array(_, val) => {
                    wasm_print("[");
                    print_inner(&val.iter().map(|v| v.borrow().clone()).collect::<Vec<_>>());
                    wasm_print("]");
                }
                Value::Ref(r) => {
                    wasm_print("ref(");
                    print_inner(&[r.borrow().clone()]);
                    wasm_print(")");
                }
            }
        }
    }
    print_inner(vals);
    wasm_print(&format!("\n"));
    Value::I32(0)
}

fn s_puts(vals: &[Value]) -> Value {
    fn puts_inner(vals: &[Value]) {
        for val in vals {
            match val {
                Value::F64(val) => wasm_print(&format!("{}", val)),
                Value::F32(val) => wasm_print(&format!("{}", val)),
                Value::I64(val) => wasm_print(&format!("{}", val)),
                Value::I32(val) => wasm_print(&format!("{}", val)),
                Value::Str(val) => wasm_print(&format!("{}", val)),
                Value::Array(_, val) => {
                    puts_inner(&val.iter().map(|v| v.borrow().clone()).collect::<Vec<_>>())
                }
                Value::Ref(r) => puts_inner(&[r.borrow().clone()]),
            }
        }
    }
    puts_inner(vals);
    Value::I32(0)
}

fn s_rectangle(vals: &[Value]) -> Value {
    let mut i32vals = vals.iter().take(4).map(|val| {
        if let Value::I32(v) = coerce_type(val, &TypeDecl::I32) {
            v
        } else {
            panic!("wrong type!");
        }
    });
    let x0 = i32vals.next().unwrap();
    let y0 = i32vals.next().unwrap();
    let x1 = i32vals.next().unwrap();
    let y1 = i32vals.next().unwrap();
    wasm_rectangle(x0, y0, x1, y1);
    Value::I32(0)
}

fn s_set_fill_style(vals: &[Value]) -> Value {
    if let [Value::Str(s), ..] = vals {
        wasm_set_fill_style(s);
    }
    Value::I32(0)
}

#[wasm_bindgen]
pub fn entry(src: &str) -> Result<(), JsValue> {
    let mut ctx = EvalContext::new();
    ctx.set_fn("print", FuncDef::Native(&s_print));
    ctx.set_fn("puts", FuncDef::Native(&s_puts));
    ctx.set_fn("set_fill_style", FuncDef::Native(&s_set_fill_style));
    ctx.set_fn("rectangle", FuncDef::Native(&s_rectangle));
    let parse_result = source(src)
        .finish()
        .map_err(|e| JsValue::from_str(&format!("Parse error: {:?}", e)))?;
    if 0 < parse_result.0.len() {
        return Err(JsValue::from_str(&format!(
            "Unexpected end of input at: {:?}",
            parse_result.0
        )));
    }
    run(&parse_result.1, &mut ctx)
        .map_err(|e| JsValue::from_str(&format!("Error on execution: {:?}", e)))?;
    Ok(())
}

#[wasm_bindgen]
pub fn parse_ast(src: &str) -> Result<String, JsValue> {
    let parse_result = source(src)
        .finish()
        .map_err(|e| JsValue::from_str(&format!("Parse error: {:?}", e)))?;
    Ok(format!("{:#?}", parse_result.1))
}
