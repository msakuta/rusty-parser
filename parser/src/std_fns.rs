use std::collections::HashMap;

use crate::{
    coerce_type,
    interpreter::{coerce_i64, EvalResult},
    type_decl::{ArraySize, ArraySizeAxis},
    value::ArrayInt,
    ArgDecl, EvalError, FuncDef, TypeDecl, Value,
};

pub(crate) fn s_print(vals: &[Value]) -> EvalResult<Value> {
    println!("print:");
    for val in vals {
        // Put a space between tokens
        print!(" {val}");
    }
    print!("\n");
    Ok(Value::I32(0))
}

fn s_puts(vals: &[Value]) -> Result<Value, EvalError> {
    fn puts_inner<'a>(vals: &mut dyn Iterator<Item = &'a Value>) {
        for val in vals {
            match val {
                Value::F64(val) => print!("{}", val),
                Value::F32(val) => print!("{}", val),
                Value::I64(val) => print!("{}", val),
                Value::I32(val) => print!("{}", val),
                Value::Str(val) => print!("{}", val),
                Value::Array(val) => puts_inner(&mut val.borrow().values.iter()),
                Value::Ref(r) => {
                    let v: &Value = &r.borrow();
                    puts_inner(&mut std::iter::once(v))
                }
                Value::ArrayRef(r, idx) => {
                    if let Some(r) = r.borrow().values.get(*idx) {
                        puts_inner(&mut std::iter::once(r))
                    }
                }
                Value::Tuple(val) => puts_inner(&mut val.borrow().iter().map(|v| &v.value)),
            }
        }
    }
    puts_inner(&mut vals.iter());
    Ok(Value::I32(0))
}

pub(crate) fn s_type(vals: &[Value]) -> Result<Value, EvalError> {
    fn type_str(val: &Value) -> String {
        match val {
            Value::F64(_) => "f64".to_string(),
            Value::F32(_) => "f32".to_string(),
            Value::I64(_) => "i64".to_string(),
            Value::I32(_) => "i32".to_string(),
            Value::Str(_) => "str".to_string(),
            Value::Array(inner) => format!("[{}]", inner.borrow().type_decl),
            Value::Ref(r) => format!("ref[{}]", type_str(&r.borrow())),
            Value::ArrayRef(r, _) => format!("aref[{}]", r.borrow().type_decl),
            Value::Tuple(inner) => format!(
                "({})",
                &inner.borrow().iter().fold(String::new(), |acc, cur| {
                    if acc.is_empty() {
                        cur.decl.to_string()
                    } else {
                        acc + ", " + &cur.decl.to_string()
                    }
                })
            ),
        }
    }
    if let [val, ..] = vals {
        Ok(Value::Str(type_str(val)))
    } else {
        Ok(Value::I32(0))
    }
}

pub(crate) fn s_len(vals: &[Value]) -> Result<Value, EvalError> {
    if let [val, ..] = vals {
        Ok(Value::I64(val.array_len()? as i64))
    } else {
        Ok(Value::I32(0))
    }
}

pub(crate) fn s_push(vals: &[Value]) -> Result<Value, EvalError> {
    if let [arr, val, ..] = vals {
        let val = val.clone().deref()?;
        arr.array_push(val).map(|_| Value::I32(0))
    } else {
        Ok(Value::I32(0))
    }
}

/// Reshape a given array with a new shape.
pub(crate) fn s_reshape(vals: &[Value]) -> Result<Value, EvalError> {
    let [arr, shape, ..] = vals else {
        return Err(EvalError::RuntimeError(
            "reshape does not have enough arguments".to_string(),
        ));
    };
    let shape = shape.clone().deref()?;
    let Value::Array(shape) = shape else {
        return Err(EvalError::RuntimeError(
            "reshape's second argument (shape) must be an array".to_string(),
        ));
    };
    let shape = shape.borrow();
    let shape = shape
        .values
        .iter()
        .map(|val| coerce_i64(val).map(|val| val as usize))
        .collect::<Result<Vec<_>, _>>()?;
    let Value::Array(arr) = arr.clone().deref()? else {
        return Err(EvalError::RuntimeError(
            "reshape's first argument (array) must be an array".to_string(),
        ));
    };
    let arr = arr
        .try_borrow()
        .map_err(|e| EvalError::Other(e.to_string()))?;
    let arr_elems: usize = arr.shape.iter().copied().product();
    let shape_elems: usize = shape.iter().copied().product();
    if arr_elems != shape_elems {
        return Err(EvalError::RuntimeError(format!(
            "reshape's array ({:?}) and new shape ({:?}) does not have the same number of elements",
            arr.shape, shape
        )));
    }
    let new_values = arr.values.clone();
    Ok(Value::Array(ArrayInt::new(
        arr.type_decl.clone(),
        shape,
        new_values,
    )))
}

pub(crate) fn s_hex_string(vals: &[Value]) -> Result<Value, EvalError> {
    if let [val, ..] = vals {
        match coerce_type(val, &TypeDecl::I64)? {
            Value::I64(i) => Ok(Value::Str(format!("{:02x}", i))),
            _ => Err(EvalError::Other(
                "hex_string() could not convert argument to i64".to_string(),
            )),
        }
    } else {
        Ok(Value::Str("".to_string()))
    }
}

pub(crate) fn std_functions<'src, 'native>() -> HashMap<String, FuncDef<'src, 'native>> {
    let mut ret = HashMap::new();
    std_functions_gen(&mut |name, code, args, ret_type| {
        ret.insert(name.to_string(), FuncDef::new_native(code, args, ret_type));
    });
    ret
}

pub(crate) fn std_functions_gen<'src, 'native>(
    set_fn: &mut impl FnMut(
        &str,
        &'native dyn Fn(&[Value]) -> Result<Value, EvalError>,
        Vec<ArgDecl<'native>>,
        Option<TypeDecl>,
    ),
) {
    // let mut functions = HashMap::new();
    set_fn("print", &s_print, vec![], None);
    set_fn(
        "puts",
        &s_puts,
        vec![ArgDecl::new("val", TypeDecl::Any)],
        None,
    );
    set_fn(
        "type",
        &s_type,
        vec![ArgDecl::new("value", TypeDecl::Any)],
        Some(TypeDecl::Str),
    );
    set_fn(
        "len",
        &s_len,
        vec![ArgDecl::new(
            "array",
            TypeDecl::Array(Box::new(TypeDecl::Any), ArraySize::default()),
        )],
        Some(TypeDecl::I64),
    );
    set_fn(
        "push",
        &s_push,
        vec![
            ArgDecl::new(
                "array",
                TypeDecl::Array(
                    Box::new(TypeDecl::Any),
                    ArraySize(vec![ArraySizeAxis::Range(0..usize::MAX)]),
                ),
            ),
            ArgDecl::new("value", TypeDecl::Any),
        ],
        None,
    );
    set_fn(
        "reshape",
        &s_reshape,
        vec![
            ArgDecl::new(
                "array",
                TypeDecl::Array(Box::new(TypeDecl::Any), ArraySize::all_dyn()),
            ),
            ArgDecl::new(
                "shape",
                TypeDecl::Array(Box::new(TypeDecl::Integer), ArraySize::all_dyn()),
            ),
        ],
        None,
    );
    set_fn(
        "hex_string",
        &s_hex_string,
        vec![ArgDecl::new("value", TypeDecl::I64)],
        Some(TypeDecl::Str),
    );
}
