use std::{
    cell::RefCell,
    io::{Read, Write},
    rc::Rc,
};

use crate::{
    interpreter::{EGetExt, EvalResult},
    type_decl::TypeDecl,
    type_tags::*,
    EvalError, ReadError,
};

#[derive(Debug, PartialEq, Clone)]
pub struct ArrayInt {
    pub(crate) type_decl: TypeDecl,
    /// Shape of multi-dimensional array.
    pub(crate) shape: Vec<usize>,
    /// Flattened payload for values. First axis changes last.
    pub(crate) values: Vec<Value>,
}

impl ArrayInt {
    pub(crate) fn new(
        type_decl: TypeDecl,
        shape: Vec<usize>,
        values: Vec<Value>,
    ) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            type_decl,
            shape,
            values,
        }))
    }

    pub fn values(&self) -> &[Value] {
        &self.values
    }

    pub fn get(&self, idx: usize) -> EvalResult<Value> {
        self.values
            .get(idx)
            .ok_or_else(|| EvalError::ArrayOutOfBounds(self.values.len(), idx))
            .cloned()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    F64(f64),
    F32(f32),
    I64(i64),
    I32(i32),
    Str(String),
    Array(Rc<RefCell<ArrayInt>>),
    Tuple(Rc<RefCell<TupleInt>>),
}

impl Default for Value {
    fn default() -> Self {
        Self::I64(0)
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F64(v) => write!(f, "{v}"),
            Self::F32(v) => write!(f, "{v}"),
            Self::I64(v) => write!(f, "{v}"),
            Self::I32(v) => write!(f, "{v}"),
            Self::Str(v) => write!(f, "{v}"),
            Self::Array(v) => {
                let v = v.borrow();
                array_recurse(f, &v.values, &v.shape, 0, true)
            }
            Self::Tuple(v) => write!(
                f,
                "({})",
                &v.borrow().iter().fold("".to_string(), |acc, cur| {
                    if acc.is_empty() {
                        cur.value.to_string()
                    } else {
                        acc + ", " + &cur.value.to_string()
                    }
                })
            ),
        }
    }
}

fn array_recurse(
    f: &mut std::fmt::Formatter<'_>,
    arr: &[Value],
    shape: &[usize],
    level: usize,
    last: bool,
) -> std::fmt::Result {
    if shape.is_empty() {
        write!(f, "{}, ", arr[0])?;
        return Ok(());
    }
    let indent = " ".repeat(2 + level);
    if shape.len() == 2 {
        write!(f, "[\n")?;
    } else {
        write!(f, "{indent}[")?;
    }
    let stride: usize = shape[1..].iter().product();
    for i in 0..shape[0] {
        if 1 < shape.len() {
            array_recurse(
                f,
                &arr[i * stride..(i + 1) * stride],
                &shape[1..],
                level + 1,
                i == shape[0] - 1,
            )?;
        } else {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", arr[i])?;
        }
    }
    if shape.len() == 1 {
        write!(f, "],{indent}\n")?;
    } else {
        if last {
            write!(f, "]")?;
        } else {
            write!(f, "],")?;
        }
    }
    Ok(())
}

impl Value {
    pub(crate) fn serialize(&self, writer: &mut impl Write) -> std::io::Result<()> {
        macro_rules! serialize_with_tag {
            ($tag:ident, $val:expr) => {{
                writer.write_all(&$tag.to_le_bytes())?;
                writer.write_all(&$val.to_le_bytes())?;
                Ok(())
            }};
        }

        match self {
            Self::F64(val) => serialize_with_tag!(F64_TAG, val),
            Self::F32(val) => serialize_with_tag!(F32_TAG, val),
            Self::I64(val) => serialize_with_tag!(I64_TAG, val),
            Self::I32(val) => serialize_with_tag!(I32_TAG, val),
            Self::Str(val) => {
                writer.write_all(&STR_TAG.to_le_bytes())?;
                writer.write_all(&(val.len() as u32).to_le_bytes())?;
                writer.write_all(val.as_bytes())?;
                Ok(())
            }
            Self::Array(rc) => {
                let ArrayInt {
                    type_decl: decl,
                    shape,
                    values,
                } = &rc.borrow() as &ArrayInt;
                writer.write_all(&ARRAY_TAG.to_le_bytes())?;
                write_sizes(shape, writer)?;
                decl.serialize(writer)?;
                for value in values {
                    value.serialize(writer)?;
                }
                Ok(())
            }
            Self::Tuple(rc) => {
                let values = rc.borrow();
                writer.write_all(&TUPLE_TAG.to_le_bytes())?;
                writer.write_all(&values.len().to_le_bytes())?;
                for entry in values.iter() {
                    entry.decl.serialize(writer)?;
                    entry.value.serialize(writer)?;
                }
                Ok(())
            }
        }
    }

    pub(crate) fn deserialize(reader: &mut impl Read) -> Result<Self, ReadError> {
        let mut tag = [0u8; 1];
        reader.read_exact(&mut tag)?;

        macro_rules! parse {
            ($typ:ty) => {{
                let mut buf = [0u8; std::mem::size_of::<$typ>()];
                reader.read_exact(&mut buf)?;
                <$typ>::from_le_bytes(buf)
            }};
        }

        Ok(match tag[0] {
            F64_TAG => Value::F64(parse!(f64)),
            F32_TAG => Value::F32(parse!(f32)),
            I64_TAG => Value::I64(parse!(i64)),
            I32_TAG => Value::I32(parse!(i32)),
            STR_TAG => Value::Str({
                let len = parse!(u32);
                let mut buf = vec![0u8; len as usize];
                reader.read_exact(&mut buf)?;
                String::from_utf8(buf)?
            }),
            ARRAY_TAG => {
                let shape = read_sizes(reader)?;
                let value_count = shape.iter().fold(1usize, |acc, cur| acc * *cur);
                let decl = TypeDecl::deserialize(reader)?;
                let values = (0..value_count)
                    .map(|_| Value::deserialize(reader))
                    .collect::<Result<_, _>>()?;
                Self::Array(ArrayInt::new(decl, shape, values))
            }
            TUPLE_TAG => {
                let value_count = parse!(usize);
                let values = (0..value_count)
                    .map(|_| -> Result<_, ReadError> {
                        Ok(TupleEntry {
                            decl: TypeDecl::deserialize(reader)?,
                            value: Value::deserialize(reader)?,
                        })
                    })
                    .collect::<Result<_, _>>()?;
                Self::Tuple(Rc::new(RefCell::new(values)))
            }
            _ => todo!(),
        })
    }

    pub fn array_assign(&self, idx: usize, value: Value) -> EvalResult<()> {
        match self {
            Value::Array(array) => {
                array.borrow_mut().values[idx] = value;
            }
            _ => return Err(EvalError::IndexNonArray),
        }
        Ok(())
    }

    pub fn array_get(&self, idx: u64) -> EvalResult<Value> {
        match self {
            Value::Array(array) => Ok(array.borrow_mut().values.eget(idx as usize)?.clone()),
            _ => Err(EvalError::IndexNonArray),
        }
    }

    pub fn array_push(&self, value: Value) -> Result<(), EvalError> {
        match self {
            Value::Array(array) => {
                let mut array_int = array.borrow_mut();
                if array_int.shape.len() != 1 {
                    return Err("push() must be called for 1-D array".to_string().into());
                }
                array_int.values.push(value);
                array_int.shape[0] += 1;
                Ok(())
            }
            _ => Err("push() must be called for an array".to_string().into()),
        }
    }

    /// Returns the length of an array, dereferencing recursively if the value was a reference.
    pub fn array_len(&self) -> EvalResult<usize> {
        match self {
            Value::Array(array) => Ok(array.borrow().values.len()),
            _ => Err("len() must be called for an array".to_string().into()),
        }
    }

    pub fn tuple_get(&self, idx: u64) -> Result<Value, EvalError> {
        Ok(match self {
            Value::Tuple(tuple) => {
                let tuple_int = tuple.borrow();
                tuple_int
                    .get(idx as usize)
                    .ok_or_else(|| EvalError::TupleOutOfBounds(idx as usize, tuple_int.len()))?
                    .value
                    .clone()
            }
            _ => return Err(EvalError::IndexNonArray),
        })
    }
}

fn write_sizes(shape: &[usize], writer: &mut impl Write) -> std::io::Result<()> {
    writer.write_all(&(shape.len() as u64).to_le_bytes())?;
    for shape_axis in shape {
        writer.write_all(&(*shape_axis as u64).to_le_bytes())?;
    }
    Ok(())
}

fn read_sizes(reader: &mut impl Read) -> std::io::Result<Vec<usize>> {
    let mut buf = [0u8; std::mem::size_of::<u64>()];
    reader.read_exact(&mut buf)?;
    let size = u64::from_le_bytes(buf) as usize;
    let ret = (0..size)
        .map(|_| -> std::io::Result<_> {
            let mut buf = [0u8; std::mem::size_of::<u64>()];
            reader.read_exact(&mut buf)?;
            Ok(u64::from_le_bytes(buf) as usize)
        })
        .collect::<Result<_, _>>()?;
    Ok(ret)
}

pub type TupleInt = Vec<TupleEntry>;

#[derive(Debug, PartialEq, Clone)]
pub struct TupleEntry {
    pub(crate) decl: TypeDecl,
    pub(crate) value: Value,
}

impl TupleEntry {
    pub fn value(&self) -> &Value {
        &self.value
    }
}
