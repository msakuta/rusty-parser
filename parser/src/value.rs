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
    pub(crate) values: Vec<Value>,
}

impl ArrayInt {
    pub(crate) fn new(type_decl: TypeDecl, values: Vec<Value>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self { type_decl, values }))
    }

    pub fn values(&self) -> &[Value] {
        &self.values
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
    Ref(Rc<RefCell<Value>>),
    ArrayRef(Rc<RefCell<ArrayInt>>, usize),
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
            Self::Array(v) => write!(
                f,
                "[{}]",
                &v.borrow().values.iter().fold("".to_string(), |acc, cur| {
                    if acc.is_empty() {
                        cur.to_string()
                    } else {
                        acc + ", " + &cur.to_string()
                    }
                })
            ),
            Self::Ref(v) => write!(f, "&{}", v.borrow()),
            Self::ArrayRef(v, idx) => {
                if let Some(v) = (*v.borrow()).values.get(*idx) {
                    v.fmt(f)
                } else {
                    write!(f, "Array index out of range")
                }
            }
        }
    }
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
                    values,
                } = &rc.borrow() as &ArrayInt;
                writer.write_all(&ARRAY_TAG.to_le_bytes())?;
                writer.write_all(&values.len().to_le_bytes())?;
                decl.serialize(writer)?;
                for value in values {
                    value.serialize(writer)?;
                }
                Ok(())
            }
            Self::Ref(val) => {
                writer.write_all(&REF_TAG.to_le_bytes())?;
                val.borrow().serialize(writer)?;
                Ok(())
            }
            Self::ArrayRef(val, idx) => {
                if let Some(v) = (*val.borrow()).values.get(*idx) {
                    writer.write_all(&REF_TAG.to_le_bytes())?;
                    v.serialize(writer)?;
                    Ok(())
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "ArrayRef out of range".to_string(),
                    ))
                }
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
                let value_count = parse!(usize);
                let decl = TypeDecl::deserialize(reader)?;
                let values = (0..value_count)
                    .map(|_| Value::deserialize(reader))
                    .collect::<Result<_, _>>()?;
                Self::Array(ArrayInt::new(decl, values))
            }
            _ => todo!(),
        })
    }

    /// We don't really need assignment operation for an array (yet), because
    /// array index will return a reference.
    fn _array_assign(&mut self, idx: usize, value: Value) -> EvalResult<()> {
        if let Value::Array(array) = self {
            array.borrow_mut().values[idx] = value.deref()?;
        } else {
            return Err(EvalError::IndexNonArray);
        }
        Ok(())
    }

    fn _array_get(&self, idx: u64) -> EvalResult<Value> {
        match self {
            Value::Ref(rc) => rc.borrow()._array_get(idx),
            Value::Array(array) => Ok(array.borrow_mut().values.eget(idx as usize)?.clone()),
            _ => Err(EvalError::IndexNonArray),
        }
    }

    pub fn array_get_ref(&self, idx: u64) -> Result<Value, EvalError> {
        Ok(match self {
            Value::Ref(rc) => rc.borrow().array_get_ref(idx)?,
            Value::Array(array) => {
                let array_int = array.borrow();
                if (idx as usize) < array_int.values.len() {
                    Value::ArrayRef(array.clone(), idx as usize)
                } else {
                    return Err(EvalError::ArrayOutOfBounds(
                        idx as usize,
                        array_int.values.len(),
                    ));
                }
            }
            Value::ArrayRef(rc, idx2) => {
                let array_int = rc.borrow();
                array_int.values.eget(*idx2)?.array_get_ref(idx)?
            }
            _ => return Err(EvalError::IndexNonArray),
        })
    }

    pub fn array_push(&self, value: Value) -> Result<(), EvalError> {
        match self {
            Value::Ref(r) => r.borrow_mut().array_push(value),
            Value::Array(array) => {
                array.borrow_mut().values.push(value.deref()?);
                Ok(())
            }
            _ => Err("push() must be called for an array".to_string().into()),
        }
    }

    /// Returns the length of an array, dereferencing recursively if the value was a reference.
    pub fn array_len(&self) -> EvalResult<usize> {
        match self {
            Value::Ref(rc) => rc.borrow().array_len(),
            Value::Array(array) => Ok(array.borrow().values.len()),
            _ => Err("len() must be called for an array".to_string().into()),
        }
    }

    /// Recursively peels off references
    pub fn deref(self) -> EvalResult<Self> {
        Ok(match self {
            Value::Ref(r) => r.borrow().clone().deref()?,
            Value::ArrayRef(r, idx) => (*r.borrow()).values.eget(idx)?.clone(),
            _ => self,
        })
    }
}
