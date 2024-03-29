use std::io::{Read, Write};

use crate::{type_tags::*, Value};

#[derive(Debug, PartialEq, Clone)]
#[repr(u8)]
pub enum TypeDecl {
    Any,
    F64,
    F32,
    I64,
    I32,
    Str,
    Array(Box<TypeDecl>),
    /// An abstract type that can match F64 or F32
    Float,
    /// An abstract type that can match I64 or I32
    Integer,
}

impl TypeDecl {
    pub(crate) fn _from_value(value: &Value) -> Self {
        match value {
            Value::F64(_) => Self::F64,
            Value::F32(_) => Self::F32,
            Value::I32(_) => Self::I32,
            Value::I64(_) => Self::I64,
            Value::Str(_) => Self::Str,
            Value::Array(a) => Self::Array(Box::new(a.borrow().type_decl.clone())),
            Value::Ref(a) => Self::_from_value(&*a.borrow()),
            Value::ArrayRef(a, _) => a.borrow().type_decl.clone(),
        }
    }

    pub(crate) fn serialize(&self, writer: &mut impl Write) -> std::io::Result<()> {
        let tag = match self {
            Self::Any => 0xff,
            Self::F64 => F64_TAG,
            Self::F32 => F32_TAG,
            Self::I64 => I64_TAG,
            Self::I32 => I32_TAG,
            Self::Str => STR_TAG,
            Self::Array(inner) => {
                writer.write_all(&ARRAY_TAG.to_le_bytes())?;
                inner.serialize(writer)?;
                return Ok(());
            }
            Self::Float => FLOAT_TAG,
            Self::Integer => INTEGER_TAG,
        };
        writer.write_all(&tag.to_le_bytes())?;
        Ok(())
    }

    pub(crate) fn deserialize(reader: &mut impl Read) -> std::io::Result<Self> {
        macro_rules! read {
            ($ty:ty) => {{
                let mut buf = [0u8; std::mem::size_of::<$ty>()];
                reader.read_exact(&mut buf)?;
                <$ty>::from_le_bytes(buf)
            }};
        }

        let tag = read!(u8);
        Ok(match tag {
            0xff => Self::Any,
            F64_TAG => Self::F64,
            F32_TAG => Self::F32,
            I64_TAG => Self::I64,
            I32_TAG => Self::I32,
            STR_TAG => Self::Str,
            ARRAY_TAG => Self::Array(Box::new(Self::deserialize(reader)?)),
            REF_TAG => todo!(),
            FLOAT_TAG => Self::Float,
            INTEGER_TAG => Self::Integer,
            _ => unreachable!(),
        })
    }
}
