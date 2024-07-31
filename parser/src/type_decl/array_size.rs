use std::io::{Read, Write};

use crate::ReadError;

#[derive(Debug, PartialEq, Clone)]
pub enum ArraySize {
    /// Either dynamic or fixed array
    Any,
    /// Only dynamic array
    Dynamic,
    /// Fixed array with a length
    Fixed(usize),
    Range(std::ops::Range<usize>),
}

impl ArraySize {
    fn tag(&self) -> u8 {
        match self {
            Self::Any => 0,
            Self::Dynamic => 1,
            Self::Fixed(_) => 2,
            Self::Range(_) => 3,
        }
    }

    pub fn zip(&self, other: &Self) -> Option<(usize, usize)> {
        match (self, other) {
            (Self::Fixed(lhs), Self::Fixed(rhs)) => Some((*lhs, *rhs)),
            _ => None,
        }
    }

    pub fn ok(&self) -> Option<usize> {
        match self {
            Self::Fixed(len) => Some(*len),
            _ => None,
        }
    }

    pub fn or(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Fixed(lhs), _) => Self::Fixed(*lhs),
            (_, Self::Fixed(rhs)) => Self::Fixed(*rhs),
            (Self::Dynamic, Self::Dynamic) => Self::Dynamic,
            _ => Self::Any,
        }
    }
}

pub(super) fn write_array_size(value: &ArraySize, writer: &mut impl Write) -> std::io::Result<()> {
    writer.write_all(&mut [value.tag()])?;
    if let ArraySize::Fixed(value) = value {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

pub(super) fn read_array_size(reader: &mut impl Read) -> Result<ArraySize, ReadError> {
    let mut tag = [0u8; 1];
    reader.read_exact(&mut tag)?;
    Ok(match tag[0] {
        0 => ArraySize::Any,
        1 => ArraySize::Dynamic,
        2 => {
            let mut buf = [0u8; std::mem::size_of::<usize>()];
            reader.read_exact(&mut buf)?;
            ArraySize::Fixed(usize::from_le_bytes(buf))
        }
        _ => return Err(ReadError::UndefinedOpCode(tag[0])),
    })
}
