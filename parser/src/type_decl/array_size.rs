use std::io::{Read, Write};

use crate::ReadError;

#[derive(Debug, PartialEq, Clone)]
pub struct ArraySize(pub Vec<ArraySizeAxis>);

impl Default for ArraySize {
    fn default() -> Self {
        Self(vec![ArraySizeAxis::Any])
    }
}

impl std::fmt::Display for ArraySize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, a) in self.0.iter().enumerate() {
            if i != 0 {
                write!(f, ", {}", a)?;
            } else {
                write!(f, "{}", a)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum ArraySizeAxis {
    /// Either dynamic or fixed array
    Any,
    /// Fixed array with a length
    Fixed(usize),
    /// Dynamic array with specified range
    Range(std::ops::Range<usize>),
}

impl ArraySizeAxis {
    fn tag(&self) -> u8 {
        match self {
            Self::Any => 0,
            Self::Fixed(_) => 1,
            Self::Range(_) => 2,
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
            (Self::Range(lhs), Self::Range(rhs)) => {
                Self::Range(lhs.start.min(rhs.start)..lhs.end.max(rhs.end))
            }
            _ => Self::Any,
        }
    }
}

impl std::fmt::Display for ArraySizeAxis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Any => write!(f, "_"),
            Self::Fixed(v) => write!(f, "{v}"),
            Self::Range(range) => match (range.start, range.end) {
                (0, usize::MAX) => write!(f, ".."),
                (start, usize::MAX) => write!(f, "{start}.."),
                (0, end) => write!(f, "..{end}"),
                (start, end) => write!(f, "{start}..{end}"),
            },
        }
    }
}

pub(super) fn write_array_size(value: &ArraySize, writer: &mut impl Write) -> std::io::Result<()> {
    writer.write_all(&(value.0.len() as u64).to_le_bytes())?;
    for v in &value.0 {
        write_array_size_axis(v, writer)?;
    }
    Ok(())
}

pub(super) fn write_array_size_axis(
    value: &ArraySizeAxis,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    writer.write_all(&mut [value.tag()])?;
    match value {
        ArraySizeAxis::Fixed(value) => writer.write_all(&(*value as u64).to_le_bytes())?,
        ArraySizeAxis::Range(range) => {
            writer.write_all(&(range.start as u64).to_le_bytes())?;
            writer.write_all(&(range.end as u64).to_le_bytes())?;
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn read_array_size(reader: &mut impl Read) -> Result<ArraySize, ReadError> {
    let mut buf = [0u8; std::mem::size_of::<u64>()];
    reader.read_exact(&mut buf)?;
    let size = usize::from_le_bytes(buf);
    let values = (0..size)
        .map(|_| read_array_size_axis(reader))
        .collect::<Result<_, _>>()?;
    Ok(ArraySize(values))
}

pub(super) fn read_array_size_axis(reader: &mut impl Read) -> Result<ArraySizeAxis, ReadError> {
    let mut tag = [0u8; 1];
    reader.read_exact(&mut tag)?;
    Ok(match tag[0] {
        0 => ArraySizeAxis::Any,
        1 => {
            let mut buf = [0u8; std::mem::size_of::<u64>()];
            reader.read_exact(&mut buf)?;
            ArraySizeAxis::Fixed(u64::from_le_bytes(buf) as usize)
        }
        2 => {
            let mut buf = [0u8; std::mem::size_of::<u64>()];
            reader.read_exact(&mut buf)?;
            let start = u64::from_le_bytes(buf) as usize;
            reader.read_exact(&mut buf)?;
            let end = u64::from_le_bytes(buf) as usize;
            ArraySizeAxis::Range(start..end)
        }
        _ => return Err(ReadError::UndefinedOpCode(tag[0])),
    })
}
