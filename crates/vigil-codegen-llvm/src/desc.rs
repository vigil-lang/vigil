//! this module defines descriptions of the memory layout of types, exported
//! with the object so that userspace can read and write maps and ring
//! buffers. The code generator makes them, because it knows where fields
//! lie. Values are little-endian, except the integer fields of
//! `@network_order` structs.

use vigil_sema::maps::MapKind;

/// the memory layout of a type, as far as userspace needs it. Offsets are
/// in bytes from the start of the value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeDesc {
    /// an integer, or a kernel pointer stored as its address
    Int {
        /// size in bytes
        bytes: u32,
        signed: bool,
        /// stored big-endian, for the fields of `@network_order` structs
        big_endian: bool,
    },
    /// a `bool`: one byte, 0 for `False`
    Bool,
    /// a `fixed[I, F]`: an `i32` holding the value times 2^F
    Fixed { int_bits: u32, frac_bits: u32 },
    Struct {
        name: String,
        /// name, offset and type of each field
        fields: Vec<(String, u64, TypeDesc)>,
    },
    /// a `u32` variant index at offset 0, then the fields of the variant
    Enum {
        name: String,
        /// the name of each variant, in index order, with the offset and
        /// type of each of its fields
        variants: Vec<(String, Vec<(u64, TypeDesc)>)>,
    },
    /// a `bool` at offset 0, then the value at `offset`
    Option { inner: Box<TypeDesc>, offset: u64 },
    /// `len` elements, `stride` bytes apart
    Array {
        element: Box<TypeDesc>,
        len: u64,
        stride: u64,
    },
    /// the unit, or any type without a description: decodes as `()` and
    /// encodes as nothing
    Unit,
}

/// a map or ring buffer of the object
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapDesc {
    /// the name of the map, also its symbol in the object
    pub name: String,
    pub kind: MapKind,
    pub key: TypeDesc,
    pub value: TypeDesc,
    /// size of a key, 0 for a ring buffer
    pub key_size: u64,
    /// size of a value. For a ring buffer, the size of an event, although
    /// the map declares none.
    pub value_size: u64,
    pub max_entries: u32,
}

/// the `len` bytes at `offset`, or `None` when they are not all there
fn read(bytes: &[u8], offset: u64, len: u32) -> Option<&[u8]> {
    let start = usize::try_from(offset).ok()?;
    bytes.get(start..start + len as usize)
}

impl TypeDesc {
    /// the value at the start of `bytes`, written as in Vigil source:
    /// `Event(pid=42, comm="bash")`, `Shape.Circle(5)`, `Some(7)`.
    /// Missing bytes at the end read as `?`, so a short buffer still shows
    /// what it has.
    pub fn decode(&self, bytes: &[u8]) -> String {
        self.decode_at(bytes, 0)
    }

    /// the value at offset `at` of `bytes`
    fn decode_at(&self, bytes: &[u8], at: u64) -> String {
        match self {
            TypeDesc::Int {
                bytes: size,
                signed,
                big_endian,
            } => {
                let Some(raw) = read(bytes, at, *size) else {
                    return "?".into();
                };
                let mut buffer = [0u8; 8];
                buffer[..raw.len()].copy_from_slice(raw);
                if *big_endian {
                    buffer[..raw.len()].reverse();
                }
                let value = u64::from_le_bytes(buffer);
                if *signed {
                    let shift = 64 - 8 * size;
                    (((value << shift) as i64) >> shift).to_string()
                } else {
                    value.to_string()
                }
            }
            TypeDesc::Bool => match read(bytes, at, 1) {
                Some([0]) => "False".into(),
                Some(_) => "True".into(),
                None => "?".into(),
            },
            TypeDesc::Fixed { frac_bits, .. } => match read(bytes, at, 4) {
                Some(raw) => {
                    let raw = i32::from_le_bytes(raw.try_into().unwrap());
                    let value = f64::from(raw) / f64::from(1u32 << frac_bits);
                    // enough digits to tell the values of the type apart,
                    // without trailing zeros but always with a fractional
                    // part, so it does not read as an integer
                    let digits =
                        (f64::from(*frac_bits) * std::f64::consts::LOG10_2).ceil() as usize;
                    let text = format!("{value:.digits$}");
                    let text = text.trim_end_matches('0').trim_end_matches('.');
                    if text.contains('.') {
                        text.to_string()
                    } else {
                        format!("{text}.0")
                    }
                }
                None => "?".into(),
            },
            TypeDesc::Struct { name, fields } => {
                let fields: Vec<_> = fields
                    .iter()
                    .map(|(field, offset, ty)| {
                        format!("{field}={}", ty.decode_at(bytes, at + offset))
                    })
                    .collect();
                format!("{name}({})", fields.join(", "))
            }
            TypeDesc::Enum { name, variants } => {
                let tag = read(bytes, at, 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
                match tag.and_then(|t| variants.get(t as usize)) {
                    Some((variant, fields)) if fields.is_empty() => format!("{name}.{variant}"),
                    Some((variant, fields)) => {
                        let fields: Vec<_> = fields
                            .iter()
                            .map(|(offset, ty)| ty.decode_at(bytes, at + offset))
                            .collect();
                        format!("{name}.{variant}({})", fields.join(", "))
                    }
                    None => format!("{name}.?"),
                }
            }
            TypeDesc::Option { inner, offset } => match read(bytes, at, 1) {
                Some([0]) => "None".into(),
                Some(_) => format!("Some({})", inner.decode_at(bytes, at + offset)),
                None => "?".into(),
            },
            TypeDesc::Array {
                element,
                len,
                stride,
            } => {
                // byte arrays with text (process names, paths) read better
                // as strings. It is text when it is printable ASCII up to
                // the first zero and zeros after.
                if matches!(
                    **element,
                    TypeDesc::Int {
                        bytes: 1,
                        signed: false,
                        ..
                    }
                ) && let Some(raw) = read(bytes, at, *len as u32)
                {
                    let text: &[u8] = raw.split(|&b| b == 0).next().unwrap_or(raw);
                    let tail_is_zero = raw[text.len()..].iter().all(|&b| b == 0);
                    if !text.is_empty()
                        && tail_is_zero
                        && text.iter().all(|b| b.is_ascii_graphic() || *b == b' ')
                    {
                        return format!("{:?}", String::from_utf8_lossy(text));
                    }
                }
                let elements: Vec<_> = (0..*len)
                    .map(|i| element.decode_at(bytes, at + i * stride))
                    .collect();
                format!("[{}]", elements.join(", "))
            }
            TypeDesc::Unit => "()".into(),
        }
    }
}

/// the bytes built by [`TypeDesc::encode`], and what was lost building them
#[derive(Debug, Default)]
pub struct Encoding {
    /// the value, padded with zeros to the size asked for
    pub bytes: Vec<u8>,
    /// numbers stored in fixed point
    pub quantized: usize,
    /// values that did not fit their type, saturated (fixed point)
    pub saturated: usize,
    /// the largest quantization error, `|value - stored|`
    pub max_error: f64,
}

impl TypeDesc {
    /// the bytes of the value given by `json`, in a buffer of `size` bytes
    ///
    /// - a struct is an object with its fields, or just the field for a
    ///   struct of one field (like `Vector`)
    /// - an array is an array of the same length
    /// - a `fixed` is a number, quantized and saturated to fit (counted in
    ///   the [`Encoding`])
    /// - an option is `null` or its value
    /// - an enum is the variant name, or `{"Variant": [fields]}`
    ///
    /// An error names the path to the part that does not fit, like
    /// `value.weights.items`.
    pub fn encode(&self, json: &serde_json::Value, size: u64) -> Result<Encoding, String> {
        let mut out = Encoding {
            bytes: vec![0; size as usize],
            ..Default::default()
        };
        self.encode_at(json, 0, &mut out, "value")?;
        Ok(out)
    }

    /// writing the value given by `json` at offset `at` of `out`; `path` is
    /// where it is, for errors
    fn encode_at(
        &self,
        json: &serde_json::Value,
        at: u64,
        out: &mut Encoding,
        path: &str,
    ) -> Result<(), String> {
        use serde_json::Value;
        let write = |out: &mut Encoding, bytes: &[u8]| -> Result<(), String> {
            let start = at as usize;
            let slot = out
                .bytes
                .get_mut(start..start + bytes.len())
                .ok_or_else(|| format!("{path}: beyond the value"))?;
            slot.copy_from_slice(bytes);
            Ok(())
        };
        match self {
            TypeDesc::Int {
                bytes,
                signed,
                big_endian,
            } => {
                let value = match json {
                    Value::Number(n) => n
                        .as_i64()
                        .map(i128::from)
                        .or_else(|| n.as_u64().map(i128::from)),
                    Value::Bool(b) => Some(i128::from(*b)),
                    _ => None,
                }
                .ok_or_else(|| format!("{path}: expected an integer, found {json}"))?;
                let bits = 8 * *bytes;
                let (min, max) = if *signed {
                    (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
                } else {
                    (0, (1i128 << bits) - 1)
                };
                if value < min || value > max {
                    return Err(format!("{path}: {value} does not fit in {bits} bits"));
                }
                let mut le = (value as u64).to_le_bytes()[..*bytes as usize].to_vec();
                if *big_endian {
                    le.reverse();
                }
                write(out, &le)
            }
            TypeDesc::Bool => match json {
                Value::Bool(b) => write(out, &[u8::from(*b)]),
                _ => Err(format!("{path}: expected a boolean, found {json}")),
            },
            TypeDesc::Fixed {
                int_bits,
                frac_bits,
            } => {
                let value = json
                    .as_f64()
                    .ok_or_else(|| format!("{path}: expected a number, found {json}"))?;
                let scaled = (value * f64::from(1u32 << frac_bits)).round();
                let hi = ((1i64 << (int_bits + frac_bits - 1)) - 1) as f64;
                let lo = -hi - 1.0;
                let raw = scaled.clamp(lo, hi);
                out.quantized += 1;
                if raw != scaled {
                    out.saturated += 1;
                }
                let stored = raw / f64::from(1u32 << frac_bits);
                out.max_error = out.max_error.max((value - stored).abs());
                write(out, &(raw as i32).to_le_bytes())
            }
            TypeDesc::Struct { fields, .. } => match json {
                Value::Object(map) => {
                    for (name, offset, ty) in fields {
                        let value = map
                            .get(name)
                            .ok_or_else(|| format!("{path}: missing field `{name}`"))?;
                        ty.encode_at(value, at + offset, out, &format!("{path}.{name}"))?;
                    }
                    if let Some(extra) = map.keys().find(|k| !fields.iter().any(|f| &f.0 == *k)) {
                        return Err(format!("{path}: no field `{extra}`"));
                    }
                    Ok(())
                }
                // a struct of one field, given as the field
                _ if fields.len() == 1 => {
                    let (name, offset, ty) = &fields[0];
                    ty.encode_at(json, at + offset, out, &format!("{path}.{name}"))
                }
                _ => Err(format!("{path}: expected an object, found {json}")),
            },
            TypeDesc::Enum { variants, .. } => {
                let (name, values) = match json {
                    Value::String(name) => (name.as_str(), &[][..]),
                    Value::Object(map) if map.len() == 1 => {
                        let (name, values) = map.iter().next().unwrap();
                        match values {
                            Value::Array(values) => (name.as_str(), values.as_slice()),
                            _ => (name.as_str(), std::slice::from_ref(values)),
                        }
                    }
                    _ => return Err(format!("{path}: expected a variant, found {json}")),
                };
                let (tag, (_, fields)) = variants
                    .iter()
                    .enumerate()
                    .find(|(_, (v, _))| v == name)
                    .ok_or_else(|| format!("{path}: no variant `{name}`"))?;
                if fields.len() != values.len() {
                    return Err(format!(
                        "{path}: `{name}` has {}, given {}",
                        vigil_diagnostics::plural(fields.len(), "field"),
                        values.len()
                    ));
                }
                write(out, &(tag as u32).to_le_bytes())?;
                for (i, ((offset, ty), value)) in fields.iter().zip(values).enumerate() {
                    ty.encode_at(value, at + offset, out, &format!("{path}.{name}.{i}"))?;
                }
                Ok(())
            }
            TypeDesc::Option { inner, offset } => {
                if json.is_null() {
                    return write(out, &[0]);
                }
                write(out, &[1])?;
                inner.encode_at(json, at + offset, out, path)
            }
            TypeDesc::Array {
                element,
                len,
                stride,
            } => {
                let Value::Array(items) = json else {
                    return Err(format!("{path}: expected an array, found {json}"));
                };
                if items.len() as u64 != *len {
                    return Err(format!(
                        "{path}: expected {}, found {}",
                        vigil_diagnostics::plural(*len, "element"),
                        items.len()
                    ));
                }
                for (i, item) in items.iter().enumerate() {
                    element.encode_at(
                        item,
                        at + i as u64 * stride,
                        out,
                        &format!("{path}[{i}]"),
                    )?;
                }
                Ok(())
            }
            TypeDesc::Unit => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_structs_arrays_and_text() {
        let desc = TypeDesc::Struct {
            name: "Event".into(),
            fields: vec![
                (
                    "pid".into(),
                    0,
                    TypeDesc::Int {
                        bytes: 4,
                        signed: false,
                        big_endian: false,
                    },
                ),
                (
                    "delta".into(),
                    4,
                    TypeDesc::Int {
                        bytes: 4,
                        signed: true,
                        big_endian: false,
                    },
                ),
                (
                    "comm".into(),
                    8,
                    TypeDesc::Array {
                        element: Box::new(TypeDesc::Int {
                            bytes: 1,
                            signed: false,
                            big_endian: false,
                        }),
                        len: 8,
                        stride: 1,
                    },
                ),
            ],
        };
        let mut bytes = Vec::new();
        bytes.extend(42u32.to_le_bytes());
        bytes.extend((-3i32).to_le_bytes());
        bytes.extend(b"bash\0\0\0\0");
        assert_eq!(
            desc.decode(&bytes),
            "Event(pid=42, delta=-3, comm=\"bash\")"
        );
    }

    #[test]
    fn decodes_options_and_enums() {
        let option = TypeDesc::Option {
            inner: Box::new(TypeDesc::Int {
                bytes: 2,
                signed: false,
                big_endian: false,
            }),
            offset: 2,
        };
        assert_eq!(option.decode(&[1, 0, 7, 0]), "Some(7)");
        assert_eq!(option.decode(&[0, 0, 7, 0]), "None");
        let shape = TypeDesc::Enum {
            name: "Shape".into(),
            variants: vec![
                ("Empty".into(), vec![]),
                (
                    "Circle".into(),
                    vec![(
                        8,
                        TypeDesc::Int {
                            bytes: 4,
                            signed: false,
                            big_endian: false,
                        },
                    )],
                ),
            ],
        };
        let mut bytes = vec![1, 0, 0, 0, 0, 0, 0, 0];
        bytes.extend(5u32.to_le_bytes());
        assert_eq!(shape.decode(&bytes), "Shape.Circle(5)");
    }

    #[test]
    fn encodes_models_from_json() {
        let q8_8 = TypeDesc::Fixed {
            int_bits: 8,
            frac_bits: 8,
        };
        // `Vector[q8_8, 3]`: a struct of one field, given as the field
        let vector = TypeDesc::Struct {
            name: "Vector".into(),
            fields: vec![(
                "items".into(),
                0,
                TypeDesc::Array {
                    element: Box::new(q8_8.clone()),
                    len: 3,
                    stride: 4,
                },
            )],
        };
        let desc = TypeDesc::Struct {
            name: "Layer".into(),
            fields: vec![
                ("weights".into(), 0, vector.clone()),
                ("bias".into(), 12, q8_8.clone()),
                (
                    "label".into(),
                    16,
                    TypeDesc::Int {
                        bytes: 2,
                        signed: false,
                        big_endian: false,
                    },
                ),
            ],
        };
        let json = serde_json::json!({"weights": [0.5, -1.25, 200.0], "bias": 0.003, "label": 7});
        let encoded = desc.encode(&json, 20).unwrap();
        let raw = |i: usize| i32::from_le_bytes(encoded.bytes[i..i + 4].try_into().unwrap());
        assert_eq!((raw(0), raw(4), raw(8), raw(12)), (128, -320, 32767, 1));
        assert_eq!(&encoded.bytes[16..18], &[7, 0]);
        // 200 does not fit in q8_8, whose largest value is a little under 128
        assert_eq!(encoded.saturated, 1);
        assert!(encoded.max_error > 72.0);
        assert_eq!(
            desc.decode(&encoded.bytes),
            "Layer(weights=Vector(items=[0.5, -1.25, 127.996]), bias=0.004, label=7)"
        );
        // the lengths of arrays and widths of integers are checked
        let short = serde_json::json!({"weights": [0.5], "bias": 0, "label": 1});
        assert_eq!(
            desc.encode(&short, 20).unwrap_err(),
            "value.weights.items: expected 3 elements, found 1"
        );
        let wide = serde_json::json!({"weights": [0, 0, 0], "bias": 0, "label": 70000});
        assert_eq!(
            desc.encode(&wide, 20).unwrap_err(),
            "value.label: 70000 does not fit in 16 bits"
        );
    }
}
