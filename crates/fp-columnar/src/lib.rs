#![forbid(unsafe_code)]

use fp_types::{DType, NullKind, Scalar, TypeError, cast_scalar_owned, common_dtype, infer_dtype};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Eq)]
pub struct ValidityMask {
    words: Vec<u64>,
    len: usize,
}

impl ValidityMask {
    #[must_use]
    pub fn from_values(values: &[Scalar]) -> Self {
        let len = values.len();
        let word_count = len.div_ceil(64);
        let mut words = vec![0_u64; word_count];
        for (idx, value) in values.iter().enumerate() {
            if !value.is_missing() {
                words[idx / 64] |= 1_u64 << (idx % 64);
            }
        }
        Self { words, len }
    }

    #[must_use]
    pub fn all_valid(len: usize) -> Self {
        let word_count = len.div_ceil(64);
        let mut words = vec![u64::MAX; word_count];
        let remainder = len % 64;
        if remainder > 0 && !words.is_empty() {
            let last = words.len() - 1;
            words[last] = (1_u64 << remainder) - 1;
        }
        Self { words, len }
    }

    #[must_use]
    pub fn all_invalid(len: usize) -> Self {
        let word_count = len.div_ceil(64);
        Self {
            words: vec![0_u64; word_count],
            len,
        }
    }

    #[must_use]
    pub fn get(&self, idx: usize) -> bool {
        if idx >= self.len {
            return false;
        }
        (self.words[idx / 64] >> (idx % 64)) & 1 == 1
    }

    pub fn set(&mut self, idx: usize, value: bool) {
        if idx >= self.len {
            return;
        }
        if value {
            self.words[idx / 64] |= 1_u64 << (idx % 64);
        } else {
            self.words[idx / 64] &= !(1_u64 << (idx % 64));
        }
    }

    #[must_use]
    pub fn count_valid(&self) -> usize {
        let full_words = self.len / 64;
        let mut count: u32 = self.words[..full_words]
            .iter()
            .map(|w| w.count_ones())
            .sum();
        let remainder = self.len % 64;
        if remainder > 0 && full_words < self.words.len() {
            let mask = (1_u64 << remainder) - 1;
            count += (self.words[full_words] & mask).count_ones();
        }
        count as usize
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn and_mask(&self, other: &Self) -> Self {
        let len = self.len.min(other.len);
        let word_count = len.div_ceil(64);
        let words = self.words[..word_count]
            .iter()
            .zip(&other.words[..word_count])
            .map(|(a, b)| a & b)
            .collect();
        Self { words, len }
    }

    #[must_use]
    pub fn or_mask(&self, other: &Self) -> Self {
        let len = self.len.min(other.len);
        let word_count = len.div_ceil(64);
        let words = self.words[..word_count]
            .iter()
            .zip(&other.words[..word_count])
            .map(|(a, b)| a | b)
            .collect();
        Self { words, len }
    }

    #[must_use]
    pub fn not_mask(&self) -> Self {
        let mut words: Vec<u64> = self.words.iter().map(|w| !w).collect();
        let remainder = self.len % 64;
        if remainder > 0 && !words.is_empty() {
            let last = words.len() - 1;
            words[last] &= (1_u64 << remainder) - 1;
        }
        Self {
            words,
            len: self.len,
        }
    }

    /// Returns an iterator yielding bool values, compatible with the previous
    /// `&[bool]` API. Materializes from the packed representation.
    pub fn bits(&self) -> impl Iterator<Item = bool> + '_ {
        (0..self.len).map(|idx| self.get(idx))
    }
}

impl PartialEq for ValidityMask {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.bits().eq(other.bits())
    }
}

impl Serialize for ValidityMask {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let bits: Vec<bool> = self.bits().collect();
        let mut state = serializer.serialize_struct("ValidityMask", 1)?;
        state.serialize_field("bits", &bits)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for ValidityMask {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            bits: Vec<bool>,
        }
        let raw = Raw::deserialize(deserializer)?;
        let len = raw.bits.len();
        let word_count = len.div_ceil(64);
        let mut words = vec![0_u64; word_count];
        for (idx, &valid) in raw.bits.iter().enumerate() {
            if valid {
                words[idx / 64] |= 1_u64 << (idx % 64);
            }
        }
        Ok(Self { words, len })
    }
}

/// AG-10: Typed array representation for vectorized batch execution.
///
/// Stores column data as contiguous typed arrays rather than `Vec<Scalar>`.
/// Validity is tracked by `ValidityMask`; invalid positions hold unspecified
/// values in the typed array (callers must check validity before reading).
///
/// This eliminates per-element enum dispatch for arithmetic operations,
/// enabling SIMD auto-vectorization on `&[f64]` / `&[i64]` slices.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnData {
    Float64(Vec<f64>),
    Int64(Vec<i64>),
    Bool(Vec<bool>),
    Utf8(Vec<String>),
}

impl ColumnData {
    /// Materialize typed array from a `Vec<Scalar>` and its `ValidityMask`.
    ///
    /// Invalid positions get a default sentinel (0 / 0.0 / false / "").
    /// The caller must pair this with the corresponding `ValidityMask` to
    /// interpret which positions are actually valid.
    #[must_use]
    pub fn from_scalars(values: &[Scalar], dtype: DType) -> Self {
        match dtype {
            DType::Float64 => {
                let data: Vec<f64> = values
                    .iter()
                    .map(|v| match v {
                        Scalar::Float64(f) => *f,
                        Scalar::Int64(i) => *i as f64,
                        Scalar::Bool(b) => if *b { 1.0 } else { 0.0 },
                        _ => 0.0, // sentinel for invalid positions
                    })
                    .collect();
                Self::Float64(data)
            }
            DType::Int64 => {
                let data: Vec<i64> = values
                    .iter()
                    .map(|v| match v {
                        Scalar::Int64(i) => *i,
                        Scalar::Bool(b) => i64::from(*b),
                        _ => 0, // sentinel for invalid positions
                    })
                    .collect();
                Self::Int64(data)
            }
            DType::Bool => {
                let data: Vec<bool> = values
                    .iter()
                    .map(|v| match v {
                        Scalar::Bool(b) => *b,
                        _ => false,
                    })
                    .collect();
                Self::Bool(data)
            }
            DType::Utf8 => {
                let data: Vec<String> = values
                    .iter()
                    .map(|v| match v {
                        Scalar::Utf8(s) => s.clone(),
                        _ => String::new(),
                    })
                    .collect();
                Self::Utf8(data)
            }
            DType::Null => Self::Float64(vec![0.0; values.len()]),
        }
    }

    /// Convert typed array back to `Vec<Scalar>`, respecting `ValidityMask`.
    #[must_use]
    pub fn to_scalars(&self, dtype: DType, validity: &ValidityMask) -> Vec<Scalar> {
        match self {
            Self::Float64(data) => data
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if !validity.get(i) {
                        Scalar::missing_for_dtype(dtype)
                    } else {
                        Scalar::Float64(*v)
                    }
                })
                .collect(),
            Self::Int64(data) => data
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if !validity.get(i) {
                        Scalar::missing_for_dtype(dtype)
                    } else {
                        Scalar::Int64(*v)
                    }
                })
                .collect(),
            Self::Bool(data) => data
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if !validity.get(i) {
                        Scalar::missing_for_dtype(dtype)
                    } else {
                        Scalar::Bool(*v)
                    }
                })
                .collect(),
            Self::Utf8(data) => data
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if !validity.get(i) {
                        Scalar::missing_for_dtype(dtype)
                    } else {
                        Scalar::Utf8(v.clone())
                    }
                })
                .collect(),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Float64(d) => d.len(),
            Self::Int64(d) => d.len(),
            Self::Bool(d) => d.len(),
            Self::Utf8(d) => d.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// AG-10: Vectorized binary arithmetic on `&[f64]` slices.
///
/// Both inputs must have the same length. The combined validity mask
/// determines which positions produce a valid output; invalid positions
/// get 0.0 as a sentinel. Returns `(result_data, result_validity)`.
fn vectorized_binary_f64(
    left: &[f64],
    right: &[f64],
    left_validity: &ValidityMask,
    right_validity: &ValidityMask,
    op: ArithmeticOp,
) -> (Vec<f64>, ValidityMask) {
    let len = left.len();
    let combined = left_validity.and_mask(right_validity);
    let mut out = vec![0.0_f64; len];

    // Tight loop over contiguous slices — the compiler can auto-vectorize this.
    match op {
        ArithmeticOp::Add => {
            for i in 0..len {
                out[i] = left[i] + right[i];
            }
        }
        ArithmeticOp::Sub => {
            for i in 0..len {
                out[i] = left[i] - right[i];
            }
        }
        ArithmeticOp::Mul => {
            for i in 0..len {
                out[i] = left[i] * right[i];
            }
        }
        ArithmeticOp::Div => {
            for i in 0..len {
                out[i] = left[i] / right[i];
            }
        }
    }

    // Zero out invalid positions so they don't carry garbage.
    for i in 0..len {
        if !combined.get(i) {
            out[i] = 0.0;
        }
    }

    (out, combined)
}

/// AG-10: Vectorized binary arithmetic on `&[i64]` slices.
///
/// Produces `i64` results for Add/Sub/Mul. For Div, returns `None`
/// to signal the caller should use the `f64` path instead.
fn vectorized_binary_i64(
    left: &[i64],
    right: &[i64],
    left_validity: &ValidityMask,
    right_validity: &ValidityMask,
    op: ArithmeticOp,
) -> Option<(Vec<i64>, ValidityMask)> {
    // Division always promotes to Float64 in pandas semantics.
    if matches!(op, ArithmeticOp::Div) {
        return None;
    }

    let len = left.len();
    let combined = left_validity.and_mask(right_validity);
    let mut out = vec![0_i64; len];

    match op {
        ArithmeticOp::Add => {
            for i in 0..len {
                out[i] = left[i].wrapping_add(right[i]);
            }
        }
        ArithmeticOp::Sub => {
            for i in 0..len {
                out[i] = left[i].wrapping_sub(right[i]);
            }
        }
        ArithmeticOp::Mul => {
            for i in 0..len {
                out[i] = left[i].wrapping_mul(right[i]);
            }
        }
        ArithmeticOp::Div => unreachable!(),
    }

    for i in 0..len {
        if !combined.get(i) {
            out[i] = 0;
        }
    }

    Some((out, combined))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Column {
    dtype: DType,
    values: Vec<Scalar>,
    validity: ValidityMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArithmeticOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ColumnError {
    #[error("column length mismatch: left={left}, right={right}")]
    LengthMismatch { left: usize, right: usize },
    #[error(transparent)]
    Type(#[from] TypeError),
}

impl Column {
    /// Construct a column, coercing values to the target dtype.
    /// AG-03: takes ownership of the values vec and uses `cast_scalar_owned`
    /// to skip cloning when values already have the correct dtype.
    pub fn new(dtype: DType, values: Vec<Scalar>) -> Result<Self, ColumnError> {
        let needs_coercion = values.iter().any(|v| {
            let d = v.dtype();
            d != dtype && d != DType::Null
        });

        let coerced = if needs_coercion {
            values
                .into_iter()
                .map(|value| cast_scalar_owned(value, dtype))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            // No coercion needed: values already match dtype.
            // Only remap Null variants to the dtype-specific missing marker.
            values
                .into_iter()
                .map(|value| match value {
                    Scalar::Null(_) => Scalar::missing_for_dtype(dtype),
                    other => other,
                })
                .collect()
        };

        let validity = ValidityMask::from_values(&coerced);

        Ok(Self {
            dtype,
            values: coerced,
            validity,
        })
    }

    pub fn from_values(values: Vec<Scalar>) -> Result<Self, ColumnError> {
        let dtype = infer_dtype(&values)?;
        Self::new(dtype, values)
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    #[must_use]
    pub fn values(&self) -> &[Scalar] {
        &self.values
    }

    #[must_use]
    pub fn value(&self, idx: usize) -> Option<&Scalar> {
        self.values.get(idx)
    }

    #[must_use]
    pub fn validity(&self) -> &ValidityMask {
        &self.validity
    }

    pub fn reindex_by_positions(&self, positions: &[Option<usize>]) -> Result<Self, ColumnError> {
        let values = positions
            .iter()
            .map(|slot| match slot {
                Some(idx) => self
                    .values
                    .get(*idx)
                    .cloned()
                    .unwrap_or_else(|| Scalar::missing_for_dtype(self.dtype)),
                None => Scalar::missing_for_dtype(self.dtype),
            })
            .collect::<Vec<_>>();

        Self::new(self.dtype, values)
    }

    /// AG-10: Attempt vectorized typed-array path for binary arithmetic.
    ///
    /// Preconditions: both columns same length, out_dtype already computed.
    /// Returns `Some(Column)` if vectorized path succeeded, `None` to
    /// signal fallback to the scalar path.
    fn try_vectorized_binary(
        &self,
        right: &Self,
        op: ArithmeticOp,
        out_dtype: DType,
    ) -> Option<Result<Self, ColumnError>> {
        // Vectorized path: both sides same numeric dtype, no NaN-vs-Null
        // distinction needed (i.e. both Int64, or both Float64 / promoted to Float64).
        match out_dtype {
            DType::Float64 => {
                let left_data = ColumnData::from_scalars(&self.values, DType::Float64);
                let right_data = ColumnData::from_scalars(&right.values, DType::Float64);
                let (ColumnData::Float64(l), ColumnData::Float64(r)) = (&left_data, &right_data)
                else {
                    return None;
                };

                // We need NaN-aware validity: original validity + NaN propagation.
                // Build validity masks that treat NaN source scalars as invalid.
                let left_nan_aware = self.nan_aware_validity();
                let right_nan_aware = right.nan_aware_validity();

                let (result_data, result_validity) =
                    vectorized_binary_f64(l, r, &left_nan_aware, &right_nan_aware, op);

                // Build output scalars respecting NaN propagation: if either
                // input was NaN (not just Null), mark output as Null(NaN).
                let values: Vec<Scalar> = result_data
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        if !result_validity.get(i) {
                            // Preserve NaN vs Null distinction from inputs.
                            if self.is_nan_at(i) || right.is_nan_at(i) {
                                Scalar::Null(NullKind::NaN)
                            } else {
                                Scalar::missing_for_dtype(out_dtype)
                            }
                        } else {
                            Scalar::Float64(*v)
                        }
                    })
                    .collect();

                Some(Self::new(out_dtype, values))
            }
            DType::Int64 if !matches!(op, ArithmeticOp::Div) => {
                // Both must actually be Int64 for the i64 fast path.
                if self.dtype != DType::Int64 || right.dtype != DType::Int64 {
                    return None;
                }
                let left_data = ColumnData::from_scalars(&self.values, DType::Int64);
                let right_data = ColumnData::from_scalars(&right.values, DType::Int64);
                let (ColumnData::Int64(l), ColumnData::Int64(r)) = (&left_data, &right_data)
                else {
                    return None;
                };

                let (result_data, result_validity) =
                    vectorized_binary_i64(l, r, &self.validity, &right.validity, op)?;

                let values: Vec<Scalar> = result_data
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        if !result_validity.get(i) {
                            Scalar::missing_for_dtype(out_dtype)
                        } else {
                            Scalar::Int64(*v)
                        }
                    })
                    .collect();

                Some(Self::new(out_dtype, values))
            }
            _ => None, // Bool, Utf8, etc. — use scalar fallback
        }
    }

    /// Validity mask that also marks NaN float values as invalid.
    #[must_use]
    fn nan_aware_validity(&self) -> ValidityMask {
        let mut mask = self.validity.clone();
        for (i, v) in self.values.iter().enumerate() {
            if matches!(v, Scalar::Float64(f) if f.is_nan()) {
                mask.set(i, false);
            }
        }
        mask
    }

    /// Check if position `i` holds a NaN-class missing value.
    fn is_nan_at(&self, i: usize) -> bool {
        self.values
            .get(i)
            .is_some_and(|v| v.is_nan())
    }

    pub fn binary_numeric(&self, right: &Self, op: ArithmeticOp) -> Result<Self, ColumnError> {
        if self.len() != right.len() {
            return Err(ColumnError::LengthMismatch {
                left: self.len(),
                right: right.len(),
            });
        }

        let mut out_dtype = common_dtype(self.dtype, right.dtype)?;
        if matches!(out_dtype, DType::Bool) {
            out_dtype = DType::Int64;
        }
        if matches!(op, ArithmeticOp::Div) {
            out_dtype = DType::Float64;
        }

        // AG-10: Try vectorized path first; fallback to scalar path.
        if let Some(result) = self.try_vectorized_binary(right, op, out_dtype) {
            return result;
        }

        // Scalar fallback path (original implementation).
        let values = self
            .values
            .iter()
            .zip(&right.values)
            .map(|(left, right)| {
                if left.is_missing() || right.is_missing() {
                    return Ok::<_, ColumnError>(if left.is_nan() || right.is_nan() {
                        Scalar::Null(NullKind::NaN)
                    } else {
                        Scalar::missing_for_dtype(out_dtype)
                    });
                }

                let lhs = left.to_f64()?;
                let rhs = right.to_f64()?;
                let result = match op {
                    ArithmeticOp::Add => lhs + rhs,
                    ArithmeticOp::Sub => lhs - rhs,
                    ArithmeticOp::Mul => lhs * rhs,
                    ArithmeticOp::Div => lhs / rhs,
                };

                if matches!(out_dtype, DType::Int64)
                    && result.is_finite()
                    && result == result.trunc()
                    && result >= i64::MIN as f64
                    && result <= i64::MAX as f64
                {
                    Ok(Scalar::Int64(result as i64))
                } else {
                    Ok(Scalar::Float64(result))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::new(out_dtype, values)
    }

    #[must_use]
    pub fn semantic_eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype
            && self.values.len() == other.values.len()
            && self
                .values
                .iter()
                .zip(&other.values)
                .all(|(left, right)| left.semantic_eq(right))
    }
}

#[cfg(test)]
mod tests {
    use fp_types::{NullKind, Scalar};

    use super::{ArithmeticOp, Column, ValidityMask};

    #[test]
    fn reindex_injects_missing_values() {
        let column = Column::from_values(vec![Scalar::Int64(10), Scalar::Int64(20)])
            .expect("column should build");

        let out = column
            .reindex_by_positions(&[Some(1), None, Some(0)])
            .expect("reindex should work");

        assert_eq!(
            out.values(),
            &[
                Scalar::Int64(20),
                Scalar::Null(NullKind::Null),
                Scalar::Int64(10)
            ]
        );
    }

    #[test]
    fn numeric_addition_propagates_missing() {
        let left = Column::from_values(vec![
            Scalar::Int64(1),
            Scalar::Null(NullKind::Null),
            Scalar::Float64(f64::NAN),
        ])
        .expect("left");
        let right = Column::from_values(vec![Scalar::Int64(2), Scalar::Int64(5), Scalar::Int64(3)])
            .expect("right");

        let out = left
            .binary_numeric(&right, ArithmeticOp::Add)
            .expect("add should pass");

        assert_eq!(out.values()[0], Scalar::Float64(3.0));
        assert_eq!(out.values()[1], Scalar::Null(NullKind::NaN));
        assert_eq!(out.values()[2], Scalar::Null(NullKind::NaN));
    }

    // === Packed Bitvec ValidityMask Tests ===

    #[test]
    fn validity_mask_from_values_packs_correctly() {
        let values = vec![
            Scalar::Int64(1),
            Scalar::Null(NullKind::Null),
            Scalar::Int64(3),
        ];
        let mask = ValidityMask::from_values(&values);
        assert_eq!(mask.len(), 3);
        assert!(mask.get(0));
        assert!(!mask.get(1));
        assert!(mask.get(2));
        assert_eq!(mask.count_valid(), 2);
    }

    #[test]
    fn validity_mask_all_valid() {
        let mask = ValidityMask::all_valid(100);
        assert_eq!(mask.len(), 100);
        assert_eq!(mask.count_valid(), 100);
        for i in 0..100 {
            assert!(mask.get(i), "bit {i} should be valid");
        }
    }

    #[test]
    fn validity_mask_all_invalid() {
        let mask = ValidityMask::all_invalid(100);
        assert_eq!(mask.len(), 100);
        assert_eq!(mask.count_valid(), 0);
        for i in 0..100 {
            assert!(!mask.get(i), "bit {i} should be invalid");
        }
    }

    #[test]
    fn validity_mask_set_and_get() {
        let mut mask = ValidityMask::all_invalid(128);
        mask.set(0, true);
        mask.set(63, true);
        mask.set(64, true);
        mask.set(127, true);
        assert!(mask.get(0));
        assert!(mask.get(63));
        assert!(mask.get(64));
        assert!(mask.get(127));
        assert!(!mask.get(1));
        assert_eq!(mask.count_valid(), 4);

        mask.set(63, false);
        assert!(!mask.get(63));
        assert_eq!(mask.count_valid(), 3);
    }

    #[test]
    fn validity_mask_and_or_not() {
        let mut a = ValidityMask::all_invalid(4);
        a.set(0, true);
        a.set(1, true);

        let mut b = ValidityMask::all_invalid(4);
        b.set(1, true);
        b.set(2, true);

        let and = a.and_mask(&b);
        assert!(and.get(1));
        assert!(!and.get(0));
        assert!(!and.get(2));
        assert_eq!(and.count_valid(), 1);

        let or = a.or_mask(&b);
        assert!(or.get(0));
        assert!(or.get(1));
        assert!(or.get(2));
        assert!(!or.get(3));
        assert_eq!(or.count_valid(), 3);

        let not_a = a.not_mask();
        assert!(!not_a.get(0));
        assert!(!not_a.get(1));
        assert!(not_a.get(2));
        assert!(not_a.get(3));
        assert_eq!(not_a.count_valid(), 2);
    }

    #[test]
    fn validity_mask_bits_iterator() {
        let values = vec![
            Scalar::Int64(1),
            Scalar::Null(NullKind::Null),
            Scalar::Int64(3),
            Scalar::Float64(f64::NAN),
        ];
        let mask = ValidityMask::from_values(&values);
        let bits: Vec<bool> = mask.bits().collect();
        assert_eq!(bits, vec![true, false, true, false]);
    }

    #[test]
    fn validity_mask_serde_round_trip() {
        let values = vec![
            Scalar::Int64(1),
            Scalar::Null(NullKind::Null),
            Scalar::Int64(3),
        ];
        let mask = ValidityMask::from_values(&values);
        let json = serde_json::to_string(&mask).expect("serialize");
        let back: ValidityMask = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(mask, back);
        // Verify backward-compatible format
        assert!(json.contains("\"bits\""), "should serialize as bits field");
    }

    #[test]
    fn validity_mask_empty() {
        let mask = ValidityMask::from_values(&[]);
        assert!(mask.is_empty());
        assert_eq!(mask.len(), 0);
        assert_eq!(mask.count_valid(), 0);
        assert_eq!(mask.bits().count(), 0);
    }

    #[test]
    fn validity_mask_boundary_65_elements() {
        let mut values = vec![Scalar::Int64(1); 65];
        values[64] = Scalar::Null(NullKind::Null);
        let mask = ValidityMask::from_values(&values);
        assert_eq!(mask.len(), 65);
        assert_eq!(mask.count_valid(), 64);
        assert!(mask.get(63));
        assert!(!mask.get(64));
    }

    #[test]
    fn validity_mask_equality() {
        let a = ValidityMask::from_values(&[Scalar::Int64(1), Scalar::Null(NullKind::Null)]);
        let b = ValidityMask::from_values(&[Scalar::Int64(1), Scalar::Null(NullKind::Null)]);
        let c = ValidityMask::from_values(&[Scalar::Null(NullKind::Null), Scalar::Int64(1)]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn validity_mask_nan_is_invalid() {
        let values = vec![
            Scalar::Float64(1.0),
            Scalar::Float64(f64::NAN),
            Scalar::Null(NullKind::NaN),
        ];
        let mask = ValidityMask::from_values(&values);
        assert!(mask.get(0));
        assert!(!mask.get(1), "Float64(NaN) should be invalid");
        assert!(!mask.get(2), "Null(NaN) should be invalid");
        assert_eq!(mask.count_valid(), 1);
    }

    #[test]
    fn validity_mask_dense_null_half() {
        let values: Vec<Scalar> = (0..1000)
            .map(|i| {
                if i % 2 == 0 {
                    Scalar::Int64(i)
                } else {
                    Scalar::Null(NullKind::Null)
                }
            })
            .collect();
        let mask = ValidityMask::from_values(&values);
        assert_eq!(mask.len(), 1000);
        assert_eq!(mask.count_valid(), 500);
    }

    // === AG-10: ColumnData and Vectorized Path Tests ===

    #[test]
    fn column_data_float64_roundtrip() {
        let values = vec![Scalar::Float64(1.5), Scalar::Null(NullKind::NaN), Scalar::Float64(3.0)];
        let validity = ValidityMask::from_values(&values);
        let data = super::ColumnData::from_scalars(&values, fp_types::DType::Float64);
        let back = data.to_scalars(fp_types::DType::Float64, &validity);
        assert_eq!(back.len(), 3);
        assert_eq!(back[0], Scalar::Float64(1.5));
        assert!(back[1].is_nan(), "position 1 should be NaN-missing");
        assert_eq!(back[2], Scalar::Float64(3.0));
    }

    #[test]
    fn column_data_int64_roundtrip() {
        let values = vec![Scalar::Int64(10), Scalar::Null(NullKind::Null), Scalar::Int64(30)];
        let validity = ValidityMask::from_values(&values);
        let data = super::ColumnData::from_scalars(&values, fp_types::DType::Int64);
        assert_eq!(data.len(), 3);
        let back = data.to_scalars(fp_types::DType::Int64, &validity);
        assert_eq!(back[0], Scalar::Int64(10));
        assert!(back[1].is_missing());
        assert_eq!(back[2], Scalar::Int64(30));
    }

    #[test]
    fn vectorized_f64_addition_matches_scalar() {
        let left = Column::from_values(vec![
            Scalar::Float64(1.0),
            Scalar::Float64(2.0),
            Scalar::Float64(3.0),
        ])
        .expect("left");
        let right = Column::from_values(vec![
            Scalar::Float64(10.0),
            Scalar::Float64(20.0),
            Scalar::Float64(30.0),
        ])
        .expect("right");

        let result = left
            .binary_numeric(&right, ArithmeticOp::Add)
            .expect("add");
        assert_eq!(result.values()[0], Scalar::Float64(11.0));
        assert_eq!(result.values()[1], Scalar::Float64(22.0));
        assert_eq!(result.values()[2], Scalar::Float64(33.0));
    }

    #[test]
    fn vectorized_i64_addition_matches_scalar() {
        let left = Column::from_values(vec![
            Scalar::Int64(1),
            Scalar::Int64(2),
            Scalar::Int64(3),
        ])
        .expect("left");
        let right = Column::from_values(vec![
            Scalar::Int64(10),
            Scalar::Int64(20),
            Scalar::Int64(30),
        ])
        .expect("right");

        let result = left
            .binary_numeric(&right, ArithmeticOp::Add)
            .expect("add");
        assert_eq!(result.values()[0], Scalar::Int64(11));
        assert_eq!(result.values()[1], Scalar::Int64(22));
        assert_eq!(result.values()[2], Scalar::Int64(33));
    }

    #[test]
    fn vectorized_f64_with_nulls_propagates_missing() {
        let left = Column::from_values(vec![
            Scalar::Float64(1.0),
            Scalar::Null(NullKind::NaN),
            Scalar::Float64(3.0),
        ])
        .expect("left");
        let right = Column::from_values(vec![
            Scalar::Float64(10.0),
            Scalar::Float64(20.0),
            Scalar::Null(NullKind::NaN),
        ])
        .expect("right");

        let result = left
            .binary_numeric(&right, ArithmeticOp::Add)
            .expect("add");
        assert_eq!(result.values()[0], Scalar::Float64(11.0));
        assert!(result.values()[1].is_nan(), "null+valid should be NaN");
        assert!(result.values()[2].is_nan(), "valid+null should be NaN");
    }

    #[test]
    fn vectorized_i64_with_nulls_propagates_missing() {
        let left = Column::from_values(vec![
            Scalar::Int64(1),
            Scalar::Null(NullKind::Null),
            Scalar::Int64(3),
        ])
        .expect("left");
        let right = Column::from_values(vec![
            Scalar::Int64(10),
            Scalar::Int64(20),
            Scalar::Null(NullKind::Null),
        ])
        .expect("right");

        let result = left
            .binary_numeric(&right, ArithmeticOp::Add)
            .expect("add");
        assert_eq!(result.values()[0], Scalar::Int64(11));
        assert!(result.values()[1].is_missing());
        assert!(result.values()[2].is_missing());
    }

    #[test]
    fn vectorized_division_promotes_to_float64() {
        let left = Column::from_values(vec![Scalar::Int64(10), Scalar::Int64(21)])
            .expect("left");
        let right = Column::from_values(vec![Scalar::Int64(3), Scalar::Int64(7)])
            .expect("right");

        let result = left
            .binary_numeric(&right, ArithmeticOp::Div)
            .expect("div");
        // Division always promotes to Float64.
        assert_eq!(result.dtype(), fp_types::DType::Float64);
        assert!(matches!(result.values()[0], Scalar::Float64(v) if (v - 10.0/3.0).abs() < 1e-10));
        assert_eq!(result.values()[1], Scalar::Float64(3.0));
    }

    #[test]
    fn vectorized_all_four_ops_f64() {
        let left = Column::from_values(vec![Scalar::Float64(10.0)]).expect("left");
        let right = Column::from_values(vec![Scalar::Float64(3.0)]).expect("right");

        let add = left.binary_numeric(&right, ArithmeticOp::Add).expect("add");
        let sub = left.binary_numeric(&right, ArithmeticOp::Sub).expect("sub");
        let mul = left.binary_numeric(&right, ArithmeticOp::Mul).expect("mul");
        let div = left.binary_numeric(&right, ArithmeticOp::Div).expect("div");

        assert_eq!(add.values()[0], Scalar::Float64(13.0));
        assert_eq!(sub.values()[0], Scalar::Float64(7.0));
        assert_eq!(mul.values()[0], Scalar::Float64(30.0));
        assert!(matches!(div.values()[0], Scalar::Float64(v) if (v - 10.0/3.0).abs() < 1e-10));
    }

    #[test]
    fn vectorized_empty_columns() {
        let left = Column::from_values(vec![]).expect("left");
        let right = Column::from_values(vec![]).expect("right");
        let result = left
            .binary_numeric(&right, ArithmeticOp::Add)
            .expect("add empty");
        assert!(result.is_empty());
    }

    #[test]
    fn vectorized_large_column_matches_scalar_semantics() {
        // Build large columns to exercise batch processing.
        let n = 4096;
        let left_values: Vec<Scalar> = (0..n).map(|i| Scalar::Float64(i as f64)).collect();
        let right_values: Vec<Scalar> = (0..n).map(|i| Scalar::Float64((n - i) as f64)).collect();

        let left = Column::from_values(left_values).expect("left");
        let right = Column::from_values(right_values).expect("right");

        let result = left
            .binary_numeric(&right, ArithmeticOp::Add)
            .expect("add");

        // Every position should sum to n.
        for (i, v) in result.values().iter().enumerate() {
            assert_eq!(
                *v,
                Scalar::Float64(n as f64),
                "position {i} should be {n}"
            );
        }
    }

    #[test]
    fn vectorized_nan_vs_null_distinction_preserved() {
        // Float64 column: NaN is a specific kind of missing.
        let left = Column::from_values(vec![
            Scalar::Float64(f64::NAN),
            Scalar::Null(NullKind::NaN),
        ])
        .expect("left");
        let right = Column::from_values(vec![
            Scalar::Float64(1.0),
            Scalar::Float64(2.0),
        ])
        .expect("right");

        let result = left
            .binary_numeric(&right, ArithmeticOp::Add)
            .expect("add");
        // Both positions should be NaN-missing (not generic Null).
        assert!(result.values()[0].is_nan(), "NaN + valid = NaN");
        assert!(result.values()[1].is_nan(), "NaN-null + valid = NaN");
    }

    #[test]
    fn vectorized_mixed_type_falls_back_to_scalar() {
        // Int64 + Float64 promotes to Float64 — vectorized path handles this.
        let left = Column::from_values(vec![Scalar::Int64(1), Scalar::Int64(2)])
            .expect("left");
        let right = Column::from_values(vec![Scalar::Float64(0.5), Scalar::Float64(1.5)])
            .expect("right");

        let result = left
            .binary_numeric(&right, ArithmeticOp::Add)
            .expect("add");
        assert_eq!(result.dtype(), fp_types::DType::Float64);
        assert_eq!(result.values()[0], Scalar::Float64(1.5));
        assert_eq!(result.values()[1], Scalar::Float64(3.5));
    }

    #[test]
    fn vectorized_i64_sub_and_mul() {
        let left = Column::from_values(vec![Scalar::Int64(10), Scalar::Int64(20)])
            .expect("left");
        let right = Column::from_values(vec![Scalar::Int64(3), Scalar::Int64(5)])
            .expect("right");

        let sub = left.binary_numeric(&right, ArithmeticOp::Sub).expect("sub");
        assert_eq!(sub.values()[0], Scalar::Int64(7));
        assert_eq!(sub.values()[1], Scalar::Int64(15));

        let mul = left.binary_numeric(&right, ArithmeticOp::Mul).expect("mul");
        assert_eq!(mul.values()[0], Scalar::Int64(30));
        assert_eq!(mul.values()[1], Scalar::Int64(100));
    }
}
