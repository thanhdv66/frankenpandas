#![forbid(unsafe_code)]

use std::cell::OnceCell;
use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum IndexLabel {
    Int64(i64),
    Utf8(String),
}

impl From<i64> for IndexLabel {
    fn from(value: i64) -> Self {
        Self::Int64(value)
    }
}

impl From<&str> for IndexLabel {
    fn from(value: &str) -> Self {
        Self::Utf8(value.to_owned())
    }
}

impl From<String> for IndexLabel {
    fn from(value: String) -> Self {
        Self::Utf8(value)
    }
}

impl fmt::Display for IndexLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int64(v) => write!(f, "{v}"),
            Self::Utf8(v) => write!(f, "{v}"),
        }
    }
}

/// AG-13: Detected sort order of an index's labels.
///
/// Enables adaptive backend selection: binary search for sorted indexes,
/// HashMap fallback for unsorted. Computed lazily via `OnceCell`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortOrder {
    /// Labels are not in any recognized sorted order.
    Unsorted,
    /// All labels are `Int64` and strictly ascending (no duplicates).
    AscendingInt64,
    /// All labels are `Utf8` and strictly ascending (no duplicates).
    AscendingUtf8,
}

/// Detect the sort order of the label slice.
fn detect_sort_order(labels: &[IndexLabel]) -> SortOrder {
    if labels.len() <= 1 {
        return match labels.first() {
            Some(IndexLabel::Int64(_)) | None => SortOrder::AscendingInt64,
            Some(IndexLabel::Utf8(_)) => SortOrder::AscendingUtf8,
        };
    }

    // Check if all Int64 and strictly ascending.
    let all_int = labels.iter().all(|l| matches!(l, IndexLabel::Int64(_)));
    if all_int {
        let is_sorted = labels.windows(2).all(|w| {
            if let (IndexLabel::Int64(a), IndexLabel::Int64(b)) = (&w[0], &w[1]) {
                a < b
            } else {
                false
            }
        });
        if is_sorted {
            return SortOrder::AscendingInt64;
        }
    }

    // Check if all Utf8 and strictly ascending.
    let all_utf8 = labels.iter().all(|l| matches!(l, IndexLabel::Utf8(_)));
    if all_utf8 {
        let is_sorted = labels.windows(2).all(|w| {
            if let (IndexLabel::Utf8(a), IndexLabel::Utf8(b)) = (&w[0], &w[1]) {
                a < b
            } else {
                false
            }
        });
        if is_sorted {
            return SortOrder::AscendingUtf8;
        }
    }

    SortOrder::Unsorted
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicateKeep {
    First,
    Last,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index {
    labels: Vec<IndexLabel>,
    #[serde(skip)]
    duplicate_cache: OnceCell<bool>,
    /// AG-13: Cached sort order for adaptive backend selection.
    #[serde(skip)]
    sort_order_cache: OnceCell<SortOrder>,
}

impl PartialEq for Index {
    fn eq(&self, other: &Self) -> bool {
        self.labels == other.labels
    }
}

impl Eq for Index {}

fn detect_duplicates(labels: &[IndexLabel]) -> bool {
    let mut seen = HashMap::<&IndexLabel, ()>::new();
    for label in labels {
        if seen.insert(label, ()).is_some() {
            return true;
        }
    }
    false
}

impl Index {
    #[must_use]
    pub fn new(labels: Vec<IndexLabel>) -> Self {
        Self {
            labels,
            duplicate_cache: OnceCell::new(),
            sort_order_cache: OnceCell::new(),
        }
    }

    #[must_use]
    pub fn from_i64(values: Vec<i64>) -> Self {
        Self::new(values.into_iter().map(IndexLabel::from).collect())
    }

    #[must_use]
    pub fn from_utf8(values: Vec<String>) -> Self {
        Self::new(values.into_iter().map(IndexLabel::from).collect())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    #[must_use]
    pub fn labels(&self) -> &[IndexLabel] {
        &self.labels
    }

    #[must_use]
    pub fn has_duplicates(&self) -> bool {
        *self
            .duplicate_cache
            .get_or_init(|| detect_duplicates(&self.labels))
    }

    /// AG-13: Lazily detect and cache the sort order of this index.
    #[must_use]
    fn sort_order(&self) -> SortOrder {
        *self
            .sort_order_cache
            .get_or_init(|| detect_sort_order(&self.labels))
    }

    /// Returns `true` if this index is sorted (strictly ascending, no duplicates).
    #[must_use]
    pub fn is_sorted(&self) -> bool {
        !matches!(self.sort_order(), SortOrder::Unsorted)
    }

    /// AG-13: Adaptive position lookup.
    ///
    /// For sorted `Int64` or `Utf8` indexes, uses binary search (O(log n)).
    /// For unsorted indexes, falls back to linear scan (O(n)).
    #[must_use]
    pub fn position(&self, needle: &IndexLabel) -> Option<usize> {
        match self.sort_order() {
            SortOrder::AscendingInt64 => {
                if let IndexLabel::Int64(target) = needle {
                    self.labels
                        .binary_search_by(|label| {
                            if let IndexLabel::Int64(v) = label {
                                v.cmp(target)
                            } else {
                                std::cmp::Ordering::Less
                            }
                        })
                        .ok()
                } else {
                    None // Type mismatch: no Int64 label can match a Utf8 needle
                }
            }
            SortOrder::AscendingUtf8 => {
                if let IndexLabel::Utf8(target) = needle {
                    self.labels
                        .binary_search_by(|label| {
                            if let IndexLabel::Utf8(v) = label {
                                v.as_str().cmp(target.as_str())
                            } else {
                                std::cmp::Ordering::Less
                            }
                        })
                        .ok()
                } else {
                    None
                }
            }
            SortOrder::Unsorted => self.labels.iter().position(|label| label == needle),
        }
    }

    #[must_use]
    pub fn position_map_first(&self) -> HashMap<IndexLabel, usize> {
        let mut positions = HashMap::with_capacity(self.labels.len());
        for (idx, label) in self.labels.iter().enumerate() {
            positions.entry(label.clone()).or_insert(idx);
        }
        positions
    }

    fn position_map_first_ref(&self) -> HashMap<&IndexLabel, usize> {
        let mut positions = HashMap::with_capacity(self.labels.len());
        for (idx, label) in self.labels.iter().enumerate() {
            positions.entry(label).or_insert(idx);
        }
        positions
    }

    // ── Pandas Index Model: lookup and membership ──────────────────────

    #[must_use]
    pub fn contains(&self, label: &IndexLabel) -> bool {
        self.position(label).is_some()
    }

    #[must_use]
    pub fn get_indexer(&self, target: &Index) -> Vec<Option<usize>> {
        let map = self.position_map_first_ref();
        target
            .labels
            .iter()
            .map(|label| map.get(label).copied())
            .collect()
    }

    #[must_use]
    pub fn isin(&self, values: &[IndexLabel]) -> Vec<bool> {
        let set: HashMap<&IndexLabel, ()> = values.iter().map(|v| (v, ())).collect();
        self.labels.iter().map(|l| set.contains_key(l)).collect()
    }

    // ── Pandas Index Model: deduplication ──────────────────────────────

    #[must_use]
    pub fn unique(&self) -> Self {
        let mut seen = HashMap::<&IndexLabel, ()>::new();
        let labels: Vec<IndexLabel> = self
            .labels
            .iter()
            .filter(|l| seen.insert(l, ()).is_none())
            .cloned()
            .collect();
        Self::new(labels)
    }

    #[must_use]
    pub fn duplicated(&self, keep: DuplicateKeep) -> Vec<bool> {
        let mut result = vec![false; self.labels.len()];
        match keep {
            DuplicateKeep::First => {
                let mut seen = HashMap::<&IndexLabel, ()>::new();
                for (i, label) in self.labels.iter().enumerate() {
                    if seen.insert(label, ()).is_some() {
                        result[i] = true;
                    }
                }
            }
            DuplicateKeep::Last => {
                let mut seen = HashMap::<&IndexLabel, ()>::new();
                for (i, label) in self.labels.iter().enumerate().rev() {
                    if seen.insert(label, ()).is_some() {
                        result[i] = true;
                    }
                }
            }
            DuplicateKeep::None => {
                let mut counts = HashMap::<&IndexLabel, usize>::new();
                for label in &self.labels {
                    *counts.entry(label).or_insert(0) += 1;
                }
                for (i, label) in self.labels.iter().enumerate() {
                    if counts[label] > 1 {
                        result[i] = true;
                    }
                }
            }
        }
        result
    }

    #[must_use]
    pub fn drop_duplicates(&self) -> Self {
        self.unique()
    }

    // ── Pandas Index Model: set operations ─────────────────────────────

    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        let other_set = other.position_map_first_ref();
        let mut seen = HashMap::<&IndexLabel, ()>::new();
        let labels: Vec<IndexLabel> = self
            .labels
            .iter()
            .filter(|l| other_set.contains_key(l) && seen.insert(l, ()).is_none())
            .cloned()
            .collect();
        Self::new(labels)
    }

    #[must_use]
    pub fn union_with(&self, other: &Self) -> Self {
        let mut seen = HashMap::<&IndexLabel, ()>::new();
        let mut labels = Vec::with_capacity(self.labels.len() + other.labels.len());
        for label in self.labels.iter().chain(other.labels.iter()) {
            if seen.insert(label, ()).is_none() {
                labels.push(label.clone());
            }
        }
        Self::new(labels)
    }

    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        let other_set = other.position_map_first_ref();
        let mut seen = HashMap::<&IndexLabel, ()>::new();
        let labels: Vec<IndexLabel> = self
            .labels
            .iter()
            .filter(|l| !other_set.contains_key(l) && seen.insert(l, ()).is_none())
            .cloned()
            .collect();
        Self::new(labels)
    }

    #[must_use]
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        let self_set = self.position_map_first_ref();
        let other_set = other.position_map_first_ref();
        let mut seen = HashMap::<&IndexLabel, ()>::new();
        let mut labels = Vec::new();
        for label in &self.labels {
            if !other_set.contains_key(label) && seen.insert(label, ()).is_none() {
                labels.push(label.clone());
            }
        }
        for label in &other.labels {
            if !self_set.contains_key(label) && seen.insert(label, ()).is_none() {
                labels.push(label.clone());
            }
        }
        Self::new(labels)
    }

    // ── Pandas Index Model: ordering and slicing ───────────────────────

    #[must_use]
    pub fn argsort(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.labels.len()).collect();
        indices.sort_by(|&a, &b| self.labels[a].cmp(&self.labels[b]));
        indices
    }

    #[must_use]
    pub fn sort_values(&self) -> Self {
        let order = self.argsort();
        Self::new(order.iter().map(|&i| self.labels[i].clone()).collect())
    }

    #[must_use]
    pub fn take(&self, indices: &[usize]) -> Self {
        Self::new(indices.iter().map(|&i| self.labels[i].clone()).collect())
    }

    #[must_use]
    pub fn slice(&self, start: usize, len: usize) -> Self {
        let end = (start + len).min(self.labels.len());
        let start = start.min(self.labels.len());
        Self::new(self.labels[start..end].to_vec())
    }

    #[must_use]
    pub fn from_range(start: i64, stop: i64, step: i64) -> Self {
        let mut labels = Vec::new();
        let mut val = start;
        if step > 0 {
            while val < stop {
                labels.push(IndexLabel::Int64(val));
                val += step;
            }
        } else if step < 0 {
            while val > stop {
                labels.push(IndexLabel::Int64(val));
                val += step;
            }
        }
        Self::new(labels)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentPlan {
    pub union_index: Index,
    pub left_positions: Vec<Option<usize>>,
    pub right_positions: Vec<Option<usize>>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum IndexError {
    #[error("alignment vectors must have equal lengths")]
    InvalidAlignmentVectors,
}

/// Alignment mode for index-level join semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignMode {
    /// Only labels present in both indexes.
    Inner,
    /// All left labels; right fills with None for missing.
    Left,
    /// All right labels; left fills with None for missing.
    Right,
    /// All labels from both indexes (union). Default for arithmetic.
    Outer,
}

/// Align two indexes using the specified join mode.
///
/// Returns an `AlignmentPlan` whose `union_index` contains the output index
/// (which may be an intersection, left-only, right-only, or union depending on mode).
pub fn align(left: &Index, right: &Index, mode: AlignMode) -> AlignmentPlan {
    match mode {
        AlignMode::Inner => align_inner(left, right),
        AlignMode::Left => align_left(left, right),
        AlignMode::Right => {
            let plan = align_left(right, left);
            AlignmentPlan {
                union_index: plan.union_index,
                left_positions: plan.right_positions,
                right_positions: plan.left_positions,
            }
        }
        AlignMode::Outer => align_union(left, right),
    }
}

/// Inner alignment: only labels present in both indexes (first-match semantics).
pub fn align_inner(left: &Index, right: &Index) -> AlignmentPlan {
    let right_map = right.position_map_first_ref();

    let mut output_labels = Vec::new();
    let mut left_positions = Vec::new();
    let mut right_positions = Vec::new();

    for (left_pos, label) in left.labels.iter().enumerate() {
        if let Some(&right_pos) = right_map.get(label) {
            output_labels.push(label.clone());
            left_positions.push(Some(left_pos));
            right_positions.push(Some(right_pos));
        }
    }

    AlignmentPlan {
        union_index: Index::new(output_labels),
        left_positions,
        right_positions,
    }
}

/// Left alignment: all left labels preserved, right fills with None for missing.
pub fn align_left(left: &Index, right: &Index) -> AlignmentPlan {
    let right_map = right.position_map_first_ref();

    let mut left_positions = Vec::with_capacity(left.len());
    let mut right_positions = Vec::with_capacity(left.len());

    for (left_pos, label) in left.labels.iter().enumerate() {
        left_positions.push(Some(left_pos));
        right_positions.push(right_map.get(label).copied());
    }

    AlignmentPlan {
        union_index: left.clone(),
        left_positions,
        right_positions,
    }
}

pub fn align_union(left: &Index, right: &Index) -> AlignmentPlan {
    let left_positions_map = left.position_map_first_ref();
    let right_positions_map = right.position_map_first_ref();

    let mut union_labels = Vec::with_capacity(left.labels.len() + right.labels.len());
    union_labels.extend(left.labels.iter().cloned());
    for label in &right.labels {
        if !left_positions_map.contains_key(&label) {
            union_labels.push(label.clone());
        }
    }

    let left_positions = union_labels
        .iter()
        .map(|label| left_positions_map.get(&label).copied())
        .collect();

    let right_positions = union_labels
        .iter()
        .map(|label| right_positions_map.get(&label).copied())
        .collect();

    AlignmentPlan {
        union_index: Index::new(union_labels),
        left_positions,
        right_positions,
    }
}

pub fn validate_alignment_plan(plan: &AlignmentPlan) -> Result<(), IndexError> {
    if plan.left_positions.len() != plan.right_positions.len()
        || plan.left_positions.len() != plan.union_index.len()
    {
        return Err(IndexError::InvalidAlignmentVectors);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Index, IndexLabel, align_union, validate_alignment_plan};

    #[test]
    fn union_alignment_preserves_left_then_right_unseen_order() {
        let left = Index::new(vec![1_i64.into(), 2_i64.into(), 4_i64.into()]);
        let right = Index::new(vec![2_i64.into(), 3_i64.into(), 4_i64.into()]);

        let plan = align_union(&left, &right);
        assert_eq!(
            plan.union_index.labels(),
            &[
                IndexLabel::Int64(1),
                IndexLabel::Int64(2),
                IndexLabel::Int64(4),
                IndexLabel::Int64(3),
            ]
        );
        assert_eq!(plan.left_positions, vec![Some(0), Some(1), Some(2), None]);
        assert_eq!(plan.right_positions, vec![None, Some(0), Some(2), Some(1)]);
        validate_alignment_plan(&plan).expect("plan must be valid");
    }

    #[test]
    fn duplicate_detection_matches_index_surface() {
        let index = Index::new(vec!["a".into(), "a".into(), "b".into()]);
        assert!(index.has_duplicates());
    }

    #[test]
    fn index_equality_ignores_duplicate_cache_state() {
        let index_with_cache = Index::new(vec!["a".into(), "a".into(), "b".into()]);
        assert!(index_with_cache.has_duplicates());

        let fresh_index = Index::new(vec!["a".into(), "a".into(), "b".into()]);
        assert_eq!(index_with_cache, fresh_index);
    }

    // === AG-13: Adaptive Index Backend Tests ===

    #[test]
    fn sorted_int64_index_detected() {
        let index = Index::from_i64(vec![1, 2, 3, 4, 5]);
        assert!(index.is_sorted());
    }

    #[test]
    fn unsorted_int64_index_detected() {
        let index = Index::from_i64(vec![3, 1, 2]);
        assert!(!index.is_sorted());
    }

    #[test]
    fn sorted_utf8_index_detected() {
        let index = Index::from_utf8(vec!["a".into(), "b".into(), "c".into()]);
        assert!(index.is_sorted());
    }

    #[test]
    fn unsorted_utf8_index_detected() {
        let index = Index::from_utf8(vec!["c".into(), "a".into(), "b".into()]);
        assert!(!index.is_sorted());
    }

    #[test]
    fn duplicate_int64_is_not_sorted() {
        let index = Index::from_i64(vec![1, 2, 2, 3]);
        assert!(!index.is_sorted());
    }

    #[test]
    fn empty_index_is_sorted() {
        let index = Index::new(vec![]);
        assert!(index.is_sorted());
    }

    #[test]
    fn single_element_is_sorted() {
        let index = Index::from_i64(vec![42]);
        assert!(index.is_sorted());
    }

    #[test]
    fn binary_search_position_sorted_int64() {
        let index = Index::from_i64(vec![10, 20, 30, 40, 50]);
        assert_eq!(index.position(&IndexLabel::Int64(10)), Some(0));
        assert_eq!(index.position(&IndexLabel::Int64(30)), Some(2));
        assert_eq!(index.position(&IndexLabel::Int64(50)), Some(4));
        assert_eq!(index.position(&IndexLabel::Int64(25)), None);
        assert_eq!(index.position(&IndexLabel::Int64(0)), None);
        assert_eq!(index.position(&IndexLabel::Int64(100)), None);
    }

    #[test]
    fn binary_search_position_sorted_utf8() {
        let index = Index::from_utf8(vec!["apple".into(), "banana".into(), "cherry".into()]);
        assert_eq!(index.position(&IndexLabel::Utf8("apple".into())), Some(0));
        assert_eq!(index.position(&IndexLabel::Utf8("banana".into())), Some(1));
        assert_eq!(index.position(&IndexLabel::Utf8("cherry".into())), Some(2));
        assert_eq!(index.position(&IndexLabel::Utf8("date".into())), None);
    }

    #[test]
    fn type_mismatch_returns_none() {
        let int_index = Index::from_i64(vec![1, 2, 3]);
        // Looking for a Utf8 needle in an Int64 index
        assert_eq!(int_index.position(&IndexLabel::Utf8("1".into())), None);

        let utf8_index = Index::from_utf8(vec!["a".into(), "b".into()]);
        // Looking for an Int64 needle in a Utf8 index
        assert_eq!(utf8_index.position(&IndexLabel::Int64(1)), None);
    }

    #[test]
    fn linear_fallback_for_unsorted_index() {
        let index = Index::from_i64(vec![30, 10, 20]);
        assert!(!index.is_sorted());
        assert_eq!(index.position(&IndexLabel::Int64(30)), Some(0));
        assert_eq!(index.position(&IndexLabel::Int64(10)), Some(1));
        assert_eq!(index.position(&IndexLabel::Int64(20)), Some(2));
        assert_eq!(index.position(&IndexLabel::Int64(99)), None);
    }

    #[test]
    fn binary_search_large_sorted_index() {
        // Verify binary search works correctly on a large sorted index.
        let labels: Vec<i64> = (0..10_000).collect();
        let index = Index::from_i64(labels);
        assert!(index.is_sorted());

        // Check first, middle, last, and missing positions.
        assert_eq!(index.position(&IndexLabel::Int64(0)), Some(0));
        assert_eq!(index.position(&IndexLabel::Int64(5000)), Some(5000));
        assert_eq!(index.position(&IndexLabel::Int64(9999)), Some(9999));
        assert_eq!(index.position(&IndexLabel::Int64(10_000)), None);
        assert_eq!(index.position(&IndexLabel::Int64(-1)), None);
    }

    #[test]
    fn sort_detection_is_cached() {
        let index = Index::from_i64(vec![1, 2, 3]);
        // First call computes and caches.
        assert!(index.is_sorted());
        // Second call should return same result from cache.
        assert!(index.is_sorted());
    }

    #[test]
    fn mixed_label_types_are_unsorted() {
        let index = Index::new(vec![IndexLabel::Int64(1), IndexLabel::Utf8("a".into())]);
        assert!(!index.is_sorted());
    }

    #[test]
    fn position_consistent_sorted_vs_unsorted() {
        // Verify that for a sorted index, binary search gives the same
        // results as a linear scan would.
        let sorted = Index::from_i64(vec![5, 10, 15, 20, 25]);
        assert!(sorted.is_sorted());

        for &target in &[5, 10, 15, 20, 25, 0, 12, 30] {
            let needle = IndexLabel::Int64(target);
            let expected = sorted.labels().iter().position(|l| l == &needle);
            assert_eq!(
                sorted.position(&needle),
                expected,
                "mismatch for target={target}"
            );
        }
    }

    // === bd-2gi.15: Alignment mode tests ===

    use super::{AlignMode, align, align_inner, align_left};

    #[test]
    fn align_inner_keeps_only_overlapping_labels() {
        let left = Index::new(vec![1_i64.into(), 2_i64.into(), 3_i64.into()]);
        let right = Index::new(vec![2_i64.into(), 3_i64.into(), 4_i64.into()]);

        let plan = align_inner(&left, &right);
        assert_eq!(
            plan.union_index.labels(),
            &[IndexLabel::Int64(2), IndexLabel::Int64(3)]
        );
        assert_eq!(plan.left_positions, vec![Some(1), Some(2)]);
        assert_eq!(plan.right_positions, vec![Some(0), Some(1)]);
        validate_alignment_plan(&plan).expect("valid");
    }

    #[test]
    fn align_inner_disjoint_yields_empty() {
        let left = Index::new(vec![1_i64.into(), 2_i64.into()]);
        let right = Index::new(vec![3_i64.into(), 4_i64.into()]);

        let plan = align_inner(&left, &right);
        assert!(plan.union_index.is_empty());
        assert!(plan.left_positions.is_empty());
        assert!(plan.right_positions.is_empty());
    }

    #[test]
    fn align_left_preserves_all_left_labels() {
        let left = Index::new(vec!["a".into(), "b".into(), "c".into()]);
        let right = Index::new(vec!["b".into(), "d".into()]);

        let plan = align_left(&left, &right);
        assert_eq!(
            plan.union_index.labels(),
            &["a".into(), "b".into(), "c".into()]
        );
        assert_eq!(plan.left_positions, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(plan.right_positions, vec![None, Some(0), None]);
        validate_alignment_plan(&plan).expect("valid");
    }

    #[test]
    fn align_right_preserves_all_right_labels() {
        let left = Index::new(vec!["a".into(), "b".into()]);
        let right = Index::new(vec!["b".into(), "c".into(), "d".into()]);

        let plan = align(&left, &right, AlignMode::Right);
        assert_eq!(
            plan.union_index.labels(),
            &["b".into(), "c".into(), "d".into()]
        );
        // Left has "b" at position 1.
        assert_eq!(plan.left_positions, vec![Some(1), None, None]);
        assert_eq!(plan.right_positions, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn align_mode_outer_matches_union() {
        let left = Index::new(vec![1_i64.into(), 2_i64.into()]);
        let right = Index::new(vec![2_i64.into(), 3_i64.into()]);

        let plan_outer = align(&left, &right, AlignMode::Outer);
        let plan_union = align_union(&left, &right);
        assert_eq!(plan_outer, plan_union);
    }

    #[test]
    fn align_inner_identical_indexes() {
        let left = Index::new(vec!["x".into(), "y".into()]);
        let right = Index::new(vec!["x".into(), "y".into()]);

        let plan = align_inner(&left, &right);
        assert_eq!(plan.union_index.labels(), &["x".into(), "y".into()]);
        assert_eq!(plan.left_positions, vec![Some(0), Some(1)]);
        assert_eq!(plan.right_positions, vec![Some(0), Some(1)]);
    }

    #[test]
    fn align_left_identical_indexes() {
        let left = Index::new(vec![1_i64.into(), 2_i64.into()]);
        let right = Index::new(vec![1_i64.into(), 2_i64.into()]);

        let plan = align_left(&left, &right);
        assert_eq!(plan.union_index.labels(), left.labels());
        assert_eq!(plan.left_positions, vec![Some(0), Some(1)]);
        assert_eq!(plan.right_positions, vec![Some(0), Some(1)]);
    }

    #[test]
    fn align_inner_empty_input() {
        let left = Index::new(Vec::new());
        let right = Index::new(vec![1_i64.into()]);

        let plan = align_inner(&left, &right);
        assert!(plan.union_index.is_empty());
    }

    #[test]
    fn align_left_empty_left() {
        let left = Index::new(Vec::new());
        let right = Index::new(vec![1_i64.into()]);

        let plan = align_left(&left, &right);
        assert!(plan.union_index.is_empty());
    }

    // === bd-2gi.13: Index model and indexer semantics ===

    use super::DuplicateKeep;

    #[test]
    fn contains_finds_existing_label() {
        let index = Index::from_i64(vec![10, 20, 30]);
        assert!(index.contains(&IndexLabel::Int64(20)));
        assert!(!index.contains(&IndexLabel::Int64(99)));
    }

    #[test]
    fn get_indexer_bulk_lookup() {
        let index = Index::new(vec!["a".into(), "b".into(), "c".into()]);
        let target = Index::new(vec!["c".into(), "a".into(), "z".into()]);
        assert_eq!(index.get_indexer(&target), vec![Some(2), Some(0), None]);
    }

    #[test]
    fn isin_membership_mask() {
        let index = Index::from_i64(vec![1, 2, 3, 4, 5]);
        let values = vec![IndexLabel::Int64(2), IndexLabel::Int64(4)];
        assert_eq!(index.isin(&values), vec![false, true, false, true, false]);
    }

    #[test]
    fn unique_preserves_first_seen_order() {
        let index = Index::new(vec![
            "b".into(),
            "a".into(),
            "b".into(),
            "c".into(),
            "a".into(),
        ]);
        let uniq = index.unique();
        assert_eq!(
            uniq.labels(),
            &["b".into(), "a".into(), "c".into()]
        );
    }

    #[test]
    fn duplicated_keep_first() {
        let index = Index::from_i64(vec![1, 2, 1, 3, 2]);
        assert_eq!(
            index.duplicated(DuplicateKeep::First),
            vec![false, false, true, false, true]
        );
    }

    #[test]
    fn duplicated_keep_last() {
        let index = Index::from_i64(vec![1, 2, 1, 3, 2]);
        assert_eq!(
            index.duplicated(DuplicateKeep::Last),
            vec![true, true, false, false, false]
        );
    }

    #[test]
    fn duplicated_keep_none_marks_all() {
        let index = Index::from_i64(vec![1, 2, 1, 3, 2]);
        assert_eq!(
            index.duplicated(DuplicateKeep::None),
            vec![true, true, true, false, true]
        );
    }

    #[test]
    fn drop_duplicates_equals_unique() {
        let index = Index::from_i64(vec![3, 1, 3, 2, 1]);
        assert_eq!(index.drop_duplicates(), index.unique());
    }

    #[test]
    fn intersection_preserves_left_order() {
        let left = Index::new(vec!["c".into(), "a".into(), "b".into()]);
        let right = Index::new(vec!["b".into(), "d".into(), "a".into()]);
        let result = left.intersection(&right);
        assert_eq!(result.labels(), &["a".into(), "b".into()]);
    }

    #[test]
    fn intersection_deduplicates() {
        let left = Index::from_i64(vec![1, 1, 2]);
        let right = Index::from_i64(vec![1, 2, 2]);
        let result = left.intersection(&right);
        assert_eq!(result.labels(), &[IndexLabel::Int64(1), IndexLabel::Int64(2)]);
    }

    #[test]
    fn union_with_combines_unique_labels() {
        let left = Index::from_i64(vec![1, 2, 3]);
        let right = Index::from_i64(vec![2, 4, 3]);
        let result = left.union_with(&right);
        assert_eq!(
            result.labels(),
            &[
                IndexLabel::Int64(1),
                IndexLabel::Int64(2),
                IndexLabel::Int64(3),
                IndexLabel::Int64(4),
            ]
        );
    }

    #[test]
    fn difference_removes_other_labels() {
        let left = Index::from_i64(vec![1, 2, 3, 4]);
        let right = Index::from_i64(vec![2, 4]);
        let result = left.difference(&right);
        assert_eq!(
            result.labels(),
            &[IndexLabel::Int64(1), IndexLabel::Int64(3)]
        );
    }

    #[test]
    fn symmetric_difference_xor() {
        let left = Index::from_i64(vec![1, 2, 3]);
        let right = Index::from_i64(vec![2, 3, 4]);
        let result = left.symmetric_difference(&right);
        assert_eq!(
            result.labels(),
            &[IndexLabel::Int64(1), IndexLabel::Int64(4)]
        );
    }

    #[test]
    fn argsort_returns_sorting_indices() {
        let index = Index::from_i64(vec![30, 10, 20]);
        assert_eq!(index.argsort(), vec![1, 2, 0]);
    }

    #[test]
    fn sort_values_produces_sorted_index() {
        let index = Index::new(vec!["c".into(), "a".into(), "b".into()]);
        let sorted = index.sort_values();
        assert_eq!(
            sorted.labels(),
            &["a".into(), "b".into(), "c".into()]
        );
    }

    #[test]
    fn take_selects_by_position() {
        let index = Index::from_i64(vec![10, 20, 30, 40, 50]);
        let taken = index.take(&[4, 0, 2]);
        assert_eq!(
            taken.labels(),
            &[
                IndexLabel::Int64(50),
                IndexLabel::Int64(10),
                IndexLabel::Int64(30),
            ]
        );
    }

    #[test]
    fn slice_extracts_subrange() {
        let index = Index::from_i64(vec![10, 20, 30, 40, 50]);
        let sliced = index.slice(1, 3);
        assert_eq!(
            sliced.labels(),
            &[
                IndexLabel::Int64(20),
                IndexLabel::Int64(30),
                IndexLabel::Int64(40),
            ]
        );
    }

    #[test]
    fn slice_clamps_to_bounds() {
        let index = Index::from_i64(vec![1, 2, 3]);
        let sliced = index.slice(1, 100);
        assert_eq!(
            sliced.labels(),
            &[IndexLabel::Int64(2), IndexLabel::Int64(3)]
        );
    }

    #[test]
    fn from_range_basic() {
        let index = Index::from_range(0, 5, 1);
        assert_eq!(
            index.labels(),
            &[
                IndexLabel::Int64(0),
                IndexLabel::Int64(1),
                IndexLabel::Int64(2),
                IndexLabel::Int64(3),
                IndexLabel::Int64(4),
            ]
        );
    }

    #[test]
    fn from_range_step_2() {
        let index = Index::from_range(0, 10, 3);
        assert_eq!(
            index.labels(),
            &[
                IndexLabel::Int64(0),
                IndexLabel::Int64(3),
                IndexLabel::Int64(6),
                IndexLabel::Int64(9),
            ]
        );
    }

    #[test]
    fn from_range_negative_step() {
        let index = Index::from_range(5, 0, -2);
        assert_eq!(
            index.labels(),
            &[
                IndexLabel::Int64(5),
                IndexLabel::Int64(3),
                IndexLabel::Int64(1),
            ]
        );
    }

    #[test]
    fn from_range_empty_when_step_zero() {
        let index = Index::from_range(0, 5, 0);
        assert!(index.is_empty());
    }

    #[test]
    fn set_ops_empty_inputs() {
        let empty = Index::new(Vec::new());
        let non_empty = Index::from_i64(vec![1, 2]);
        assert!(empty.intersection(&non_empty).is_empty());
        assert_eq!(empty.union_with(&non_empty), non_empty);
        assert!(empty.difference(&non_empty).is_empty());
        assert_eq!(empty.symmetric_difference(&non_empty), non_empty);
    }
}
