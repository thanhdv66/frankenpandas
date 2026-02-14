#![forbid(unsafe_code)]

use std::{collections::HashMap, mem::size_of};

use bumpalo::{collections::Vec as BumpVec, Bump};
use fp_columnar::{Column, ColumnError};
use fp_frame::{FrameError, Series};
use fp_index::{Index, IndexLabel};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JoinedSeries {
    pub index: Index,
    pub left_values: Column,
    pub right_values: Column,
}

#[derive(Debug, Error)]
pub enum JoinError {
    #[error(transparent)]
    Frame(#[from] FrameError),
    #[error(transparent)]
    Column(#[from] ColumnError),
}

pub const DEFAULT_ARENA_BUDGET_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinExecutionOptions {
    pub use_arena: bool,
    pub arena_budget_bytes: usize,
}

impl Default for JoinExecutionOptions {
    fn default() -> Self {
        Self {
            use_arena: true,
            arena_budget_bytes: DEFAULT_ARENA_BUDGET_BYTES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct JoinExecutionTrace {
    used_arena: bool,
    output_rows: usize,
    estimated_bytes: usize,
}

pub fn join_series(
    left: &Series,
    right: &Series,
    join_type: JoinType,
) -> Result<JoinedSeries, JoinError> {
    join_series_with_options(left, right, join_type, JoinExecutionOptions::default())
}

pub fn join_series_with_options(
    left: &Series,
    right: &Series,
    join_type: JoinType,
    options: JoinExecutionOptions,
) -> Result<JoinedSeries, JoinError> {
    let (joined, _) = join_series_with_trace(left, right, join_type, options)?;
    Ok(joined)
}

fn join_series_with_trace(
    left: &Series,
    right: &Series,
    join_type: JoinType,
    options: JoinExecutionOptions,
) -> Result<(JoinedSeries, JoinExecutionTrace), JoinError> {
    // AG-02: borrowed-key HashMap eliminates right-index label clones during build phase.
    let mut right_map = HashMap::<&IndexLabel, Vec<usize>>::new();
    for (pos, label) in right.index().labels().iter().enumerate() {
        right_map.entry(label).or_default().push(pos);
    }

    let output_rows = estimate_output_rows(left, &right_map, join_type);
    let estimated_bytes = estimate_intermediate_bytes(output_rows);
    let use_arena = options.use_arena && estimated_bytes <= options.arena_budget_bytes;

    let joined = if use_arena {
        join_series_with_arena(left, right, join_type, &right_map, output_rows)?
    } else {
        join_series_with_global_allocator(left, right, join_type, &right_map, output_rows)?
    };

    Ok((
        joined,
        JoinExecutionTrace {
            used_arena: use_arena,
            output_rows,
            estimated_bytes,
        },
    ))
}

fn estimate_output_rows(
    left: &Series,
    right_map: &HashMap<&IndexLabel, Vec<usize>>,
    join_type: JoinType,
) -> usize {
    left.index()
        .labels()
        .iter()
        .map(|label| match right_map.get(label) {
            Some(matches) => matches.len(),
            None if matches!(join_type, JoinType::Left) => 1,
            None => 0,
        })
        .sum()
}

fn estimate_intermediate_bytes(output_rows: usize) -> usize {
    output_rows.saturating_mul(
        size_of::<Option<usize>>()
            .saturating_mul(2)
            .saturating_add(size_of::<IndexLabel>()),
    )
}

fn join_series_with_global_allocator(
    left: &Series,
    right: &Series,
    join_type: JoinType,
    right_map: &HashMap<&IndexLabel, Vec<usize>>,
    output_rows: usize,
) -> Result<JoinedSeries, JoinError> {
    let mut out_labels = Vec::with_capacity(output_rows);
    let mut left_positions = Vec::<Option<usize>>::with_capacity(output_rows);
    let mut right_positions = Vec::<Option<usize>>::with_capacity(output_rows);

    for (left_pos, label) in left.index().labels().iter().enumerate() {
        if let Some(matches) = right_map.get(label) {
            for right_pos in matches {
                out_labels.push(label.clone());
                left_positions.push(Some(left_pos));
                right_positions.push(Some(*right_pos));
            }
            continue;
        }

        if matches!(join_type, JoinType::Left) {
            out_labels.push(label.clone());
            left_positions.push(Some(left_pos));
            right_positions.push(None);
        }
    }

    let left_values = left.column().reindex_by_positions(&left_positions)?;
    let right_values = right.column().reindex_by_positions(&right_positions)?;

    Ok(JoinedSeries {
        index: Index::new(out_labels),
        left_values,
        right_values,
    })
}

fn join_series_with_arena(
    left: &Series,
    right: &Series,
    join_type: JoinType,
    right_map: &HashMap<&IndexLabel, Vec<usize>>,
    output_rows: usize,
) -> Result<JoinedSeries, JoinError> {
    let arena = Bump::new();
    let mut out_labels = Vec::with_capacity(output_rows);
    let mut left_positions = BumpVec::<Option<usize>>::with_capacity_in(output_rows, &arena);
    let mut right_positions = BumpVec::<Option<usize>>::with_capacity_in(output_rows, &arena);

    for (left_pos, label) in left.index().labels().iter().enumerate() {
        if let Some(matches) = right_map.get(label) {
            for right_pos in matches {
                out_labels.push(label.clone());
                left_positions.push(Some(left_pos));
                right_positions.push(Some(*right_pos));
            }
            continue;
        }

        if matches!(join_type, JoinType::Left) {
            out_labels.push(label.clone());
            left_positions.push(Some(left_pos));
            right_positions.push(None);
        }
    }

    let left_values = left
        .column()
        .reindex_by_positions(left_positions.as_slice())?;
    let right_values = right
        .column()
        .reindex_by_positions(right_positions.as_slice())?;

    Ok(JoinedSeries {
        index: Index::new(out_labels),
        left_values,
        right_values,
    })
}

#[cfg(test)]
mod tests {
    use fp_types::{NullKind, Scalar};

    use super::{
        join_series, join_series_with_options, join_series_with_trace, JoinExecutionOptions,
        JoinType,
    };
    use fp_frame::Series;

    #[test]
    fn inner_join_multiplies_cardinality_for_duplicates() {
        let left = Series::from_values(
            "left",
            vec!["k".into(), "k".into(), "x".into()],
            vec![Scalar::Int64(1), Scalar::Int64(2), Scalar::Int64(3)],
        )
        .expect("left");

        let right = Series::from_values(
            "right",
            vec!["k".into(), "k".into()],
            vec![Scalar::Int64(10), Scalar::Int64(20)],
        )
        .expect("right");

        let out = join_series(&left, &right, JoinType::Inner).expect("join");
        assert_eq!(out.index.labels().len(), 4);
        assert_eq!(
            out.left_values.values(),
            &[
                Scalar::Int64(1),
                Scalar::Int64(1),
                Scalar::Int64(2),
                Scalar::Int64(2)
            ]
        );
    }

    #[test]
    fn left_join_injects_missing_for_unmatched_right_rows() {
        let left = Series::from_values(
            "left",
            vec!["a".into(), "b".into()],
            vec![Scalar::Int64(1), Scalar::Int64(2)],
        )
        .expect("left");
        let right =
            Series::from_values("right", vec!["a".into()], vec![Scalar::Int64(10)]).expect("right");

        let out = join_series(&left, &right, JoinType::Left).expect("join");
        assert_eq!(
            out.right_values.values(),
            &[Scalar::Int64(10), Scalar::Null(NullKind::Null)]
        );
    }

    #[test]
    fn arena_join_matches_global_allocator_behavior() {
        let left = Series::from_values(
            "left",
            vec!["k".into(), "k".into(), "x".into()],
            vec![Scalar::Int64(1), Scalar::Int64(2), Scalar::Int64(3)],
        )
        .expect("left");

        let right = Series::from_values(
            "right",
            vec!["k".into(), "k".into()],
            vec![Scalar::Int64(10), Scalar::Int64(20)],
        )
        .expect("right");

        let global = join_series_with_options(
            &left,
            &right,
            JoinType::Inner,
            JoinExecutionOptions {
                use_arena: false,
                arena_budget_bytes: 0,
            },
        )
        .expect("global join");

        let arena = join_series_with_options(
            &left,
            &right,
            JoinType::Inner,
            JoinExecutionOptions::default(),
        )
        .expect("arena join");

        assert_eq!(arena, global);
    }

    #[test]
    fn arena_join_falls_back_when_budget_is_too_small() {
        let left = Series::from_values(
            "left",
            vec!["a".into(), "a".into(), "a".into()],
            vec![Scalar::Int64(1), Scalar::Int64(2), Scalar::Int64(3)],
        )
        .expect("left");
        let right = Series::from_values(
            "right",
            vec!["a".into(), "a".into()],
            vec![Scalar::Int64(10), Scalar::Int64(20)],
        )
        .expect("right");

        let options = JoinExecutionOptions {
            use_arena: true,
            arena_budget_bytes: 1,
        };
        let (fallback_out, trace) =
            join_series_with_trace(&left, &right, JoinType::Inner, options).expect("fallback join");
        let global_out = join_series_with_options(
            &left,
            &right,
            JoinType::Inner,
            JoinExecutionOptions {
                use_arena: false,
                arena_budget_bytes: 0,
            },
        )
        .expect("global join");

        assert_eq!(fallback_out, global_out);
        assert!(!trace.used_arena);
        assert!(trace.estimated_bytes > options.arena_budget_bytes);
    }

    #[test]
    fn arena_join_is_stable_across_many_small_operations() {
        let left = Series::from_values(
            "left",
            vec!["a".into(), "b".into()],
            vec![Scalar::Int64(1), Scalar::Int64(2)],
        )
        .expect("left");
        let right = Series::from_values(
            "right",
            vec!["a".into(), "b".into()],
            vec![Scalar::Int64(10), Scalar::Int64(20)],
        )
        .expect("right");

        let options = JoinExecutionOptions::default();
        for _ in 0..1_000 {
            let out = join_series_with_options(&left, &right, JoinType::Inner, options)
                .expect("arena join");
            assert_eq!(out.index.labels().len(), 2);
            assert_eq!(
                out.right_values.values(),
                &[Scalar::Int64(10), Scalar::Int64(20)]
            );
        }
    }
}
