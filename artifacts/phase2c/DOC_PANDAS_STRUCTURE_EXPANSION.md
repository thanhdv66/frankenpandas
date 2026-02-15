# EXISTING_PANDAS_STRUCTURE.md -- Expansion Draft (Pass A)

**Bead:** bd-2gi.23.11 (DOC-PASS-10)
**Date:** 2026-02-14
**Status:** Complete
**Synthesized From:** DOC_GAP_MATRIX, DOC_MODULE_CARTOGRAPHY, DOC_API_CENSUS, DOC_DATA_MODEL_MAP, DOC_EXECUTION_PATHS
**Target:** Definitive structural reference for pandas internals informing FrankenPandas Rust reimplementation

---

## Table of Contents

1. [Overview and Scale](#1-overview-and-scale)
2. [Layered Architecture](#2-layered-architecture)
3. [Subsystem-Level Topology](#3-subsystem-level-topology)
   - 3.1 Foundation Layer (Layer 0)
   - 3.2 Cython/C Extension Layer (Layer 1)
   - 3.3 Core Engine (Layer 2)
   - 3.4 I/O Layer (Layer 3)
   - 3.5 Visualization Layer (Layer 4)
   - 3.6 Public API Surface (Layer 5)
4. [Architecture Decisions](#4-architecture-decisions)
   - 4.1 BlockManager Design
   - 4.2 ExtensionArray System
   - 4.3 Copy-on-Write (CoW)
   - 4.4 DType System and Type Algebra
   - 4.5 Index Engine Architecture
5. [State Machine Descriptions](#5-state-machine-descriptions)
   - 5.1 Block Consolidation State Machine
   - 5.2 Copy-on-Write Reference Tracking
   - 5.3 Index Caching and Memoization
   - 5.4 DType Promotion State Transitions
   - 5.5 Lazy Evaluation Points
6. [Boundary Narratives](#6-boundary-narratives)
   - 6.1 Dependency Direction Rules
   - 6.2 Boundary Violations
   - 6.3 Import Count Matrix
   - 6.4 Internal Cross-Dependencies
7. [Cross-Subsystem Integration](#7-cross-subsystem-integration)
   - 7.1 DataFrame Construction Pipeline
   - 7.2 Binary Operations and Alignment
   - 7.3 GroupBy into Ops Pipeline
   - 7.4 Merge/Join Using Indexing and Sorting
   - 7.5 CSV Read Through Parser to DataFrame
   - 7.6 loc/iloc Indexing Dispatch
8. [Semantic Hotspots](#8-semantic-hotspots)
   - 8.1 Constructor Alignment and Copy/View
   - 8.2 Series Alignment During Arithmetic
   - 8.3 Index Object Contracts
   - 8.4 BlockManager Invariants
   - 8.5 loc/iloc Distinction
   - 8.6 GroupBy/Window Defaults
   - 8.7 Operator Dispatch and DType Promotion
9. [Compatibility-Critical Behaviors](#9-compatibility-critical-behaviors)
10. [Security and Stability Risk Areas](#10-security-and-stability-risk-areas)
11. [V1 Extraction Boundary](#11-v1-extraction-boundary)
12. [Conformance Fixture Families](#12-conformance-fixture-families)
13. [FrankenPandas Structural Divergences](#13-frankenpandas-structural-divergences)
14. [Extraction Notes for Rust Spec](#14-extraction-notes-for-rust-spec)

---

## 1. Overview and Scale

### 1.1 Codebase Dimensions

The pandas source tree (excluding tests) comprises **293 Python modules** and **40 Cython modules** across **51 packages**, with the following LOC distribution:

| Category | LOC | Notes |
|----------|-----|-------|
| Python source | 249,270 | core + io + plotting + util + api + etc. |
| Cython source (.pyx) | 45,723 | _libs + tslibs + window |
| C source | 9,840 | Vendored parsers, datetime, ujson |
| C headers | 2,559 | klib, numpy datetime, public API |
| Cython templates (.pxi.in) | 3,251 | Typed code generation for hashtable, algos, sparse |
| **Total non-test** | **~310,643** | |
| Test source | 384,382 | 1,120 test files across all domains |

### 1.2 LOC Distribution by Top-Level Package

| Package | Python LOC | Cython LOC | Total | % of Total |
|---------|-----------|------------|-------|------------|
| `core/` | 179,016 | -- | 179,016 | 60.7% |
| `_libs/` | 116 | 45,723 | 45,839 | 15.5% |
| `io/` | 47,386 | -- | 47,386 | 16.1% |
| `plotting/` | 9,470 | -- | 9,470 | 3.2% |
| `_testing/` | 2,813 | -- | 2,813 | 1.0% |
| `util/` | 2,132 | -- | 2,132 | 0.7% |
| `tseries/` | 1,426 | -- | 1,426 | 0.5% |
| `_config/` | 1,263 | -- | 1,263 | 0.4% |
| `errors/` | 1,154 | -- | 1,154 | 0.4% |
| `compat/` | 1,000 | -- | 1,000 | 0.3% |
| Other (api, arrays, root) | ~1,494 | -- | ~1,494 | 0.5% |

### 1.3 API Surface Scale

The pandas public API exposes **~545 distinct symbols** across all object types:

| Domain | Symbol Count | FrankenPandas Coverage |
|--------|-------------|----------------------|
| Top-level API (`pd.*`) | 95 | 19% |
| DataFrame methods | ~250 | ~25% |
| Series methods | ~200 | ~20% |
| Index methods | ~120 | 44% |
| GroupBy methods | ~80 | 23% |
| Window methods | ~50 | 0% |
| String accessor | ~60 | 0% |
| DateTime accessor | ~35 | 0% |
| IO functions | 21 | 10% |
| Reshape functions | 14 | 14% |

---

## 2. Layered Architecture

The pandas codebase follows a six-layer architecture with well-defined (though occasionally violated) dependency constraints:

```
Layer 5: api/           Public API surface re-exports (375 LOC)
Layer 4: plotting/      Visualization (9,470 LOC) -- depends on core, io.formats
Layer 3: io/            I/O readers/writers (47,386 LOC) -- depends on core, _libs
Layer 2: core/          Engine: DataFrame, Series, indexes, groupby, etc. (179,016 LOC)
         tseries/       Time series offsets/frequencies (1,426 LOC)
Layer 1: _libs/         Cython/C extensions: algos, parsers, datetime (45,839 LOC)
Layer 0: _config/       Configuration system (1,263 LOC)
         errors/        Exception hierarchy (1,154 LOC)
         compat/        Compatibility shims (1,000 LOC)
         _typing.py     Type aliases (578 LOC)
         util/          Decorators, validators, helpers (2,132 LOC)
```

**Layering constraints (design intent):**

- Layer 0 packages SHOULD NOT import from layers above.
- `_libs/` (Layer 1) SHOULD NOT import from `core/` (Layer 2). **VIOLATED** -- see Section 6.2.
- `core/` (Layer 2) SHOULD NOT import from `io/` (Layer 3). **VIOLATED** -- see Section 6.2.
- `io/` (Layer 3) depends on `core/` legitimately (constructs DataFrames from parsed data).
- `plotting/` (Layer 4) depends on `core/` and `io.formats` legitimately.
- `api/` (Layer 5) re-exports from all layers by design.

### Package-Level Dependency Graph

```
                    +--------+
                    |  api/  |  <-- re-export surface
                    +--------+
                        |
            +-----------+-----------+
            |           |           |
       +--------+  +--------+  +----------+
       |plotting|  |  io/   |  |_testing/ |
       +--------+  +--------+  +----------+
            |       /   |           |
            +------+    |           |
            |           |           |
       +--------+       |           |
       | core/  |<------+-----------|
       +--------+
          |   |
    +-----+   +-----+
    |               |
+--------+    +--------+
| _libs/ |    |tseries/|
+--------+    +--------+
    |
+--------+    +--------+    +--------+    +--------+
|_config/|    |errors/ |    |compat/ |    | util/  |
+--------+    +--------+    +--------+    +--------+
```

---

## 3. Subsystem-Level Topology

### 3.1 Foundation Layer (Layer 0)

#### `_config/` -- Configuration System (1,263 LOC, 5 modules)

Maintains a global registry of configuration options with validators, defaults, and deprecation handling. Core API: `register_option`, `get_option`, `set_option`, `option_context`.

| Module | LOC | Key Responsibility |
|--------|-----|--------------------|
| `config.py` | 954 | Core option registration/retrieval engine |
| `localization.py` | 176 | Locale detection for number/date formatting |
| `display.py` | 62 | Terminal detection, Unicode support |
| `dates.py` | 26 | Date-related display configuration constants |
| `__init__.py` | 45 | Re-exports |

**Depended on by:** `core` (31 imports), `io` (14), `plotting` (1), `_testing` (2).

#### `errors/` -- Exception Hierarchy (1,154 LOC, 2 modules)

Defines ALL pandas exception and warning classes in a single authoritative location. Key types: `PerformanceWarning`, `UnsortedIndexError`, `ParserError`, `MergeError`, `OptionError`, `OutOfBoundsDatetime`, `InvalidIndexError`, `AbstractMethodError`, `IntCastingNaNError`, `DataError`, `DuplicateLabelError`.

The `cow.py` submodule (43 LOC) contains Copy-on-Write specific warning helpers.

**Depended on by:** `core` (55 imports), `io` (18), `_testing` (2), `tseries` (1).

#### `compat/` -- Compatibility Shims (1,000 LOC, 7 modules)

Platform detection (Windows, macOS, ARM, 32/64-bit), Python version checks, and the critical `import_optional_dependency()` function that gates all optional dependencies (numba, pyarrow, openpyxl, etc.) with version checking. Also includes NumPy compatibility wrappers for argument validation and backward-compatible unpickling.

#### `util/` -- Utilities (2,132 LOC, 9 modules)

Cross-cutting utilities including docstring substitution decorators (`@Substitution`, `@Appender`, `@doc`), argument validation helpers (`validate_bool_kwarg`, `validate_axis`, `validate_percentile`), PEP 440 version parsing, and `pd.show_versions()`.

### 3.2 Cython/C Extension Layer (Layer 1)

#### `_libs/` Top-Level Cython Modules (17,275 LOC pyx, 24 modules)

The performance foundation of pandas. These modules implement the hot algorithmic paths that would be too slow in pure Python.

**Critical modules by LOC and impact:**

| Module | LOC (pyx) | Responsibility |
|--------|-----------|----------------|
| `lib.pyx` | 3,325 | Utility megamodule: `maybe_convert_objects`, `infer_dtype`, `is_float/integer`, `no_default` sentinel, type inference |
| `groupby.pyx` | 2,325 | Cython groupby kernels: `group_sum`, `group_mean`, `group_var`, `group_nth`, `group_last`, `group_rank`, `group_cumsum`, `group_shift` |
| `parsers.pyx` | 2,182 | C-backed CSV parser interface wrapping C tokenizer: `TextReader` class |
| `algos.pyx` | 1,445 | Core algorithm primitives: `kth_smallest`, `nancorr`, `nancov`, `is_monotonic`, `pad`/`backfill`, `rank_1d`, `take_2d` |
| `index.pyx` | 1,325 | Index engines: `IndexEngine`, `DatetimeEngine`, `TimedeltaEngine`, `PeriodEngine`, `MaskedIndexEngine` |
| `internals.pyx` | 1,026 | Cython-optimized BlockManager ops: `BlockPlacement`, `SharedBlock`, `NumpyBlock` |
| `join.pyx` | 880 | Join algorithms: `left_join_indexer`, `inner_join_indexer`, `outer_join_indexer`, `asof_join_backward/nearest` |
| `sparse.pyx` | 732 | Sparse array operations: `BlockIndex`, `IntIndex`, COO/CSR conversion |
| `interval.pyx` | 684 | `Interval`, `IntervalTree` implementation |
| `tslib.pyx` | 582 | Timestamp array operations: bulk datetime conversion |
| `missing.pyx` | 549 | Fast NA detection: `checknull`, `isnaobj`, `is_matching_na` |
| `ops.pyx` | 310 | Object-array comparison/logical operations with NA propagation |
| `hashtable.pyx` | 128 | Hash table implementations (templated): `Int64HashTable`, `Float64HashTable`, `StringHashTable`, `PyObjectHashTable` |

#### `_libs/tslibs/` -- Time Series Library (26,418 LOC pyx, 18 modules)

The most Cython-dense subpackage. Implements all datetime, timedelta, period, and offset operations at C speed.

**Top files by LOC and semantic weight:**

| Module | LOC (pyx) | Responsibility |
|--------|-----------|----------------|
| `offsets.pyx` | 7,730 | **Largest Cython file.** ALL date offset classes: `BusinessDay`, `MonthEnd`, `QuarterEnd`, `YearEnd`, `Week`, `Easter`, `CustomBusinessDay`, etc. Offset arithmetic engine. |
| `timestamps.pyx` | 3,613 | `Timestamp` class: nanosecond-resolution point in time. Subclasses `datetime.datetime`. Timezone handling, rounding, frequency snapping. |
| `period.pyx` | 3,214 | `Period` class: fiscal period arithmetic, frequency conversion, start/end time computation. |
| `timedeltas.pyx` | 2,762 | `Timedelta` class: nanosecond-resolution duration. Arithmetic, comparison, component extraction. |
| `nattype.pyx` | 1,908 | `NaT` singleton: implements all datetime/timedelta methods to propagate NaT. Must quack like both Timestamp and Timedelta. |
| `parsing.pyx` | 1,124 | Date string parsing with resolution detection. |
| `strptime.pyx` | 1,000 | Vectorized strptime for bulk date string parsing. |
| `fields.pyx` | 842 | Datetime field extraction (year/month/day/hour/minute/second) from int64 arrays. |
| `conversion.pyx` | 831 | Datetime conversion: `localize_pydatetime`, timezone-aware int64-to-datetime. |
| `tzconversion.pyx` | 839 | Timezone conversion handling DST transitions. |

#### `_libs/window/` -- Window Aggregation Extensions (2,269 LOC pyx)

Rolling/expanding window aggregation kernels including `roll_sum`, `roll_mean`, `roll_var`, `roll_skew`, `roll_kurt`, `roll_median_c`, `roll_min`/`roll_max`, `roll_quantile`, `roll_rank`, and EWM (exponentially weighted) variants. Both fixed and variable-length window support.

#### `_libs/` C Source (9,840 LOC) and Templates (3,251 LOC)

Vendored C code for the CSV tokenizer (2,255 LOC), NumPy datetime (1,995 LOC), and UltraJSON encoder/decoder (4,988 LOC). Cython template files (`hashtable_class_helper.pxi.in` at 1,572 LOC is the largest) generate typed specializations for hash tables, take operations, and sparse arithmetic.

### 3.3 Core Engine (Layer 2)

The `core/` package at 179,016 LOC is the heart of pandas, containing all runtime semantics for data manipulation.

#### 3.3.1 Top-Level Container Classes

| Module | LOC | Key Classes | Role |
|--------|-----|------------|------|
| `frame.py` | 18,679 | `DataFrame` | Central 2D labeled data structure. ~250 methods including constructors, indexing, merging, groupby access, I/O dispatch, arithmetic, aggregation, iteration, reshaping. |
| `generic.py` | 12,788 | `NDFrame` | Shared base for DataFrame and Series. `.loc`/`.iloc`, `.reindex()`, `.drop()`, `.rename()`, `.astype()`, `.copy()`, `.pipe()`, `.apply()`, metadata propagation, alignment logic. |
| `series.py` | 9,860 | `Series` | 1D labeled array. String/datetime/categorical accessor dispatch, arithmetic, comparison, aggregation. Inherits from NDFrame. |
| `indexing.py` | 3,384 | `_LocIndexer`, `_iLocIndexer`, `_AtIndexer`, `_iAtIndexer` | Complex label-based and position-based indexing with slice, boolean, and fancy indexing. |
| `apply.py` | 2,147 | `FrameApply`, `SeriesApply`, `GroupByApply` | `.apply()` and `.agg()` dispatch for user-defined functions. |

**Architectural note:** `generic.py` (12,788 LOC) is effectively a God Object. NDFrame contains alignment, indexing, metadata, aggregation, and many other concerns in a single class. The Rust port decomposes this into separate traits.

`frame.py` (18,679 LOC) is the single largest file. Many of its methods are I/O dispatch stubs (`.to_parquet()`, `.to_stata()`, etc.) that simply delegate to `io/` modules.

#### 3.3.2 `core/arrays/` -- Extension Array Types (30,774 LOC, 27 modules)

The "column engine" of pandas. Implements all array-level storage types.

**Class hierarchy:**

```
ExtensionArray (base.py, 3,042 LOC)  -- abstract base, 40+ method interface
  |
  +-- NDArrayBackedExtensionArray (_mixins.py, 631 LOC)
  |     |
  |     +-- DatetimeLikeArrayMixin (datetimelike.py, 2,757 LOC)
  |     |     +-- DatetimeArray (datetimes.py, 3,123 LOC)
  |     |     +-- TimedeltaArray (timedeltas.py, 1,310 LOC)
  |     |     +-- PeriodArray (period.py, 1,493 LOC)
  |     |
  |     +-- Categorical (categorical.py, 3,194 LOC)
  |
  +-- BaseMaskedArray (masked.py, 2,011 LOC) -- two-array: data + mask
  |     +-- BooleanArray (boolean.py, 438 LOC)
  |     +-- IntegerArray (integer.py, 296 LOC)
  |     +-- FloatingArray (floating.py, 192 LOC)
  |
  +-- IntervalArray (interval.py, 1,889 LOC)
  +-- SparseArray (sparse/array.py, 1,993 LOC)
  +-- StringArray (string_.py, 1,232 LOC)
  +-- ArrowExtensionArray (arrow/array.py, 3,417 LOC) -- PyArrow bridge
  +-- NumpyExtensionArray (numpy_.py, 652 LOC) -- ndarray wrapper
```

The `ExtensionArray` ABC defines 11 required abstract methods (`_from_sequence`, `__getitem__`, `__len__`, `__eq__`, `dtype`, `nbytes`, `isna`, `take`, `copy`, `_concat_same_type`, `interpolate`) and ~15 performance-critical optional methods (`fillna`, `unique`, `factorize`, `_reduce`, `_accumulate`, `searchsorted`, etc.).

#### 3.3.3 `core/dtypes/` -- Type System (9,102 LOC, 10 modules)

The "type algebra" of pandas. Detection, casting, missing value logic, and ExtensionDtype definitions.

**Key modules:**

| Module | LOC | Responsibility |
|--------|-----|----------------|
| `common.py` | 1,988 | ~80 public type-checking predicates: `is_integer_dtype`, `is_float_dtype`, `is_bool_dtype`, `is_datetime64_dtype`, `is_categorical_dtype`, `is_extension_array_dtype`, `pandas_dtype()`, etc. |
| `cast.py` | 1,879 | Casting workhorse: `maybe_downcast_to_dtype`, `maybe_upcast`, `infer_dtype_from_scalar`, `maybe_convert_objects`, `convert_dtypes`, `find_common_type`. |
| `dtypes.py` | 2,480 | Concrete ExtensionDtype implementations: `CategoricalDtype`, `DatetimeTZDtype`, `PeriodDtype`, `IntervalDtype`, `SparseDtype`, `ArrowDtype`, `NumpyEADtype`. |
| `missing.py` | 739 | `isna`, `notna`, `array_equivalent`, `is_valid_na_for_dtype`, `na_value_for_dtype`. Central NA detection. |
| `base.py` | 597 | `ExtensionDtype` ABC: `name`, `type`, `na_value`, `construct_array_type()`, dtype registry. |
| `inference.py` | 518 | Type inference: `is_list_like`, `is_dict_like`, `is_hashable`, `is_number`, `is_scalar`, etc. |
| `concat.py` | 342 | `find_common_type`, `concat_compat`: dtype negotiation for `pd.concat()`. |

#### 3.3.4 `core/indexes/` -- Index Types (21,816 LOC, 14 modules)

All index (axis label) implementations. Indexes provide O(1) label-based lookup via Cython hash-table engines and alignment operations.

**Class hierarchy:**

```
Index (base.py, 8,082 LOC) -- O(1) lookup via _engine hash table
  |
  +-- RangeIndex (range.py, 1,584 LOC) -- memory-efficient start/stop/step
  +-- MultiIndex (multi.py, 4,807 LOC) -- hierarchical codes + levels
  +-- DatetimeIndex (datetimes.py, 1,618 LOC) -> DatetimeTimedeltaMixin
  +-- TimedeltaIndex (timedeltas.py, 417 LOC) -> DatetimeTimedeltaMixin
  +-- PeriodIndex (period.py, 793 LOC) -> DatetimeIndexOpsMixin
  +-- CategoricalIndex (category.py, 552 LOC)
  +-- IntervalIndex (interval.py, 1,470 LOC)
```

The base `Index` class provides the foundation for all alignment. Key methods: `get_loc` (O(1) single-label lookup), `get_indexer` (many-label indexer computation), `reindex` (produces indexer for aligning to a new label set), `union`/`intersection`/`difference` (set operations), `join` (the core alignment primitive).

`MultiIndex` (4,807 LOC) is the most complex subclass, storing `levels` (FrozenList of Index objects) and `codes` (FrozenList of integer code arrays). It supports partial key matching via `get_loc` and cross-level operations.

#### 3.3.5 `core/internals/` -- Block Manager (6,874 LOC, 6 modules)

The storage layer managing the internal columnar blocks of a DataFrame.

| Module | LOC | Key Classes |
|--------|-----|------------|
| `managers.py` | 2,548 | `BlockManager`, `SingleBlockManager`: the core data containers |
| `blocks.py` | 2,395 | `Block`, `NumpyBlock`, `ExtensionBlock`: storage units with dtype-homogeneous arrays |
| `construction.py` | 1,055 | `arrays_to_mgr`, `dict_to_mgr`, `ndarray_to_mgr`: input conversion to BlockManager |
| `concat.py` | 479 | `concatenate_managers`: BlockManager concatenation for `pd.concat()` |
| `api.py` | 177 | Pseudo-public API for downstream libraries |
| `ops.py` | 155 | `operate_blockwise`: block-level binary operations |

**Block hierarchy:** As of pandas 2.x, blocks are either `NumpyBlock` (holds `np.ndarray` values, consolidatable) or `ExtensionBlock` (holds 1D `ExtensionArray`, not consolidatable). Older specialized block types (`IntBlock`, `FloatBlock`, `ObjectBlock`, `DatetimeLikeBlock`) have been consolidated into this two-class model.

#### 3.3.6 `core/groupby/` -- GroupBy (13,045 LOC, 9 modules)

Split-apply-combine framework. One of pandas' most-used features.

| Module | LOC | Key Classes |
|--------|-----|------------|
| `groupby.py` | 6,036 | `GroupBy` base class: core split-apply-combine engine with `_cython_agg_general`, `_agg_py_fallback` dispatch |
| `generic.py` | 3,987 | `DataFrameGroupBy`, `SeriesGroupBy`: concrete subclasses with column-specific methods |
| `ops.py` | 1,281 | `WrappedCythonOp`, `BaseGrouper`, `BinGrouper`: low-level execution engine mapping ops to Cython kernels |
| `grouper.py` | 961 | `Grouper`, `Grouping`: constructs grouping metadata from user input |
| `indexing.py` | 378 | `GroupByIndexingMixin`: `.iloc` positional indexing within groups |
| `numba_.py` | 183 | Numba JIT compilation for custom aggregation functions |
| `categorical.py` | 83 | Categorical code remapping for groupby operations |

#### 3.3.7 `core/reshape/` -- Reshaping (8,555 LOC, 8 modules)

Data restructuring: merge, concat, pivot, melt, stack/unstack, get_dummies, cut/qcut.

| Module | LOC | Key Function |
|--------|-----|-------------|
| `merge.py` | 3,135 | `pd.merge()`: the join engine with `_MergeOperation`, `_AsofMerge`, `_OrderedMerge` |
| `pivot.py` | 1,310 | `pivot_table()`, `pivot()`, `crosstab()` |
| `reshape.py` | 1,124 | `stack()`/`unstack()` core logic |
| `concat.py` | 998 | `pd.concat()`: `Concatenator` class with index alignment and block concatenation |
| `tile.py` | 682 | `pd.cut()`/`pd.qcut()`: discretization/binning |
| `melt.py` | 676 | `melt()`/`wide_to_long()`/`lreshape()` |
| `encoding.py` | 589 | `get_dummies()`/`from_dummies()` |

#### 3.3.8 `core/window/` -- Window Operations (6,771 LOC, 7 modules)

Rolling, expanding, and exponentially weighted (EWM) computations.

| Module | LOC | Key Class |
|--------|-----|----------|
| `rolling.py` | 3,516 | `Rolling`, `Window`: fixed/variable window ops with ~20 aggregation methods |
| `expanding.py` | 1,412 | `Expanding`: cumulative window |
| `ewm.py` | 1,175 | `ExponentialMovingWindow`: span/halflife/alpha weighted computations |
| `numba_.py` | 357 | Numba JIT compilation for custom window aggregation |
| `common.py` | 171 | Shared utilities: `flex_binary_moment`, `zsqrt` |
| `online.py` | 117 | `OnlineExponentialMovingWindow`: streaming EWM |

#### 3.3.9 `core/ops/` -- Operator Dispatch (1,347 LOC, 6 modules)

Routes arithmetic, comparison, and logical operations based on array types.

| Module | LOC | Key Function |
|--------|-----|-------------|
| `array_ops.py` | 620 | `arithmetic_op`, `comparison_op`, `logical_op`: core binary operation dispatch |
| `mask_ops.py` | 194 | `kleene_or`, `kleene_and`, `kleene_xor`: three-valued logic for nullable booleans |
| `missing.py` | 176 | `dispatch_fill_zeros`: division-by-zero semantics |
| `common.py` | 157 | `unpack_zerodim_and_defer`: dunder method wrapper |
| `dispatch.py` | 31 | `should_extension_dispatch`: ExtensionArray vs numpy dispatch |
| `invalid.py` | 77 | `invalid_comparison`: error generation for impossible comparisons |

#### 3.3.10 Other Core Subsystems

| Subsystem | LOC | Key Responsibility |
|-----------|-----|--------------------|
| `core/strings/` | 5,433 | `.str` accessor with 50+ vectorized string operations |
| `core/computation/` | 3,877 | `pd.eval()`/`DataFrame.query()` expression engine (security-critical) |
| `core/resample.py` | 3,188 | Time-based resampling: `Resampler`, `DatetimeIndexResampler` |
| `core/tools/` | 1,939 | `pd.to_datetime()`, `pd.to_numeric()`, `pd.to_timedelta()` |
| `core/interchange/` | 1,991 | DataFrame interchange protocol (PEP-proposed cross-library exchange) |
| `core/_numba/` | 1,770 | JIT compilation infrastructure |
| `core/algorithms.py` | 1,712 | `unique`, `factorize`, `duplicated`, `isin`, `mode`, `rank`, `safe_sort`, `diff` |
| `core/nanops.py` | 1,777 | NA-aware aggregation: `nansum`, `nanmean`, `nanstd`, `nanvar`, `nanmin`, `nanmax`, `nanskew`, `nankurt`, `nanprod`, `nansem`, `nanmedian` |
| `core/array_algos/` | 1,509 | Array algorithm dispatch: `take`, `putmask`, `quantile`, `replace` |
| `core/indexers/` | 1,317 | Window indexer objects and indexing validation utilities |
| `core/missing.py` | 1,103 | Fill/interpolation logic: `interpolate_2d`, `clean_fill_method` |
| `core/methods/` | 964 | Extracted methods: `describe()`, `nlargest()`/`nsmallest()`, `to_dict()` |
| `core/sorting.py` | 736 | Sorting utilities: `nargsort`, `get_group_index`, `lexsort_indexer` |

### 3.4 I/O Layer (Layer 3)

The `io/` package at 47,386 LOC handles all data serialization and deserialization.

#### 3.4.1 `io/parsers/` -- CSV/Text Parsers (5,675 LOC)

Three parsing engines with automatic fallback:

| Engine | Module | LOC | Speed | Feature Coverage |
|--------|--------|-----|-------|-----------------|
| C parser | `c_parser_wrapper.py` | 395 | Fastest | Most common CSV patterns |
| Python parser | `python_parser.py` | 1,557 | Slowest | Full feature coverage (regex sep, multi-char sep, skipfooter) |
| PyArrow parser | `arrow_parser_wrapper.py` | 328 | Fast | Columnar, no chunking |

The `readers.py` (2,389 LOC) module contains `read_csv()`/`read_table()` with 100+ parameters and `TextFileReader` orchestration.

#### 3.4.2 `io/formats/` -- Output Formatting (14,213 LOC)

Controls how pandas objects are displayed and serialized to text.

| Module | LOC | Key Responsibility |
|--------|-----|--------------------|
| `style.py` | 4,536 | `Styler`: CSS-based DataFrame styling, conditional formatting, color gradients |
| `style_render.py` | 2,681 | Rendering backend for Styler (HTML/LaTeX generation) |
| `format.py` | 2,076 | Core formatting engine: `DataFrameFormatter`, `SeriesFormatter`, `GenericArrayFormatter` |
| `info.py` | 848 | `.info()` implementation with memory usage calculation |
| `html.py` | 657 | HTML table rendering (including Jupyter display) |
| `printing.py` | 597 | String utilities used by core (architecturally misplaced) |
| `xml.py` | 566 | DataFrame-to-XML serialization |
| `excel.py` | 1,023 | Cell-by-cell Excel format conversion |

**Architectural note:** `io/formats/printing.py` is imported by 16+ `core/` modules for repr/display but lives in the I/O layer. This is the most prominent layering violation by import count.

#### 3.4.3 Other I/O Subsystems

| Subsystem | LOC | Format |
|-----------|-----|--------|
| `io/pytables.py` | 5,595 | HDF5 via PyTables (most complex single I/O module) |
| `io/excel/` | 4,183 | Excel (.xlsx, .xls, .xlsb, .ods) via openpyxl/xlrd/calamine |
| `io/stata.py` | 3,934 | Stata .dta binary format (v104-119) |
| `io/sql.py` | 2,960 | SQL via SQLAlchemy/ADBC |
| `io/json/` | 2,541 | JSON with multiple orient modes + `json_normalize()` |
| `io/common.py` | 1,327 | Shared infrastructure: file handle management, URL detection, compression |
| `io/html.py` | 1,245 | HTML table scraping |
| `io/xml.py` | 1,155 | XML parsing via lxml/etree |
| `io/sas/` | 1,749 | SAS7BDAT and XPORT format readers |
| `io/parquet.py` | 680 | Apache Parquet via PyArrow/fastparquet |
| `io/orc.py` | 243 | Apache ORC format |
| `io/pickle.py` | 239 | Python pickle with compression |
| `io/feather_format.py` | 181 | Apache Feather (Arrow IPC) |

### 3.5 Visualization Layer (Layer 4)

The `plotting/` package (9,470 LOC) provides the `.plot` accessor and matplotlib backend. `_core.py` (2,255 LOC) implements the `PlotAccessor` class with backend registration. The `_matplotlib/` subpackage (5,845 LOC) contains `MPLPlot`, `LinePlot`, `BarPlot`, `ScatterPlot`, `HexBinPlot`, and pandas-specific matplotlib unit converters for datetime/timedelta/period types.

**V1 exclusion:** Plotting is excluded from FrankenPandas V1 scope.

### 3.6 Public API Surface (Layer 5)

The `api/` package (375 LOC) aggregates sub-namespaces: `api.types` (type checking), `api.indexers` (window indexer classes), `api.extensions` (ExtensionDtype/ExtensionArray registration), `api.interchange` (DataFrame interchange protocol), `api.typing` (type aliases), `api.executors` (ThreadPoolExecutor for CSV).

Also includes `api.internals.py` (62 LOC): a pseudo-public API for downstream libraries like Dask and Modin that need access to BlockManager internals.

---

## 4. Architecture Decisions

### 4.1 BlockManager Design

The BlockManager is the central storage abstraction separating logical column layout from physical memory layout.

**Structure:**

```
BlockManager
  +-- axes[0]: Index          # Column labels
  +-- axes[1]: Index          # Row index
  +-- blocks: tuple[Block]    # Homogeneous-dtype data blocks
  |     +-- values: ndarray | ExtensionArray    # 2D for numpy, 1D for EA
  |     +-- _mgr_locs: BlockPlacement           # Which columns this block owns
  |     +-- refs: BlockValuesRefs               # CoW reference tracking
  +-- _blknos[i] -> block number for column i
  +-- _blklocs[i] -> within-block position for column i
```

**Design rationale:** By grouping same-dtype columns into contiguous 2D numpy arrays, the BlockManager enables:
1. SIMD-friendly memory layout for homogeneous operations (e.g., all float64 columns in one block).
2. Reduced metadata overhead compared to per-column storage.
3. Efficient block-wise operations via `BlockManager.apply(func)`.

**Trade-offs:**
- Column insertion requires block splitting and `_blknos`/`_blklocs` rebuilding.
- Mixed-dtype DataFrames produce many small blocks (fragmentation).
- ExtensionArray blocks cannot consolidate with numpy blocks.
- The `PerformanceWarning` at 100+ non-extension blocks signals fragmentation.

**SingleBlockManager:** A simplified version for Series with exactly one block and one axis. Always consolidated. Exposes convenience properties: `_block`, `array`, `index`, `dtype`.

### 4.2 ExtensionArray System

The ExtensionArray system allows third-party dtypes to participate natively in pandas operations.

**Contract (from `core/arrays/base.py:112`):** An ExtensionArray must implement 11 abstract methods and optionally override ~15 performance-critical methods. The key contract requirements:
- `_from_sequence(scalars, dtype)` -- construct from iterable
- `take(indices, allow_fill, fill_value)` -- positional indexing with fill
- `isna()` -- boolean array of missing values
- `_concat_same_type(to_concat)` -- concatenation
- `_reduce(name)` -- aggregation dispatch

**Two-array nullable pattern:** `BaseMaskedArray` stores `_data` (values) + `_mask` (boolean NA indicator) separately, enabling nullable integer/float/boolean without the float-promotion hack.

**Arrow bridge:** `ArrowExtensionArray` (3,417 LOC) wraps a PyArrow `ChunkedArray` as a pandas ExtensionArray, providing the primary bridge between pandas and Arrow memory models.

### 4.3 Copy-on-Write (CoW)

Pandas 3.0 enforces Copy-on-Write throughout. Every `Block` tracks references via `BlockValuesRefs`, a weak-reference-based reference counter.

**Mechanism:**
1. `BlockValuesRefs` maintains `referenced_blocks` (weak references to blocks sharing the same data).
2. `has_reference()` returns `True` if any other block still references the data.
3. On mutation: if `has_reference() == True`, the block is copied first.
4. `add_references(mgr)` links refs between managers with identical block structure.

**Behavior by operation:**

| Operation | Creates Copy? | Mechanism |
|-----------|--------------|-----------|
| `df[col]` (column access) | No -- view | Shares `refs` with parent block |
| `df.copy(deep=False)` | No data copy | New BlockManager with shared `refs` |
| `df.copy(deep=True)` | Full copy | New blocks, new `refs`, consolidation triggered |
| `df[col] = new_values` | Copy if refs exist | `_has_no_reference()` check; splits if shared |
| `df.iloc[row, col] = val` | Copy column if refs | `column_setitem` copies block if CoW active |
| Slicing `df[1:5]` | No -- view | Blocks share refs with parent |
| Binary ops `df + df2` | Always new | New blocks, no ref sharing |

### 4.4 DType System and Type Algebra

The pandas type system has two parallel hierarchies:

**NumPy dtypes:** `float16/32/64`, `int8/16/32/64`, `uint8-64`, `bool_`, `object_`, `datetime64[ns]`, `timedelta64[ns]`, `complex64/128`, `str_`.

**ExtensionDtype hierarchy:**

```
ExtensionDtype (core/dtypes/base.py)
  +-- PandasExtensionDtype
  |     +-- CategoricalDtype (categories + ordered flag)
  |     +-- DatetimeTZDtype (datetime64[ns, tz])
  |     +-- PeriodDtype (period[freq])
  |     +-- IntervalDtype (interval[subtype, closed])
  |     +-- SparseDtype (Sparse[subtype, fill_value])
  +-- BaseMaskedDtype (nullable numeric/bool base)
  |     +-- Int8Dtype...Int64Dtype, UInt8Dtype...UInt64Dtype
  |     +-- Float32Dtype, Float64Dtype
  |     +-- BooleanDtype
  +-- StringDtype (object or arrow backed)
  +-- ArrowDtype (general pyarrow-backed)
  +-- NumpyEADtype (numpy dtype wrapper)
```

**Promotion rules (`core/dtypes/cast.py:find_common_type`):**

| Operation | Input Types | Result |
|-----------|-------------|--------|
| Arithmetic | int + float | float64 |
| Arithmetic | int + int | int64 (may promote to float64 if NaN introduced) |
| Arithmetic | bool + int | int64 |
| Division | any numeric | float64 (always) |
| Comparison | any + any | bool |
| Concatenation | int + float | float64 |
| Concatenation | int + object | object |
| Setting NaN | int column | float64 (numpy) or stays Int64 (nullable) |
| Mixed | numeric + string | object |
| Extension | Int64 + float | Float64 (nullable) |

### 4.5 Index Engine Architecture

Each Index object lazily constructs a typed hash-table engine (via `@cache_readonly`) for O(1) label lookups.

**Engine selection from `_masked_engines` dict based on dtype:**
- `Int64Engine`, `UInt64Engine`, `Float64Engine` for numeric indexes
- `ObjectEngine` for string/mixed-type indexes
- `DatetimeEngine`, `TimedeltaEngine`, `PeriodEngine` for temporal indexes
- `MaskedIndexEngine` for nullable-typed indexes

**Key operations on Index engines:**
- `get_loc(key)` -- returns integer position, slice, or boolean mask for single label
- `get_indexer(target)` -- returns integer array mapping target labels to source positions (-1 for missing)
- `get_indexer_non_unique(target)` -- same but handles duplicate labels in source

**RangeIndex optimization:** `RangeIndex` stores only `start/stop/step` without materializing an array. Optimized arithmetic and set operations avoid array allocation for many common patterns (the default index type for DataFrames).

---

## 5. State Machine Descriptions

### 5.1 Block Consolidation State Machine

Consolidation merges multiple blocks of the same dtype into a single block for better memory locality.

**State flags:**
- `_known_consolidated: bool` -- whether the consolidation state has been checked
- `_is_consolidated: bool` -- whether same-dtype blocks have been merged

**State transitions:**

```
                     +-------------------+
                     | UNKNOWN STATE     |
                     | _known = False    |
                     | _is = False       |
                     +-------------------+
                           |
                    _consolidate_check()
                           |
              +------------+-------------+
              |                          |
    +---------v--------+    +-----------v----------+
    | CONSOLIDATED     |    | FRAGMENTED           |
    | _known = True    |    | _known = True        |
    | _is = True       |    | _is = False          |
    +------------------+    +----------------------+
         |                        |
     mutation                  consolidate_inplace()
    (insert,iset)                 |
         |                  +-----v-----+
    +----v---------+        | CONSOLIDATED |
    | UNKNOWN STATE|        +-------------+
    +--------------+
```

**Consolidation triggers:**
1. `BlockManager.copy(deep=True)` -- consolidates after copy
2. `replace_list` -- consolidates after multi-value replacement
3. Manual `DataFrame._consolidate()` -- explicit user call

**Constraints:**
- Only non-extension blocks can consolidate (`_can_consolidate` is `False` for `ExtensionBlock`)
- `DatetimeTZDtype` blocks explicitly do not consolidate despite being 2D-capable
- `PerformanceWarning` raised at >100 non-extension blocks

### 5.2 Copy-on-Write Reference Tracking

CoW reference state is tracked per-block via `BlockValuesRefs`.

```
STATE: UNSHARED (has_reference() == False)
  |
  +-- Shallow copy / column access / slice
  |     adds weak reference to refs
  |
  v
STATE: SHARED (has_reference() == True)
  |
  +-- Mutation requested (setitem, putmask, where)
  |     |
  |     +-- Copy block data (lazy copy triggered)
  |     +-- New block with new refs (back to UNSHARED)
  |
  +-- All other references dropped (weak refs expire)
        |
        v
STATE: UNSHARED (has_reference() == False)
  |
  +-- Mutation is in-place (no copy needed)
```

### 5.3 Index Caching and Memoization

Indexes are immutable. The `_cache` dictionary stores memoized computed properties, never invalidated (because the underlying data never changes).

**Cached properties (computed once on first access):**
- `is_unique` -- whether all labels are distinct
- `is_monotonic_increasing` / `is_monotonic_decreasing`
- `_engine` -- the hash-table engine for O(1) lookups
- `inferred_type` -- inferred type string
- `hasnans` -- whether index contains NaN values

**Exception:** `Index.name` can be set in-place (it does not affect data), but `_no_setting_name` flag can prevent this.

Any "modification" returns a new Index object (e.g., `index.insert()`, `index.delete()`, `index.set_names()`), which starts with an empty cache.

### 5.4 DType Promotion State Transitions

When a value is assigned to a Block whose dtype cannot hold the value, a state transition occurs:

```
Block (dtype=int64)
  |
  +-- setitem with float value
  |     blk.should_store(value) returns False
  |     |
  |     +-- Block is SPLIT:
  |     |     _iset_split_block(loc)
  |     |     - Creates new single-column Block(dtype=float64) for that column
  |     |     - Remaining columns stay as int64 Block
  |     |
  |     +-- _blknos/_blklocs REBUILT
  |     +-- _known_consolidated = False
  |
  +-- setitem with NaN
  |     For numpy int64: promotes entire column to float64
  |     For nullable Int64: NaN stored via mask (no promotion)
  |
  +-- binary op with float column
        Result block has dtype=float64
        Original blocks unchanged (new blocks created)
```

### 5.5 Lazy Evaluation Points

pandas is generally eager (not lazy), but several key components use deferred computation:

1. **Index engine construction:** `_engine` property is `@cache_readonly` -- the hash table is not built until first `get_loc()` call.
2. **Index monotonicity:** `is_monotonic_increasing` is computed and cached on first access.
3. **GroupBy grouping:** `get_grouper()` constructs `BaseGrouper` with `codes`, `uniques`, and `ngroups` eagerly, but actual aggregation kernels run only when `.sum()`, `.mean()`, etc. are called.
4. **TextFileReader:** When `chunksize` or `iterator=True`, `read_csv()` returns a lazy `TextFileReader` object that reads chunks on demand.
5. **Styler:** Style computations are deferred until rendering (`.to_html()`, `._repr_html_()`).
6. **Expression evaluation:** `pd.eval()` parses the expression string into an AST, then evaluates lazily against the provided scope.

---

## 6. Boundary Narratives

### 6.1 Dependency Direction Rules

| Direction | Status | Count | Notes |
|-----------|--------|-------|-------|
| `core -> _libs` | ALLOWED | 192 | Core engine uses Cython extensions |
| `core -> errors` | ALLOWED | 55 | Core raises pandas exceptions |
| `core -> compat` | ALLOWED | 51 | Core uses compatibility shims |
| `core -> util` | ALLOWED | 121 | Core uses decorators/validators |
| `core -> _config` | ALLOWED | 31 | Core reads configuration |
| `io -> core` | ALLOWED | 92 | I/O constructs DataFrames |
| `io -> _libs` | ALLOWED | 41 | I/O uses Cython parsers |
| `plotting -> core` | ALLOWED | 22 | Plotting reads DataFrames |
| `tseries -> _libs` | ALLOWED | 11 | Time series uses Cython offsets |

### 6.2 Boundary Violations

#### VIOLATION 1: `_libs/` imports from `core/` (upward dependency)

The Cython extension layer reaches up into the Python core layer:

| Source | Target | Purpose |
|--------|--------|---------|
| `_libs/internals.pyx` | `core.internals.blocks`, `core.construction` | Block construction in Cython internals |
| `_libs/lib.pyx` | `core.dtypes.missing`, `core.dtypes.generic`, `core.dtypes.cast`, `core.arrays.*` | Type inference and array construction |
| `_libs/parsers.pyx` | `core.arrays`, `core.dtypes.dtypes`, `core.dtypes.inference` | Parser needs to construct typed arrays |
| `_libs/testing.pyx` | `core.dtypes.missing` | Array comparison utility |
| `_libs/tslibs/offsets.pyx` | `core.dtypes.cast` | Scalar unboxing |

**Severity:** Moderate. Mostly deferred (inside-function) imports to break import cycles.

**Rust port implication:** The Rust trait system cleanly separates interface (`fp-types` traits) from implementation (`fp-columnar`, `fp-frame`), eliminating this circular dependency.

#### VIOLATION 2: `core/` imports from `io/` (upward dependency)

The core engine reaches into the I/O layer, primarily for formatting:

| Source | Target | Purpose |
|--------|--------|---------|
| `core/frame.py` | `io.formats.*`, `io.stata`, `io.parquet`, `io.orc` | DataFrame serialization methods |
| `core/arrays/*.py` | `io.formats.printing`, `io.formats.format` | Array repr formatting |
| `core/indexes/*.py` | `io.formats.printing`, `io.formats.format` | Index repr formatting |
| `core/dtypes/cast.py` | `io._util._arrow_dtype_mapping` | Arrow dtype lookup |

**Severity:** Low-moderate. `io.formats.printing` (597 LOC) is arguably misplaced -- it is a utility used pervasively by core but lives in io.

**Rust port implication:** In FrankenPandas, formatting lives in `fp-types` (Display trait impls) and `fp-frame` (Display for DataFrame/Series), not in `fp-io`.

### 6.3 Import Count Matrix

| From \ To | core | _libs | util | errors | compat | _config | io | tseries |
|-----------|------|-------|------|--------|--------|---------|----|---------|
| **core/** | 879 | 192 | 121 | 55 | 51 | 31 | 22 | 8 |
| **io/** | 92 | 41 | 49 | 18 | 28 | 14 | 95 | 1 |
| **plotting/** | 22 | 7 | 9 | 1 | -- | 1 | 5 | 1 |
| **tseries/** | 4 | 11 | 1 | 1 | -- | -- | -- | 3 |
| **_testing/** | 10 | 4 | 1 | 2 | 4 | 2 | 2 | 1 |
| **compat/** | 2 | 3 | 5 | 1 | 3 | -- | -- | -- |
| **_config/** | -- | -- | 1 | -- | -- | 7 | -- | -- |
| **errors/** | -- | 1 | 1 | -- | -- | 1 | -- | -- |
| **api/** | 19 | 5 | -- | -- | -- | -- | 3 | -- |

### 6.4 Internal Cross-Dependencies Within `core/`

The `core/dtypes/` package is the most depended-on subsystem within core:

| Subpackage | Depended On By (import count) |
|------------|-------------------------------|
| `core/dtypes/` | arrays (73), indexes (43), reshape (27), internals (26), groupby (13), window (9) |
| `core/arrays/` | indexes (14), internals (7), reshape (9), groupby (6), tools (6) |
| `core/indexes/` | groupby (4), internals (3), reshape (7), window (2) |
| `core/_libs (via core)` | arrays (52), indexes (28), dtypes (21), tools (11), groupby (8) |

This makes `core/dtypes/` the "foundation within the foundation" -- any Rust reimplementation of the dtype system must be one of the earliest crates stabilized.

---

## 7. Cross-Subsystem Integration

### 7.1 DataFrame Construction Pipeline

Entry point: `pd.DataFrame(data)` at `core/frame.py:455`.

The constructor branches on `type(data)` into multiple paths, all converging on `NDFrame.__init__(mgr)`:

```
DataFrame.__init__(data, index, columns, dtype, copy)
    |
    +-- isinstance(data, dict)  -->  dict_to_mgr()  -->  arrays_to_mgr()
    |                                                     -->  create_block_manager_from_column_arrays()
    |
    +-- isinstance(data, ndarray/Series/Index/EA)  -->  ndarray_to_mgr()
    |                                                    -->  create_block_manager_from_blocks()
    |
    +-- isinstance(data, DataFrame)  -->  extract _mgr, shallow copy
    |
    +-- is_list_like(data)  -->  nested_data_to_arrays() or ndarray_to_mgr()
    |
    +-- scalar  -->  construct_2d_arraylike_from_scalar()  -->  ndarray_to_mgr()
    |
    +-- NDFrame.__init__(self, mgr)
```

**The dict path** (most common): `dict_to_mgr()` iterates over columns, resolves missing columns as NaN placeholders, infers index from arrays (if not provided), and calls `arrays_to_mgr()`. This groups arrays by dtype into homogeneous `Block` objects via `create_block_manager_from_column_arrays()`.

**The ndarray path:** `ndarray_to_mgr()` ensures 2D shape, validates `(len(index), len(columns))` match, transposes to column-major (for block storage), and creates blocks. Object-type inference via `lib.maybe_convert_objects` may detect embedded datetime data.

**Subsystems touched:** `core/frame.py` -> `core/internals/construction.py` -> `core/internals/managers.py` -> `core/internals/blocks.py` -> `core/dtypes/cast.py` (type inference) -> `core/indexes/api.py` (index creation).

### 7.2 Binary Operations and Alignment

Entry point: `df + df2` at `core/arraylike.py:101`.

**Full call chain:**

1. `OpsMixin.__add__` -> `@unpack_zerodim_and_defer` -> `self._arith_method(other, operator.add)`
2. `DataFrame._arith_method` checks `_should_reindex_frame_op` (do columns differ?)
3. If columns differ: `_arith_method_with_reindex` computes `cols = left.columns.intersection(right.columns)`, reindexes both, operates on intersection, then reindexes result to full union.
4. `_align_for_op` handles Series-vs-DataFrame alignment and axis broadcasting.
5. `_dispatch_frame_op` branches on operand type:
   - Scalar: `self._mgr.apply(array_op, right=right)` -- applies to every block
   - DataFrame: `self._mgr.operate_blockwise(right._mgr, array_op)` -- paired block ops
   - Series (axis=1): iterate columns, apply per-column
6. `ops.arithmetic_op` dispatches to the appropriate kernel based on array types.

**For Series arithmetic:** Index alignment happens via `self._align_for_op(other)` which calls `left.align(right)` producing union-aligned left and right, then `ops.arithmetic_op(lvalues, rvalues, op)`.

**Subsystems touched:** `core/arraylike.py` -> `core/frame.py` or `core/series.py` -> `core/generic.py` (alignment) -> `core/ops/array_ops.py` -> `core/internals/managers.py` (block-wise dispatch) -> `_libs/ops.pyx` (C-level operations for object dtype).

### 7.3 GroupBy into Ops Pipeline

Entry point: `df.groupby('key').sum()` at `core/frame.py:12360`.

**Phase 1: Group construction**
1. `DataFrame.groupby('key')` creates `DataFrameGroupBy` object.
2. `get_grouper(obj, keys, sort)` resolves user input into `Grouping` objects.
3. For column-name keys: extracts `obj[key]` as grouping values.
4. `BaseGrouper` computes `codes`, `uniques`, and `ngroups`.

**Phase 2: Aggregation dispatch**
1. `GroupBy.sum()` calls `_agg_general()` -> `_cython_agg_general(how="sum")`.
2. `_get_data_to_aggregate(numeric_only)` returns a BlockManager with relevant columns.
3. For each block: try `_grouper._cython_operation("aggregate", values, "sum")`.
4. This calls the Cython kernel `_libs.groupby.group_sum()` for numeric data.
5. If Cython raises `NotImplementedError` (ExtensionArray or object dtype): fall back to `_agg_py_fallback` which iterates per-group in Python.
6. `data.grouped_reduce(array_func)` applies the kernel to each block.
7. `_wrap_agged_manager` constructs the result DataFrame.

**Decision matrix:**

| Condition | Path | Performance |
|-----------|------|-------------|
| Default, numeric dtype | Cython `_libs.groupby.group_sum` | Fastest (C-level loop) |
| `engine='numba'` | `_numba_agg_general` | JIT-compiled |
| Non-numeric/EA dtype | `_agg_py_fallback` | Python loop (orders of magnitude slower) |

**Subsystems touched:** `core/frame.py` -> `core/groupby/groupby.py` -> `core/groupby/grouper.py` -> `core/groupby/ops.py` -> `_libs/groupby.pyx` -> `core/internals/managers.py` (grouped_reduce) -> `core/nanops.py` (fallback).

### 7.4 Merge/Join Using Indexing and Sorting

Entry point: `pd.merge(left, right, on='key')` at `core/reshape/merge.py:146`.

**Phase 1: Key extraction**
1. `_MergeOperation.__init__` validates `how`, resolves `on` into `left_on`/`right_on`.
2. `_get_merge_keys()` extracts actual join key arrays from column names, indexes, or array-like inputs.
3. `_maybe_coerce_merge_keys()` ensures dtype compatibility between left and right keys.

**Phase 2: Join algorithm selection**
The critical decision point in `get_join_indexers()`:

| Condition | Algorithm | Complexity |
|-----------|-----------|-----------|
| Both keys sorted, at least one unique | Sort-merge via `Index.join()` | O(n + m) |
| Keys not sorted or both non-unique | Hash join via `_factorize_keys` + Cython `libjoin` | O(n + m) average |
| Multi-column keys | Factorize + flatten to composite key, then same decision | Extra O(n * k) for factorization |

**Phase 3: Result construction**
1. `join_index, left_indexer, right_indexer` define the output mapping.
2. Both sides are reindexed via `BlockManager.reindex_indexer()`.
3. Results concatenated along axis=1.
4. Join keys, indicator columns, and index levels restored.

**Subsystems touched:** `core/reshape/merge.py` -> `core/indexes/base.py` (Index.join) -> `_libs/join.pyx` (Cython join kernels) -> `_libs/hashtable.pyx` (factorization) -> `core/sorting.py` -> `core/internals/managers.py` (reindex_indexer) -> `core/reshape/concat.py`.

### 7.5 CSV Read Through Parser to DataFrame

Entry point: `pd.read_csv('file.csv')` at `io/parsers/readers.py:350`.

**Engine selection cascade:**
1. `_refine_defaults_read()` resolves engine: regex/multi-char sep forces 'python', `skipfooter > 0` forces 'python', default is 'c'.
2. `TextFileReader.__init__` creates engine wrapper via `_make_engine()`.
3. File handle opened via `get_handle()` (handles URL, compression, memory_map).

**Engine mapping:**
- `"c"` -> `CParserWrapper` -> `_libs.parsers.TextReader` (C tokenizer)
- `"python"` -> `PythonParser` (pure Python, full feature support)
- `"pyarrow"` -> `ArrowParserWrapper` (columnar Arrow-based parsing)

**Post-parsing construction:**
1. Engine returns `(index, columns, col_dict)` tuple.
2. User-specified dtypes applied per column.
3. `DataFrame(col_dict, columns=columns, index=index)` goes through `dict_to_mgr` path.

**Subsystems touched:** `io/parsers/readers.py` -> `io/parsers/c_parser_wrapper.py` or `python_parser.py` -> `_libs/parsers.pyx` (C parser) -> `io/common.py` (file handles) -> `core/internals/construction.py` (dict_to_mgr) -> `core/frame.py`.

### 7.6 loc/iloc Indexing Dispatch

Entry point: `df.loc[rows, cols]` at `core/indexing.py:1190`.

**loc dispatch tree:**

```
_LocIndexer.__getitem__(key)
    |
    +-- tuple(row, col):
    |     +-- _is_scalar_access? (all scalar, unique axes, no MultiIndex)
    |     |     YES -> obj._get_value(row, col)  [FAST PATH - direct scalar]
    |     +-- _getitem_tuple -> _getitem_lowerdim / _multi_take / _getitem_tuple_same_dim
    |
    +-- non-tuple:
          _getitem_axis(key, axis=0):
            +-- slice -> label-based INCLUSIVE slicing
            +-- bool indexer -> nonzero() + take
            +-- list-like -> _get_listlike_indexer -> reindex
            +-- scalar -> xs(label) [cross-section]
```

**iloc dispatch tree (analogous but positional):**
```
_iLocIndexer._getitem_axis(key, axis)
    +-- slice -> Python-convention EXCLUSIVE slicing
    +-- bool indexer -> same as loc
    +-- list-like -> take(key, axis)
    +-- scalar integer -> bounds check + _ixs(key, axis)
```

**Key behavioral difference:** loc slice `'a':'c'` is INCLUSIVE of both endpoints (label semantics). iloc slice `1:3` is EXCLUSIVE of stop (Python convention). This is one of the most frequently confused behaviors in pandas.

**Subsystems touched:** `core/indexing.py` -> `core/indexes/base.py` (get_loc, get_indexer, slice_indexer) -> `_libs/index.pyx` (engine lookup) -> `core/internals/managers.py` (iget, take) -> `core/frame.py` (DataFrame._get_value) or `core/series.py`.

---

## 8. Semantic Hotspots

### 8.1 Constructor Alignment and Copy/View

DataFrame construction from mixed inputs (dict of Series, list of dicts, ndarray) involves complex alignment:
- When input is a dict of Series with different indexes, `_extract_index(arrays)` computes the union index and reindexes each array.
- Copy semantics differ: EA types are copied eagerly; numpy arrays may share memory until consolidated.
- Under CoW, `DataFrame(other_df)` creates a shallow copy with shared block references.

### 8.2 Series Alignment During Arithmetic

Binary operations between Series with different indexes trigger automatic alignment via `align(other)`. The default is outer (union) join -- positions where one Series has a label but the other does not are filled with NaN. This alignment-aware computation is the core semantic that FrankenPandas' AACE (Alignment-Aware Columnar Execution) layer preserves.

### 8.3 Index Object Contracts

Index objects are immutable and hashable. Key contracts:
- `get_loc(key)` returns int, slice, or boolean mask (depending on duplicate labels).
- `get_indexer(target)` returns `np.ndarray[intp]` with -1 for missing labels.
- Duplicate labels are allowed in construction but `reindex` raises `InvalidIndexError` on duplicate source.
- Set operations (`union`, `intersection`) deduplicate implicitly.
- `_validate_can_reindex(indexer)` checks for -1 values when source has duplicates.

### 8.4 BlockManager Invariants

Six invariants must hold at all times:

1. **Index-Length == Rows:** `len(axes[1]) == block.shape[1]` for all blocks.
2. **Block shape alignment:** `block.shape[1:] == manager.shape[1:]` for every block.
3. **Total column coverage:** `sum(len(block.mgr_locs)) == len(items)`.
4. **Exclusive column ownership:** Each column position 0..n-1 owned by exactly one block.
5. **DType homogeneity within blocks:** All values in a block share one dtype.
6. **`_blknos`/`_blklocs` consistency:** These arrays accurately map column positions to block/within-block positions.

`BlockManager._verify_integrity()` checks invariants 1-4. The `_blknos`/`_blklocs` arrays are rebuilt after mutations via `_rebuild_blknos_and_blklocs()`.

### 8.5 loc/iloc Distinction

| Aspect | loc | iloc |
|--------|-----|------|
| Key type | Labels (any hashable) | Integer positions |
| Slice semantics | Inclusive of both endpoints | Exclusive of stop |
| Boolean mask | Aligned to index first | Must match exact length |
| Out-of-bounds | `KeyError` | `IndexError` |
| MultiIndex | Partial indexing supported | No partial indexing |
| Fast path | `_is_scalar_access`: scalar keys, unique axes, no MultiIndex | `_is_scalar_access`: scalar integers |

### 8.6 GroupBy/Window Defaults

**GroupBy critical defaults:**
- `sort=True`: groups appear in sorted order (not first-occurrence order)
- `as_index=True`: grouping columns become the result index
- `dropna=True`: groups with NA keys are dropped
- `observed=True` (pandas 2.2+): only observed categories for Categorical groupers

**Window critical defaults:**
- `min_periods=None` (defaults to window size for rolling, 1 for expanding)
- `center=False`: window is right-aligned
- `closed='right'` for fixed windows, configurable for variable

### 8.7 Operator Dispatch and DType Promotion

The `core/ops/array_ops.py` module implements the dispatch chain:

1. `arithmetic_op(left, right, op)` -> checks if either side is ExtensionArray
2. If extension: delegates to `ExtensionArray._arith_method` -> EA-specific implementation
3. If numpy: `_na_arithmetic_op(left, right, op)` with NA propagation
4. Result dtype determined by `find_common_type` from `core/dtypes/cast.py`

Nullable boolean uses three-valued (Kleene) logic via `core/ops/mask_ops.py`:
- `True & NA == NA`
- `False & NA == False`
- `True | NA == True`
- `False | NA == NA`

---

## 9. Compatibility-Critical Behaviors

These behaviors must be preserved in any reimplementation for user-code compatibility:

1. **Assignment and reindex interactions:** `df[col] = series` aligns the series to the DataFrame's index before assignment. Missing labels in the series produce NaN in the DataFrame.

2. **Chained-assignment warning/error:** In pandas 3.0 with CoW, `df['col'][0] = value` raises `ChainedAssignmentError`. The old `SettingWithCopyWarning` is replaced by hard errors.

3. **Concat/merge around missing labels:** `pd.concat` with `join='outer'` (default) produces NaN for labels missing in some inputs. `pd.merge` produces cross-product rows for duplicate keys.

4. **ExtensionArray interoperability:** ExtensionArrays must participate in constructors, arithmetic, and aggregation pipelines. The `_from_sequence` / `take` / `_reduce` / `_concat_same_type` contract must be honored.

5. **NA propagation per dtype:** Float uses IEEE 754 NaN propagation. Nullable integer uses `pd.NA` with Kleene logic. Object dtype treats `None` and `np.nan` as missing.

6. **Sort stability:** All sort operations use stable sort (mergesort/Timsort). Group ordering and multi-key sort ties are deterministic.

7. **Index immutability:** Index objects never mutate after construction. Methods like `insert()`, `delete()`, `rename()` return new Index objects.

---

## 10. Security and Stability Risk Areas

1. **Expression evaluation** (`core/computation/`, 3,877 LOC): `pd.eval()` and `DataFrame.query()` execute string-based expressions. The `Scope` class resolves variable names from caller frames. This is an injection risk surface that FrankenPandas' `fp-expr` must carefully bound.

2. **SQL adapters** (`io/sql.py`, 2,960 LOC): Parameterization and injection risk. SQL query construction must use parameterized queries.

3. **Pickle** (`io/pickle.py`): Python pickle deserialization can execute arbitrary code. FrankenPandas deliberately does not implement pickle support.

4. **CSV parsing resource exhaustion** (`_libs/parsers.pyx` + `io/parsers/`): Malformed quoting, extremely long lines, and exponential-blowup patterns can cause memory exhaustion. The C parser has fixed-size buffers; the Python parser does not.

5. **Heavy third-party IO dependencies:** HDF5 (PyTables), Excel (openpyxl/xlrd), Parquet (pyarrow), SQL (SQLAlchemy) each bring their own security surface. FrankenPandas uses only the Rust `csv` and `serde_json` crates for V1.

6. **UltraJSON vendored C code** (4,988 LOC): Vendored ujson for JSON parsing/writing. Potential buffer overflow surface in C code. FrankenPandas uses safe Rust `serde_json` instead.

---

## 11. V1 Extraction Boundary

### Include in V1 (FrankenPandas current scope)

- **Core containers:** DataFrame, Series construction and basic operations
- **Index model:** Flat Index with alignment (union, intersection, inner, left, right, outer)
- **Column storage:** DType-homogeneous columns with validity mask
- **Arithmetic/comparison:** Binary operations with alignment and NA propagation
- **Aggregation:** sum, mean, median, min, max, std, var, count
- **GroupBy:** Split-apply-combine for core aggregations
- **Join/Merge:** Inner, left, right, outer joins on single-column keys
- **Concat:** Series and DataFrame concatenation
- **IO:** CSV and JSON read/write
- **Expression engine:** Basic expression evaluation with IVM delta propagation

### Exclude from V1

- **MultiIndex** and hierarchical operations
- **Datetime/Timedelta/Period** types and temporal indexes
- **Categorical** type and categorical index
- **Window functions** (rolling, expanding, EWM)
- **String accessor** methods
- **Resample** (time-based groupby)
- **Apply/map** with user-defined Python functions
- **Full IO ecosystem** (Excel, Parquet, SQL, HDF5, Stata, SAS, HTML, XML)
- **Plotting**
- **Styler/formatting** beyond basic Display
- **Interchange protocol**
- **Sorting** (beyond index sort_values)
- **Configuration** system

### V1 API Coverage

FrankenPandas V1 covers approximately **47% of CORE-tier** pandas APIs (daily-use functions), **11% of COMMON-tier**, and **2% of NICHE-tier**, for an overall **12% of the total ~545 public symbols**.

---

## 12. Conformance Fixture Families

### Test Domain Scale

| Test Domain | Files | Test Domain | Files |
|-------------|-------|-------------|-------|
| `tests/indexes` | 191 | `tests/io` (all) | 137 |
| `tests/frame` | 119 | `tests/series` | 105 |
| `tests/arrays` | 101 | `tests/extension` | 48 |
| `tests/groupby` | 45 | `tests/scalar` | 39 |
| `tests/tseries` | 34 | `tests/indexing` | 32 |
| `tests/reshape` | 31 | `tests/util` | 25 |
| `tests/window` | 24 | `tests/copy_view` | 23 |
| `tests/plotting` | 20 | `tests/dtypes` | 19 |

### Priority Fixture Families for FrankenPandas

1. **tests/frame:** Constructor, assign, setitem, getitem, indexing, arithmetic, aggregation behavior. 119 files covering the DataFrame contract.

2. **tests/series and tests/indexes:** Scalar/index/dtype contracts. 105 + 191 files. The indexes tests are the largest family due to the breadth of the index hierarchy.

3. **tests/indexing:** loc/iloc/boolean edge semantics. 32 files with intensive edge-case coverage.

4. **tests/groupby:** Aggregation, transform, filter, and method behavior. 45 files.

5. **tests/reshape:** Merge/concat/pivot semantics. 31 files.

6. **tests/arithmetic:** Binary-op parity, dtype promotion, NA propagation. 13 files.

7. **tests/arrays and tests/extension:** Extension storage and API contract conformance. 149 files combined.

8. **tests/copy_view:** CoW behavior verification. 23 files (not applicable to FrankenPandas but useful for understanding behavioral expectations).

---

## 13. FrankenPandas Structural Divergences

### D1: No Block Manager -- Per-Column Storage

**pandas:** DataFrame uses BlockManager grouping same-dtype columns into 2D blocks with `_blknos`/`_blklocs` indirection.

**FrankenPandas:** DataFrame uses `BTreeMap<String, Column>` where each column is independent with its own `Vec<Scalar>`, `ValidityMask`, and `DType`. No block concept, no consolidation, no cross-column data sharing.

**Implications:** No consolidation overhead or fragmentation warnings. No cross-column SIMD from shared memory. Simpler mutation model. Column access is O(log n) via BTreeMap vs O(1) via ndarray.

### D2: No Copy-on-Write -- Rust Ownership

**pandas:** CoW with `BlockValuesRefs` weak references, lazy copies on mutation.

**FrankenPandas:** Rust ownership and borrowing eliminates CoW. Data is owned, borrowed, or explicitly cloned. No view/copy semantics. All operations return new values.

**Implications:** No reference-counting overhead. No view-vs-copy bugs. Higher memory for operations that would be views in pandas.

### D3: Scalar-Level Typing vs Array-Level Typing

**pandas:** Each block has a single dtype. Homogeneous numpy arrays or ExtensionArrays.

**FrankenPandas:** `Column` stores `Vec<Scalar>` where each `Scalar` is a tagged enum. `ColumnData` enum provides vectorized fast path. DType is inferred from the collection.

**Implications:** Each scalar carries 1-byte enum discriminant overhead. Mixed-type columns impossible by design. Type coercion per-scalar via `cast_scalar_owned`.

### D4: Flat Index Only -- No Hierarchy

**pandas:** Full Index hierarchy: MultiIndex, DatetimeIndex, PeriodIndex, CategoricalIndex, IntervalIndex.

**FrankenPandas:** Single `Index` struct with `Vec<IndexLabel>` where `IndexLabel` is `Int64(i64) | Utf8(String)`. Sorted indexes get binary search, unsorted get HashMap fallback.

### D5: Explicit NA Discrimination

**pandas:** NA handling is dtype-dependent: `np.nan` for float, `NaT` for datetime, `pd.NA` for nullable extension types.

**FrankenPandas:** `NullKind` enum distinguishes `Null`, `NaN`, `NaT`. Integer columns CAN hold nulls without float promotion. NaN-vs-Null distinction preserved for round-tripping.

### D6: BTreeMap Column Ordering vs Positional

**pandas:** Columns maintain insertion order. Positional access via integer index is O(1).

**FrankenPandas:** `BTreeMap<String, Column>` sorts columns alphabetically. No positional column access. `from_dict` accepts `column_order` but BTreeMap still sorts.

### D7: Immutable-by-Default Operations

**pandas:** Many operations support `inplace=True`. BlockManager supports in-place and CoW mutation.

**FrankenPandas:** All operations return new values. `with_column`, `drop_column`, `rename_columns`, `filter_rows` all return new `DataFrame` instances. Natural in Rust's ownership model.

### D8: Runtime Policy Layer (FrankenPandas-specific)

**pandas:** No equivalent. Operations always proceed.

**FrankenPandas:** `RuntimePolicy` (Strict/Hardened/Permissive) gates operations. Strict mode rejects duplicate-label alignment. Hardened mode enforces size bounds. `EvidenceLedger` records all decisions for audit. This is the AACE layer.

### Invariant Mapping

| pandas Invariant | FrankenPandas Equivalent |
|-----------------|------------------------|
| `len(index) == nrows` for all blocks | `index.len() == column.len()` for all columns |
| DType homogeneity within block | DType homogeneity within Column |
| Index uniqueness for reindex | `has_duplicates()` with OnceCell memoization |
| NA propagation per dtype | Unified via `Scalar::is_missing()` |
| Sort stability (mergesort) | Rust `sort_by` (TimSort, stable) |
| CoW prevents unintended mutation | Rust ownership (compile-time guarantee) |
| Block `_can_hold_na` | All dtypes hold NA via `Scalar::Null(_)` |

---

## 14. Extraction Notes for Rust Spec

### Priority Observations

1. **Treat `core/internals` + `core/indexing` + `core/indexes` as the first-order compatibility contract.** These subsystems define the observable behavior that users depend on. Preserve behavioral semantics before replacing internal storage strategy.

2. **The `io.formats.printing` problem must be resolved.** This 597-LOC module is imported by 16+ core modules but lives in io. In FrankenPandas, formatting should live in `fp-types` (Display traits) or a shared `fp-format` crate.

3. **`core/generic.py` (12,788 LOC) must be decomposed.** NDFrame is a God Object containing alignment, indexing, metadata, aggregation, and more. The Rust port should use separate traits: alignment (AACE), indexing, metadata management, aggregation dispatch.

4. **`core/frame.py` (18,679 LOC) I/O dispatch stubs belong in `fp-io`.** Many DataFrame methods (`.to_parquet()`, `.to_stata()`, `.to_excel()`, etc.) simply delegate to io modules. In Rust, implement these as trait impls in the io crate, not methods on DataFrame.

5. **The dtype system (`core/dtypes/`, 9,102 LOC) is central to everything.** FrankenPandas' `DType` enum approach is correct but must expand to cover all casting/coercion logic in `cast.py` (1,879 LOC) and `common.py` (1,988 LOC). This is the highest-priority expansion target.

6. **`_libs -> core` circular dependency must not be replicated.** The Rust trait system cleanly separates interface from implementation. `fp-types` defines traits; `fp-columnar` and `fp-frame` implement them.

7. **Build parity reports per feature family before performance work.** Correctness first, then optimize. The conformance test harness in `fp-conformance` should map to the priority fixture families identified in Section 12.

### FrankenPandas Crate Mapping

| pandas Layer | FP Crate(s) | Status |
|-------------|-------------|--------|
| `_libs/` algorithms | `fp-index` (hash engines), `fp-columnar` | Partial |
| `_libs/` hashtable | `fp-index` | Partial |
| `_libs/tslibs/` | Not yet started | Planned |
| `core/dtypes/` | `fp-types` (DType enum) | Partial |
| `core/arrays/` | `fp-columnar` (Column) | Partial |
| `core/indexes/` | `fp-index` (Index) | Partial |
| `core/internals/` | `fp-frame` (no BlockManager) | Partial |
| `core/frame.py` | `fp-frame` (DataFrame) | Partial |
| `core/series.py` | `fp-frame` (Series) | Partial |
| `core/groupby/` | `fp-groupby` | Partial |
| `core/reshape/merge.py` | `fp-join` | Partial |
| `core/ops/` | `fp-columnar` (binary ops) | Partial |
| `core/nanops.py` | `fp-types` (nanops) | Partial |
| `io/` | `fp-io` (CSV + JSON) | Partial |
| `core/computation/` | `fp-expr` | Partial |
| `errors/` | `fp-types` + domain crates (Error types) | Partial |

---

*Generated for FrankenPandas Phase-2C, bead bd-2gi.23.11 (DOC-PASS-10). Synthesized from DOC_GAP_MATRIX, DOC_MODULE_CARTOGRAPHY, DOC_API_CENSUS, DOC_DATA_MODEL_MAP, and DOC_EXECUTION_PATHS.*
