# DOC-PASS-05: Complexity, Performance, and Memory Characterization

**Bead:** bd-2gi.23.6
**Status:** Complete
**Date:** 2026-02-14
**Source Trees:** `legacy_pandas_code/pandas/pandas/` and `crates/fp-*/src/lib.rs`

---

## Table of Contents

1. [Complexity Taxonomy](#1-complexity-taxonomy)
2. [Memory Growth Model](#2-memory-growth-model)
3. [Hotspot Families](#3-hotspot-families)
4. [FrankenPandas Performance Characteristics](#4-frankenpandas-performance-characteristics)
5. [Optimization Opportunities](#5-optimization-opportunities)
6. [Performance Sentinels](#6-performance-sentinels)

---

## 1. Complexity Taxonomy

This section classifies all major pandas operations by their time and space complexity, annotated with the conditions that determine which complexity class applies. Variables: `n` = number of rows, `m` = number of columns, `g` = number of groups, `k` = number of join keys, `u` = union cardinality.

### 1.1 DataFrame Construction

| Operation | Best Case | Typical Case | Worst Case | Space | Conditions |
|-----------|-----------|-------------|------------|-------|------------|
| `DataFrame(dict)` | O(m) | O(n*m) | O(n*m) | O(n*m) | Best: all columns already numpy arrays of same dtype. Typical: column validation + consolidation. |
| `dict_to_mgr` | O(m) | O(n*m) | O(n*m) | O(n*m) | Iterates columns, validates lengths, calls `arrays_to_mgr`. File: `core/internals/construction.py:375`. |
| `ndarray_to_mgr` | O(n*m) | O(n*m) | O(n*m) | O(n*m) | Transpose + shape validation + dtype inference. File: `core/internals/construction.py:193`. |
| `arrays_to_mgr` | O(n*m) | O(n*m) | O(n*m) | O(n*m) | Homogenizes arrays, extracts index. File: `core/internals/construction.py:96`. |
| Block consolidation | O(m log m) | O(n*m) | O(n*m) | O(n*m) | Sorts blocks by dtype key, merges homogeneous blocks via `np.concatenate`. File: `core/internals/managers.py:2441`. |
| `_extract_index` | O(n) | O(n * m_series) | O(n * m_series) | O(n) | When inputs are Series, inspects each for index. File: `core/internals/construction.py`. |

**Key insight:** Construction always requires O(n*m) work to validate and store data. The consolidation step groups columns by dtype into 2D blocks, paying an O(m log m) sort on dtype keys plus O(n*m) for `np.concatenate`. For homogeneous-dtype DataFrames, consolidation produces a single block (best case); for heterogeneous dtypes (e.g., mixed int/float/object), each dtype gets its own block.

### 1.2 Column Selection

| Operation | Best Case | Typical Case | Worst Case | Space | Conditions |
|-----------|-----------|-------------|------------|-------|------------|
| `df['col']` (unique columns) | O(1) | O(1) | O(m) | O(n) | Best: hash lookup in unique column index. Worst: `drop_duplicates` check when `is_unique` is False. Returns view (no copy). |
| `df[['col1', 'col2']]` | O(k) | O(k) + O(n*k) | O(n*k) | O(n*k) | k = number of selected columns. Column lookup via `_get_indexer_strict`, then `take` on axis=1. File: `core/frame.py:4198`. |
| `df[bool_mask]` | O(n) | O(n) | O(n) | O(n*m) | Validates mask length, computes `nonzero()`, then `take`. File: `core/frame.py:4231`. |
| `df.loc[row, col]` scalar | O(1) | O(log n) | O(n) | O(1) | `_is_scalar_access` fast path bypasses branching. Hash-based index engine: amortized O(1). Sorted index: O(log n). File: `core/indexing.py:1608`. |
| `df.loc[label_list]` | O(k) | O(k) | O(n) | O(n*k) | k = number of requested labels. Uses `get_indexer` which builds/consults hash table. |
| `df.iloc[int_list]` | O(k) | O(k) | O(n) | O(n*k) | Direct positional `take`, no hash table needed. |

**Key insight:** The single-column fast path (`df['col']` with unique columns) is O(1) via the `IndexEngine` hash table. The `is_unique` check is cached after first computation (`_libs/index.pyx`). Non-unique columns degrade to O(m) for `drop_duplicates`.

### 1.3 Index Operations

| Operation | Best Case | Typical Case | Worst Case | Space | Conditions |
|-----------|-----------|-------------|------------|-------|------------|
| `Index.get_loc(label)` | O(1) amortized | O(1) amortized | O(n) | O(n) | IndexEngine hash table is lazily built on first access. Sorted monotonic indexes use binary search O(log n). File: `_libs/index.pyx:1325`. |
| `Index.get_indexer(target)` | O(t) | O(t) | O(n + t) | O(n + t) | t = target length. Builds hash table from self (O(n) first time), then probes for each target label (O(t)). |
| `Index.join(other)` | O(n + m) | O(n + m) | O(n * m) | O(n + m) | Sorted + unique: merge in O(n + m). Non-unique: can produce cross-product O(n * m) output rows. File: `_libs/join.pyx`. |
| `Index.intersection(other)` | O(n + m) | O(n + m) | O(n + m) | O(min(n,m)) | Hash-based intersection. |
| `Index.union(other)` | O(n + m) | O(n + m) | O(n + m) | O(n + m) | Hash-based union with deduplication. |
| `Index.is_monotonic_increasing` | O(n) | O(n) | O(n) | O(1) | Cached after first computation. Full scan of values. File: `_libs/algos.pyx`. |
| `Index.has_duplicates` | O(n) | O(n) | O(n) | O(n) | Builds hash table to detect collisions. Cached. |
| `Index.factorize` | O(n) | O(n) | O(n) | O(n + u) | u = number of unique values. Uses `Int64HashTable` or `PyObjectHashTable`. File: `_libs/hashtable.pyx`. |

### 1.4 Binary Operations (Element-wise Arithmetic)

| Operation | Best Case | Typical Case | Worst Case | Space | Conditions |
|-----------|-----------|-------------|------------|-------|------------|
| `s1 + s2` (same index) | O(n) | O(n) | O(n) | O(n) | When indexes are identical (`is` check), no alignment needed. Direct element-wise numpy add. |
| `s1 + s2` (different index) | O(n + m) | O(n + m) | O(n + m) | O(n + m + u) | Alignment via `NDFrame.align()`: union index computation O(n + m), then reindex both sides O(u), then element-wise O(u). u = union size. |
| `df1 + df2` (block-wise) | O(n * m) | O(n * m) | O(n * m) | O(n * m) | `operate_blockwise`: paired block iteration. Same-structure DataFrames: fast block-level numpy operations. File: `core/frame.py:8975`. |
| `df1 + df2` (reindex) | O(n * m) | O(n * m) + O(reindex) | O(n * m * 2) | O(n * m * 2) | `_arith_method_with_reindex`: intersect columns, reindex both, operate, reindex to union. File: `core/frame.py:9065`. |
| `df + scalar` | O(n * m) | O(n * m) | O(n * m) | O(n * m) | `BlockManager.apply(array_op, right=scalar)`: applies op to each block. |

**Complexity bottleneck:** Index alignment is the hidden cost. For Series with disjoint indexes of length n, the union index has 2n entries, and two full reindex operations (each O(n)) must produce the aligned arrays before the O(n) element-wise operation.

### 1.5 GroupBy Aggregation

| Operation | Best Case | Typical Case | Worst Case | Space | Conditions |
|-----------|-----------|-------------|------------|-------|------------|
| `get_grouper()` | O(n) | O(n) | O(n) | O(n) | Extracts grouping column, computes codes via factorize. File: `core/groupby/grouper.py:722`. |
| `group_sum` (Cython) | O(n) | O(n) | O(n) | O(g) | g = number of groups. Single pass over data with array-indexed accumulators. File: `_libs/groupby.pyx:702`. |
| `group_mean` (Cython) | O(n) | O(n) | O(n) | O(g) | Two accumulators per group (sum + count). File: `_libs/groupby.pyx:1193`. |
| `group_var` (Cython) | O(n) | O(n) | O(n) | O(2*g) | Welford's online algorithm: sum, sum-of-squares, count per group. File: `_libs/groupby.pyx:906`. |
| `_agg_py_fallback` | O(n * g) | O(n * g) | O(n * g) | O(n) | Triggered for ExtensionArray/object dtypes. Iterates groups, applies Python function per group. Catastrophically slow for large data. File: `core/groupby/groupby.py:1462`. |
| `_cython_agg_general` | O(n * b) | O(n * b) | O(n * b) | O(g * b) | b = number of blocks. Applies Cython kernel to each block. File: `core/groupby/groupby.py:1510`. |
| `grouped_reduce` | O(n * m) | O(n * m) | O(n * m) | O(g * m) | Applies array_func per block in BlockManager. |
| Numba path | O(n) | O(n) | O(n + JIT) | O(g) | First call pays JIT compilation cost. Subsequent calls: pure O(n). |

**Complexity hierarchy:** Cython path >> Numba path > Python fallback. The Cython `group_sum` is a tight C loop: one pass over n values, indexing into a g-length accumulator array. The Python fallback iterates each group separately, paying interpreter overhead per element.

### 1.6 Merge / Join

| Operation | Best Case | Typical Case | Worst Case | Space | Conditions |
|-----------|-----------|-------------|------------|-------|------------|
| Hash join (single key) | O(n + m) | O(n + m) | O(n * m) | O(n + m + output) | Best: unique keys, O(n + m) output. Worst: all keys identical, cross-product O(n*m). File: `core/reshape/merge.py:2121`. |
| Sort-merge join | O(n + m) | O(n + m) | O(n + m) | O(n + m) | Requires both keys sorted + at least one unique. Uses `Index.join()` with binary search. File: `core/reshape/merge.py:2103`. |
| Multi-key factorize | O(n * k) | O(n * k) | O(n * k) | O(n * k) | k = number of key columns. Factorizes each key pair, then flattens via `_get_join_keys`. |
| `_reindex_and_concat` | O(output * m) | O(output * m) | O(output * m) | O(output * m) | Reindexes both sides' BlockManagers to output indexer, then column-concatenates. File: `core/reshape/merge.py:1081`. |
| `_factorize_keys` | O(n + m) | O(n + m) | O(n + m) | O(n + m + u) | Builds hash table, assigns integer codes. u = unique key count. |

**Critical path:** The hash join algorithm in `get_join_indexers_non_unique` (`merge.py:2121`) first factorizes both key arrays into integer codes (O(n + m)), then calls the appropriate Cython join function (O(n + m) for unique keys, O(n * m) worst case for many-to-many). The sort-merge path is chosen only when `left.is_monotonic_increasing AND right.is_monotonic_increasing AND (left.is_unique OR right.is_unique)`.

### 1.7 CSV Read

| Operation | Best Case | Typical Case | Worst Case | Space | Conditions |
|-----------|-----------|-------------|------------|-------|------------|
| C parser tokenize | O(bytes) | O(bytes) | O(bytes) | O(n * m) | Single-pass tokenizer in C. File: `_libs/parsers.pyx:2182`, `_libs/src/parser/`. |
| Python parser tokenize | O(bytes) | O(bytes * 3-5x) | O(bytes * 10x) | O(n * m) | Python string splitting, much slower constant factor. |
| PyArrow parser | O(bytes) | O(bytes) | O(bytes) | O(n * m) | Columnar read, no chunking support. Arrow memory pool. |
| dtype inference | O(n * m) | O(n * m) | O(n * m) | O(n * m) | `maybe_convert_objects` per column. File: `_libs/lib.pyx:3325`. |
| DataFrame assembly | O(n * m) | O(n * m) | O(n * m) | O(n * m) | `dict_to_mgr` + block consolidation. |
| Memory-mapped read | O(1) | O(page faults) | O(bytes) | O(1) virtual | OS handles paging; no explicit read. Only with `memory_map=True`. |

### 1.8 Sorting

| Operation | Best Case | Typical Case | Worst Case | Space | Conditions |
|-----------|-----------|-------------|------------|-------|------------|
| `sort_values` (single col) | O(n log n) | O(n log n) | O(n log n) | O(n) | numpy argsort (introsort/timsort). |
| `sort_values` (multi col) | O(n * k * log n) | O(n * k * log n) | O(n * k * log n) | O(n * k) | k = number of sort keys. `lexsort_indexer`. File: `core/sorting.py:736`. |
| `sort_index` | O(n log n) | O(n log n) | O(n log n) | O(n) | Delegates to `Index.argsort()`. |
| `rank` | O(n log n) | O(n log n) | O(n log n) | O(n) | Sort-based ranking. File: `_libs/algos.pyx:rank_1d`. |

### 1.9 Reshaping

| Operation | Best Case | Typical Case | Worst Case | Space | Conditions |
|-----------|-----------|-------------|------------|-------|------------|
| `concat` (axis=0) | O(sum(n_i)) | O(sum(n_i) * m) | O(sum(n_i) * m) | O(total * m) | Union or intersection of columns; reindex + block concatenation. |
| `concat` (axis=1) | O(sum(m_i) * n) | O(sum(m_i) * n) | O(sum(m_i) * n) | O(n * total_m) | Column-wise concatenation; index alignment required. |
| `pivot_table` | O(n * g) | O(n * g) | O(n * g) | O(g * v) | g = groups, v = value columns. Internally: groupby + unstack. |
| `stack/unstack` | O(n * m) | O(n * m) | O(n * m) | O(n * m) | Reshape indexer generation + take. File: `_libs/reshape.pyx`. |
| `melt` | O(n * v) | O(n * v) | O(n * v) | O(n * v) | v = number of value columns melted. |

### 1.10 Window Operations

| Operation | Best Case | Typical Case | Worst Case | Space | Conditions |
|-----------|-----------|-------------|------------|-------|------------|
| `rolling(w).sum()` | O(n) | O(n) | O(n) | O(1) | Sliding window sum via `roll_sum` with running accumulator. File: `_libs/window/aggregations.pyx:2118`. |
| `rolling(w).mean()` | O(n) | O(n) | O(n) | O(1) | Running mean from sum/count. |
| `rolling(w).var()` | O(n) | O(n) | O(n) | O(1) | Welford's online variance in sliding window. |
| `rolling(w).median()` | O(n * log w) | O(n * log w) | O(n * log w) | O(w) | Binary heap or skiplist median tracking. |
| `rolling(w).apply(func)` | O(n * w) | O(n * w) | O(n * w) | O(w) | Python function called per window. No vectorization. |
| Variable window bounds | O(n log n) | O(n log n) | O(n log n) | O(n) | Binary search for window boundaries. File: `_libs/window/indexers.pyx:151`. |

---

## 2. Memory Growth Model

### 2.1 BlockManager Memory Layout

The pandas `BlockManager` (`core/internals/managers.py`) is the central memory management structure. It groups columns by dtype into contiguous 2D blocks:

```
BlockManager
  axes: [Index(columns), Index(index)]
  blocks: [
    Block(dtype=float64, values=ndarray[float64, (k1, n)], mgr_locs=[0,3,7]),
    Block(dtype=int64,   values=ndarray[int64,   (k2, n)], mgr_locs=[1,4]),
    Block(dtype=object,  values=ndarray[object,  (k3, n)], mgr_locs=[2,5,6]),
  ]
  _blknos:  ndarray[int8,  (m,)]  -- maps col position -> block number
  _blklocs: ndarray[intp, (m,)]  -- maps col position -> position within block
```

**Memory formula for a consolidated BlockManager:**

```
M_BM(n, m, dtypes) = sum_d( sizeof(dtype_d) * n * count_d ) + overhead
```

Where `count_d` = number of columns with dtype `d`, and `overhead` includes:
- `_blknos` array: m bytes (int8)
- `_blklocs` array: m * sizeof(intp) bytes
- Index objects: 2 * (label storage + hash table)
- Block metadata: ~100 bytes per block
- Python object headers: ~56 bytes per Block, ~200 bytes per BlockManager

**Concrete example:** A 1M-row, 10-column DataFrame with 7 float64 + 3 int64 columns:

```
Data:   7 * 8 * 1M + 3 * 8 * 1M = 80 MB
Index:  1M * 8 (RangeIndex: virtual, ~0 bytes actual)
Columns: 10 * ~50 bytes (string labels) = 500 bytes
Overhead: ~1 KB
Total:  ~80 MB
```

### 2.2 Consolidation Costs

**File:** `core/internals/managers.py:2441-2475`

Consolidation merges same-dtype blocks. This occurs:
1. After column insertion (`df['new'] = values`) -- triggers `_consolidate_inplace`
2. After row concatenation -- result blocks may be fragmented
3. After certain mutations that split blocks

**Memory during consolidation:**

```
M_consolidate = M_existing + M_new_blocks + M_temp_concatenation
              = M_existing + sizeof(dtype) * n * k_same_dtype + temp_overhead
```

The `_merge_blocks` function (`managers.py:2458`) calls `np.concatenate` on block values, which allocates a new contiguous array. During consolidation, both old and new blocks coexist briefly, causing ~2x peak memory for the largest dtype group.

**Cost timeline:**
1. Sort blocks by consolidation key: O(b log b), b = block count
2. Group by dtype: O(b)
3. Per group: `np.concatenate` -- O(n * k) copy, O(n * k) allocation
4. Rebuild `_blknos`/`_blklocs`: O(m)

### 2.3 Index Memory

| Index Type | Memory per Label | Notes |
|------------|-----------------|-------|
| `RangeIndex` | O(1) total | Virtual: stores only start/stop/step. No materialization until needed. |
| `Int64Index` | 8 bytes | Plus hash table (~16 bytes/entry) lazily allocated on first `get_loc`. |
| `Float64Index` | 8 bytes | Same hash table overhead. |
| `DatetimeIndex` | 8 bytes (int64) | Plus timezone object reference. |
| `object Index` (strings) | ~70+ bytes | Python str object (~49 bytes) + pointer (8 bytes) + hash table entry (16 bytes). |
| `MultiIndex` | variable | `levels` (shared) + `codes` (int8/int16/int32 per level per row). |
| `CategoricalIndex` | 1-4 bytes | Integer codes + shared categories Index. Very compact for low-cardinality. |

**Hash table construction cost:** The `IndexEngine` (`_libs/index.pyx`) builds a hash table on first `get_loc` call. For n labels:
- Time: O(n) -- single pass, insert each label
- Space: O(n * 16 bytes) -- khash-based hash table with load factor ~0.7
- This is a one-time cost, amortized over subsequent lookups

### 2.4 GroupBy Intermediates

The `_cython_agg_general` path creates these intermediate structures:

```
M_groupby(n, g, m_numeric) =
    codes_array:    n * sizeof(intp)           -- group assignment for each row
  + uniques:        g * label_size             -- unique group keys
  + result_values:  g * m_numeric * sizeof(f64) -- one accumulator per group per column
  + nobs_array:     g * m_numeric * sizeof(i64) -- count per group per column (some aggs)
  + comp_array:     g * m_numeric * sizeof(f64) -- compensated sum correction (group_sum)
```

**For `groupby.sum()` on 1M rows, 5 numeric columns, 1000 groups:**

```
codes:    1M * 8 = 8 MB
uniques:  1000 * ~50 = 50 KB
results:  1000 * 5 * 8 = 40 KB
nobs:     1000 * 5 * 8 = 40 KB
comp:     1000 * 5 * 8 = 40 KB
Total intermediate: ~8.2 MB
```

The `_agg_py_fallback` path is worse: it materializes each group as a full Series before applying the aggregation, costing O(n) additional memory for the splitter.

### 2.5 Merge Expansion

Merge operations can dramatically expand memory. The output size depends on key uniqueness:

| Scenario | Output Rows | Memory Growth |
|----------|------------|---------------|
| Inner, both keys unique | min(n, m) | 1x |
| Inner, one-to-many | n * avg_matches | 1x-10x typical |
| Outer, both unique | n + m - overlap | ~2x |
| Many-to-many | up to n * m | Catastrophic: 10^6 * 10^6 = 10^12 rows |

**Intermediate memory during merge:**

```
M_merge(n, m, output) =
    left_indexer:   output * sizeof(intp)   -- maps output -> left row
  + right_indexer:  output * sizeof(intp)   -- maps output -> right row
  + factorized_keys: (n + m) * sizeof(i64)  -- if multi-key
  + hash_table:     min(n, m) * ~32 bytes   -- right-side hash table
  + reindexed_blocks: output * (m_left + m_right) * sizeof(dtype)  -- output data
```

**Example: merging two 1M-row DataFrames, 5 columns each, unique keys, inner join:**

```
Indexers:     2 * 1M * 8 = 16 MB
Hash table:   1M * 32 = 32 MB
Output data:  1M * 10 * 8 = 80 MB
Total:        ~128 MB (for ~80 MB input)
```

### 2.6 Copy-on-Write (CoW) Memory Implications

Pandas 3.0 enforces CoW. Key memory implications:

1. **Shallow copies cost O(1):** `df.copy(deep=False)` creates new Block objects pointing to the same numpy arrays.
2. **Mutation triggers O(n*k) copy:** First write to a shared block copies the entire block (not just the modified column).
3. **Reference tracking overhead:** Each Block gets a `BlockValuesRefs` weak-reference set (~100 bytes per block).
4. **Peak memory on mutation:** Up to 2x the block size during copy-on-write.

File: `core/internals/managers.py`, `_libs/internals.pyx:1026` (BlockValuesRefs tracking).

---

## 3. Hotspot Families

### 3.1 Family H1: Cython Aggregation Kernels

**Location:** `_libs/groupby.pyx` (2,325 LOC), `_libs/algos.pyx` (1,445 LOC)

These are the tightest performance-critical loops in pandas. They operate on C-contiguous arrays with typed inner loops, avoiding Python interpreter overhead entirely.

**Key functions and their characteristics:**

| Function | LOC | Complexity | Hotness | Notes |
|----------|-----|-----------|---------|-------|
| `group_sum` (pyx:702) | ~200 | O(n) | Extreme | Compensated (Kahan) summation with NA masking. Inner loop: `val = values[i, j]; if val == val: sumx[lab, j] += val`. |
| `group_mean` (pyx:1193) | ~90 | O(n) | Extreme | Two arrays: sum + count. Division at the end. |
| `group_var` (pyx:906) | ~100 | O(n) | High | Welford's online algorithm. Three arrays per group: mean, M2, count. |
| `group_nth` / `group_last` | ~60 | O(n) | Medium | Conditional assignment with rank tracking. |
| `group_rank` | ~150 | O(n log n) | Medium | Sort within each group, then assign ranks. |
| `group_cumsum` | ~80 | O(n) | Medium | Running sum with group-level reset. |
| `rank_1d` (algos.pyx) | ~200 | O(n log n) | High | Argsort + rank assignment with tie-breaking. |
| `take_2d` (algos.pyx) | ~100 | O(n*m) | Extreme | Reindexing kernel. Called during every merge, reindex, and alignment. |
| `groupsort_indexer` | ~50 | O(n) | High | Counting sort for group indices. |

**Performance characteristics:**
- Inner loops compile to SIMD-friendly C code via Cython
- Templated (via `.pxi.in` files) for int64, float64, object, etc.
- NA handling via `val == val` (NaN check) or explicit mask arrays
- Compensated summation (`group_sum`) prevents floating-point drift

### 3.2 Family H2: Hash Table Operations

**Location:** `_libs/hashtable.pyx` (128 LOC + 1,572 LOC templates), `_libs/index.pyx` (1,325 LOC)

Hash tables are the backbone of index lookups, factorization, merge key matching, and deduplication.

**Key operations:**

| Operation | Location | Time | Space | Frequency |
|-----------|----------|------|-------|-----------|
| `IndexEngine.get_loc` | `index.pyx` | O(1) amortized | O(n) (first call) | Every `df['col']`, `df.loc[label]`, alignment |
| `Int64HashTable.factorize` | `hashtable.pyx` template | O(n) | O(n + u) | Every `groupby`, `merge`, `unique`, `duplicated` |
| `_factorize_keys` (merge) | `merge.py:2121` -> `hashtable` | O(n + m) | O(n + m) | Every merge operation |
| `PyObjectHashTable.get_item` | `hashtable.pyx` template | O(1) amortized | O(n) | Object-dtype index lookups |

**Template generation:** The `hashtable_class_helper.pxi.in` (1,572 LOC) generates specialized hash tables for each dtype. This avoids Python-level type dispatch but produces ~8x code bloat at compile time.

**Performance concern:** Object-dtype hash tables (`PyObjectHashTable`) are dramatically slower than numeric-dtype tables because:
1. Each hash requires a Python `__hash__` call (interpreter overhead)
2. Each comparison requires a Python `__eq__` call
3. No SIMD vectorization possible
4. GIL contention on multi-threaded access

### 3.3 Family H3: Index Alignment and Reindexing

**Location:** `core/generic.py` (align), `core/internals/managers.py` (reindex_indexer)

Every binary operation, merge, and concat must align indexes. This is the "hidden tax" of pandas' alignment-aware semantics.

**Alignment call chain:**

```
Series._arith_method(other, op)
  -> Series._align_for_op(other)
    -> left.align(right)                           -- O(n + m) index union
      -> Index.join(other, how='outer')
        -> _libs.join.outer_join_indexer()          -- Cython join
        -> left.reindex(join_index)                 -- O(n) take
        -> right.reindex(join_index)                -- O(m) take
```

**Reindex path (`managers.py:792`):**

```
BlockManager.reindex_indexer(new_axis, indexer, axis, ...)
  -> For each block:
    -> Block.take_nd(indexer)                       -- fancy indexing
      -> algos.take_nd(values, indexer, fill_value) -- Cython take
```

The `take_nd` function (`_libs/algos.pyx`) is templated for all numeric dtypes and handles fill values for missing positions. This is one of the most frequently called Cython functions in the entire codebase.

**Overhead analysis:** For a simple `s1 + s2` with different indexes of size n each:
1. Index union: O(n) hash table build + O(n) probe = O(n)
2. Left reindex: O(n) indexer computation + O(n) take = O(n)
3. Right reindex: O(n) indexer computation + O(n) take = O(n)
4. Element-wise add: O(n)
5. Result construction: O(n) (new Block, new Index)
Total: ~5n operations for a conceptually O(n) operation.

### 3.4 Family H4: Dtype Coercion Cascades

**Location:** `core/dtypes/cast.py` (2,282 LOC), `core/construction.py` (852 LOC), `_libs/lib.pyx` (`maybe_convert_objects`)

Dtype coercion is a pervasive cost that triggers in unexpected places:

1. **On construction:** `sanitize_array` → `maybe_convert_objects` → type inference per element
2. **On merge:** `_maybe_coerce_merge_keys` → cast left/right keys to common type
3. **On consolidation:** blocks of different dtypes that should be same type
4. **On arithmetic:** `common_dtype` resolution → potential upcasting

**The `maybe_convert_objects` bottleneck:**

**File:** `_libs/lib.pyx:3325` (inside the 3,325-LOC utility megamodule)

This function scans an object-dtype array to detect if all elements are:
- Numeric (int/float) → convert to numeric array
- Boolean → convert to bool array
- Datetime-like → convert to datetime64 array
- Complex → convert to complex array
- All same type → downcast

**Complexity:** O(n * type_checks), where type_checks involves `isinstance` calls and numeric parsing per element. For 1M-row object arrays, this can take hundreds of milliseconds.

### 3.5 Family H5: Python Fallback Paths

**Location:** `core/groupby/groupby.py:1462` (`_agg_py_fallback`), `core/apply.py` (`FrameApply`)

When Cython kernels cannot handle a dtype (ExtensionArrays, object columns), pandas falls back to Python-level iteration:

```
_agg_py_fallback(how, values, ndim, alt)
  -> GroupBy._grouper.agg_series(ser, alt, preserve_dtype=True)
    -> for group_idx, group_values in splitter:    -- Python loop
        result[group_idx] = alt(group_values)      -- Python function call per group
```

**Performance cliff:**

| Path | Time for 1M rows, 1000 groups | Factor |
|------|------------------------------|--------|
| Cython `group_sum` (float64) | ~2 ms | 1x |
| Numba `grouped_sum` | ~5 ms | 2.5x |
| Python fallback (np.sum) | ~200 ms | 100x |
| Python fallback (custom func) | ~500 ms | 250x |

This cliff is one of the most common performance surprises in pandas. Users writing `df.groupby('key').agg(custom_func)` unknowingly bypass the Cython fast path.

### 3.6 Family H6: String Operations

**Location:** `core/strings/accessor.py`, `core/arrays/string_.py`, `core/arrays/string_arrow.py`

String operations are inherently expensive due to:
1. Variable-length data (no SIMD vectorization)
2. Python object overhead for `object`-dtype strings (~49 bytes per string + pointer)
3. Regex compilation per call (cached via `re.compile`)
4. No in-place mutation (new string objects for every operation)

**ArrowStringArray** (`core/arrays/string_arrow.py:592`) mitigates this by using PyArrow's `large_string` type with zero-copy slicing and Arrow compute kernels written in C++. Benchmarks show 2-10x speedup for common string operations.

### 3.7 Family H7: CSV Parsing and Type Inference

**Location:** `_libs/parsers.pyx` (2,182 LOC), `_libs/src/parser/` (2,255 LOC C)

The C parser tokenizer is the fastest path:
1. C-level tokenization: byte-by-byte scanning, delimiter detection, quoting
2. Type inference per column: scan first N rows, detect numeric/date/string
3. Batch conversion: `strtod`/`strtol` for numeric, `datetime_parse` for dates

**Bottleneck hierarchy:**
1. I/O read: dominated by disk/network, not CPU (unless memory-mapped)
2. Tokenization: C loop, ~1 GB/s throughput
3. Type inference: O(n * m * type_checks), can be significant for wide DataFrames
4. DataFrame assembly: O(n * m) for block creation + consolidation

---

## 4. FrankenPandas Performance Characteristics

### 4.1 Architecture Differences

FrankenPandas eliminates several pandas performance bottlenecks through architectural choices:

| Pandas Overhead | FP Architecture | Performance Impact |
|----------------|-----------------|-------------------|
| BlockManager consolidation | Flat `BTreeMap<String, Column>` per-column storage | No consolidation cost. O(1) column insertion. File: `fp-frame/src/lib.rs:617`. |
| `_blknos`/`_blklocs` indirection | Direct column access via `BTreeMap::get` | O(log m) lookup vs O(1) array index, but no consolidation tax. |
| Python object Scalar | Rust `enum Scalar { Int64(i64), Float64(f64), ... }` | 24 bytes (enum) vs 28+ bytes (Python int) + no GC pressure. File: `fp-types/src/lib.rs:26`. |
| Copy-on-Write tracking | Rust ownership model (move/clone semantics) | No reference counting overhead. Explicit clones. |
| IndexEngine hash table (lazy) | `OnceCell`-cached sort order + HashMap | `OnceCell` initialization: thread-safe, zero-cost after first use. File: `fp-index/src/lib.rs:111`. |
| GIL contention | No GIL (Rust is inherently thread-safe) | Future parallelism path is unblocked. |

### 4.2 Index Lookup Complexity

FP implements an adaptive index backend (AG-13):

```
Index::position(&self, needle) -> Option<usize>
  match self.sort_order():
    AscendingInt64 -> binary_search: O(log n)
    AscendingUtf8  -> binary_search: O(log n)
    Unsorted       -> linear scan:   O(n)
```

**File:** `fp-index/src/lib.rs:196-229`

**Comparison with pandas:**

| Index State | pandas | FP | Advantage |
|-------------|--------|-----|-----------|
| Sorted, unique | O(log n) binary search | O(log n) binary search | Parity |
| Unsorted, unique | O(1) hash table (after O(n) build) | O(n) linear scan | Pandas wins (amortized) |
| Unsorted, repeated lookups | O(1) per lookup | O(n) per lookup | Pandas wins dramatically |
| First lookup, cold | O(n) hash table build | O(n) sort order detection | Parity |

**Gap:** FP does not build an IndexEngine-style hash table for unsorted indexes. For repeated point lookups on unsorted indexes, pandas is significantly faster after the first call. The `position_map_first_ref()` method provides HashMap-based bulk lookup but is not cached.

### 4.3 Alignment Complexity

FP's alignment (`fp-index/src/lib.rs:461-546`) uses borrowed-key HashMaps:

```
align_union(left, right) -> AlignmentPlan:
  left_map  = left.position_map_first_ref()    -- O(n), HashMap<&IndexLabel, usize>
  right_map = right.position_map_first_ref()    -- O(m), HashMap<&IndexLabel, usize>
  union_labels = left.labels.clone()            -- O(n)
  for label in right.labels:                    -- O(m)
    if !left_map.contains(label):
      union_labels.push(label.clone())
  left_positions  = union.map(|l| left_map.get(l))   -- O(u)
  right_positions = union.map(|l| right_map.get(l))   -- O(u)
```

**Total:** O(n + m + u), where u = |union| <= n + m.
**Space:** O(n + m) for the two HashMaps, O(u) for output vectors.

**AG-02 optimization (borrowed keys):** The `position_map_first_ref()` method returns `HashMap<&IndexLabel, usize>` instead of `HashMap<IndexLabel, usize>`, avoiding cloning every index label during map construction. This saves O(n) allocations for string-typed indexes.

### 4.4 GroupBy Complexity

FP provides three paths for `groupby_sum` (`fp-groupby/src/lib.rs`):

**Path 1: Dense Int64 bucket path**

```
try_groupby_sum_dense_int64(keys, values, dropna):
  Scan keys: find (min, max)               -- O(n)
  if span <= 65,536:
    Allocate sums[span], seen[span]        -- O(span)
    Single pass: sums[key - min] += val    -- O(n)
    Emit results in first-seen order       -- O(g)
```

**File:** `fp-groupby/src/lib.rs:324-377`
**Complexity:** O(n + span) time, O(span) space.
**Condition:** All keys Int64, span <= 65,536 (`DENSE_INT_KEY_RANGE_LIMIT`).

**Path 2: HashMap with borrowed keys (AG-08)**

```
groupby_sum_with_global_allocator:
  HashMap<GroupKeyRef<'_>, (usize, f64)>   -- borrowed key refs
  Single pass: accumulate sums per group   -- O(n)
  Emit results in first-seen order         -- O(g)
```

**File:** `fp-groupby/src/lib.rs:149-191`
**Complexity:** O(n) amortized time, O(g) space for HashMap + ordering vector.

**Path 3: Arena-backed (AG-06)**

Same algorithm as Path 2, but ordering vector allocated in `bumpalo::Bump` arena. Reduces allocator pressure for large group counts.

**File:** `fp-groupby/src/lib.rs:193-239`

**Comparison with pandas Cython `group_sum`:**

| Aspect | pandas `group_sum` | FP `groupby_sum` |
|--------|-------------------|------------------|
| Inner loop | C array-indexed accumulators | HashMap probe + insert |
| NA handling | `val == val` (NaN check) | `value.is_missing()` (enum match) |
| Kahan summation | Yes (compensated) | No (naive summation) |
| Numeric stability | Better for large sums | Potential drift |
| Dense-int path | No (always uses codes) | Yes, O(span) direct indexing |
| Object-dtype | Falls to Python `_agg_py_fallback` | Handled uniformly via `Scalar` enum |

### 4.5 Join Complexity

FP's join (`fp-join/src/lib.rs:76-131`):

```
join_series_with_trace(left, right, join_type, options):
  Build right_map: HashMap<&IndexLabel, Vec<usize>>   -- O(m)
  If Right/Outer: build left_map similarly              -- O(n)
  Estimate output rows                                  -- O(n + m)
  Iterate left labels, probe right_map                  -- O(n * avg_matches)
  If Outer: append unmatched right labels               -- O(m)
  Reindex left_values by left_positions                 -- O(output)
  Reindex right_values by right_positions               -- O(output)
```

**Total:** O(n + m + output), same asymptotic as pandas.
**Space:** O(n + m + output).

**AG-02 optimization:** Hash maps use `&IndexLabel` (borrowed keys) rather than cloned labels. For the right_map build phase, this saves m allocations.

**Arena path (AG-06):** Position vectors (`left_positions`, `right_positions`) allocated in `bumpalo::Bump` arena when estimated intermediate size <= 256 MB budget (`DEFAULT_ARENA_BUDGET_BYTES`). This reduces global allocator fragmentation.

**File:** `fp-join/src/lib.rs:259-332`

### 4.6 Vectorized Binary Arithmetic (AG-10)

FP's `Column::binary_numeric` tries a vectorized path before falling back to scalar:

**Vectorized path (`fp-columnar/src/lib.rs:572-648`):**

```
try_vectorized_binary(right, op, out_dtype):
  if Float64:
    ColumnData::from_scalars(values, Float64)      -- O(n) conversion
    vectorized_binary_f64(left, right, validity)    -- O(n) contiguous slice ops
    Build output scalars from result + validity     -- O(n)
  if Int64 (non-Div):
    Same pattern with i64 slices
  else: None (fallback to scalar)
```

**Scalar fallback path:**

```
for (left, right) in self.values.zip(right.values):
  if left.is_missing() || right.is_missing():
    -> null propagation (enum match)
  else:
    left.to_f64()? op right.to_f64()?              -- 2 enum matches + f64 op
```

**Performance difference:** The vectorized path operates on `&[f64]` slices, which LLVM can auto-vectorize using SIMD instructions. The scalar fallback requires per-element enum discrimination (branch prediction cost). Expected speedup: 2-4x for large columns.

**File:** `fp-columnar/src/lib.rs:362-398` (vectorized_binary_f64)

### 4.7 ValidityMask Bit-Packing (AG-04)

FP uses a packed bitvec (`Vec<u64>`) for validity masks instead of `Vec<bool>`:

```
ValidityMask:
  words: Vec<u64>     -- 1 bit per element, 64 elements per word
  len: usize
```

**File:** `fp-columnar/src/lib.rs:10-137`

**Memory savings:** For a 1M-element column:
- `Vec<bool>`: 1,000,000 bytes = ~1 MB
- `ValidityMask`: 1,000,000 / 64 = 15,625 words * 8 bytes = ~122 KB
- **Savings: 8x**

**Operations:**
- `count_valid()`: uses hardware `popcnt` via `u64::count_ones()` -- O(n/64) vs O(n)
- `and_mask()`: word-level AND -- O(n/64) vs O(n)
- `or_mask()`: word-level OR -- O(n/64) vs O(n)
- `not_mask()`: word-level NOT -- O(n/64)
- `get(idx)`: bit shift + mask -- O(1)

### 4.8 Leapfrog Triejoin (AG-11)

FP's multi-way index alignment uses a min-heap merge instead of iterative pairwise union:

**Pairwise approach (before AG-11):**
```
union(A, B)         -- O(|A| + |B|)
union(result, C)    -- O(|A| + |B| + |C|)
union(result, D)    -- O(|A| + |B| + |C| + |D|)
...
Total: O(K * N)  where K = number of inputs, N = total labels
```

**Leapfrog union (`fp-index/src/lib.rs:572-619`):**
```
Sort + dedup each input: O(N log(N/K))
Min-heap merge: O(N log K)
Total: O(N log K)
```

**File:** `fp-index/src/lib.rs:572` (`leapfrog_union`), `fp-index/src/lib.rs:626` (`leapfrog_intersection`)

**Improvement factor:** For K=10 inputs of 10,000 labels each, the pairwise approach does ~10 * 100,000 = 1M comparisons. The leapfrog approach does ~100,000 * log(10) = ~330K comparisons. The improvement grows with K.

### 4.9 IVM Delta Propagation (AG-15)

FP's expression evaluator supports incremental view maintenance:

```
MaterializedView::apply_deltas(deltas, policy, ledger):
  For Expr::Add { left, right }:
    Extract deltas for left_name and right_name
    Build new_left = Series::from_values(delta.new_labels, delta.new_values)
    Build new_right from other delta (or empty)
    Compute delta_result = new_left.add(new_right)?
    Concatenate base_result with delta_result
```

**File:** `fp-expr/src/lib.rs:94-180`

**Complexity:** O(d) where d = delta size, instead of O(n + d) for full re-evaluation. This is a significant win for streaming/append workloads.

---

## 5. Optimization Opportunities

### 5.1 Completed AG Optimizations

| AG Round | Optimization | Crate | Impact | Status |
|----------|-------------|-------|--------|--------|
| AG-02 | Borrowed-key HashMap in `align_union` | fp-index | Eliminates O(n) label clones during alignment | CLOSED |
| AG-03 | `cast_scalar_owned` in `Column::new` | fp-columnar | Eliminates unnecessary clone when values match dtype | CLOSED |
| AG-04 | Packed bitvec `ValidityMask` | fp-columnar | 8x memory reduction, hardware popcnt for `count_valid` | CLOSED |
| AG-05 | Identity-alignment fast path for `groupby_sum` | fp-groupby | Skips alignment when key/value indexes match | CLOSED |
| AG-06 | Bumpalo arena allocation for groupby/join intermediates | fp-groupby, fp-join | Reduces allocator fragmentation, bulk deallocation | CLOSED |
| AG-07 | Vec-based CSV column accumulation | fp-io | O(1) per cell vs O(log c) BTreeMap insert | CLOSED |
| AG-08 | `GroupKeyRef` borrowed keys in groupby | fp-groupby | Eliminates per-group key cloning | CLOSED |
| AG-10 | Vectorized `f64`/`i64` binary arithmetic | fp-columnar | SIMD-friendly contiguous slice operations | CLOSED |
| AG-11 | Leapfrog triejoin for multi-way alignment | fp-index | O(N log K) vs O(K*N) for K-way union | CLOSED |
| AG-13 | Adaptive index backend (binary search for sorted) | fp-index | O(log n) vs O(n) for sorted-index lookups | CLOSED |
| AG-15 | IVM delta propagation in fp-expr | fp-expr | O(delta) incremental update vs O(n) full re-eval | CLOSED |

### 5.2 Remaining Optimization Gaps

#### Gap G1: No Cached Hash Table for Unsorted Indexes

**Current:** Unsorted index `position()` is O(n) linear scan per call.
**Pandas:** O(1) amortized after O(n) IndexEngine build.
**Fix:** Add `OnceCell<HashMap<&IndexLabel, usize>>` to `Index`, lazily built on first `position()` call to unsorted index. Requires self-referential borrow (or `IndexLabel -> usize` with owned keys).
**Impact:** Critical for workloads with repeated lookups on unsorted indexes.
**Estimated speedup:** 100x+ for repeated lookups (O(1) vs O(n)).

#### Gap G2: No Sort-Merge Join Path

**Current:** FP always uses hash join for `join_series`.
**Pandas:** Uses sort-merge when both keys are sorted and at least one is unique (O(n + m) with better constants than hash join).
**Fix:** Add a pre-check in `join_series_with_trace`: if both indexes `is_sorted()` and at least one `!has_duplicates()`, use a merge-join sweep.
**Impact:** Moderate. Better cache locality for large sorted datasets.

#### Gap G3: No Compensated (Kahan) Summation

**Current:** FP's `groupby_sum` uses naive `+=` accumulation.
**Pandas:** Uses compensated summation in `group_sum` Cython kernel.
**Fix:** Implement Kahan-Babushka-Neumaier summation in the groupby accumulation loop.
**Impact:** Numerical accuracy for large sums with wide value ranges. Minimal performance cost (~5% overhead).

#### Gap G4: No Chunked/Streaming CSV Read

**Current:** FP reads entire CSV into memory as a string, then parses.
**Pandas:** C parser supports chunked reading, memory-mapping, and iterator-based lazy evaluation.
**Fix:** Use `csv::Reader` on a `BufReader<File>` for streaming row processing without loading the entire file.
**Impact:** Memory-critical for large files (>1 GB). Current FP approach requires 2x file size in memory (string + parsed data).

#### Gap G5: No Type-Specialized Column Storage

**Current:** FP stores all column data as `Vec<Scalar>`, with `Scalar` being a 24-byte enum. The `ColumnData` typed array representation (AG-10) is only used transiently during binary arithmetic.
**Pandas:** Stores columns as contiguous numpy arrays of their native dtype (8 bytes for int64/float64).
**Fix:** Make `ColumnData` the primary storage, with `Vec<Scalar>` as a compatibility/materialization layer.
**Impact:** 3x memory reduction for numeric columns (24 bytes/element -> 8 bytes). SIMD vectorization for all operations, not just binary arithmetic.

#### Gap G6: No Parallel Execution

**Current:** All FP operations are single-threaded.
**Pandas:** Also mostly single-threaded (GIL), but Numba path supports nogil parallelism.
**Fix:** Rust's `rayon` crate for data-parallel operations (groupby, column-wise operations, CSV parsing).
**Impact:** Near-linear speedup with core count for embarrassingly parallel operations.

#### Gap G7: No Expression Tree Optimization

**Current:** FP's `Expr` evaluator materializes each intermediate result.
**Pandas:** Also materializes intermediates (no lazy evaluation in core).
**Fix:** Add expression fusion: `(a + b) * c` -> single fused pass over three columns instead of two passes.
**Impact:** 2x reduction in memory allocation and cache misses for compound expressions.

#### Gap G8: Outer Join Unmatched-Right Linear Scan

**Current:** The outer join path in `join_series_with_global_allocator` (`fp-join/src/lib.rs:219-227`) uses `left.index().labels().contains(label)` to check if a right label is unmatched -- this is O(n) per right label, making the total O(n * m).
**Fix:** Use the already-computed `left_map` (or build one) to check membership in O(1).
**Impact:** Critical for large outer joins. O(n * m) -> O(n + m).

### 5.3 Optimization Priority Matrix

| Gap | Effort | Impact | Priority |
|-----|--------|--------|----------|
| G1: Cached hash table | Medium | High (100x for hot lookups) | P0 |
| G5: Type-specialized storage | High | High (3x memory, SIMD everywhere) | P0 |
| G8: Outer join linear scan | Low | High (O(n*m) -> O(n+m)) | P0 |
| G4: Streaming CSV | Medium | Medium (large file support) | P1 |
| G3: Kahan summation | Low | Medium (numerical accuracy) | P1 |
| G2: Sort-merge join | Medium | Low-Medium (better constants) | P2 |
| G7: Expression fusion | High | Medium (2x for compound exprs) | P2 |
| G6: Parallel execution | High | High (multi-core scaling) | P2 (future) |

---

## 6. Performance Sentinels

Performance sentinels are benchmark anchors that detect regressions and validate optimizations. Each sentinel defines an operation, input characteristics, and expected complexity bounds.

### 6.1 Sentinel S1: Index Lookup Scaling

**What:** `Index::position()` for sorted vs unsorted indexes.
**Input:** Indexes of size 10, 100, 1K, 10K, 100K, 1M.
**Expected:**
- Sorted: O(log n) -- lookup time should grow by ~3.3x per 10x size increase
- Unsorted: O(n) -- lookup time should grow by 10x per 10x size increase
- With cached hash (future G1): O(1) -- constant after first call

**Regression threshold:** >20% deviation from expected scaling.

**File references:**
- FP: `fp-index/src/lib.rs:196` (`Index::position`)
- pandas: `_libs/index.pyx` (`IndexEngine.get_loc`)

### 6.2 Sentinel S2: Alignment Overhead Ratio

**What:** Ratio of `s1.add(s2)` time to raw element-wise addition time.
**Input:** Two Series of size n with varying overlap ratios (100%, 50%, 0%).
**Expected:**
- 100% overlap, same index identity: ~1.2x (fast path, no alignment)
- 100% overlap, different objects: ~3x (alignment + reindex + add)
- 50% overlap: ~4x (alignment + reindex + add + null filling)
- 0% overlap: ~5x (full alignment + all nulls)

**Regression threshold:** >50% increase in overhead ratio.

**File references:**
- FP: `fp-frame/src/lib.rs:107` (`binary_op_with_policy`)
- pandas: `core/series.py:6939` (`Series._arith_method`)

### 6.3 Sentinel S3: GroupBy Sum Throughput

**What:** Rows processed per second for `groupby_sum`.
**Input:** n=1M rows, varying group counts: 10, 100, 1K, 10K, 100K.
**Expected paths:**
- Dense int64 path (g <= 65K, Int64 keys): highest throughput
- HashMap path (g > 65K or non-Int64 keys): ~2-5x slower
- Arena vs global allocator: ~1.1-1.3x improvement

**Regression threshold:** >30% throughput decrease.

**File references:**
- FP: `fp-groupby/src/lib.rs:88` (`groupby_sum_with_trace`)
- FP dense: `fp-groupby/src/lib.rs:324` (`try_groupby_sum_dense_int64`)
- pandas: `_libs/groupby.pyx:702` (`group_sum`)

### 6.4 Sentinel S4: Join Output/Input Ratio

**What:** Time per output row for hash join.
**Input:** Left=1M, Right=1M, varying selectivity: unique keys (1:1), 1:10, 1:100.
**Expected:**
- 1:1 join: ~2x input time (hash build + probe + reindex)
- 1:10 join: ~10x input rows in output, proportional time increase
- 1:100 join: output-dominated, time proportional to output size

**Regression threshold:** >40% increase in per-output-row time.

**File references:**
- FP: `fp-join/src/lib.rs:76` (`join_series_with_trace`)
- pandas: `core/reshape/merge.py:2043` (`get_join_indexers`)

### 6.5 Sentinel S5: CSV Parse Rate

**What:** MB/s throughput for CSV parsing.
**Input:** 100MB CSV file, 10 columns: 5 int, 3 float, 2 string.
**Expected:**
- FP (Rust `csv` crate): ~200-400 MB/s
- pandas C parser: ~400-800 MB/s
- pandas Python parser: ~50-100 MB/s

**Regression threshold:** >25% throughput decrease.

**File references:**
- FP: `fp-io/src/lib.rs:59` (`read_csv_str`)
- pandas: `_libs/parsers.pyx:2182`, `_libs/src/parser/`

### 6.6 Sentinel S6: Memory Efficiency Ratio

**What:** RSS memory per data byte for numeric DataFrames.
**Input:** 1M rows x 10 float64 columns = 80 MB logical data.
**Expected:**
- FP current (Vec<Scalar>): ~240 MB (3x: 24-byte Scalar enum)
- FP future (ColumnData): ~82 MB (~1.03x: 8-byte f64 + validity mask)
- pandas (BlockManager): ~82 MB (~1.03x: consolidated float64 block)
- pandas (fragmented): ~100 MB (~1.25x: multiple blocks + overhead)

**Regression threshold:** >20% increase in memory ratio.

### 6.7 Sentinel S7: Leapfrog vs Pairwise Multi-Way Alignment

**What:** Ratio of leapfrog_union time to iterative pairwise union time.
**Input:** K indexes of size N/K each (total N labels, 50% overlap between adjacent pairs).
**Expected:**
- K=2: ~1x (leapfrog has overhead for small K)
- K=5: ~0.7x (leapfrog wins)
- K=10: ~0.5x (leapfrog wins clearly)
- K=50: ~0.3x (leapfrog dominates)

**Regression threshold:** Leapfrog should never be >1.5x slower than pairwise for K >= 3.

**File references:**
- FP leapfrog: `fp-index/src/lib.rs:572` (`leapfrog_union`)
- FP pairwise: `fp-index/src/lib.rs:519` (`align_union`)

### 6.8 Sentinel S8: ValidityMask Operations

**What:** `count_valid()` and `and_mask()` throughput.
**Input:** Masks of size 1K, 10K, 100K, 1M, 10M.
**Expected:**
- `count_valid`: O(n/64) via `popcnt` -- should be ~64x faster than naive O(n) bool count
- `and_mask`: O(n/64) via word-level AND -- same 64x factor
- `or_mask`: O(n/64)

**Regression threshold:** If packed-bitvec path regresses to >1/32x of theoretical speedup.

**File references:**
- FP: `fp-columnar/src/lib.rs:70` (`ValidityMask::count_valid`)

### 6.9 Sentinel S9: IVM Delta Amortization

**What:** Amortized cost of `apply_deltas` vs full `evaluate` for streaming appends.
**Input:** Base Series of 1M rows, deltas of 1K, 10K, 100K rows.
**Expected:**
- 1K delta: IVM should be ~1000x faster than full re-eval
- 10K delta: ~100x faster
- 100K delta: ~10x faster
- At some delta size (~50% of base), full re-eval is cheaper.

**Crossover threshold:** IVM should win for deltas <= 30% of base size.

**File references:**
- FP: `fp-expr/src/lib.rs:94` (`MaterializedView::apply_deltas`)

### 6.10 Sentinel S10: Vectorized vs Scalar Binary Arithmetic

**What:** Throughput ratio of AG-10 vectorized path vs scalar fallback.
**Input:** Two Float64 columns of size 1K, 10K, 100K, 1M.
**Expected:**
- Vectorized (contiguous f64 slice): ~2-4x faster than scalar path
- Improvement should be consistent across sizes (dominated by per-element cost)
- Int64 non-Div path: ~2x faster (less conversion overhead)

**Regression threshold:** Vectorized path should always be >= 1.5x faster for n >= 1K.

**File references:**
- FP vectorized: `fp-columnar/src/lib.rs:367` (`vectorized_binary_f64`)
- FP scalar: `fp-columnar/src/lib.rs:688` (scalar fallback in `binary_numeric`)

### 6.11 Sentinel S11: ASUPERSYNC Join-Admission Decision Path

**What:** Compare baseline vs optimized `RuntimePolicy::decide_join_admission` behavior and latency/allocation profile under mixed-cap workloads.

**Input:** Hardened policy with cap `2048`; 256 deterministic mixed rows (`<= cap` and `> cap`) as executed by:
- `asupersync_join_admission_optimized_path_is_isomorphic_to_baseline`
- `asupersync_join_admission_profile_snapshot_reports_allocation_delta`

**Expected:**
- **Isomorphism:** Identical action, issue payload, prior, metrics, and evidence coefficients between optimized and baseline paths.
- **Allocation budget:** `EvidenceTerm.name` allocation for decision evidence is zero on the optimized path (borrowed labels via `Cow<'static, str>`), versus baseline per-call owned strings.
- **Latency snapshot:** Track `p50/p95/p99` for both paths; on 2026-02-15 snapshot, baseline `430/571/5029 ns`, optimized `380/2695/4458 ns` (local fallback run via `rch`).

**Regression threshold:**
- Any semantic mismatch in optimized-vs-baseline record comparison is a hard fail.
- Any non-zero optimized evidence-name allocation budget is a hard fail.

**File references:**
- Runtime policy path: `crates/fp-runtime/src/lib.rs:254` (`decide_join_admission`)
- Optimization constants: `crates/fp-runtime/src/lib.rs:76`, `crates/fp-runtime/src/lib.rs:89`, `crates/fp-runtime/src/lib.rs:102`, `crates/fp-runtime/src/lib.rs:115`
- Isomorphism/profiling proofs: `crates/fp-runtime/src/lib.rs:747`, `crates/fp-runtime/src/lib.rs:769`
- Replay command: `rch exec -- cargo test -p fp-runtime --lib asupersync_join_admission_profile_snapshot_reports_allocation_delta -- --nocapture`

---

## Appendix A: Complexity Summary Table

| Operation Class | pandas Best | pandas Typical | FP Best | FP Typical | Notes |
|----------------|------------|---------------|---------|-----------|-------|
| Point lookup (sorted) | O(log n) | O(log n) | O(log n) | O(log n) | Parity |
| Point lookup (unsorted) | O(1)* | O(1)* | O(n) | O(n) | *After O(n) IndexEngine build |
| Union alignment | O(n + m) | O(n + m) | O(n + m) | O(n + m) | Parity (both use hash) |
| Binary add (same idx) | O(n) | O(n) | O(n) | O(n) | Parity |
| Binary add (diff idx) | O(n + m) | O(n + m) | O(n + m) | O(n + m) | Parity |
| GroupBy sum (Cython/dense) | O(n) | O(n) | O(n) | O(n) | FP dense path for Int64 keys |
| GroupBy sum (generic) | O(n) | O(n) | O(n) | O(n) | Parity (hash-based) |
| GroupBy sum (object/EA) | O(n * g) | O(n * g) | O(n) | O(n) | FP wins: no Python fallback |
| Hash join (unique keys) | O(n + m) | O(n + m) | O(n + m) | O(n + m) | Parity |
| Hash join (many-to-many) | O(n * m) | O(output) | O(n * m) | O(output) | Parity (output-dominated) |
| Outer join unmatched check | O(n + m) | O(n + m) | O(n * m) | O(n * m) | FP worse: linear scan (Gap G8) |
| K-way alignment | O(K * N) | O(K * N) | O(N log K) | O(N log K) | FP wins: leapfrog (AG-11) |
| CSV parse | O(bytes) | O(bytes) | O(bytes) | O(bytes) | Parity (both single-pass) |
| Construction (dict) | O(n * m) | O(n * m) | O(n * m) | O(n * m) | FP: no consolidation cost |
| Column access | O(1) | O(1) | O(log m) | O(log m) | BTreeMap vs array index |

---

## Appendix B: Memory Growth Summary

| Data Structure | pandas Footprint | FP Current | FP Future (G5) | Ratio |
|---------------|-----------------|------------|----------------|-------|
| Float64 column (n rows) | 8n bytes | 24n bytes | 8n + n/8 bytes | 3x -> 1.02x |
| Int64 column (n rows) | 8n bytes | 24n bytes | 8n + n/8 bytes | 3x -> 1.02x |
| Bool column (n rows) | n bytes | 24n bytes | n/8 + n/8 bytes | 24x -> 0.25x |
| String column (n rows, avg L) | (49+L)*n bytes | (24+L+24)*n bytes | (L+8)*n + n/8 bytes | ~1x -> ~0.5x |
| Validity mask (n elements) | n bytes (bool) | n/8 bytes (bitvec) | n/8 bytes | 0.125x |
| Index (n Int64 labels) | 8n + 16n (engine) | 8n + 24n (labels) | 8n + 16n (cached map) | ~1x |
| Index (n String labels) | (49+L)*n + 16n | (24+L)*n | (24+L)*n + 16n | ~0.8x |
| GroupBy intermediate (g groups) | 8g * accumulators | 96g (HashMap entry) | 96g | ~8-12x overhead |
| Join hash table (m right rows) | 32m bytes | 56m bytes | 56m bytes | ~1.75x |

---

## Appendix C: Source File Reference Index

### Pandas Performance-Critical Files

| File | LOC | Hotspot Family | Key Functions |
|------|-----|---------------|---------------|
| `_libs/groupby.pyx` | 2,325 | H1 | `group_sum`, `group_mean`, `group_var`, `group_rank` |
| `_libs/algos.pyx` | 1,445 | H1 | `rank_1d`, `take_2d`, `groupsort_indexer`, `kth_smallest` |
| `_libs/hashtable.pyx` + templates | 1,700+ | H2 | `Int64HashTable`, `factorize`, `unique`, `duplicated` |
| `_libs/index.pyx` | 1,325 | H2 | `IndexEngine.get_loc`, `DatetimeEngine`, binary search |
| `_libs/join.pyx` | 880 | H1, H3 | `left_join_indexer`, `inner_join_indexer`, `outer_join_indexer` |
| `_libs/lib.pyx` | 3,325 | H4 | `maybe_convert_objects`, `infer_dtype`, `no_default` |
| `_libs/parsers.pyx` | 2,182 | H7 | `TextReader`, C parser bridge |
| `_libs/src/parser/` | 2,255 (C) | H7 | C tokenizer, `strtod`, field parsing |
| `_libs/window/aggregations.pyx` | 2,118 | H1 | `roll_sum`, `roll_mean`, `roll_var`, `roll_median_c` |
| `core/nanops.py` | 1,777 | H1, H4 | `nansum`, `nanmean`, `nanvar`, `nanstd` |
| `core/internals/managers.py` | 2,500+ | H3 | `reindex_indexer`, `_consolidate`, `operate_blockwise` |
| `core/reshape/merge.py` | 2,200+ | H2, H3 | `get_join_indexers`, `_factorize_keys` |
| `core/groupby/groupby.py` | 3,000+ | H5 | `_cython_agg_general`, `_agg_py_fallback` |
| `core/dtypes/cast.py` | 2,282 | H4 | `find_common_type`, `np_can_hold_element` |

### FrankenPandas Performance-Critical Files

| File | LOC (approx) | AG Optimizations | Key Functions |
|------|-------------|-----------------|---------------|
| `fp-index/src/lib.rs` | 1,613 | AG-02, AG-11, AG-13 | `Index::position`, `align_union`, `leapfrog_union`, `leapfrog_intersection` |
| `fp-columnar/src/lib.rs` | 800+ | AG-03, AG-04, AG-10 | `ValidityMask`, `vectorized_binary_f64`, `Column::binary_numeric` |
| `fp-groupby/src/lib.rs` | 700+ | AG-05, AG-06, AG-08 | `groupby_sum_with_trace`, `try_groupby_sum_dense_int64`, `groupby_agg` |
| `fp-join/src/lib.rs` | 500+ | AG-02, AG-06 | `join_series_with_trace`, `merge_dataframes` |
| `fp-frame/src/lib.rs` | 1,000+ | AG-05 | `binary_op_with_policy`, `DataFrame::from_series` |
| `fp-io/src/lib.rs` | 300+ | AG-07 | `read_csv_str`, Vec-based accumulation |
| `fp-expr/src/lib.rs` | 200+ | AG-15 | `evaluate`, `MaterializedView::apply_deltas` |
| `fp-types/src/lib.rs` | 200+ | -- | `Scalar`, `DType`, `cast_scalar_owned` |

---

## Appendix D: Glossary

| Term | Definition |
|------|-----------|
| **AACE** | Alignment-Aware Columnar Execution -- FP's core execution model |
| **AG** | Optimization round identifier (AG-02 through AG-15) |
| **Arena allocation** | Bump allocator (`bumpalo`) for batch allocation/deallocation |
| **BlockManager** | pandas' internal columnar storage engine; groups columns by dtype into 2D blocks |
| **Borrowed-key HashMap** | HashMap keyed by references (`&IndexLabel`) instead of owned values, avoiding clones |
| **Consolidation** | BlockManager operation that merges same-dtype blocks into single contiguous arrays |
| **Dense-int path** | FP optimization for Int64 keys with small span: uses array-indexed accumulators instead of HashMap |
| **GroupKeyRef** | FP's borrowed-reference enum for group keys, avoiding per-group Scalar cloning |
| **IVM** | Incremental View Maintenance -- recomputing only changed portions of derived data |
| **Leapfrog triejoin** | K-way merge algorithm using sorted iterators and a min-heap |
| **OnceCell** | Rust's lazy-initialization cell; computes value on first access, caches forever |
| **ValidityMask** | Bit-packed boolean array tracking which column positions hold valid (non-null) data |
| **Vectorized path** | FP's AG-10 optimization: operating on `&[f64]` slices instead of `Vec<Scalar>` for SIMD |
