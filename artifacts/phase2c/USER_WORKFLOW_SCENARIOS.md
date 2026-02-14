# User Workflow Scenario Corpus + Golden Journeys

User-centric E2E workflow scenarios reflecting realistic data processing patterns.
Frozen per bead **bd-2gi.20**.

---

## 1. Purpose

This corpus defines realistic user workflows that span multiple FrankenPandas operations. Each scenario mirrors a common pandas usage pattern and serves as:

1. An E2E regression test target
2. A compatibility validation surface
3. A performance benchmark anchor
4. A documentation aid for downstream consumers

Scenarios are numbered UW-NNN and organized by domain.

---

## 2. Scenario Format

Each scenario specifies:

| Field | Description |
|---|---|
| **ID** | `UW-NNN` |
| **Title** | Short descriptive name |
| **Domain** | Analytics, ETL, Finance, Scientific, etc. |
| **Operations** | Ordered list of FrankenPandas operations |
| **Packets** | Which FP-P2C-NNN packets are exercised |
| **Input** | Description of input data shape |
| **Expected** | Expected final output characteristics |
| **Edge Cases** | Nulls, duplicates, type coercion, etc. |
| **Golden Journey** | Exact sequence of API calls with expected intermediate states |

---

## 3. Scenarios

### UW-001: CSV Load + Column Arithmetic + Export

**Domain**: Analytics
**Operations**: `read_csv_str` -> `Series::add` -> `write_csv_string`
**Packets**: FP-P2C-001, FP-P2C-008

**Story**: User loads a CSV with sales data, computes a total column by adding two numeric columns, then exports the result.

**Golden Journey**:
```
1. read_csv_str("product,q1,q2\nA,100,200\nB,150,\nC,,300\n")
   -> DataFrame with 3 rows, 3 columns
   -> q1: [Int64(100), Int64(150), Null]
   -> q2: [Int64(200), Null, Int64(300)]

2. Extract q1 and q2 as Series, compute q1.add(&q2)
   -> Union index alignment (all labels match)
   -> Result: [Float64(300.0), NaN, NaN]
   -> Null propagation: missing + value = NaN

3. write_csv_string(&result_frame)
   -> Valid CSV with null cells as empty strings
```

**Edge Cases**:
- Missing values in numeric columns -> NaN propagation
- Mixed int/null columns -> Float64 coercion via common_dtype

---

### UW-002: Multi-Source Join + GroupBy Aggregation

**Domain**: Analytics
**Operations**: `join_series` (inner) -> `groupby_sum`
**Packets**: FP-P2C-004, FP-P2C-005

**Story**: User joins sales data with product categories, then groups by category to compute total sales.

**Golden Journey**:
```
1. sales = Series::from_values("sales", [A, B, C, A], [100, 200, 150, 50])
   categories = Series::from_values("cat", [A, B, D], ["electronics", "clothing", "food"])

2. join_series(&sales, &categories, JoinType::Inner)
   -> Inner join on index: only A, B match
   -> Left values: [100, 200, 50] (A appears twice in sales)
   -> Right values: ["electronics", "clothing", "electronics"]

3. groupby_sum on joined result by category
   -> electronics: 150 (100 + 50)
   -> clothing: 200
```

**Edge Cases**:
- Duplicate keys in left (A appears twice) -> multiplied cardinality
- Unmatched key (D in categories, C in sales) -> dropped in inner join
- Left join variant preserves all left rows with NaN for unmatched

---

### UW-003: Index Alignment with Mismatched Labels

**Domain**: Finance
**Operations**: `Series::add` with union alignment
**Packets**: FP-P2C-001, FP-P2C-003

**Story**: User adds two time series with different trading days. Missing days produce NaN.

**Golden Journey**:
```
1. portfolio_a = Series::from_values("a", [Mon, Tue, Wed], [100, 110, 105])
   portfolio_b = Series::from_values("b", [Tue, Wed, Thu], [200, 210, 220])

2. portfolio_a.add(&portfolio_b)
   -> Union index: [Mon, Tue, Wed, Thu]
   -> Values: [NaN, 310, 315, NaN]
   -> Mon: only in a -> NaN (no b value)
   -> Thu: only in b -> NaN (no a value)
```

**Edge Cases**:
- Completely disjoint indices -> all NaN result
- Identical indices -> element-wise addition, no nulls
- Single overlapping label -> one valid value, rest NaN

---

### UW-004: CSV with Complex Types + Round-Trip Stability

**Domain**: ETL
**Operations**: `read_csv_str` -> type inference -> `write_csv_string` -> `read_csv_str`
**Packets**: FP-P2C-001, FP-P2C-008

**Story**: User loads a CSV with mixed types, processes it, and verifies the output CSV can be re-ingested identically.

**Golden Journey**:
```
1. read_csv_str("id,name,score,active\n1,Alice,95.5,true\n2,Bob,,false\n3,,100,true\n")
   -> id: Int64, name: Utf8, score: Float64, active: Bool
   -> Row 2: score is Null(NaN) in Float64 column
   -> Row 3: name is Null(NaN) in Utf8 column

2. write_csv_string(&frame)
   -> CSV output with empty strings for nulls

3. read_csv_str(output)
   -> Semantically equal to original frame
   -> Column::semantic_eq() passes for all columns
```

**Edge Cases**:
- Bool coercion: `true`/`false` parsed as Bool, not Utf8
- Empty field -> Null, not empty string
- Unicode in string fields preserved through round-trip
- Trailing newline presence/absence produces identical frame

---

### UW-005: Duplicate Index Detection + Policy Enforcement

**Domain**: Data Validation
**Operations**: `Index::has_duplicates` -> `Series::add` with policy
**Packets**: FP-P2C-002, FP-P2C-003

**Story**: User detects duplicate labels in their index and handles them according to runtime policy.

**Golden Journey**:
```
1. index = Index::new([A, B, A, C])
   index.has_duplicates() -> true

2. series_with_dups = Series::from_values("x", [A, B, A, C], [1, 2, 3, 4])
   other = Series::from_values("y", [A, B], [10, 20])

3. Strict mode: series_with_dups.add(&other)
   -> Err(DuplicateIndexUnsupported)
   -> Evidence ledger records rejection

4. Hardened mode: series_with_dups.add_with_policy(&other, &hardened_policy, &mut ledger)
   -> Allowed with evidence record
   -> Evidence ledger captures decision with mode=hardened
```

**Edge Cases**:
- All labels identical -> duplicates detected
- Empty index -> no duplicates
- Single label -> no duplicates

---

### UW-006: GroupBy with Null Keys + DropNA Behavior

**Domain**: Analytics
**Operations**: `groupby_sum` with `drop_na: true` vs `drop_na: false`
**Packets**: FP-P2C-005

**Story**: User groups data containing null keys and controls whether null groups appear in output.

**Golden Journey**:
```
1. keys = Series::from_values("key", [0,1,2,3], [A, Null, B, A])
   values = Series::from_values("val", [0,1,2,3], [10, 20, 30, 40])

2. groupby_sum(keys, values, {drop_na: true})
   -> Groups: A=50, B=30
   -> Null key group dropped

3. groupby_sum(keys, values, {drop_na: false})
   -> Groups: A=50, Null=20, B=30
   -> Null key group preserved
```

**Edge Cases**:
- All keys null + drop_na=true -> empty result
- NaN as key vs Null as key -> both treated as missing
- Single-element groups -> sum equals the element

---

### UW-007: Left Join with Missing Values

**Domain**: ETL
**Operations**: `join_series` (left)
**Packets**: FP-P2C-004

**Story**: User left-joins a master list with a lookup table. Unmatched rows get NaN.

**Golden Journey**:
```
1. master = Series::from_values("ids", [1, 2, 3, 4], [A, B, C, D])
   lookup = Series::from_values("prices", [1, 3], [9.99, 29.99])

2. join_series(&master, &lookup, JoinType::Left)
   -> Index: [1, 2, 3, 4] (all left labels preserved)
   -> Left values: [A, B, C, D]
   -> Right values: [9.99, NaN, 29.99, NaN]
   -> Labels 2 and 4 have no match -> NaN
```

**Edge Cases**:
- All right keys match -> no NaN injected
- No right keys match -> all right values NaN
- Duplicate keys in right -> multiplied cardinality

---

### UW-008: Index First Positions Lookup

**Domain**: Data Engineering
**Operations**: `Index::first_positions`
**Packets**: FP-P2C-002

**Story**: User looks up the first occurrence position of each target label in an index.

**Golden Journey**:
```
1. source = Index::new([A, B, C, A, D])
   targets = [C, A, E]

2. source.first_positions(&targets)
   -> [Some(2), Some(0), None]
   -> C first at position 2
   -> A first at position 0 (not 3)
   -> E not found -> None
```

**Edge Cases**:
- Empty source index -> all None
- Empty target list -> empty result
- All targets present -> all Some
- Duplicate targets in target list -> each resolved independently

---

### UW-009: Multi-Column DataFrame Construction

**Domain**: Scientific
**Operations**: `DataFrame::from_series` with mismatched indices
**Packets**: FP-P2C-001

**Story**: User constructs a DataFrame from multiple Series with different indices. Missing values are filled with NaN.

**Golden Journey**:
```
1. temp = Series::from_values("temp", [Mon, Tue, Wed], [20.5, 22.1, 19.8])
   humidity = Series::from_values("humidity", [Tue, Wed, Thu], [65, 70, 80])

2. DataFrame::from_series(vec![temp, humidity])
   -> Union index: [Mon, Tue, Wed, Thu]
   -> temp column: [20.5, 22.1, 19.8, NaN]
   -> humidity column: [NaN, 65, 70, 80]
```

**Edge Cases**:
- Single series -> DataFrame with that series's index
- Empty series list -> empty DataFrame
- All series share same index -> no NaN fill needed

---

### UW-010: Pipeline: CSV -> Filter -> GroupBy -> Export

**Domain**: Reporting
**Operations**: `read_csv_str` -> column extraction -> `groupby_sum` -> `write_csv_string`
**Packets**: FP-P2C-001, FP-P2C-005, FP-P2C-008

**Story**: User loads a CSV report, groups by a category column, sums a value column, and exports the aggregated result.

**Golden Journey**:
```
1. read_csv_str("region,product,sales\nEast,Widget,100\nWest,Widget,200\nEast,Gadget,150\nWest,Gadget,50\n")
   -> DataFrame with 4 rows, 3 columns

2. Extract "region" as keys, "sales" as values
   -> keys: Series with [East, West, East, West]
   -> values: Series with [100, 200, 150, 50]

3. groupby_sum(keys, values, {drop_na: true})
   -> East: 250
   -> West: 250

4. write_csv_string(&result)
   -> "region,sales\nEast,250\nWest,250\n"
```

**Edge Cases**:
- Empty CSV (headers only) -> empty groupby -> empty result
- All same region -> single group
- Null in region column -> dropped if drop_na=true

---

## 4. Golden Journey Test Integration

Each scenario maps to one or more conformance fixtures. The mapping:

| Scenario | Fixture Pattern |
|---|---|
| UW-001 | `fp_p2c_001_*` + `fp_p2c_008_*` |
| UW-002 | `fp_p2c_004_*` + `fp_p2c_005_*` |
| UW-003 | `fp_p2c_001_*` + `fp_p2c_003_*` |
| UW-004 | `fp_p2c_001_*` + `fp_p2c_008_*` |
| UW-005 | `fp_p2c_002_*` + `fp_p2c_003_*` |
| UW-006 | `fp_p2c_005_*` |
| UW-007 | `fp_p2c_004_*` |
| UW-008 | `fp_p2c_002_*` |
| UW-009 | `fp_p2c_001_*` |
| UW-010 | `fp_p2c_001_*` + `fp_p2c_005_*` + `fp_p2c_008_*` |

### Scenario Coverage Matrix

| Packet | Scenarios |
|---|---|
| FP-P2C-001 | UW-001, UW-003, UW-004, UW-009, UW-010 |
| FP-P2C-002 | UW-005, UW-008 |
| FP-P2C-003 | UW-003, UW-005 |
| FP-P2C-004 | UW-002, UW-007 |
| FP-P2C-005 | UW-002, UW-006, UW-010 |
| FP-P2C-008 | UW-001, UW-004, UW-010 |

---

## 5. Workflow Complexity Tiers

| Tier | Description | Scenarios |
|---|---|---|
| **Simple** | Single operation, no alignment | UW-008 |
| **Standard** | 2-3 operations, basic alignment | UW-001, UW-003, UW-005, UW-006, UW-007, UW-009 |
| **Pipeline** | Multi-step with I/O | UW-002, UW-004, UW-010 |

Each tier has different reliability expectations:
- **Simple**: 100% pass rate, no flakes
- **Standard**: 100% pass rate, deterministic null handling
- **Pipeline**: 100% pass rate, round-trip stability

---

## 6. Non-Goals (Excluded Workflows)

These patterns are explicitly out of scope for Phase-2C:

| Pattern | Reason |
|---|---|
| Multi-threaded DataFrame access | No concurrency model yet |
| In-place mutation (`.iloc[0] = x`) | Copy-on-write design |
| String method chains (`.str.upper()`) | Not in MVP slice |
| DateTime operations | No datetime dtype yet |
| Hierarchical indexing (MultiIndex) | Single-level index only |
| Pivot tables | Composite of unimplemented operations |
| Window functions (rolling, expanding) | Not in MVP slice |

---

## Changelog

- **bd-2gi.20** (2026-02-14): Initial corpus with 10 user workflow scenarios covering CSV I/O, arithmetic alignment, joins, groupby, index operations, round-trip stability, and multi-step pipelines. Includes scenario-to-packet mapping, coverage matrix, and complexity tiers.
