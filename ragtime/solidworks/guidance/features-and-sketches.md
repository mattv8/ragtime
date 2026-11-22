# Feature tree, sketches and dimensions

## Operations

**`features.list`** `document`, `include_system` (default false; hides folders such as Comments
and Sensors), `types` (optional list of feature type names to keep). Paged, in tree order.
Items: `name`, `type`, `id`, `suppressed`, `error_code`, `warning`, `dimension_count`,
and `parent_names` when present. Use `error_code` and `warning` to find trouble quickly.

**`feature.get`** `document`, `feature` (the feature name). Result: `name`, `type`, `id`,
`suppressed`, `error_code`, `warning`, `whats_wrong`, `parents` (names), `children` (names),
`dimensions` (dimension items), and `sketch` (the sketch name, when the feature has one).
`whats_wrong` is SolidWorks' own explanation; quote it rather than paraphrasing.

**`sketch.get`** `document`, `sketch` (the sketch name), `part` (optional):
- `summary` (default): `name`, `is_3d`, `constrained_status`, `segment_count`, `point_count`,
  `relation_count`, `dimension_count`, `plane_or_face`, `units`.
- `segments`, `points`, `relations`, `dimensions`: paged lists.
- Segment items: `id`, `type` (`line`, `arc`, `circle`, `ellipse`, `spline`, `parabola`, `text`,
  `other`), `construction`, `length`, and where relevant `start`, `end`, `center` ([x, y, z]),
  `radius`. Coordinates are in sketch space, metres.
- Relation items: `type` and `entities` (each `{kind, id}`), which name the segments or points
  they constrain.

**`dimensions.list`** `document`, `feature` (optional feature name). Paged. Items: `full_name`,
`name`, `feature`, `value`, `type` (`linear`, `angular`, `radial`, `diameter`, `other`), `driven`,
`read_only`, `tolerance` (`{type, max, min}`, optional), `units`. `full_name` looks like
`D1@Boss-Extrude1@Part1.Part`; use it exactly when you refer to one dimension. Lengths are metres
and angles radians.

## Reading the data

- `constrained_status` is SolidWorks' own word for the sketch state (for example fully defined,
  under defined, over defined). Report it as given. Under defined means some geometry can still
  move; over defined means redundant constraints exist.
- `driven` dimensions report a value but do not control the geometry. Do not treat them as inputs.
- Sketch coordinates are local to the sketch plane or face (`plane_or_face`). Two sketches
  with the same numbers may be in different places in the model.
- Segment lengths and coordinates are in metres. Convert before quoting.
- `error_code` on a feature is an error; `warning` is a lesser issue. Both are set by SolidWorks.

## Recipes

**Why is this sketch under-defined?**
1. `sketch.get` with `part` `summary`: confirm `constrained_status`, `dimension_count`,
   `relation_count`.
2. `sketch.get` with `part` `segments`: find the lines and arcs. Report those whose `start` or
   `end` is not pinned.
3. `sketch.get` with `part` `relations`: a segment with no relation pinning it is the likely
   free geometry. Name the segments by `id` in your answer.

**Which features have rebuild errors?**
1. `model.summary`: `features_with_errors` and `features_with_warnings` give the counts.
2. `features.list` (paged, `types` optional): read the items with `error_code` or `warning`.
3. `feature.get` for each one, to quote `whats_wrong`.

**What drives this dimension?**
1. `dimensions.list` with `feature` set to the feature that owns it, or search the list for
   the `full_name`.
2. Check `driven` and `read_only`. A driven or read-only dimension is set by something else,
   usually an equation or another dimension. Use `equations.list` to see the equations.

**Walk from a feature to its sketch geometry.**
1. `feature.get` for the feature: note `sketch` and `parents`.
2. `sketch.get` with `sketch` set to that name, `part` `summary`, then `segments`.
3. For a feature with no `sketch` (for example a fillet), use `parents` to reach the sketch.
