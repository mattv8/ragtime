# Assemblies, components and mates

Use these operations only on assemblies. On a part or drawing they return
`not_supported_for_document_type`. Check `type` in `documents.list` or `model.summary` first.

## Operations

**`components.list`** `document` (assembly), `top_level_only` (default true), paged. Items:
`name`, `path`, `configuration`, `suppressed`, `hidden`, `fixed`, `is_virtual`, `depth`,
`parent`, `transform`, `children_count`.
- `top_level_only` true lists only the direct children of the assembly. Set it to false to
  walk the whole tree; use `depth` and `parent` to rebuild the nesting.
- `transform` is 16 numbers in SolidWorks `MathTransform.ArrayData` order, in the assembly frame:
  - elements 0-8: the 3x3 rotation matrix, row by row (0-2 first row, 3-5 second, 6-8 third);
  - elements 9-11: the translation (x, y, z) in metres;
  - element 12: the scale factor;
  - elements 13-15: unused.
- `suppressed` components are switched off; `hidden` ones are not drawn. `fixed` components
  do not move under mates. `is_virtual` components are virtual copies.

**`mates.list`** `document` (assembly), paged. Items: `name`, `type`, `alignment`, `suppressed`,
`error_code`, `warning`, `value` (optional), `entities` (each `{component, entity_type}`).
A mate with `error_code` set cannot be satisfied as defined; quote the error code and the
entities involved.

## Notes

- A component is an instance of a file, not its own model. Its features, sketches and
  dimensions are not in this assembly's tree. To inspect them, open the component's file as
  its own document with `document.open` using its `path`, then use the feature operations.
  Close it afterwards.
- `path` is the component's file path. It is usable only if it is a local drive path;
  a UNC path is rejected by `document.open`.
- The same part may appear many times. Each entry is one instance, so count instances by
  grouping on `path` (and `configuration` when it matters).

## Recipes

**List the mates with errors.**
1. `model.summary`: check `mate_count`.
2. `mates.list` (paged): keep items where `error_code` is set or `warning` is set.
3. Report `name`, `type`, `error_code` and the components named in `entities`.

**Which components are fixed, or are candidates for under-constrained?**
1. `components.list` with `top_level_only` true.
2. Report the items with `fixed` true. Components that are not fixed and appear in no mate
   (check `entities` in `mates.list`) are the candidates to inspect. Say they are candidates,
   not confirmed problems.

**Where is component X positioned?**
1. `components.list` and find the item by `name` (paged; follow `next_offset` if needed).
2. Read elements 9-11 of `transform` as the position in metres (see above); convert it to
   millimetres for the user.

**Count instances of a part.**
1. `components.list` with `top_level_only` false, following every page.
2. Count items whose `path` matches the part. Report the total and the number of distinct
   configurations if relevant. Use `total` from the last page to confirm you read everything.
