# Assemblies, components and mates

Use these operations only on assemblies. On a part or drawing they return
`not_supported_for_document_type`. Check `type` in `documents.list` or `model.summary` first.

## Operations

**`components.list`** `document` (assembly), `top_level_only` (default true), `only_problems` (default
false), paged. Items: `name`, `path`, `configuration`, `suppressed`, `hidden`, `fixed`, `is_virtual`,
`depth`, `parent`, `transform`, `children_count`, `file_exists`, `suppression`.
- `top_level_only` true lists only the direct children of the assembly. Set it to false to
  walk the whole tree; use `depth` and `parent` to rebuild the nesting.
- `file_exists` is true when the referenced file is on this machine. `suppression` is `resolved`,
  `lightweight`, `suppressed` or `missing_file`. `missing_file` means the file does not exist; the
  other values come from SolidWorks.
- `only_problems` true lists only the components whose file is missing.
- `transform` is 16 numbers in SolidWorks `MathTransform.ArrayData` order, in the assembly frame:
  - elements 0-8: the 3x3 rotation matrix, row by row (0-2 first row, 3-5 second, 6-8 third);
  - elements 9-11: the translation (x, y, z) in metres;
  - element 12: the scale factor;
  - elements 13-15: unused.
- `suppressed` components are switched off; `hidden` ones are not drawn. `fixed` components
  do not move under mates. `is_virtual` components are virtual copies.

**`mates.list`** `document` (assembly), `only_problems` (default false), paged. Items: `name`, `type`,
`alignment`, `suppressed`, `error_code`, `error`, `warning`, `whats_wrong`, `value` (optional),
`entities` (each `{component, component_path, entity_type, resolved}`).
- `error_code` is the raw SolidWorks value. `error` is `null` for a healthy mate, otherwise
  `{code, name, meaning}`: `name` is the `swFeatureError_e` name, or `unknown`, and `meaning` is one
  sentence. The table below lists the mate errors.
- `whats_wrong` is SolidWorks' own explanation, the same as `feature.get`. Quote it.
- `component` is empty and `entity_type` is `unknown` when SolidWorks cannot return the entity.
  `resolved` is false for such an entity, and for one whose reference component is missing.
- `component_path` joins the component names from the top assembly down, with `/`, for example
  `Sub-1/Bracket-2`. It is `null` when there is no path.
- `only_problems` true lists only mates with an error, a warning, or an entity with `resolved` false.
- A mate with `error` set cannot be satisfied as defined. Quote the error name and the entities.

## Mate error names

These are the `swFeatureError_e` mate values that `mates.list` reports. Other codes come back as
`unknown`; quote the raw `code`.

| Code | Name | Meaning |
|---|---|---|
| 38 | `swFeatureErrorMateInvalidEdge` | An edge of the mate is suppressed, invalid or no longer present. |
| 39 | `swFeatureErrorMateInvalidFace` | A face of the mate is suppressed, invalid or no longer present. |
| 40 | `swFeatureErrorMateFailedCreatingSurface` | The mating surface type is not supported. |
| 41 | `swFeatureErrorMateInvalidEntity` | An entity of the mate is suppressed, invalid or no longer present. |
| 42 | `swFeatureErrorMateUnknownTangent` | The tangent is not satisfied. |
| 43 | `swFeatureErrorMateDanglingGeometry` | The mate points to dangling geometry. |
| 44 | `swFeatureErrorMateEntityNotLinear` | Non-linear edges cannot be used for mating. |
| 45 | `swFeatureErrorMateEntityFailed` | Mating is not supported for a component, or a component cannot be modified. |
| 46 | `swFeatureErrorMateOverdefined` | The mate over-defines the assembly. |
| 47 | `swFeatureErrorMateIlldefined` | The mate cannot be solved as defined. |
| 48 | `swFeatureErrorMateBroken` | One or more mate entities were suppressed. |

A mate that over-defines the assembly shows as 46. A missing reference does not always show as a mate
error: in one checked assembly every mate reported 0 while 36 of 37 components were suppressed
because their files were in another vault view. A clean mate list does not prove the assembly is
healthy; check the components too. Repair steps are in `repair-assembly`.

## Notes

- A component is an instance of a file, not its own model. Its features, sketches and
  dimensions are not in this assembly's tree. To inspect them, open the component's file as
  its own document with `document.open` using its `path`, then use the feature operations.
  Close it afterwards.
- `path` is the component's file path. It is usable only if it is a local drive path;
  a UNC path is rejected by `document.open`.
- The same part may appear many times. Each entry is one instance, so count instances by
  grouping on `path` (and `configuration` when it matters).
- A component shown as suppressed may have its file in a different vault view, not on this
  machine. Check `file_exists` and `suppression` before you call it a deliberate suppression, and
  check the vault views in `pdm.status`.

## Recipes

**List the mates with errors.**
1. `model.summary`: check `mate_count`.
2. `mates.list` with `only_problems` true (paged): keep the items where `error` is set.
3. Report `name`, `type`, `error.name` and `whats_wrong`, and the components named in `entities`.

**Which components are missing their files?**
1. `model.summary`: `missing_component_count`.
2. `components.list` with `only_problems` true, and `top_level_only` false to see nested ones.
3. Report each `path`. Those files are not on this machine. Look for them with `pdm.search` if the
   linked vault may hold them, or name the vault they belong to.

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
