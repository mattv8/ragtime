# Learn how existing models are built

Before you propose or change a model, read how the house already builds similar ones. This routine
only reads. It finds two or three comparable models in the vault, opens them and writes down the
conventions they share. Do not treat one model as the rule.

Use this routine to learn the conventions, then give the user the result. To build a model with
them, use `script.run` when the tool allows scripts (see `scripts`).

## 1. Choose the models

1. `pdm.search` with `query` set to the product family, a word from the description, or the
   part-number pattern (for example a prefix). Set `document_type` to `SLDPRT` for parts or
   `SLDASM` for assemblies.
2. Choose two or three comparable items. Prefer released ones and the latest `revision`. Check
   `workflow_state` with `pdm.file.info` in the next step. Say which models you chose and why.
3. Use the `configurations` and `bom_count` of the search items as a first hint about how varied
   each model is.

## 2. Get the files

For each chosen model:

1. `pdm.file.info` for `vault_path` (from `vault_paths[0]`). Note `workflow_state` and
   `latest_version`.
2. `pdm.file.get_latest` with `include_references` false. For an assembly, keep the default true
   only if you must open its parts, and study one or two assemblies at most. A file skipped as
   `local_modified` or `local_untracked` keeps its local copy; open that copy and say so.
3. `document.open` with `path` set to `file.local_path`.

## 3. Read in this order

Read each open model with the operations below, in this order. Use `paging` rules for every list.

1. **`model.summary`**: `units`, `feature_count`, `features_with_errors`, `custom_property_count`.
   Look for the unit system. A model with errors is a weak example; say so.
2. **`configurations.list`**: how variants are organised and named (size steps, short and long
   forms, suffixes). Look for a naming pattern and whether every variant has a `description`.
3. **`properties.get`**: file level first, then `configuration` for each variant. Look for which
   custom properties exist, how they are named, and which change between configurations. Compare
   with the data-card values from **`pdm.variables.get`** (file level, then per configuration). Note
   any name that appears in one but not the other.
4. **`features.list`** (`include_system` false): feature names and their order. Look for custom
   names (a plain `Boss-Extrude1` tells you nothing) and for how the base feature, holes and
   fillets are ordered.
5. **`feature.get`** on the base feature and one more key feature, then **`sketch.get`** on the
   sketch with `part` `summary`. Look for how the sketch is constrained (`constrained_status`) and
   how many dimensions it has.
6. **`dimensions.list`** (filter by `feature` for the base feature) and **`equations.list`**. Look
   for dimensions driven by equations or global variables, and how the parameters are named
   (`full_name`).
7. **Assemblies only**: **`components.list`** with `top_level_only` true for the structure, the
   sub-assemblies, which components are `fixed`, and the configurations used. Then **`mates.list`**
   for the mate types and how mates are named.

## 4. Summarise before you propose

Write a short list of the conventions you found. Give evidence for each: the model file names and the
feature, property, dimension or equation names that show it. Use this form:

- `<convention>`: `<model A>`, `<model B>`; `<evidence>`.

If the models disagree, say where. When the disagreement changes what you would build, ask the user
which to follow.

## 5. Close what you opened

Call `document.close` for each document you opened (`opened_by_host` true). Never close documents the
user opened.

## Limits

- Reading is cheap, but each open takes seconds and holds a SolidWorks licence until it is closed.
- Drawings (`.slddrw`) open, but their feature data is not what this routine needs. Skip them.
- The vault index holds document metadata only: file names, configurations, descriptions and BOM
  counts. It does not hold feature data. Feature data comes only from opening the file.
- The index is a snapshot and can lag the vault. Confirm the live state with `pdm.file.info`.
- See `scripts` for the build, save, add and check-in flow.
