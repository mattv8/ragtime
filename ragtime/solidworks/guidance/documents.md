# Documents, properties and mass

## The `document` argument

Most operations take `document`. It accepts either:

- the `document_id` from `documents.list` or `document.open` (for example `"doc-1"`), or
- the full path of a document that is already open (for example `C:\Parts\bracket.sldprt`).

If you get `document_not_open`, call `document.open` with the path first.

## Operations

**`documents.list`** (paged, no arguments). Open documents in every instance. Key items:
`document_id`, `title`, `path`, `type` (`part`, `assembly`, `drawing`), `active_configuration`,
`opened_by_host`, `read_only`, `modified`.

**`document.open`** `path` (required), `configuration` (optional). Returns the document item.
`path` must be an absolute local drive path ending in `.sldprt`, `.sldasm` or `.slddrw`.
The document is opened read-only and silently. A missing file gives `document_not_found`;
UNC or relative paths give `invalid_arguments`.

**`document.close`** `document`. Closes a document the host opened (`opened_by_host` true) and
returns `{closed: true}`. It never saves. Use it when you are finished, because background
instances hold a licence. Documents the user opened are refused with `invalid_arguments`.

**`model.summary`** `document`. One call to see what the model is. Key fields:
`type`, `title`, `active_configuration`, `configuration_count`, `feature_count`,
`features_with_errors`, `features_with_warnings`, `material`, `custom_property_count`,
`needs_rebuild`, `units`. Parts also have `solid_body_count`; assemblies have `component_count`
and `mate_count`. Read this before any large listing.

**`configurations.list`** `document` (paged). Items: `name`, `is_active`, `description`, `parent`.

**`properties.get`** `document`, `configuration` (optional; omit for file-level properties).
Result: `{scope, properties: [{name, type, value, resolved_value}]}`.
- `value` is the property as stored in the file, which may be a formula or a reference.
- `resolved_value` is the value SolidWorks computes for it. Quote `resolved_value` to the user
  unless it is empty, then fall back to `value`.
- File-level properties (no `configuration`) are the document's own. Configuration properties
  override them for that configuration, so call once per configuration when they differ.

**`equations.list`** `document` (paged). Items: `index`, `equation`, `value`,
`is_global_variable`, `suppressed`. Equations are the model's named relations between dimensions.

**`mass_properties.get`** `document`. Result: `mass`, `volume`, `surface_area`, `density`,
`center_of_mass` ([x, y, z]), `principal_moments` ([3]), `moments_of_inertia` ([9]), `bounding_box` ([6], optional), `units`. Mass is kilograms, volume cubic metres,
area square metres, density kg/m^3, lengths metres. Convert for the user, for example
grams, cubic centimetres, or mm for the centre of mass.

## Recipes

**What is this part made of, and how heavy is it?**
1. `documents.list` or `document.open` to get a `document_id`.
2. `model.summary`: check `material`, `solid_body_count`, `needs_rebuild`.
3. `mass_properties.get`: report `mass` (kg) and `density`. If `mass` is missing or zero,
   say the model may need a rebuild or contains no solid body, and do not guess.

**Which configurations exist?**
1. `model.summary` for `configuration_count` and `active_configuration`.
2. `configurations.list` for the names, the active flag and each `description`.

**Read the custom properties for the BOM.**
1. `properties.get` with no `configuration` for file-level values.
2. For each configuration in `configurations.list`, call `properties.get` with
   `configuration` set to its `name`. Build one row per configuration, using `resolved_value`.
3. Report the properties that differ between configurations, and mention which values were
   taken from file-level properties.
