# Find a model

Identifiers differ by system. A catalog or product number (SKU) may be an ERP number, not a file
name. The vault may use internal part numbers (for example `22-100-xxxxx`), custom copies named
`CU-<catalog>-<yy><letter>-<rev>`, and quoting folders that hold only PDF, DWG or XLSM files. Try
each form you have. This routine only reads.

## Routine

1. `pdm.lookup` with `filename`, and with `part_number` if you have one. Set `document_type` to
   `SLDASM` for an assembly or `SLDPRT` for a part. Lookup matches substrings, and exact file name
   matches come first. It returns 25 items at most; follow `next_offset` with `offset`.
2. No hit: search the base number without the trailing variant (`IDB0061-19` becomes `IDB0061`),
   then the custom form `CU-<base>`. Use each with `document_type` `SLDASM`, then `SLDPRT`. Page with
   `offset` and `next_offset`.
3. `pdm.search` with the same words, for family names and descriptions that lookup does not match.
   An item with `match` set to `filename` is a file-name prefix hit. A `semantic` item came from
   meaning, so check it before you rely on it.
4. A hit with a different variant is a candidate, not a confirmed match. For example,
   `CU-IDB0061-26A-A` for `IDB0061-19` is a candidate. Report it, and ask the user to confirm before
   you act on it.
5. Check the candidate's references from `document.open` rather than by walking components. Fetch
   it with `pdm.file.get_latest`, open it with `document.open`, and read `load_warnings` and
   `missing_references`. Those name the referenced paths that are not on disk. Call `components.list`
   with `only_problems` true only when you need component names. A reference under a vault view
   that is not the linked vault is not in this vault: name that vault and stop with a precise
   question to the user.
6. No hit anywhere: the model may be in another vault that this tool does not link. Read the views
   in `pdm.status` before you browse folders. Name any view that is not the linked vault or has
   `logged_in` false, and ask the user. Browse only if the search points at this vault.
7. Browse with `pdm.folder.list` on the product-line folders that the candidates suggest. Go down
   one level per call, and stop when the folder names no longer match.
8. The tool cannot look up ERP data. If another available tool can (an ERP or Odoo tool), use it to
   map the SKU to its BOM part numbers, then repeat steps 1 and 3 with those numbers. If none can,
   ask the user for the file name or part number.

## Limits

- The index can lag the vault. Confirm the chosen file with `pdm.file.info` before you report its
  state.

## Report

List every candidate you found. For each one give its vault path, its `matched_on`, and whether it is
the target, with the reason. If none matches, say so and name what you tried, including the views
you checked.
