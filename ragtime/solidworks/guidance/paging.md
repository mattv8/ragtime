# Paging, snapshots and large results

Every `*.list` operation, and `sketch.get` with `part` set to anything other than `summary`,
returns one page:

```json
{"items": [], "offset": 0, "next_offset": null, "total": 0, "snapshot": "<hex>", "units": "SI: m, kg, s, rad"}
```

## Parameters

- `limit`: 1 to 500, default 100. Ask for the smallest page that answers the question.
- `offset`: where to start, counting items from 0. Use `next_offset` from the previous page.
- `next_offset`: the offset of the next page, or `null` when the list is finished.
- `total`: the number of items in the whole list. Use it to know how much is left.

## Byte budget

The SolidWorks agent (the Ragtime program on the Windows machine) also stops adding items when a page would exceed its size budget (about 32 KB of JSON).
So a page can hold fewer items than `limit` even when more exist. It is not an error. Always
follow `next_offset`, not `offset + limit`. A page always holds at least one item. If a single
item is too large, the call fails with `result_too_large`.

## Snapshot

`snapshot` is a hash of the whole list as it was when the page was made. To read a list in
several calls:

1. First call: no `snapshot` argument. Note the `snapshot` from the response.
2. Next calls: send the same `snapshot` with the next `offset`.
3. If the model changed in between, the SolidWorks agent returns `stale_snapshot` with the current hash.
   Restart the list from `offset` 0, without `snapshot`, and tell the user the model changed
   while you were reading.

Calls that omit `snapshot` simply get pages with no consistency check. Use the check when you
need every page to describe the same model.

## Large results

`result_too_large` means the response would not fit. Narrow the request instead of repeating it:

- `features.list`: set `types` to the feature types you need, or leave `include_system` false.
- `dimensions.list`: set `feature` to one feature.
- Any list: lower `limit`, then follow the pages.
- `sketch.get`: request one `part` (`segments`, `relations` or `dimensions`), not everything.
  Use `summary` first.
- `components.list`: keep `top_level_only` true unless you need the whole tree.

## Before you list

Call `model.summary` first. Its counts (`feature_count`, `configuration_count`,
`component_count`, `mate_count`, `custom_property_count`, `features_with_errors`) tell you how
many pages to expect, and whether a listing is worth the time. For a large model, say so to the
user and summarise from the first pages or the counts, rather than fetching everything.

## Recipe: read a whole list safely

1. `model.summary`, then the relevant count.
2. Call the list with `limit` 100 (or less if the items are large). Keep `snapshot` and
   `total`.
3. While `next_offset` is not null, call again with `offset` set to `next_offset` and the
   same `snapshot`.
4. On `stale_snapshot`, start again from step 2 without `snapshot`.
5. Stop after a sensible number of pages if the user only needs a sample, and say the result
   is partial (`next_offset` was not null).
