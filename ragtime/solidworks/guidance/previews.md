# Previews, exports and the 3D view

Two read-only operations give the user a picture or a file of the model. Neither changes or saves
the source document.

- `model.preview`: a PNG render. Arguments: `document` (a `doc-N` id or the full path of an open
  document, as in the other operations; optional when exactly one document is open), `view`
  (`isometric` by default, or `front`, `top`, `right`), `width` and `height` (256 to 2048;
  SolidWorks may use the graphics window size instead, and a warning says so). For a drawing it
  renders the sheet.
- `model.export`: a file. Arguments: `document` (as above), `format` (`step`, `stl`, `3mf`, `glb`
  or `native`), `configuration` (optional) and `include_preview` (true to render a preview image
  with the file).

```json
{"operation": "model.preview", "arguments": {"document": "doc-1", "view": "front"}}
{"operation": "model.export", "arguments": {"document": "C:\\Parts\\bracket.sldprt", "format": "step"}}
```

## What the result gives you

`result.artifact` describes the main file: `page_url`, `content_url`, `file_name`, `format`,
`size_bytes`, `expires_at`. For a preview it also has `preview_image_url`. `result.artifacts` lists any
other files, and `result.warnings` may explain a skipped step. The result never contains a local path.

## When to call `model.preview`

- After each build step that changes the geometry, so the user sees design progress. Ragtime chat
  shows the image in the conversation, and the model sees the latest preview image too.
- Start with `isometric`. Use `front`, `top` or `right` when the user asks about a profile, a hole
  position or a dimension.
- Do not call it again for the same model state. If nothing changed, the picture is the same.

## When to call `model.export`

- At milestones: a finished part, an approved assembly, or when the user asks for a file. Not after
  every step.
- `step` for exchange with other CAD systems. `stl` or `3mf` for printing or meshes. `glb` for fast
  3D viewing.
- `native` gives the SolidWorks file itself. A part or drawing comes as a copy of its file. An
  assembly comes as a `zip` with the assembly and the files it references, in the same folder
  structure.
- Drawings support only `native` and `model.preview`. Any other format returns `invalid_arguments`.
- Assemblies support `step`, `3mf`, `glb` and `native`, not `stl` (SolidWorks writes one STL per component).
- The document must be open. In attach mode, the user must have it open. In background mode, open it
  with `document.open` first.
- Files are kept for 7 days (`expires_at`). After that the link stops working; export again if the
  user still needs the file.
- A file over 100 MB is refused. Export a part on its own, or ask the user what to leave out.

## Errors

- `export_failed`: SolidWorks could not write that format, for example `glb` or `3mf` on an older
  SolidWorks. Try `step`, or `stl` for a part.
- `preview_unavailable`: SolidWorks could not render a picture, often in background mode. Export a
  file instead, or ask the user to open the model in attach mode.
- `artifact_too_large`: over 100 MB. `artifact_transfer_failed`: the file could not be copied to
  Ragtime or failed its check. Report it; retry once at most.

## Telling the user

- Give the user the `page_url` and say what it is: the model's page, with the 3D view, file facts and
  download. Downloads and the 3D view need Ragtime sign-in, so say so.
- Do not paste file paths or `content_url` values into the answer. Say that the link or the card in
  the chat has the file.
- If a preview or link says it has expired, say so and export the file again.
- Clients outside Ragtime chat may show the picture inline and offer the link instead.
- Do not call `model.preview` or `model.export` in a loop. Repeat a call only after the model changed.
