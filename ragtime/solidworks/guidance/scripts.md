# Scripts: C# 5 syntax to avoid

Do not use `$"…"` string interpolation, `?.`, `nameof`, expression-bodied members,
`out var`, auto-property initializers, `using static`, tuples, or pattern matching.
Scripts compile as C# 5 against the assemblies shipped with the SolidWorks agent (the Ragtime program on the Windows machine).
Source is limited to 16 KB UTF-8; keep scripts small; split big jobs into several runs.

Use `public static class RagtimeScript` with `public static object Run(ScriptContext ctx)`.
The context exposes `App` (the SolidWorks application), `Arguments` (a JSON object),
`WorkFolder`, `Log(string)`, `NewWorkPath(string)` for a unique work-folder path,
and `DefaultTemplate(int)` to find an installed template or return null.
Activate the target document with `ActivateDoc3` before sketch or feature calls.
Use silent options and never show dialogs. Return plain JSON values, never SolidWorks
objects. Prefer bulk APIs for heavy work to avoid excessive COM calls.

Scripts are not sandboxed; they run with the permissions of the Windows account the SolidWorks agent runs as. Content protection
reviews each script when enabled; surface-level protection may review the call as well.
If a script times out, it may have partly changed open documents: re-read with
`documents.list` before retrying. Compile errors include line, column and code; common
modern-syntax errors include a C# 5 hint. Check the indicated source line.
Run time is limited by the tool's timeout setting (default 100 s) and by 600 s, whichever is lower.
The script gets at least 15 s less than the requested timeout, kept for the connection and the cleanup afterwards.
Compiling and attaching come out of what is left, and a cold SolidWorks start costs about 10 s more. Request at
least 30 s (`timeout` 30 leaves roughly 10 s for `Run`); shorter requests are refused with "Not enough time remains".

Build flow: run the script, save into `work_folder`, then use `pdm.file.add` and
`pdm.file.checkin`. Do not call PDM APIs directly from scripts.

Editing an existing model: a script may open a copy in `work_folder` read-write (`OpenDoc6` with
`swOpenDocOptions_Silent`, without the read-only option), change it and save it. Make the copy with
`document.copy` first. Save only that copy. Never save a file under a vault view (a folder in a
vault's local view, or any view listed by `pdm.status`), and never save over a source file. Pass
the copy path in `arguments` and read it with `(string)ctx.Arguments["path"]`. The complete repair
scripts are in `repair-assembly`.

Activate the copy with `ctx.App.ActivateDoc3` after `OpenDoc6`, and close it with `ctx.App.CloseDoc` in a
`finally` block, so a failed run leaves no open copy behind.

When scripts are off, `script.run` is refused with `scripts_not_allowed`. Read `error.details`, tell
the user what an admin can change, and do not retry.

## Example: 20 × 30 × 10 mm block

```csharp
using System;
using Ragtime.SolidWorks.Scripting;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

public static class RagtimeScript
{
    public static object Run(ScriptContext ctx)
    {
        string template = ctx.DefaultTemplate((int)swDocumentTypes_e.swDocPART);
        if (template == null) throw new InvalidOperationException("No part template found.");
        ModelDoc2 model = (ModelDoc2)ctx.App.NewDocument(template, 0, 0, 0);
        if (model == null) throw new InvalidOperationException("Part creation failed.");
        int activationError = 0;
        ctx.App.ActivateDoc3(model.GetTitle(), false, 0, ref activationError);
        if (activationError != 0) throw new InvalidOperationException("Activation failed.");
        if (!model.Extension.SelectByID2("Front Plane", "PLANE", 0, 0, 0, false, 0, null, 0))
            throw new InvalidOperationException("Front Plane was not found.");
        model.SketchManager.InsertSketch(true);
        model.SketchManager.CreateCornerRectangle(0, 0, 0, 0.02, 0.03, 0);
        model.SketchManager.InsertSketch(true);
        Feature feature = model.FeatureManager.FeatureExtrusion2(
            true, false, false, 0, 0, 0.01, 0, false, false, false, false, 0, 0,
            false, false, false, false, true, true, true, 0, 0, false);
        if (feature == null) throw new InvalidOperationException("Extrusion failed.");
        string path = ctx.NewWorkPath("block.SLDPRT");
        int errors = 0;
        int warnings = 0;
        bool saved = model.Extension.SaveAs3(path, 0,
            (int)swSaveAsOptions_e.swSaveAsOptions_Silent, null, null, ref errors, ref warnings);
        if (!saved || errors != 0) throw new InvalidOperationException("Save failed: " + errors);
        ctx.Log("Saved a 20 x 30 x 10 mm block.");
        return path;
    }
}
```
