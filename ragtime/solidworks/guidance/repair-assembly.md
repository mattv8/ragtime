# Repair an assembly on a copy

Diagnose first, without writing. Repair only a copy in the work folder. The source files, on disk
and in the vault, are never changed by this routine.

## 1. Diagnose (read only)

1. Get the files: `pdm.file.get_latest` on the assembly's vault `path` (references included). Read
   `skipped`. Then `document.open` with `file.local_path`.
2. `document.open` returns `load_warnings` (for example `references_not_found`) and
   `missing_references` (up to 50 referenced paths that are not on disk). Read them first. They name
   the missing files without walking the tree.
3. `model.summary`: check `missing_component_count`, `mate_count` and `features_with_errors`.
4. `components.list` with `only_problems` true: the missing files. `file_exists` false or
   `suppression` `missing_file` means the referenced file is not on this machine.
5. `mates.list` with `only_problems` true: each mate's `error` (`code`, `name`, `meaning`) and
   `whats_wrong`. An entity with `resolved` false has a missing or suppressed reference.
6. `feature.get` for a feature in a rebuild error, to quote `whats_wrong`.

Classify each problem before you change anything:

- **Missing files.** Fetch the file from the linked vault (`pdm.search`, then `pdm.file.get_latest`).
  If the path is under another vault view, stop: the file is in that vault, which this tool cannot
  use. Mates on a missing file cannot be repaired here.
- **Mate errors.** Use the table in `assemblies`. Suppress a mate whose entities are gone, or
  re-create it between reference planes.
- **Over-defined or conflicting mates.** Name the mates and quote their errors. Do not guess which
  one to remove; ask the user.

Stop here when scripts are off: `status` reports `scripts_enabled` false in its `ragtime` object.
Report the diagnosis.

## 2. Copy the model

`document.copy` with `document` set to the open assembly. The result has `path`, the copy's top
file in `<work folder>\copies\...`, and `warnings`. Work on that `path`. A missing reference is still
missing in the copy, so report every warning.

## 3. Repair the copy with scripts

Call `script.run` with `source` (the snippet text), `purpose` (one sentence), `arguments` (for example
`{"path": "<copy path>"}`) and `timeout` 60 to 90. `describe` gives the schema. A run like these takes
2 to 3 s on a small assembly. Opening a large assembly read-write can take far longer: a 37-component
assembly took 16.5 s just to open read-only. Use `timeout` 60 to 90 for assembly scripts.

Each snippet below is a full script. It works only on a copy under the work folder's `copies` folder,
and it closes the copy in a `finally` block. The first only reads. The other two change the copy,
rebuild it and save it. Scripts do not save into the vault.

List the mates with feature errors (`arguments`: `path`). It opens the copy read-only, walks the
`MateGroup` features and reads each mate's `GetErrorCode2`:

<!-- live-check: verified on SolidWorks 2024 SP5 -->
```csharp
using System;
using System.Collections.Generic;
using System.IO;
using Ragtime.SolidWorks.Scripting;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

public static class RagtimeScript
{
    public static object Run(ScriptContext ctx)
    {
        string path = Path.GetFullPath((string)ctx.Arguments["path"]);
        string copies = Path.GetFullPath(Path.Combine(ctx.WorkFolder, "copies")) + "\\";
        if (!path.StartsWith(copies, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Work only on a copy under the work folder's copies folder.");
        int errors = 0;
        int warnings = 0;
        ModelDoc2 model = null;
        try
        {
            int options = (int)swOpenDocOptions_e.swOpenDocOptions_Silent | (int)swOpenDocOptions_e.swOpenDocOptions_ReadOnly;
            model = ctx.App.OpenDoc6(path, (int)swDocumentTypes_e.swDocASSEMBLY, options, "", ref errors, ref warnings);
            if (model == null) throw new InvalidOperationException("Open failed: " + errors);
            int activationError = 0;
            ctx.App.ActivateDoc3(model.GetTitle(), false, 0, ref activationError);
            var problems = new List<string>();
            Feature feature = (Feature)model.FirstFeature();
            while (feature != null)
            {
                if (feature.GetTypeName2() == "MateGroup")
                {
                    Feature mate = (Feature)feature.GetFirstSubFeature();
                    while (mate != null)
                    {
                        bool warning;
                        int code = mate.GetErrorCode2(out warning);
                        if (code != 0) problems.Add(mate.Name + ": error " + code);
                        mate = (Feature)mate.GetNextSubFeature();
                    }
                }
                feature = (Feature)feature.GetNextFeature();
            }
            return problems;
        }
        finally
        {
            if (model != null) ctx.App.CloseDoc(model.GetTitle());
        }
    }
}
```

Suppress a mate that cannot be solved (`arguments`: `path`, `suppress` set to the mate name):

<!-- live-check: verified on SolidWorks 2024 SP5 -->
```csharp
using System;
using System.IO;
using Ragtime.SolidWorks.Scripting;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

public static class RagtimeScript
{
    public static object Run(ScriptContext ctx)
    {
        string path = Path.GetFullPath((string)ctx.Arguments["path"]);
        string copies = Path.GetFullPath(Path.Combine(ctx.WorkFolder, "copies")) + "\\";
        if (!path.StartsWith(copies, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Work only on a copy under the work folder's copies folder.");
        int errors = 0;
        int warnings = 0;
        ModelDoc2 model = null;
        try
        {
            int options = (int)swOpenDocOptions_e.swOpenDocOptions_Silent;
            model = ctx.App.OpenDoc6(path, (int)swDocumentTypes_e.swDocASSEMBLY, options, "", ref errors, ref warnings);
            if (model == null) throw new InvalidOperationException("Open failed: " + errors);
            int activationError = 0;
            ctx.App.ActivateDoc3(model.GetTitle(), false, 0, ref activationError);
            string mateName = (string)ctx.Arguments["suppress"];
            model.ClearSelection2(true);
            if (!model.Extension.SelectByID2(mateName, "MATE", 0, 0, 0, false, 0, null, 0))
                throw new InvalidOperationException("Mate not found: " + mateName);
            model.EditSuppress2();
            model.ForceRebuild3(false);
            bool saved = model.Save3((int)swSaveAsOptions_e.swSaveAsOptions_Silent, ref errors, ref warnings);
            if (!saved) throw new InvalidOperationException("Save failed: " + errors + " " + warnings);
            return mateName + " suppressed.";
        }
        finally
        {
            if (model != null) ctx.App.CloseDoc(model.GetTitle());
        }
    }
}
```

Re-create a coincident mate between reference planes (`arguments`: `path`, `replace` set to the old mate
name, `first_component` and `second_component` set to component names, `plane_index` set to the plane
position). Component names come from `components.list` (`name`, for example `box-1`).

<!-- live-check: verified on SolidWorks 2024 SP5 -->
```csharp
using System;
using System.IO;
using Ragtime.SolidWorks.Scripting;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

public static class RagtimeScript
{
    public static object Run(ScriptContext ctx)
    {
        string path = Path.GetFullPath((string)ctx.Arguments["path"]);
        string copies = Path.GetFullPath(Path.Combine(ctx.WorkFolder, "copies")) + "\\";
        if (!path.StartsWith(copies, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Work only on a copy under the work folder's copies folder.");
        int errors = 0;
        int warnings = 0;
        ModelDoc2 model = null;
        try
        {
            int options = (int)swOpenDocOptions_e.swOpenDocOptions_Silent;
            model = ctx.App.OpenDoc6(path, (int)swDocumentTypes_e.swDocASSEMBLY, options, "", ref errors, ref warnings);
            if (model == null) throw new InvalidOperationException("Open failed: " + errors);
            int activationError = 0;
            ctx.App.ActivateDoc3(model.GetTitle(), false, 0, ref activationError);
            AssemblyDoc assembly = (AssemblyDoc)model;
            model.ClearSelection2(true);
            if (!model.Extension.SelectByID2((string)ctx.Arguments["replace"], "MATE", 0, 0, 0, false, 0, null, 0))
                throw new InvalidOperationException("Mate not found.");
            model.EditDelete();
            model.ClearSelection2(true);
            int planeIndex = (int)ctx.Arguments["plane_index"];
            Feature firstPlane = FindPlane(assembly.GetComponentByName((string)ctx.Arguments["first_component"]), planeIndex);
            Feature secondPlane = FindPlane(assembly.GetComponentByName((string)ctx.Arguments["second_component"]), planeIndex);
            if (firstPlane == null || secondPlane == null)
                throw new InvalidOperationException("Reference plane not found.");
            if (!firstPlane.Select2(false, 1) || !secondPlane.Select2(true, 1))
                throw new InvalidOperationException("Plane selection failed.");
            int mateError = 0;
            Mate2 mate = (Mate2)assembly.AddMate5((int)swMateType_e.swMateCOINCIDENT,
                (int)swMateAlign_e.swMateAlignALIGNED, false, 0, 0, 0, 1, 1, 0, 0, 0, false, false, 0, out mateError);
            model.ClearSelection2(true);
            if (mate == null || mateError != (int)swAddMateError_e.swAddMateError_NoError)
                throw new InvalidOperationException("Mate failed: " + mateError);
            model.ForceRebuild3(false);
            bool saved = model.Save3((int)swSaveAsOptions_e.swSaveAsOptions_Silent, ref errors, ref warnings);
            if (!saved) throw new InvalidOperationException("Save failed: " + errors + " " + warnings);
            return "Mate re-created.";
        }
        finally
        {
            if (model != null) ctx.App.CloseDoc(model.GetTitle());
        }
    }

    // Returns the index-th reference plane of a component (0 is its first). Plane names vary between
    // models, so the position is the stable key. Returns null when the component or plane is missing.
    private static Feature FindPlane(Component2 component, int index)
    {
        if (component == null) return null;
        int found = 0;
        Feature feature = (Feature)component.FirstFeature();
        while (feature != null)
        {
            if (feature.GetTypeName2() == "RefPlane")
            {
                if (found == index) return feature;
                found++;
            }
            feature = (Feature)feature.GetNextFeature();
        }
        return null;
    }
}
```

Snippet 3 finds the planes by position, not by name, because plane names vary between models (this
sample's planes are named `Front`, not `Front Plane`). `plane_index` 0 is each component's first
reference plane. The script deletes the mate named in `replace` first, then adds the coincident mate.
`AddMate5` returns an error status: `1` (`swAddMateError_NoError`) is success, and `0` is an unknown
error.

Re-attaching to faces needs names from the user, or a geometric search that a script writes itself.
This snippet does not do either. `mates.list` gives component and entity type only, not face names.

`AddMate5` is marked obsolete in the SolidWorks 2026 API help; `IAssemblyDoc::CreateMate` replaces it
in newer APIs. The gear ratios are `1, 1`, which the coincident mate ignores.

The repair scripts rebuild with `ForceRebuild3` and save with `Save3`; a save that returns false is a
failure. After a repair, check the copy again: `document.open` on `path`, then `mates.list` with
`only_problems` true. The scripts close the copy they open. Close any document you opened with
`document.open` (`document.close`) when you finish.

## 4. Deliver

No operation replaces an existing vault file with a copy. Never overwrite the checked-out or local
source file with the repaired copy. Give the user one of these choices:

- Add the copy to an approved folder under a new name: `pdm.file.add` with `source` set to the copy in
  the work folder and a `name` that differs from the original. Check it in as a new file (see `pdm`).
  Say the original is unchanged.
- Keep the copy on this machine and ask the user what goes into the vault.

Show the user the copy with `model.preview`, then list the mates that are still in error. When the
vault refuses the change, the copy's `path` from `document.copy` is the deliverable on this machine.
A native export needs the copy open: `document.open` on `path`, then `model.export` with format
`native`. For an assembly that export is a zip with every referenced file. Tell the user which setting
would allow check-in: "PDM vault: Allow changes", or an approved folder. Report what you did not fix,
and why.
