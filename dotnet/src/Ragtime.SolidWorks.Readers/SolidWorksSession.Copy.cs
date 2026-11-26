using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using SW = global::SolidWorks.Interop.sldworks;
using SWC = global::SolidWorks.Interop.swconst;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>
    /// <c>document.copy</c>: copies the open document and its references into a new folder with Pack and Go. Silent: the
    /// settings are set through the API, so no dialog opens. The source is never saved, renamed or modified. Without
    /// references, only the top document is saved as a copy (SaveAs3 Copy), so no reference is written.
    /// Every refusal (non-local source, reparse point, destination outside the folder, file count, size and quota) happens
    /// before the first save. Unverified against live SolidWorks.
    /// </summary>
    internal sealed partial class SolidWorksSession
    {
        private const string StatusFailedMessage = "SolidWorks did not finish the copy.";

        /// <summary>Copies the document into <see cref="DocumentCopyRequest.FolderPath"/>; the folder does not exist yet.</summary>
        public DocumentCopyOutcome CopyDocument(DocumentCopyRequest request) => RunAs(ErrorCodes.SolidworksError, "copying the document", () =>
        {
            var model = RequireOpenDocument(request.Document);
            var source = model.GetPathName();
            if (string.IsNullOrEmpty(source))
            {
                throw HostError.Coded(ErrorCodes.SolidworksError, "The document has not been saved, so there is no file to copy. Save it first.");
            }

            CopyDestination.RequireLocalSources(new[] { source });
            LocalFileCheck.Require(source, "path");
            CopyGuards.RequirePlainChain(request.FolderPath);
            var warnings = new List<string>();
            if (model.GetSaveFlag())
            {
                warnings.Add("The document has unsaved changes; the copy holds its last saved version.");
            }

            Directory.CreateDirectory(request.FolderPath);
            return request.IncludeReferences
                ? CopyWithReferences(model, source, request, warnings)
                : CopyTopOnly(model, source, request.FolderPath, warnings);
        });

        private DocumentCopyOutcome CopyTopOnly(SW.ModelDoc2 model, string source, string folder, List<string> warnings)
        {
            var target = Path.Combine(folder, Path.GetFileName(source));
            CopyDestination.RequireInside(folder, new[] { target });
            CopyGuards.RequireBudget(new[] { source }, folder);
            SaveCopy(model, target, ErrorCodes.SolidworksError);
            return new DocumentCopyOutcome(target, new[] { new CopiedFile(source, target) }, warnings);
        }

        private DocumentCopyOutcome CopyWithReferences(SW.ModelDoc2 model, string source, DocumentCopyRequest request, List<string> warnings)
        {
            var folder = request.FolderPath;
            var extension = model.Extension;
            var pack = ComHelper.Require(extension.GetPackAndGo(), "the Pack and Go settings");
            ApplyPackAndGoSettings(pack);
            if (!pack.SetSaveToName(true, folder))
            {
                throw HostError.Coded(ErrorCodes.SolidworksError, "SolidWorks did not accept the copy folder.");
            }

            if (!pack.GetDocumentNames(out object names))
            {
                throw HostError.Coded(ErrorCodes.SolidworksError, "SolidWorks did not list the documents to copy.");
            }

            if (!pack.GetDocumentSaveToNames(out object targets, out _))
            {
                throw HostError.Coded(ErrorCodes.SolidworksError, "SolidWorks did not list the copy names.");
            }

            var sources = ComHelper.Strings(names);
            var copyPaths = ComHelper.Strings(targets);
            if (sources.Count != copyPaths.Count)
            {
                throw HostError.Coded(ErrorCodes.SolidworksError, "SolidWorks listed the copy names inconsistently.");
            }

            CopyDestination.RequireLocalSources(sources);
            foreach (var file in sources)
            {
                LocalFileCheck.Require(file, "path");
            }

            if (sources.Count > request.MaxFiles)
            {
                throw HostError.Coded(
                    ErrorCodes.ResultTooLarge,
                    "The copy would hold " + sources.Count + " files; the limit is " + request.MaxFiles + ".",
                    new JObject
                    {
                        ["reason"] = "file_count",
                        ["limit"] = request.MaxFiles,
                        ["count"] = sources.Count,
                    });
            }

            CopyDestination.RequireInside(folder, copyPaths);
            CopyGuards.RequireBudget(sources, folder);
            AddMissingSourceWarnings(sources, warnings);
            var previous = ActiveDocumentOrNull();
            object? status;
            try
            {
                status = extension.SavePackAndGo(pack);
            }
            finally
            {
                RestoreActiveDocument(previous);
            }

            AddStatusWarnings(status, warnings);
            return OutcomeOf(source, folder, sources, copyPaths, warnings);
        }

        /// <summary>
        /// The Pack and Go flags of document.copy (spec §5.5): no drawings or simulation results, toolbox components and
        /// suppressed components included, the folder structure kept, and no name prefix or suffix.
        /// </summary>
        private static void ApplyPackAndGoSettings(SW.PackAndGo pack)
        {
            pack.IncludeDrawings = false;
            pack.IncludeSimulationResults = false;
            pack.IncludeToolboxComponents = true;
            pack.IncludeSuppressed = true;
            pack.FlattenToSingleFolder = false;
            pack.AddPrefix = string.Empty;
            pack.AddSuffix = string.Empty;
        }

        private static void AddMissingSourceWarnings(IEnumerable<string> sources, List<string> warnings)
        {
            foreach (var source in sources.Where(file => !File.Exists(file)))
            {
                warnings.Add("Missing on disk: " + source);
            }
        }

        /// <summary>Adds one warning per status Pack and Go reports other than success. The status list is read as an array or one value.</summary>
        private static void AddStatusWarnings(object? status, List<string> warnings)
        {
            var items = status is Array ? ComHelper.Objects(status) : new List<object?> { status };
            foreach (var item in items.Where(item => item != null))
            {
                var code = Convert.ToInt32(item, CultureInfo.InvariantCulture);
                if (code != (int)SWC.swPackAndGoSaveStatus_e.swPackAndGoSaveStatus_Succeed)
                {
                    var name = Enum.GetName(typeof(SWC.swPackAndGoSaveStatus_e), code) ?? "unknown";
                    warnings.Add("Pack and Go reported status " + code + " (" + name + ") for a file.");
                }
            }
        }

        /// <summary>
        /// Builds the outcome: the top document first, then every other file that Pack and Go wrote inside the folder.
        /// A file written outside the folder, or not written at all, becomes a warning and is left out of the list.
        /// </summary>
        private static DocumentCopyOutcome OutcomeOf(
            string source,
            string folder,
            IReadOnlyList<string> sources,
            IReadOnlyList<string> copyPaths,
            List<string> warnings)
        {
            var topIndex = IndexOfSource(sources, source);
            if (topIndex < 0)
            {
                throw HostError.Coded(ErrorCodes.SolidworksError, StatusFailedMessage + " The document itself was not in the copy list.");
            }

            var top = copyPaths[topIndex];
            if (!IsInsideFolder(top, folder) || !File.Exists(top))
            {
                throw HostError.Coded(ErrorCodes.SolidworksError, StatusFailedMessage + " The copied document was not written.");
            }

            var files = new List<CopiedFile> { new CopiedFile(source, top) };
            for (var index = 0; index < sources.Count; index++)
            {
                if (index == topIndex)
                {
                    continue;
                }

                var copy = copyPaths[index];
                if (!IsInsideFolder(copy, folder))
                {
                    warnings.Add("Left outside the copy folder: " + copy);
                }
                else if (!File.Exists(copy))
                {
                    warnings.Add("Not written: " + copy);
                }
                else
                {
                    files.Add(new CopiedFile(sources[index], copy));
                }
            }

            return new DocumentCopyOutcome(top, files, warnings);
        }

        private static int IndexOfSource(IReadOnlyList<string> sources, string source)
        {
            for (var index = 0; index < sources.Count; index++)
            {
                if (DocumentPaths.SamePath(sources[index], source))
                {
                    return index;
                }
            }

            return -1;
        }

        private static bool IsInsideFolder(string path, string folder) =>
            path.StartsWith(folder.TrimEnd('\\') + "\\", StringComparison.OrdinalIgnoreCase);

        /// <summary>Puts the previously active document back when Pack and Go changed the active one. Never throws.</summary>
        private void RestoreActiveDocument(SW.ModelDoc2? previous)
        {
            if (previous == null)
            {
                return;
            }

            var current = ActiveDocumentOrNull();
            if (current == null || !SameDocument(current, previous))
            {
                ReactivateDocument(previous);
            }
        }
    }
}
