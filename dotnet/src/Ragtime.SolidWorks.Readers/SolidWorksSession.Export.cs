using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using SW = global::SolidWorks.Interop.sldworks;
using SWC = global::SolidWorks.Interop.swconst;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>
    /// model.export and model.preview. Writes one file to the path Core chose; never saves the document. Model formats
    /// are a silent copy (the document keeps its title and path). Native parts and drawings are copied from disk; a
    /// native assembly is a zip of the assembly and its local referenced files. Unverified against live SolidWorks.
    /// </summary>
    internal sealed partial class SolidWorksSession
    {
        private const int CurrentSaveVersion = 0;
        private const int DefaultNamedViewType = -1;
        private const string ViewIsometric = "isometric";
        private const string ExportFailedMessage = "The file could not be written.";
        private const string PreviewFailedMessage = "The picture could not be written.";

        private static readonly int CopySaveOptions =
            (int)SWC.swSaveAsOptions_e.swSaveAsOptions_Silent | (int)SWC.swSaveAsOptions_e.swSaveAsOptions_Copy;

        /// <summary>Writes one export file. Core checks the path and measures the file afterwards.</summary>
        public ExportResult ExportModel(ExportRequest request) => RunAs(ErrorCodes.ExportFailed, "exporting the model", () =>
        {
            var model = RequireOpenDocument(request.Document);
            var type = ReaderNames.DocumentType(model.GetType());
            var warnings = new List<string>();
            if (model.GetSaveFlag())
            {
                warnings.Add(request.Format == NativeFiles.Format
                    ? "The document has unsaved changes; the native file is the last saved version."
                    : "The document has unsaved changes; the export includes them.");
            }

            if (request.Format == NativeFiles.Format)
            {
                WriteNative(model, type, request.TargetPath, warnings);
                return new ExportResult(
                    new[] { new ExportedFile(NativeFiles.Format, request.TargetPath, NativeFiles.Extension(type)) },
                    ActiveConfigurationName(model),
                    warnings);
            }

            var previous = ActiveConfigurationName(model);
            var requested = request.Configuration;
            var switched = requested != null && requested != previous;
            try
            {
                if (requested != null && switched)
                {
                    ShowConfiguration(model, requested);
                }

                SaveCopy(model, request.TargetPath, ErrorCodes.ExportFailed);
            }
            finally
            {
                if (switched && previous != null)
                {
                    RestoreConfiguration(model, previous);
                }
            }

            return new ExportResult(
                new[] { new ExportedFile("model", request.TargetPath, request.Format) },
                requested ?? previous,
                warnings);
        });

        /// <summary>
        /// Renders one PNG of a named view of the document, in the requested configuration. The document and the
        /// configuration are activated for the picture and restored afterwards, and the view pose is put back. Drawings
        /// use the sheet as it is.
        /// </summary>
        public ExportResult RenderPreview(PreviewRequest request) => RunAs(ErrorCodes.PreviewUnavailable, "rendering the preview", () =>
        {
            var model = RequireOpenDocument(request.Document);
            var type = ReaderNames.DocumentType(model.GetType());
            var isSheet = type == ProtocolInfo.DocumentTypes.Drawing;
            var warnings = new List<string> { "The preview uses the SolidWorks graphics window size; the requested width and height were not applied." };
            if (model.GetSaveFlag())
            {
                warnings.Add("The document has unsaved changes; the preview shows them.");
            }

            var previousDocument = ActiveDocumentOrNull();
            var switchedDocument = previousDocument == null || !SameDocument(previousDocument, model);
            var previousConfiguration = ActiveConfigurationName(model);
            var requested = request.Configuration;
            var switchedConfiguration = requested != null && requested != previousConfiguration;
            SW.IModelView? view = null;
            ViewPose? pose = null;
            try
            {
                if (switchedDocument)
                {
                    ActivateDocument(model);
                }

                if (requested != null && switchedConfiguration)
                {
                    ShowConfiguration(model, requested);
                }

                if (!isSheet)
                {
                    view = (SW.IModelView)model.ActiveView;
                    pose = CapturePose(view);
                    // ShowNamedView2 reports nothing; a missing view leaves the current orientation in the picture.
                    model.ShowNamedView2(NamedViewName(request.View), DefaultNamedViewType);
                    model.ViewZoomtofit2();
                }

                SaveCopy(model, request.TargetPath, ErrorCodes.PreviewUnavailable);
            }
            finally
            {
                if (view != null && pose != null)
                {
                    RestorePose(view, pose);
                }

                if (switchedConfiguration && previousConfiguration != null)
                {
                    RestoreConfiguration(model, previousConfiguration);
                }

                if (switchedDocument && previousDocument != null)
                {
                    ReactivateDocument(previousDocument);
                }
            }

            return new ExportResult(
                new[] { new ExportedFile("preview", request.TargetPath, "png") },
                requested ?? previousConfiguration,
                warnings);
        });

        /// <summary>
        /// Runs one export or preview step. A COM failure is reported under the operation's own code, and so are the
        /// I/O failures of writing the file; the messages stay generic (no paths).
        /// </summary>
        private static T RunAs<T>(string code, string action, Func<T> work)
        {
            try
            {
                return Run(action, work);
            }
            catch (HostErrorException error) when (error.Code == ErrorCodes.SolidworksError)
            {
                throw HostError.Coded(code, error.Message, error.Details);
            }
            catch (IOException)
            {
                throw HostError.Coded(code, FailureMessage(code));
            }
            catch (UnauthorizedAccessException)
            {
                throw HostError.Coded(code, FailureMessage(code));
            }
            catch (InvalidDataException)
            {
                throw HostError.Coded(code, FailureMessage(code));
            }
        }

        private static string FailureMessage(string code) =>
            code == ErrorCodes.PreviewUnavailable ? PreviewFailedMessage : ExportFailedMessage;

        /// <summary>Saves a silent copy in the format implied by the target extension. Never saves the document itself.</summary>
        private static void SaveCopy(SW.ModelDoc2 model, string target, string code)
        {
            var errors = 0;
            var warnings = 0;
            var saved = model.Extension.SaveAs3(target, CurrentSaveVersion, CopySaveOptions, null, null, ref errors, ref warnings);
            if (!saved || !File.Exists(target) || new FileInfo(target).Length == 0)
            {
                throw HostError.Coded(
                    code,
                    "SolidWorks could not save the file in that format.",
                    new JObject { ["errors"] = errors, ["warnings"] = warnings });
            }
        }

        /// <summary>
        /// Activates a configuration. ShowConfiguration2 reports nothing, so the active name is read back and a mismatch
        /// is an export failure.
        /// </summary>
        private static void ShowConfiguration(SW.ModelDoc2 model, string configuration)
        {
            model.ShowConfiguration2(configuration);
            if (!string.Equals(ActiveConfigurationName(model), configuration, StringComparison.Ordinal))
            {
                throw HostError.Coded(ErrorCodes.ExportFailed, "SolidWorks could not show the requested configuration.");
            }
        }

        /// <summary>Puts the previous configuration back. Never throws: it must not hide the original result or error.</summary>
        private static void RestoreConfiguration(SW.ModelDoc2 model, string configuration)
        {
            try
            {
                model.ShowConfiguration2(configuration);
            }
            catch (Exception)
            {
                // Best effort: the configuration stays as it is; the original outcome is what the caller sees.
            }
        }

        /// <summary>
        /// Activates the target document for the picture and verifies it: the active document must match the target by
        /// path (or title when there is no path). Otherwise the picture would show the wrong model.
        /// </summary>
        private void ActivateDocument(SW.ModelDoc2 document)
        {
            _application.ActivateDoc2(document.GetTitle(), false, 0);
            var active = ActiveDocumentOrNull();
            if (active == null || !SameDocument(active, document))
            {
                throw HostError.Coded(ErrorCodes.PreviewUnavailable, "SolidWorks did not make the document active for the picture.");
            }
        }

        /// <summary>Activates the previously active document again. Never throws (cleanup must not mask the result).</summary>
        private void ReactivateDocument(SW.ModelDoc2 document)
        {
            try
            {
                _application.ActivateDoc2(document.GetTitle(), false, 0);
            }
            catch (Exception)
            {
                // Best effort: the user's active document may stay as the picture left it.
            }
        }

        private SW.ModelDoc2? ActiveDocumentOrNull() => ComHelper.TryObject(() => _application.ActiveDoc as SW.ModelDoc2);

        /// <summary>Same document by path when both have one, else by title.</summary>
        private static bool SameDocument(SW.ModelDoc2 first, SW.ModelDoc2 second)
        {
            var firstPath = ComHelper.TryText(() => first.GetPathName());
            var secondPath = ComHelper.TryText(() => second.GetPathName());
            if (!string.IsNullOrEmpty(firstPath) && !string.IsNullOrEmpty(secondPath))
            {
                return string.Equals(firstPath, secondPath, StringComparison.OrdinalIgnoreCase);
            }

            return string.Equals(first.GetTitle(), second.GetTitle(), StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>The view pose that a named view change replaces: orientation, translation and scale.</summary>
        private static ViewPose CapturePose(SW.IModelView view) =>
            new ViewPose(view.Orientation3, view.Translation3, view.Scale2);

        /// <summary>Puts the captured pose back. Never throws: the picture is already taken.</summary>
        private static void RestorePose(SW.IModelView view, ViewPose pose)
        {
            try
            {
                view.Orientation3 = pose.Orientation!;
                view.Translation3 = pose.Translation!;
                view.Scale2 = pose.Scale;
            }
            catch (Exception)
            {
                // Best effort: the view may keep the picture's orientation; the warning above says the size is not restored.
            }
        }

        /// <summary>A native file is copied from disk, after the same local-path check the open uses. An assembly becomes a zip.</summary>
        private void WriteNative(SW.ModelDoc2 model, string type, string target, List<string> warnings)
        {
            var source = model.GetPathName();
            if (string.IsNullOrEmpty(source))
            {
                throw HostError.Coded(ErrorCodes.ExportFailed, "The document has not been saved, so there is no native file to copy. Save it first.");
            }

            LocalFileCheck.Require(source, "path");
            if (!File.Exists(source))
            {
                throw HostError.Coded(ErrorCodes.ExportFailed, "The document file was not found on disk.");
            }

            if (type != ProtocolInfo.DocumentTypes.Assembly)
            {
                File.Copy(source, target, false);
                return;
            }

            WriteAssemblyZip(source, target, warnings);
        }

        /// <summary>
        /// Writes the assembly and its referenced files into a zip. Each reference is checked with LocalFileCheck before
        /// it is read; the total is checked against the ceiling before the first byte is written.
        /// </summary>
        private void WriteAssemblyZip(string source, string target, List<string> warnings)
        {
            var root = Path.GetDirectoryName(source) ?? string.Empty;
            var references = ReferencedFiles(source, warnings);

            long total = new FileInfo(source).Length;
            foreach (var reference in references)
            {
                total += new FileInfo(reference).Length;
            }

            NativeFiles.RequireWithinLimit(total);
            WarnAboutUnsavedReferences(references, warnings);

            var taken = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var entries = new List<KeyValuePair<string, string>>
            {
                new KeyValuePair<string, string>(source, NativeFiles.Unique(Path.GetFileName(source), taken, out _)),
            };
            foreach (var reference in references)
            {
                var name = NativeFiles.Unique(NativeFiles.EntryNameFor(root, reference), taken, out var renamed);
                if (renamed)
                {
                    warnings.Add("Two referenced files share a name; " + Path.GetFileName(reference) + " is stored as " + name + ".");
                }

                entries.Add(new KeyValuePair<string, string>(reference, name));
            }

            using (var stream = File.Create(target))
            using (var archive = new ZipArchive(stream, ZipArchiveMode.Create))
            {
                foreach (var entry in entries)
                {
                    using (var input = File.OpenRead(entry.Key))
                    using (var output = archive.CreateEntry(entry.Value, CompressionLevel.Optimal).Open())
                    {
                        input.CopyTo(output);
                    }
                }
            }
        }

        /// <summary>
        /// The local SolidWorks references of the assembly, each as a full path. Non-SolidWorks entries, files on network
        /// or linked locations and missing files are left out, each with a warning that gives the count only.
        /// </summary>
        private List<string> ReferencedFiles(string source, List<string> warnings)
        {
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { source };
            var result = new List<string>();
            var skipped = 0;
            var unreadable = 0;
            var missing = 0;
            // Traverseflag true: every level (nested sub-assembly parts too). Searchflag false: the locations the
            // references were last saved at, as the document loaded them. AddReadOnlyInfo false: name and path only.
            var pairs = ComHelper.Strings(_application.GetDocumentDependencies2(source, true, false, false));
            foreach (var dependency in NativeFiles.PathsFromDependencyPairs(pairs))
            {
                if (!NativeFiles.IsSolidWorksFile(dependency))
                {
                    skipped++;
                    continue;
                }

                if (!seen.Add(dependency))
                {
                    continue;
                }

                try
                {
                    LocalFileCheck.Require(dependency, "path");
                }
                catch (HostErrorException)
                {
                    unreadable++;
                    continue;
                }

                if (!File.Exists(dependency))
                {
                    missing++;
                    continue;
                }

                result.Add(dependency);
            }

            if (skipped > 0)
            {
                warnings.Add(skipped + " referenced file(s) are not SolidWorks parts, assemblies or drawings and were left out.");
            }

            if (unreadable > 0)
            {
                warnings.Add(unreadable + " referenced file(s) are on a network or linked location and were left out.");
            }

            if (missing > 0)
            {
                warnings.Add(missing + " referenced file(s) were not found on disk and were left out.");
            }

            return result;
        }

        /// <summary>A referenced document open with unsaved changes: the zip holds its saved version, so say so.</summary>
        private void WarnAboutUnsavedReferences(List<string> references, List<string> warnings)
        {
            var paths = new HashSet<string>(references, StringComparer.OrdinalIgnoreCase);
            var unsaved = 0;
            foreach (var open in OpenDocuments())
            {
                var path = ComHelper.TryText(() => open.GetPathName());
                if (path != null && path.Length > 0 && paths.Contains(path) && open.GetSaveFlag())
                {
                    unsaved++;
                }
            }

            if (unsaved > 0)
            {
                warnings.Add(unsaved + " referenced file(s) have unsaved changes; the zip holds their saved versions.");
            }
        }

        private static string NamedViewName(string view)
        {
            switch (view)
            {
                case ViewIsometric:
                    return "*Isometric";
                case "front":
                    return "*Front";
                case "top":
                    return "*Top";
                case "right":
                    return "*Right";
                default:
                    throw HostError.InvalidArguments("view", "The view must be isometric, front, top or right.");
            }
        }

        /// <summary>The view transform, translation and scale, captured before a named view replaces them.</summary>
        private sealed class ViewPose
        {
            public ViewPose(SW.MathTransform? orientation, SW.MathVector? translation, double scale)
            {
                Orientation = orientation;
                Translation = translation;
                Scale = scale;
            }

            public SW.MathTransform? Orientation { get; }

            public SW.MathVector? Translation { get; }

            public double Scale { get; }
        }
    }
}
