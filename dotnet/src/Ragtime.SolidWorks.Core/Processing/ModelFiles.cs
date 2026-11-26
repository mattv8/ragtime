using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// model.export and model.preview. Validates the arguments, has the session write files into the agent work folder
    /// under generated names, then adds the facts Ragtime checks before it fetches a file: display name, size and
    /// SHA-256. Runs on the instance's STA thread; the session never sees a user-supplied output path.
    /// </summary>
    internal static class ModelFiles
    {
        /// <summary>Operation name of the export.</summary>
        public const string ExportOperation = "model.export";

        /// <summary>Operation name of the preview.</summary>
        public const string PreviewOperation = "model.preview";

        /// <summary>Smallest preview side in pixels; smaller values are clamped up.</summary>
        public const int MinPreviewSize = 256;

        /// <summary>Largest preview side in pixels; larger values are clamped down.</summary>
        public const int MaxPreviewSize = 2048;

        private const int DefaultPreviewWidth = 1024;
        private const int DefaultPreviewHeight = 768;
        private const string DefaultView = "isometric";
        private const string PreviewFormat = "png";
        private const int MaxDisplayStemLength = 120;
        private const string FallbackDisplayStem = "model";
        private static readonly string[] KnownExtensions = { "sldprt", "sldasm", "slddrw" };
        private static readonly string[] ExportFormats = { "step", "stl", "3mf", "glb", NativeFiles.Format };
        private static readonly string[] PreviewViews = { "isometric", "front", "top", "right" };

        /// <summary>True for the two file operations.</summary>
        public static bool Handles(string operation) => operation == ExportOperation || operation == PreviewOperation;

        /// <summary>
        /// Validates one file operation's arguments against the document. Throws invalid_arguments for a bad request and
        /// internal_error when the agent has no work folder. Runs before any SolidWorks call.
        /// </summary>
        public static ModelFileRequest Parse(string operation, JObject arguments, DocumentEntry document, string? workFolder)
        {
            if (string.IsNullOrWhiteSpace(workFolder))
            {
                throw HostError.Coded(
                    ErrorCodes.InternalError,
                    "The agent work folder is not available, so no file can be written. Restart the SolidWorks agent.");
            }

            return operation == ExportOperation
                ? ParseExport(arguments, document, workFolder!)
                : ParsePreview(arguments, workFolder!);
        }

        /// <summary>Runs a validated request on the session and returns the result object (files, document, warnings).</summary>
        public static JToken Execute(ModelFileRequest request, ISolidWorksSession session, DocumentEntry document, RequestContext context)
        {
            var written = new List<string>();
            try
            {
                return Build(request, session, document, context, written);
            }
            catch (Exception)
            {
                // A failed request leaves no file of its own behind; files of other requests are not touched.
                foreach (var path in written)
                {
                    TryDelete(path);
                }

                throw;
            }
        }

        /// <summary>Display name from the saved file stem, or the title for unsaved documents.</summary>
        public static string DisplayName(string title, string path, string extension)
        {
            string stem;
            if (!string.IsNullOrEmpty(path))
            {
                stem = Path.GetFileNameWithoutExtension(path.Replace('\\', '/'));
            }
            else
            {
                stem = (title ?? string.Empty).Trim();
                foreach (var known in KnownExtensions)
                {
                    var suffix = "." + known;
                    if (stem.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
                    {
                        stem = stem.Substring(0, stem.Length - suffix.Length);
                    }
                }
            }

            var builder = new StringBuilder(stem.Length);
            foreach (var character in stem)
            {
                builder.Append(IsSafeNameCharacter(character) ? character : '_');
            }

            var safe = builder.ToString().Trim(' ', '.');
            if (safe.Length > MaxDisplayStemLength)
            {
                safe = safe.Substring(0, MaxDisplayStemLength).TrimEnd(' ', '.');
            }

            return (safe.Length == 0 ? FallbackDisplayStem : safe) + "." + extension;
        }

        private static JToken Build(
            ModelFileRequest request,
            ISolidWorksSession session,
            DocumentEntry document,
            RequestContext context,
            List<string> written)
        {
            var reference = document.ToRef();
            var files = new JArray();
            var warnings = new List<string>();
            var configuration = request.Configuration ?? document.ActiveConfiguration;

            if (request.Operation == ExportOperation)
            {
                var isNative = request.Format == NativeFiles.Format;
                if (isNative && request.Configuration != null)
                {
                    warnings.Add("The configuration is not used for native files; the file is the document as saved.");
                }

                var target = GeneratedPath(request.WorkFolder, "export", ExportExtension(request.Format, document.Type));
                var exported = WriteFile(target, written, () => session.ExportModel(
                    new ExportRequest(reference, request.Format, request.Configuration, target)));
                files.Add(FileEntry(SingleFile(exported, target, ErrorCodes.ExportFailed), target, request.WorkFolder, document));
                warnings.AddRange(exported.Warnings);
                configuration = exported.Configuration ?? configuration;

                if (request.IncludePreview)
                {
                    // The preview shows the configuration that was exported; native files have no configuration to match.
                    AddExportPreview(request, session, reference, document, files, warnings, written, isNative ? null : request.Configuration);
                }
            }
            else
            {
                var target = GeneratedPath(request.WorkFolder, "preview", PreviewFormat);
                var preview = WriteFile(target, written, () => session.RenderPreview(
                    new PreviewRequest(reference, request.View, request.Width, request.Height, target, null)));
                files.Add(FileEntry(SingleFile(preview, target, ErrorCodes.PreviewUnavailable), target, request.WorkFolder, document));
                warnings.AddRange(preview.Warnings);
                configuration = preview.Configuration ?? configuration;
            }

            var result = new JObject
            {
                ["files"] = files,
                ["document"] = new JObject
                {
                    ["path"] = WireJson.Value(document.Path),
                    ["type"] = WireJson.Value(document.Type),
                    ["title"] = WireJson.Value(document.Title),
                    ["configuration"] = WireJson.Value(configuration),
                },
                ["warnings"] = JArray.FromObject(warnings),
            };
            return Pager.RequireFits(result, context.MaxResultBytes);
        }

        private static ModelFileRequest ParseExport(JObject arguments, DocumentEntry document, string workFolder)
        {
            var format = ArgumentReader.RequiredString(arguments, "format");
            if (Array.IndexOf(ExportFormats, format) < 0)
            {
                throw HostError.InvalidArguments("format", "The format must be step, stl, 3mf, glb or native.");
            }

            if (document.Type == ProtocolInfo.DocumentTypes.Drawing && format != NativeFiles.Format)
            {
                throw HostError.InvalidArguments(
                    "format",
                    "Drawings can be exported as native only. Use model.preview for a picture of the sheet.");
            }

            if (document.Type == ProtocolInfo.DocumentTypes.Assembly && format == "stl")
            {
                throw HostError.InvalidArguments(
                    "format",
                    "STL export of an assembly writes one file per component. Export the assembly as 3mf or glb for a single mesh, or step.");
            }

            return new ModelFileRequest(
                ExportOperation,
                format,
                ArgumentReader.OptionalString(arguments, "configuration"),
                ArgumentReader.OptionalBoolean(arguments, "include_preview", true),
                DefaultView,
                DefaultPreviewWidth,
                DefaultPreviewHeight,
                workFolder);
        }

        private static ModelFileRequest ParsePreview(JObject arguments, string workFolder)
        {
            var view = ArgumentReader.OptionalString(arguments, "view") ?? DefaultView;
            if (Array.IndexOf(PreviewViews, view) < 0)
            {
                throw HostError.InvalidArguments("view", "The view must be isometric, front, top or right.");
            }

            return new ModelFileRequest(
                PreviewOperation,
                PreviewFormat,
                configuration: null,
                includePreview: false,
                view: view,
                width: ClampedSize(arguments, "width", DefaultPreviewWidth),
                height: ClampedSize(arguments, "height", DefaultPreviewHeight),
                workFolder: workFolder);
        }

        private static int ClampedSize(JObject arguments, string name, int fallback)
        {
            if (!WireJson.TryGetInteger(arguments[name], out var value))
            {
                return fallback;
            }

            return (int)Math.Min(MaxPreviewSize, Math.Max(MinPreviewSize, value));
        }

        /// <summary>Adds the preview that accompanies an export. A preview failure is a warning, not an error.</summary>
        private static void AddExportPreview(
            ModelFileRequest request,
            ISolidWorksSession session,
            DocumentRef reference,
            DocumentEntry document,
            JArray files,
            List<string> warnings,
            List<string> written,
            string? configuration)
        {
            var target = GeneratedPath(request.WorkFolder, "preview", PreviewFormat);
            try
            {
                var preview = WriteFile(target, written, () => session.RenderPreview(
                    new PreviewRequest(reference, DefaultView, DefaultPreviewWidth, DefaultPreviewHeight, target, configuration)));
                files.Add(FileEntry(SingleFile(preview, target, ErrorCodes.PreviewUnavailable), target, request.WorkFolder, document));
                warnings.AddRange(preview.Warnings);
            }
            catch (HostErrorException error) when (error.Code == ErrorCodes.PreviewUnavailable)
            {
                warnings.Add("No preview was made: " + error.Message);
            }
        }

        private static ExportedFile SingleFile(ExportResult result, string target, string failureCode)
        {
            if (result.Files.Count != 1 || !string.Equals(result.Files[0].Path, target, StringComparison.OrdinalIgnoreCase))
            {
                throw HostError.Coded(failureCode, "SolidWorks did not write the expected file.");
            }

            return result.Files[0];
        }

        /// <summary>
        /// Records the target as written by this request, then runs the write. A failed write removes its partial file
        /// at once; <see cref="Execute"/> removes the rest of the request's files if the request fails later.
        /// </summary>
        private static ExportResult WriteFile(string target, List<string> written, Func<ExportResult> write)
        {
            written.Add(target);
            try
            {
                return write();
            }
            catch (Exception)
            {
                TryDelete(target);
                throw;
            }
            finally
            {
                ModelFileCleanup.DeleteStrays(target);
            }
        }

        /// <summary>The file entry for one written file: checks it is inside the work folder, then measures it.</summary>
        private static JObject FileEntry(ExportedFile file, string target, string workFolder, DocumentEntry document)
        {
            var info = new FileInfo(target);
            if (!info.Exists || !IsInside(info.FullName, workFolder))
            {
                throw HostError.Coded(ErrorCodes.ExportFailed, "The file was not written to the agent work folder.");
            }

            return new JObject
            {
                ["role"] = WireJson.Value(file.Role),
                ["path"] = WireJson.Value(info.FullName),
                ["file_name"] = WireJson.Value(DisplayName(document.Title, document.Path, file.Format)),
                ["format"] = WireJson.Value(file.Format),
                ["size_bytes"] = WireJson.Value(info.Length),
                ["sha256"] = WireJson.Value(Sha256Hex(info.FullName)),
            };
        }

        private static string ExportExtension(string format, string documentType) =>
            format == NativeFiles.Format ? NativeFiles.Extension(documentType) : format;

        /// <summary>A file name generated here (never user-supplied) inside the work folder.</summary>
        private static string GeneratedPath(string workFolder, string prefix, string extension) =>
            Path.Combine(workFolder, prefix + "-" + Guid.NewGuid().ToString("N") + "." + extension);

        /// <summary>True when <paramref name="fullPath"/> is inside <paramref name="folder"/> (case-insensitive, any depth).</summary>
        internal static bool IsInside(string fullPath, string folder)
        {
            var root = Path.GetFullPath(folder).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
                + Path.DirectorySeparatorChar;
            return fullPath.StartsWith(root, StringComparison.OrdinalIgnoreCase);
        }

        private static string Sha256Hex(string path)
        {
            using (var stream = File.OpenRead(path))
            using (var sha = SHA256.Create())
            {
                var hash = sha.ComputeHash(stream);
                var hex = new StringBuilder(hash.Length * 2);
                foreach (var b in hash)
                {
                    hex.Append(b.ToString("x2"));
                }

                return hex.ToString();
            }
        }

        private static bool IsSafeNameCharacter(char character) =>
            char.IsLetterOrDigit(character) || character == ' ' || character == '-' || character == '_' || character == '.'
            || character == '(' || character == ')';

        private static void TryDelete(string path)
        {
            try
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
            catch (IOException)
            {
                // Best effort: the partial file stays in the work folder and is not returned to the caller.
            }
            catch (UnauthorizedAccessException)
            {
                // Best effort, as above.
            }
        }
    }

    /// <summary>Validated arguments of model.export or model.preview.</summary>
    internal sealed class ModelFileRequest
    {
        /// <summary>Creates the request.</summary>
        public ModelFileRequest(
            string operation,
            string format,
            string? configuration,
            bool includePreview,
            string view,
            int width,
            int height,
            string workFolder)
        {
            Operation = operation;
            Format = format;
            Configuration = configuration;
            IncludePreview = includePreview;
            View = view;
            Width = width;
            Height = height;
            WorkFolder = workFolder;
        }

        /// <summary>model.export or model.preview.</summary>
        public string Operation { get; }

        /// <summary>Export format (step, stl, 3mf, glb, native), or png for a preview.</summary>
        public string Format { get; }

        /// <summary>Configuration to export, or null for the active one.</summary>
        public string? Configuration { get; }

        /// <summary>For an export: also write the isometric preview.</summary>
        public bool IncludePreview { get; }

        /// <summary>Named view of a preview.</summary>
        public string View { get; }

        /// <summary>Preview width in pixels, already clamped.</summary>
        public int Width { get; }

        /// <summary>Preview height in pixels, already clamped.</summary>
        public int Height { get; }

        /// <summary>Agent work folder the files are written into.</summary>
        public string WorkFolder { get; }
    }
}
