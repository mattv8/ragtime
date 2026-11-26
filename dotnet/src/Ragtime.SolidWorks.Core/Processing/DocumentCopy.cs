using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// document.copy. Core generates the copy folder under the work folder's copies tree, asks the session to copy into it,
    /// then checks what came back. A failed copy removes its folder, so no partial copy is left without a caller.
    /// Runs on the instance's STA thread; the session never sees a user-supplied folder.
    /// </summary>
    internal static class DocumentCopy
    {
        /// <summary>Operation name.</summary>
        public const string Operation = "document.copy";

        /// <summary>Largest number of files one copy may contain.</summary>
        public const int MaxFiles = 500;

        /// <summary>Most entries of the <c>files</c> list; the rest are counted and reported by <c>files_truncated</c>.</summary>
        public const int MaxListedFiles = 50;

        /// <summary>Refuses a copy when there is no work folder. Called before any SolidWorks instance is used.</summary>
        public static void RequireWorkFolder(string? workFolder)
        {
            if (string.IsNullOrWhiteSpace(workFolder))
            {
                throw HostError.Coded(
                    ErrorCodes.InternalError,
                    "The agent work folder is not available, so no copy can be made. Restart the SolidWorks agent.");
            }
        }

        /// <summary>Copies the document and returns the result object (folder, path, files, count, warnings).</summary>
        public static JToken Execute(
            ISolidWorksSession session,
            DocumentEntry document,
            JObject arguments,
            RequestContext context,
            string? workFolder)
        {
            RequireWorkFolder(workFolder);
            var folder = Path.Combine(CopyRetention.CopiesRoot(workFolder!), Guid.NewGuid().ToString("N"));
            var includeReferences = ArgumentReader.OptionalBoolean(arguments, "include_references", true);
            try
            {
                var outcome = session.CopyDocument(new DocumentCopyRequest(document.ToRef(), folder, includeReferences, MaxFiles));
                return Pager.RequireFits(WireJson.ToToken(ToResult(outcome, folder)), context.MaxResultBytes);
            }
            catch (Exception)
            {
                RemoveFolder(folder);
                throw;
            }
        }

        private static DocumentCopyResult ToResult(DocumentCopyOutcome outcome, string folder)
        {
            if (outcome.Files.Count > MaxFiles)
            {
                throw TooManyFiles(outcome.Files.Count);
            }

            if (outcome.Files.Count == 0
                || !string.Equals(outcome.Files[0].Path, outcome.TopPath, StringComparison.OrdinalIgnoreCase))
            {
                throw Failed("SolidWorks did not list the copied document first.");
            }

            var files = new List<CopyFileItem>();
            foreach (var file in outcome.Files)
            {
                if (!ModelFiles.IsInside(file.Path, folder))
                {
                    throw Failed("A copied file was not written to the copy folder.");
                }

                if (files.Count < MaxListedFiles)
                {
                    files.Add(new CopyFileItem { Source = file.Source, Path = file.Path });
                }
            }

            if (!File.Exists(outcome.TopPath))
            {
                throw Failed("SolidWorks did not write the copied document.");
            }

            return new DocumentCopyResult
            {
                Folder = folder,
                Path = outcome.TopPath,
                Files = files,
                Count = outcome.Files.Count,
                FilesTruncated = outcome.Files.Count > MaxListedFiles,
                Warnings = outcome.Warnings.ToList(),
            };
        }

        private static HostErrorException TooManyFiles(int count) =>
            HostError.Coded(
                ErrorCodes.ResultTooLarge,
                $"The copy would need {count} files; the limit is {MaxFiles}. Copy a smaller model, or copy with include_references false.",
                new JObject
                {
                    ["reason"] = WireJson.Value("file_count"),
                    ["limit"] = WireJson.Value(MaxFiles),
                    ["count"] = WireJson.Value(count),
                });

        private static HostErrorException Failed(string message) => HostError.Coded(ErrorCodes.SolidworksError, message);

        /// <summary>Best effort: removes the copy folder Core generated for this request. Never masks the original error.</summary>
        private static void RemoveFolder(string folder)
        {
            try
            {
                if (Directory.Exists(folder))
                {
                    Directory.Delete(folder, recursive: true);
                }
            }
            catch (IOException)
            {
                // The files stay until the copy retention sweep removes the folder.
            }
            catch (UnauthorizedAccessException)
            {
                // As above.
            }
        }
    }
}
