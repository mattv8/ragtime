using System;
using System.Collections.Generic;
using System.IO;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// Pure rules for native files, shared by the export (Core) and the COM reader: the extension per document type,
    /// which references go into an assembly zip, zip entry names that stay inside the archive, unique names, and the
    /// size ceiling. No COM and no file access, so the rules are testable on any platform.
    /// </summary>
    public static class NativeFiles
    {
        /// <summary>Format name of the native export.</summary>
        public const string Format = "native";

        /// <summary>Uncompressed total (the file plus its references) above which a native export is refused.</summary>
        public const long MaxArchiveBytes = 100L * 1024 * 1024;

        private const string FallbackEntryName = "reference";
        private const int MaxEntryNameLength = 260;
        private static readonly string[] ReferenceExtensions = { ".sldprt", ".sldasm", ".slddrw" };

        /// <summary>File extension of a native file for a document type: zip for assemblies, else the SolidWorks extension.</summary>
        public static string Extension(string documentType)
        {
            switch (documentType)
            {
                case ProtocolInfo.DocumentTypes.Assembly:
                    return "zip";
                case ProtocolInfo.DocumentTypes.Drawing:
                    return "slddrw";
                default:
                    return "sldprt";
            }
        }

        /// <summary>True when the file name has a SolidWorks part, assembly or drawing extension (case-insensitive).</summary>
        public static bool IsSolidWorksFile(string fileName)
        {
            if (string.IsNullOrEmpty(fileName))
            {
                return false;
            }

            return Array.IndexOf(ReferenceExtensions, Path.GetExtension(fileName).ToLowerInvariant()) >= 0;
        }

        /// <summary>
        /// The file paths from the name/path pairs that GetDocumentDependencies2 returns (two strings per document: the
        /// name, then the path). Each path is returned once, in order. Names, unpaired trailing entries and entries that
        /// are not paths (no folder part) are ignored. The list is flat: with traversal on it holds every level.
        /// </summary>
        public static IReadOnlyList<string> PathsFromDependencyPairs(IReadOnlyList<string> pairs)
        {
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var paths = new List<string>();
            for (var index = 0; index + 1 < pairs.Count; index += 2)
            {
                var path = pairs[index + 1];
                if (!string.IsNullOrEmpty(path) && path.IndexOf('\\') >= 0 && seen.Add(path))
                {
                    paths.Add(path);
                }
            }

            return paths;
        }

        /// <summary>
        /// Throws export_failed when the uncompressed total is above <see cref="MaxArchiveBytes"/>. Called before any
        /// archive byte is written.
        /// </summary>
        public static void RequireWithinLimit(long totalBytes)
        {
            if (totalBytes > MaxArchiveBytes)
            {
                throw HostError.Coded(
                    ErrorCodes.ExportFailed,
                    "The assembly and its referenced files add up to more than 100 MB, the limit for files Ragtime can hand out. Export a smaller part or a subassembly instead.");
            }
        }

        /// <summary>
        /// The zip entry name of a referenced file: its path relative to the assembly folder (forward slashes) when it is
        /// inside that folder and the name is safe, otherwise its file name. Never returns an unsafe name.
        /// <paramref name="fullPath"/> must already be a full path (the reader normalises it with Path.GetFullPath).
        /// </summary>
        public static string EntryNameFor(string root, string fullPath)
        {
            var fileName = LastSegment(fullPath);
            if (TryRelativeTo(root, fullPath, out var relative) && IsSafeEntryName(relative))
            {
                return relative;
            }

            return IsSafeEntryName(fileName) ? fileName : FallbackEntryName;
        }

        /// <summary>
        /// True for a zip entry name that stays inside the archive: relative, forward slashes only, no empty, "." or ".."
        /// segments, no drive colon, no leading slash, no UNC prefix, and a sane length.
        /// </summary>
        public static bool IsSafeEntryName(string entryName)
        {
            if (string.IsNullOrEmpty(entryName) || entryName.Length > MaxEntryNameLength)
            {
                return false;
            }

            if (entryName.IndexOf(':') >= 0 || entryName.IndexOf('\\') >= 0 || entryName[0] == '/')
            {
                return false;
            }

            foreach (var segment in entryName.Split('/'))
            {
                if (segment.Length == 0 || segment == "." || segment == "..")
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Returns <paramref name="entryName"/> when it is free in <paramref name="taken"/>, otherwise a free variant
        /// "stem (2).ext", "stem (3).ext" and so on, and sets <paramref name="renamed"/>. The chosen name is added to
        /// <paramref name="taken"/>. Pass a set that compares ordinal-ignore-case.
        /// </summary>
        public static string Unique(string entryName, ISet<string> taken, out bool renamed)
        {
            if (taken.Add(entryName))
            {
                renamed = false;
                return entryName;
            }

            renamed = true;
            var directory = entryName.Substring(0, entryName.LastIndexOf('/') + 1);
            var file = entryName.Substring(directory.Length);
            var dot = file.LastIndexOf('.');
            var stem = dot > 0 ? file.Substring(0, dot) : file;
            var extension = dot > 0 ? file.Substring(dot) : string.Empty;
            for (var number = 2; ; number++)
            {
                var candidate = directory + stem + " (" + number.ToString(System.Globalization.CultureInfo.InvariantCulture) + ")" + extension;
                if (taken.Add(candidate))
                {
                    return candidate;
                }
            }
        }

        private static bool TryRelativeTo(string root, string fullPath, out string relative)
        {
            relative = string.Empty;
            if (string.IsNullOrEmpty(root) || string.IsNullOrEmpty(fullPath))
            {
                return false;
            }

            var prefix = root.TrimEnd('\\', '/') + "\\";
            if (!fullPath.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            relative = fullPath.Substring(prefix.Length).Replace('\\', '/');
            return true;
        }

        private static string LastSegment(string path)
        {
            var cut = Math.Max(path.LastIndexOf('\\'), path.LastIndexOf('/'));
            return cut < 0 ? path : path.Substring(cut + 1);
        }
    }
}
