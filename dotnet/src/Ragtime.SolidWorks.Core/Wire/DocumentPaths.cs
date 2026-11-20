using System;
using System.Text.RegularExpressions;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// Path rules for document.open (contract section 5 and decision D13). The rules are string rules, so
    /// they give the same answer on every operating system. Only local drive paths to SolidWorks files pass.
    /// </summary>
    public static class DocumentPaths
    {
        /// <summary>Longest accepted path, in characters.</summary>
        public const int MaxLength = 260;

        private static readonly string[] AllowedExtensions = { ".sldprt", ".sldasm", ".slddrw" };
        private static readonly Regex DrivePattern = new Regex(@"^[A-Za-z]:\\", RegexOptions.CultureInvariant);
        private static readonly char[] ForbiddenInName = { '*', '?', '"', '<', '>', '|' };

        /// <summary>
        /// Returns null when the path may be opened, otherwise one readable sentence explaining the refusal.
        /// Rules: non-empty; at most 260 characters; no control characters; not UNC, \\?\ or \\.\; absolute
        /// drive path X:\...; backslashes only; no empty or ".." segments; no ':' after the drive (alternate data
        /// streams); no wildcard characters; extension .sldprt, .sldasm or .slddrw (case-insensitive).
        /// </summary>
        public static string? FindProblem(string? path)
        {
            var problem = FindLocalFileProblem(path);
            if (problem != null) return problem;
            var fileName = path!.Substring(path.LastIndexOf('\\') + 1);
            var dot = fileName.LastIndexOf('.');
            if (dot < 0 || !IsAllowedExtension(fileName.Substring(dot)))
                return "The file must end in .sldprt, .sldasm or .slddrw.";
            return null;
        }

        /// <summary>Checks local-file rules without requiring a SolidWorks extension.</summary>
        public static string? FindLocalFileProblem(string? path)
        {
            if (path == null || path.Length == 0)
            {
                return "The path is required.";
            }

            if (path.Length > MaxLength)
            {
                return "The path is longer than 260 characters.";
            }

            foreach (var ch in path)
            {
                if (ch < 0x20 || ch == 0x7F)
                {
                    return "The path contains control characters.";
                }
            }

            if (path.StartsWith(@"\\", StringComparison.Ordinal))
            {
                return @"UNC and device paths (\\server, \\?\, \\.\) are not allowed; use a local drive path.";
            }

            if (!DrivePattern.IsMatch(path))
            {
                return @"The path must be absolute on a local drive, such as C:\Models\part.sldprt.";
            }

            if (path.IndexOf('/') >= 0)
            {
                return "Use backslashes in the path.";
            }

            if (path.IndexOf(':', 2) >= 0)
            {
                return "The path must not contain ':' after the drive letter.";
            }

            if (path.IndexOfAny(ForbiddenInName) >= 0)
            {
                return "The path contains characters that are not allowed in a file name.";
            }

            var segments = path.Substring(3).Split('\\');
            foreach (var segment in segments)
            {
                if (segment.Length == 0 || segment == "..")
                {
                    return "The path must not contain empty or '..' segments.";
                }
            }

            return null;
        }

        /// <summary>Throws invalid_arguments (field "path") when <see cref="FindProblem"/> reports a problem.</summary>
        public static void RequireOpenable(string? path)
        {
            var problem = FindProblem(path);
            if (problem != null)
            {
                throw HostError.InvalidArguments("path", problem);
            }
        }

        /// <summary>Case-insensitive comparison used for matching open documents by path.</summary>
        public static bool SamePath(string? left, string? right) =>
            left != null && right != null && string.Equals(left, right, StringComparison.OrdinalIgnoreCase);

        private static bool IsAllowedExtension(string extension)
        {
            foreach (var allowed in AllowedExtensions)
            {
                if (string.Equals(extension, allowed, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }
    }
}
