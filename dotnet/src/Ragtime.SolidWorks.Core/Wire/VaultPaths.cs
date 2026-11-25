using System;
using System.Collections.Generic;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Platform-independent rules for vault paths.</summary>
    public static class VaultPaths
    {
        private static readonly char[] Forbidden = { ':', '*', '?', '"', '<', '>', '|' };

        /// <summary>Normalizes a vault-relative path or a path under the local view root.</summary>
        public static string Normalize(string? input, string? viewRoot, string field)
        {
            if (string.IsNullOrEmpty(input))
                throw HostError.InvalidArguments(field, "The path is required.");
            if (input!.Length > 260)
                throw HostError.InvalidArguments(field, "The path is longer than 260 characters.");
            foreach (var ch in input)
                if (ch < 0x20 || ch == 0x7f)
                    throw HostError.InvalidArguments(field, "The path contains control characters.");

            var path = input.Replace('/', '\\');
            if (viewRoot != null && path.StartsWith(viewRoot, StringComparison.OrdinalIgnoreCase)
                && (path.Length == viewRoot.Length || path[viewRoot.Length] == '\\'))
                path = path.Substring(viewRoot.Length);
            if (path.Length == 0)
                path = "\\";
            if (path[0] != '\\' || path.StartsWith(@"\\", StringComparison.Ordinal))
                throw HostError.InvalidArguments(field, "Give the path relative to the vault, starting with a backslash.");
            if (path.IndexOfAny(Forbidden) >= 0)
                throw HostError.InvalidArguments(field, "The path contains characters that are not allowed in a file name.");
            if (path == "\\")
                return path;
            if (path.EndsWith("\\", StringComparison.Ordinal))
                path = path.Substring(0, path.Length - 1);
            foreach (var segment in path.Substring(1).Split('\\'))
                if (segment.Length == 0 || segment == "." || segment == ".."
                    || segment[0] == ' ' || segment[segment.Length - 1] == ' ' || segment[segment.Length - 1] == '.')
                    throw HostError.InvalidArguments(field, "The path contains an invalid segment.");
            return path;
        }

        /// <summary>Tests whether a path lies in an approved folder.</summary>
        public static bool IsApproved(string normalizedPath, IReadOnlyList<string> approvedFolders)
        {
            if (normalizedPath == "\\") return false;
            foreach (var folder in approvedFolders)
                if (folder != "\\" && (string.Equals(normalizedPath, folder, StringComparison.OrdinalIgnoreCase)
                    || normalizedPath.StartsWith(folder + "\\", StringComparison.OrdinalIgnoreCase)))
                    return true;
            return false;
        }

        /// <summary>Returns the parent of a normalized path.</summary>
        public static string Parent(string path)
        {
            var index = path.LastIndexOf('\\');
            return index <= 0 ? "\\" : path.Substring(0, index);
        }

        /// <summary>Joins a normalized folder and a name.</summary>
        public static string Combine(string folder, string name) => folder == "\\" ? folder + name : folder + "\\" + name;

        /// <summary>Returns a problem for an invalid single file name.</summary>
        public static string? FindNameProblem(string? name)
        {
            if (string.IsNullOrEmpty(name) || name!.Length > 255 || name == "." || name == ".."
                || name[0] == ' ' || name[name.Length - 1] == ' ' || name[name.Length - 1] == '.'
                || name.IndexOfAny(Forbidden) >= 0 || name.IndexOf('\\') >= 0 || name.IndexOf('/') >= 0)
                return "The name is not a valid file name.";
            foreach (var ch in name)
                if (ch < 0x20 || ch == 0x7f)
                    return "The name contains control characters.";
            return null;
        }
    }
}
