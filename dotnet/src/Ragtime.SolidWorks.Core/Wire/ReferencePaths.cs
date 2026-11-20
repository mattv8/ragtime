using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// Maps a PDM reference to its path in the local view (the rule that decides whether get_latest may fetch it).
    /// The rules are string rules, so they give the same answer on every operating system: the path is normalised
    /// the way GetFullPath would, without touching the file system.
    /// </summary>
    public static class ReferencePaths
    {
        private static readonly Regex DrivePattern = new Regex(@"^[A-Za-z]:\\", RegexOptions.CultureInvariant);
        private static readonly char[] InvalidNameChars = { '"', '<', '>', '|', ':', '*', '?', '\\', '/' };
        private static readonly char[] InvalidFolderChars = { '"', '<', '>', '|', ':', '*', '?' };

        /// <summary>
        /// Returns the vault path (with its leading backslash) of the file <paramref name="name"/> in the folder at
        /// <paramref name="folderLocalPath"/>, or null when the file is not inside the view at <paramref name="root"/>.
        /// The folder must be an absolute drive path. The path is normalised first, so ".", ".." and repeated separators
        /// resolve, and a path that escapes the root fails. The name must be a valid Windows file name, not "." or "..",
        /// and must not end in a dot or a space.
        /// </summary>
        public static string? VaultPath(string root, string? folderLocalPath, string? name)
        {
            if (!IsRepresentableName(name) || string.IsNullOrEmpty(folderLocalPath)) return null;
            var localPath = FullLocalPath(folderLocalPath!, name!);
            if (localPath == null) return null;
            var rootPrefix = root.TrimEnd('\\') + @"\";
            if (!localPath.StartsWith(rootPrefix, StringComparison.OrdinalIgnoreCase)) return null;
            return localPath.Substring(rootPrefix.Length - 1);
        }

        private static bool IsRepresentableName(string? name) =>
            !string.IsNullOrEmpty(name)
            && name != "."
            && name != ".."
            && !name!.EndsWith(".", StringComparison.Ordinal)
            && !name.EndsWith(" ", StringComparison.Ordinal)
            && name.IndexOfAny(InvalidNameChars) < 0
            && !HasControlCharacter(name);

        /// <summary>Joins the folder and name, then resolves "." and ".." segments. Null when the folder is not a drive path.</summary>
        private static string? FullLocalPath(string folderLocalPath, string name)
        {
            var folder = folderLocalPath.Replace('/', '\\');
            if (!DrivePattern.IsMatch(folder) || folder.IndexOfAny(InvalidFolderChars, 2) >= 0 || HasControlCharacter(folder))
            {
                return null;
            }

            var combined = folder.TrimEnd('\\') + @"\" + name;
            var segments = new List<string>();
            foreach (var segment in combined.Substring(3).Split('\\'))
            {
                if (segment.Length == 0 || segment == ".") continue;
                if (segment != "..")
                {
                    segments.Add(segment);
                }
                else if (segments.Count > 0)
                {
                    segments.RemoveAt(segments.Count - 1);
                }
            }

            return combined.Substring(0, 2) + @"\" + string.Join(@"\", segments);
        }

        private static bool HasControlCharacter(string text)
        {
            foreach (var ch in text)
            {
                if (ch < 0x20) return true;
            }

            return false;
        }
    }
}
