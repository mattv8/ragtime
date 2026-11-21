using System;
using System.IO;
using System.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Creates a unique, unreserved path in a work folder.</summary>
    public static class ScriptWorkPath
    {
        public static string Create(string folder, string fileName)
        {
            if (string.IsNullOrWhiteSpace(fileName) || fileName == "." || fileName == ".." || fileName.Any(char.IsControl)
                || fileName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0
                || fileName.IndexOfAny(new[] { '/', '\\', '<', '>', ':', '"', '|', '?', '*' }) >= 0)
            {
                throw new ArgumentException("A plain valid file name is required.", nameof(fileName));
            }
            var stem = Path.GetFileNameWithoutExtension(fileName);
            var extension = Path.GetExtension(fileName);
            return Path.Combine(Path.GetFullPath(folder), stem + "-" + Guid.NewGuid().ToString("N") + extension);
        }
    }
}
