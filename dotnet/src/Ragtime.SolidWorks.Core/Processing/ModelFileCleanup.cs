using System;
using System.IO;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Identifies and removes extra files written alongside generated model files.</summary>
    public static class ModelFileCleanup
    {
        /// <summary>Matches a sibling name beginning with the target stem, excluding the target itself.</summary>
        public static bool IsStrayFile(string targetName, string candidateName) =>
            !string.Equals(targetName, candidateName, StringComparison.OrdinalIgnoreCase)
            && candidateName.StartsWith(Path.GetFileNameWithoutExtension(targetName), StringComparison.OrdinalIgnoreCase);

        internal static void DeleteStrays(string target)
        {
            try
            {
                var folder = Path.GetDirectoryName(Path.GetFullPath(target))!;
                foreach (var file in Directory.EnumerateFiles(folder))
                {
                    try
                    {
                        if (IsStrayFile(Path.GetFileName(target), Path.GetFileName(file))
                            && (File.GetAttributes(file) & (FileAttributes.ReparsePoint | FileAttributes.Directory)) == 0)
                        {
                            File.Delete(file);
                        }
                    }
                    catch (Exception)
                    {
                        // Best effort: continue with the remaining files.
                    }
                }
            }
            catch (Exception)
            {
                // Best effort: never mask the export or preview result.
            }
        }
    }
}
