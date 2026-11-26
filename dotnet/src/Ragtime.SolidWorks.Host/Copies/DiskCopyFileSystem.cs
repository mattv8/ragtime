using System;
using System.Collections.Generic;
using System.IO;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// The copy retention file operations over the disk. Junctions and links are never followed and never deleted: a link
    /// directly under the copies root is skipped, and a link anywhere inside a copy folder makes that folder fail, so a sweep
    /// cannot reach files outside the copies tree.
    /// </summary>
    internal sealed class DiskCopyFileSystem : ICopyFileSystem
    {
        /// <summary>Direct child folders of the root that are not links. A missing root has none.</summary>
        public IReadOnlyList<CopyFolder> ListFolders(string root)
        {
            var folders = new List<CopyFolder>();
            if (!Directory.Exists(root))
            {
                return folders;
            }

            RequirePlainFolder(root);
            foreach (var folder in Directory.GetDirectories(root))
            {
                if (!IsLink(folder))
                {
                    folders.Add(new CopyFolder(folder, Directory.GetCreationTimeUtc(folder)));
                }
            }

            return folders;
        }

        /// <summary>Every file under the folder. Throws when the folder or a folder below it is a link.</summary>
        public IReadOnlyList<string> ListFiles(string folder)
        {
            var files = new List<string>();
            CollectFiles(folder, files);
            return files;
        }

        /// <summary>Deletes one file, clearing its read-only attribute first. Throws when it is in use.</summary>
        public void DeleteFile(string path)
        {
            var attributes = File.GetAttributes(path);
            if ((attributes & FileAttributes.ReadOnly) != 0)
            {
                File.SetAttributes(path, attributes & ~FileAttributes.ReadOnly);
            }

            File.Delete(path);
        }

        /// <summary>Deletes an empty folder tree. Throws when a file is left in it or the folder is a link.</summary>
        public void DeleteFolder(string folder)
        {
            RequirePlainFolder(folder);
            if (Directory.GetFiles(folder, "*", SearchOption.AllDirectories).Length > 0)
            {
                throw new IOException("The copy folder still holds files.");
            }

            Directory.Delete(folder, true);
        }

        private static void CollectFiles(string folder, List<string> files)
        {
            RequirePlainFolder(folder);
            files.AddRange(Directory.GetFiles(folder));
            foreach (var subfolder in Directory.GetDirectories(folder))
            {
                CollectFiles(subfolder, files);
            }
        }

        private static void RequirePlainFolder(string folder)
        {
            if (IsLink(folder))
            {
                throw new IOException("A junction or link is not followed by copy retention.");
            }
        }

        private static bool IsLink(string path) => (File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0;
    }
}
