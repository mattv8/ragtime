using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>One folder directly under the copies root.</summary>
    public sealed class CopyFolder
    {
        /// <summary>Creates the entry.</summary>
        public CopyFolder(string path, DateTime createdUtc)
        {
            Path = path;
            CreatedUtc = createdUtc;
        }

        /// <summary>Full path of the folder.</summary>
        public string Path { get; }

        /// <summary>When the folder was created, in UTC.</summary>
        public DateTime CreatedUtc { get; }
    }

    /// <summary>
    /// The file operations the copy retention sweep needs. D2 implements it over the disk. Every member may throw; the sweep
    /// records the failure and continues.
    /// </summary>
    public interface ICopyFileSystem
    {
        /// <summary>Direct child folders of <paramref name="root"/> (files are not returned), each with its creation time.</summary>
        IReadOnlyList<CopyFolder> ListFolders(string root);

        /// <summary>Every file under <paramref name="folder"/>, at any depth.</summary>
        IReadOnlyList<string> ListFiles(string folder);

        /// <summary>Deletes one file. Throws when the file cannot be deleted (for example it is in use).</summary>
        void DeleteFile(string path);

        /// <summary>Deletes the folder and its subfolders. Throws when anything is left in it.</summary>
        void DeleteFolder(string folder);
    }

    /// <summary>What one retention sweep did.</summary>
    public sealed class CopyRetentionResult
    {
        /// <summary>Creates the result. <paramref name="openStateKnown"/> is false when an open document's path could not be read.</summary>
        public CopyRetentionResult(int deletedFolders, int skippedOpenFolders, IReadOnlyList<string> failedPaths, bool openStateKnown = true)
        {
            DeletedFolders = deletedFolders;
            SkippedOpenFolders = skippedOpenFolders;
            FailedPaths = failedPaths;
            OpenStateKnown = openStateKnown;
        }

        /// <summary>Copy folders removed completely.</summary>
        public int DeletedFolders { get; }

        /// <summary>Expired copy folders kept because a document in them is open, or because the open state is unknown.</summary>
        public int SkippedOpenFolders { get; }

        /// <summary>Files or folders that could not be deleted this time. They are retried by the next sweep.</summary>
        public IReadOnlyList<string> FailedPaths { get; }

        /// <summary>False when an open document's path could not be read: the sweep then deleted nothing (review finding 3).</summary>
        public bool OpenStateKnown { get; }
    }

    /// <summary>
    /// Deletes copy folders under the work folder's copies tree once they are older than <see cref="MaxAge"/> (contract §5.5).
    /// Only folders named with a generated copy id (32 hex digits, as <c>Guid.ToString("N")</c>) are considered. A folder that
    /// holds a document open in an agent-started instance is kept. When any open document's path is unknown (null), nothing
    /// is deleted that run. Per-file failures are tolerated: the folder stays and the next sweep tries again. The sweep itself
    /// never throws. D2 runs it at serve start and hourly.
    /// </summary>
    public sealed class CopyRetention
    {
        /// <summary>Name of the folder under the work folder that holds the copies.</summary>
        public const string CopiesFolderName = "copies";

        /// <summary>Copy folders older than this are deleted.</summary>
        public static readonly TimeSpan MaxAge = TimeSpan.FromDays(7);

        /// <summary>Length of a generated copy folder name (<c>Guid.ToString("N")</c>).</summary>
        private const int GeneratedNameLength = 32;

        private readonly ICopyFileSystem _files;

        /// <summary>Creates the sweeper over a file system.</summary>
        public CopyRetention(ICopyFileSystem files)
        {
            _files = files ?? throw new ArgumentNullException(nameof(files));
        }

        /// <summary>The copies root for a work folder: <c>&lt;work folder&gt;\copies</c>.</summary>
        public static string CopiesRoot(string workFolder) => Path.Combine(workFolder, CopiesFolderName);

        /// <summary>True when a folder created at <paramref name="createdUtc"/> is older than <see cref="MaxAge"/> at <paramref name="nowUtc"/>.</summary>
        public static bool IsExpired(DateTime createdUtc, DateTime nowUtc) => nowUtc - createdUtc > MaxAge;

        /// <summary>True when the folder's own name is a generated copy id: 32 hexadecimal digits, nothing else.</summary>
        public static bool IsGeneratedFolder(string folderPath)
        {
            // Split on the Windows separator by hand: Path.GetFileName is platform-specific and the tests run on Linux.
            var trimmed = folderPath.TrimEnd('\\');
            var name = trimmed.Substring(trimmed.LastIndexOf('\\') + 1);
            return name.Length == GeneratedNameLength && name.All(IsHexDigit);
        }

        /// <summary>
        /// Deletes the expired generated-id folders directly under <paramref name="copiesRoot"/>. A folder containing any of
        /// <paramref name="openDocumentPaths"/> is skipped. A null entry means an open document's path could not be read: the
        /// sweep then deletes nothing. Returns the counts and the paths that could not be deleted.
        /// </summary>
        public CopyRetentionResult Sweep(string copiesRoot, DateTime nowUtc, IEnumerable<string?> openDocumentPaths)
        {
            if (openDocumentPaths == null)
            {
                throw new ArgumentNullException(nameof(openDocumentPaths));
            }

            var open = openDocumentPaths.ToList();
            var failed = new List<string>();
            var deleted = 0;
            var skipped = 0;
            var openStateKnown = !open.Any(path => path == null);

            var folders = ListFoldersOrNull(copiesRoot, failed)
                .Where(folder => IsGeneratedFolder(folder.Path))
                .Where(folder => IsExpired(folder.CreatedUtc, nowUtc))
                .ToList();

            if (!openStateKnown)
            {
                return new CopyRetentionResult(0, folders.Count, failed, openStateKnown: false);
            }

            foreach (var folder in folders)
            {
                if (open.Any(path => IsInside(path!, folder.Path)))
                {
                    skipped++;
                    continue;
                }

                if (DeleteFolder(folder.Path, failed))
                {
                    deleted++;
                }
            }

            return new CopyRetentionResult(deleted, skipped, failed);
        }

        private IReadOnlyList<CopyFolder> ListFoldersOrNull(string copiesRoot, List<string> failed)
        {
            try
            {
                return _files.ListFolders(copiesRoot);
            }
            catch (Exception)
            {
                failed.Add(copiesRoot);
                return new List<CopyFolder>();
            }
        }

        /// <summary>Deletes every file, then the folder. Returns false (and records the failures) when anything stays.</summary>
        private bool DeleteFolder(string folder, List<string> failed)
        {
            IReadOnlyList<string> files;
            try
            {
                files = _files.ListFiles(folder);
            }
            catch (Exception)
            {
                failed.Add(folder);
                return false;
            }

            var allFilesDeleted = true;
            foreach (var file in files)
            {
                try
                {
                    _files.DeleteFile(file);
                }
                catch (Exception)
                {
                    failed.Add(file);
                    allFilesDeleted = false;
                }
            }

            if (!allFilesDeleted)
            {
                return false;
            }

            try
            {
                _files.DeleteFolder(folder);
                return true;
            }
            catch (Exception)
            {
                failed.Add(folder);
                return false;
            }
        }

        private static bool IsHexDigit(char value) =>
            (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') || (value >= 'A' && value <= 'F');

        /// <summary>True when <paramref name="path"/> is inside <paramref name="folder"/> at any depth (case-insensitive).</summary>
        private static bool IsInside(string path, string folder) =>
            path.StartsWith(folder.TrimEnd('\\') + "\\", StringComparison.OrdinalIgnoreCase);
    }
}
