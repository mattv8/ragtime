using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using EPDM.Interop.epdm;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Pdm
{
    /// <summary>Read operations: existence, file state, listings, variables, latest-version copies and link checks.</summary>
    internal sealed partial class PdmVault
    {
        /// <inheritdoc />
        public bool FolderExists(string vaultPath) => Read("check the folder", () => FindFolder(vaultPath) != null);

        /// <inheritdoc />
        public bool FileExists(string vaultPath) => Read("check the file", () => FindFile(vaultPath, out _) != null);

        /// <inheritdoc />
        public PdmFileState GetFile(string vaultPath) => Read("read the file", () =>
        {
            var (file, folder) = RequireFile(vaultPath);
            return ReadState(vaultPath, file, folder);
        });

        /// <inheritdoc />
        public IReadOnlyList<PdmFolderEntry> ListFolder(string vaultPath) => Read("list the folder", () =>
        {
            var folder = FindFolder(vaultPath) ?? throw PdmErrors.NotFound(vaultPath);
            var entries = new List<PdmFolderEntry>();
            AddSubFolders(folder, entries);
            AddFiles(folder, entries, vaultPath);
            return entries;
        });

        /// <inheritdoc />
        public PdmVariableSet GetVariables(string vaultPath, string configuration) => Read("read the variables", () =>
        {
            var (file, _) = RequireFile(vaultPath);
            return new PdmVariableSet
            {
                Configurations = ReadConfigurations(file),
                Variables = ReadVariables(file, configuration),
            };
        });

        /// <inheritdoc />
        public PdmGetOutcome GetLatest(string vaultPath, bool includeReferences, int maxFiles, Func<bool> shouldStop)
        {
            var outcome = new PdmGetOutcome();
            var status = "error";
            try
            {
                return Read("get the latest version", () =>
                {
                    var (file, folder) = RequireFile(vaultPath);
                    var targets = new List<FetchTarget> { new FetchTarget(vaultPath, file, folder.ID) };
                    if (targets.Count > maxFiles) throw PdmErrors.TooManyFiles(maxFiles);
                    if (includeReferences)
                    {
                        CollectReferences(file, folder, targets, outcome.Skipped, maxFiles);
                    }

                    foreach (var target in targets) RequireNoLinks(target.VaultPath);

                    var stopped = false;
                    foreach (var target in targets)
                    {
                        Fetch(target, outcome, shouldStop, ref stopped);
                    }
                    status = "ok";
                    return outcome;
                });
            }
            finally
            {
                var counts = new SortedDictionary<string, int>(StringComparer.Ordinal);
                foreach (var skipped in outcome.Skipped)
                {
                    counts.TryGetValue(skipped.Reason, out var count);
                    counts[skipped.Reason] = count + 1;
                }
                var summary = new List<string>();
                foreach (var count in counts) summary.Add(count.Key + "=" + count.Value);
                Log("file.get_latest", vaultPath, null, null,
                    status + " fetched=" + outcome.Fetched.Count + " skipped={" + string.Join(",", summary) + "}");
            }
        }

        /// <inheritdoc />
        public void RequireNoLinks(string vaultPath)
        {
            var local = LocalPath(vaultPath);
            while (local.Length > Root.Length)
            {
                if (PdmLinkCheck.IsNameSurrogate(local))
                {
                    throw PdmErrors.LinkInPath(vaultPath);
                }

                local = Path.GetDirectoryName(local) ?? Root;
            }
        }

        private PdmFileState ReadState(string vaultPath, IEdmFile5 file, IEdmFolder5 folder)
        {
            var localPath = LocalPath(vaultPath);
            var existsLocally = File.Exists(localPath);
            var locked = file.IsLocked;
            var latestVersion = LatestVersionOf(file);
            return new PdmFileState
            {
                VaultPath = vaultPath,
                LocalPath = localPath,
                Name = file.Name,
                FileId = file.ID,
                ExistsLocally = existsLocally,
                LocalVersion = existsLocally ? ReadLocalVersion(file, folder) : null,
                LatestVersion = latestVersion,
                EverCheckedIn = HasCheckedInVersion(file, latestVersion),
                CheckedOut = locked,
                CheckedOutBy = locked ? file.LockedByUser?.Name : null,
                CheckedOutOn = locked ? file.LockedOnComputer : null,
                LockedByUserId = locked ? file.LockedByUserID : (int?)null,
                LockPath = locked ? file.LockPath : null,
                CheckedOutHere = IsCheckedOutHere(file),
                WorkflowState = ReadWorkflowState(file),
            };
        }

        /// <summary>The version reported by PDM, or 0 when it is not positive.</summary>
        private static int LatestVersionOf(IEdmFile5 file) => file.CurrentVersion > 0 ? file.CurrentVersion : 0;

        /// <summary>Positive evidence of a checked-in version; false means unknown, not never.</summary>
        private static bool HasCheckedInVersion(IEdmFile5 file, int currentVersion)
        {
            if (currentVersion > 1) return true;
            try
            {
                return file.CurrentState is IEdmState6 state && state.WorkflowID > 0;
            }
            catch (COMException)
            {
                return false;
            }
        }

        private static string? ReadWorkflowState(IEdmFile5 file)
        {
            try { return file.CurrentState?.Name; }
            catch (COMException) { return null; }
        }

        private static int? ReadLocalVersion(IEdmFile5 file, IEdmFolder5 folder)
        {
            object folderId = folder.ID;
            var version = file.GetLocalVersionNo(ref folderId);
            return version < 0 ? (int?)null : version;
        }

        private static void AddSubFolders(IEdmFolder5 folder, List<PdmFolderEntry> entries)
        {
            foreach (var child in SubFolders(folder))
            {
                entries.Add(new PdmFolderEntry { Name = child.Name, IsFolder = true });
            }
        }

        private static IEnumerable<IEdmFolder5> SubFolders(IEdmFolder5 folder)
        {
            var position = folder.GetFirstSubFolderPosition();
            while (!position.IsNull)
            {
                yield return folder.GetNextSubFolder(position);
            }
        }

        private void AddFiles(IEdmFolder5 folder, List<PdmFolderEntry> entries, string vaultPath)
        {
            var position = folder.GetFirstFilePosition();
            while (!position.IsNull)
            {
                var file = folder.GetNextFile(position);
                entries.Add(new PdmFolderEntry
                {
                    Name = file.Name,
                    IsFolder = false,
                    LatestVersion = LatestVersionOf(file),
                    LocalVersion = File.Exists(LocalPath(JoinVault(vaultPath, file.Name))) ? ReadLocalVersion(file, folder) : null,
                    CheckedOut = file.IsLocked,
                });
            }
        }

        private static List<string> ReadConfigurations(IEdmFile5 file)
        {
            object version = 0;
            var names = new List<string>();
            var list = file.GetConfigurations(ref version);
            var position = list.GetHeadPosition();
            while (!position.IsNull)
            {
                names.Add(list.GetNext(position));
            }

            return names;
        }

        private List<PdmVariableValue> ReadVariables(IEdmFile5 file, string configuration)
        {
            var values = new List<PdmVariableValue>();
            var enumerator = file.GetEnumeratorVariable(string.Empty);
            try
            {
                foreach (var name in VariableNames())
                {
                    object? raw;
                    if (!enumerator.GetVar(name, configuration, out raw))
                    {
                        continue;
                    }

                    var text = Convert.ToString(raw, CultureInfo.InvariantCulture);
                    if (!string.IsNullOrEmpty(text))
                    {
                        values.Add(new PdmVariableValue { Name = name, Value = text });
                    }
                }
            }
            finally
            {
                ((IEdmEnumeratorVariable8)enumerator).CloseFile(false);
            }

            return values;
        }

        private void CollectReferences(IEdmFile5 root, IEdmFolder5 rootFolder, List<FetchTarget> targets, List<PdmSkippedFile> skipped, int maxFiles)
        {
            var seen = new HashSet<int> { root.ID };
            // A child without a File is reached once per parent; report each of its display paths once.
            var skippedPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var stack = new Stack<IEdmReference5>();
            stack.Push(root.GetReferenceTree(rootFolder.ID, 0));
            while (stack.Count > 0)
            {
                var parent = stack.Pop();
                var configuration = "@";
                var position = parent.GetFirstChildPosition(ref configuration, false, false, 0);
                while (!position.IsNull)
                {
                    var child = parent.GetNextChild(position);
                    var file = child.File;
                    if (file != null && !seen.Add(file.ID)) continue;

                    var hasPath = TryReferenceVaultPath(child.Folder, file?.Name ?? child.Name, out var vaultPath);
                    // Internal components can report a folder/path but live inside their parent, not as fetchable files.
                    if (file is IEdmFile8 kind && kind.FileType == EdmFileType.Edmft_Normal && child.FolderID > 0 && hasPath)
                    {
                        targets.Add(new FetchTarget(vaultPath, file, child.FolderID));
                        if (targets.Count > maxFiles) throw PdmErrors.TooManyFiles(maxFiles);
                    }
                    else if (file != null || skippedPaths.Add(vaultPath))
                    {
                        skipped.Add(new PdmSkippedFile { VaultPath = vaultPath, Reason = "not_in_vault" });
                    }

                    if (file != null) stack.Push(child);
                }
            }
        }

        /// <summary>Copies the latest version of one file to the view. Never overwrites a file checked out here.</summary>
        private void Fetch(FetchTarget target, PdmGetOutcome outcome, Func<bool> shouldStop, ref bool stopped)
        {
            string? reason = null;
            stopped = stopped || shouldStop();
            if (stopped)
            {
                reason = "deadline";
            }
            else if (IsCheckedOutHere(target.File))
            {
                reason = "checked_out_here";
            }
            else
            {
                var localPath = LocalPath(target.VaultPath);
                if (File.Exists(localPath))
                {
                    object localFolderId = target.FolderId;
                    if (target.File.GetLocalVersionNo(ref localFolderId) <= 0)
                    {
                        reason = "local_untracked";
                    }
                    else if (IsLocalModified(target, localPath))
                    {
                        reason = "local_modified";
                    }
                }
            }

            if (reason == null && shouldStop())
            {
                stopped = true;
                reason = "deadline";
            }
            if (reason != null)
            {
                outcome.Skipped.Add(new PdmSkippedFile { VaultPath = target.VaultPath, Reason = reason });
                return;
            }
            object version = 0;
            object folderId = target.FolderId;
            // Window handle 0, EdmGet_Simple: no dialog, no read-only switch, no reference cascade.
            target.File.GetFileCopy(0, ref version, ref folderId, (int)EdmGetFlag.EdmGet_Simple, string.Empty);
            outcome.Fetched.Add(target.VaultPath);
        }

        private static bool IsLocalModified(FetchTarget target, string localPath)
        {
            try
            {
                object folderId = target.FolderId;
                var date = target.File.GetLocalFileDate(ref folderId);
                if (!(date is DateTime recorded) || recorded == DateTime.MinValue) return true;
                var actual = File.GetLastWriteTimeUtc(localPath);
                return Math.Abs((actual - recorded.ToUniversalTime()).TotalSeconds) > 2;
            }
            catch (Exception)
            {
                return true;
            }
        }

        private sealed class FetchTarget
        {
            public FetchTarget(string vaultPath, IEdmFile5 file, int folderId)
            {
                VaultPath = vaultPath;
                File = file;
                FolderId = folderId;
            }

            public string VaultPath { get; }

            public IEdmFile5 File { get; }

            public int FolderId { get; }
        }
    }
}
