using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using EPDM.Interop.epdm;
using Newtonsoft.Json;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Pdm
{
    /// <summary>
    /// Write operations. Every PDM write call goes through <see cref="Mutate"/>, which logs one line per call that
    /// reached PDM. Preconditions run before it and log nothing. No call asks for a window or a dialog.
    /// </summary>
    internal sealed partial class PdmVault
    {
        private const long MaxSourceBytes = 512L * 1024 * 1024;

        /// <inheritdoc />
        public void CreateFolder(string vaultPath)
        {
            var index = vaultPath.LastIndexOf('\\');
            var parentPath = index == 0 ? @"\" : vaultPath.Substring(0, index);
            var name = vaultPath.Substring(index + 1);
            var parent = Read("find the parent folder", () => FindFolder(parentPath)) ?? throw PdmErrors.NotFound(parentPath);
            Mutate("folder.create", vaultPath, null, () =>
            {
                // Window handle 0; no folder data, so no property dialog.
                parent.AddFolder(0, name, null);
            });
        }

        /// <inheritdoc />
        /// <remarks>An in-place add (the source already is the target) adds without copying. Any other target that already exists on disk is refused.</remarks>
        public int AddFile(string sourcePath, string folderVaultPath, string name)
        {
            CheckSource(sourcePath);
            var folder = Read("find the folder", () => FindFolder(folderVaultPath)) ?? throw PdmErrors.NotFound(folderVaultPath);
            var inPlace = IsInFolderView(sourcePath, folderVaultPath)
                && string.Equals(Path.GetFileName(sourcePath), name, StringComparison.OrdinalIgnoreCase);
            if (!inPlace)
            {
                var localTarget = Path.Combine(LocalPath(folderVaultPath), name);
                if (File.Exists(localTarget) || Directory.Exists(localTarget))
                {
                    throw PdmErrors.LocalFileExists(JoinVault(folderVaultPath, name));
                }
            }

            var targetName = inPlace ? string.Empty : name;
            var id = 0;
            Mutate("file.add", JoinVault(folderVaultPath, name), null, () =>
            {
                // Window handle 0 and flags 0 (EdmAdd_Simple): no replace-local or duplicate-variable dialog.
                id = folder.AddFile(0, sourcePath, targetName, 0);
            });
            return id;
        }

        /// <inheritdoc />
        public void CheckOut(string vaultPath)
        {
            var (file, folder) = RequireFile(vaultPath);
            Mutate("file.checkout", vaultPath, null, () =>
            {
                // Window handle 0, EdmLock_Simple: no dialog.
                file.LockFile(folder.ID, 0, (int)EdmLockFlag.EdmLock_Simple);
            }, () => file.CurrentVersion);
        }

        /// <inheritdoc />
        public void UndoCheckOut(string vaultPath)
        {
            var (file, _) = RequireFile(vaultPath);
            Mutate("file.undo_checkout", vaultPath, null, () =>
            {
                // Window handle 0; bGetLatestVersion true. Discards local changes only.
                file.UndoLockFile(0, true);
            }, () => file.CurrentVersion);
        }

        /// <inheritdoc />
        /// <remarks>
        /// Checks in this one file only: EdmUnlock_IgnoreReferences stops the cascade to referenced files. The flags
        /// never change workflow state, remove the local copy or overwrite the latest version. No callback is passed.
        /// </remarks>
        public void CheckIn(string vaultPath, string comment, bool keepCheckedOut)
        {
            var (file, _) = RequireFile(vaultPath);
            var flags = (int)EdmUnlockFlag.EdmUnlock_IgnoreReferences
                | (keepCheckedOut ? (int)EdmUnlockFlag.EdmUnlock_KeepLocked : (int)EdmUnlockFlag.EdmUnlock_Simple);
            Mutate("file.checkin", vaultPath, comment, () =>
            {
                file.UnlockFile(0, comment, flags, null);
            }, () => file.CurrentVersion);
        }

        /// <inheritdoc />
        public IReadOnlyList<string> SetVariables(string vaultPath, string configuration, IReadOnlyDictionary<string, object> values)
        {
            var known = Read("read variable names", VariableNames);
            foreach (var name in values.Keys)
            {
                if (!known.Contains(name))
                {
                    throw PdmErrors.UnknownVariable(name);
                }
            }

            var (file, _) = RequireFile(vaultPath);
            var written = new List<string>();
            Mutate("variables.set", vaultPath, null, () =>
            {
                var enumerator = file.GetEnumeratorVariable(string.Empty);
                var succeeded = false;
                try
                {
                    foreach (var pair in values)
                    {
                        object value = pair.Value;
                        enumerator.SetVar(pair.Key, configuration, ref value, false);
                        written.Add(pair.Key);
                    }
                    succeeded = true;
                }
                catch
                {
                    try { ((IEdmEnumeratorVariable8)enumerator).CloseFile(false); }
                    catch (Exception) { }
                    throw;
                }
                if (succeeded) ((IEdmEnumeratorVariable8)enumerator).CloseFile(true);
            }, () => file.CurrentVersion);
            return written;
        }

        /// <summary>Runs one PDM write call. Logs one line for it; a COM failure is logged and mapped to Api.</summary>
        private void Mutate(string operation, string vaultPath, string? comment, Action call, Func<int>? readVersion = null)
        {
            try
            {
                call();
            }
            catch (COMException error)
            {
                Log(operation, vaultPath, comment, null, $"api_error hresult=0x{error.ErrorCode:X8}");
                throw PdmErrors.Api(operation, error);
            }

            int? version = null;
            if (readVersion != null)
            {
                try { version = readVersion(); }
                catch (Exception error) { Log(operation, vaultPath, null, null, "version_read_failed " + error.GetType().Name); }
            }
            Log(operation, vaultPath, comment, version, "ok");
        }

        private void Log(string operation, string vaultPath, string? comment, int? version, string outcome)
        {
            var line = $"{DateTime.UtcNow:O} op={JsonConvert.ToString(operation)} vault={JsonConvert.ToString(Vault)} path={JsonConvert.ToString(vaultPath)} outcome={JsonConvert.ToString(outcome)}";
            if (version.HasValue)
            {
                line += $" version={version.Value}";
            }

            if (comment != null)
            {
                line += $" comment={JsonConvert.ToString(comment)}";
            }

            try { _log(line); }
            catch (Exception) { }
        }

        private bool IsInFolderView(string sourcePath, string folderVaultPath)
        {
            var directory = Path.GetDirectoryName(sourcePath) ?? string.Empty;
            return string.Equals(
                directory.TrimEnd('\\'),
                LocalPath(folderVaultPath).TrimEnd('\\'),
                StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>Checks the source the way the contract requires; throws InvalidSource otherwise.</summary>
        private static void CheckSource(string sourcePath)
        {
            var problem = DocumentPaths.FindLocalFileProblem(sourcePath);
            if (problem != null)
            {
                throw PdmErrors.InvalidSource(sourcePath, problem);
            }

            LocalFileCheck.Require(sourcePath, "source");

            try
            {
                var info = new FileInfo(sourcePath);
                if (!info.Exists)
                {
                    throw PdmErrors.InvalidSource(sourcePath, "The source file does not exist.");
                }

                if ((info.Attributes & FileAttributes.Directory) != 0)
                {
                    throw PdmErrors.InvalidSource(sourcePath, "The source is not a regular file.");
                }
                if (info.Length > MaxSourceBytes)
                {
                    throw PdmErrors.InvalidSource(sourcePath, "The source file is larger than 512 MiB.");
                }
            }
            catch (Exception error) when (error is IOException || error is UnauthorizedAccessException)
            {
                throw PdmErrors.InvalidSource(sourcePath, error.Message);
            }
        }
    }
}
