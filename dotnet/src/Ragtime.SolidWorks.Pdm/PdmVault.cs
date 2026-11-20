using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using EPDM.Interop.epdm;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Pdm
{
    /// <summary>
    /// One logged-in vault session, created and used only on the PDM thread. Reads and writes live in the partial files;
    /// this file holds the shared lookups. Every call passes window handle 0 and no dialog option.
    /// </summary>
    internal sealed partial class PdmVault : IPdmVault
    {
        private readonly IEdmVault5 _vault;
        private readonly Action<string> _log;
        private readonly int _userId;
        private HashSet<string>? _variableNames;

        internal PdmVault(IEdmVault5 vault, string name, string root, string user, int userId, Action<string> log)
        {
            _vault = vault;
            Vault = name;
            Root = root;
            User = user;
            _userId = userId;
            _log = log;
        }

        /// <inheritdoc />
        public string Vault { get; }

        /// <inheritdoc />
        public string Root { get; }

        /// <inheritdoc />
        public string User { get; }

        /// <summary>True while PDM still reports this session as logged in.</summary>
        internal bool IsLoggedIn
        {
            get
            {
                try
                {
                    return _vault.IsLoggedIn;
                }
                catch (COMException)
                {
                    return false;
                }
            }
        }

        /// <summary>Drops the COM reference. Runs on the PDM thread.</summary>
        internal void Release()
        {
            Marshal.ReleaseComObject(_vault);
        }

        private string LocalPath(string vaultPath) => vaultPath == @"\" ? Root : Root + vaultPath;

        private static string JoinVault(string folderVaultPath, string name) =>
            folderVaultPath == @"\" ? @"\" + name : folderVaultPath + @"\" + name;

        private IEdmFolder5? FindFolder(string vaultPath)
        {
            try
            {
                return _vault.GetFolderFromPath(LocalPath(vaultPath));
            }
            catch (COMException error) when (PdmErrors.IsNotFound(error))
            {
                return null;
            }
        }

        private IEdmFile5? FindFile(string vaultPath, out IEdmFolder5? folder)
        {
            try
            {
                return _vault.GetFileFromPath(LocalPath(vaultPath), out folder);
            }
            catch (COMException error) when (PdmErrors.IsNotFound(error))
            {
                folder = null;
                return null;
            }
        }

        private (IEdmFile5, IEdmFolder5) RequireFile(string vaultPath)
        {
            IEdmFolder5? folder = null;
            var file = Read("find the file", () => FindFile(vaultPath, out folder));
            if (file == null || folder == null)
            {
                throw PdmErrors.NotFound(vaultPath);
            }

            return (file, folder);
        }

        private T Read<T>(string operation, Func<T> read)
        {
            try
            {
                return read();
            }
            catch (COMException error)
            {
                throw PdmErrors.Api(operation, error);
            }
        }

        /// <summary>True when the file is checked out by this user on this computer.</summary>
        private bool IsCheckedOutHere(IEdmFile5 file) =>
            file.IsLocked
            && file.LockedByUserID == _userId
            && string.Equals(file.LockedOnComputer, Environment.MachineName, StringComparison.OrdinalIgnoreCase);

        /// <summary>Vault path of a folder, built from its parent chain up to the vault root.</summary>
        private string FolderVaultPath(IEdmFolder5 folder)
        {
            var segments = new List<string>();
            var rootId = _vault.RootFolderID;
            var current = folder;
            while (current != null && current.ID != rootId)
            {
                segments.Insert(0, current.Name);
                current = current.ParentFolder;
            }

            return segments.Count == 0 ? @"\" : @"\" + string.Join(@"\", segments);
        }

        private string VaultPathOf(IEdmFolder5? folder, string name) =>
            folder == null ? @"\" + name : JoinVault(FolderVaultPath(folder), name);

        /// <summary>Names of every vault variable, read once per session.</summary>
        private HashSet<string> VariableNames()
        {
            if (_variableNames != null)
            {
                return _variableNames;
            }

            var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            // CreateUtility returns a VARIANT, which the embedded interop types as dynamic; widen it to object before the cast.
            object utility = ((IEdmVault8)_vault).CreateUtility(EdmUtility.EdmUtil_VariableMgr);
            var manager = (IEdmVariableMgr5)utility;
            var position = manager.GetFirstVariablePosition();
            while (!position.IsNull)
            {
                names.Add(manager.GetNextVariable(position).Name);
            }

            _variableNames = names;
            return names;
        }
    }
}
