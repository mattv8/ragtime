using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Ragtime.SolidWorks.Core
{
    public sealed class PdmViewFacts
    {
        public PdmViewFacts(string vault, string root, bool loggedIn, string? user)
        { Vault = vault; Root = root; LoggedIn = loggedIn; User = user; }
        public string Vault { get; }
        /// <summary>Local view root without a trailing backslash, e.g. C:\PDM.</summary>
        public string Root { get; }
        /// <summary>True only when this host process holds a logged-in session for the vault.</summary>
        public bool LoggedIn { get; }
        public string? User { get; }
    }

    public sealed class PdmClientFacts
    {
        public PdmClientFacts(bool clientInstalled, string? clientVersion, IReadOnlyList<PdmViewFacts> views)
        { ClientInstalled = clientInstalled; ClientVersion = clientVersion; Views = views; }
        public bool ClientInstalled { get; }
        public string? ClientVersion { get; }
        public IReadOnlyList<PdmViewFacts> Views { get; }
        public static PdmClientFacts NotInstalled { get; } = new PdmClientFacts(false, null, new PdmViewFacts[0]);
    }

    public enum PdmFailureKind
    {
        ClientMissing, ViewMissing, LoginFailed, Busy, Timeout,
        NotFound, LinkInPath, InvalidSource, UnknownVariable, TooManyFiles, Api,
        LocalFileExists,
    }

    /// <summary>The only exception type a gateway or vault session throws on purpose.</summary>
    public sealed class PdmGatewayException : Exception
    {
        public PdmGatewayException(PdmFailureKind kind, string message, string? subject = null, int? hresult = null, string? pdmError = null, Exception? inner = null)
            : base(message, inner)
        { Kind = kind; Subject = subject; HResultCode = hresult; PdmError = pdmError; }
        public PdmFailureKind Kind { get; }
        /// <summary>Vault path (NotFound, LinkInPath), variable name (UnknownVariable) or source path (InvalidSource).</summary>
        public string? Subject { get; }
        public int? HResultCode { get; }
        /// <summary>Symbolic PDM error name when known, e.g. E_EDM_PERMISSION_DENIED.</summary>
        public string? PdmError { get; }
    }

    public sealed class PdmFileState
    {
        public string VaultPath { get; set; } = string.Empty;
        public string LocalPath { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public int FileId { get; set; }
        public bool ExistsLocally { get; set; }
        /// <summary>Null when there is no local copy.</summary>
        public int? LocalVersion { get; set; }
        public int LatestVersion { get; set; }
        /// <summary>True only with positive evidence of a checked-in version. False means unknown, not never. Core does not use this for decisions.</summary>
        public bool EverCheckedIn { get; set; }
        public bool CheckedOut { get; set; }
        public string? CheckedOutBy { get; set; }
        public string? CheckedOutOn { get; set; }
        /// <summary>PDM user id holding the check-out; null when not checked out.</summary>
        public int? LockedByUserId { get; set; }
        /// <summary>Local path PDM recorded for the check-out; null when not checked out.</summary>
        public string? LockPath { get; set; }
        /// <summary>Checked out by the logged-in user on this computer.</summary>
        public bool CheckedOutHere { get; set; }
        public string? WorkflowState { get; set; }
    }

    public sealed class PdmFolderEntry
    {
        public string Name { get; set; } = string.Empty;
        public bool IsFolder { get; set; }
        public int? LatestVersion { get; set; }
        public int? LocalVersion { get; set; }
        public bool? CheckedOut { get; set; }
    }

    public sealed class PdmVariableValue
    {
        public string Name { get; set; } = string.Empty;
        public string Value { get; set; } = string.Empty;
    }

    public sealed class PdmVariableSet
    {
        public List<string> Configurations { get; set; } = new List<string>();
        public List<PdmVariableValue> Variables { get; set; } = new List<PdmVariableValue>();
    }

    public sealed class PdmSkippedFile
    {
        public string VaultPath { get; set; } = string.Empty;
        public string Reason { get; set; } = string.Empty;
    }

    public sealed class PdmGetOutcome
    {
        public List<string> Fetched { get; set; } = new List<string>();
        public List<PdmSkippedFile> Skipped { get; set; } = new List<PdmSkippedFile>();
    }

    /// <summary>
    /// One logged-in vault. Synchronous; every call happens on the gateway's STA thread inside
    /// <see cref="IPdmGateway.RunAsync{T}"/>. All paths are normalized vault paths (leading backslash, "\" = root).
    /// Methods never check approval, write permission or check-out ownership: Core does, before calling.
    /// Failures are <see cref="PdmGatewayException"/>.
    /// </summary>
    public interface IPdmVault
    {
        string Vault { get; }
        string Root { get; }
        string User { get; }

        bool FolderExists(string vaultPath);
        bool FileExists(string vaultPath);
        /// <summary>NotFound when no file is at that path (a folder there is also NotFound).</summary>
        PdmFileState GetFile(string vaultPath);
        /// <summary>Direct children. NotFound when the folder does not exist. Order is not significant.</summary>
        IReadOnlyList<PdmFolderEntry> ListFolder(string vaultPath);
        /// <summary>configuration "@" = file level. Only variables that have a value. NotFound for a missing file.</summary>
        PdmVariableSet GetVariables(string vaultPath, string configuration);
        /// <summary>
        /// Copies the latest version to the local view; with includeReferences also the latest versions of referenced
        /// files. Every target is checked before the first copy: a link in its local path is LinkInPath and nothing is
        /// fetched. Never overwrites local work: a target checked out here is skipped with reason "checked_out_here", a
        /// local file PDM does not track with "local_untracked", and a local copy changed since PDM wrote it with
        /// "local_modified". When shouldStop returns true before a copy, that file and all later ones are skipped with
        /// reason "deadline". More than maxFiles files in total: TooManyFiles, nothing fetched.
        /// </summary>
        PdmGetOutcome GetLatest(string vaultPath, bool includeReferences, int maxFiles, Func<bool> shouldStop);
        /// <summary>
        /// LinkInPath when the local path of vaultPath, or of any existing parent up to (not including) the view root,
        /// is a name-surrogate reparse point (symbolic link or junction). A component that does not exist passes.
        /// </summary>
        void RequireNoLinks(string vaultPath);

        /// <summary>Creates exactly one folder level; its parent exists.</summary>
        void CreateFolder(string vaultPath);
        /// <summary>
        /// Adds a local file to an existing vault folder under the given name and leaves it checked out to this user.
        /// The gateway checks the source itself: absolute local-drive path, exists, regular file, not reached through a
        /// link (LocalFileCheck), at most 512 MiB; otherwise InvalidSource. A source already inside the folder's local
        /// view directory with the same name is added in place; anything else is copied in.
        /// </summary>
        /// <returns>The PDM file id of the added file, as returned by the add call itself.</returns>
        int AddFile(string sourcePath, string folderVaultPath, string name);
        void CheckOut(string vaultPath);
        /// <summary>Discards local changes.</summary>
        void UndoCheckOut(string vaultPath);
        /// <summary>Checks in exactly this one file with exactly this comment. Never its references, never a workflow state change.</summary>
        void CheckIn(string vaultPath, string comment, bool keepCheckedOut);
        /// <summary>Values are string, long, double or bool. Returns the names written. UnknownVariable names the first unknown one; then nothing is written.</summary>
        IReadOnlyList<string> SetVariables(string vaultPath, string configuration, IReadOnlyDictionary<string, object> values);
    }

    /// <summary>Evidence that one tool started the current check-out of one file.</summary>
    public sealed class PdmClaim
    {
        public string Vault { get; set; } = string.Empty;
        public int FileId { get; set; }
        public string ToolId { get; set; } = string.Empty;
        /// <summary>True until this tool checks in a file it added. Missing legacy evidence defaults to true, refusing undo safely.</summary>
        public bool AddedByTool { get; set; } = true;
        public int LockedByUserId { get; set; }
        public string LockedOnComputer { get; set; } = string.Empty;
        public string LockPath { get; set; } = string.Empty;
        public DateTime ClaimedUtc { get; set; }
    }

    /// <summary>
    /// Remembers which check-outs a tool started itself, so that it never checks in, changes or discards a check-out a
    /// person or another tool started on the same machine. One claim per vault and file. Thread-safe.
    /// </summary>
    public interface IPdmClaimStore
    {
        PdmClaim? Find(string vault, int fileId);
        void Put(PdmClaim claim);
        void Remove(string vault, int fileId);
    }

    /// <summary>Claims kept for the life of the process. The host supplies a file-backed store instead.</summary>
    public sealed class InMemoryPdmClaimStore : IPdmClaimStore
    {
        private readonly Dictionary<string, PdmClaim> _claims = new Dictionary<string, PdmClaim>(StringComparer.OrdinalIgnoreCase);

        public PdmClaim? Find(string vault, int fileId)
        {
            lock (_claims) { return _claims.TryGetValue(Key(vault, fileId), out var claim) ? claim : null; }
        }

        public void Put(PdmClaim claim)
        {
            lock (_claims) { _claims[Key(claim.Vault, claim.FileId)] = claim; }
        }

        public void Remove(string vault, int fileId)
        {
            lock (_claims) { _claims.Remove(Key(vault, fileId)); }
        }

        private static string Key(string vault, int fileId) =>
            vault + "|" + fileId.ToString(System.Globalization.CultureInfo.InvariantCulture);
    }

    public interface IPdmGateway
    {
        /// <summary>No login, no network wait; safe on every status request. Never throws.</summary>
        PdmClientFacts GetClientFacts();

        /// <summary>
        /// Runs work on the gateway's single STA thread against the named vault, logging in with the machine's
        /// automatic login on first use and keeping the session. Throws PdmGatewayException: ClientMissing, ViewMissing,
        /// LoginFailed, Busy (an earlier call is still stuck), Timeout (deadline passed; the gateway stays Busy until
        /// the work returns), or whatever the work throws.
        /// </summary>
        Task<T> RunAsync<T>(string vault, Func<IPdmVault, T> work, Deadline deadline, CancellationToken cancellationToken);
    }
}
