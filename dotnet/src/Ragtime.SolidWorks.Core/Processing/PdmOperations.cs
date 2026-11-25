using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// The <c>pdm.*</c> operations (contract section 2, seam-pdm.md section 4). The processor sends every such
    /// operation here before any SolidWorks instance code runs. Order: link, path normalization, plain-ASCII and
    /// approval checks for write targets, and only then one gateway call. Vault work runs inside
    /// <see cref="IPdmGateway.RunAsync{T}"/>.
    /// </summary>
    internal sealed class PdmOperations
    {
        /// <summary>Most files one pdm.file.get_latest may fetch (contract 2.2).</summary>
        private const int MaxFetchedFiles = 500;

        /// <summary>Configuration name of the file-level data card.</summary>
        private const string FileLevelConfiguration = "@";

        private readonly IPdmGateway? _gateway;
        private readonly RequestContext _context;
        private readonly Deadline _deadline;
        private readonly CancellationToken _cancellationToken;
        private readonly IPdmClaimStore _claims;
        private readonly string? _workFolder;
        private readonly Func<DateTime> _utcNow;
        private readonly Action<Exception> _reportUnexpected;

        /// <summary>Creates the operations for one request.</summary>
        public PdmOperations(IPdmGateway? gateway, RequestContext context, Deadline deadline, CancellationToken cancellationToken,
            IPdmClaimStore claims, string? workFolder, Func<DateTime> utcNow, Action<Exception> reportUnexpected)
        {
            _gateway = gateway;
            _context = context;
            _deadline = deadline;
            _cancellationToken = cancellationToken;
            _claims = claims;
            _workFolder = workFolder;
            _utcNow = utcNow;
            _reportUnexpected = reportUnexpected;
        }

        /// <summary>True for every operation this class serves.</summary>
        public static bool IsPdmOperation(string operation) => operation.StartsWith("pdm.", StringComparison.Ordinal);

        /// <summary>The <c>status.pdm</c> block: client facts only, never a login.</summary>
        public static PdmStatusBlock ClientStatusOf(IPdmGateway? gateway)
        {
            var facts = FactsOf(gateway);
            return new PdmStatusBlock
            {
                ClientInstalled = facts.ClientInstalled,
                ClientVersion = facts.ClientVersion,
                Views = facts.Views.Select(ToViewInfo).ToList(),
            };
        }

        /// <summary>Runs one pdm.* operation. Every failure surfaces as a <see cref="HostErrorException"/>.</summary>
        public async Task<JToken> ExecuteAsync(string operation, JObject arguments)
        {
            try
            {
                return await DispatchAsync(operation, arguments).ConfigureAwait(false);
            }
            catch (PdmGatewayException error)
            {
                throw MapGatewayError(error, _context.Pdm?.Vault);
            }
        }

        private async Task<JToken> DispatchAsync(string operation, JObject arguments)
        {
            if (operation == "pdm.status")
            {
                return await StatusAsync().ConfigureAwait(false);
            }

            var link = _context.Pdm ?? throw HostError.Coded(
                ErrorCodes.PdmNotLinked,
                "No PDM tool is linked to this SolidWorks tool.",
                new JObject { ["reason"] = WireJson.Value("no_link") });

            switch (operation)
            {
                case "pdm.folder.list":
                    return await FolderListAsync(link, arguments).ConfigureAwait(false);
                case "pdm.file.info":
                    return await FileInfoAsync(link, arguments).ConfigureAwait(false);
                case "pdm.variables.get":
                    return await VariablesGetAsync(link, arguments).ConfigureAwait(false);
                case "pdm.folder.rights":
                    return await FolderRightsAsync(link, arguments).ConfigureAwait(false);
                case "pdm.folder.children":
                    return await FolderChildrenAsync(link, arguments).ConfigureAwait(false);
                case "pdm.file.get_latest":
                    return await GetLatestAsync(link, arguments).ConfigureAwait(false);
                case "pdm.folder.create":
                    return await FolderCreateAsync(link, arguments).ConfigureAwait(false);
                case "pdm.file.add":
                    return await FileAddAsync(link, arguments).ConfigureAwait(false);
                case "pdm.file.checkout":
                    return await CheckOutAsync(link, arguments).ConfigureAwait(false);
                case "pdm.file.undo_checkout":
                    return await UndoCheckOutAsync(link, arguments).ConfigureAwait(false);
                case "pdm.file.checkin":
                    return await CheckInAsync(link, arguments).ConfigureAwait(false);
                case "pdm.variables.set":
                    return await VariablesSetAsync(link, arguments).ConfigureAwait(false);
                default:
                    throw new InvalidOperationException($"No handler is registered for operation '{operation}'.");
            }
        }

        private async Task<JToken> StatusAsync()
        {
            var client = ClientStatusOf(_gateway);
            var result = new PdmStatusResult
            {
                WorkFolder = _workFolder,
                ClientInstalled = client.ClientInstalled,
                ClientVersion = client.ClientVersion,
                Views = client.Views,
            };

            var link = _context.Pdm;
            if (link != null)
            {
                result.Linked = await LinkedStatusAsync(link, client.Views).ConfigureAwait(false);
            }

            return Fits(result);
        }

        /// <summary>Logs in to the linked vault to report its state. An unavailable vault is reported, not thrown.</summary>
        private async Task<PdmLinkedInfo> LinkedStatusAsync(PdmContext link, IReadOnlyList<PdmViewInfo> views)
        {
            var linked = new PdmLinkedInfo
            {
                Vault = link.Vault,
                ApprovedFolders = link.ApprovedFolders.ToList(),
            };

            try
            {
                var session = await RunVaultAsync(link.Vault, vault => (vault.Root, vault.User)).ConfigureAwait(false);
                linked.Root = session.Root;
                linked.LoggedIn = true;
                linked.User = session.User;
                linked.WritesEnabled = _context.AllowWrite && link.ApprovedFolders.Count > 0;

                var view = views.FirstOrDefault(candidate => SameVault(candidate.Vault, link.Vault));
                if (view != null)
                {
                    view.LoggedIn = true;
                    view.User = session.User;
                }
            }
            catch (PdmGatewayException error) when (UnavailableReason(error.Kind) != null)
            {
                linked.Root = FactsOf(_gateway).Views.FirstOrDefault(view => SameVault(view.Vault, link.Vault))?.Root;
                linked.Problem = UnavailableReason(error.Kind);
            }

            return linked;
        }

        private async Task<JToken> FolderRightsAsync(PdmContext link, JObject arguments)
        {
            var input = (JArray)arguments["paths"]!;
            if (input.Count == 0) throw HostError.InvalidArguments("paths", "At least one path is required.");
            var paths = new List<string>();
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (var i = 0; i < input.Count; i++)
            {
                var path = VaultPaths.Normalize((string?)input[i], ViewRootOf(link), $"paths[{i}]");
                if (seen.Add(path)) paths.Add(path);
            }
            var result = await RunVaultAsync(link.Vault, vault =>
            {
                var output = new PdmFolderRightsResult { User = vault.User };
                foreach (var path in paths)
                {
                    var item = new PdmFolderRightsItem { VaultPath = path };
                    if (!_deadline.Expired)
                    {
                        try
                        {
                            item.Rights = vault.GetFolderRights(path);
                            item.Checked = true;
                            item.Exists = true;
                        }
                        catch (PdmGatewayException ex)
                        {
                            if (ex.Kind == PdmFailureKind.NotFound) { item.Checked = true; item.Exists = false; }
                        }
                    }
                    output.Items.Add(item);
                }
                return output;
            }).ConfigureAwait(false);
            return Fits(result);
        }

        private async Task<JToken> FolderChildrenAsync(PdmContext link, JObject arguments)
        {
            var folder = NormalizeArgument(link, arguments, "path");
            var offset = (int?)arguments["offset"] ?? 0;
            var limit = (int?)arguments["limit"] ?? 50;
            var requestedSnapshot = (string?)arguments["snapshot"];
            var page = await RunVaultAsync(link.Vault, vault =>
            {
                var names = vault.ListSubFolderNames(folder).OrderBy(name => name, StringComparer.OrdinalIgnoreCase)
                    .ThenBy(name => name, StringComparer.Ordinal).ToList();
                var paths = names.Select(name => VaultPaths.Combine(folder, name)).ToList();
                var snapshot = Pager.ComputeSnapshot(paths.Select(path => Newtonsoft.Json.JsonConvert.ToString(path)).ToList());
                if (requestedSnapshot != null && !string.Equals(requestedSnapshot, snapshot, StringComparison.Ordinal))
                    throw HostError.Coded(ErrorCodes.StaleSnapshot,
                        "The list changed since the snapshot was taken. Request the page again without snapshot.",
                        new JObject { ["snapshot"] = WireJson.Value(snapshot) });
                var items = new List<PdmFolderChildItem>();
                for (var i = Math.Min(offset, names.Count); i < names.Count && items.Count < limit; i++)
                {
                    if (_deadline.Expired)
                    {
                        if (items.Count == 0) throw HostError.Coded(ErrorCodes.OperationTimeout, "The request deadline passed before the folder rights could be read.");
                        break;
                    }
                    var item = new PdmFolderChildItem { Name = names[i], VaultPath = paths[i] };
                    try { item.Rights = vault.GetFolderRights(paths[i]); }
                    catch (PdmGatewayException) { }
                    items.Add(item);
                }
                return (items, total: names.Count, snapshot);
            }).ConfigureAwait(false);
            return Pager.FitPage(page.items, offset, page.total, page.snapshot, ProtocolInfo.SiUnits, _context.MaxResultBytes);
        }

        private async Task<JToken> FolderListAsync(PdmContext link, JObject arguments)
        {
            var folder = NormalizeArgument(link, arguments, "path");
            var entries = await RunVaultAsync(link.Vault, vault => vault.ListFolder(folder)).ConfigureAwait(false);

            var items = entries
                .OrderBy(entry => entry.IsFolder ? 0 : 1)
                .ThenBy(entry => entry.Name, StringComparer.OrdinalIgnoreCase)
                .Select(entry => ToFolderItem(folder, entry))
                .ToList();
            return Pager.Page(items, PageRequest.FromArguments(arguments), ProtocolInfo.SiUnits, _context.MaxResultBytes);
        }

        private async Task<JToken> FileInfoAsync(PdmContext link, JObject arguments)
        {
            var path = NormalizeArgument(link, arguments, "path");
            var file = await RunVaultAsync(link.Vault, vault => ReadFile(vault, link, path)).ConfigureAwait(false);
            return Fits(ToFileInfo(file, link));
        }

        private async Task<JToken> VariablesGetAsync(PdmContext link, JObject arguments)
        {
            var path = NormalizeArgument(link, arguments, "path");
            var configuration = ArgumentReader.OptionalString(arguments, "configuration") ?? FileLevelConfiguration;
            var set = await RunVaultAsync(link.Vault, vault => vault.GetVariables(path, configuration)).ConfigureAwait(false);

            return Fits(new PdmVariablesResult
            {
                VaultPath = path,
                Configurations = set.Configurations,
                Configuration = configuration,
                Variables = set.Variables.Select(variable => new PdmVariableItem { Name = variable.Name, Value = variable.Value }).ToList(),
            });
        }

        private async Task<JToken> GetLatestAsync(PdmContext link, JObject arguments)
        {
            var path = NormalizeArgument(link, arguments, "path");
            var includeReferences = ArgumentReader.OptionalBoolean(arguments, "include_references", true);
            var latest = await RunVaultAsync(link.Vault, vault =>
            {
                var fetch = vault.GetLatest(path, includeReferences, MaxFetchedFiles, () => _deadline.Expired);
                return (Fetch: fetch, File: ReadFile(vault, link, path));
            }).ConfigureAwait(false);

            return Fits(new PdmGetLatestResult
            {
                Fetched = latest.Fetch.Fetched,
                Skipped = latest.Fetch.Skipped
                    .Select(skipped => new PdmSkippedItem { VaultPath = skipped.VaultPath, Reason = skipped.Reason })
                    .ToList(),
                File = ToFileInfo(latest.File, link),
            });
        }

        private async Task<JToken> FolderCreateAsync(PdmContext link, JObject arguments)
        {
            var path = NormalizeTarget(link, arguments, "path");
            RequireApproved(link, path);
            var result = await RunVaultAsync(link.Vault, vault => CreateFolders(vault, link, path)).ConfigureAwait(false);
            return Fits(result);
        }

        private async Task<JToken> FileAddAsync(PdmContext link, JObject arguments)
        {
            var source = ArgumentReader.RequiredString(arguments, "source");
            var sourceProblem = DocumentPaths.FindLocalFileProblem(source);
            if (sourceProblem != null)
            {
                throw HostError.InvalidArguments("source", sourceProblem);
            }

            var folder = NormalizeTarget(link, arguments, "folder");
            var name = ArgumentReader.OptionalString(arguments, "name") ?? source.Substring(source.LastIndexOf('\\') + 1);
            RequirePlainAscii(name, "name");
            var nameProblem = VaultPaths.FindNameProblem(name);
            if (nameProblem != null)
            {
                throw HostError.InvalidArguments("name", nameProblem);
            }

            var target = VaultPaths.Normalize(VaultPaths.Combine(folder, name), null, "name");
            RequireSourceLocation(link, source);
            RequireApproved(link, target);
            var file = await RunVaultAsync(link.Vault, vault => AddFile(vault, link, source, folder, name, target)).ConfigureAwait(false);
            return Fits(ToFileInfo(file, link));
        }

        private async Task<JToken> CheckOutAsync(PdmContext link, JObject arguments)
        {
            var path = NormalizeTarget(link, arguments, "path");
            RequireApproved(link, path);
            var file = await RunVaultAsync(link.Vault, vault => CheckOutFile(vault, link, path)).ConfigureAwait(false);
            return Fits(ToFileInfo(file, link));
        }

        private async Task<JToken> UndoCheckOutAsync(PdmContext link, JObject arguments)
        {
            var path = NormalizeTarget(link, arguments, "path");
            RequireApproved(link, path);
            var file = await RunVaultAsync(link.Vault, vault => UndoCheckOutFile(vault, link, path)).ConfigureAwait(false);
            return Fits(ToFileInfo(file, link));
        }

        private async Task<JToken> CheckInAsync(PdmContext link, JObject arguments)
        {
            var comment = CheckInComment(arguments);
            var path = NormalizeTarget(link, arguments, "path");
            RequireApproved(link, path);
            var keepCheckedOut = ArgumentReader.OptionalBoolean(arguments, "keep_checked_out", false);
            var file = await RunVaultAsync(link.Vault, vault => CheckInFile(vault, link, path, link.CommentPrefix + comment, keepCheckedOut)).ConfigureAwait(false);
            return Fits(ToFileInfo(file, link));
        }

        private async Task<JToken> VariablesSetAsync(PdmContext link, JObject arguments)
        {
            var path = NormalizeTarget(link, arguments, "path");
            RequireApproved(link, path);
            var configuration = ArgumentReader.OptionalString(arguments, "configuration") ?? FileLevelConfiguration;
            var values = ToVariableValues(arguments["variables"] as JObject
                ?? throw HostError.InvalidArguments("variables", "'variables' must be an object."));

            var updated = await RunVaultAsync(link.Vault, vault => SetVariablesOf(vault, link, path, configuration, values)).ConfigureAwait(false);
            return Fits(new PdmVariablesSetResult
            {
                VaultPath = path,
                Configuration = configuration,
                Updated = updated,
            });
        }

        /// <summary>
        /// The check-in comment as stored: no control characters, surrounding whitespace removed, never empty.
        /// Checked before any vault call.
        /// </summary>
        private static string CheckInComment(JObject arguments)
        {
            var raw = ArgumentReader.RequiredString(arguments, "comment");
            if (raw.Any(ch => ch < 0x20 || ch == 0x7F))
            {
                throw HostError.InvalidArguments("comment", "'comment' must not contain control characters.");
            }

            var text = raw.Trim();
            if (text.Length == 0)
            {
                throw HostError.InvalidArguments("comment", "'comment' must not be empty.");
            }

            return text;
        }

        private PdmFolderCreateResult CreateFolders(IPdmVault vault, PdmContext link, string path)
        {
            vault.RequireNoLinks(path);
            if (vault.FolderExists(path))
            {
                return FolderResult(vault, path, created: false);
            }

            if (vault.FileExists(path))
            {
                throw Conflict("already_exists", "A file already has that name.");
            }

            // Walk up to the first existing folder; every level that must be created has to be approved first.
            var missing = new List<string>();
            for (var level = path; !vault.FolderExists(level); level = VaultPaths.Parent(level))
            {
                RequireApproved(link, level);
                missing.Add(level);
            }

            missing.Reverse();
            foreach (var level in missing)
            {
                RequireMutationDeadline();
                vault.CreateFolder(level);
            }

            return FolderResult(vault, path, created: true);
        }

        private static PdmFolderCreateResult FolderResult(IPdmVault vault, string path, bool created) =>
            new PdmFolderCreateResult { VaultPath = path, LocalPath = vault.Root + path, Created = created };

        private PdmFileState AddFile(IPdmVault vault, PdmContext link, string source, string folder, string name, string target)
        {
            vault.RequireNoLinks(target);
            if (!vault.FolderExists(folder))
            {
                throw NotFound(folder);
            }

            if (vault.FileExists(target))
            {
                throw Conflict("already_exists", "A file with that name already exists in the folder.");
            }

            RequireMutationDeadline();
            var fileId = vault.AddFile(source, folder, name);
            return CompleteClaim(vault, link, fileId, target, addedByTool: true);
        }

        private PdmFileState CheckOutFile(IPdmVault vault, PdmContext link, string path)
        {
            vault.RequireNoLinks(path);
            var state = ReadFile(vault, link, path);
            if (state.CheckedOutHere)
            {
                RequireClaimed(link, state);
                return state;
            }

            if (state.CheckedOut)
            {
                throw Conflict("checked_out_by_other", "The file is checked out by another user or on another computer.", state);
            }

            RequireMutationDeadline();
            vault.CheckOut(path);
            return CompleteClaim(vault, link, state.FileId, path, addedByTool: false);
        }

        private PdmFileState UndoCheckOutFile(IPdmVault vault, PdmContext link, string path)
        {
            vault.RequireNoLinks(path);
            var state = ReadFile(vault, link, path);
            RequireCheckedOutHere(state);
            RequireClaimed(link, state);
            if (_claims.Find(link.Vault, state.FileId)!.AddedByTool)
            {
                throw Conflict("never_checked_in", "A file that was added and never checked in cannot be undone. Check it in instead.");
            }

            RequireMutationDeadline();
            RevokeThenMutate(link, state.FileId, () => vault.UndoCheckOut(path));
            return ReadFile(vault, link, path);
        }

        private PdmFileState CheckInFile(IPdmVault vault, PdmContext link, string path, string storedComment, bool keepCheckedOut)
        {
            vault.RequireNoLinks(path);
            var state = ReadFile(vault, link, path);
            RequireCheckedOutHere(state);
            RequireClaimed(link, state);

            RequireMutationDeadline();
            if (keepCheckedOut)
            {
                vault.CheckIn(path, storedComment, true);
            }
            else
            {
                RevokeThenMutate(link, state.FileId, () => vault.CheckIn(path, storedComment, false));
            }

            return keepCheckedOut
                ? CompleteClaim(vault, link, state.FileId, path, addedByTool: false)
                : ReadFile(vault, link, path);
        }

        private void RevokeThenMutate(PdmContext link, int fileId, Action mutate)
        {
            var claim = _claims.Find(link.Vault, fileId);
            try
            {
                _claims.Remove(link.Vault, fileId);
            }
            catch (Exception error)
            {
                _reportUnexpected(error);
                throw HostError.Coded(ErrorCodes.InternalError,
                    "The tool could not record the change safely; nothing was changed.");
            }

            try
            {
                RequireMutationDeadline();
                mutate();
            }
            catch (Exception)
            {
                if (claim != null)
                {
                    try
                    {
                        _claims.Put(claim);
                    }
                    catch (Exception error)
                    {
                        _reportUnexpected(error);
                    }
                }
                throw;
            }
        }

        private List<string> SetVariablesOf(IPdmVault vault, PdmContext link, string path, string configuration, IReadOnlyDictionary<string, object> values)
        {
            vault.RequireNoLinks(path);
            var state = ReadFile(vault, link, path);
            RequireCheckedOutHere(state);
            RequireClaimed(link, state);
            RequireMutationDeadline();
            return vault.SetVariables(path, configuration, values).ToList();
        }

        private void RequireMutationDeadline()
        {
            if (_deadline.Expired)
            {
                throw new PdmGatewayException(PdmFailureKind.Timeout, "The request deadline passed before the PDM write started.");
            }
        }

        /// <summary>Refuses a file that is not checked out by this user on this computer.</summary>
        private static void RequireCheckedOutHere(PdmFileState state)
        {
            if (!state.CheckedOutHere)
            {
                throw Conflict("not_checked_out_here", "Only a file checked out by you on this computer can be changed.");
            }
        }

        /// <summary>Refuses a file checked out here that the tool did not check out (or add) itself.</summary>
        private void RequireClaimed(PdmContext link, PdmFileState state)
        {
            if (!HasValidClaim(link, state))
            {
                var claim = _claims.Find(link.Vault, state.FileId);
                if (claim != null && string.Equals(claim.ToolId, link.ToolId, StringComparison.Ordinal))
                {
                    _claims.Remove(link.Vault, state.FileId);
                }
                throw Conflict(
                    "not_checked_out_by_tool",
                    "This file was checked out on this machine outside the tool; leave it to the person who did.");
            }
        }

        private PdmFileState ReadFile(IPdmVault vault, PdmContext link, string path)
        {
            var state = vault.GetFile(path);
            if (!state.CheckedOutHere && _claims.Find(link.Vault, state.FileId) != null)
            {
                _claims.Remove(link.Vault, state.FileId);
            }
            return state;
        }

        private PdmFileState CompleteClaim(IPdmVault vault, PdmContext link, int fileId, string path, bool addedByTool)
        {
            // Record success before the follow-up read; incomplete evidence must never authorize another write.
            _claims.Put(new PdmClaim
            {
                Vault = link.Vault, FileId = fileId, ToolId = link.ToolId,
                AddedByTool = addedByTool, ClaimedUtc = _utcNow(),
            });
            var state = ReadFile(vault, link, path);
            if (state.CheckedOutHere)
            {
                _claims.Put(new PdmClaim
                {
                    Vault = link.Vault,
                    FileId = fileId,
                    ToolId = link.ToolId,
                    AddedByTool = addedByTool,
                    LockedByUserId = state.LockedByUserId ?? 0,
                    LockedOnComputer = state.CheckedOutOn ?? string.Empty,
                    LockPath = state.LockPath ?? string.Empty,
                    ClaimedUtc = _utcNow(),
                });
            }
            return state;
        }

        private bool HasValidClaim(PdmContext link, PdmFileState state)
        {
            var claim = _claims.Find(link.Vault, state.FileId);
            if (claim == null || !state.CheckedOutHere)
            {
                return false;
            }
            var age = _utcNow() - claim.ClaimedUtc;
            return string.Equals(claim.ToolId, link.ToolId, StringComparison.Ordinal)
                && state.LockedByUserId.HasValue && claim.LockedByUserId == state.LockedByUserId.Value
                && claim.LockedByUserId > 0
                && !string.IsNullOrEmpty(claim.LockedOnComputer) && !string.IsNullOrEmpty(claim.LockPath)
                && string.Equals(claim.LockedOnComputer, state.CheckedOutOn, StringComparison.OrdinalIgnoreCase)
                && string.Equals(claim.LockPath, state.LockPath, StringComparison.OrdinalIgnoreCase)
                && age >= TimeSpan.Zero && age < TimeSpan.FromHours(24);
        }

        private void RequireSourceLocation(PdmContext link, string source)
        {
            var normalized = NormalizeLocalPath(source);
            var root = ViewRootOf(link);
            if ((root != null && link.ApprovedFolders.Any(folder => IsInside(normalized, root + folder)))
                || (_workFolder != null && IsInside(normalized, _workFolder)))
            {
                return;
            }
            throw HostError.InvalidArguments("source",
                $"Add files from an approved vault folder or from the tool's work folder {_workFolder}.");
        }

        private static bool IsInside(string path, string folder) =>
            path.StartsWith(NormalizeLocalPath(folder).TrimEnd('\\') + "\\", StringComparison.OrdinalIgnoreCase);

        private static string NormalizeLocalPath(string path)
        {
            var segments = path.TrimEnd('\\').Split('\\').Where(segment => segment != ".").ToArray();
            if (segments.Skip(1).Any(segment => segment.Length == 0 || segment == ".."
                || segment.EndsWith(".", StringComparison.Ordinal) || segment.EndsWith(" ", StringComparison.Ordinal)))
            {
                throw HostError.InvalidArguments("source", "The source path contains an invalid segment.");
            }
            return string.Join("\\", segments);
        }

        private static Dictionary<string, object> ToVariableValues(JObject variables)
        {
            var values = new Dictionary<string, object>();
            foreach (var property in variables.Properties())
            {
                values[property.Name] = ToVariableValue(property.Name, property.Value);
            }

            return values;
        }

        private static object ToVariableValue(string name, JToken value)
        {
            switch (value.Type)
            {
                case JTokenType.String:
                    return (string?)value ?? string.Empty;
                case JTokenType.Boolean:
                    return (bool)value;
                case JTokenType.Float:
                    return (double)value;
                case JTokenType.Integer:
                    if (value is JValue constant && constant.Value is long integer)
                    {
                        return integer;
                    }

                    break;
            }

            throw HostError.InvalidArguments("variables", $"The value of '{name}' must be a string, a number or a boolean.");
        }

        /// <summary>Refuses a vault path outside the approved folders, before any vault call.</summary>
        private static void RequireApproved(PdmContext link, string vaultPath)
        {
            if (!VaultPaths.IsApproved(vaultPath, link.ApprovedFolders))
            {
                throw NotApproved(link, vaultPath);
            }
        }

        private static HostErrorException NotApproved(PdmContext link, string vaultPath)
        {
            var message = link.ApprovedFolders.Count == 0
                ? "No vault folders are approved for changes. Ask an administrator to approve a folder."
                : $"'{vaultPath}' is outside the vault folders approved for changes.";
            return HostError.Coded(
                ErrorCodes.PdmFolderNotApproved,
                message,
                new JObject
                {
                    ["vault_path"] = WireJson.Value(vaultPath),
                    ["approved_folders"] = JArray.FromObject(link.ApprovedFolders),
                });
        }

        private static HostErrorException NotFound(string? vaultPath) =>
            HostError.Coded(
                ErrorCodes.PdmNotFound,
                "No file or folder exists at that vault path.",
                new JObject { ["vault_path"] = WireJson.Value(vaultPath) });

        private static HostErrorException Conflict(string reason, string message, PdmFileState? holder = null)
        {
            var details = new JObject { ["reason"] = WireJson.Value(reason) };
            if (holder?.CheckedOutBy != null)
            {
                details["checked_out_by"] = WireJson.Value(holder.CheckedOutBy);
            }

            if (holder?.CheckedOutOn != null)
            {
                details["checked_out_on"] = WireJson.Value(holder.CheckedOutOn);
            }

            return HostError.Coded(ErrorCodes.PdmConflict, message, details);
        }

        /// <summary>Plain ASCII (0x20-0x7E) only: the rule for anything that decides a vault write.</summary>
        private static void RequirePlainAscii(string value, string field)
        {
            if (value.Any(ch => ch < 0x20 || ch > 0x7E))
            {
                throw HostError.InvalidArguments(field, "Vault changes are limited to paths with plain ASCII characters.");
            }
        }

        private string NormalizeArgument(PdmContext link, JObject arguments, string field) =>
            VaultPaths.Normalize(ArgumentReader.OptionalString(arguments, field), ViewRootOf(link), field);

        /// <summary>A path a write may change: normalized, then limited to plain ASCII, before the approval check.</summary>
        private string NormalizeTarget(PdmContext link, JObject arguments, string field)
        {
            var path = NormalizeArgument(link, arguments, field);
            RequirePlainAscii(path, field);
            return path;
        }

        /// <summary>The local view root of the linked vault, or null when the client does not report that vault.</summary>
        private string? ViewRootOf(PdmContext link) =>
            FactsOf(_gateway).Views.FirstOrDefault(view => SameVault(view.Vault, link.Vault))?.Root;

        /// <summary>Runs work on the gateway against one vault within the request deadline.</summary>
        private async Task<T> RunVaultAsync<T>(string vault, Func<IPdmVault, T> work)
        {
            if (_deadline.Expired)
            {
                throw HostError.Coded(
                    ErrorCodes.OperationTimeout,
                    $"The request passed its {(int)_deadline.Total.TotalSeconds}-second deadline before the next PDM step started. Nothing more was run; try again.");
            }

            if (_gateway == null)
            {
                throw new PdmGatewayException(PdmFailureKind.ClientMissing, "The SolidWorks PDM client is not installed on this computer.");
            }

            return await _gateway.RunAsync(vault, work, _deadline, _cancellationToken).ConfigureAwait(false);
        }

        private JToken Fits(object dto) => Pager.RequireFits(WireJson.ToToken(dto), _context.MaxResultBytes);

        private static PdmClientFacts FactsOf(IPdmGateway? gateway) => gateway?.GetClientFacts() ?? PdmClientFacts.NotInstalled;

        private static bool SameVault(string? left, string? right) =>
            string.Equals(left, right, StringComparison.OrdinalIgnoreCase);

        private static PdmViewInfo ToViewInfo(PdmViewFacts view) =>
            new PdmViewInfo { Vault = view.Vault, Root = view.Root, LoggedIn = view.LoggedIn, User = view.User };

        private static PdmFolderItem ToFolderItem(string folder, PdmFolderEntry entry)
        {
            var item = new PdmFolderItem
            {
                Name = entry.Name,
                Kind = entry.IsFolder ? "folder" : "file",
                VaultPath = VaultPaths.Combine(folder, entry.Name),
            };
            if (!entry.IsFolder)
            {
                item.LatestVersion = entry.LatestVersion;
                item.LocalVersion = entry.LocalVersion;
                item.CheckedOut = entry.CheckedOut;
            }

            return item;
        }

        private PdmFileInfo ToFileInfo(PdmFileState file, PdmContext link) =>
            new PdmFileInfo
            {
                VaultPath = file.VaultPath,
                LocalPath = file.LocalPath,
                Name = file.Name,
                FileId = file.FileId,
                ExistsLocally = file.ExistsLocally,
                LocalVersion = file.LocalVersion,
                LatestVersion = file.LatestVersion,
                CheckedOut = file.CheckedOut,
                CheckedOutBy = file.CheckedOutBy,
                CheckedOutOn = file.CheckedOutOn,
                CheckedOutHere = file.CheckedOutHere,
                CheckedOutByTool = HasValidClaim(link, file),
                WorkflowState = file.WorkflowState,
                InApprovedFolder = VaultPaths.IsApproved(file.VaultPath, link.ApprovedFolders),
            };

        /// <summary>The reason text of pdm_unavailable for a kind, or null when the kind is not an unavailable one.</summary>
        private static string? UnavailableReason(PdmFailureKind kind)
        {
            switch (kind)
            {
                case PdmFailureKind.ClientMissing:
                    return "client_missing";
                case PdmFailureKind.ViewMissing:
                    return "view_missing";
                case PdmFailureKind.LoginFailed:
                    return "login_failed";
                case PdmFailureKind.Busy:
                    return "busy";
                default:
                    return null;
            }
        }

        private static HostErrorException MapGatewayError(PdmGatewayException error, string? vault)
        {
            var reason = UnavailableReason(error.Kind);
            if (reason != null)
            {
                return HostError.Coded(
                    ErrorCodes.PdmUnavailable,
                    UnavailableMessage(reason),
                    new JObject
                    {
                        ["reason"] = WireJson.Value(reason),
                        ["vault"] = WireJson.Value(vault),
                    });
            }

            switch (error.Kind)
            {
                case PdmFailureKind.Timeout:
                    return HostError.Coded(
                        ErrorCodes.OperationTimeout,
                        "The PDM call did not finish before the request deadline. The PDM worker stays busy until it returns.");
                case PdmFailureKind.NotFound:
                    return NotFound(error.Subject);
                case PdmFailureKind.LinkInPath:
                    return HostError.InvalidArguments("path", "The path goes through a link (symbolic link or junction).");
                case PdmFailureKind.InvalidSource:
                    return HostError.InvalidArguments("source", error.Message);
                case PdmFailureKind.UnknownVariable:
                    return HostError.InvalidArguments("variables", $"There is no variable named '{error.Subject}' for this file.");
                case PdmFailureKind.TooManyFiles:
                    return HostError.Coded(
                        ErrorCodes.ResultTooLarge,
                        $"The request would fetch more than {MaxFetchedFiles} files. Fetch one file without its references.");
                case PdmFailureKind.LocalFileExists:
                    return Conflict(
                        "local_file_exists",
                        "A file already exists on this computer where the new vault file would go. Nothing was changed.");
                default:
                    return HostError.Coded(ErrorCodes.PdmError, error.Message, PdmErrorDetails(error));
            }
        }

        private static string UnavailableMessage(string reason)
        {
            switch (reason)
            {
                case "client_missing":
                    return "The SolidWorks PDM client is not installed on this computer.";
                case "view_missing":
                    return "The linked vault has no local view on this computer.";
                case "login_failed":
                    return "Logging in to the linked PDM vault failed.";
                default:
                    return "The PDM worker is still busy with an earlier call. Try again in a moment.";
            }
        }

        private static JObject PdmErrorDetails(PdmGatewayException error) =>
            new JObject
            {
                ["hresult"] = WireJson.Value(error.HResultCode.HasValue ? "0x" + error.HResultCode.Value.ToString("X8", CultureInfo.InvariantCulture) : null),
                ["pdm_error"] = WireJson.Value(error.PdmError),
            };
    }
}
