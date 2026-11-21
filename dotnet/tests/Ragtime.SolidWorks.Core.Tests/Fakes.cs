using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Core.Tests
{
    /// <summary>In-memory session. Lists are small fixed sets; <see cref="FeatureCount"/> sizes the feature list for paging tests.</summary>
    internal sealed class FakeSession : ISolidWorksSession
    {
        public int ProcessId { get; set; } = 4242;
        public DateTime ProcessStartUtc { get; set; } = new DateTime(2026, 11, 20, 0, 0, 0, DateTimeKind.Utc);
        public bool IsAttached { get; set; }
        public ScriptSessionState ScriptState { get; set; } = new ScriptSessionState();
        public ScriptSessionState? RestoredScriptState { get; private set; }
        public bool FailScriptStateCapture { get; set; }
        public bool FailScriptStateRestore { get; set; }

        public ScriptSessionState CaptureScriptState()
        {
            Calls.Add(nameof(CaptureScriptState));
            if (FailScriptStateCapture) throw new InvalidOperationException("Capture failed.");
            return ScriptState;
        }

        public void RestoreScriptState(ScriptSessionState state)
        {
            Calls.Add(nameof(RestoreScriptState));
            if (FailScriptStateRestore) throw HostError.Coded(ErrorCodes.InstanceTimeout, "Restore timed out.");
            RestoredScriptState = state;
        }

        public void ReapplyWindowHiding()
        {
            if (!IsAttached) Calls.Add(nameof(ReapplyWindowHiding));
        }

        public List<SessionDocumentInfo> Open { get; } = new List<SessionDocumentInfo>();

        public List<string> Closed { get; } = new List<string>();

        public List<string> Calls { get; } = new List<string>();

        /// <summary>References the session received through <see cref="GetModelSummary"/>, in call order.</summary>
        public List<DocumentRef> Refs { get; } = new List<DocumentRef>();

        public int FeatureCount { get; set; } = 3;

        public int SegmentCount { get; set; } = 3;

        public bool ThrowUnexpected { get; set; }

        public string Revision { get; set; } = "32.5.0";

        public InstanceFacts GetInstanceFacts()
        {
            Calls.Add(nameof(GetInstanceFacts));
            return new InstanceFacts(Revision, 4242, false);
        }

        public IReadOnlyList<SessionDocumentInfo> ListOpenDocuments()
        {
            Calls.Add(nameof(ListOpenDocuments));
            return Open.ToList();
        }

        public SessionDocumentInfo OpenReadOnlySilent(string path, string? configuration)
        {
            Calls.Add(nameof(OpenReadOnlySilent));
            if (path.Contains("missing"))
            {
                throw HostError.Coded(ErrorCodes.DocumentNotFound, "The file does not exist.");
            }

            var existing = Open.FirstOrDefault(d => DocumentPaths.SamePath(d.Path, path));
            if (existing != null)
            {
                return new SessionDocumentInfo(existing.Title, existing.Path, existing.Type, existing.ActiveConfiguration, true, false, wasAlreadyOpen: true);
            }

            var info = new SessionDocumentInfo(TitleOf(path), path, TypeOf(path), "Default", true, false);
            Open.Add(info);
            return info;
        }

        public void CloseDocument(DocumentRef document)
        {
            Calls.Add(nameof(CloseDocument));
            Closed.Add(document.Path ?? string.Empty);
            Open.RemoveAll(d => DocumentPaths.SamePath(d.Path, document.Path));
        }

        public ModelSummary GetModelSummary(DocumentRef document)
        {
            Calls.Add(nameof(GetModelSummary));
            Refs.Add(document);
            return new ModelSummary
            {
                Type = TypeOf(document.Path ?? document.Title ?? string.Empty),
                Title = document.Title ?? TitleOf(document.Path ?? string.Empty),
                Path = document.Path ?? string.Empty,
                ConfigurationCount = 1,
                FeatureCount = FeatureCount,
                Units = ProtocolInfo.SiUnits,
                SolidBodyCount = 1,
                ActiveConfiguration = "Default",
            };
        }

        public IReadOnlyList<ConfigurationItem> ListConfigurations(DocumentRef document)
        {
            Calls.Add(nameof(ListConfigurations));
            return new List<ConfigurationItem>
            {
                new ConfigurationItem { Name = "Default", IsActive = true },
                new ConfigurationItem { Name = "Small", Parent = "Default" },
            };
        }

        public IReadOnlyList<FeatureItem> ListFeatures(DocumentRef document, bool includeSystem, IReadOnlyList<string>? types)
        {
            Calls.Add(nameof(ListFeatures));
            if (ThrowUnexpected)
            {
                throw new InvalidOperationException("secret detail C:\\private\\path");
            }

            return Enumerable.Range(0, FeatureCount)
                .Select(i => new FeatureItem { Name = i == 0 ? "Boss-Extrude1" : "Feature" + i, Type = "Extrusion", Id = i.ToString() })
                .ToList();
        }

        public FeatureDetail GetFeature(DocumentRef document, string feature)
        {
            Calls.Add(nameof(GetFeature));
            if (feature != "Boss-Extrude1")
            {
                throw HostError.Coded(ErrorCodes.EntityNotFound, $"No feature named {feature}.");
            }

            return new FeatureDetail { Name = feature, Type = "Extrusion", Id = "1", Sketch = "Sketch1" };
        }

        public SketchSummary GetSketchSummary(DocumentRef document, string sketch)
        {
            Calls.Add(nameof(GetSketchSummary));
            RequireSketch(sketch);
            return new SketchSummary { Name = sketch, Units = ProtocolInfo.SiUnits, SegmentCount = SegmentCount };
        }

        public IReadOnlyList<SegmentItem> ListSketchSegments(DocumentRef document, string sketch)
        {
            Calls.Add(nameof(ListSketchSegments));
            RequireSketch(sketch);
            return Enumerable.Range(0, SegmentCount)
                .Select(i => new SegmentItem { Id = i.ToString(), Type = "line", Length = 0.05, Start = new double[] { 0, 0, 0 }, End = new double[] { 0.05, 0, 0 } })
                .ToList();
        }

        public IReadOnlyList<PointItem> ListSketchPoints(DocumentRef document, string sketch)
        {
            Calls.Add(nameof(ListSketchPoints));
            RequireSketch(sketch);
            return new List<PointItem> { new PointItem { Id = "p1" }, new PointItem { Id = "p2", Construction = true } };
        }

        public IReadOnlyList<RelationItem> ListSketchRelations(DocumentRef document, string sketch)
        {
            Calls.Add(nameof(ListSketchRelations));
            RequireSketch(sketch);
            return new List<RelationItem> { new RelationItem { Type = "Horizontal" } };
        }

        public IReadOnlyList<DimensionItem> ListSketchDimensions(DocumentRef document, string sketch)
        {
            Calls.Add(nameof(ListSketchDimensions));
            RequireSketch(sketch);
            return new List<DimensionItem> { new DimensionItem { Name = "D1", Units = ProtocolInfo.SiUnits } };
        }

        public IReadOnlyList<DimensionItem> ListDimensions(DocumentRef document, string? feature)
        {
            Calls.Add(nameof(ListDimensions));
            return new List<DimensionItem>
            {
                new DimensionItem { Name = "D1", Feature = feature ?? "Boss-Extrude1", Value = 0.02, Type = "linear" },
                new DimensionItem { Name = "D2", Value = 0.5, Type = "angular" },
            };
        }

        public IReadOnlyList<MateItem> ListMates(DocumentRef document)
        {
            Calls.Add(nameof(ListMates));
            return new List<MateItem> { new MateItem { Name = "Coincident1", Type = "Coincident", Alignment = "Aligned" } };
        }

        public IReadOnlyList<ComponentItem> ListComponents(DocumentRef document, bool topLevelOnly)
        {
            Calls.Add(nameof(ListComponents) + ":" + topLevelOnly);
            return new List<ComponentItem> { new ComponentItem { Name = "Part1-1", Transform = new double[16] } };
        }

        public PropertiesResult GetProperties(DocumentRef document, string? configuration)
        {
            Calls.Add(nameof(GetProperties));
            return new PropertiesResult { Scope = configuration == null ? "file" : "configuration:" + configuration };
        }

        public MassPropertiesResult GetMassProperties(DocumentRef document)
        {
            Calls.Add(nameof(GetMassProperties));
            return new MassPropertiesResult { Mass = 1.5, Units = ProtocolInfo.SiUnits, CenterOfMass = new double[3], PrincipalMoments = new double[3], MomentsOfInertia = new double[9] };
        }

        public IReadOnlyList<EquationItem> ListEquations(DocumentRef document)
        {
            Calls.Add(nameof(ListEquations));
            return new List<EquationItem> { new EquationItem { Index = 0, Equation = "\"D1\"=0.02" } };
        }

        private static void RequireSketch(string sketch)
        {
            if (sketch != "Sketch1")
            {
                throw HostError.Coded(ErrorCodes.EntityNotFound, $"No sketch named {sketch}.");
            }
        }

        public static string TitleOf(string path) => path.Substring(path.LastIndexOf('\\') + 1);

        public static string TypeOf(string path)
        {
            if (path.EndsWith(".sldasm", StringComparison.OrdinalIgnoreCase))
            {
                return "assembly";
            }

            return path.EndsWith(".slddrw", StringComparison.OrdinalIgnoreCase) ? "drawing" : "part";
        }
    }

    /// <summary>In-memory instance manager. Creates sessions for new instances and reports them as snapshots.</summary>
    internal sealed class FakeInstanceManager : ISolidWorksInstanceManager
    {
        public List<string> RecoveredInstances { get; } = new List<string>();
        public string Recovery { get; set; } = ScriptRecovery.ProbeOk;

        public Task<string> RecoverAfterScriptAsync(string instanceId, CancellationToken cancellationToken)
        {
            RecoveredInstances.Add(instanceId);
            return Task.FromResult(Recovery);
        }

        private int _counter;

        public List<int> Installed { get; } = new List<int> { 2024 };

        public List<InstanceSnapshot> Instances { get; } = new List<InstanceSnapshot>();

        public Dictionary<string, FakeSession> Sessions { get; } = new Dictionary<string, FakeSession>();

        public List<string> Targets { get; } = new List<string>();
        public bool RunningWork { get; private set; }

        /// <summary>The deadline each RunAsync call received, in call order.</summary>
        public List<Deadline> Deadlines { get; } = new List<Deadline>();
        public List<CancellationToken> RunTokens { get; } = new List<CancellationToken>();

        /// <summary>When set together with <see cref="CostMs"/>, every call advances this clock: it simulates time spent in SolidWorks.</summary>
        public FakeClock? Clock { get; set; }

        /// <summary>Milliseconds each RunAsync call costs on <see cref="Clock"/>.</summary>
        public long CostMs { get; set; }

        public bool ShutdownCalled { get; private set; }

        public event Action<string>? InstanceStopped;

        public FakeSession SessionFor(string instanceId) => Sessions[instanceId];

        public InstanceSnapshot AddBackground(string id, int documents = 0, bool busy = false)
        {
            var snapshot = new InstanceSnapshot(id, ProtocolInfo.InstanceModes.Background, true, 2024, "32.5.0", 100, false, busy, documents);
            Instances.Add(snapshot);
            Sessions[id] = new FakeSession();
            _counter = Math.Max(_counter, int.Parse(id.Substring(1)));
            return snapshot;
        }

        public InstanceSnapshot AddAttached(int pid, int year = 2024)
        {
            var id = "a" + pid.ToString();
            var snapshot = new InstanceSnapshot(id, ProtocolInfo.InstanceModes.Attach, false, year, year == 2024 ? "32.5.0" : "33.0.0", pid, true, false, 0);
            Instances.Add(snapshot);
            Sessions[id] = new FakeSession { IsAttached = true };
            return snapshot;
        }

        private (InstanceSnapshot Snapshot, FakeSession Session)? _pendingAttach;

        /// <summary>
        /// A SolidWorks that is not registered yet: it appears (with the given open documents) when the first attach
        /// request runs, as on a fresh host. Used to check that a request sees the instance its own refresh created.
        /// </summary>
        public void AddPendingAttached(int pid, params SessionDocumentInfo[] openDocuments)
        {
            var session = new FakeSession { IsAttached = true };
            session.Open.AddRange(openDocuments);
            var snapshot = new InstanceSnapshot("a" + pid.ToString(), ProtocolInfo.InstanceModes.Attach, false, 2024, "32.5.0", pid, true, false, 0);
            _pendingAttach = (snapshot, session);
        }

        /// <summary>Instance ids closed to make room for a launch (F6), in order.</summary>
        public List<string> Retired { get; } = new List<string>();

        public void Stop(string instanceId)
        {
            Instances.RemoveAll(i => i.InstanceId == instanceId);
            InstanceStopped?.Invoke(instanceId);
        }

        public IReadOnlyList<int> GetInstalledVersions() => Installed.ToList();

        public IReadOnlyList<InstanceSnapshot> GetInstances() => Instances.ToList();

        public Task<InstanceRun<T>> RunAsync<T>(
            InstanceTarget target,
            RequestContext context,
            Deadline deadline,
            Func<ISolidWorksSession, T> work,
            CancellationToken cancellationToken)
        {
            Targets.Add(target.Kind + (target.InstanceId == null ? string.Empty : ":" + target.InstanceId));
            Deadlines.Add(deadline);
            RunTokens.Add(cancellationToken);
            if (Clock != null)
            {
                Clock.Now += CostMs;
            }

            InstanceSnapshot snapshot;
            switch (target.Kind)
            {
                case InstanceTargetKind.Specific:
                    snapshot = Instances.FirstOrDefault(i => i.InstanceId == target.InstanceId)
                        ?? throw HostError.Coded(ErrorCodes.SolidworksNotRunning, "That instance is not running.");
                    break;
                case InstanceTargetKind.StartNew:
                    if (target.RetireInstanceId != null)
                    {
                        Instances.RemoveAll(i => i.InstanceId == target.RetireInstanceId);
                        Retired.Add(target.RetireInstanceId);
                        InstanceStopped?.Invoke(target.RetireInstanceId);
                    }

                    _counter++;
                    snapshot = AddBackground("i" + _counter.ToString());
                    break;
                default:
                    if (_pendingAttach is { } pending && Instances.All(i => i.Mode != ProtocolInfo.InstanceModes.Attach))
                    {
                        Instances.Add(pending.Snapshot);
                        Sessions[pending.Snapshot.InstanceId] = pending.Session;
                        _pendingAttach = null;
                    }

                    snapshot = Instances.FirstOrDefault(i => i.Mode == ProtocolInfo.InstanceModes.Attach)
                        ?? throw HostError.Coded(ErrorCodes.SolidworksNotRunning, "No SolidWorks instance is running in this session.");
                    break;
            }

            RunningWork = true;
            try
            {
                var value = work(Sessions[snapshot.InstanceId]);
                return Task.FromResult(new InstanceRun<T>(snapshot, value));
            }
            finally
            {
                RunningWork = false;
            }
        }

        public Task ShutdownAsync(CancellationToken cancellationToken)
        {
            ShutdownCalled = true;
            return Task.CompletedTask;
        }
    }

    internal sealed class FakeHostFacts : IHostFactsProvider
    {
        public HostFacts GetHostFacts() => new HostFacts("DESK\\elfy", TestData.Sid, false, 1, true, 42.5);
    }

    /// <summary>
    /// In-memory PDM vault. Folders, files and data-card values persist across gateway runs, as a logged-in session
    /// would. Every call is recorded in <see cref="Calls"/> as "Method" or "Method:argument". A method scripted in
    /// <see cref="Failures"/> throws its exception instead of acting.
    /// </summary>
    internal sealed class FakePdmVault : IPdmVault
    {
        public const string VaultName = "PDM";
        public const string RootPath = @"C:\PDM";
        public const string UserName = "mvisnovsky";
        public const string CheckedOutOn = "2026-11-20 09:15";

        private readonly Dictionary<string, PdmFileState> _files = new Dictionary<string, PdmFileState>(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> _folders = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { @"\", @"\13_Ragtime", @"\13_Ragtime2", @"\Other" };
        private int _nextFileId = 100;

        public string Vault => VaultName;

        public string Root => RootPath;

        public string User => UserName;

        public List<string> Calls { get; } = new List<string>();

        /// <summary>Exceptions thrown instead of the named method's work (keyed by method name).</summary>
        public Dictionary<string, Exception> Failures { get; } = new Dictionary<string, Exception>();

        public Action<string, int>? OnCall { get; set; }

        private readonly Dictionary<string, int> _callCounts = new Dictionary<string, int>();

        public Dictionary<string, Dictionary<string, string>> ConfigurationVariables { get; } =
            new Dictionary<string, Dictionary<string, string>>();

        /// <summary>Vault paths that are links; a path at or under one fails RequireNoLinks with LinkInPath.</summary>
        public HashSet<string> LinkPaths { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        /// <summary>Referenced files that GetLatest fetches when asked to include references.</summary>
        public List<string> References { get; } = new List<string>();

        /// <summary>Data-card variables of the vault; the names are the known variables.</summary>
        public Dictionary<string, string> Variables { get; } = new Dictionary<string, string>
        {
            ["Description"] = "Bracket",
            ["Revision"] = "A",
            ["Count"] = "0",
            ["Mass"] = "0",
            ["Flag"] = "False",
        };

        /// <summary>Values written by SetVariables, one dictionary per call.</summary>
        public List<Dictionary<string, object>> VariableWrites { get; } = new List<Dictionary<string, object>>();

        /// <summary>Check-ins in order: vault path, stored comment, keep_checked_out.</summary>
        public List<(string Path, string Comment, bool KeepCheckedOut)> CheckIns { get; } = new List<(string Path, string Comment, bool KeepCheckedOut)>();

        /// <summary>Files added by AddFile: source, folder, name.</summary>
        public List<(string Source, string Folder, string Name)> Added { get; } = new List<(string Source, string Folder, string Name)>();

        /// <summary>Creates a file with the given state in the vault (a checked-in file unless adjusted).</summary>
        public PdmFileState SeedFile(string vaultPath, Action<PdmFileState>? adjust = null)
        {
            var file = new PdmFileState
            {
                VaultPath = vaultPath,
                LocalPath = RootPath + vaultPath,
                Name = LastSegment(vaultPath),
                FileId = _nextFileId++,
                ExistsLocally = true,
                LocalVersion = 3,
                LatestVersion = 3,
                EverCheckedIn = true,
            };
            adjust?.Invoke(file);
            _files[vaultPath] = file;
            return file;
        }

        public void SeedFolder(string vaultPath) => _folders.Add(vaultPath);

        public bool FolderExists(string vaultPath)
        {
            Enter(nameof(FolderExists), vaultPath);
            return _folders.Contains(vaultPath);
        }

        public bool FileExists(string vaultPath)
        {
            Enter(nameof(FileExists), vaultPath);
            return _files.ContainsKey(vaultPath);
        }

        public PdmFileState GetFile(string vaultPath)
        {
            Enter(nameof(GetFile), vaultPath);
            return Find(vaultPath);
        }

        public IReadOnlyList<PdmFolderEntry> ListFolder(string vaultPath)
        {
            Enter(nameof(ListFolder), vaultPath);
            if (!_folders.Contains(vaultPath))
            {
                throw NotFoundAt(vaultPath);
            }

            var entries = new List<PdmFolderEntry>();
            foreach (var folder in _folders)
            {
                if (!string.Equals(folder, vaultPath, StringComparison.OrdinalIgnoreCase)
                    && string.Equals(VaultPaths.Parent(folder), vaultPath, StringComparison.OrdinalIgnoreCase))
                {
                    entries.Add(new PdmFolderEntry { Name = LastSegment(folder), IsFolder = true });
                }
            }

            foreach (var file in _files.Values)
            {
                if (string.Equals(VaultPaths.Parent(file.VaultPath), vaultPath, StringComparison.OrdinalIgnoreCase))
                {
                    entries.Add(new PdmFolderEntry
                    {
                        Name = file.Name,
                        IsFolder = false,
                        LatestVersion = file.LatestVersion,
                        LocalVersion = file.LocalVersion,
                        CheckedOut = file.CheckedOut,
                    });
                }
            }

            return entries;
        }

        public PdmVariableSet GetVariables(string vaultPath, string configuration)
        {
            Enter(nameof(GetVariables), vaultPath + "#" + configuration);
            Find(vaultPath);
            return new PdmVariableSet
            {
                Configurations = new List<string> { "Default", "Small" },
                Variables = (ConfigurationVariables.TryGetValue(configuration, out var configured) ? configured : Variables)
                    .Select(pair => new PdmVariableValue { Name = pair.Key, Value = pair.Value }).ToList(),
            };
        }

        public PdmGetOutcome GetLatest(string vaultPath, bool includeReferences, int maxFiles, Func<bool> shouldStop)
        {
            Enter(nameof(GetLatest), vaultPath);
            Find(vaultPath);
            var names = new List<string> { vaultPath };
            if (includeReferences)
            {
                names.AddRange(References);
            }

            if (names.Count > maxFiles)
            {
                throw new PdmGatewayException(PdmFailureKind.TooManyFiles, "More files than the limit.");
            }

            var outcome = new PdmGetOutcome();
            foreach (var name in names)
            {
                if (shouldStop())
                {
                    outcome.Skipped.Add(new PdmSkippedFile { VaultPath = name, Reason = "deadline" });
                    continue;
                }
                if (!_files.TryGetValue(name, out var state))
                {
                    continue;
                }

                if (state.CheckedOutHere)
                {
                    outcome.Skipped.Add(new PdmSkippedFile { VaultPath = name, Reason = "checked_out_here" });
                    continue;
                }

                state.LocalVersion = state.LatestVersion;
                state.ExistsLocally = true;
                outcome.Fetched.Add(name);
            }

            return outcome;
        }

        public void RequireNoLinks(string vaultPath)
        {
            Enter(nameof(RequireNoLinks), vaultPath);
            foreach (var link in LinkPaths)
            {
                if (string.Equals(vaultPath, link, StringComparison.OrdinalIgnoreCase)
                    || vaultPath.StartsWith(link + "\\", StringComparison.OrdinalIgnoreCase))
                {
                    throw new PdmGatewayException(PdmFailureKind.LinkInPath, "The path goes through a link.", vaultPath);
                }
            }
        }

        public void CreateFolder(string vaultPath)
        {
            Enter(nameof(CreateFolder), vaultPath);
            _folders.Add(vaultPath);
        }

        public int AddFile(string sourcePath, string folderVaultPath, string name)
        {
            var target = VaultPaths.Combine(folderVaultPath, name);
            Enter(nameof(AddFile), target);
            Added.Add((sourcePath, folderVaultPath, name));
            return SeedFile(target, file =>
            {
                file.LatestVersion = 0;
                file.LocalVersion = 1;
                file.EverCheckedIn = false;
                file.CheckedOut = true;
                file.CheckedOutBy = UserName;
                file.CheckedOutOn = CheckedOutOn;
                file.CheckedOutHere = true;
                file.LockedByUserId = 42;
                file.LockPath = file.LocalPath;
            }).FileId;
        }

        public void CheckOut(string vaultPath)
        {
            Enter(nameof(CheckOut), vaultPath);
            var file = Find(vaultPath);
            file.CheckedOut = true;
            file.CheckedOutBy = UserName;
            file.CheckedOutOn = CheckedOutOn;
            file.CheckedOutHere = true;
            file.LockedByUserId = 42;
            file.LockPath = file.LocalPath;
        }

        public void UndoCheckOut(string vaultPath)
        {
            Enter(nameof(UndoCheckOut), vaultPath);
            var file = Find(vaultPath);
            file.CheckedOut = false;
            file.CheckedOutBy = null;
            file.CheckedOutOn = null;
            file.CheckedOutHere = false;
            file.LockedByUserId = null;
            file.LockPath = null;
        }

        public void CheckIn(string vaultPath, string comment, bool keepCheckedOut)
        {
            Enter(nameof(CheckIn), vaultPath);
            var file = Find(vaultPath);
            CheckIns.Add((vaultPath, comment, keepCheckedOut));
            file.EverCheckedIn = true;
            file.LatestVersion++;
            file.LocalVersion = file.LatestVersion;
            file.CheckedOut = keepCheckedOut;
            file.CheckedOutHere = keepCheckedOut;
            file.CheckedOutBy = keepCheckedOut ? UserName : null;
            file.CheckedOutOn = keepCheckedOut ? CheckedOutOn : null;
            file.LockedByUserId = keepCheckedOut ? 42 : (int?)null;
            file.LockPath = keepCheckedOut ? file.LocalPath : null;
        }

        public IReadOnlyList<string> SetVariables(string vaultPath, string configuration, IReadOnlyDictionary<string, object> values)
        {
            Enter(nameof(SetVariables), vaultPath + "#" + configuration);
            Find(vaultPath);
            foreach (var name in values.Keys)
            {
                if (!Variables.ContainsKey(name))
                {
                    throw new PdmGatewayException(PdmFailureKind.UnknownVariable, "Unknown variable.", name);
                }
            }

            foreach (var pair in values)
            {
                Variables[pair.Key] = Convert.ToString(pair.Value, CultureInfo.InvariantCulture) ?? string.Empty;
            }

            VariableWrites.Add(new Dictionary<string, object>(values));
            return values.Keys.ToList();
        }

        private void Enter(string method, string argument)
        {
            Calls.Add(method + ":" + argument);
            _callCounts.TryGetValue(method, out var count);
            _callCounts[method] = ++count;
            OnCall?.Invoke(method, count);
            if (Failures.TryGetValue(method, out var failure))
            {
                throw failure;
            }
        }

        private PdmFileState Find(string vaultPath) =>
            _files.TryGetValue(vaultPath, out var file) ? file : throw NotFoundAt(vaultPath);

        private static PdmGatewayException NotFoundAt(string vaultPath) =>
            new PdmGatewayException(PdmFailureKind.NotFound, "Not found.", vaultPath);

        private static string LastSegment(string vaultPath) => vaultPath.Substring(vaultPath.LastIndexOf('\\') + 1);
    }

    /// <summary>Gateway over one <see cref="FakePdmVault"/>. Scripts the client facts and whole-run failures.</summary>
    internal sealed class FakePdmGateway : IPdmGateway
    {
        public FakePdmGateway()
        {
            Facts = new PdmClientFacts(true, "32.3.1", new[] { new PdmViewFacts(FakePdmVault.VaultName, FakePdmVault.RootPath, false, null) });
        }

        /// <summary>What GetClientFacts reports.</summary>
        public PdmClientFacts Facts { get; set; }

        public FakePdmVault Vault { get; } = new FakePdmVault();

        /// <summary>Vault names passed to RunAsync, in call order.</summary>
        public List<string> RunVaults { get; } = new List<string>();

        /// <summary>When set, RunAsync fails with this exception before any work runs.</summary>
        public PdmGatewayException? RunFailure { get; set; }

        public int RunCount => RunVaults.Count;

        public PdmClientFacts GetClientFacts() => Facts;

        public Task<T> RunAsync<T>(string vault, Func<IPdmVault, T> work, Deadline deadline, CancellationToken cancellationToken)
        {
            RunVaults.Add(vault);
            if (RunFailure != null)
            {
                return Task.FromException<T>(RunFailure);
            }

            try
            {
                return Task.FromResult(work(Vault));
            }
            catch (Exception error)
            {
                return Task.FromException<T>(error);
            }
        }
    }

    /// <summary>Scripted claim persistence; a failed removal leaves the stored claim unchanged.</summary>
    internal sealed class FakePdmClaimStore : IPdmClaimStore
    {
        private readonly InMemoryPdmClaimStore _inner = new InMemoryPdmClaimStore();

        public List<string> Calls { get; } = new List<string>();
        public Exception? RemoveFailure { get; set; }
        public Exception? PutFailure { get; set; }

        public PdmClaim? Find(string vault, int fileId) => _inner.Find(vault, fileId);

        public void Put(PdmClaim claim)
        {
            Calls.Add("Put");
            if (PutFailure != null) throw PutFailure;
            _inner.Put(claim);
        }

        public void Remove(string vault, int fileId)
        {
            Calls.Add("Remove");
            if (RemoveFailure != null) throw RemoveFailure;
            _inner.Remove(vault, fileId);
        }
    }

    /// <summary>Processor wired to the fakes. Sends request lines and parses the single response line.</summary>
    internal sealed class Harness
    {
        public Harness(OperationRegistry? registry = null, FakePdmGateway? pdm = null, string? workFolder = TestData.WorkFolder,
            IScriptRunner? scriptRunner = null)
        {
            Clock = new FakeClock();
            Manager.Clock = Clock;
            Pdm = pdm ?? new FakePdmGateway();
            Processor = new RequestProcessor(
                Manager,
                new FakeHostFacts(),
                registry ?? OperationRegistry.Default,
                TestData.HostVersion,
                ct =>
                {
                    ShutdownCalls++;
                    return Task.CompletedTask;
                },
                Clock,
                exception => Unexpected.Add(exception),
                Pdm,
                Claims,
                workFolder,
                () => UtcNow,
                scriptRunner);
        }

        public FakeInstanceManager Manager { get; } = new FakeInstanceManager();

        public FakePdmGateway Pdm { get; }

        /// <summary>The claim store the processor uses (files the tool checked out or added).</summary>
        public FakePdmClaimStore Claims { get; } = new FakePdmClaimStore();

        public DateTime UtcNow { get; set; } = new DateTime(2026, 11, 20, 12, 0, 0, DateTimeKind.Utc);

        public void Claim(PdmFileState file, string toolId = TestData.ToolId, bool addedByTool = false) =>
            Claims.Put(new PdmClaim
            {
                Vault = FakePdmVault.VaultName, FileId = file.FileId, ToolId = toolId, AddedByTool = addedByTool,
                LockedByUserId = file.LockedByUserId ?? 0, LockedOnComputer = file.CheckedOutOn ?? string.Empty,
                LockPath = file.LockPath ?? string.Empty, ClaimedUtc = UtcNow,
            });

        public FakeClock Clock { get; }

        public RequestProcessor Processor { get; }

        public int ShutdownCalls { get; private set; }

        public List<Exception> Unexpected { get; } = new List<Exception>();

        public async Task<string> SendRawAsync(string line) => await Processor.ProcessLineAsync(line, CancellationToken.None);

        public async Task<JObject> SendAsync(string line) => JObject.Parse(await SendRawAsync(line));

        public Task<JObject> CallAsync(string operation, JObject? arguments = null, JObject? context = null, string id = "req-1") =>
            SendAsync(TestData.Request(operation, arguments, context ?? TestData.Context(), id));

        public static JObject Args(params (string Name, JToken Value)[] pairs)
        {
            var args = new JObject();
            foreach (var pair in pairs)
            {
                args[pair.Name] = pair.Value;
            }

            return args;
        }

        public static JToken S(string value) => WireJson.Value(value);

        public static JToken I(int value) => WireJson.Value(value);
    }
}
