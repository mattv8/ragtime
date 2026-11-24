using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// The host's request entry point. <see cref="ProcessLineAsync"/> takes one request line and returns one
    /// response line; it never throws. Flow: parse, look up the operation, write gate, argument validation,
    /// dispatch, paging or budget check, envelope.
    /// <para>
    /// Construction: the Host lane creates one processor per host process, with the Runtime's instance manager,
    /// the Host's facts provider, <see cref="OperationRegistry.Default"/>, the assembly version, and a shutdown
    /// callback that stops the pipe server and calls <see cref="ISolidWorksInstanceManager.ShutdownAsync"/>.
    /// The processor subscribes to <see cref="ISolidWorksInstanceManager.InstanceStopped"/> to forget documents.
    /// </para>
    /// <para>
    /// Threading: <see cref="ProcessLineAsync"/> may be called concurrently (one call per pipe connection). The
    /// document registry is thread-safe. Session work is done on the instance threads by the instance manager.
    /// </para>
    /// <para>
    /// Deadlines: one <see cref="Deadline"/> is created per request, when it starts, and every instance-manager
    /// call of that request receives it (refreshes, document resolution and the operation share one budget). Once
    /// it has expired, no further instance call is made and the request answers operation_timeout.
    /// </para>
    /// <para>
    /// Instance modes: a request sees and addresses only the documents of instances of its own
    /// <see cref="RequestContext.InstanceMode"/>. Documents of the other mode are neither found nor listed, and
    /// refreshes never remove them from the registry.
    /// </para>
    /// </summary>
    public sealed class RequestProcessor
    {
        private readonly ISolidWorksInstanceManager _instances;
        private readonly IHostFactsProvider _hostFacts;
        private readonly OperationRegistry _registry;
        private readonly string _hostVersion;
        private readonly Func<CancellationToken, Task> _shutdown;
        private readonly IMonotonicClock _clock;
        private readonly Action<Exception>? _unexpectedErrorSink;
        private readonly IPdmGateway? _pdm;
        private readonly IPdmClaimStore _pdmClaims;
        private readonly string? _pdmWorkFolder;
        private readonly Func<DateTime> _utcNow;
        private readonly IScriptRunner? _scriptRunner;
        private readonly Action<string>? _hostLog;
        private readonly DocumentRegistry _documents = new DocumentRegistry();
        private readonly object _activityLock = new object();
        private int _inFlight;
        private long _lastWorkEndedMs;
        private long _activityGeneration;
        private bool _stopping;

        /// <summary>Creates the processor.</summary>
        /// <param name="instances">Instance manager (Runtime lane).</param>
        /// <param name="hostFacts">Facts for status (Host lane).</param>
        /// <param name="registry">Operation registry; normally <see cref="OperationRegistry.Default"/>.</param>
        /// <param name="hostVersion">Assembly informational version, e.g. "0.1.0".</param>
        /// <param name="shutdown">Called for host.shutdown; must stop the host (the response is sent first).</param>
        /// <param name="clock">Clock for elapsed_ms and the request deadline; default is a system stopwatch.</param>
        /// <param name="unexpectedErrorSink">Receives exceptions that become internal_error, for the host log. Optional.</param>
        /// <param name="pdm">PDM gateway for the pdm.* operations. Null means no PDM client: every pdm.* operation except
        /// pdm.status answers pdm_unavailable (client_missing), and status reports client_installed false.</param>
        /// <param name="pdmClaims">Files this tool checked out or added, which only it may finish. Default: in memory.</param>
        /// <param name="pdmWorkFolder">Host work folder accepted as an add source, in addition to approved view folders.</param>
        /// <param name="utcNow">UTC clock for check-out claim expiry.</param>
        /// <param name="scriptRunner">Script worker runner; null means scripts are unavailable in this host.</param>
        /// <param name="hostLog">Optional informational host log sink.</param>
        public RequestProcessor(
            ISolidWorksInstanceManager instances,
            IHostFactsProvider hostFacts,
            OperationRegistry registry,
            string hostVersion,
            Func<CancellationToken, Task> shutdown,
            IMonotonicClock? clock = null,
            Action<Exception>? unexpectedErrorSink = null,
            IPdmGateway? pdm = null,
            IPdmClaimStore? pdmClaims = null,
            string? pdmWorkFolder = null,
            Func<DateTime>? utcNow = null,
            IScriptRunner? scriptRunner = null,
            Action<string>? hostLog = null)
        {
            _instances = instances ?? throw new ArgumentNullException(nameof(instances));
            _hostFacts = hostFacts ?? throw new ArgumentNullException(nameof(hostFacts));
            _registry = registry ?? throw new ArgumentNullException(nameof(registry));
            _hostVersion = hostVersion ?? throw new ArgumentNullException(nameof(hostVersion));
            _shutdown = shutdown ?? throw new ArgumentNullException(nameof(shutdown));
            _clock = clock ?? new SystemMonotonicClock();
            _lastWorkEndedMs = _clock.ElapsedMilliseconds;
            _unexpectedErrorSink = unexpectedErrorSink;
            _pdm = pdm;
            _pdmClaims = pdmClaims ?? new InMemoryPdmClaimStore();
            _pdmWorkFolder = pdmWorkFolder;
            _utcNow = utcNow ?? (() => DateTime.UtcNow);
            _scriptRunner = scriptRunner;
            _hostLog = hostLog;
            _instances.InstanceStopped += OnInstanceStopped;
        }

        /// <summary>
        /// Processes one request line (without the trailing newline) and returns the response envelope as one
        /// compact line. Never throws; every failure becomes an error envelope.
        /// </summary>
        public async Task<string> ProcessLineAsync(string requestLine, CancellationToken cancellationToken)
        {
            var started = _clock.ElapsedMilliseconds;
            var id = RequestParser.TryReadString(requestLine, "id");
            var operation = RequestParser.TryReadString(requestLine, "operation");

            try
            {
                var request = RequestParser.Parse(requestLine);
                id = request.Id;
                operation = request.Operation;

                var outcome = await ExecuteAsync(request, cancellationToken).ConfigureAwait(false);
                return ResponseEnvelope.ToLine(
                    ResponseEnvelope.Success(id, operation, outcome.Result, outcome.Host, ElapsedSince(started)));
            }
            catch (HostErrorException error)
            {
                return ResponseEnvelope.ToLine(ResponseEnvelope.FromException(id, operation, error, _hostVersion, ElapsedSince(started)));
            }
            catch (OperationCanceledException)
            {
                return ResponseEnvelope.ToLine(ResponseEnvelope.Failure(
                    id,
                    operation,
                    ErrorCodes.InternalError,
                    "The request was cancelled before it finished.",
                    null,
                    HostInfo.ForHost(_hostVersion),
                    ElapsedSince(started)));
            }
            catch (Exception exception)
            {
                ReportUnexpected(exception);
                return ResponseEnvelope.ToLine(ResponseEnvelope.Failure(
                    id,
                    operation,
                    ErrorCodes.InternalError,
                    "The SolidWorks agent failed while handling the request. See the SolidWorks agent log.",
                    new JObject { ["exception_type"] = WireJson.Value(exception.GetType().Name) },
                    HostInfo.ForHost(_hostVersion),
                    ElapsedSince(started)));
            }
        }

        private async Task<Outcome> ExecuteAsync(ParsedRequest request, CancellationToken cancellationToken)
        {
            var definition = _registry.Find(request.Operation)
                ?? throw HostError.Coded(
                    ErrorCodes.UnknownOperation,
                    $"There is no operation named '{request.Operation}'. Call describe for the list.");

            var isWork = definition.Name != "describe" && definition.Name != "status"
                && definition.Name != "pdm.status" && definition.Name != "host.shutdown";
            if (isWork)
            {
                lock (_activityLock)
                {
                    if (_stopping)
                    {
                        throw HostError.Coded(ErrorCodes.HostStopping, "The SolidWorks agent is stopping for an update or restart. Try again in a minute.");
                    }
                    _inFlight++;
                    _activityGeneration++;
                }
            }

            try
            {
                return await ExecuteRegisteredAsync(definition, request, cancellationToken).ConfigureAwait(false);
            }
            finally
            {
                if (isWork)
                {
                    lock (_activityLock)
                    {
                        _inFlight--;
                        _lastWorkEndedMs = _clock.ElapsedMilliseconds;
                        _activityGeneration++;
                    }
                }
            }
        }

        private async Task<Outcome> ExecuteRegisteredAsync(OperationDefinition definition, ParsedRequest request, CancellationToken cancellationToken)
        {
            switch (definition.Access)
            {
                case ProtocolInfo.Access.Write when !request.Context.AllowWrite:
                    throw HostError.Coded(
                        ErrorCodes.WriteNotAllowed,
                        $"'{definition.Name}' changes SolidWorks state, which this tool's configuration does not allow.");
                case ProtocolInfo.Access.Script when !request.Context.AllowScripts:
                    throw HostError.Coded(ErrorCodes.ScriptsNotAllowed, "Scripts are not enabled for this tool.");
            }

            ArgumentValidator.Validate(definition.ArgumentsSchema, request.Arguments);
            var context = request.Context;
            if (definition.Name == "script.run")
            {
                if (context.DeadlineSeconds > 600)
                {
                    throw HostError.InvalidArguments("timeout", "Script timeout must be at most 600 seconds.");
                }
                var scriptDeadline = new Deadline(context.DeadlineSeconds, _clock);
                return await RunScriptAsync(request, scriptDeadline, cancellationToken).ConfigureAwait(false);
            }
            if (PdmOperations.IsPdmOperation(definition.Name))
            {
                // PDM operations never start or use a SolidWorks instance, so they skip the version pin check.
                var pdmDeadline = new Deadline(context.DeadlineSeconds, _clock);
                var pdm = new PdmOperations(_pdm, context, pdmDeadline, cancellationToken, _pdmClaims, _pdmWorkFolder, _utcNow, ReportUnexpected);
                return new Outcome(
                    await pdm.ExecuteAsync(definition.Name, request.Arguments).ConfigureAwait(false),
                    HostInfo.ForHost(_hostVersion));
            }

            RequireInstalledPin(definition.Name, context);
            var deadline = new Deadline(context.DeadlineSeconds, _clock);

            switch (definition.Name)
            {
                case "describe":
                    return Plain(DescribeBuilder.Build(_registry, _hostVersion), context);

                case "status":
                    return Plain(BuildStatus(), context);

                case "host.shutdown":
                    await ShutdownAsync(request.Arguments, cancellationToken).ConfigureAwait(false);
                    return Plain(new ShutdownResult(), context);

                case "documents.list":
                    await RefreshDocumentsAsync(context, deadline, cancellationToken).ConfigureAwait(false);
                    var scope = ScopeOf(context);
                    return new Outcome(
                        Pager.Page(
                            _documents.Snapshot().Where(scope).Select(entry => entry.ToItem()).ToList(),
                            PageRequest.FromArguments(request.Arguments),
                            ProtocolInfo.SiUnits,
                            context.MaxResultBytes),
                        HostInfo.ForHost(_hostVersion));

                case "document.open":
                    return await OpenDocumentAsync(request, deadline, cancellationToken).ConfigureAwait(false);

                case "document.close":
                    return await CloseDocumentAsync(request, deadline, cancellationToken).ConfigureAwait(false);

                default:
                    return await ModelOperationAsync(definition, request, deadline, cancellationToken).ConfigureAwait(false);
            }
        }

        private async Task ShutdownAsync(JObject arguments, CancellationToken cancellationToken)
        {
            var threshold = (int?)arguments["if_idle_seconds"];
            if (threshold.HasValue)
            {
                StopIfIdle(threshold.Value);
            }
            else
            {
                lock (_activityLock)
                {
                    _stopping = true;
                }
            }
            await _shutdown(cancellationToken).ConfigureAwait(false);
        }

        private void StopIfIdle(int threshold)
        {
            for (var attempt = 0; attempt < 3; attempt++)
            {
                long generation;
                lock (_activityLock)
                {
                    if (_stopping) return;
                    generation = _activityGeneration;
                }

                var instances = CurrentInstances();
                var pdmBusy = _pdm?.IsBusy ?? false;
                lock (_activityLock)
                {
                    if (_stopping) return;
                    if (_activityGeneration != generation)
                    {
                        if (attempt < 2) continue;
                        throw HostError.Coded(ErrorCodes.HostBusy, "The SolidWorks agent is in use.",
                            new JObject { ["reasons"] = new JArray(HostIdleRule.RequestInFlight) });
                    }

                    var reasons = HostIdleRule.BlockingReasons(_inFlight, IdleSeconds(), threshold, instances, pdmBusy);
                    if (reasons.Count > 0)
                    {
                        throw HostError.Coded(ErrorCodes.HostBusy, "The SolidWorks agent is in use.",
                            new JObject { ["reasons"] = new JArray(reasons) });
                    }
                    _stopping = true;
                    return;
                }
            }
        }

        private int IdleSeconds() => _inFlight > 0
            ? 0
            : (int)Math.Min(int.MaxValue, (_clock.ElapsedMilliseconds - _lastWorkEndedMs) / 1000);

        private async Task<Outcome> RunScriptAsync(ParsedRequest request, Deadline deadline, CancellationToken cancellationToken)
        {
            if (_scriptRunner == null)
            {
                throw HostError.Coded(ErrorCodes.InternalError, "Scripts are not available in this SolidWorks agent.");
            }
            RequireInstalledPin(request.Operation, request.Context);
            var instance = ArgumentReader.OptionalString(request.Arguments, "instance");
            var target = instance != null ? InstanceTarget.Specific(instance)
                : request.Context.IsAttachMode ? InstanceTarget.Attach() : ChooseOpenTarget(request.Context);
            var operation = new ScriptOperations(_instances, _documents, _scriptRunner, _pdmWorkFolder ?? string.Empty, _hostLog);
            var run = await operation.ExecuteAsync(target, request, deadline, cancellationToken).ConfigureAwait(false);
            return new Outcome(Fits(run.Value, request.Context), HostInfoOf(run.Instance));
        }

        private async Task<Outcome> OpenDocumentAsync(ParsedRequest request, Deadline deadline, CancellationToken cancellationToken)
        {
            var context = request.Context;
            if (context.IsAttachMode)
            {
                throw HostError.InvalidArguments(
                    "path",
                    "document.open is not available in attach mode. Open the model in SolidWorks and use its path with the other operations.");
            }

            var path = ArgumentReader.RequiredString(request.Arguments, "path");
            DocumentPaths.RequireOpenable(path);
            var configuration = ArgumentReader.OptionalString(request.Arguments, "configuration");

            var target = ChooseOpenTarget(context);
            var run = await RunOnInstanceAsync(
                target,
                context,
                deadline,
                session => session.OpenReadOnlySilent(path, configuration),
                cancellationToken).ConfigureAwait(false);

            var entry = _documents.Register(run.Instance.InstanceId, run.Value, openedByHost: !run.Value.WasAlreadyOpen);
            return new Outcome(Fits(entry.ToItem(), context), HostInfoOf(run.Instance));
        }

        private InstanceTarget ChooseOpenTarget(RequestContext context)
        {
            var instances = CurrentInstances();
            var choice = InstancePool.ChooseForOpen(instances, context.MaxInstances, context.SolidworksVersion);
            if (choice.Action == PoolAction.LimitReached)
            {
                throw InstanceLimitError(context);
            }

            InstanceTarget target;
            if (choice.Action == PoolAction.StartNew)
            {
                // The host's admission counts every background instance: when the pool is already at max_instances,
                // an idle other-year instance must be closed first, or the launch is refused here.
                string? retire = null;
                if (instances.Count(instance => instance.Mode == ProtocolInfo.InstanceModes.Background) >= context.MaxInstances)
                {
                    retire = (InstanceRoom.InstanceToRetire(instances, context.SolidworksVersion)?.InstanceId)
                        ?? throw InstanceLimitError(context);
                }

                target = InstanceTarget.StartNew(retire);
            }
            else
            {
                target = InstanceTarget.Specific(choice.InstanceId ?? throw new InvalidOperationException("Pool chose an instance without an id."));
            }

            return target;
        }

        private async Task<Outcome> CloseDocumentAsync(ParsedRequest request, Deadline deadline, CancellationToken cancellationToken)
        {
            var context = request.Context;
            if (context.IsAttachMode)
            {
                // Refused before any lookup: attach mode never closes a document, even one the host opened earlier.
                throw HostError.InvalidArguments(
                    "document",
                    "document.close is not available in attach mode. The SolidWorks agent closes only the documents it opened in background mode.");
            }

            var document = await ResolveDocumentAsync(
                ArgumentReader.RequiredString(request.Arguments, "document"),
                context,
                deadline,
                cancellationToken).ConfigureAwait(false);

            if (!document.OpenedByHost)
            {
                throw HostError.InvalidArguments(
                    "document",
                    "Only documents that the SolidWorks agent opened can be closed. This one was already open in SolidWorks, so it was left alone.");
            }

            var run = await RunOnInstanceAsync(
                InstanceTarget.Specific(document.InstanceId),
                context,
                deadline,
                session =>
                {
                    session.CloseDocument(document.ToRef());
                    return true;
                },
                cancellationToken).ConfigureAwait(false);

            _documents.Remove(document.DocumentId);
            return new Outcome(Fits(new ClosedResult(), context), HostInfoOf(run.Instance));
        }

        private async Task<Outcome> ModelOperationAsync(OperationDefinition definition, ParsedRequest request, Deadline deadline, CancellationToken cancellationToken)
        {
            var context = request.Context;
            var document = await ResolveDocumentAsync(
                ArgumentReader.RequiredString(request.Arguments, "document"),
                context,
                deadline,
                cancellationToken).ConfigureAwait(false);

            if (ModelOperations.RequiresAssembly(definition.Name) && document.Type != ProtocolInfo.DocumentTypes.Assembly)
            {
                throw HostError.Coded(
                    ErrorCodes.NotSupportedForDocumentType,
                    $"{definition.Name} applies only to assemblies, and '{document.Title}' is a {document.Type}.",
                    new JObject { ["document_type"] = WireJson.Value(document.Type) });
            }

            var arguments = request.Arguments;
            var run = await RunOnInstanceAsync(
                InstanceTarget.Specific(document.InstanceId),
                context,
                deadline,
                session => ModelOperations.Execute(definition.Name, session, document, arguments, context),
                cancellationToken).ConfigureAwait(false);

            return new Outcome(run.Value, HostInfoOf(run.Instance));
        }

        /// <summary>
        /// Finds a document of this request's instance mode by id or path; if unknown, refreshes the registry from
        /// the instances once and looks again.
        /// </summary>
        private async Task<DocumentEntry> ResolveDocumentAsync(string argument, RequestContext context, Deadline deadline, CancellationToken cancellationToken)
        {
            var entry = _documents.Find(argument, ScopeOf(context));
            if (entry != null)
            {
                return entry;
            }

            await RefreshDocumentsAsync(context, deadline, cancellationToken).ConfigureAwait(false);

            // The refresh may have registered the instance (first attach): the scope must be rebuilt after it.
            entry = _documents.Find(argument, ScopeOf(context));
            if (entry != null)
            {
                return entry;
            }

            if (DocumentRegistry.LooksLikeDocumentId(argument))
            {
                throw HostError.Coded(
                    ErrorCodes.DocumentNotFound,
                    $"No open document has the id {argument}. Call documents.list to see the current ids.");
            }

            throw HostError.Coded(
                ErrorCodes.DocumentNotOpen,
                "That file is not open in SolidWorks. Open it with document.open first, or check the path.");
        }

        /// <summary>
        /// Replaces the registry's view of each live instance of the request's mode with what the session reports.
        /// Background mode queries every background instance; attach mode queries the attached instance (and fails with
        /// solidworks_not_running when none exists). Documents of other instances are kept, whatever their mode.
        /// A background instance of another version pin is skipped (its documents are kept).
        /// </summary>
        private async Task RefreshDocumentsAsync(RequestContext context, Deadline deadline, CancellationToken cancellationToken)
        {
            if (context.IsAttachMode)
            {
                var attached = await RunOnInstanceAsync(
                    InstanceTarget.Attach(),
                    context,
                    deadline,
                    session => session.ListOpenDocuments(),
                    cancellationToken).ConfigureAwait(false);

                _documents.RetainInstances(LiveInstanceIds());
                _documents.Synchronize(attached.Instance.InstanceId, attached.Value);
                return;
            }

            _documents.RetainInstances(LiveInstanceIds());
            var backgroundInstances = _instances.GetInstances()
                .Where(instance => instance.Mode == ProtocolInfo.InstanceModes.Background)
                .ToList();

            foreach (var instance in backgroundInstances)
            {
                try
                {
                    var listed = await RunOnInstanceAsync(
                        InstanceTarget.Specific(instance.InstanceId),
                        context,
                        deadline,
                        session => session.ListOpenDocuments(),
                        cancellationToken).ConfigureAwait(false);

                    _documents.Synchronize(listed.Instance.InstanceId, listed.Value);
                }
                catch (HostErrorException error) when (error.Code == ErrorCodes.SolidworksNotRunning)
                {
                    _documents.ForgetInstance(instance.InstanceId);
                }
                catch (HostErrorException error) when (error.Code == ErrorCodes.VersionMismatch)
                {
                    // Another version's instance: not part of this pin. Its documents stay as they are.
                }
            }
        }

        /// <summary>
        /// Runs one instance call of the request. The request's deadline is shared by every call, so when it has
        /// already expired nothing more is scheduled and the request answers operation_timeout.
        /// </summary>
        private async Task<InstanceRun<T>> RunOnInstanceAsync<T>(
            InstanceTarget target,
            RequestContext context,
            Deadline deadline,
            Func<ISolidWorksSession, T> work,
            CancellationToken cancellationToken)
        {
            if (deadline.Expired)
            {
                throw HostError.Coded(
                    ErrorCodes.OperationTimeout,
                    $"The request passed its {(int)deadline.Total.TotalSeconds}-second deadline before the next SolidWorks step started. Nothing more was run; try again.");
            }

            return await _instances.RunAsync(target, context, deadline, work, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Background requests with a pin need that year installed. Answered before any pool decision or instance call.
        /// Attach requests and the operations that never touch an instance are not checked.
        /// </summary>
        private void RequireInstalledPin(string operation, RequestContext context)
        {
            if (context.IsAttachMode || !RunsOnInstances(operation) || !context.SolidworksVersion.HasValue)
            {
                return;
            }

            var pin = context.SolidworksVersion.Value;
            var installed = _instances.GetInstalledVersions();
            if (installed.Contains(pin))
            {
                return;
            }

            var installedArray = new JArray();
            foreach (var year in installed.OrderBy(year => year))
            {
                installedArray.Add(WireJson.Value(year));
            }

            throw HostError.Coded(
                ErrorCodes.VersionNotInstalled,
                $"SolidWorks {pin} is not installed on this computer. Install it or change the pinned version.",
                new JObject
                {
                    ["pinned_version"] = WireJson.Value(pin),
                    ["installed_versions"] = installedArray,
                });
        }

        private static bool RunsOnInstances(string operation) =>
            operation != "describe" && operation != "status" && operation != "host.shutdown";

        /// <summary>
        /// instance_limit with an accurate message: "busy" when every background instance is busy, otherwise the
        /// pool is full of instances that are idle but hold documents or belong to another pinned version.
        /// </summary>
        private HostErrorException InstanceLimitError(RequestContext context)
        {
            var background = CurrentInstances()
                .Where(instance => instance.Mode == ProtocolInfo.InstanceModes.Background)
                .ToList();
            var allBusy = background.Count > 0 && background.All(instance => instance.Busy);

            var message = allBusy
                ? $"All {context.MaxInstances} SolidWorks instances are busy. Try again in a moment."
                : $"All {context.MaxInstances} background instances are in use (another pinned version or open documents). Close documents or wait for idle shutdown.";
            return HostError.Coded(
                ErrorCodes.InstanceLimit,
                message,
                new JObject { ["max_instances"] = WireJson.Value(context.MaxInstances) });
        }

        /// <summary>Predicate: the document's instance has the request's instance mode.</summary>
        private Func<DocumentEntry, bool> ScopeOf(RequestContext context)
        {
            var modes = _instances.GetInstances().ToDictionary(instance => instance.InstanceId, instance => instance.Mode);
            return entry => modes.TryGetValue(entry.InstanceId, out var mode) && mode == context.InstanceMode;
        }

        private IReadOnlyList<string> LiveInstanceIds() =>
            _instances.GetInstances().Select(instance => instance.InstanceId).ToList();

        private void OnInstanceStopped(string instanceId) => _documents.ForgetInstance(instanceId);

        /// <summary>
        /// Instance snapshots with the document counts from the registry. The instance manager does not track
        /// documents, so Core's count is the one the pool and status must use.
        /// </summary>
        private IReadOnlyList<InstanceSnapshot> CurrentInstances() =>
            _instances.GetInstances()
                .Select(instance => instance.WithDocumentCount(_documents.CountForInstance(instance.InstanceId)))
                .ToList();

        private StatusResult BuildStatus()
        {
            var facts = _hostFacts.GetHostFacts();
            var instances = CurrentInstances()
                .Select(ToInstanceInfo)
                .ToList();
            HostActivityInfo activity;
            lock (_activityLock)
            {
                activity = new HostActivityInfo
                {
                    InFlight = _inFlight,
                    IdleSeconds = IdleSeconds(),
                    PdmBusy = _pdm?.IsBusy ?? false,
                };
            }

            return new StatusResult
            {
                HostVersion = _hostVersion,
                Protocol = ProtocolInfo.Version,
                Account = new AccountInfo
                {
                    Name = facts.AccountName,
                    Sid = facts.Sid,
                    IsAdministrator = facts.IsAdministrator,
                },
                Session = new SessionInfo
                {
                    Id = facts.SessionId,
                    Interactive = facts.Interactive,
                },
                Solidworks = new SolidworksStatus
                {
                    InstalledVersions = _instances.GetInstalledVersions().OrderBy(year => year).ToList(),
                    Instances = instances,
                },
                Pdm = PdmOperations.ClientStatusOf(_pdm),
                Activity = activity,
                UptimeSeconds = facts.UptimeSeconds,
            };
        }

        private static InstanceInfo ToInstanceInfo(InstanceSnapshot instance) =>
            new InstanceInfo
            {
                InstanceId = instance.InstanceId,
                Mode = instance.Mode,
                Owned = instance.Owned,
                Version = instance.Version,
                Revision = instance.Revision,
                Pid = instance.ProcessId,
                Visible = instance.Visible,
                Documents = instance.DocumentCount,
                Busy = instance.Busy,
            };

        private HostInfo HostInfoOf(InstanceSnapshot instance) =>
            new HostInfo(_hostVersion, instance.Revision, instance.Version, instance.InstanceId);

        private Outcome Plain(object dto, RequestContext context) => new Outcome(Fits(dto, context), HostInfo.ForHost(_hostVersion));

        private static JToken Fits(object dto, RequestContext context) =>
            Pager.RequireFits(WireJson.ToToken(dto), context.MaxResultBytes);

        private long ElapsedSince(long started) => _clock.ElapsedMilliseconds - started;

        private void ReportUnexpected(Exception exception)
        {
            try
            {
                _unexpectedErrorSink?.Invoke(exception);
            }
            catch (Exception)
            {
                // The log sink must never turn an error response into a crash.
            }
        }

        /// <summary>Result of one operation: the JSON result and the host block of the envelope.</summary>
        private sealed class Outcome
        {
            public Outcome(JToken result, HostInfo host)
            {
                Result = result;
                Host = host;
            }

            public JToken Result { get; }

            public HostInfo Host { get; }
        }
    }
}
