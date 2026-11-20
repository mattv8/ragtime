using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Pdm
{
    /// <summary>
    /// Link to the SOLIDWORKS PDM client on this computer. Every PDM COM object is created and used on one dedicated
    /// STA thread. Client facts come from a short cache and never wait long for that thread. One per host process.
    /// </summary>
    public sealed class PdmGateway : IPdmGateway, IDisposable
    {
        private static readonly TimeSpan FactsLifetime = TimeSpan.FromSeconds(30);
        private static readonly TimeSpan ProbeWait = TimeSpan.FromSeconds(2);
        private static readonly Action<string> NoLog = _ => { };

        private readonly object _factsGate = new object();
        private readonly PdmSessions _sessions;
        private readonly PdmWorker _worker;
        private readonly Action<string> _log;
        private CachedClient? _cached;
        private Task? _refresh;

        /// <summary>Creates the gateway. No COM object and no thread exist until a request needs them.</summary>
        /// <param name="log">Receives one line per PDM write call that reached PDM. Never file contents or data-card values.</param>
        public PdmGateway(Action<string>? log = null)
        {
            _log = log ?? NoLog;
            _sessions = new PdmSessions(_log);
            _worker = new PdmWorker(_sessions.ReleaseAll);
        }

        /// <inheritdoc />
        /// <remarks>Never logs in and never throws. The PDM thread is asked for views at most every 30 seconds.</remarks>
        public PdmClientFacts GetClientFacts()
        {
            try
            {
                return Describe(Current());
            }
            catch (Exception)
            {
                return PdmClientFacts.NotInstalled;
            }
        }

        /// <inheritdoc />
        public async Task<T> RunAsync<T>(string vault, Func<IPdmVault, T> work, Deadline deadline, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var item = _worker.Enqueue(() => work(_sessions.Get(vault)), error =>
                _log($"{DateTime.UtcNow:O} op={JsonConvert.ToString(work.Method.Name)} vault={JsonConvert.ToString(vault)} completed_late outcome={JsonConvert.ToString(error == null ? "ok" : error.GetType().Name)}"));
            var timeout = Task.Delay(deadline.Remaining, cancellationToken);
            var finished = await Task.WhenAny(item.Task, timeout).ConfigureAwait(false);
            if (ReferenceEquals(finished, item.Task))
            {
                return await item.Task.ConfigureAwait(false);
            }

            switch (item.Abandon())
            {
                case PdmAbandonOutcome.Done:
                    return await item.Task.ConfigureAwait(false);
                case PdmAbandonOutcome.Cancelled when cancellationToken.IsCancellationRequested:
                    throw new OperationCanceledException(cancellationToken);
                default:
                    throw PdmErrors.TimedOut();
            }
        }

        /// <summary>Stops the PDM thread after its current call; queued calls fail. Vault sessions are released on that thread.</summary>
        public void Dispose()
        {
            _worker.Stop();
        }

        private CachedClient Current()
        {
            Task? first = null;
            lock (_factsGate)
            {
                if (_cached == null)
                {
                    _cached = new CachedClient(false, null, new PdmViewFacts[0], DateTime.MinValue);
                    first = StartRefresh();
                }
                else if (DateTime.UtcNow - _cached.At >= FactsLifetime)
                {
                    StartRefresh();
                }
            }
            if (first != null) first.Wait(ProbeWait);
            lock (_factsGate) return _cached!;
        }

        private Task StartRefresh()
        {
            if (_refresh != null && !_refresh.IsCompleted) return _refresh;
            _refresh = Task.Run(() =>
            {
                var installed = IsInstalled();
                var probe = installed ? ProbeWorker() : null;
                lock (_factsGate)
                {
                    var views = installed ? probe?.Views ?? _cached!.Views : new PdmViewFacts[0];
                    _cached = new CachedClient(installed, probe?.Version ?? (installed ? _cached!.Version : null), views, DateTime.UtcNow);
                }
            });
            return _refresh;
        }

        private PdmClientFacts Describe(CachedClient client)
        {
            if (!client.Installed)
            {
                return PdmClientFacts.NotInstalled;
            }

            var loggedIn = _sessions.LoggedIn();
            var views = new List<PdmViewFacts>();
            foreach (var view in client.Views)
            {
                var isLoggedIn = loggedIn.TryGetValue(view.Vault, out var user);
                views.Add(new PdmViewFacts(view.Vault, view.Root, isLoggedIn, isLoggedIn ? user : null));
            }

            return new PdmClientFacts(true, client.Version, views);
        }

        private PdmProbe? ProbeWorker()
        {
            try
            {
                var item = _worker.Enqueue(() => PdmClientProbe.Probe());
                if (item.Task.Wait(ProbeWait))
                {
                    return item.Task.Result;
                }

                item.Abandon();
                return null;
            }
            catch (Exception)
            {
                return null;
            }
        }

        private static bool IsInstalled()
        {
            try
            {
                return PdmClientRegistry.IsInstalled();
            }
            catch (Exception)
            {
                return false;
            }
        }

        private sealed class CachedClient
        {
            public CachedClient(bool installed, string? version, IReadOnlyList<PdmViewFacts> views, DateTime at)
            {
                Installed = installed;
                Version = version;
                Views = views;
                At = at;
            }

            public bool Installed { get; }

            public string? Version { get; }

            /// <summary>Views without login state; the login state is added when facts are described.</summary>
            public IReadOnlyList<PdmViewFacts> Views { get; }

            public DateTime At { get; }
        }
    }
}
