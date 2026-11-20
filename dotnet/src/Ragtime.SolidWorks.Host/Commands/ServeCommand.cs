using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Ragtime.SolidWorks.Core;
using Ragtime.SolidWorks.Pdm;
using Ragtime.SolidWorks.Readers;
using Ragtime.SolidWorks.Runtime;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// <c>serve</c> and the composition root. Wires Runtime + Readers + Pdm + Core + the pipe server, then blocks until a
    /// shutdown is requested, stops accepting, drains, closes owned instances and exits.
    /// </summary>
    internal static class ServeCommand
    {
        private static readonly TimeSpan DrainTimeout = TimeSpan.FromSeconds(5);
        private static readonly TimeSpan ManagerShutdownTimeout = TimeSpan.FromSeconds(15);
        private const string WorkFolderRelative = @"RagtimeSolidWorks\work";

        public static int Run(HostEnvironment env)
        {
            var log = env.Log;
            var sessionId = Process.GetCurrentProcess().SessionId;
            if (sessionId == 0 || !Environment.UserInteractive)
            {
                var reason = "serve needs the interactive desktop session of the user; this process runs in session " + sessionId + ".";
                Console.Error.WriteLine("ragtime-solidworks-host: " + reason);
                log.Write("serve refused: " + reason);
                return ExitCodes.Failure;
            }

            ConsoleControl.HideWindow();
            HostMutex? mutex;
            try
            {
                mutex = HostMutex.TryAcquire(env.MutexName, env.Sid);
            }
            catch (HostMutexException exception)
            {
                log.Write("serve refused: " + exception.Message + ".");
                Console.Error.WriteLine("ragtime-solidworks-host: serve refused: " + exception.Message + ".");
                return ExitCodes.Failure;
            }

            if (mutex == null)
            {
                log.Write("serve: a host is already running for this user; exiting.");
                return ExitCodes.Success;
            }

            using (mutex)
            {
                return RunUntilShutdown(env, sessionId);
            }
        }

        private static int RunUntilShutdown(HostEnvironment env, int sessionId)
        {
            var log = env.Log;
            log.Write($"serve starting: version {env.HostVersion}, account {env.AccountName}, session {sessionId}");

            var shutdown = new ShutdownSignal();
            if (!ConsoleControl.RegisterStopHandler(shutdown.Request))
            {
                log.Write("serve: could not register the console stop handler; logoff will not stop the host cleanly.");
            }

            using (var manager = new SolidWorksInstanceManager(new SolidWorksSessionFactory(), log.Write))
            using (var pdm = CreatePdmGateway(log))
            {
                var processor = new RequestProcessor(
                    manager,
                    new HostFactsProvider(),
                    OperationRegistry.Default,
                    env.HostVersion,
                    shutdown.RequestAsync,
                    clock: null,
                    unexpectedErrorSink: log.WriteException,
                    pdm: pdm,
                    pdmClaims: FilePdmClaimStore.ForCurrentUser(log.Write),
                    pdmWorkFolder: EnsureWorkFolder(log));

                using (var server = new PipeServer(env.PipeName, PipeAccess.ForUser(env.Sid), processor, env.HostVersion, log))
                {
                    try
                    {
                        server.Start();
                    }
                    catch (PipeNameInUseException exception)
                    {
                        log.Write("pipe name already in use by another process; serve exits. " + exception.Message);
                        Console.Error.WriteLine("ragtime-solidworks-host: the pipe name is already in use by another process.");
                        return ExitCodes.Failure;
                    }

                    shutdown.Wait();

                    log.Write("serve: shutdown requested; no longer accepting connections.");
                    server.StopAccepting();
                    server.DrainAsync(DrainTimeout).GetAwaiter().GetResult();
                    StopInstances(manager, log);
                }
            }

            log.Write("serve stopped.");
            return ExitCodes.Success;
        }

        /// <summary>
        /// Creates %LOCALAPPDATA%\RagtimeSolidWorks\work\ and returns its full path, the folder pdm.file.add may read from
        /// besides approved vault folders. Null when it cannot be created (failure logged). Never throws.
        /// </summary>
        private static string? EnsureWorkFolder(HostLog log)
        {
            try
            {
                var root = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                if (string.IsNullOrEmpty(root))
                {
                    log.Write("PDM work folder: the local application data folder is unavailable; pdm.file.add accepts approved vault folders only.");
                    return null;
                }

                var path = Path.GetFullPath(Path.Combine(root, WorkFolderRelative));
                Directory.CreateDirectory(path);
                return path;
            }
            catch (Exception exception)
            {
                log.Write("PDM work folder could not be created; pdm.file.add accepts approved vault folders only. " + exception.Message);
                return null;
            }
        }

        /// <summary>The PDM gateway, or null when it cannot be created: pdm.* then answers pdm_unavailable. Never throws.</summary>
        private static PdmGateway? CreatePdmGateway(HostLog log)
        {
            try
            {
                return new PdmGateway(log.Write);
            }
            catch (Exception exception)
            {
                log.WriteException(exception);
                return null;
            }
        }

        private static void StopInstances(ISolidWorksInstanceManager manager, HostLog log)
        {
            try
            {
                using (var cancellation = new CancellationTokenSource(ManagerShutdownTimeout))
                {
                    var pending = manager.ShutdownAsync(cancellation.Token);
                    if (!pending.Wait(ManagerShutdownTimeout))
                    {
                        log.Write("serve: closing the owned SolidWorks instances took longer than 15 s.");
                    }
                }
            }
            catch (Exception exception)
            {
                log.WriteException(exception);
            }
        }
    }
}
