using System;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Text;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// <c>install</c>: creates or replaces the logon task, starts it, waits for the pipe and prints the status response.
    /// Exit 0 only when the host answered status with ok true. Runs only from the canonical install folder, and changes
    /// nothing when the pipe name is owned by another account.
    /// </summary>
    internal static class InstallCommand
    {
        private const string IncompleteResponseMessage = "The SolidWorks agent closed the connection before answering completely.";
        private static readonly TimeSpan PipeWait = TimeSpan.FromSeconds(20);
        private static readonly TimeSpan StatusTimeout = TimeSpan.FromSeconds(20);

        public static int Run(HostEnvironment env)
        {
            var watch = Stopwatch.StartNew();
            var log = env.Log;
            var id = Guid.NewGuid().ToString();

            if (!env.RunsFromCanonicalInstall)
            {
                log.Write("install refused: this copy runs from " + env.ExePath + "; the task must point at the install folder.");
                return ClientOutcome.Print(
                    ClientOutcome.Failure(
                        env,
                        id,
                        "status",
                        ErrorCodes.InternalError,
                        "Install only from the install folder " + HostEnvironment.CanonicalInstallDirectory + ".",
                        watch.ElapsedMilliseconds),
                    ExitCodes.Failure);
            }

            try
            {
                return Install(env, id, watch);
            }
            catch (PipeOwnerMismatchException)
            {
                log.Write("install refused: the pipe is owned by another account; the task was not changed.");
                return ClientOutcome.Print(ClientOutcome.OwnerMismatch(env, id, "status", watch.ElapsedMilliseconds), ExitCodes.Failure);
            }
        }

        private static int Install(HostEnvironment env, string id, Stopwatch watch)
        {
            var log = env.Log;

            // Refuse before the task is touched when another account owns the pipe name.
            PipeClient.TryConnect(env.PipeName, env.Sid, PipeClient.ProbeTimeout)?.Dispose();

            var schtasks = new Schtasks(log);
            var created = CreateTask(env, schtasks);
            if (!created.Succeeded)
            {
                return ClientOutcome.Print(
                    ClientOutcome.Failure(env, id, "status", ErrorCodes.InternalError, "Could not create the logon task: " + created.Summary, watch.ElapsedMilliseconds),
                    ExitCodes.Failure);
            }

            schtasks.RunNow(env.TaskName, Schtasks.FixedLimit);
            var pipe = PipeClient.WaitForPipe(env.PipeName, env.Sid, PipeWait);
            if (pipe == null)
            {
                log.Write("install: the task was created but the host pipe did not appear within 20 s.");
                return ClientOutcome.Print(
                    ClientOutcome.NotRunning(env, id, "status", true, watch.ElapsedMilliseconds),
                    ExitCodes.Failure);
            }

            using (pipe)
            {
                return ReadStatus(pipe, env, id, watch);
            }
        }

        private static int ReadStatus(NamedPipeClientStream pipe, HostEnvironment env, string id, Stopwatch watch)
        {
            string? response;
            try
            {
                response = PipeClient.ExchangeAsync(pipe, ClientRequests.StatusLine(id), StatusTimeout).GetAwaiter().GetResult();
            }
            catch (TimeoutException)
            {
                return ClientOutcome.Print(
                    ClientOutcome.Failure(env, id, "status", ErrorCodes.OperationTimeout, "The SolidWorks agent did not answer the status request in time.", watch.ElapsedMilliseconds),
                    ExitCodes.Failure);
            }
            catch (Exception exception)
            {
                env.Log.WriteException(exception);
                response = null;
            }

            if (response == null || !ClientOutcome.IsCompleteResponse(response, id))
            {
                env.Log.Write("install: the status response was missing, incomplete or for another request.");
                return ClientOutcome.Print(
                    ClientOutcome.Failure(env, id, "status", ErrorCodes.InternalError, IncompleteResponseMessage, watch.ElapsedMilliseconds),
                    ExitCodes.Failure);
            }

            JsonOutput.WriteLine(response);
            return ClientOutcome.IsOk(response) ? ExitCodes.Success : ExitCodes.Failure;
        }

        private static SchtasksResult CreateTask(HostEnvironment env, Schtasks schtasks)
        {
            var xmlPath = Path.Combine(Path.GetTempPath(), "ragtime-sw-task-" + Guid.NewGuid().ToString("N") + ".xml");
            try
            {
                var xml = TaskXml.Build(env.AccountName, env.Sid, HostEnvironment.CanonicalExePath, HostEnvironment.CanonicalInstallDirectory);
                File.WriteAllText(xmlPath, xml, Encoding.Unicode);
                return schtasks.Create(env.TaskName, xmlPath, Schtasks.FixedLimit);
            }
            finally
            {
                DeleteQuietly(xmlPath);
            }
        }

        private static void DeleteQuietly(string path)
        {
            try
            {
                File.Delete(path);
            }
            catch (Exception)
            {
                // Temp file; the system cleans it up eventually.
            }
        }
    }
}
