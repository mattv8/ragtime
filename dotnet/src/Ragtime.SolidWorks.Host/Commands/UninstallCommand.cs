using System.Diagnostics;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// <c>uninstall</c>: stop, then delete the logon task (absent is fine). Deletes no files. A failed stop leaves the
    /// task in place so nothing is half-removed.
    /// </summary>
    internal static class UninstallCommand
    {
        public static int Run(HostEnvironment env)
        {
            var watch = Stopwatch.StartNew();
            var stopFailure = StopCommand.Stop(env);
            if (stopFailure != null)
            {
                return ClientOutcome.Print(stopFailure, ExitCodes.Failure);
            }

            var schtasks = new Schtasks(env.Log);
            if (schtasks.Exists(env.TaskName, Schtasks.FixedLimit))
            {
                var deleted = schtasks.Delete(env.TaskName, Schtasks.FixedLimit);
                if (!deleted.Succeeded)
                {
                    return ClientOutcome.Print(
                        ClientOutcome.Failure(env, null, null, ErrorCodes.InternalError, "Could not delete the logon task: " + deleted.Summary, watch.ElapsedMilliseconds),
                        ExitCodes.Failure);
                }
            }

            return ClientOutcome.Print(new JObject { ["uninstalled"] = WireJson.Value(true) }, ExitCodes.Success);
        }
    }
}
