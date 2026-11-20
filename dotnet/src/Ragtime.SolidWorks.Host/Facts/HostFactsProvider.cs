using System.Diagnostics;
using System.Security.Principal;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>Facts for <c>status</c>: account, SID, administrator role, session and uptime since serve started.</summary>
    internal sealed class HostFactsProvider : IHostFactsProvider
    {
        private readonly Stopwatch _uptime = Stopwatch.StartNew();

        /// <inheritdoc />
        public HostFacts GetHostFacts()
        {
            using (var identity = WindowsIdentity.GetCurrent())
            {
                var sessionId = Process.GetCurrentProcess().SessionId;
                var isAdministrator = new WindowsPrincipal(identity).IsInRole(WindowsBuiltInRole.Administrator);
                var interactive = System.Environment.UserInteractive && sessionId != 0;
                return new HostFacts(
                    identity.Name,
                    identity.User.Value,
                    isAdministrator,
                    sessionId,
                    interactive,
                    _uptime.Elapsed.TotalSeconds);
            }
        }
    }
}
