using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary><c>version</c>: prints {"host_version","protocol"} and exits 0.</summary>
    internal static class VersionCommand
    {
        public static int Run()
        {
            var line = new JObject
            {
                ["host_version"] = WireJson.Value(HostEnvironment.ReadHostVersion()),
                ["protocol"] = WireJson.Value(ProtocolInfo.Version),
            };
            return ClientOutcome.Print(line, ExitCodes.Success);
        }
    }
}
