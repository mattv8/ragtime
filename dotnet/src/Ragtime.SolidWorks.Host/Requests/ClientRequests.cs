using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>Requests the host command itself sends over the pipe (protocol 1, background context).</summary>
    internal static class ClientRequests
    {
        private const int DefaultIdleShutdownSeconds = 900;
        private const int StatusDeadlineSeconds = 15;
        private const int ShutdownDeadlineSeconds = 10;

        /// <summary>The status request used by install.</summary>
        public static string StatusLine(string id) => Line(Build(id, "status", StatusDeadlineSeconds));

        /// <summary>
        /// The internal host.shutdown request used by stop and uninstall. With <paramref name="ifIdleSeconds"/> the host
        /// stops only when it has been idle that long (<c>if_idle_seconds</c>); without it the stop is unconditional.
        /// </summary>
        public static string ShutdownLine(string id, int? ifIdleSeconds = null)
        {
            var request = Build(id, "host.shutdown", ShutdownDeadlineSeconds);
            if (ifIdleSeconds is int seconds)
            {
                ((JObject)request["arguments"]!)["if_idle_seconds"] = WireJson.Value(seconds);
            }

            return Line(request);
        }

        private static string Line(JObject request) => ResponseEnvelope.ToLine(request);

        private static JObject Build(string id, string operation, int deadlineSeconds) =>
            new JObject
            {
                ["protocol"] = WireJson.Value(ProtocolInfo.Version),
                ["id"] = WireJson.Value(id),
                ["operation"] = WireJson.Value(operation),
                ["arguments"] = new JObject(),
                ["context"] = new JObject
                {
                    ["solidworks_version"] = JValue.CreateNull(),
                    ["instance_mode"] = WireJson.Value(ProtocolInfo.InstanceModes.Background),
                    ["max_instances"] = WireJson.Value(ProtocolInfo.MinMaxInstances),
                    ["idle_shutdown_seconds"] = WireJson.Value(DefaultIdleShutdownSeconds),
                    ["allow_write"] = WireJson.Value(false),
                    ["deadline_seconds"] = WireJson.Value(deadlineSeconds),
                    ["max_result_bytes"] = WireJson.Value(ProtocolInfo.MaxResultBytes),
                },
            };
    }
}
