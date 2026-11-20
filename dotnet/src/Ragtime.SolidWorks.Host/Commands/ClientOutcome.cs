using System;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>Shared output and failure helpers for the client-side commands.</summary>
    internal static class ClientOutcome
    {
        /// <summary>Prints one JSON line and returns the exit code.</summary>
        public static int Print(JObject line, int exitCode)
        {
            JsonOutput.WriteLine(ResponseEnvelope.ToLine(line));
            return exitCode;
        }

        /// <summary>
        /// host_not_running envelope with the best-effort interactive-session fact. Pass null for taskRegistered when the
        /// task was not queried (no time left); details.task_registered is then JSON null.
        /// </summary>
        public static JObject NotRunning(HostEnvironment env, string? id, string? operation, bool? taskRegistered, long elapsedMs)
        {
            var envelope = ClientEnvelopes.HostNotRunning(
                id,
                operation,
                env.HostVersion,
                taskRegistered ?? false,
                UserSessionProbe.HasSessionOtherThanZero(),
                elapsedMs);
            if (taskRegistered == null)
            {
                ((JObject)envelope["error"]!["details"]!)["task_registered"] = JValue.CreateNull();
            }

            return envelope;
        }

        /// <summary>
        /// host_not_running with details.pipe_owner_mismatch true: the pipe answers but belongs to another account. Nothing
        /// was sent to it and no host was started.
        /// </summary>
        public static JObject OwnerMismatch(HostEnvironment env, string? id, string? operation, long elapsedMs) =>
            ResponseEnvelope.Failure(
                id,
                operation,
                ErrorCodes.HostNotRunning,
                "The host pipe is owned by another account, so it was not used and no host was started. Ask that account to stop its host.",
                new JObject { ["pipe_owner_mismatch"] = WireJson.Value(true) },
                HostInfo.ForHost(env.HostVersion),
                elapsedMs);

        /// <summary>Client-side error envelope.</summary>
        public static JObject Failure(HostEnvironment env, string? id, string? operation, string code, string message, long elapsedMs) =>
            ClientEnvelopes.Failure(id, operation, env.HostVersion, code, message, elapsedMs);

        /// <summary>
        /// True when the line is a complete host response for this request: it parses as a JSON object with a boolean
        /// ok and the matching id. Callers check this before printing a host response.
        /// </summary>
        public static bool IsCompleteResponse(string? line, string id)
        {
            if (line == null)
            {
                return false;
            }

            try
            {
                var root = JObject.Parse(line);
                var ok = root["ok"];
                var responseId = root["id"];
                return ok != null
                    && ok.Type == JTokenType.Boolean
                    && responseId != null
                    && responseId.Type == JTokenType.String
                    && string.Equals((string?)responseId, id, StringComparison.Ordinal);
            }
            catch (JsonException)
            {
                return false;
            }
        }

        /// <summary>True when a response line parses and has ok true.</summary>
        public static bool IsOk(string line)
        {
            try
            {
                return (bool?)JObject.Parse(line)["ok"] == true;
            }
            catch (JsonException)
            {
                return false;
            }
        }
    }
}
