using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Helpers for the <c>stop --if-idle</c> command: argument parsing and reply interpretation.</summary>
    public static class StopRequest
    {
        /// <summary>Largest idle condition, in seconds, that <c>stop --if-idle</c> accepts (one day).</summary>
        public const int MaxIfIdleSeconds = 86400;

        private const string UnreadableAnswer = "The SolidWorks agent gave an unreadable answer.";
        private const string GenericError = "The SolidWorks agent reported an error.";

        /// <summary>
        /// Parses the N of <c>--if-idle N</c>. Accepts ASCII digits only (no sign, spaces or exponent) with a value
        /// from 0 to <see cref="MaxIfIdleSeconds"/>. Returns null for anything else.
        /// </summary>
        public static int? TryParseIfIdle(string? text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return null;
            }

            var value = 0;
            foreach (var ch in text!)
            {
                if (ch < '0' || ch > '9')
                {
                    return null;
                }

                value = value * 10 + (ch - '0');
                if (value > MaxIfIdleSeconds)
                {
                    return null;
                }
            }

            return value;
        }

        /// <summary>
        /// Interprets the single JSON reply line to a <c>host.shutdown</c> request with id <paramref name="expectedId"/>.
        /// Anything that is not a readable envelope for that id is a <see cref="StopReplyKind.Failed"/> reply, so the
        /// caller never stops without a confirmed answer.
        /// </summary>
        public static StopReply Interpret(string? replyLine, string expectedId)
        {
            var root = TryParseObject(replyLine);
            if (root == null || root["ok"]?.Type != JTokenType.Boolean || !IsId(root["id"], expectedId))
            {
                return StopReply.Failed(ErrorCodes.InternalError, UnreadableAnswer);
            }

            if ((bool)root["ok"]!)
            {
                return StopReply.Accepted();
            }

            if (!(root["error"] is JObject error))
            {
                return StopReply.Failed(ErrorCodes.InternalError, GenericError);
            }

            var code = ReadString(error["code"]);
            if (code == ErrorCodes.HostBusy)
            {
                return StopReply.Busy(ReadReasons(error["details"] as JObject));
            }

            return StopReply.Failed(code ?? ErrorCodes.InternalError, ReadString(error["message"]) ?? GenericError);
        }

        private static string? ReadString(JToken? token) =>
            token?.Type == JTokenType.String ? (string?)token : null;

        private static JObject? TryParseObject(string? line)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                return null;
            }

            try
            {
                return JToken.Parse(line!) as JObject;
            }
            catch (JsonReaderException)
            {
                return null;
            }
        }

        private static bool IsId(JToken? id, string expectedId) =>
            id?.Type == JTokenType.String && (string?)id == expectedId;

        private static IReadOnlyList<string> ReadReasons(JObject? details)
        {
            var reasons = new List<string>();
            if (details?["reasons"] is JArray items)
            {
                foreach (var item in items)
                {
                    if (item.Type == JTokenType.String)
                    {
                        reasons.Add((string)item!);
                    }
                }
            }

            return reasons;
        }
    }

    /// <summary>How a <c>host.shutdown</c> reply was read.</summary>
    public enum StopReplyKind
    {
        /// <summary>The host accepted the stop.</summary>
        Accepted,

        /// <summary>The host refused because it is in use (<c>host_busy</c>).</summary>
        Busy,

        /// <summary>Any other error, or an unreadable reply.</summary>
        Failed,
    }

    /// <summary>The interpreted reply to a stop request.</summary>
    public sealed class StopReply
    {
        private StopReply(StopReplyKind kind, IReadOnlyList<string> reasons, string? code, string? message)
        {
            Kind = kind;
            Reasons = reasons;
            Code = code;
            Message = message;
        }

        /// <summary>Whether the host accepted, refused as busy, or failed.</summary>
        public StopReplyKind Kind { get; }

        /// <summary>Busy reasons from the host; empty unless <see cref="Kind"/> is Busy.</summary>
        public IReadOnlyList<string> Reasons { get; }

        /// <summary>Error code when the reply failed; null otherwise.</summary>
        public string? Code { get; }

        /// <summary>Readable message when the reply failed; null otherwise.</summary>
        public string? Message { get; }

        /// <summary>An accepted stop.</summary>
        public static StopReply Accepted() => new StopReply(StopReplyKind.Accepted, new string[0], null, null);

        /// <summary>A refused stop with the host's reasons.</summary>
        public static StopReply Busy(IReadOnlyList<string> reasons) =>
            new StopReply(StopReplyKind.Busy, reasons, null, null);

        /// <summary>A failed stop with a contract code and message.</summary>
        public static StopReply Failed(string code, string message) =>
            new StopReply(StopReplyKind.Failed, new string[0], code, message);
    }
}
