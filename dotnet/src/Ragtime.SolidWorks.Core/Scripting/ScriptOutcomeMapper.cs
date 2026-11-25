using System;
using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Maps a worker's <c>result.json</c> and <c>log.jsonl</c> to a <see cref="ScriptRunOutcome"/>.</summary>
    public static class ScriptOutcomeMapper
    {
        /// <summary>Reads <c>log.jsonl</c>: one JSON string per line. Blank or malformed lines are skipped.</summary>
        public static List<string> ReadLog(string? text)
        {
            var lines = new List<string>();
            if (text == null || text.Length == 0)
            {
                return lines;
            }

            foreach (var raw in text.Split('\n'))
            {
                var line = raw.Trim();
                if (line.Length == 0)
                {
                    continue;
                }

                try
                {
                    var value = JsonConvert.DeserializeObject<string>(line);
                    if (value != null)
                    {
                        lines.Add(value);
                    }
                }
                catch (JsonException)
                {
                    // A line the worker did not write whole; the rest of the log is still useful.
                }
            }

            return lines;
        }

        /// <summary>
        /// Maps a parsed <c>result.json</c>. Every field is type-checked: a wrong type, an unknown status, or a completed
        /// result without a "result" field throws <see cref="ScriptResultFormatException"/>. The caller reports that with
        /// <see cref="Malformed"/>.
        /// </summary>
        public static ScriptRunOutcome FromResult(JObject result, IEnumerable<string> log)
        {
            var status = StatusOf(RequiredString(result, "status"));
            var outcome = new ScriptRunOutcome
            {
                Status = status,
                Log = new List<string>(log),
                LogTruncated = OptionalBool(result, "log_truncated"),
                ExceptionType = OptionalString(result, "exception_type"),
                Message = OptionalString(result, "message"),
                Stack = OptionalString(result, "stack"),
                CompileMilliseconds = OptionalLong(result, "compile_ms"),
                RunMilliseconds = OptionalLong(result, "run_ms"),
            };

            if (status == ScriptRunStatus.Completed)
            {
                outcome.Result = CompletedValue(result);
            }

            outcome.CompileErrors.AddRange(CompileErrors(result));
            return outcome;
        }

        /// <summary>The worker exited without a usable <c>result.json</c>.</summary>
        public static ScriptRunOutcome NoResult(int exitCode, IEnumerable<string> log) => new ScriptRunOutcome
        {
            Status = ScriptRunStatus.Failed,
            ExitCode = exitCode,
            Log = new List<string>(log),
            Message = "The script worker exited without a result (exit code " + exitCode + ").",
        };

        /// <summary><c>result.json</c> exists but cannot be read or has wrong fields.</summary>
        public static ScriptRunOutcome Malformed(int exitCode, string reason, IEnumerable<string> log) => new ScriptRunOutcome
        {
            Status = ScriptRunStatus.Failed,
            ExitCode = exitCode,
            Log = new List<string>(log),
            Message = "The script worker wrote an unreadable result: " + reason,
        };

        /// <summary>The timeout passed (or the call was cancelled) and the worker was killed.</summary>
        public static ScriptRunOutcome TimedOut(IEnumerable<string> log) => new ScriptRunOutcome
        {
            Status = ScriptRunStatus.TimedOut,
            Log = new List<string>(log),
            Message = "The script timed out and its worker was stopped.",
        };

        private static ScriptRunStatus StatusOf(string status)
        {
            switch (status)
            {
                case "completed": return ScriptRunStatus.Completed;
                case "compile_failed": return ScriptRunStatus.CompileFailed;
                case "compile_unavailable": return ScriptRunStatus.CompileUnavailable;
                case "invalid": return ScriptRunStatus.Invalid;
                case "failed": return ScriptRunStatus.Failed;
                case "attach_failed": return ScriptRunStatus.AttachFailed;
                default: throw new ScriptResultFormatException("\"status\" has an unknown value \"" + status + "\".");
            }
        }

        private static JToken? CompletedValue(JObject result)
        {
            if (!result.ContainsKey("result"))
            {
                throw new ScriptResultFormatException("A completed result has no \"result\" field.");
            }

            var value = result["result"];
            return value == null || value.Type == JTokenType.Null ? null : value;
        }

        private static List<ScriptCompileError> CompileErrors(JObject result)
        {
            var errors = new List<ScriptCompileError>();
            var token = result["compile_errors"];
            if (token == null || token.Type == JTokenType.Null)
            {
                return errors;
            }

            if (!(token is JArray items))
            {
                throw new ScriptResultFormatException("\"compile_errors\" must be an array.");
            }

            foreach (var item in items)
            {
                if (!(item is JObject entry))
                {
                    throw new ScriptResultFormatException("Each compile error must be an object.");
                }

                errors.Add(new ScriptCompileError
                {
                    Line = OptionalInt(entry, "line"),
                    Column = OptionalInt(entry, "column"),
                    Code = OptionalString(entry, "code") ?? string.Empty,
                    Message = OptionalString(entry, "message") ?? string.Empty,
                    Hint = OptionalString(entry, "hint"),
                });
            }

            return errors;
        }

        private static string RequiredString(JObject obj, string name) =>
            OptionalString(obj, name) ?? throw Wrong(name, "a string");

        private static string? OptionalString(JObject obj, string name)
        {
            var token = obj[name];
            if (token == null || token.Type == JTokenType.Null)
            {
                return null;
            }

            if (token.Type != JTokenType.String)
            {
                throw Wrong(name, "a string");
            }

            return (string?)token;
        }

        private static bool OptionalBool(JObject obj, string name)
        {
            var token = obj[name];
            if (token == null || token.Type == JTokenType.Null)
            {
                return false;
            }

            if (token.Type != JTokenType.Boolean)
            {
                throw Wrong(name, "a boolean");
            }

            return (bool)token;
        }

        private static long OptionalLong(JObject obj, string name)
        {
            var token = obj[name];
            if (token == null || token.Type == JTokenType.Null)
            {
                return 0;
            }

            if (token.Type != JTokenType.Integer)
            {
                throw Wrong(name, "an integer");
            }

            try
            {
                return (long)token;
            }
            catch (OverflowException)
            {
                // Newtonsoft keeps integers beyond Int64 as BigInteger; the explicit conversion then overflows.
                throw Wrong(name, "an integer in the 64-bit range");
            }
        }

        private static int OptionalInt(JObject obj, string name)
        {
            var value = OptionalLong(obj, name);
            if (value < int.MinValue || value > int.MaxValue)
            {
                throw Wrong(name, "a 32-bit integer");
            }

            return (int)value;
        }

        private static ScriptResultFormatException Wrong(string name, string kind) =>
            new ScriptResultFormatException("\"" + name + "\" must be " + kind + ".");
    }
}
