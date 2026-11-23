using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>Command names of the host executable (contract section 3).</summary>
    internal static class CommandNames
    {
        public const string Serve = "serve";
        public const string Call = "call";
        public const string Install = "install";
        public const string Stop = "stop";
        public const string Uninstall = "uninstall";
        public const string Version = "version";
        public const string ScriptWorker = "script-worker";
    }

    /// <summary>
    /// Exit codes: 0 success, 2 failure (usage errors and client-side errors included), 3 busy (<c>stop --if-idle</c>
    /// only: the host refused to stop because it is in use and is still running).
    /// </summary>
    internal static class ExitCodes
    {
        public const int Success = 0;
        public const int Failure = 2;

        /// <summary><c>stop --if-idle</c>: the host is in use and was left running.</summary>
        public const int Busy = 3;
    }

    /// <summary>
    /// The parsed command line: one command and, for <c>call</c>, an optional base64 request; for <c>stop</c>, an
    /// optional idle condition.
    /// </summary>
    internal sealed class CommandLine
    {
        private const string RequestOption = "--request-b64";
        private const string IfIdleOption = "--if-idle";

        /// <summary>Usage text printed to stderr for unknown or malformed command lines.</summary>
        public const string Usage =
            "Usage: ragtime-solidworks-host <command> [options]\n" +
            "  serve                          run the pipe server in the current desktop session\n" +
            "  call --request-b64 <base64>    forward one request line to the host (stdin when the option is absent)\n" +
            "  install                        create the logon task, start it and print the status response\n" +
            "  stop [--if-idle <seconds>]     ask a running host to exit; with --if-idle only when it is idle\n" +
            "  uninstall                      stop the host and delete the logon task (files are kept)\n" +
            "  version                        print the host version and protocol";

        private CommandLine(string name, string? requestBase64, int? ifIdleSeconds = null)
        {
            Name = name;
            RequestBase64 = requestBase64;
            IfIdleSeconds = ifIdleSeconds;
        }

        /// <summary>Command name, one of <see cref="CommandNames"/>.</summary>
        public string Name { get; }

        /// <summary>Value of --request-b64, or null when the option is absent.</summary>
        public string? RequestBase64 { get; }

        /// <summary>Value of <c>stop --if-idle</c> (0 to 86400 seconds), or null when the option is absent.</summary>
        public int? IfIdleSeconds { get; }

        /// <summary>True for commands whose stdout carries exactly one JSON line (everything except serve).</summary>
        public bool WritesJson => Name != CommandNames.Serve;

        /// <summary>Parses the arguments; returns null for an unknown command or an invalid option.</summary>
        public static CommandLine? Parse(string[] args)
        {
            if (args == null || args.Length == 0)
            {
                return null;
            }

            var name = args[0];
            if (name == CommandNames.ScriptWorker)
            {
                return args.Length == 3 && args[1] == "--run-folder" && args[2].Length > 0
                    ? new CommandLine(name, args[2]) : null;
            }
            if (name == CommandNames.Call)
            {
                return ParseCall(args);
            }

            if (name == CommandNames.Stop)
            {
                return ParseStop(args);
            }

            if (!IsKnownWithoutOptions(name) || args.Length != 1)
            {
                return null;
            }

            return new CommandLine(name, null);
        }

        private static CommandLine? ParseStop(string[] args)
        {
            if (args.Length == 1)
            {
                return new CommandLine(CommandNames.Stop, null);
            }

            if (args.Length == 3 && args[1] == IfIdleOption)
            {
                var seconds = StopRequest.TryParseIfIdle(args[2]);
                return seconds == null ? null : new CommandLine(CommandNames.Stop, null, seconds);
            }

            return null;
        }

        private static CommandLine? ParseCall(string[] args)
        {
            string? requestBase64 = null;
            for (var index = 1; index < args.Length; index++)
            {
                if (args[index] == RequestOption && index + 1 < args.Length && requestBase64 == null)
                {
                    requestBase64 = args[index + 1];
                    index++;
                    continue;
                }

                return null;
            }

            return new CommandLine(CommandNames.Call, requestBase64);
        }

        private static bool IsKnownWithoutOptions(string name) =>
            name == CommandNames.Serve
            || name == CommandNames.Install
            || name == CommandNames.Uninstall
            || name == CommandNames.Version;
    }
}
