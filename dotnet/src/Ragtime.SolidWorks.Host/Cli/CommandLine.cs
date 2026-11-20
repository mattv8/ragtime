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
    }

    /// <summary>Exit codes: 0 success, 2 failure (usage errors and client-side errors included).</summary>
    internal static class ExitCodes
    {
        public const int Success = 0;
        public const int Failure = 2;
    }

    /// <summary>The parsed command line: one command and, for <c>call</c>, an optional base64 request.</summary>
    internal sealed class CommandLine
    {
        private const string RequestOption = "--request-b64";

        /// <summary>Usage text printed to stderr for unknown or malformed command lines.</summary>
        public const string Usage =
            "Usage: ragtime-solidworks-host <command> [options]\n" +
            "  serve                          run the pipe server in the current desktop session\n" +
            "  call --request-b64 <base64>    forward one request line to the host (stdin when the option is absent)\n" +
            "  install                        create the logon task, start it and print the status response\n" +
            "  stop                           ask a running host to exit\n" +
            "  uninstall                      stop the host and delete the logon task (files are kept)\n" +
            "  version                        print the host version and protocol";

        private CommandLine(string name, string? requestBase64)
        {
            Name = name;
            RequestBase64 = requestBase64;
        }

        /// <summary>Command name, one of <see cref="CommandNames"/>.</summary>
        public string Name { get; }

        /// <summary>Value of --request-b64, or null when the option is absent.</summary>
        public string? RequestBase64 { get; }

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
            if (name == CommandNames.Call)
            {
                return ParseCall(args);
            }

            if (!IsKnownWithoutOptions(name) || args.Length != 1)
            {
                return null;
            }

            return new CommandLine(name, null);
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
            || name == CommandNames.Stop
            || name == CommandNames.Uninstall
            || name == CommandNames.Version;
    }
}
