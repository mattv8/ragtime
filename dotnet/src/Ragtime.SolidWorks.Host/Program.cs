using System;
using System.Text;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// Entry point. Parses the command line, runs one command and maps any escaped exception to an error
    /// envelope (JSON commands) or a stderr line, with exit code 2.
    /// </summary>
    internal static class Program
    {
        private static int Main(string[] args)
        {
            ConfigureConsole();

            var command = CommandLine.Parse(args);
            if (command == null)
            {
                Console.Error.WriteLine(CommandLine.Usage);
                return ExitCodes.Failure;
            }

            try
            {
                return Dispatch(command);
            }
            catch (Exception exception)
            {
                return ReportUnexpected(command, exception);
            }
        }

        private static int Dispatch(CommandLine command)
        {
            switch (command.Name)
            {
                case CommandNames.Version:
                    return VersionCommand.Run();
                case CommandNames.Serve:
                    return ServeCommand.Run(HostEnvironment.Create(HostLog.ForCurrentUser()));
                case CommandNames.Call:
                    return CallCommand.Run(command, HostEnvironment.Create(HostLog.ForCurrentUser()));
                case CommandNames.Install:
                    return InstallCommand.Run(HostEnvironment.Create(HostLog.ForCurrentUser()));
                case CommandNames.Stop:
                    return StopCommand.Run(HostEnvironment.Create(HostLog.ForCurrentUser()));
                case CommandNames.Uninstall:
                    return UninstallCommand.Run(HostEnvironment.Create(HostLog.ForCurrentUser()));
                default:
                    return ExitCodes.Failure;
            }
        }

        private static int ReportUnexpected(CommandLine command, Exception exception)
        {
            HostLog.ForCurrentUser().WriteException(exception);
            if (command.WritesJson)
            {
                JsonOutput.WriteLine(ResponseEnvelope.ToLine(ClientEnvelopes.Failure(
                    null,
                    null,
                    HostEnvironment.ReadHostVersion(),
                    ErrorCodes.InternalError,
                    "The host tool failed unexpectedly. See host.log for details.",
                    0)));
            }

            Console.Error.WriteLine("ragtime-solidworks-host: unexpected error; see host.log.");
            return ExitCodes.Failure;
        }

        private static void ConfigureConsole()
        {
            try
            {
                Console.OutputEncoding = new UTF8Encoding(false);
            }
            catch (Exception)
            {
                // No console attached (scheduled task or redirected handles). Output is written as raw bytes anyway.
            }
        }
    }
}
