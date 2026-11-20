using System;
using System.IO;
using System.Text;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>Writes the single JSON line of a JSON command to stdout: UTF-8 without BOM, terminated by LF.</summary>
    internal static class JsonOutput
    {
        public static void WriteLine(string line)
        {
            try
            {
                var bytes = new UTF8Encoding(false).GetBytes(line + "\n");
                var stdout = Console.OpenStandardOutput();
                stdout.Write(bytes, 0, bytes.Length);
                stdout.Flush();
            }
            catch (IOException)
            {
                // stdout closed by the caller; nothing else can be reported.
            }
        }
    }
}
