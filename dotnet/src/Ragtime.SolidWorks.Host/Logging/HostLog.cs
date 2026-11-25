using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// Host log at %LOCALAPPDATA%\RagtimeSolidWorks\logs\host.log. Thread-safe, one physical line per message, rotated
    /// to host.log.1 at 2 MB. Every failure inside the logger is swallowed: logging must never break a command.
    /// Lines never carry request bodies or arguments. Control characters in every message are escaped, so a value
    /// from a request or another process cannot forge or split a line.
    /// </summary>
    internal sealed class HostLog
    {
        private const long MaxBytes = 2L * 1024 * 1024;
        private const int ExceptionFrames = 3;
        private static readonly object Gate = new object();
        private static readonly UTF8Encoding Utf8 = new UTF8Encoding(false);
        private static readonly int ProcessId = Process.GetCurrentProcess().Id;
        private readonly string? _path;

        /// <summary>Creates a logger for the given file; null disables logging.</summary>
        public HostLog(string? path)
        {
            _path = path;
        }

        /// <summary>The logger for the current user's log file. Never throws.</summary>
        public static HostLog ForCurrentUser()
        {
            try
            {
                var root = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                return new HostLog(string.IsNullOrEmpty(root) ? null : Path.Combine(root, HostNames.LogRelative));
            }
            catch (Exception)
            {
                return new HostLog(null);
            }
        }

        /// <summary>Appends one timestamped line. Control characters in the message are escaped.</summary>
        public void Write(string message)
        {
            var path = _path;
            if (path == null)
            {
                return;
            }

            try
            {
                var line = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff") + " [" + ProcessId + "] " + LogText.Escape(message) + "\r\n";
                var bytes = Utf8.GetBytes(line);
                lock (Gate)
                {
                    Append(path, bytes);
                }
            }
            catch (Exception)
            {
                // Logging failures are never reported to the caller.
            }
        }

        /// <summary>Appends the type, message and the first stack frames of an exception on one line.</summary>
        public void WriteException(Exception exception)
        {
            var frames = (exception.StackTrace ?? string.Empty)
                .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(frame => frame.Trim())
                .Where(frame => frame.Length > 0)
                .Take(ExceptionFrames)
                .ToArray();
            var text = "exception " + exception.GetType().FullName + ": " + exception.Message;
            if (frames.Length > 0)
            {
                text += " at " + string.Join(" | ", frames);
            }

            Write(text);
        }

        private static void Append(string path, byte[] bytes)
        {
            var directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            RotateIfLarge(path);
            using (var stream = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.ReadWrite | FileShare.Delete))
            {
                stream.Write(bytes, 0, bytes.Length);
            }
        }

        private static void RotateIfLarge(string path)
        {
            var info = new FileInfo(path);
            if (!info.Exists || info.Length <= MaxBytes)
            {
                return;
            }

            var backup = path + ".1";
            if (File.Exists(backup))
            {
                File.Delete(backup);
            }

            File.Move(path, backup);
        }
    }

    /// <summary>Escapes text so that it is printed as one physical line.</summary>
    internal static class LogText
    {
        /// <summary>Replaces CR, LF and TAB with \r, \n and \t, and other control characters with \uXXXX.</summary>
        public static string Escape(string? value)
        {
            if (value == null || value.Length == 0)
            {
                return string.Empty;
            }

            var builder = new StringBuilder(value.Length);
            foreach (var character in value)
            {
                switch (character)
                {
                    case '\r':
                        builder.Append("\\r");
                        break;
                    case '\n':
                        builder.Append("\\n");
                        break;
                    case '\t':
                        builder.Append("\\t");
                        break;
                    default:
                        if (character < ' ' || character == '\u007F')
                        {
                            builder.Append("\\u").Append(((int)character).ToString("X4"));
                        }
                        else
                        {
                            builder.Append(character);
                        }

                        break;
                }
            }

            return builder.ToString();
        }
    }
}
