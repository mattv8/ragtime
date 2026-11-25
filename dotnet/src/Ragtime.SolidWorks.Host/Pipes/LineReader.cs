using System;
using System.IO;
using System.Threading.Tasks;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>Outcome of reading one line from a pipe.</summary>
    internal enum LineReadKind
    {
        /// <summary>A line was read; <see cref="LineReadResult.Bytes"/> holds its content without the LF.</summary>
        Line,

        /// <summary>The content is longer than the limit. Nothing beyond the limit was read.</summary>
        TooLong,

        /// <summary>The peer closed the pipe before sending anything.</summary>
        Closed,
    }

    /// <summary>Result of <see cref="LineReader.ReadAsync"/>.</summary>
    internal sealed class LineReadResult
    {
        public LineReadResult(LineReadKind kind, byte[] bytes, bool terminated)
        {
            Kind = kind;
            Bytes = bytes;
            Terminated = terminated;
        }

        public LineReadKind Kind { get; }

        public byte[] Bytes { get; }

        /// <summary>True when the line ended with LF; false when the peer closed the pipe in the middle of it.</summary>
        public bool Terminated { get; }
    }

    /// <summary>
    /// Reads one LF-terminated line without ever consuming more than <c>limit + 1</c> bytes, so a client that never
    /// sends a newline cannot make the host buffer without bound.
    /// </summary>
    internal static class LineReader
    {
        private const byte NewLine = (byte)'\n';

        /// <summary>Reads at most one line whose content is at most <paramref name="maxContentBytes"/> bytes.</summary>
        public static async Task<LineReadResult> ReadAsync(Stream stream, int maxContentBytes)
        {
            var limit = maxContentBytes + 1;
            var buffer = new byte[4096];
            using (var content = new MemoryStream())
            {
                while (content.Length < limit)
                {
                    var wanted = Math.Min(buffer.Length, limit - (int)content.Length);
                    var read = await stream.ReadAsync(buffer, 0, wanted).ConfigureAwait(false);
                    if (read == 0)
                    {
                        return Classify(content, maxContentBytes, terminated: false);
                    }

                    var newline = Array.IndexOf(buffer, NewLine, 0, read);
                    if (newline >= 0)
                    {
                        content.Write(buffer, 0, newline);
                        return Classify(content, maxContentBytes, terminated: true);
                    }

                    content.Write(buffer, 0, read);
                }

                return Classify(content, maxContentBytes, terminated: false);
            }
        }

        private static LineReadResult Classify(MemoryStream content, int maxContentBytes, bool terminated)
        {
            if (content.Length > maxContentBytes)
            {
                return new LineReadResult(LineReadKind.TooLong, Array.Empty<byte>(), terminated: false);
            }

            if (!terminated && content.Length == 0)
            {
                return new LineReadResult(LineReadKind.Closed, Array.Empty<byte>(), terminated: false);
            }

            return new LineReadResult(LineReadKind.Line, content.ToArray(), terminated);
        }
    }
}
