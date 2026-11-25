using System;
using System.IO;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Shared names and JSON file operations for script workers.</summary>
    public static class ScriptWorkerFiles
    {
        public const string RequestName = "request.json";
        public const string SourceName = "script.cs";
        public const string ResultName = "result.json";
        public const string LogName = "log.jsonl";

        public static JObject ReadRequest(string folder) => Read(folder, RequestName);
        public static JObject ReadResult(string folder) => Read(folder, ResultName);

        /// <summary>
        /// Reads <c>log.jsonl</c> as UTF-8 while a worker may still hold it open for writing. Returns null when the file is
        /// missing. Throws the usual IO exceptions when the file cannot be opened.
        /// </summary>
        public static string? ReadLogText(string folder)
        {
            var path = Path.Combine(folder, LogName);
            if (!File.Exists(path)) return null;
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
            using (var reader = new StreamReader(stream, Encoding.UTF8))
            {
                return reader.ReadToEnd();
            }
        }

        public static void WriteRequest(string folder, JObject request) => Write(folder, RequestName, request);
        public static void WriteResult(string folder, JObject result)
        {
            var target = Path.Combine(folder, ResultName);
            var temp = Path.Combine(folder, "." + Guid.NewGuid().ToString("N") + ".tmp");
            try
            {
                using (var file = new FileStream(temp, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                {
                    try
                    {
                        WriteBoundedResult(file, result);
                    }
                    catch (ResultTooLargeException)
                    {
                        file.SetLength(0);
                        file.Position = 0;
                        var failure = new JObject
                        {
                            ["status"] = "failed",
                            ["message"] = "The script result was too large.",
                        };
                        if (result["exception_type"]?.Type == JTokenType.String)
                            failure["exception_type"] = Truncate((string)result["exception_type"]!, 2048);
                        if (result["stack"]?.Type == JTokenType.String)
                            failure["stack"] = Truncate((string)result["stack"]!, 8192);
                        WriteBoundedResult(file, failure);
                    }
                }
                File.Move(temp, target);
            }
            finally
            {
                if (File.Exists(temp)) File.Delete(temp);
            }
        }

        private static void WriteBoundedResult(Stream stream, JObject result)
        {
            using (var limited = new ResultStream(stream))
            using (var text = new StreamWriter(limited, new UTF8Encoding(false), 1024, true))
            using (var json = new JsonTextWriter(text))
            {
                json.WriteStartObject();
                foreach (var property in result.Properties())
                {
                    json.WritePropertyName(property.Name);
                    var limit = property.Name == "stack" ? 8192
                        : property.Name == "message" || property.Name == "exception_type" ? 2048 : 0;
                    if (limit != 0 && property.Value.Type == JTokenType.String)
                        json.WriteValue(Truncate((string)property.Value!, limit));
                    else
                        property.Value.WriteTo(json);
                }
                json.WriteEndObject();
                json.Flush();
            }
        }

        private static string Truncate(string value, int maxBytes)
        {
            if (Encoding.UTF8.GetByteCount(value) <= maxBytes) return value;
            var low = 0;
            var high = Math.Min(value.Length, maxBytes);
            while (low < high)
            {
                var middle = low + (high - low + 1) / 2;
                if (Encoding.UTF8.GetByteCount(value.Substring(0, middle)) <= maxBytes) low = middle;
                else high = middle - 1;
            }
            if (low > 0 && char.IsHighSurrogate(value[low - 1])) low--;
            return value.Substring(0, low);
        }

        private sealed class ResultTooLargeException : IOException { }

        private sealed class ResultStream : Stream
        {
            private readonly Stream _inner;
            private long _written;
            public ResultStream(Stream inner) => _inner = inner;
            public override bool CanRead => false;
            public override bool CanSeek => false;
            public override bool CanWrite => true;
            public override long Length => throw new NotSupportedException();
            public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
            public override void Flush() => _inner.Flush();
            public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();
            public override void Write(byte[] buffer, int offset, int count)
            {
                if (count > 1048576 - _written) throw new ResultTooLargeException();
                _inner.Write(buffer, offset, count);
                _written += count;
            }
        }

        private static JObject Read(string folder, string name) =>
            JObject.Parse(File.ReadAllText(Path.Combine(folder, name), Encoding.UTF8));

        private static void Write(string folder, string name, JObject value)
        {
            var target = Path.Combine(folder, name);
            var temp = Path.Combine(folder, "." + Guid.NewGuid().ToString("N") + ".tmp");
            try
            {
                File.WriteAllText(temp, value.ToString(Formatting.None), new UTF8Encoding(false));
                File.Move(temp, target);
            }
            finally
            {
                if (File.Exists(temp))
                {
                    File.Delete(temp);
                }
            }
        }
    }
}
