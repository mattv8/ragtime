using System;
using System.Text;
using Newtonsoft.Json;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Limits a stream of script log lines.</summary>
    public sealed class ScriptLogBuffer
    {
        private readonly Action<string> _append;
        private readonly object _gate = new object();
        private int _lines;
        private int _bytes;

        public ScriptLogBuffer(Action<string> append) => _append = append ?? throw new ArgumentNullException(nameof(append));

        public bool Truncated { get; private set; }

        public void Log(string message)
        {
            lock (_gate)
            {
                if (Truncated)
                {
                    return;
                }
                var bytes = Encoding.UTF8.GetByteCount(JsonConvert.SerializeObject(message)) + Encoding.UTF8.GetByteCount(Environment.NewLine);
                if (_lines >= 2000 || bytes > 262144 - _bytes)
                {
                    Truncated = true;
                    _append("[log truncated]");
                    return;
                }
                _append(message);
                _lines++;
                _bytes += bytes;
            }
        }
    }
}
