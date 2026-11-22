using System;
using System.IO;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json.Serialization;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Serializes a script result with bounded size and without COM objects.</summary>
    public static class ScriptResultGuard
    {
        public static JToken Serialize(object? value, Func<object, bool> isComObject)
        {
            if (isComObject == null) throw new ArgumentNullException(nameof(isComObject));
            using (var stream = new LimitedStream(1024 * 1024))
            using (var text = new StreamWriter(stream, new UTF8Encoding(false), 1024, true))
            using (var writer = new DepthWriter(text))
            {
                var serializer = JsonSerializer.Create(new JsonSerializerSettings
                {
                    ReferenceLoopHandling = ReferenceLoopHandling.Error,
                    MaxDepth = 32,
                    ContractResolver = new GuardResolver(isComObject),
                });
                if (value != null && isComObject(value))
                    throw new JsonSerializationException("Return plain values, not SolidWorks objects.");
                serializer.Serialize(writer, value);
                writer.Flush();
                stream.Position = 0;
                using (var reader = new StreamReader(stream, Encoding.UTF8))
                using (var json = new JsonTextReader(reader) { MaxDepth = 32 })
                    return JToken.ReadFrom(json);
            }
        }

        private sealed class GuardResolver : DefaultContractResolver
        {
            private readonly Func<object, bool> _isCom;
            public GuardResolver(Func<object, bool> isCom) => _isCom = isCom;
            protected override JsonContract CreateContract(Type type)
            {
                var contract = base.CreateContract(type);
                contract.OnSerializingCallbacks.Add((value, context) =>
                {
                    if (_isCom(value))
                        throw new JsonSerializationException("Return plain values, not SolidWorks objects.");
                });
                return contract;
            }
        }

        private sealed class DepthWriter : JsonTextWriter
        {
            private int _depth;
            public DepthWriter(TextWriter writer) : base(writer) { }
            public override void WriteStartObject() { Enter(); base.WriteStartObject(); }
            public override void WriteStartArray() { Enter(); base.WriteStartArray(); }
            public override void WriteEndObject() { base.WriteEndObject(); _depth--; }
            public override void WriteEndArray() { base.WriteEndArray(); _depth--; }
            private void Enter()
            {
                if (++_depth > 32) throw new JsonSerializationException("Result exceeds the maximum depth.");
            }
        }

        private sealed class LimitedStream : MemoryStream
        {
            private readonly int _limit;
            public LimitedStream(int limit) => _limit = limit;
            public override void Write(byte[] buffer, int offset, int count)
            {
                if (Position + count > _limit) throw new JsonSerializationException("The script result was too large.");
                base.Write(buffer, offset, count);
            }
        }
    }
}
