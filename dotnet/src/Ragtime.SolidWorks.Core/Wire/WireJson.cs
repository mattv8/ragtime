using System.Globalization;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json.Serialization;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// JSON helpers. DTOs serialize with snake_case names and omit null properties; envelopes are
    /// built as <see cref="JObject"/> so that null host fields are written as JSON null.
    /// </summary>
    public static class WireJson
    {
        private static readonly JsonSerializer DtoSerializer = JsonSerializer.Create(new JsonSerializerSettings
        {
            NullValueHandling = NullValueHandling.Ignore,
            ContractResolver = new DefaultContractResolver { NamingStrategy = new SnakeCaseNamingStrategy() },
        });

        /// <summary>Converts a DTO (or null) to a JSON token using snake_case and omitting null properties.</summary>
        public static JToken ToToken(object? value) =>
            value == null ? JValue.CreateNull() : JToken.FromObject(value, DtoSerializer);

        /// <summary>Compact single-line JSON text. Newlines inside strings are escaped by the serializer.</summary>
        public static string Compact(JToken token) => token.ToString(Formatting.None);

        /// <summary>UTF-8 byte count of a string.</summary>
        public static int ByteCount(string text) => Encoding.UTF8.GetByteCount(text);

        /// <summary>Wraps a primitive as a JSON value; null becomes JSON null.</summary>
        public static JToken Value(object? value) => value == null ? JValue.CreateNull() : new JValue(value);

        /// <summary>True when the token is a JSON integer that fits in a long.</summary>
        public static bool TryGetInteger(JToken? token, out long value)
        {
            value = 0;
            if (token == null || token.Type != JTokenType.Integer)
            {
                return false;
            }

            return long.TryParse(
                token.ToString(Formatting.None),
                NumberStyles.AllowLeadingSign,
                CultureInfo.InvariantCulture,
                out value);
        }
    }
}
