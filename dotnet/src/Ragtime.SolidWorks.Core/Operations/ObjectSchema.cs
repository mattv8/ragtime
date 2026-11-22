using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// Builds the JSON Schema (object) of an operation's arguments. Always emits
    /// <c>additionalProperties: false</c>; paged operations also accept offset, limit and snapshot.
    /// </summary>
    public sealed class ObjectSchema
    {
        private readonly JObject _properties = new JObject();
        private readonly JArray _required = new JArray();

        /// <summary>Adds a property schema.</summary>
        public ObjectSchema Property(string name, JObject schema, bool required = false)
        {
            _properties[name] = schema;
            if (required)
            {
                _required.Add(WireJson.Value(name));
            }

            return this;
        }

        /// <summary>Adds offset, limit and snapshot, the paging arguments of every list operation.</summary>
        public ObjectSchema Paged()
        {
            Property("offset", SchemaTypes.Integer(0, int.MaxValue, "Index of the first item (0-based). Default 0."));
            Property("limit", SchemaTypes.Integer(ProtocolInfo.MinPageLimit, ProtocolInfo.MaxPageLimit, "Maximum items in this page. Default 100."));
            Property("snapshot", SchemaTypes.Snapshot());
            return this;
        }

        /// <summary>Builds the schema object.</summary>
        public JObject Build()
        {
            var schema = new JObject
            {
                ["type"] = WireJson.Value("object"),
                ["properties"] = _properties,
                ["additionalProperties"] = WireJson.Value(false),
            };
            if (_required.Count > 0)
            {
                schema["required"] = _required;
            }

            return schema;
        }
    }

    /// <summary>Small constructors for the schema subset the validator understands.</summary>
    public static class SchemaTypes
    {
        /// <summary>A document reference: a document id or a full path, 1-260 characters.</summary>
        public static JObject Document(string description) => Text(description);

        /// <summary>A non-empty string of at most 260 characters.</summary>
        public static JObject Text(string description) =>
            new JObject
            {
                ["type"] = WireJson.Value("string"),
                ["minLength"] = WireJson.Value(1),
                ["maxLength"] = WireJson.Value(ProtocolInfo.MaxPathLength),
                ["description"] = WireJson.Value(description),
            };

        /// <summary>A boolean.</summary>
        public static JObject Boolean(string description) =>
            new JObject
            {
                ["type"] = WireJson.Value("boolean"),
                ["description"] = WireJson.Value(description),
            };

        /// <summary>An integer from <paramref name="min"/> to <paramref name="max"/> inclusive.</summary>
        public static JObject Integer(int min, int max, string description) =>
            new JObject
            {
                ["type"] = WireJson.Value("integer"),
                ["minimum"] = WireJson.Value(min),
                ["maximum"] = WireJson.Value(max),
                ["description"] = WireJson.Value(description),
            };

        /// <summary>A string that must be one of <paramref name="values"/>.</summary>
        public static JObject Enum(string description, params string[] values)
        {
            var allowed = new JArray();
            foreach (var value in values)
            {
                allowed.Add(WireJson.Value(value));
            }

            return new JObject
            {
                ["type"] = WireJson.Value("string"),
                ["enum"] = allowed,
                ["description"] = WireJson.Value(description),
            };
        }

        /// <summary>An array of non-empty strings, at most <paramref name="maxItems"/> items.</summary>
        public static JObject StringArray(string description, int maxItems) =>
            new JObject
            {
                ["type"] = WireJson.Value("array"),
                ["maxItems"] = WireJson.Value(maxItems),
                ["items"] = new JObject
                {
                    ["type"] = WireJson.Value("string"),
                    ["minLength"] = WireJson.Value(1),
                    ["maxLength"] = WireJson.Value(64),
                },
                ["description"] = WireJson.Value(description),
            };

        /// <summary>The snapshot argument: a lowercase SHA-256 hex string.</summary>
        public static JObject Snapshot() =>
            new JObject
            {
                ["type"] = WireJson.Value("string"),
                ["pattern"] = WireJson.Value("^[0-9a-f]{64}$"),
                ["description"] = WireJson.Value("Snapshot from an earlier page of this list. Fails with stale_snapshot when the list changed."),
            };

        /// <summary>The schema of an operation that takes no arguments.</summary>
        public static JObject NoArguments() => new ObjectSchema().Build();
    }
}
