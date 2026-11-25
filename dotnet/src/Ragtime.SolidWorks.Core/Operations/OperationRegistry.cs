using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>One operation: name, summary, access, paging, argument schema and result text.</summary>
    public sealed class OperationDefinition
    {
        /// <summary>Creates a definition.</summary>
        /// <param name="name">Operation name, unique in the registry.</param>
        /// <param name="summary">One-sentence summary for describe.</param>
        /// <param name="access">"read" or "write" (see <see cref="ProtocolInfo.Access"/>).</param>
        /// <param name="paged">True when the result is a page.</param>
        /// <param name="argumentsSchema">JSON Schema object of the arguments.</param>
        /// <param name="result">Text description of the result.</param>
        /// <param name="isInternal">True for operations never listed by describe (host.shutdown).</param>
        public OperationDefinition(
            string name,
            string summary,
            string access,
            bool paged,
            JObject argumentsSchema,
            string result,
            bool isInternal = false)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Summary = summary ?? throw new ArgumentNullException(nameof(summary));
            Access = access ?? throw new ArgumentNullException(nameof(access));
            Paged = paged;
            ArgumentsSchema = argumentsSchema ?? throw new ArgumentNullException(nameof(argumentsSchema));
            Result = result ?? throw new ArgumentNullException(nameof(result));
            IsInternal = isInternal;
        }

        /// <summary>Operation name such as "features.list".</summary>
        public string Name { get; }

        /// <summary>One-sentence summary.</summary>
        public string Summary { get; }

        /// <summary>"read" or "write".</summary>
        public string Access { get; }

        /// <summary>True when the result is a page.</summary>
        public bool Paged { get; }

        /// <summary>JSON Schema object of the arguments.</summary>
        public JObject ArgumentsSchema { get; }

        /// <summary>Text description of the result.</summary>
        public string Result { get; }

        /// <summary>True for internal operations that describe must not list.</summary>
        public bool IsInternal { get; }
    }

    /// <summary>The operations of one host build. Names are unique; lookup is case-sensitive.</summary>
    public sealed class OperationRegistry
    {
        private readonly List<OperationDefinition> _operations;
        private readonly Dictionary<string, OperationDefinition> _byName;

        /// <summary>Creates a registry. Throws when a name is duplicated.</summary>
        public OperationRegistry(IEnumerable<OperationDefinition> operations)
        {
            _operations = (operations ?? throw new ArgumentNullException(nameof(operations))).ToList();
            _byName = new Dictionary<string, OperationDefinition>(StringComparer.Ordinal);
            foreach (var operation in _operations)
            {
                if (_byName.ContainsKey(operation.Name))
                {
                    throw new ArgumentException($"Duplicate operation '{operation.Name}'.", nameof(operations));
                }

                _byName.Add(operation.Name, operation);
            }
        }

        /// <summary>The v1 operation set (contract section 5 plus the internal host.shutdown).</summary>
        public static OperationRegistry Default { get; } = new OperationRegistry(OperationCatalog.Create());

        /// <summary>Every operation including internal ones, in registration order.</summary>
        public IReadOnlyList<OperationDefinition> Operations => _operations;

        /// <summary>Operations listed by describe (internal ones excluded), in registration order.</summary>
        public IReadOnlyList<OperationDefinition> Public => _operations.Where(o => !o.IsInternal).ToList();

        /// <summary>Finds an operation by exact name; null when unknown.</summary>
        public OperationDefinition? Find(string name) =>
            name != null && _byName.TryGetValue(name, out var operation) ? operation : null;
    }
}
