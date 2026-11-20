using System.Collections.Generic;

namespace Ragtime.SolidWorks.Core
{
    // DTOs for document, model, configuration, feature and dimension results (contract section 5).

    /// <summary>A document item of <c>documents.list</c> and the result of <c>document.open</c>.</summary>
    public sealed class DocumentItem
    {
        /// <summary>Host-assigned id such as "doc-1".</summary>
        public string DocumentId { get; set; } = string.Empty;

        /// <summary>Instance that holds the document.</summary>
        public string InstanceId { get; set; } = string.Empty;

        /// <summary>Document title (file name).</summary>
        public string Title { get; set; } = string.Empty;

        /// <summary>Full local path.</summary>
        public string Path { get; set; } = string.Empty;

        /// <summary>"part", "assembly" or "drawing".</summary>
        public string Type { get; set; } = string.Empty;

        /// <summary>Active configuration name, or null.</summary>
        [Newtonsoft.Json.JsonProperty(NullValueHandling = Newtonsoft.Json.NullValueHandling.Include)]
        public string? ActiveConfiguration { get; set; }

        /// <summary>True when the host opened the document (only those can be closed by document.close).</summary>
        public bool OpenedByHost { get; set; }

        /// <summary>True when the document is open read-only.</summary>
        public bool ReadOnly { get; set; }

        /// <summary>True when the document has unsaved changes.</summary>
        public bool Modified { get; set; }
    }

    /// <summary>Result of <c>model.summary</c>. Part-only and assembly-only fields are null for other types.</summary>
    public sealed class ModelSummary
    {
        /// <summary>Document id; filled in by Core, not by the session.</summary>
        public string DocumentId { get; set; } = string.Empty;

        /// <summary>"part", "assembly" or "drawing".</summary>
        public string Type { get; set; } = string.Empty;

        /// <summary>Document title.</summary>
        public string Title { get; set; } = string.Empty;

        /// <summary>Full local path.</summary>
        public string Path { get; set; } = string.Empty;

        /// <summary>Active configuration name, or null.</summary>
        [Newtonsoft.Json.JsonProperty(NullValueHandling = Newtonsoft.Json.NullValueHandling.Include)]
        public string? ActiveConfiguration { get; set; }

        /// <summary>Number of configurations.</summary>
        public int ConfigurationCount { get; set; }

        /// <summary>Number of features (system folders excluded).</summary>
        public int FeatureCount { get; set; }

        /// <summary>Features with an error.</summary>
        public int FeaturesWithErrors { get; set; }

        /// <summary>Features with a warning.</summary>
        public int FeaturesWithWarnings { get; set; }

        /// <summary>Material name, or null.</summary>
        [Newtonsoft.Json.JsonProperty(NullValueHandling = Newtonsoft.Json.NullValueHandling.Include)]
        public string? Material { get; set; }

        /// <summary>Number of file-level custom properties.</summary>
        public int CustomPropertyCount { get; set; }

        /// <summary>True when the model needs a rebuild.</summary>
        public bool NeedsRebuild { get; set; }

        /// <summary>Unit string, "SI: m, kg, s, rad".</summary>
        public string Units { get; set; } = string.Empty;

        /// <summary>Part only: number of solid bodies.</summary>
        public int? SolidBodyCount { get; set; }

        /// <summary>Assembly only: number of components.</summary>
        public int? ComponentCount { get; set; }

        /// <summary>Assembly only: number of mates.</summary>
        public int? MateCount { get; set; }
    }

    /// <summary>One item of <c>configurations.list</c>.</summary>
    public sealed class ConfigurationItem
    {
        /// <summary>Configuration name.</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>True for the active configuration.</summary>
        public bool IsActive { get; set; }

        /// <summary>Description, or null.</summary>
        [Newtonsoft.Json.JsonProperty(NullValueHandling = Newtonsoft.Json.NullValueHandling.Include)]
        public string? Description { get; set; }

        /// <summary>Parent configuration name; null for a root configuration.</summary>
        [Newtonsoft.Json.JsonProperty(NullValueHandling = Newtonsoft.Json.NullValueHandling.Include)]
        public string? Parent { get; set; }
    }

    /// <summary>Defaults of the always-present <c>error_code</c> and <c>warning</c> fields of feature and mate items.</summary>
    internal static class FeatureStatusDefaults
    {
        /// <summary>error_code when the item has no error.</summary>
        public const string NoErrorCode = "0";
    }

    /// <summary>One item of <c>features.list</c>.</summary>
    public sealed class FeatureItem
    {
        private string _errorCode = FeatureStatusDefaults.NoErrorCode;
        private object _warning = false;

        /// <summary>Feature name as shown in the tree.</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>SolidWorks feature type name, e.g. "Extrusion".</summary>
        public string Type { get; set; } = string.Empty;

        /// <summary>Feature id as text.</summary>
        public string Id { get; set; } = string.Empty;

        /// <summary>True when suppressed.</summary>
        public bool Suppressed { get; set; }

        /// <summary>Error code text; "0" when the feature has no error. Always serialized.</summary>
        public string? ErrorCode
        {
            get => _errorCode;
            set => _errorCode = value ?? FeatureStatusDefaults.NoErrorCode;
        }

        /// <summary>Warning code text, or false when the feature has no warning. Always serialized.</summary>
        public object? Warning
        {
            get => _warning;
            set => _warning = value ?? false;
        }

        /// <summary>Number of dimensions owned by the feature.</summary>
        public int DimensionCount { get; set; }

        /// <summary>Names of the parent features; filled by the session when available.</summary>
        public List<string>? ParentNames { get; set; }
    }

    /// <summary>Result of <c>feature.get</c>.</summary>
    public sealed class FeatureDetail
    {
        private string _errorCode = FeatureStatusDefaults.NoErrorCode;
        private object _warning = false;

        /// <summary>Feature name.</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>SolidWorks feature type name.</summary>
        public string Type { get; set; } = string.Empty;

        /// <summary>Feature id as text.</summary>
        public string Id { get; set; } = string.Empty;

        /// <summary>True when suppressed.</summary>
        public bool Suppressed { get; set; }

        /// <summary>Error code text; "0" when the feature has no error. Always serialized.</summary>
        public string? ErrorCode
        {
            get => _errorCode;
            set => _errorCode = value ?? FeatureStatusDefaults.NoErrorCode;
        }

        /// <summary>Warning code text, or false when the feature has no warning. Always serialized.</summary>
        public object? Warning
        {
            get => _warning;
            set => _warning = value ?? false;
        }

        /// <summary>Explanation of the error or warning, or null.</summary>
        [Newtonsoft.Json.JsonProperty(NullValueHandling = Newtonsoft.Json.NullValueHandling.Include)]
        public string? WhatsWrong { get; set; }

        /// <summary>Names of the parent features.</summary>
        public List<string> Parents { get; set; } = new List<string>();

        /// <summary>Names of the child features.</summary>
        public List<string> Children { get; set; } = new List<string>();

        /// <summary>Dimensions owned by the feature.</summary>
        public List<DimensionItem> Dimensions { get; set; } = new List<DimensionItem>();

        /// <summary>Name of the sketch the feature is built from, or null.</summary>
        public string? Sketch { get; set; }
    }

    /// <summary>Tolerance of a dimension.</summary>
    public sealed class ToleranceInfo
    {
        /// <summary>Tolerance type text, e.g. "bilateral".</summary>
        public string Type { get; set; } = string.Empty;

        /// <summary>Upper limit in SI units (metres, radians for angles).</summary>
        public double Max { get; set; }

        /// <summary>Lower limit in SI units (metres, radians for angles).</summary>
        public double Min { get; set; }
    }

    /// <summary>One dimension item of <c>dimensions.list</c>, <c>feature.get</c> and <c>sketch.get</c> (dimensions part).</summary>
    public sealed class DimensionItem
    {
        /// <summary>Full dimension name such as "D1@Sketch1@Part.SLDPRT".</summary>
        public string FullName { get; set; } = string.Empty;

        /// <summary>Short name such as "D1".</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>Owning feature name, or null.</summary>
        [Newtonsoft.Json.JsonProperty(NullValueHandling = Newtonsoft.Json.NullValueHandling.Include)]
        public string? Feature { get; set; }

        /// <summary>Value in SI units (metres, radians for angles).</summary>
        public double Value { get; set; }

        /// <summary>"linear", "angular", "radial", "diameter" or "other".</summary>
        public string Type { get; set; } = string.Empty;

        /// <summary>True when the dimension is driven (reference only).</summary>
        public bool Driven { get; set; }

        /// <summary>True when the dimension cannot be edited.</summary>
        public bool ReadOnly { get; set; }

        /// <summary>Tolerance, or null when none.</summary>
        public ToleranceInfo? Tolerance { get; set; }

        /// <summary>Unit string, "SI: m, kg, s, rad".</summary>
        public string Units { get; set; } = string.Empty;
    }

    /// <summary>One item of <c>equations.list</c>.</summary>
    public sealed class EquationItem
    {
        /// <summary>Index in the equation list.</summary>
        public int Index { get; set; }

        /// <summary>Equation text.</summary>
        public string Equation { get; set; } = string.Empty;

        /// <summary>Evaluated value, or null when it cannot be evaluated.</summary>
        [Newtonsoft.Json.JsonProperty(NullValueHandling = Newtonsoft.Json.NullValueHandling.Include)]
        public double? Value { get; set; }

        /// <summary>True for global variables.</summary>
        public bool IsGlobalVariable { get; set; }

        /// <summary>True when suppressed.</summary>
        public bool Suppressed { get; set; }
    }
}
