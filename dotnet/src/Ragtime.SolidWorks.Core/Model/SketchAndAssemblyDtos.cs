using System.Collections.Generic;
using Newtonsoft.Json;

namespace Ragtime.SolidWorks.Core
{
    // DTOs for sketch, mate, component, property and mass-property results (contract section 5).

    /// <summary>Result of <c>sketch.get</c> with part "summary" (the default).</summary>
    public sealed class SketchSummary
    {
        /// <summary>Sketch name.</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>True for a 3D sketch.</summary>
        [JsonProperty("is_3d")]
        public bool Is3D { get; set; }

        /// <summary>SolidWorks constrained status text, e.g. "Fully Defined".</summary>
        public string ConstrainedStatus { get; set; } = string.Empty;

        /// <summary>Number of segments.</summary>
        public int SegmentCount { get; set; }

        /// <summary>Number of sketch points.</summary>
        public int PointCount { get; set; }

        /// <summary>Number of relations.</summary>
        public int RelationCount { get; set; }

        /// <summary>Number of dimensions.</summary>
        public int DimensionCount { get; set; }

        /// <summary>Plane or face the sketch lies on, or null.</summary>
        [Newtonsoft.Json.JsonProperty(NullValueHandling = Newtonsoft.Json.NullValueHandling.Include)]
        public string? PlaneOrFace { get; set; }

        /// <summary>Unit string, "SI: m, kg, s, rad".</summary>
        public string Units { get; set; } = string.Empty;
    }

    /// <summary>Reference to a sketch entity in a relation.</summary>
    public sealed class EntityRef
    {
        /// <summary>Entity kind, e.g. "segment" or "point".</summary>
        public string Kind { get; set; } = string.Empty;

        /// <summary>Entity id as text.</summary>
        public string Id { get; set; } = string.Empty;
    }

    /// <summary>One item of <c>sketch.get</c> part "segments". Coordinates are in sketch space, metres.</summary>
    public sealed class SegmentItem
    {
        /// <summary>Segment id as text.</summary>
        public string Id { get; set; } = string.Empty;

        /// <summary>"line", "arc", "circle", "ellipse", "spline", "parabola", "text" or "other".</summary>
        public string Type { get; set; } = string.Empty;

        /// <summary>True for construction geometry.</summary>
        public bool Construction { get; set; }

        /// <summary>Length in metres.</summary>
        public double Length { get; set; }

        /// <summary>Start point [x, y, z], or null.</summary>
        public double[]? Start { get; set; }

        /// <summary>End point [x, y, z], or null.</summary>
        public double[]? End { get; set; }

        /// <summary>Centre [x, y, z] for arcs and circles, or null.</summary>
        public double[]? Center { get; set; }

        /// <summary>Radius in metres for arcs and circles, or null.</summary>
        public double? Radius { get; set; }
    }

    /// <summary>One item of <c>sketch.get</c> part "points". Position is sketch space, metres.</summary>
    public sealed class PointItem
    {
        /// <summary>Point id as text.</summary>
        public string Id { get; set; } = string.Empty;

        /// <summary>Position [x, y, z].</summary>
        public double[] Position { get; set; } = new double[3];

        /// <summary>True for construction points.</summary>
        public bool Construction { get; set; }
    }

    /// <summary>One item of <c>sketch.get</c> part "relations".</summary>
    public sealed class RelationItem
    {
        /// <summary>Relation type text, e.g. "Horizontal" or "Coincident".</summary>
        public string Type { get; set; } = string.Empty;

        /// <summary>Entities the relation applies to.</summary>
        public List<EntityRef> Entities { get; set; } = new List<EntityRef>();
    }

    /// <summary>One mate entity reference.</summary>
    public sealed class MateEntity
    {
        /// <summary>Component name that owns the entity; empty when SolidWorks cannot return the entity.</summary>
        public string Component { get; set; } = string.Empty;

        /// <summary>
        /// Component names from the top assembly down, joined with "/" (for example "Sub-1/Bracket-2"), or null when there
        /// is no path. Always serialized.
        /// </summary>
        [Newtonsoft.Json.JsonProperty(NullValueHandling = Newtonsoft.Json.NullValueHandling.Include)]
        public string? ComponentPath { get; set; }

        /// <summary>Entity type, e.g. "face" or "edge"; "unknown" when SolidWorks cannot return the entity.</summary>
        public string EntityType { get; set; } = string.Empty;

        /// <summary>
        /// True when the entity is present and its reference component, if any, is not missing. The session sets it;
        /// the default is false so an entity that was never checked shows as a problem.
        /// </summary>
        public bool Resolved { get; set; }
    }

    /// <summary>One item of <c>mates.list</c>.</summary>
    public sealed class MateItem
    {
        private string _errorCode = FeatureStatusDefaults.NoErrorCode;
        private object _warning = false;

        /// <summary>Mate name.</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>Mate type text, e.g. "Coincident".</summary>
        public string Type { get; set; } = string.Empty;

        /// <summary>Alignment text, e.g. "Aligned".</summary>
        public string Alignment { get; set; } = string.Empty;

        /// <summary>True when suppressed.</summary>
        public bool Suppressed { get; set; }

        /// <summary>Error code text; "0" when the mate has no error. Always serialized.</summary>
        public string? ErrorCode
        {
            get => _errorCode;
            set => _errorCode = value ?? FeatureStatusDefaults.NoErrorCode;
        }

        /// <summary>Warning code text, or false when the mate has no warning. Always serialized.</summary>
        public object? Warning
        {
            get => _warning;
            set => _warning = value ?? false;
        }

        /// <summary>
        /// The error object: null for a healthy mate, otherwise the raw code with its <c>swFeatureError_e</c> name and
        /// meaning (see <see cref="FeatureErrorCodes"/>). Derived from <see cref="ErrorCode"/>; always serialized.
        /// </summary>
        [Newtonsoft.Json.JsonProperty(NullValueHandling = Newtonsoft.Json.NullValueHandling.Include)]
        public FeatureErrorInfo? Error => FeatureErrorCodes.Describe(ErrorCode);

        /// <summary>
        /// SolidWorks' own explanation of the mate's error (the same source as <c>feature.get</c>), or null when none.
        /// Always serialized.
        /// </summary>
        [Newtonsoft.Json.JsonProperty(NullValueHandling = Newtonsoft.Json.NullValueHandling.Include)]
        public string? WhatsWrong { get; set; }

        /// <summary>Mate value in SI units, or null for mates without a value.</summary>
        public double? Value { get; set; }

        /// <summary>Entities the mate references.</summary>
        public List<MateEntity> Entities { get; set; } = new List<MateEntity>();

        /// <summary>True when the mate has a warning (the warning value is anything but false).</summary>
        internal bool HasWarning => !Equals(Warning, false);
    }

    /// <summary>One item of <c>components.list</c>.</summary>
    public sealed class ComponentItem
    {
        /// <summary>Component name in the tree, e.g. "Part1-1".</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>Path of the component document.</summary>
        public string Path { get; set; } = string.Empty;

        /// <summary>Configuration name.</summary>
        public string Configuration { get; set; } = string.Empty;

        /// <summary>True when suppressed.</summary>
        public bool Suppressed { get; set; }

        /// <summary>True when hidden.</summary>
        public bool Hidden { get; set; }

        /// <summary>True when fixed.</summary>
        public bool Fixed { get; set; }

        /// <summary>True for virtual components.</summary>
        public bool IsVirtual { get; set; }

        /// <summary>Depth in the assembly tree; 0 for top-level components.</summary>
        public int Depth { get; set; }

        /// <summary>Parent component name, or null for top-level components.</summary>
        [Newtonsoft.Json.JsonProperty(NullValueHandling = Newtonsoft.Json.NullValueHandling.Include)]
        public string? Parent { get; set; }

        /// <summary>
        /// Component transform: 16 numbers in SolidWorks MathTransform.ArrayData order: [0-8] the rotation matrix row
        /// by row, [9-11] the translation in metres, [12] the scale, [13-15] unused.
        /// </summary>
        public double[] Transform { get; set; } = new double[16];

        /// <summary>Number of child components.</summary>
        public int ChildrenCount { get; set; }

        /// <summary>True when the referenced file is on local disk. The session sets it; the default is false.</summary>
        public bool FileExists { get; set; }

        /// <summary>
        /// "resolved", "lightweight", "suppressed" or "missing_file". The session reports the SolidWorks state; Core sets
        /// "missing_file" when <see cref="FileExists"/> is false.
        /// </summary>
        public string Suppression { get; set; } = ComponentSuppression.Resolved;
    }

    /// <summary>The values of <see cref="ComponentItem.Suppression"/>.</summary>
    public static class ComponentSuppression
    {
        /// <summary>Fully resolved.</summary>
        public const string Resolved = "resolved";

        /// <summary>Lightweight.</summary>
        public const string Lightweight = "lightweight";

        /// <summary>Suppressed.</summary>
        public const string Suppressed = "suppressed";

        /// <summary>The referenced file is not on disk.</summary>
        public const string MissingFile = "missing_file";
    }

    /// <summary>One custom property of <c>properties.get</c>.</summary>
    public sealed class PropertyItem
    {
        /// <summary>Property name.</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>Property type text.</summary>
        public string Type { get; set; } = string.Empty;

        /// <summary>Stored value text.</summary>
        public string Value { get; set; } = string.Empty;

        /// <summary>Resolved value text (after evaluation of linked values).</summary>
        /// <summary>
        /// Resolved value text, present only when SolidWorks has an evaluated value cached. Null (omitted from JSON)
        /// otherwise; the tool never forces an evaluation.
        /// </summary>
        public string? ResolvedValue { get; set; }
    }

    /// <summary>Result of <c>properties.get</c>.</summary>
    public sealed class PropertiesResult
    {
        /// <summary>"file" or "configuration:&lt;name&gt;".</summary>
        public string Scope { get; set; } = string.Empty;

        /// <summary>The properties of that scope.</summary>
        public List<PropertyItem> Properties { get; set; } = new List<PropertyItem>();
    }

    /// <summary>Result of <c>mass_properties.get</c>. Quantities are SI.</summary>
    public sealed class MassPropertiesResult
    {
        /// <summary>Mass in kilograms.</summary>
        public double Mass { get; set; }

        /// <summary>Volume in cubic metres.</summary>
        public double Volume { get; set; }

        /// <summary>Surface area in square metres.</summary>
        public double SurfaceArea { get; set; }

        /// <summary>Density in kilograms per cubic metre.</summary>
        public double Density { get; set; }

        /// <summary>Centre of mass [x, y, z] in metres.</summary>
        public double[] CenterOfMass { get; set; } = new double[3];

        /// <summary>Principal moments of inertia [3] in kg·m².</summary>
        public double[] PrincipalMoments { get; set; } = new double[3];

        /// <summary>Moments of inertia about the origin, 9 numbers row-major, kg·m².</summary>
        public double[] MomentsOfInertia { get; set; } = new double[9];

        /// <summary>Bounding box [xmin, ymin, zmin, xmax, ymax, zmax] in metres, or null.</summary>
        public double[]? BoundingBox { get; set; }

        /// <summary>Unit string, "SI: m, kg, s, rad".</summary>
        public string Units { get; set; } = string.Empty;
    }
}
