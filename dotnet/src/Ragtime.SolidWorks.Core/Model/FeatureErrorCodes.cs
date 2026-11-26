using System.Collections.Generic;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// The <c>error</c> object of a mate: the raw code, the <c>swFeatureError_e</c> name and one sentence of meaning.
    /// Serialized only for a mate with an error (the mate item writes <c>error</c> as null when healthy).
    /// </summary>
    public sealed class FeatureErrorInfo
    {
        /// <summary>Creates the object.</summary>
        public FeatureErrorInfo(string code, string name, string meaning)
        {
            Code = code;
            Name = name;
            Meaning = meaning;
        }

        /// <summary>Raw SolidWorks error code as text, for example "46".</summary>
        public string Code { get; }

        /// <summary>swFeatureError_e name, or "unknown" when the table does not name the code.</summary>
        public string Name { get; }

        /// <summary>One sentence explaining the error.</summary>
        public string Meaning { get; }
    }

    /// <summary>
    /// Names and meanings of the <c>swFeatureError_e</c> values that <c>mates.list</c> reports (contract §5.5). Other codes
    /// map to the name "unknown" and keep their raw code. The table is wire contract: the chat guidance quotes these names.
    /// </summary>
    public static class FeatureErrorCodes
    {
        /// <summary>Name for a code the table does not know.</summary>
        public const string UnknownName = "unknown";

        /// <summary>Meaning for a code the table does not know.</summary>
        public const string UnknownMeaning = "SolidWorks reported an error code that this agent does not name; quote the raw code.";

        private const string NoErrorCode = "0";

        private static readonly IReadOnlyDictionary<string, FeatureErrorInfo> Table = new Dictionary<string, FeatureErrorInfo>
        {
            ["38"] = new FeatureErrorInfo("38", "swFeatureErrorMateInvalidEdge", "An edge of the mate is suppressed, invalid or no longer present."),
            ["39"] = new FeatureErrorInfo("39", "swFeatureErrorMateInvalidFace", "A face of the mate is suppressed, invalid or no longer present."),
            ["40"] = new FeatureErrorInfo("40", "swFeatureErrorMateFailedCreatingSurface", "The mating surface type is not supported."),
            ["41"] = new FeatureErrorInfo("41", "swFeatureErrorMateInvalidEntity", "An entity of the mate is suppressed, invalid or no longer present."),
            ["42"] = new FeatureErrorInfo("42", "swFeatureErrorMateUnknownTangent", "The tangent is not satisfied."),
            ["43"] = new FeatureErrorInfo("43", "swFeatureErrorMateDanglingGeometry", "The mate points to dangling geometry."),
            ["44"] = new FeatureErrorInfo("44", "swFeatureErrorMateEntityNotLinear", "Non-linear edges cannot be used for mating."),
            ["45"] = new FeatureErrorInfo("45", "swFeatureErrorMateEntityFailed", "Mating is not supported for a component, or a component cannot be modified."),
            ["46"] = new FeatureErrorInfo("46", "swFeatureErrorMateOverdefined", "The mate over-defines the assembly."),
            ["47"] = new FeatureErrorInfo("47", "swFeatureErrorMateIlldefined", "The mate cannot be solved as defined."),
            ["48"] = new FeatureErrorInfo("48", "swFeatureErrorMateBroken", "One or more mate entities were suppressed."),
        };

        /// <summary>
        /// The error object for a mate's raw code, or null when the code is "0" (no error). Unknown codes give name
        /// "unknown" with the raw code kept.
        /// </summary>
        public static FeatureErrorInfo? Describe(string? code)
        {
            if (string.IsNullOrEmpty(code) || code == NoErrorCode)
            {
                return null;
            }

            return Table.TryGetValue(code!, out var known)
                ? known
                : new FeatureErrorInfo(code!, UnknownName, UnknownMeaning);
        }
    }
}
