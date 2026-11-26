using System.Linq;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    /// <summary>
    /// The mate error table is wire contract. These literals match the "Mate error names" table in
    /// ragtime/solidworks/guidance/assemblies.md exactly; change both together.
    /// </summary>
    public class FeatureErrorCodesTests
    {
        public static System.Collections.Generic.IEnumerable<object[]> GuidanceEntries()
        {
            yield return new object[] { "38", "swFeatureErrorMateInvalidEdge", "An edge of the mate is suppressed, invalid or no longer present." };
            yield return new object[] { "39", "swFeatureErrorMateInvalidFace", "A face of the mate is suppressed, invalid or no longer present." };
            yield return new object[] { "40", "swFeatureErrorMateFailedCreatingSurface", "The mating surface type is not supported." };
            yield return new object[] { "41", "swFeatureErrorMateInvalidEntity", "An entity of the mate is suppressed, invalid or no longer present." };
            yield return new object[] { "42", "swFeatureErrorMateUnknownTangent", "The tangent is not satisfied." };
            yield return new object[] { "43", "swFeatureErrorMateDanglingGeometry", "The mate points to dangling geometry." };
            yield return new object[] { "44", "swFeatureErrorMateEntityNotLinear", "Non-linear edges cannot be used for mating." };
            yield return new object[] { "45", "swFeatureErrorMateEntityFailed", "Mating is not supported for a component, or a component cannot be modified." };
            yield return new object[] { "46", "swFeatureErrorMateOverdefined", "The mate over-defines the assembly." };
            yield return new object[] { "47", "swFeatureErrorMateIlldefined", "The mate cannot be solved as defined." };
            yield return new object[] { "48", "swFeatureErrorMateBroken", "One or more mate entities were suppressed." };
        }

        [Theory]
        [MemberData(nameof(GuidanceEntries))]
        public void EveryGuidanceCodeHasItsNameAndMeaning(string code, string name, string meaning)
        {
            var error = FeatureErrorCodes.Describe(code);

            Assert.NotNull(error);
            Assert.Equal(code, error!.Code);
            Assert.Equal(name, error.Name);
            Assert.Equal(meaning, error.Meaning);
        }

        [Fact]
        public void TheTableNamesExactlyTheGuidanceCodes()
        {
            var codes = GuidanceEntries().Select(entry => (string)entry[0]).OrderBy(code => code).ToArray();

            Assert.Equal(Enumerable.Range(38, 11).Select(code => code.ToString()).ToArray(), codes);
        }

        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("0")]
        public void HealthyCodesHaveNoErrorObject(string? code)
        {
            Assert.Null(FeatureErrorCodes.Describe(code));
        }

        [Theory]
        [InlineData("1")]
        [InlineData("37")]
        [InlineData("49")]
        [InlineData("73")]
        public void CodesOutsideTheTableAreUnknown(string code)
        {
            var error = FeatureErrorCodes.Describe(code);

            Assert.NotNull(error);
            Assert.Equal(code, error!.Code);
            Assert.Equal("unknown", error.Name);
            Assert.Equal(FeatureErrorCodes.UnknownMeaning, error.Meaning);
        }

        [Fact]
        public void MateErrorSerializesCodeNameAndMeaningAsSnakeCaseKeys()
        {
            var json = WireJson.Compact(WireJson.ToToken(new MateItem { Name = "M", Type = "Coincident", Alignment = "Aligned", ErrorCode = "46" }));

            Assert.Contains("\"error\":{\"code\":\"46\",\"name\":\"swFeatureErrorMateOverdefined\",\"meaning\":\"The mate over-defines the assembly.\"}", json);
        }
    }

    /// <summary>The load-warning names of <c>document.open</c>, by swFileLoadWarning_e bit value.</summary>
    public class FileLoadWarningsTests
    {
        [Fact]
        public void NoBitsGiveAnEmptyList()
        {
            Assert.Empty(FileLoadWarnings.NamesOf(0));
        }

        [Theory]
        [InlineData(0x2, "read_only")]
        [InlineData(0x80, "already_open")]
        [InlineData(0x100000, "references_not_found")]
        [InlineData(0x20, "needs_regen")]
        public void EachKnownBitHasItsName(int bits, string name)
        {
            Assert.Equal(new[] { name }, FileLoadWarnings.NamesOf(bits));
        }

        [Fact]
        public void SeveralBitsComeInBitOrder()
        {
            Assert.Equal(new[] { "read_only", "already_open" }, FileLoadWarnings.NamesOf(0x2 | 0x80));
        }

        [Fact]
        public void AnUnnamedBitIsReportedOnceAsUnknown()
        {
            Assert.Equal(new[] { "read_only", "unknown" }, FileLoadWarnings.NamesOf(0x2 | 0x40000000 | 0x20000000));
        }
    }
}
