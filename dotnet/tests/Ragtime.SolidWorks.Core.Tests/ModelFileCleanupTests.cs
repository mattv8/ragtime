using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public sealed class ModelFileCleanupTests
    {
        [Theory]
        [InlineData("export-abc.stl", "export-abc - box-1.STL", true)]
        [InlineData("export-abc.stl", "export-abc.stl", false)]
        [InlineData("export-abc.stl", "export-def - box-1.STL", false)]
        [InlineData("preview-abc.png", "preview-abc-extra.png", true)]
        public void MatchesOnlyStrayFilesWithTargetStem(string target, string candidate, bool expected)
        {
            Assert.Equal(expected, ModelFileCleanup.IsStrayFile(target, candidate));
        }
    }
}
