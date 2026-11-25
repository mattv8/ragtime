using System.IO;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class ScriptTemplatePathTests
    {
        [Theory]
        [InlineData("32.5.0", 2024)]
        [InlineData("31.0", 2023)]
        public void UsesOnlyRunningRelease(string revision, int year)
        {
            Assert.Equal(Path.Combine("root", "SOLIDWORKS", "SOLIDWORKS " + year, "templates"), ScriptTemplatePath.ForRevision("root", revision));
        }

        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("unknown")]
        public void UnknownRevisionHasNoFallback(string? revision)
        {
            Assert.Null(ScriptTemplatePath.ForRevision("root", revision));
        }
    }
}
