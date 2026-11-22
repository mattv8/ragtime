using System;
using System.IO;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class ScriptWorkPathTests
    {
        [Theory]
        [InlineData("../escape")]
        [InlineData("foo\\bar")]
        [InlineData("foo/bar")]
        [InlineData("")]
        [InlineData("bad:name")]
        [InlineData("bad\u0001name")]
        public void RejectsInvalidFileNames(string name)
        {
            Assert.Throws<ArgumentException>(() => ScriptWorkPath.Create(Path.GetTempPath(), name));
        }

        [Fact]
        public void CreatesUniquePathsWithoutFiles()
        {
            var folder = Path.GetTempPath();
            var first = ScriptWorkPath.Create(folder, "part.sldprt");
            var second = ScriptWorkPath.Create(folder, "part.sldprt");
            Assert.NotEqual(first, second);
            Assert.Equal(Path.GetFullPath(folder).TrimEnd(Path.DirectorySeparatorChar), Path.GetDirectoryName(first));
            Assert.False(File.Exists(first));
        }
    }
}
