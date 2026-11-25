using System.Collections.Generic;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class ScriptLogBufferTests
    {
        [Fact]
        public void LineCapWritesOneMarker()
        {
            var lines = new List<string>();
            var buffer = new ScriptLogBuffer(lines.Add);
            for (var i = 0; i < 2002; i++)
            {
                buffer.Log("x");
            }

            Assert.True(buffer.Truncated);
            Assert.Equal(2001, lines.Count);
            Assert.Equal("[log truncated]", lines[2000]);
        }

        [Fact]
        public void ByteCapWritesOneMarker()
        {
            var lines = new List<string>();
            var buffer = new ScriptLogBuffer(lines.Add);
            buffer.Log(new string('x', 262145));
            buffer.Log("after");
            Assert.True(buffer.Truncated);
            Assert.Equal(new[] { "[log truncated]" }, lines);
        }
    }
}
