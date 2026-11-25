using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class ScriptCompileHintsTests
    {
        [Theory]
        [InlineData("CS1056", "Unexpected character '$'")]
        [InlineData("CS1525", "Invalid expression term '.'")]
        [InlineData("CS1003", "Syntax error")]
        [InlineData("CS0103", "The name 'nameof' does not exist in the current context")]
        public void NewerSyntaxGetsHint(string code, string message)
        {
            Assert.Equal("C# 5 only; see the scripts guidance.", ScriptCompileHints.HintFor(code, message));
        }

        [Theory]
        [InlineData("CS0103", "The name 'other' does not exist in the current context")]
        [InlineData("CS0103", "The name 'nameofOther' does not exist in the current context")]
        [InlineData("CS9999", "nameof")]
        public void OtherErrorsHaveNoHint(string code, string message)
        {
            Assert.Null(ScriptCompileHints.HintFor(code, message));
        }
    }
}
