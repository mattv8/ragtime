using System;
using System.IO;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class ScriptRunEntryRuleTests
    {
        private static readonly DateTime Now = new DateTime(2026, 11, 20, 12, 0, 0, DateTimeKind.Utc);
        private static readonly DateTime Old = Now.AddHours(-25);
        private static readonly DateTime Fresh = Now.AddMinutes(-5);

        [Fact]
        public void PlainFolderIsNotALink()
        {
            Assert.False(ScriptRunEntryRule.IsLink(FileAttributes.Directory));
        }

        [Fact]
        public void ReparsePointIsALink()
        {
            Assert.True(ScriptRunEntryRule.IsLink(FileAttributes.Directory | FileAttributes.ReparsePoint));
        }

        [Fact]
        public void ExpiredPlainFolderIsDeleted()
        {
            Assert.Equal(ScriptRunEntryAction.Delete, ScriptRunEntryRule.Decide(FileAttributes.Directory, Old, Now));
        }

        [Fact]
        public void ExpiredReparsePointIsUnlinkedNotFollowed()
        {
            var action = ScriptRunEntryRule.Decide(FileAttributes.Directory | FileAttributes.ReparsePoint, Old, Now);

            Assert.Equal(ScriptRunEntryAction.UnlinkOnly, action);
        }

        [Fact]
        public void FreshFolderIsKeptWhateverItIs()
        {
            Assert.Equal(ScriptRunEntryAction.Keep, ScriptRunEntryRule.Decide(FileAttributes.Directory, Fresh, Now));
            Assert.Equal(ScriptRunEntryAction.Keep, ScriptRunEntryRule.Decide(FileAttributes.Directory | FileAttributes.ReparsePoint, Fresh, Now));
        }
    }
}
