using System;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class ScriptRunRetentionTests
    {
        private static readonly DateTime Now = new DateTime(2026, 11, 20, 12, 0, 0, DateTimeKind.Utc);

        [Fact]
        public void FolderOlderThan24HoursIsExpired()
        {
            Assert.True(ScriptRunRetention.IsExpired(Now.AddHours(-25), Now));
        }

        [Fact]
        public void FolderExactlyAt24HoursIsKept()
        {
            Assert.False(ScriptRunRetention.IsExpired(Now.AddHours(-24), Now));
        }

        [Fact]
        public void RecentFolderIsKept()
        {
            Assert.False(ScriptRunRetention.IsExpired(Now.AddMinutes(-5), Now));
        }
    }
}
