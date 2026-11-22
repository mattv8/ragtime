using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class ScriptResultGuardTests
    {
        [Fact]
        public void SerializesPlainValues()
        {
            var result = ScriptResultGuard.Serialize(new { value = 4, items = new[] { true, false } }, _ => false);
            Assert.Equal(4, (int)result["value"]!);
            Assert.Equal(JTokenType.Array, result["items"]!.Type);
            Assert.Equal(7, (int)ScriptResultGuard.Serialize(7, _ => false));
        }

        [Fact]
        public void RejectsCycles()
        {
            var list = new List<object>();
            list.Add(list);
            Assert.ThrowsAny<Exception>(() => ScriptResultGuard.Serialize(list, _ => false));
        }

        [Fact]
        public void RejectsNestedComObject()
        {
            var com = new object();
            Assert.ThrowsAny<Exception>(() => ScriptResultGuard.Serialize(new[] { com }, item => ReferenceEquals(item, com)));
        }

        [Fact]
        public void RejectsOversizeAndExcessDepth()
        {
            Assert.ThrowsAny<Exception>(() => ScriptResultGuard.Serialize(new string('x', 1048577), _ => false));
            object value = 1;
            for (var i = 0; i < 34; i++)
            {
                value = new[] { value };
            }
            Assert.ThrowsAny<Exception>(() => ScriptResultGuard.Serialize(value, _ => false));
        }
    }
}
