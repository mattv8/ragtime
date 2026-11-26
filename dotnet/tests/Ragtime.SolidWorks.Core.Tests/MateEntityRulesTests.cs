using System.Collections.Generic;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    /// <summary>
    /// The component path and resolved rules of a mate entity (review findings 4 and 5). The nested fake models SolidWorks'
    /// component tree: every node has a parent link, and Name2 already names the path from the top assembly.
    /// </summary>
    public class MateEntityRulesTests
    {
        /// <summary>A component of the fake tree: its Name2 holds the hierarchy, and its parent is kept for the tree only.</summary>
        private sealed class FakeComponent
        {
            public FakeComponent(string name2, FakeComponent? parent)
            {
                Name2 = name2;
                Parent = parent;
            }

            public string Name2 { get; }

            public FakeComponent? Parent { get; }
        }

        [Fact]
        public void ANestedComponentPathIsItsName2AndNotTheJoinOfItsAncestors()
        {
            var sub = new FakeComponent("Sub-1", null);
            var bracket = new FakeComponent("Sub-1/Bracket-2", sub);

            var path = MateEntityRules.ComponentPath(bracket.Name2);

            Assert.Equal("Sub-1/Bracket-2", path);
            Assert.NotEqual("Sub-1/Sub-1/Bracket-2", path);
        }

        [Fact]
        public void ATopLevelComponentPathIsItsName()
        {
            var top = new FakeComponent("Bracket-1", null);

            Assert.Equal("Bracket-1", MateEntityRules.ComponentPath(top.Name2));
        }

        [Fact]
        public void ABackslashSeparatorBecomesASlash()
        {
            Assert.Equal("Sub-1/Bracket-2", MateEntityRules.ComponentPath(@"Sub-1\Bracket-2"));
        }

        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("   ")]
        public void AMissingNameGivesNoPath(string? name2)
        {
            Assert.Null(MateEntityRules.ComponentPath(name2));
        }

        [Fact]
        public void ANestedFakeTreeGivesOnePathPerComponent()
        {
            var top = new FakeComponent("Top-1", null);
            var sub = new FakeComponent("Sub-1", null);
            var leaf = new FakeComponent("Sub-1/Bracket-2", sub);
            var tree = new List<FakeComponent> { top, sub, leaf };

            var paths = tree.ConvertAll(component => MateEntityRules.ComponentPath(component.Name2));

            Assert.Equal(new[] { "Top-1", "Sub-1", "Sub-1/Bracket-2" }, paths);
        }

        [Fact]
        public void AThrownReadOfTheReferenceComponentIsNotResolved()
        {
            Assert.False(MateEntityRules.Resolved(componentReadFailed: true, hasComponent: false, componentResolved: false));
            Assert.False(MateEntityRules.Resolved(componentReadFailed: true, hasComponent: true, componentResolved: true));
        }

        [Fact]
        public void ALegitimateNullComponentIsResolvedByTheEntityAlone()
        {
            // An assembly-level plane has no reference component: nothing can be missing.
            Assert.True(MateEntityRules.Resolved(componentReadFailed: false, hasComponent: false, componentResolved: false));
        }

        [Fact]
        public void AComponentIsResolvedOnlyWhenItsFileExistsOrItIsInMemory()
        {
            Assert.True(MateEntityRules.Resolved(componentReadFailed: false, hasComponent: true, componentResolved: true));
            Assert.False(MateEntityRules.Resolved(componentReadFailed: false, hasComponent: true, componentResolved: false));
        }
    }
}
