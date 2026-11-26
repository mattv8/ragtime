using System.Collections.Generic;
using System.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// The Core rules for the <c>only_problems</c> arguments and the component suppression state. The session returns the
    /// raw lists; these rules decide what a caller sees.
    /// </summary>
    internal static class ModelListings
    {
        /// <summary>
        /// Mates in session order. With <paramref name="onlyProblems"/>, only mates with an error, a warning, or an entity
        /// that does not resolve.
        /// </summary>
        public static IReadOnlyList<MateItem> Mates(IReadOnlyList<MateItem> mates, bool onlyProblems) =>
            onlyProblems ? mates.Where(HasMateProblem).ToList() : mates;

        /// <summary>
        /// Components in session order. A component whose file is not on disk gets suppression "missing_file". With
        /// <paramref name="onlyProblems"/>, only those components.
        /// </summary>
        public static IReadOnlyList<ComponentItem> Components(IReadOnlyList<ComponentItem> components, bool onlyProblems)
        {
            foreach (var component in components)
            {
                if (!component.FileExists)
                {
                    component.Suppression = ComponentSuppression.MissingFile;
                }
            }

            return onlyProblems ? components.Where(component => !component.FileExists).ToList() : components;
        }

        private static bool HasMateProblem(MateItem mate) =>
            mate.Error != null || mate.HasWarning || mate.Entities.Any(entity => !entity.Resolved);
    }
}
