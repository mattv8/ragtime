namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// The rules for a mate entity's <c>component_path</c> and <c>resolved</c> (review findings 4 and 5). The Readers lane reads
    /// SolidWorks and passes the facts here, so the rules are tested without COM. Stateless.
    /// </summary>
    public static class MateEntityRules
    {
        /// <summary>
        /// The component path of an entity: <c>Name2</c> already holds the hierarchy of a nested component (for example
        /// "Sub-1/Bracket-2"), so it is used as it is, with backslashes turned into "/". Null for no component name.
        /// </summary>
        public static string? ComponentPath(string? name2)
        {
            if (string.IsNullOrWhiteSpace(name2))
            {
                return null;
            }

            return name2!.Trim().Replace('\\', '/');
        }

        /// <summary>
        /// Whether an entity is resolved. A thrown read of its reference component is unknown, so the entity is not resolved.
        /// A legitimate null (no reference component, for example an assembly-level plane) depends on the entity alone, so
        /// it is resolved. A component is resolved when it is in memory or its file exists (<paramref name="componentResolved"/>).
        /// </summary>
        public static bool Resolved(bool componentReadFailed, bool hasComponent, bool componentResolved)
        {
            if (componentReadFailed)
            {
                return false;
            }

            return !hasComponent || componentResolved;
        }
    }
}
