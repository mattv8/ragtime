using System;
using System.Collections.Generic;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// Finds the referenced paths that are not on disk for <c>document.open</c> (review finding 6). Probing stops as soon as
    /// <paramref name="maxMissing"/> missing paths are found, and before every probe when the request deadline has run out;
    /// the result is then the prefix that was checked. Stateless.
    /// </summary>
    public static class MissingReferenceScan
    {
        /// <summary>
        /// The first <paramref name="maxMissing"/> references for which <paramref name="isPresent"/> is false, in order.
        /// <paramref name="isPresent"/> is called at most once per reference, and never after the deadline has expired.
        /// </summary>
        public static IReadOnlyList<string> Collect(
            IEnumerable<string> references,
            Func<string, bool> isPresent,
            Deadline deadline,
            int maxMissing)
        {
            if (references == null) throw new ArgumentNullException(nameof(references));
            if (isPresent == null) throw new ArgumentNullException(nameof(isPresent));
            if (deadline == null) throw new ArgumentNullException(nameof(deadline));

            var missing = new List<string>();
            foreach (var reference in references)
            {
                if (missing.Count >= maxMissing || deadline.Remaining <= TimeSpan.Zero)
                {
                    break;
                }

                if (!isPresent(reference))
                {
                    missing.Add(reference);
                }
            }

            return missing;
        }
    }
}
