using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>
    /// Bounded COM traversals. A GetFirst/GetNext chain fails with solidworks_error when an item repeats (the same COM
    /// object seen twice) and when its safety cap is reached; a walk never returns a shortened list. Collections
    /// returned as arrays get the same cap. Stateless.
    /// </summary>
    internal static class ComWalk
    {
        /// <summary>Largest feature tree or sub-feature list read in one walk.</summary>
        public const int FeatureLimit = 100000;

        /// <summary>Largest array of components, equations, properties, configurations or sketch entities.</summary>
        public const int ItemLimit = 100000;

        /// <summary>Largest number of open documents.</summary>
        public const int DocumentLimit = 10000;

        /// <summary>Largest number of display dimensions owned by one feature.</summary>
        public const int DimensionLimit = 10000;

        /// <summary>
        /// Walks a chain from the first item through <paramref name="next"/> until it returns null. <paramref name="subject"/>
        /// names the chain in messages, e.g. "The feature tree".
        /// </summary>
        public static List<T> Chain<T>(string subject, int limit, Func<T?> first, Func<T, T?> next) where T : class
        {
            var items = new List<T>();
            var identities = new HashSet<IntPtr>();
            var current = first();
            while (current != null)
            {
                if (!identities.Add(Identity(current)))
                {
                    throw DidNotTerminate(subject);
                }

                if (items.Count >= limit)
                {
                    throw LimitReached(subject, limit);
                }

                items.Add(current);
                current = next(current);
            }

            return items;
        }

        /// <summary>The items of a COM array, failing when there are more than <see cref="ItemLimit"/>. Null gives none.</summary>
        public static List<object?> Collection(object? value, string subject)
        {
            var items = ComHelper.Objects(value);
            RequireWithinLimit(items.Count, ItemLimit, subject);
            return items;
        }

        /// <summary>Throws solidworks_error with details.limit when <paramref name="count"/> is above <paramref name="limit"/>.</summary>
        public static void RequireWithinLimit(int count, int limit, string subject)
        {
            if (count > limit)
            {
                throw LimitReached(subject, limit);
            }
        }

        /// <summary>solidworks_error for a walk that repeated an item (a COM object seen twice).</summary>
        public static HostErrorException DidNotTerminate(string subject) =>
            Failure(subject + " did not terminate: an item repeated.", subject, null);

        /// <summary>solidworks_error for a walk that reached its safety cap.</summary>
        public static HostErrorException LimitReached(string subject, int limit) =>
            Failure(subject + " is larger than the safety limit of " + limit + " items.", subject, limit);

        /// <summary>The COM identity of an object (its IUnknown pointer), the same for every wrapper of one COM object.</summary>
        public static IntPtr Identity(object item)
        {
            var unknown = Marshal.GetIUnknownForObject(item);
            Marshal.Release(unknown);
            return unknown;
        }

        private static HostErrorException Failure(string message, string operation, int? limit)
        {
            var details = new JObject { ["operation"] = operation };
            if (limit.HasValue)
            {
                details["limit"] = limit.Value;
            }

            return HostError.Coded(ErrorCodes.SolidworksError, message, details);
        }
    }
}
