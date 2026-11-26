using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Disk limits for <c>document.copy</c> (review finding 7). Sizes in bytes.</summary>
    public static class DocumentCopyLimits
    {
        /// <summary>Largest total size of one copy's source files.</summary>
        public const long MaxCopyBytes = 2L * 1024 * 1024 * 1024;

        /// <summary>Largest total size the existing <c>copies</c> folder may hold before a new copy is refused.</summary>
        public const long MaxCopiesFolderBytes = 10L * 1024 * 1024 * 1024;

        /// <summary>Least free space the volume must keep after the copy is written.</summary>
        public const long MinFreeBytesAfterCopy = 5L * 1024 * 1024 * 1024;
    }

    /// <summary>
    /// Lexical checks on the destination of <c>document.copy</c> (review finding 1). They run before any save. The
    /// canonical form is computed with Windows rules (drive, backslashes, "." and ".." segments) and does not touch the
    /// file system; the reparse-point checks are the session's job. Stateless.
    /// </summary>
    public static class CopyDestination
    {
        private const string Field = "destination";
        private const string Separator = "\\";
        private const string OutsideMessage = "A copy destination is outside the copy folder. Nothing was saved.";
        private const string NonLocalMessage = "Only files on local drives can be copied. Mapped network drives and links to network locations are not.";
        private static readonly char[] ForbiddenInSegment = { '/', ':', '*', '?', '"', '<', '>', '|' };

        /// <summary>
        /// The canonical form of an absolute local path: the drive, then segments with "." removed and each ".." applied.
        /// Throws invalid_arguments when the path is not absolute on a drive, has a forbidden character, or ".." leaves the root.
        /// </summary>
        public static string Canonical(string? path)
        {
            if (string.IsNullOrEmpty(path) || path!.Length < 3 || !char.IsLetter(path[0]) || path[1] != ':' || path[2] != '\\')
            {
                throw Outside();
            }

            var segments = new List<string>();
            foreach (var segment in path.Substring(3).Split('\\'))
            {
                if (segment.Length == 0 || segment == ".")
                {
                    continue;
                }

                if (segment == "..")
                {
                    if (segments.Count == 0)
                    {
                        throw Outside();
                    }

                    segments.RemoveAt(segments.Count - 1);
                    continue;
                }

                if (segment.IndexOfAny(ForbiddenInSegment) >= 0 || segment.Any(char.IsControl))
                {
                    throw Outside();
                }

                segments.Add(segment);
            }

            return char.ToUpperInvariant(path[0]) + ":" + Separator + string.Join(Separator, segments);
        }

        /// <summary>
        /// Requires every target to be inside <paramref name="folder"/> (at any depth, never the folder itself), after
        /// canonicalization. The check is a prefix on the canonical forms with a trailing separator, so "folder\..\outside"
        /// is refused. Throws invalid_arguments and changes nothing.
        /// </summary>
        public static void RequireInside(string folder, IEnumerable<string?> targets)
        {
            var canonicalFolder = Canonical(folder);
            var prefix = canonicalFolder + Separator;
            foreach (var target in targets)
            {
                var canonical = Canonical(target);
                if (canonical.Length <= prefix.Length || !canonical.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                {
                    throw Outside();
                }
            }
        }

        /// <summary>
        /// Refuses any source that is not on a local drive (UNC, device, mapped or malformed paths) before anything is
        /// saved. Uses the same path rules as <c>document.open</c>; the Win32 link and drive-type check runs after this.
        /// </summary>
        public static void RequireLocalSources(IEnumerable<string?> sources)
        {
            foreach (var source in sources)
            {
                if (DocumentPaths.FindLocalFileProblem(source) != null)
                {
                    throw HostError.InvalidArguments("path", NonLocalMessage);
                }
            }
        }

        private static HostErrorException Outside() => HostError.InvalidArguments(Field, OutsideMessage);
    }

    /// <summary>
    /// The copy size rules (review finding 7). All sums saturate instead of overflowing, so a huge file cannot wrap to a
    /// small number. Stateless.
    /// </summary>
    public static class CopyQuota
    {
        /// <summary>Saturating addition of two non-negative sizes.</summary>
        public static long Add(long left, long right)
        {
            if (left < 0 || right < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(left), "Sizes are not negative.");
            }

            return right > long.MaxValue - left ? long.MaxValue : left + right;
        }

        /// <summary>Saturating sum of file sizes.</summary>
        public static long SumSizes(IEnumerable<long> sizes) => sizes.Aggregate(0L, Add);

        /// <summary>
        /// Refuses a copy before it is written. Checks, in order: the copy's size against
        /// <see cref="DocumentCopyLimits.MaxCopyBytes"/> (reason copy_size); the existing copies folder against
        /// <see cref="DocumentCopyLimits.MaxCopiesFolderBytes"/>, and the free space left after the copy against
        /// <see cref="DocumentCopyLimits.MinFreeBytesAfterCopy"/> (reason copies_quota). Throws result_too_large.
        /// </summary>
        public static void RequireWithinLimits(long copyBytes, long existingCopiesBytes, long freeBytes)
        {
            if (copyBytes > DocumentCopyLimits.MaxCopyBytes)
            {
                throw Refused("copy_size", DocumentCopyLimits.MaxCopyBytes, copyBytes, "The model is too large to copy.");
            }

            if (existingCopiesBytes > DocumentCopyLimits.MaxCopiesFolderBytes)
            {
                throw Refused(
                    "copies_quota",
                    DocumentCopyLimits.MaxCopiesFolderBytes,
                    existingCopiesBytes,
                    "The copies folder is full. Wait for the copies to expire, then copy again.");
            }

            var required = Add(copyBytes, DocumentCopyLimits.MinFreeBytesAfterCopy);
            if (freeBytes < required)
            {
                throw Refused(
                    "copies_quota",
                    DocumentCopyLimits.MinFreeBytesAfterCopy,
                    copyBytes,
                    "The disk has too little free space for this copy. Free some space, then copy again.",
                    freeBytes);
            }
        }

        private static HostErrorException Refused(string reason, long limit, long size, string message, long? freeBytes = null)
        {
            var details = new JObject
            {
                ["reason"] = reason,
                ["limit_bytes"] = limit,
                ["size_bytes"] = size,
            };
            if (freeBytes.HasValue)
            {
                details["free_bytes"] = freeBytes.Value;
            }

            return HostError.Coded(ErrorCodes.ResultTooLarge, message, details);
        }
    }
}
