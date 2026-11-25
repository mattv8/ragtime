using System.Globalization;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Release year <-> revision mapping and version pin checks (contract sections 2 and 4).</summary>
    public static class SolidWorksVersions
    {
        /// <summary>SolidWorks release year = major revision number + this offset (32 -> 2024).</summary>
        public const int MajorToYearOffset = 1992;

        /// <summary>Year of a major revision number, e.g. 32 -> 2024.</summary>
        public static int YearFromMajor(int major) => major + MajorToYearOffset;

        /// <summary>Major revision number of a release year, e.g. 2024 -> 32.</summary>
        public static int MajorFromYear(int year) => year - MajorToYearOffset;

        /// <summary>
        /// Release year of a revision string such as "32.5.0" (major 32 -> 2024). Returns null when the string
        /// does not start with a positive integer.
        /// </summary>
        public static int? RevisionToYear(string? revision)
        {
            if (revision == null || revision.Trim().Length == 0)
            {
                return null;
            }

            var dot = revision.IndexOf('.');
            var majorText = dot < 0 ? revision : revision.Substring(0, dot);
            if (!int.TryParse(majorText, NumberStyles.None, CultureInfo.InvariantCulture, out var major) || major <= 0)
            {
                return null;
            }

            return YearFromMajor(major);
        }

        /// <summary>True for years in the supported range 2022-2035.</summary>
        public static bool IsSupportedYear(int year) =>
            year >= ProtocolInfo.MinSolidworksYear && year <= ProtocolInfo.MaxSolidworksYear;

        /// <summary>
        /// True when an instance of <paramref name="actualYear"/> satisfies the pin. No pin (null) matches
        /// everything; a pin matches only the same year, and an unknown actual year never matches a pin.
        /// </summary>
        public static bool MatchesPin(int? pinnedYear, int? actualYear) =>
            pinnedYear == null || (actualYear != null && actualYear.Value == pinnedYear.Value);
    }
}
