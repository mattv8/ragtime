using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>
    /// Helpers for COM values: best-effort reads for optional fields, safe conversion of the arrays SolidWorks returns
    /// as <c>object</c>, text formatting, and translation of COM failures. Stateless.
    /// </summary>
    internal static class ComHelper
    {
        /// <summary>Reads a value type; returns null when the read throws (the field is then left empty).</summary>
        public static T? TryValue<T>(Func<T> read) where T : struct
        {
            try
            {
                return read();
            }
            catch (Exception)
            {
                return null;
            }
        }

        /// <summary>Reads a reference; returns null when the read throws or returns null.</summary>
        public static T? TryObject<T>(Func<T?> read) where T : class
        {
            try
            {
                return read();
            }
            catch (Exception)
            {
                return null;
            }
        }

        /// <summary>Reads text; returns null when the read throws.</summary>
        public static string? TryText(Func<string?> read) => TryObject(read);

        /// <summary>Returns the value, or throws solidworks_error naming what SolidWorks did not return.</summary>
        public static T Require<T>(T? value, string what) where T : class
        {
            return value ?? throw HostError.Coded(ErrorCodes.SolidworksError, "SolidWorks did not return " + what + ".");
        }

        /// <summary>Builds solidworks_error for a COM, cast or null failure of one operation. COM errors carry the HRESULT.</summary>
        public static HostErrorException Translate(string action, Exception exception)
        {
            var message = "SolidWorks failed while " + action + ".";
            if (exception is COMException comException)
            {
                return HostError.Coded(
                    ErrorCodes.SolidworksError,
                    message,
                    new JObject { ["hresult"] = HResultText(comException.HResult) });
            }

            return HostError.Coded(ErrorCodes.SolidworksError, message);
        }

        /// <summary>entity_not_found naming the missing kind (feature, sketch, configuration) in details.</summary>
        public static HostErrorException NotFound(string kind, string name) =>
            HostError.Coded(
                ErrorCodes.EntityNotFound,
                "No " + kind + " named '" + name + "' exists in this document.",
                new JObject { [kind] = name });

        /// <summary>HRESULT as eight upper-case hex digits with a 0x prefix.</summary>
        public static string HResultText(int hresult) => "0x" + hresult.ToString("X8", CultureInfo.InvariantCulture);

        /// <summary>Elements of a COM array as objects. A null or non-array value gives an empty list.</summary>
        public static List<object?> Objects(object? value)
        {
            var items = new List<object?>();
            if (value is Array array)
            {
                foreach (var item in array)
                {
                    items.Add(item);
                }
            }

            return items;
        }

        /// <summary>Elements of a COM array as text. Null elements give empty strings.</summary>
        public static List<string> Strings(object? value) =>
            Objects(value).Select(item => Convert.ToString(item, CultureInfo.InvariantCulture)).ToList();

        /// <summary>Elements of a COM array as doubles. Null elements give zero.</summary>
        public static double[] Doubles(object? value) =>
            Objects(value).Select(item => item == null ? 0.0 : Convert.ToDouble(item, CultureInfo.InvariantCulture)).ToArray();

        /// <summary>The first element as a double, or null when the array is empty or missing.</summary>
        public static double? FirstDouble(object? value)
        {
            var values = Doubles(value);
            return values.Length > 0 ? values[0] : (double?)null;
        }

        /// <summary>
        /// The first <paramref name="length"/> elements of a required numeric COM array. Throws solidworks_error when the
        /// array is missing, short or has a null element; never pads with zeros.
        /// </summary>
        public static double[] RequireDoubles(object? value, int length, string what)
        {
            var items = Objects(value);
            if (items.Count < length)
            {
                throw HostError.Coded(ErrorCodes.SolidworksError, "SolidWorks did not return " + what + ".");
            }

            var result = new double[length];
            for (var index = 0; index < length; index++)
            {
                var item = items[index];
                if (item == null)
                {
                    throw HostError.Coded(ErrorCodes.SolidworksError, "SolidWorks did not return " + what + ".");
                }

                result[index] = Convert.ToDouble(item, CultureInfo.InvariantCulture);
            }

            return result;
        }

        /// <summary>
        /// Text for a SolidWorks id. An int array such as GetID()'s pair becomes "a-b"; a scalar becomes its text.
        /// </summary>
        public static string IdText(object? value)
        {
            if (value is Array)
            {
                return string.Join("-", Objects(value).Select(item => Convert.ToString(item, CultureInfo.InvariantCulture)));
            }

            return value == null ? string.Empty : Convert.ToString(value, CultureInfo.InvariantCulture);
        }

        /// <summary>Decimal text of an integer, or empty when it is null.</summary>
        public static string IntText(int? value) => value?.ToString(CultureInfo.InvariantCulture) ?? string.Empty;

        /// <summary>Lower snake case of a CamelCase or UPPER_CASE SolidWorks enum name, e.g. FullyConstrained → fully_constrained.</summary>
        public static string Snake(string text)
        {
            var builder = new StringBuilder(text.Length + 4);
            for (var index = 0; index < text.Length; index++)
            {
                var current = text[index];
                if (index > 0 && char.IsUpper(current) && StartsNewWord(text, index))
                {
                    builder.Append('_');
                }

                builder.Append(char.ToLowerInvariant(current));
            }

            return builder.ToString();
        }

        /// <summary>Removes a leading enum prefix such as "swMate" when present.</summary>
        public static string StripPrefix(string text, string prefix) =>
            text.StartsWith(prefix, StringComparison.Ordinal) ? text.Substring(prefix.Length) : text;

        private static bool StartsNewWord(string text, int index)
        {
            var previous = text[index - 1];
            var nextIsLower = index + 1 < text.Length && char.IsLower(text[index + 1]);
            return char.IsLower(previous) || (char.IsUpper(previous) && nextIsLower);
        }
    }
}
