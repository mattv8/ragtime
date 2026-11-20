using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// Validates an arguments object against an operation schema built with <see cref="ObjectSchema"/>.
    /// Supported keywords: type (object is implied; string, integer, boolean, array, scalar-valued object), properties,
    /// required, additionalProperties (only false), minLength, maxLength, minimum, maximum, pattern, enum,
    /// items, maxItems, minProperties, maxProperties. Any other schema shape is a programming error and surfaces as internal_error.
    /// A property sent as JSON null is treated as omitted, so it fails only when it is required.
    /// Failures are invalid_arguments with details.field naming the offending argument (for arrays, field[i]).
    /// </summary>
    public static class ArgumentValidator
    {
        /// <summary>Throws invalid_arguments when <paramref name="arguments"/> does not satisfy <paramref name="schema"/>.</summary>
        public static void Validate(JObject schema, JObject arguments)
        {
            var properties = schema["properties"] as JObject ?? new JObject();

            foreach (var property in arguments.Properties())
            {
                var propertySchema = properties[property.Name] as JObject;
                if (propertySchema == null)
                {
                    throw HostError.InvalidArguments(property.Name, $"'{property.Name}' is not an argument of this operation.");
                }

                if (property.Value.Type == JTokenType.Null)
                {
                    continue;
                }

                CheckValue(property.Name, propertySchema, property.Value);
            }

            if (schema["required"] is JArray required)
            {
                foreach (var entry in required)
                {
                    var name = (string?)entry;
                    if (name == null)
                    {
                        continue;
                    }

                    var value = arguments[name];
                    if (value == null || value.Type == JTokenType.Null)
                    {
                        throw HostError.InvalidArguments(name, $"'{name}' is required.");
                    }
                }
            }
        }

        private static void CheckValue(string field, JObject schema, JToken value)
        {
            var type = (string?)schema["type"];
            switch (type)
            {
                case "string":
                    CheckString(field, schema, value);
                    break;
                case "integer":
                    CheckInteger(field, schema, value);
                    break;
                case "boolean":
                    if (value.Type != JTokenType.Boolean)
                    {
                        throw HostError.InvalidArguments(field, $"'{field}' must be true or false.");
                    }

                    break;
                case "array":
                    CheckArray(field, schema, value);
                    break;
                case "object":
                    CheckObject(field, schema, value);
                    break;
                default:
                    throw new InvalidOperationException($"Unsupported schema type '{type}' for '{field}'.");
            }
        }

        private static void CheckObject(string field, JObject schema, JToken value)
        {
            if (!(value is JObject obj))
            {
                throw HostError.InvalidArguments(field, $"'{field}' must be an object.");
            }

            if (WireJson.TryGetInteger(schema["minProperties"], out var min) && obj.Count < min)
            {
                throw HostError.InvalidArguments(field, $"'{field}' must have at least {min} properties.");
            }

            if (WireJson.TryGetInteger(schema["maxProperties"], out var max) && obj.Count > max)
            {
                throw HostError.InvalidArguments(field, $"'{field}' must have at most {max} properties.");
            }

            var types = schema["additionalProperties"]?["type"] as JArray;
            if (types == null || !JToken.DeepEquals(types, new JArray("string", "number", "boolean")))
            {
                throw new InvalidOperationException($"Unsupported object schema for '{field}'.");
            }

            foreach (var property in obj.Properties())
            {
                switch (property.Value.Type)
                {
                    case JTokenType.String:
                    case JTokenType.Integer:
                    case JTokenType.Float:
                    case JTokenType.Boolean:
                        break;
                    default:
                        throw HostError.InvalidArguments(field, $"'{field}' values must be strings, numbers or booleans.");
                }
            }
        }

        private static void CheckString(string field, JObject schema, JToken value)
        {
            if (value.Type != JTokenType.String)
            {
                throw HostError.InvalidArguments(field, $"'{field}' must be a string.");
            }

            var text = (string?)value ?? string.Empty;
            if (WireJson.TryGetInteger(schema["minLength"], out var minLength) && text.Length < minLength)
            {
                throw HostError.InvalidArguments(field, $"'{field}' must be at least {minLength} characters.");
            }

            if (WireJson.TryGetInteger(schema["maxLength"], out var maxLength) && text.Length > maxLength)
            {
                throw HostError.InvalidArguments(field, $"'{field}' must be at most {maxLength} characters.");
            }

            var pattern = (string?)schema["pattern"];
            if (pattern != null && !Regex.IsMatch(text, pattern, RegexOptions.CultureInvariant))
            {
                throw HostError.InvalidArguments(field, $"'{field}' has an invalid format.");
            }

            if (schema["enum"] is JArray allowed && !ContainsString(allowed, text))
            {
                throw HostError.InvalidArguments(field, $"'{field}' must be one of the allowed values.");
            }
        }

        private static void CheckInteger(string field, JObject schema, JToken value)
        {
            if (!WireJson.TryGetInteger(value, out var number))
            {
                throw HostError.InvalidArguments(field, $"'{field}' must be an integer.");
            }

            if (WireJson.TryGetInteger(schema["minimum"], out var minimum) && number < minimum)
            {
                throw HostError.InvalidArguments(field, $"'{field}' must be at least {minimum}.");
            }

            if (WireJson.TryGetInteger(schema["maximum"], out var maximum) && number > maximum)
            {
                throw HostError.InvalidArguments(field, $"'{field}' must be at most {maximum}.");
            }
        }

        private static void CheckArray(string field, JObject schema, JToken value)
        {
            if (!(value is JArray array))
            {
                throw HostError.InvalidArguments(field, $"'{field}' must be an array.");
            }

            if (WireJson.TryGetInteger(schema["maxItems"], out var maxItems) && array.Count > maxItems)
            {
                throw HostError.InvalidArguments(field, $"'{field}' must have at most {maxItems} items.");
            }

            var itemSchema = schema["items"] as JObject;
            if (itemSchema == null)
            {
                return;
            }

            for (var index = 0; index < array.Count; index++)
            {
                CheckValue($"{field}[{index}]", itemSchema, array[index]);
            }
        }

        private static bool ContainsString(JArray allowed, string text)
        {
            foreach (var entry in allowed)
            {
                if (string.Equals((string?)entry, text, StringComparison.Ordinal))
                {
                    return true;
                }
            }

            return false;
        }
    }

    /// <summary>Typed reads of validated arguments. Call only after <see cref="ArgumentValidator.Validate"/>.</summary>
    internal static class ArgumentReader
    {
        public static string RequiredString(JObject arguments, string name)
        {
            var value = OptionalString(arguments, name);
            if (value == null)
            {
                throw HostError.InvalidArguments(name, $"'{name}' is required.");
            }

            return value;
        }

        public static string? OptionalString(JObject arguments, string name)
        {
            var token = arguments[name];
            return token == null || token.Type != JTokenType.String ? null : (string?)token;
        }

        public static bool OptionalBoolean(JObject arguments, string name, bool fallback)
        {
            var token = arguments[name];
            return token == null || token.Type != JTokenType.Boolean ? fallback : (bool)token;
        }

        public static IReadOnlyList<string>? OptionalStringList(JObject arguments, string name)
        {
            if (!(arguments[name] is JArray array))
            {
                return null;
            }

            var list = new List<string>();
            foreach (var entry in array)
            {
                list.Add((string?)entry ?? string.Empty);
            }

            return list;
        }
    }
}
