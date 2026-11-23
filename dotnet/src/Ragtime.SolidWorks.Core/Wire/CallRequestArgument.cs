using System;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>The base64 request that <c>call --request-b64</c> takes on its command line (contract section 3).</summary>
    public static class CallRequestArgument
    {
        /// <summary>
        /// Longest accepted value, in characters. It must cover the largest request Ragtime sends, a <c>script.run</c>
        /// request (MAX_ENCODED_SCRIPT_REQUEST_LENGTH in ragtime/solidworks/protocol.py), and keep the command line
        /// within the 32,767 characters Windows allows.
        /// </summary>
        public const int MaxLength = 32000;

        /// <summary>Decodes the value to the request bytes. Throws invalid_request when it is too long or not base64.</summary>
        public static byte[] Decode(string value)
        {
            if (value.Length > MaxLength)
            {
                throw HostError.InvalidRequest($"The --request-b64 value is longer than {MaxLength} characters.");
            }

            try
            {
                return Convert.FromBase64String(value);
            }
            catch (FormatException)
            {
                throw HostError.InvalidRequest("The --request-b64 value is not valid base64.");
            }
        }
    }
}
