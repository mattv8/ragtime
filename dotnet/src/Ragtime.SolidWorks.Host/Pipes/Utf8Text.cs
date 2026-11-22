using System.Text;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>Strict UTF-8 helpers for pipe lines: no BOM on output, invalid input is reported, not replaced.</summary>
    internal static class Utf8Text
    {
        private static readonly UTF8Encoding Strict = new UTF8Encoding(false, true);
        private static readonly UTF8Encoding Plain = new UTF8Encoding(false);

        /// <summary>Decodes UTF-8 and removes one trailing carriage return. False when the bytes are not valid UTF-8.</summary>
        public static bool TryDecode(byte[] bytes, out string text)
        {
            try
            {
                text = Strict.GetString(bytes).TrimEnd('\r');
                return true;
            }
            catch (DecoderFallbackException)
            {
                text = string.Empty;
                return false;
            }
        }

        /// <summary>Encodes text as UTF-8 without a byte order mark.</summary>
        public static byte[] Encode(string text) => Plain.GetBytes(text);
    }
}
