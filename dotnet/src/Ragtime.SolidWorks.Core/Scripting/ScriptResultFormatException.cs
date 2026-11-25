using System;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>A <c>result.json</c> whose fields have the wrong type, an unknown status, or a missing required field.</summary>
    public sealed class ScriptResultFormatException : Exception
    {
        /// <summary>Creates the exception with the field that is wrong.</summary>
        public ScriptResultFormatException(string message)
            : base(message)
        {
        }
    }
}
