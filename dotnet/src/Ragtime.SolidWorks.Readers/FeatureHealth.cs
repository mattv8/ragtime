namespace Ragtime.SolidWorks.Readers
{
    /// <summary>Error state of one feature, as reported by <c>Feature.GetErrorCode2</c>.</summary>
    internal readonly struct FeatureHealth
    {
        private const string NoErrorCode = "0";

        private readonly int _code;
        private readonly bool _warningOnly;

        public FeatureHealth(int code, bool warningOnly)
        {
            _code = code;
            _warningOnly = warningOnly;
        }

        /// <summary>True when the feature has an error (a non-zero code that is not warning-only).</summary>
        public bool HasError => _code != 0 && !_warningOnly;

        /// <summary>True when the feature has a warning (a non-zero code flagged warning-only).</summary>
        public bool HasWarning => _code != 0 && _warningOnly;

        /// <summary>The error code as text; "0" when the feature has no error.</summary>
        public string ErrorCode => HasError ? ComHelper.IntText(_code) : NoErrorCode;

        /// <summary>The warning code as text, or false when the feature has no warning.</summary>
        public object Warning => HasWarning ? (object)ComHelper.IntText(_code) : false;
    }
}
