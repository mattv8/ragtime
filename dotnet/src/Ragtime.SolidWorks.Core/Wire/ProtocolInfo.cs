namespace Ragtime.SolidWorks.Core
{
    /// <summary>Protocol constants shared by the host and Ragtime (contract sections 3 and 4).</summary>
    public static class ProtocolInfo
    {
        /// <summary>Protocol version carried by every request and response.</summary>
        public const int Version = 1;

        /// <summary>Largest request line the host accepts, in UTF-8 bytes (64 KiB).</summary>
        public const int MaxRequestBytes = 64 * 1024;

        /// <summary>Serialized result budget in UTF-8 bytes (constant in v1).</summary>
        public const int MaxResultBytes = 32768;

        /// <summary>Longest string accepted for a path or document reference in arguments (characters).</summary>
        public const int MaxPathLength = 260;

        /// <summary>Unit string carried by every result that has quantities.</summary>
        public const string SiUnits = "SI: m, kg, s, rad";

        /// <summary>Default page size for paged results.</summary>
        public const int DefaultPageLimit = 100;

        /// <summary>Smallest accepted page size.</summary>
        public const int MinPageLimit = 1;

        /// <summary>Largest accepted page size.</summary>
        public const int MaxPageLimit = 500;

        /// <summary>Seconds added to deadline_seconds for the client-side overall cap of <c>call</c>.</summary>
        public const int CallGraceSeconds = 5;

        /// <summary>Oldest SolidWorks release year accepted (support floor).</summary>
        public const int MinSolidworksYear = 2022;

        /// <summary>Newest SolidWorks release year accepted by the request validation.</summary>
        public const int MaxSolidworksYear = 2035;

        /// <summary>Smallest allowed max_instances.</summary>
        public const int MinMaxInstances = 1;

        /// <summary>Largest allowed max_instances.</summary>
        public const int MaxMaxInstances = 4;

        /// <summary>Smallest allowed idle_shutdown_seconds.</summary>
        public const int MinIdleShutdownSeconds = 60;

        /// <summary>Largest allowed idle_shutdown_seconds.</summary>
        public const int MaxIdleShutdownSeconds = 86400;

        /// <summary>Largest allowed deadline_seconds.</summary>
        public const int MaxDeadlineSeconds = 86400;

        /// <summary>Instance modes (contract section 2).</summary>
        public static class InstanceModes
        {
            /// <summary>The host launches and owns hidden instances.</summary>
            public const string Background = "background";

            /// <summary>The host uses a running instance in the same desktop session.</summary>
            public const string Attach = "attach";
        }

        /// <summary>Operation access values (contract section 5).</summary>
        public static class Access
        {
            /// <summary>Read-only operation.</summary>
            public const string Read = "read";

            /// <summary>Operation that changes state; refused unless allow_write is set.</summary>
            public const string Write = "write";
        }

        /// <summary>Document type values reported by documents.list and model.summary.</summary>
        public static class DocumentTypes
        {
            /// <summary>Part document (.sldprt).</summary>
            public const string Part = "part";

            /// <summary>Assembly document (.sldasm).</summary>
            public const string Assembly = "assembly";

            /// <summary>Drawing document (.slddrw).</summary>
            public const string Drawing = "drawing";
        }
    }
}
