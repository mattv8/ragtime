using System.Collections.Generic;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Linked vault configuration supplied by the caller.</summary>
    public sealed class PdmContext
    {
        public PdmContext(string vault, IReadOnlyList<string> approvedFolders, string commentPrefix, string toolId)
        {
            Vault = vault;
            ApprovedFolders = approvedFolders;
            CommentPrefix = commentPrefix;
            ToolId = toolId;
        }

        public string Vault { get; }
        public IReadOnlyList<string> ApprovedFolders { get; }
        public string CommentPrefix { get; }
        public string ToolId { get; }
    }

    /// <summary>The <c>context</c> object of a request. Built by Ragtime from the tool config (contract section 4).</summary>
    public sealed class RequestContext
    {
        /// <summary>Creates a context.</summary>
        public RequestContext(
            int? solidworksVersion,
            string instanceMode,
            int maxInstances,
            int idleShutdownSeconds,
            bool allowWrite,
            int deadlineSeconds,
            int maxResultBytes,
            PdmContext? pdm = null,
            bool allowScripts = false)
        {
            SolidworksVersion = solidworksVersion;
            InstanceMode = instanceMode;
            MaxInstances = maxInstances;
            IdleShutdownSeconds = idleShutdownSeconds;
            AllowWrite = allowWrite;
            DeadlineSeconds = deadlineSeconds;
            MaxResultBytes = maxResultBytes;
            Pdm = pdm;
            AllowScripts = allowScripts;
        }

        /// <summary>Pinned release year, or null for the newest installed version.</summary>
        public int? SolidworksVersion { get; }

        /// <summary>"background" or "attach".</summary>
        public string InstanceMode { get; }

        /// <summary>Pool size, 1-4 (background only).</summary>
        public int MaxInstances { get; }

        /// <summary>Idle seconds after which a background instance exits.</summary>
        public int IdleShutdownSeconds { get; }

        /// <summary>Whether operations with access "write" may run.</summary>
        public bool AllowWrite { get; }

        /// <summary>Whether scripts may run, independently of allow_write.</summary>
        public bool AllowScripts { get; }

        /// <summary>Time the host may spend on this request.</summary>
        public int DeadlineSeconds { get; }

        /// <summary>Result budget in UTF-8 bytes.</summary>
        public int MaxResultBytes { get; }

        /// <summary>Linked vault configuration, if present.</summary>
        public PdmContext? Pdm { get; }

        /// <summary>True when the instance mode is attach.</summary>
        public bool IsAttachMode => InstanceMode == ProtocolInfo.InstanceModes.Attach;
    }

    /// <summary>A parsed and structurally valid request. Arguments are not yet validated against a schema.</summary>
    public sealed class ParsedRequest
    {
        /// <summary>Creates a parsed request.</summary>
        public ParsedRequest(string id, string operation, JObject arguments, RequestContext context)
        {
            Id = id;
            Operation = operation;
            Arguments = arguments;
            Context = context;
        }

        /// <summary>Caller-chosen request id, echoed in the response.</summary>
        public string Id { get; }

        /// <summary>Operation name.</summary>
        public string Operation { get; }

        /// <summary>Arguments object (never null; an absent or null field becomes an empty object).</summary>
        public JObject Arguments { get; }

        /// <summary>Request context.</summary>
        public RequestContext Context { get; }
    }
}
