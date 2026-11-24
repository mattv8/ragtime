using System.Collections.Generic;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    // DTOs for the host-level operations (contract section 5). Serialized with snake_case names;
    // null properties are omitted. Property names are the PascalCase form of the JSON names.

    /// <summary>The <c>account</c> object of <c>status</c>.</summary>
    public sealed class AccountInfo
    {
        /// <summary>DOMAIN\user name of the account the host runs as.</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>SID of that account.</summary>
        public string Sid { get; set; } = string.Empty;

        /// <summary>True when the account is in the Administrators group.</summary>
        public bool IsAdministrator { get; set; }
    }

    /// <summary>The <c>session</c> object of <c>status</c>.</summary>
    public sealed class SessionInfo
    {
        /// <summary>Windows session id of the host process.</summary>
        public int Id { get; set; }

        /// <summary>True when the host runs in an interactive desktop session.</summary>
        public bool Interactive { get; set; }
    }

    /// <summary>One entry of <c>status.solidworks.instances</c>.</summary>
    public sealed class InstanceInfo
    {
        /// <summary>Host-assigned instance id such as "i1".</summary>
        public string InstanceId { get; set; } = string.Empty;

        /// <summary>"background" or "attach".</summary>
        public string Mode { get; set; } = string.Empty;

        /// <summary>True when the host launched the process.</summary>
        public bool Owned { get; set; }

        /// <summary>Release year, or null when unknown.</summary>
        [Newtonsoft.Json.JsonProperty("version", NullValueHandling = Newtonsoft.Json.NullValueHandling.Include)]
        public int? Version { get; set; }

        /// <summary>Revision number such as "32.5.0", or null when unknown.</summary>
        [Newtonsoft.Json.JsonProperty(NullValueHandling = Newtonsoft.Json.NullValueHandling.Include)]
        public string? Revision { get; set; }

        /// <summary>Process id of the SLDWORKS process.</summary>
        public int Pid { get; set; }

        /// <summary>True when the instance window is visible.</summary>
        public bool Visible { get; set; }

        /// <summary>Number of documents the host tracks in this instance.</summary>
        public int Documents { get; set; }

        /// <summary>True while a call is running on the instance (or a timed-out call has not returned).</summary>
        public bool Busy { get; set; }
    }

    /// <summary>The <c>solidworks</c> object of <c>status</c>.</summary>
    public sealed class SolidworksStatus
    {
        /// <summary>Installed release years, ascending.</summary>
        public List<int> InstalledVersions { get; set; } = new List<int>();

        /// <summary>Current instances.</summary>
        public List<InstanceInfo> Instances { get; set; } = new List<InstanceInfo>();
    }

    /// <summary>Result of <c>status</c>.</summary>
    public sealed class StatusResult
    {
        /// <summary>Assembly informational version of the host.</summary>
        public string HostVersion { get; set; } = string.Empty;

        /// <summary>Protocol version (1).</summary>
        public int Protocol { get; set; }

        /// <summary>Account the host runs as.</summary>
        public AccountInfo Account { get; set; } = new AccountInfo();

        /// <summary>Session of the host process.</summary>
        public SessionInfo Session { get; set; } = new SessionInfo();

        /// <summary>SolidWorks installations and instances.</summary>
        public SolidworksStatus Solidworks { get; set; } = new SolidworksStatus();

        /// <summary>PDM client facts. Never logs in; the views' logged_in and user are false/null before the first pdm.* call.</summary>
        public PdmStatusBlock Pdm { get; set; } = new PdmStatusBlock();

        /// <summary>Seconds since the host process started.</summary>
        public double UptimeSeconds { get; set; }

        /// <summary>Work in flight and idle time, used by the idle-stop gate.</summary>
        public HostActivityInfo Activity { get; set; } = new HostActivityInfo();
    }

    /// <summary>The <c>activity</c> object of <c>status</c>.</summary>
    public sealed class HostActivityInfo
    {
        /// <summary>Work requests currently running (0 or more).</summary>
        public int InFlight { get; set; }

        /// <summary>Whole seconds since the last work finished; 0 while work is in flight.</summary>
        public int IdleSeconds { get; set; }

        /// <summary>True while a PDM item whose caller gave up is still running.</summary>
        public bool PdmBusy { get; set; }
    }

    /// <summary>One operation in the <c>describe</c> result.</summary>
    public sealed class OperationDescriptor
    {
        /// <summary>Operation name such as "features.list".</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>One-sentence summary.</summary>
        public string Summary { get; set; } = string.Empty;

        /// <summary>Access: always "read" in v1.</summary>
        public string Access { get; set; } = string.Empty;

        /// <summary>True when the result is a page (items, offset, next_offset, total, snapshot, units).</summary>
        public bool Paged { get; set; }

        /// <summary>JSON Schema (object) of the arguments.</summary>
        public JObject Arguments { get; set; } = new JObject();

        /// <summary>Text description of the result.</summary>
        public string Result { get; set; } = string.Empty;
    }

    /// <summary>Result of <c>describe</c>: the host contract generated from the operation registry.</summary>
    public sealed class DescribeResult
    {
        /// <summary>Protocol version (1).</summary>
        public int Protocol { get; set; }

        /// <summary>Assembly informational version of the host.</summary>
        public string HostVersion { get; set; } = string.Empty;

        /// <summary>Every public operation. The internal host.shutdown is never listed.</summary>
        public List<OperationDescriptor> Operations { get; set; } = new List<OperationDescriptor>();
    }

    /// <summary>Result of the internal <c>host.shutdown</c> operation.</summary>
    public sealed class ShutdownResult
    {
        /// <summary>Always true: the host is stopping.</summary>
        public bool Stopping { get; set; } = true;
    }

    /// <summary>Result of <c>document.close</c>.</summary>
    public sealed class ClosedResult
    {
        /// <summary>Always true.</summary>
        public bool Closed { get; set; } = true;
    }
}
