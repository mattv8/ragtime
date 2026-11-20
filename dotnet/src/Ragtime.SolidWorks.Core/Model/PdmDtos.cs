using System.Collections.Generic;
using Newtonsoft.Json;

namespace Ragtime.SolidWorks.Core
{
    public sealed class PdmViewInfo
    {
        public string Vault { get; set; } = string.Empty;
        public string Root { get; set; } = string.Empty;
        public bool LoggedIn { get; set; }
        [JsonProperty(NullValueHandling = NullValueHandling.Include)]
        public string? User { get; set; }
    }

    public sealed class PdmStatusBlock
    {
        public bool ClientInstalled { get; set; }
        [JsonProperty(NullValueHandling = NullValueHandling.Include)]
        public string? ClientVersion { get; set; }
        public List<PdmViewInfo> Views { get; set; } = new List<PdmViewInfo>();
    }

    public sealed class PdmLinkedInfo
    {
        public string Vault { get; set; } = string.Empty;
        [JsonProperty(NullValueHandling = NullValueHandling.Include)]
        public string? Root { get; set; }
        public bool LoggedIn { get; set; }
        [JsonProperty(NullValueHandling = NullValueHandling.Include)]
        public string? User { get; set; }
        public bool WritesEnabled { get; set; }
        public List<string> ApprovedFolders { get; set; } = new List<string>();
        [JsonProperty(NullValueHandling = NullValueHandling.Include)]
        public string? Problem { get; set; }
    }

    public sealed class PdmStatusResult
    {
        [JsonProperty(NullValueHandling = NullValueHandling.Include)]
        public string? WorkFolder { get; set; }
        public bool ClientInstalled { get; set; }
        [JsonProperty(NullValueHandling = NullValueHandling.Include)]
        public string? ClientVersion { get; set; }
        public List<PdmViewInfo> Views { get; set; } = new List<PdmViewInfo>();
        [JsonProperty(NullValueHandling = NullValueHandling.Include)]
        public PdmLinkedInfo? Linked { get; set; }
    }

    public sealed class PdmFolderItem
    {
        public string Name { get; set; } = string.Empty;
        public string Kind { get; set; } = string.Empty;
        public string VaultPath { get; set; } = string.Empty;
        [JsonProperty(NullValueHandling = NullValueHandling.Include)]
        public int? LatestVersion { get; set; }
        [JsonProperty(NullValueHandling = NullValueHandling.Include)]
        public int? LocalVersion { get; set; }
        [JsonProperty(NullValueHandling = NullValueHandling.Include)]
        public bool? CheckedOut { get; set; }
    }

    public sealed class PdmFileInfo
    {
        public string VaultPath { get; set; } = string.Empty;
        public string LocalPath { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public int FileId { get; set; }
        public bool ExistsLocally { get; set; }
        [JsonProperty(NullValueHandling = NullValueHandling.Include)]
        public int? LocalVersion { get; set; }
        public int LatestVersion { get; set; }
        public bool CheckedOut { get; set; }
        [JsonProperty(NullValueHandling = NullValueHandling.Include)]
        public string? CheckedOutBy { get; set; }
        [JsonProperty(NullValueHandling = NullValueHandling.Include)]
        public string? CheckedOutOn { get; set; }
        public bool CheckedOutHere { get; set; }
        /// <summary>Checked out here and checked out by this tool (so this tool may finish it).</summary>
        public bool CheckedOutByTool { get; set; }
        [JsonProperty(NullValueHandling = NullValueHandling.Include)]
        public string? WorkflowState { get; set; }
        public bool InApprovedFolder { get; set; }
    }

    public sealed class PdmVariableItem
    {
        public string Name { get; set; } = string.Empty;
        public string Value { get; set; } = string.Empty;
    }

    public sealed class PdmVariablesResult
    {
        public string VaultPath { get; set; } = string.Empty;
        public List<string> Configurations { get; set; } = new List<string>();
        public string Configuration { get; set; } = string.Empty;
        public List<PdmVariableItem> Variables { get; set; } = new List<PdmVariableItem>();
    }

    public sealed class PdmSkippedItem
    {
        public string VaultPath { get; set; } = string.Empty;
        public string Reason { get; set; } = string.Empty;
    }

    public sealed class PdmGetLatestResult
    {
        public List<string> Fetched { get; set; } = new List<string>();
        public List<PdmSkippedItem> Skipped { get; set; } = new List<PdmSkippedItem>();
        public PdmFileInfo File { get; set; } = new PdmFileInfo();
    }

    public sealed class PdmFolderCreateResult
    {
        public string VaultPath { get; set; } = string.Empty;
        public string LocalPath { get; set; } = string.Empty;
        public bool Created { get; set; }
    }

    public sealed class PdmVariablesSetResult
    {
        public string VaultPath { get; set; } = string.Empty;
        public string Configuration { get; set; } = string.Empty;
        public List<string> Updated { get; set; } = new List<string>();
    }
}
