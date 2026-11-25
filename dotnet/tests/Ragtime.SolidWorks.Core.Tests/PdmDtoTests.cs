using System;
using System.Linq;
using Newtonsoft.Json.Linq;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class PdmDtoTests
    {
        [Theory]
        [InlineData(typeof(PdmViewInfo), "vault,root,logged_in,user", "user")]
        [InlineData(typeof(PdmStatusBlock), "client_installed,client_version,views", "client_version")]
        [InlineData(typeof(PdmLinkedInfo), "vault,root,logged_in,user,writes_enabled,approved_folders,problem", "root,user,problem")]
        [InlineData(typeof(PdmStatusResult), "client_installed,client_version,views,linked,work_folder", "client_version,linked,work_folder")]
        [InlineData(typeof(PdmFolderItem), "name,kind,vault_path,latest_version,local_version,checked_out", "latest_version,local_version,checked_out")]
        [InlineData(typeof(PdmFileInfo), "vault_path,local_path,name,file_id,exists_locally,local_version,latest_version,checked_out,checked_out_by,checked_out_on,checked_out_here,checked_out_by_tool,workflow_state,in_approved_folder", "local_version,checked_out_by,checked_out_on,workflow_state")]
        [InlineData(typeof(PdmVariableItem), "name,value", "")]
        [InlineData(typeof(PdmVariablesResult), "vault_path,configurations,configuration,variables", "")]
        [InlineData(typeof(PdmSkippedItem), "vault_path,reason", "")]
        [InlineData(typeof(PdmGetLatestResult), "fetched,skipped,file", "")]
        [InlineData(typeof(PdmFolderCreateResult), "vault_path,local_path,created", "")]
        [InlineData(typeof(PdmVariablesSetResult), "vault_path,configuration,updated", "")]
        public void SerializesContractFieldsIncludingNulls(Type type, string fields, string nullable)
        {
            var result = (JObject)WireJson.ToToken(Activator.CreateInstance(type)!);
            Assert.Equal(fields.Split(',').OrderBy(x => x), result.Properties().Select(p => p.Name).OrderBy(x => x));
            foreach (var name in nullable.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries))
            {
                Assert.Equal(JTokenType.Null, result[name]!.Type);
            }
        }
    }
}
