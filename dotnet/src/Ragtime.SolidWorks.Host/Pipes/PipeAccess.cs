using System.IO.Pipes;
using System.Security.AccessControl;
using System.Security.Principal;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>Pipe ACL: FullControl for the current user SID and nothing else (verified in spike S1).</summary>
    internal static class PipeAccess
    {
        public static PipeSecurity ForUser(string sid)
        {
            var security = new PipeSecurity();
            var identity = new SecurityIdentifier(sid);
            security.AddAccessRule(new PipeAccessRule(identity, PipeAccessRights.FullControl, AccessControlType.Allow));
            return security;
        }
    }
}
