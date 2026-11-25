using System.Globalization;
using System.Security;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// Task Scheduler 1.2 definition of the per-user logon task. Runs this exe with "serve" at logon of the
    /// account, least privilege, interactive token, no execution time limit, restart on failure.
    /// </summary>
    internal static class TaskXml
    {
        private const string Template =
@"<?xml version=""1.0"" encoding=""UTF-16""?>
<Task version=""1.2"" xmlns=""http://schemas.microsoft.com/windows/2004/02/mit/task"">
  <RegistrationInfo>
    <Description>Starts the Ragtime SolidWorks agent when this user logs on. Managed by Ragtime install and uninstall.</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{0}</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id=""Author"">
      <UserId>{1}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>true</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context=""Author"">
    <Exec>
      <Command>{2}</Command>
      <Arguments>serve</Arguments>
      <WorkingDirectory>{3}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>";

        /// <summary>
        /// Builds the XML. <paramref name="accountName"/> is DOMAIN\user for the trigger, <paramref name="sid"/> is the
        /// principal. All values are XML-escaped.
        /// </summary>
        public static string Build(string accountName, string sid, string exePath, string installDirectory) =>
            string.Format(
                CultureInfo.InvariantCulture,
                Template,
                Escape(accountName),
                Escape(sid),
                Escape(exePath),
                Escape(installDirectory));

        private static string Escape(string value) => SecurityElement.Escape(value) ?? string.Empty;
    }
}
