[CmdletBinding()]
param(
    [string]$ServerUrl,
    [string]$Token,
    [ValidateSet('auto', 'direct', 'tunnel', 'offline')][string]$Mode = 'auto',
    [ValidateRange(1, 65535)][int]$Port,
    [string]$CaFile,
    [string]$TunnelClient,
    [string]$PairingId,
    [string]$PublicKey,
    [string]$TargetUser,
    [string]$Uninstall,
    [string]$CleanupMode,
    [switch]$ConfirmMachineWide,
    [switch]$RunFinalizer,
    [string]$FinalizerContext
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$tagPrefix = '# ragtime-remote:'

function Stop-Install([string]$Message) { throw "Ragtime Remote installer: $Message" }
function Get-OriginalIdentity([string]$RequestedUser) {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    if ($env:SUDO_USER) { Stop-Install 'sudo is unsupported on Windows; start PowerShell as the target user.' }
    if (-not $RequestedUser -or $RequestedUser -eq $identity.Name) {
        return [pscustomobject]@{ Name = $identity.Name; Sid = $identity.User.Value; Profile = (Get-SidProfile $identity.User.Value) }
    }
    try {
        $sid = ([Security.Principal.NTAccount]$RequestedUser).Translate([Security.Principal.SecurityIdentifier])
        $name = $sid.Translate([Security.Principal.NTAccount]).Value
        $profile = (Get-ItemProperty -LiteralPath "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList\$($sid.Value)" -Name ProfileImagePath -ErrorAction Stop).ProfileImagePath
        $profile = [Environment]::ExpandEnvironmentVariables($profile)
    } catch { Stop-Install "Cannot resolve target account $RequestedUser to a SID and profile." }
    [pscustomobject]@{ Name = $name; Sid = $sid.Value; Profile = (Get-SidProfile $sid.Value) }
}
function Test-HttpsUrl([string]$Url) {
    try { $uri = [Uri]$Url } catch { return $false }
    return $uri.Scheme -eq 'https'
}
function Get-EnrollmentMaterial {
    $headers = @{ Authorization = "Bearer $Token" }
    $parameters = @{ Uri = "$($ServerUrl.TrimEnd('/'))/indexes/remote/enroll/material"; Headers = $headers; Method = 'Get'; UseBasicParsing = $true }
    if ($CaFile) { Stop-Install 'Custom CA files require installation into the Windows trust store; no TLS bypass is supported.' }
    return Invoke-RestMethod @parameters
}
function Get-OpenSshExecutable {
    $command = Get-Command sshd.exe -ErrorAction SilentlyContinue
    if ($command) {
        $path = if ($command.Path) { $command.Path } elseif ($command.Source) { $command.Source } else { $command.Definition }
        if ($path) { return [pscustomobject]@{ Path = $path } }
    }
    $fallbackPath = Join-Path $env:WINDIR 'System32\OpenSSH\sshd.exe'
    if (Test-Path -LiteralPath $fallbackPath -PathType Leaf) {
        return [pscustomobject]@{ Path = (Get-Item -LiteralPath $fallbackPath).FullName }
    }
    return $null
}
function Invoke-OpenSshConfigurationProbe {
    param([object]$OpenSsh, [string[]]$Arguments)
    $originalPreference = $ErrorActionPreference
    $output = @()
    $exitCode = 1
    try {
        $ErrorActionPreference = 'Continue'
        $output = @(& $OpenSsh.Path @Arguments 2>$null)
        if ($null -ne $LASTEXITCODE) { $exitCode = $LASTEXITCODE }
    } finally {
        $ErrorActionPreference = $originalPreference
    }
    return [pscustomobject]@{ Output = $output; ExitCode = $exitCode; Succeeded = ($exitCode -eq 0) }
}
function Get-OpenSshConfiguration {
    param([object]$OpenSsh = (Get-OpenSshExecutable))
    if (-not $OpenSsh) { return $null }
    $probe = Invoke-OpenSshConfigurationProbe $OpenSsh @('-T')
    $configuredPort = ($probe.Output | Where-Object { $_ -match '^port ' } | Select-Object -First 1) -replace '^port\s+', ''
    if (-not $configuredPort -or $configuredPort -notmatch '^\d+$') { $configuredPort = 22 }
    return [pscustomobject]@{ Path = $OpenSsh.Path; Port = [int]$configuredPort }
}
function Ensure-OpenSshServer {
    param([object]$OpenSsh)
    if ($OpenSsh) {
        $service = Get-Service sshd -ErrorAction SilentlyContinue
        if (-not $service) { Stop-Install 'OpenSSH Server executable was found, but the sshd service is unavailable. Repair the Windows OpenSSH Server installation.' }
        if ($service -and $service.Status -ne 'Running') {
            if (-not (Test-Elevated)) { Stop-Install 'OpenSSH Server is installed but stopped; re-run elevated to start it.' }
            Set-Service sshd -StartupType Automatic; Start-Service sshd
        }
        return $OpenSsh
    }
    if (-not (Test-Elevated)) {
        Stop-Install 'OpenSSH Server is absent. Re-run elevated, or install it through Windows Optional Features.'
    }
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
    $OpenSsh = Get-OpenSshExecutable
    if (-not $OpenSsh) { Stop-Install 'OpenSSH Server installation completed but sshd.exe was not found.' }
    if (-not (Get-Service sshd -ErrorAction SilentlyContinue)) { Stop-Install 'OpenSSH Server installation completed but the sshd service is unavailable.' }
    Set-Service sshd -StartupType Automatic
    Start-Service sshd
    return $OpenSsh
}
function Get-AuthorizedKeyPath {
    param([object]$Identity, [object]$OpenSsh)
    if (-not $OpenSsh) { Stop-Install 'Cannot determine the effective sshd AuthorizedKeysFile because sshd.exe is unavailable.' }
    $probe = Invoke-OpenSshConfigurationProbe $OpenSsh @('-T', '-C', "user=$($Identity.Name),host=$env:COMPUTERNAME,addr=127.0.0.1")
    $effectiveConfiguration = $probe.Output
    if (-not $probe.Succeeded -or $effectiveConfiguration.Count -eq 0) {
        if (Test-Elevated) { Stop-Install 'Cannot determine the effective sshd AuthorizedKeysFile. Refusing to select a key path.' }
        return Get-NonElevatedAuthorizedKeyPath $Identity ([Security.Principal.WindowsIdentity]::GetCurrent())
    }
    $profileKey = [IO.Path]::GetFullPath((Join-Path $Identity.Profile '.ssh\authorized_keys'))
    $adminKey = [IO.Path]::GetFullPath((Join-Path $env:ProgramData 'ssh\administrators_authorized_keys'))
    foreach ($line in $effectiveConfiguration) {
        if ($line -notmatch '^authorizedkeysfile\s+(.+)$') { continue }
        foreach ($configuredPath in $Matches[1].Split(' ', [StringSplitOptions]::RemoveEmptyEntries)) {
            $candidate = $configuredPath.Replace('/', '\\').Replace('__PROGRAMDATA__', $env:ProgramData)
            if (-not [IO.Path]::IsPathRooted($candidate)) { $candidate = Join-Path $Identity.Profile $candidate }
            try { $candidate = [IO.Path]::GetFullPath($candidate) } catch { continue }
            if ($candidate -eq $adminKey) {
                if (-not (Test-Elevated)) { Stop-Install 'The effective sshd rule uses administrators_authorized_keys. Re-run elevated to write that protected file.' }
                return $adminKey
            }
            if ($candidate -eq $profileKey) { return $profileKey }
        }
    }
    Stop-Install 'The effective sshd AuthorizedKeysFile is not a supported target profile or administrators key path.'
}
function Test-Elevated { return ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator) }
function Test-AdministratorMembership([object]$Identity) {
    $administratorsSid = 'S-1-5-32-544'
    if (@($Identity.Groups | ForEach-Object { $_.Value }) -contains $administratorsSid) { return $true }
    return @($Identity.Claims | Where-Object { $_.Type -eq [Security.Claims.ClaimTypes]::DenyOnlySid -and $_.Value -eq $administratorsSid }).Count -gt 0
}
function Get-NonElevatedAuthorizedKeyPath([object]$Identity, [object]$CurrentIdentity) {
    if ($Identity.Sid -ne $currentIdentity.User.Value) { Stop-Install 'Cannot validate a different target account without elevation and effective sshd configuration.' }
    if (Test-AdministratorMembership $CurrentIdentity) {
        Stop-Install 'This administrator account may use sshd administrators_authorized_keys. Re-run elevated to determine and write the effective key file.'
    }
    $profileKey = [IO.Path]::GetFullPath((Join-Path $Identity.Profile '.ssh\authorized_keys'))
    $configPath = Join-Path $env:ProgramData 'ssh\sshd_config'
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) { return $profileKey }
    $configuredPaths = @()
    $inMatchBlock = $false
    try { $configLines = @(Get-Content -LiteralPath $configPath -ErrorAction Stop) }
    catch { Stop-Install 'Cannot read sshd_config without elevation. Re-run elevated or have an administrator validate the effective AuthorizedKeysFile.' }
    foreach ($rawLine in $configLines) {
        $line = ($rawLine -replace '\s+#.*$', '').Trim()
        if (-not $line) { continue }
        if ($line -match '(?i)^Include\s+') { Stop-Install 'The readable sshd_config uses Include directives. Re-run elevated or have an administrator validate the effective AuthorizedKeysFile.' }
        if ($line -match '^Match\s+(.+)$') {
            $inMatchBlock = $true
            if ($Matches[1] -notmatch '(?i)^Group\s+administrators$') {
                Stop-Install 'The readable sshd_config has a custom Match rule; re-run elevated to validate the effective AuthorizedKeysFile.'
            }
            continue
        }
        if ($line -match '(?i)^AuthorizedKeysFile\s+(.+)$') {
            if ($inMatchBlock) { continue }
            $configuredPaths += $Matches[1].Split(' ', [StringSplitOptions]::RemoveEmptyEntries)
        }
    }
    if ($configuredPaths.Count -eq 0) { return $profileKey }
    foreach ($configuredPath in $configuredPaths) {
        $candidate = $configuredPath.Replace('/', '\\').Replace('%h', $Identity.Profile)
        if (-not [IO.Path]::IsPathRooted($candidate)) { $candidate = Join-Path $Identity.Profile $candidate }
        try { $candidate = [IO.Path]::GetFullPath($candidate) } catch { continue }
        if ($candidate -eq $profileKey) { return $profileKey }
    }
    Stop-Install 'The readable sshd_config AuthorizedKeysFile is not the supported target profile path; re-run elevated to validate it.'
}
function Set-FileSystemAcl([string]$Path, [object]$Acl) {
    if (Get-Command Set-Acl -ErrorAction SilentlyContinue) {
        Set-Acl -LiteralPath $Path -AclObject $Acl
        return
    }
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.PSIsContainer) { [IO.FileSystemAclExtensions]::SetAccessControl([IO.DirectoryInfo]$item, [Security.AccessControl.DirectorySecurity]$Acl) }
    else { [IO.FileSystemAclExtensions]::SetAccessControl([IO.FileInfo]$item, [Security.AccessControl.FileSecurity]$Acl) }
}
function Set-ExactKeyAcl([string]$Path, [string]$Sid) {
    $acl = New-Object Security.AccessControl.FileSecurity
    $acl.SetAccessRuleProtection($true, $false)
    $account = New-Object Security.Principal.SecurityIdentifier($Sid)
    $rule = New-Object Security.AccessControl.FileSystemAccessRule($account, 'FullControl', 'Allow')
    $acl.SetOwner($account)
    $acl.AddAccessRule($rule)
    foreach ($builtInSid in @('S-1-5-18', 'S-1-5-32-544')) {
        $principal = New-Object Security.Principal.SecurityIdentifier($builtInSid)
        $acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($principal, 'FullControl', 'Allow')))
    }
    Set-FileSystemAcl $Path $acl
}
function Set-AdministratorKeyAcl([string]$Path) {
    $acl = New-Object Security.AccessControl.FileSecurity
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($sid in @('S-1-5-18', 'S-1-5-32-544')) {
        $principal = New-Object Security.Principal.SecurityIdentifier($sid)
        $acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($principal, 'FullControl', 'Allow')))
    }
    Set-FileSystemAcl $Path $acl
}
function Set-AuthorizedKey([string]$Path, [string]$Tag, [string]$Key) {
    $encoding = New-Object Text.UTF8Encoding($false)
    $lines = if (Test-Path -LiteralPath $Path) { @(Get-Content -LiteralPath $Path) } else { @() }
    $replacement = "$Key $Tag"
    $written = $false
    $result = @(foreach ($line in $lines) {
        if ($line.Contains($Tag)) { if (-not $written) { $written = $true; $replacement }; continue }
        $line
    })
    if (-not $written) { $result += $replacement }
    [IO.File]::WriteAllText($Path, (($result -join "`n") + "`n"), $encoding)
}
function Get-Nonce([string]$Path) {
    if (Test-Path -LiteralPath $Path) { return [IO.File]::ReadAllText($Path) }
    $bytes = New-Object byte[] 24; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $nonce = [Convert]::ToBase64String($bytes).Replace('+', '').Replace('/', '').Replace('=', '')
    [IO.File]::WriteAllText($Path, $nonce, [Text.Encoding]::UTF8)
    return $nonce
}
function Get-Artifact([string]$Uri, [string]$ExpectedHash, [string]$Destination) {
    if (-not $ExpectedHash -or $ExpectedHash -notmatch '^[A-Fa-f0-9]{64}$') { Stop-Install 'Enrollment material omitted a valid tunnel-client SHA-256.' }
    Invoke-WebRequest -Uri $Uri -OutFile $Destination -UseBasicParsing
    if ((Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -ne $ExpectedHash.ToUpperInvariant()) {
        Remove-Item -LiteralPath $Destination -Force -ErrorAction SilentlyContinue
        Stop-Install 'Tunnel client SHA-256 verification failed.'
    }
}
function Set-SystemOnlyAcl([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { Stop-Install "Refusing reparse point in SYSTEM-owned install path: $Path" }
    $acl = if ($item.PSIsContainer) { New-Object Security.AccessControl.DirectorySecurity } else { New-Object Security.AccessControl.FileSecurity }
    $acl.SetAccessRuleProtection($true, $false)
    $administrators = New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')
    $acl.SetOwner($administrators)
    foreach ($sid in @('S-1-5-18', 'S-1-5-32-544')) {
        $principal = New-Object Security.Principal.SecurityIdentifier($sid)
        $acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($principal, 'FullControl', 'Allow')))
    }
    Set-FileSystemAcl $Path $acl
}
function Initialize-SystemDirectory([string]$Path) {
    Assert-NoReparseAncestors $Path
    if (Test-Path -LiteralPath $Path) {
        $item = Get-Item -LiteralPath $Path -Force
        if (-not $item.PSIsContainer) { Stop-Install "SYSTEM-owned install path is not a directory: $Path" }
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { Stop-Install "Refusing reparse point in SYSTEM-owned install path: $Path" }
    } else {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
    Set-SystemOnlyAcl $Path
}
function Set-ExactDirectoryAcl([string]$Path, [string]$Sid) {
    Assert-NoReparseAncestors $Path
    $acl = New-Object Security.AccessControl.DirectorySecurity
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($accountSid in @($Sid, 'S-1-5-18', 'S-1-5-32-544')) {
        $account = New-Object Security.Principal.SecurityIdentifier($accountSid)
        $acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($account, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')))
    }
    $acl.SetOwner((New-Object Security.Principal.SecurityIdentifier($Sid)))
    Set-FileSystemAcl $Path $acl
}
function Get-Facts([object]$Identity, [int]$SshdPort, [object]$OpenSsh) {
    $addresses = @([Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() | ForEach-Object { $_.GetIPProperties().UnicastAddresses.Address } | Where-Object { $_ -and -not $_.IsIPv6LinkLocal -and -not $_.IsIPv6Loopback -and -not $_.IPAddressToString.StartsWith('127.') -and -not $_.IPAddressToString.StartsWith('169.254.') } | ForEach-Object { $_.ToString() } | Select-Object -Unique -First 8)
    $hostKeys = @()
    Get-ChildItem "$env:ProgramData\ssh\ssh_host_*_key.pub" -ErrorAction SilentlyContinue | Select-Object -First 6 | ForEach-Object { $hostKeys += (Get-Content -LiteralPath $_.FullName -Raw).Trim() }
    [ordered]@{ os = 'windows'; username = $Identity.Name; hostname = $env:COMPUTERNAME; addresses = $addresses; sshd_port = $SshdPort; host_keys = $hostKeys; shared_admin_keyfile = $false; errors = @(); python_available = [bool](Get-Command python.exe -ErrorAction SilentlyContinue) }
}
function Assert-PairingId([string]$PairingId) {
    if ($PairingId -notmatch '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\z') {
        Stop-Install 'Uninstall requires a valid UUID pairing ID.'
    }
    return $PairingId.ToLowerInvariant()
}
function Assert-CleanupMode([string]$Mode) {
    if ($Mode -notin @('DisableSsh', 'PairingOnly')) { Stop-Install 'CleanupMode must be DisableSsh or PairingOnly.' }
}
function Test-InteractiveConsole {
    return [Environment]::UserInteractive -and -not [Console]::IsInputRedirected -and -not [Console]::IsOutputRedirected
}
function Test-TaggedKeyLine([string]$Line, [string]$Tag) {
    $trimmedLine = $Line.TrimEnd()
    if (-not $trimmedLine.EndsWith($Tag, [StringComparison]::Ordinal)) { return $false }
    $tagStart = $trimmedLine.Length - $Tag.Length
    return $tagStart -eq 0 -or [char]::IsWhiteSpace($trimmedLine[$tagStart - 1])
}
function Remove-PairingKeyLines([string]$Path, [string]$PairingId) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return }
    $tag = "$tagPrefix$PairingId"
    $lines = @(Get-Content -LiteralPath $Path)
    $kept = @($lines | Where-Object { -not (Test-TaggedKeyLine $_ $tag) })
    if ($kept.Count -eq $lines.Count) { return }
    $content = if ($kept.Count) { ($kept -join "`n") + "`n" } else { '' }
    [IO.File]::WriteAllText($Path, $content, (New-Object Text.UTF8Encoding($false)))
}
function Remove-PairingDirectory([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    if (Test-Elevated) { Assert-ProtectedSystemPath $Path }
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { Stop-Install "Refusing reparse point in pairing path: $Path" }
    if ($item.PSIsContainer) {
        foreach ($child in @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop)) {
            Remove-PairingDirectory $child.FullName
        }
    }
    Remove-Item -LiteralPath $Path -Force
}
function Assert-NoReparseAncestors([string]$Path) {
    $current = [IO.Path]::GetFullPath($Path)
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force -ErrorAction Stop
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { Stop-Install "Refusing reparse point in path ancestor: $current" }
        }
        $parent = [IO.Path]::GetDirectoryName($current)
        if ($parent -eq $current) { break }
        $current = $parent
    }
}
function Assert-ProtectedSystemPath([string]$Path) {
    Assert-NoReparseAncestors $Path
    $root = [IO.Path]::GetFullPath((Join-Path $env:ProgramData 'RagtimeRemote'))
    $current = [IO.Path]::GetFullPath($Path)
    if ($current -ne $root -and -not $current.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { Stop-Install 'System removal path is outside the protected root.' }
    while ($current -and $current.Length -ge $root.Length) {
        if (Test-Path -LiteralPath $current) {
            $acl = Get-Acl -LiteralPath $current -ErrorAction Stop
            $trusted = @('S-1-5-18', 'S-1-5-32-544')
            if ($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -notin $trusted) { Stop-Install "Untrusted owner in system removal path: $current" }
            foreach ($rule in $acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])) {
                $writes = [Security.AccessControl.FileSystemRights]::Write -bor [Security.AccessControl.FileSystemRights]::Delete -bor [Security.AccessControl.FileSystemRights]::ChangePermissions -bor [Security.AccessControl.FileSystemRights]::TakeOwnership -bor [Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles
                if ($rule.AccessControlType -eq 'Allow' -and $rule.IdentityReference.Value -notin $trusted -and ($rule.FileSystemRights -band $writes)) { Stop-Install "Untrusted write access in system removal path: $current" }
            }
        }
        if ($current -eq $root) { break }
        $current = [IO.Path]::GetDirectoryName($current)
    }
}
function Get-SidProfile([string]$Sid) {
    if ($Sid -notmatch '^S-1-[0-9-]+\z') { Stop-Install 'Invalid target SID.' }
    $profile = (Get-ItemProperty -LiteralPath "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList\$Sid" -Name ProfileImagePath -ErrorAction Stop).ProfileImagePath
    $profile = [Environment]::ExpandEnvironmentVariables([string]$profile)
    if ($profile -notmatch '^[A-Za-z]:[\\/]' -or $profile.Contains('..')) { Stop-Install 'Target SID has an unsafe profile path.' }
    return [IO.Path]::GetFullPath($profile)
}
function New-RemovalSteps([string]$Mode) {
    [ordered]@{ pairing_cleanup = 'confirmed'; ssh_service = $(if ($Mode -eq 'DisableSsh') { 'confirmed' } else { 'not_requested' }) }
}
function Set-RemovalStepStatus([Collections.IDictionary]$Steps, [string]$Step, [string]$Status) {
    $rank = @{ confirmed = 0; not_requested = 0; unconfirmed = 1; failed = 2 }
    if ($rank[$Status] -gt $rank[$Steps[$Step]]) { $Steps[$Step] = $Status }
}
function Write-RemovalResult([Collections.IDictionary]$Steps, [string[]]$Warnings) {
    Write-Output "Ragtime removal: pairing cleanup $($Steps.pairing_cleanup); SSH service $($Steps.ssh_service)."
    $result = [ordered]@{ steps = $Steps; warnings = @($Warnings) } | ConvertTo-Json -Compress
    Write-Output "RAGTIME_REMOVAL_RESULT:$result"
}
function Get-RemovalReceiptPath([string]$PairingId, [object]$Identity, [bool]$Elevated) {
    $root = if ($Elevated) { Join-Path $env:ProgramData 'RagtimeRemote\receipts' } else { Join-Path $Identity.Profile 'AppData\Local\RagtimeRemote\receipts' }
    return Join-Path $root "$PairingId.json"
}
function Write-RemovalReceipt([string]$PairingId, [object]$Identity, [bool]$Elevated, [Collections.IDictionary]$Steps, [string[]]$Warnings) {
    $receipt = Get-RemovalReceiptPath $PairingId $Identity $Elevated
    $parent = Split-Path -Parent $receipt
    Assert-NoReparseAncestors $parent
    if ($Elevated) { Assert-ProtectedSystemPath $parent; Initialize-SystemDirectory (Split-Path -Parent $parent); Initialize-SystemDirectory $parent }
    elseif (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    if (-not $Elevated) { Set-ExactDirectoryAcl $parent $Identity.Sid }
    if (Test-Path -LiteralPath $receipt) {
        $existing = Get-Item -LiteralPath $receipt -Force
        if ($existing.PSIsContainer -or (($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) { Stop-Install "Unsafe removal receipt path: $receipt" }
        [IO.File]::Delete($receipt)
    }
    $stream = [IO.File]::Open($receipt, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $writer = New-Object IO.StreamWriter($stream, (New-Object Text.UTF8Encoding($false)))
        try { $writer.Write(([ordered]@{ steps = $Steps; warnings = @($Warnings) } | ConvertTo-Json -Compress)) }
        finally { $writer.Dispose() }
    } finally { $stream.Dispose() }
    if ($Elevated) { Set-SystemOnlyAcl $receipt } else { Set-ExactKeyAcl $receipt $Identity.Sid }
}
function Invoke-PairingCleanup([string]$PairingId, [object]$Identity, [Collections.IDictionary]$Steps, [System.Collections.ArrayList]$Warnings) {
    $elevated = Test-Elevated
    $operations = @(
        { Remove-PairingKeyLines (Join-Path $Identity.Profile '.ssh\authorized_keys') $PairingId },
        {
            if (-not $elevated) { Stop-Install 'Administrator privileges are required to inspect protected OpenSSH keys and system pairing state.' }
            Remove-PairingKeyLines (Join-Path $env:ProgramData 'ssh\administrators_authorized_keys') $PairingId
        },
        {
            $taskName = "RagtimeRemote-$PairingId"
            try { $task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop }
            catch {
                if ($_.CategoryInfo.Category -eq [Management.Automation.ErrorCategory]::ObjectNotFound) { return }
                throw
            }
            if ($task) {
                Stop-ScheduledTask -TaskName $taskName -ErrorAction Stop
                Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop
            }
        },
        {
            $profileState = Join-Path $Identity.Profile "AppData\Local\RagtimeRemote\$PairingId"
            if ($elevated) {
                if (Test-Path -LiteralPath $profileState) { Stop-Install 'Refusing elevated recursive cleanup of a target-user-controlled pairing directory.' }
                return
            }
            Remove-PairingDirectory $profileState
        },
        {
            if (-not $elevated) { Stop-Install 'Administrator privileges are required to inspect protected system pairing state.' }
            Assert-ProtectedSystemPath (Join-Path $env:ProgramData "RagtimeRemote\$PairingId")
            Remove-PairingDirectory (Join-Path $env:ProgramData "RagtimeRemote\$PairingId")
        }
    )
    foreach ($operation in $operations) {
        try { & $operation } catch {
            Set-RemovalStepStatus $Steps 'pairing_cleanup' $(if ($_.Exception.Message -match 'Administrator privileges') { 'unconfirmed' } else { 'failed' })
            [void]$Warnings.Add("Pairing cleanup failed: $($_.Exception.Message)")
        }
    }
}
function Invoke-SshServiceCleanup([Collections.IDictionary]$Steps, [System.Collections.ArrayList]$Warnings) {
    if (-not (Test-Elevated)) {
        Set-RemovalStepStatus $Steps 'ssh_service' 'failed'
        [void]$Warnings.Add('Administrator privileges are required to disable and stop sshd.')
        return
    }
    try {
        try { $service = Get-Service sshd -ErrorAction Stop }
        catch {
            if ($_.CategoryInfo.Category -eq [Management.Automation.ErrorCategory]::ObjectNotFound) { [void]$Warnings.Add('sshd is absent; it was already secured.'); return }
            throw
        }
        Set-Service sshd -StartupType Disabled
        if ($service.Status -ne 'Stopped') { Stop-Service sshd -ErrorAction Stop }
    } catch {
        Set-RemovalStepStatus $Steps 'ssh_service' 'failed'
        [void]$Warnings.Add("Could not disable and stop sshd: $($_.Exception.Message)")
    }
}
function Test-TunnelTaskPresent([string]$PairingId) {
    return [bool](Get-ScheduledTask -TaskName "RagtimeRemote-$PairingId" -ErrorAction SilentlyContinue)
}
function Get-FinalizerIdentity([string]$Context) {
    if (-not $Context) { Stop-Install 'Finalizer context is required.' }
    try { $identity = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Context)) | ConvertFrom-Json -ErrorAction Stop }
    catch { Stop-Install 'Finalizer context is invalid.' }
    if (-not $identity.Name -or -not $identity.Sid -or -not $identity.Profile -or $identity.Sid -notmatch '^S-1-[0-9-]+\z' -or -not [IO.Path]::IsPathRooted($identity.Profile)) {
        Stop-Install 'Finalizer context has unsafe target identity paths.'
    }
    $registeredProfile = Get-SidProfile $identity.Sid
    if ($identity.Profile -cne $registeredProfile) { Stop-Install 'Finalizer profile does not match the registered target SID.' }
    return [pscustomobject]@{ Name = [string]$identity.Name; Sid = [string]$identity.Sid; Profile = $registeredProfile }
}
function Start-RemovalFinalizer([string]$PairingId, [string]$Mode, [object]$Identity, [Collections.IDictionary]$Steps, [System.Collections.ArrayList]$Warnings) {
    $elevated = Test-Elevated
    $stageRoot = if ($elevated) { Join-Path $env:ProgramData 'RagtimeRemote\removal-payloads' } else { Join-Path $Identity.Profile 'AppData\Local\RagtimeRemote\removal-payloads' }
    $taskName = "RagtimeRemoteRemoval-$PairingId"
    $payload = $null
    try {
        if ($Identity.Profile -cne (Get-SidProfile $Identity.Sid)) { Stop-Install 'Target profile does not match its registered SID.' }
        Assert-NoReparseAncestors $stageRoot
        if ($elevated) { Assert-ProtectedSystemPath $stageRoot; Initialize-SystemDirectory (Split-Path -Parent $stageRoot); Initialize-SystemDirectory $stageRoot }
        else { New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null; Set-ExactDirectoryAcl $stageRoot $Identity.Sid }
        $payload = Join-Path $stageRoot "$PairingId.ps1"
        Remove-Item -LiteralPath "$payload.ready" -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $payload) { Remove-PairingDirectory $payload }
        if (-not $PSCommandPath) { Stop-Install 'Cannot stage a detached removal finalizer without a script file path.' }
        Copy-Item -LiteralPath $PSCommandPath -Destination $payload -Force
        if ($elevated) { Set-SystemOnlyAcl $payload } else { Set-ExactKeyAcl $payload $Identity.Sid }
        $context = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(([ordered]@{ Name = $Identity.Name; Sid = $Identity.Sid; Profile = $Identity.Profile } | ConvertTo-Json -Compress)))
        $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$payload`" -Uninstall $PairingId -CleanupMode $Mode -FinalizerContext $context -RunFinalizer"
        if ($Mode -eq 'DisableSsh') { $arguments += ' -ConfirmMachineWide' }
        $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments
        $trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddSeconds(5))
        $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 2) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
        if ($elevated) { Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -User 'SYSTEM' -RunLevel Highest -Force | Out-Null }
        else { Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -User $Identity.Name -RunLevel Limited -Force | Out-Null }
        Set-RemovalStepStatus $Steps 'pairing_cleanup' 'unconfirmed'
        if ($Mode -eq 'DisableSsh') { Set-RemovalStepStatus $Steps 'ssh_service' 'unconfirmed' }
        [void]$Warnings.Add('Removal finalizer was launched; inspect the local receipt for its final outcome.')
        return $true
    } catch {
        Set-RemovalStepStatus $Steps 'pairing_cleanup' 'failed'
        if ($Mode -eq 'DisableSsh') { Set-RemovalStepStatus $Steps 'ssh_service' 'failed' }
        [void]$Warnings.Add("Could not launch removal finalizer: $($_.Exception.Message)")
        if ($payload) {
            Remove-Item -LiteralPath "$payload.ready" -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $payload -Force -ErrorAction SilentlyContinue
        }
        return $false
    }
}
function Complete-RemovalFinalizer([string]$PairingId) {
    $payload = $PSCommandPath
    try {
        $validatedId = Assert-PairingId $PairingId
        $elevated = Test-Elevated
        $root = if ($elevated) { Join-Path $env:ProgramData 'RagtimeRemote\removal-payloads' }
        else { Join-Path (Get-OriginalIdentity '').Profile 'AppData\Local\RagtimeRemote\removal-payloads' }
        if (-not $payload -or [IO.Path]::GetFullPath($payload) -ne [IO.Path]::GetFullPath((Join-Path $root "$validatedId.ps1"))) { return }
        Assert-NoReparseAncestors $payload
        if ($elevated) { Assert-ProtectedSystemPath $payload }
    } catch { return }
    Remove-Item -LiteralPath "$payload.ready" -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $payload -Force -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName "RagtimeRemoteRemoval-$PairingId" -Confirm:$false -ErrorAction SilentlyContinue
}
function Invoke-Removal([string]$PairingId, [string]$Mode, [object]$Identity) {
    $steps = New-RemovalSteps $Mode
    $warnings = New-Object System.Collections.ArrayList
    if ($Mode -eq 'DisableSsh' -and -not (Test-Elevated)) {
        Set-RemovalStepStatus $steps 'ssh_service' 'failed'
        [void]$warnings.Add('Administrator privileges are required to disable and stop sshd.')
    }
    $defer = -not $RunFinalizer -and ($Mode -eq 'DisableSsh' -or (Test-TunnelTaskPresent $PairingId))
    if ($defer) {
        $launched = Start-RemovalFinalizer $PairingId $Mode $Identity $steps $warnings
        Write-RemovalResult $steps $warnings
        if ($launched) {
            [Console]::Out.Flush()
            $root = if (Test-Elevated) { Join-Path $env:ProgramData 'RagtimeRemote\removal-payloads' } else { Join-Path $Identity.Profile 'AppData\Local\RagtimeRemote\removal-payloads' }
            [IO.File]::WriteAllText((Join-Path $root "$PairingId.ps1.ready"), 'ready')
            if (Test-Elevated) { Set-SystemOnlyAcl (Join-Path $root "$PairingId.ps1.ready") }
            else { Set-ExactKeyAcl (Join-Path $root "$PairingId.ps1.ready") $Identity.Sid }
        }
        return
    }
    Invoke-PairingCleanup $PairingId $Identity $steps $warnings
    if ($Mode -eq 'DisableSsh') { Invoke-SshServiceCleanup $steps $warnings }
    if ($RunFinalizer) {
        try { Write-RemovalReceipt $PairingId $Identity (Test-Elevated) $steps $warnings }
        catch {
            Set-RemovalStepStatus $steps 'pairing_cleanup' 'failed'
            [void]$warnings.Add("Could not write removal receipt: $($_.Exception.Message)")
        }
    }
    Write-RemovalResult $steps $warnings
}
function Invoke-RemovalFinalizer([string]$PairingId, [string]$Mode, [string]$Context) {
    try {
        $identity = Get-FinalizerIdentity $Context
        $expectedRoot = if (Test-Elevated) { Join-Path $env:ProgramData 'RagtimeRemote\removal-payloads' } else { Join-Path $identity.Profile 'AppData\Local\RagtimeRemote\removal-payloads' }
        if ($PSCommandPath -ne (Join-Path $expectedRoot "$PairingId.ps1")) { Stop-Install 'Finalizer must run from its protected staging path.' }
        Assert-NoReparseAncestors $PSCommandPath
        if (Test-Elevated) { Assert-ProtectedSystemPath $PSCommandPath }
        $deadline = [DateTime]::UtcNow.AddSeconds(20)
        while (-not (Test-Path -LiteralPath "$PSCommandPath.ready")) {
            if ([DateTime]::UtcNow -ge $deadline) { Stop-Install 'Finalizer readiness acknowledgement timed out; no cleanup performed.' }
            Start-Sleep -Milliseconds 200
        }
        Invoke-Removal $PairingId $Mode $identity
    } finally { Complete-RemovalFinalizer $PairingId }
}

# RAGTIME_INSTALLER_ENTRYPOINT
if ($Uninstall) {
    $Uninstall = Assert-PairingId $Uninstall
    $cleanupModeSupplied = $PSBoundParameters.ContainsKey('CleanupMode')
    if (-not $CleanupMode) {
        if (-not (Test-InteractiveConsole)) { Stop-Install 'Noninteractive uninstall requires -CleanupMode.' }
        $enteredMode = Read-Host 'Cleanup mode: DisableSsh (default) or PairingOnly'
        $CleanupMode = if ($enteredMode) { $enteredMode } else { 'DisableSsh' }
    }
    Assert-CleanupMode $CleanupMode
    if ($CleanupMode -eq 'PairingOnly' -and $ConfirmMachineWide) { Stop-Install 'ConfirmMachineWide is only valid with DisableSsh.' }
    if ($CleanupMode -eq 'DisableSsh' -and -not $ConfirmMachineWide) {
        if ($cleanupModeSupplied -or $RunFinalizer -or -not (Test-InteractiveConsole)) { Stop-Install 'DisableSsh requires -ConfirmMachineWide.' }
        if ((Read-Host 'This disables SSH for all users. Type YES to continue') -ne 'YES') { Stop-Install 'Machine-wide SSH disablement was not confirmed.' }
    }
    if ($RunFinalizer) {
        Invoke-RemovalFinalizer $Uninstall $CleanupMode $FinalizerContext
    } else {
        $identity = Get-OriginalIdentity $TargetUser
        Invoke-Removal $Uninstall $CleanupMode $identity
    }
    exit 0
}

$identity = Get-OriginalIdentity $TargetUser
if ($Mode -eq 'offline') {
    if (-not $PairingId -or -not $PublicKey) { Stop-Install 'Offline mode requires -PairingId and -PublicKey.' }
} else {
    if (-not (Test-HttpsUrl $ServerUrl)) { Stop-Install '--ServerUrl must be an HTTPS URL.' }
    if (-not $Token) { Stop-Install '-Token is required.' }
}
if ($CaFile -and -not (Test-Path -LiteralPath $CaFile)) { Stop-Install "CA file does not exist: $CaFile" }

$material = if ($Mode -eq 'offline') { [pscustomobject]@{ pairing_id = $PairingId; public_key = $PublicKey } } else { Get-EnrollmentMaterial }
if (-not $material.pairing_id -or -not $material.public_key -or $material.public_key -notmatch '^ssh-[^ ]+ [^ ]+') { Stop-Install 'Enrollment material is incomplete or has an invalid public key.' }
$openSsh = Get-OpenSshExecutable
if ($Mode -ne 'offline') { $openSsh = Ensure-OpenSshServer $openSsh }
$openSsh = Get-OpenSshConfiguration $openSsh
$sshdPort = if ($Port) { $Port } elseif ($openSsh) { $openSsh.Port } else { 22 }
$keyPath = Get-AuthorizedKeyPath $identity $openSsh
$keyDirectory = Split-Path -Parent $keyPath
New-Item -ItemType Directory -Force -Path $keyDirectory | Out-Null
if (-not (Test-Path -LiteralPath $keyPath)) { New-Item -ItemType File -Force -Path $keyPath | Out-Null }
Set-AuthorizedKey $keyPath "$tagPrefix$($material.pairing_id)" $material.public_key
if ($keyPath -eq (Join-Path $env:ProgramData 'ssh\administrators_authorized_keys')) { Set-AdministratorKeyAcl $keyPath } else { Set-ExactKeyAcl $keyPath $identity.Sid }
$facts = Get-Facts $identity $sshdPort $openSsh
$facts.shared_admin_keyfile = $keyPath -eq (Join-Path $env:ProgramData 'ssh\administrators_authorized_keys')
if ($facts.host_keys.Count -eq 0) { Stop-Install 'No SSH host keys were found; start sshd and rerun before enrollment.' }
if ($Mode -eq 'offline') {
    $offlineState = Join-Path $env:LOCALAPPDATA "RagtimeRemote\$($material.pairing_id)"
    New-Item -ItemType Directory -Force -Path $offlineState | Out-Null
    $offline = [ordered]@{ pairing_id = $material.pairing_id; public_key = $material.public_key; nonce = (Get-Nonce (Join-Path $offlineState 'nonce')); facts = $facts; mode = 'offline' } | ConvertTo-Json -Compress -Depth 5
    Write-Output ([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($offline)))
    exit 0
}
$stateDirectory = Join-Path $env:LOCALAPPDATA "RagtimeRemote\$($material.pairing_id)"
New-Item -ItemType Directory -Force -Path $stateDirectory | Out-Null
$nonce = Get-Nonce (Join-Path $stateDirectory 'nonce')
$body = [ordered]@{ nonce = $nonce; facts = $facts; mode = $Mode } | ConvertTo-Json -Compress -Depth 5
$headers = @{ Authorization = "Bearer $Token" }
for ($attempt = 1; $attempt -le 3; $attempt++) {
    try {
        $response = Invoke-RestMethod -Uri "$($ServerUrl.TrimEnd('/'))/indexes/remote/enroll/complete" -Headers $headers -Method Post -ContentType 'application/json' -Body $body -UseBasicParsing
        [IO.File]::WriteAllText((Join-Path $stateDirectory 'complete.json'), ($response | ConvertTo-Json -Depth 8), (New-Object Text.UTF8Encoding($false)))
        if ($response.status -in @('failed', 'revoked', 'expired') -or ($response.transport -ne 'tunnel' -and $response.status -ne 'paired')) { Stop-Install "Enrollment did not pair the host (status: $($response.status))." }
        if ($response.transport -eq 'tunnel') {
            if (-not $response.credential) { Stop-Install 'Enrollment response omitted credential.' }
            $isAdmin = Test-Elevated
            $installDirectory = if ($isAdmin) { Join-Path $env:ProgramData "RagtimeRemote\$($material.pairing_id)" } else { $stateDirectory }
            if ($isAdmin) { Initialize-SystemDirectory (Join-Path $env:ProgramData 'RagtimeRemote'); Initialize-SystemDirectory $installDirectory } else { New-Item -ItemType Directory -Force -Path $installDirectory | Out-Null }
            $clientPath = if ($TunnelClient) { $TunnelClient } else { Join-Path $installDirectory 'tunnel-client.ps1' }
            if (-not $TunnelClient) { Get-Artifact $(if ($material.tunnel_client_powershell_url) { $material.tunnel_client_powershell_url } else { "$($ServerUrl.TrimEnd('/'))/indexes/remote/tunnel-client.ps1" }) $material.artifact_hashes.'tunnel-client.ps1' $clientPath }
            if (-not (Test-Path -LiteralPath $clientPath -PathType Leaf)) { Stop-Install "Tunnel client does not exist: $clientPath" }
            $configPath = Join-Path $installDirectory 'tunnel.json'
            [IO.File]::WriteAllText($configPath, ([ordered]@{ server_url = $(if ($response.ws_url) { $response.ws_url } else { $material.ws_url }); pairing_id = $material.pairing_id; credential = $response.credential; sshd_port = $sshdPort } | ConvertTo-Json -Compress), (New-Object Text.UTF8Encoding($false)))
            if ($isAdmin) { Set-SystemOnlyAcl $clientPath; Set-SystemOnlyAcl $configPath } else { Set-ExactKeyAcl $configPath $identity.Sid }
            $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$clientPath`" `"$configPath`""
            $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew
            if ($isAdmin) {
                Register-ScheduledTask -TaskName "RagtimeRemote-$($material.pairing_id)" -Action $action -Trigger (New-ScheduledTaskTrigger -AtStartup) -Settings $settings -User 'SYSTEM' -RunLevel Highest -Force | Out-Null
            } else {
                Register-ScheduledTask -TaskName "RagtimeRemote-$($material.pairing_id)" -Action $action -Trigger (New-ScheduledTaskTrigger -AtLogOn -User $identity.Name) -Settings $settings -User $identity.Name -RunLevel Limited -Force | Out-Null
                Write-Warning 'Non-elevated tunnel task runs only while this user is logged on.'
            }
            Start-ScheduledTask -TaskName "RagtimeRemote-$($material.pairing_id)"
        }
        Write-Output "Paired $($material.pairing_id) for exact account $($identity.Name) on SSH port $sshdPort. Existing SSH policy was not weakened."
        exit 0
    } catch { if ($attempt -eq 3) { throw }; Start-Sleep -Seconds $attempt }
}
