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
    [string]$Uninstall
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$tagPrefix = '# ragtime-remote:'

function Stop-Install([string]$Message) { throw "Ragtime Remote installer: $Message" }
function Get-OriginalIdentity([string]$RequestedUser) {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    if ($env:SUDO_USER) { Stop-Install 'sudo is unsupported on Windows; start PowerShell as the target user.' }
    if (-not $RequestedUser -or $RequestedUser -eq $identity.Name) {
        return [pscustomobject]@{ Name = $identity.Name; Sid = $identity.User.Value; Profile = $env:USERPROFILE }
    }
    try {
        $sid = ([Security.Principal.NTAccount]$RequestedUser).Translate([Security.Principal.SecurityIdentifier])
        $name = $sid.Translate([Security.Principal.NTAccount]).Value
        $profile = (Get-ItemProperty -LiteralPath "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList\$($sid.Value)" -Name ProfileImagePath -ErrorAction Stop).ProfileImagePath
        $profile = [Environment]::ExpandEnvironmentVariables($profile)
    } catch { Stop-Install "Cannot resolve target account $RequestedUser to a SID and profile." }
    if (-not [IO.Path]::IsPathRooted($profile)) { Stop-Install "Target account $name has an unsafe profile path." }
    [pscustomobject]@{ Name = $name; Sid = $sid.Value; Profile = [IO.Path]::GetFullPath($profile) }
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
    if (Test-Path -LiteralPath $Path) {
        $item = Get-Item -LiteralPath $Path -Force
        if (-not $item.PSIsContainer) { Stop-Install "SYSTEM-owned install path is not a directory: $Path" }
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { Stop-Install "Refusing reparse point in SYSTEM-owned install path: $Path" }
    } else {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
    Set-SystemOnlyAcl $Path
}
function Get-Facts([object]$Identity, [int]$SshdPort, [object]$OpenSsh) {
    $addresses = @([Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() | ForEach-Object { $_.GetIPProperties().UnicastAddresses.Address } | Where-Object { $_ -and -not $_.IsIPv6LinkLocal -and -not $_.IsIPv6Loopback -and -not $_.IPAddressToString.StartsWith('127.') -and -not $_.IPAddressToString.StartsWith('169.254.') } | ForEach-Object { $_.ToString() } | Select-Object -Unique -First 8)
    $hostKeys = @()
    Get-ChildItem "$env:ProgramData\ssh\ssh_host_*_key.pub" -ErrorAction SilentlyContinue | Select-Object -First 6 | ForEach-Object { $hostKeys += (Get-Content -LiteralPath $_.FullName -Raw).Trim() }
    [ordered]@{ os = 'windows'; username = $Identity.Name; hostname = $env:COMPUTERNAME; addresses = $addresses; sshd_port = $SshdPort; host_keys = $hostKeys; shared_admin_keyfile = $false; errors = @(); python_available = [bool](Get-Command python.exe -ErrorAction SilentlyContinue) }
}
function Remove-OwnedInstall([string]$PairingId, [object]$Identity) {
    $possible = @((Join-Path $Identity.Profile '.ssh\authorized_keys'), (Join-Path $env:ProgramData 'ssh\administrators_authorized_keys'))
    foreach ($path in $possible) {
        if (Test-Path -LiteralPath $path) {
            $kept = Get-Content -LiteralPath $path | Where-Object { $_ -notlike "*$tagPrefix$PairingId*" }
            [IO.File]::WriteAllText($path, (($kept -join "`n") + $(if ($kept.Count) { "`n" } else { '' })), (New-Object Text.UTF8Encoding($false)))
        }
    }
    Stop-ScheduledTask -TaskName "RagtimeRemote-$PairingId" -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $env:LOCALAPPDATA "RagtimeRemote\$PairingId") -Force -Recurse -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $env:ProgramData "RagtimeRemote\$PairingId") -Force -Recurse -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName "RagtimeRemote-$PairingId" -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "Removed only Ragtime-owned configuration for pairing $PairingId. sshd was left unchanged."
}

$identity = Get-OriginalIdentity $TargetUser
if ($Uninstall) { Remove-OwnedInstall $Uninstall $identity; exit 0 }
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
