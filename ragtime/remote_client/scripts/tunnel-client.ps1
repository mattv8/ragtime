param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateNotNullOrEmpty()]
    [string]$ConfigPath
)

$ErrorActionPreference = 'Stop'
if ([Net.ServicePointManager]::SecurityProtocol -ne [Net.SecurityProtocolType]::SystemDefault) {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
}
$Open = 1; $Data = 2; $Close = 3; $Ping = 4; $Pong = 5
$MaxData = 65536

function Get-Config {
    param([string]$Path)
    $config = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    foreach ($name in @('server_url', 'pairing_id', 'credential', 'sshd_port')) {
        if ($null -eq $config.$name -or [string]::IsNullOrWhiteSpace([string]$config.$name)) { throw "Missing $name in config" }
    }
    if ([int]$config.sshd_port -lt 1 -or [int]$config.sshd_port -gt 65535) { throw 'sshd_port must be between 1 and 65535' }
    if ($config.ca_file) { throw 'Custom CA files require installation in the Windows trusted root store before using the PowerShell client.' }
    return $config
}

function New-Frame {
    param([byte]$Type, [uint32]$StreamId, [byte[]]$Payload = @())
    if ($Type -eq $Data -and $Payload.Length -gt $MaxData) { throw 'Tunnel payload exceeds 64 KiB' }
    $frame = New-Object byte[] (5 + $Payload.Length)
    $frame[0] = $Type
    [Array]::Copy([BitConverter]::GetBytes([Net.IPAddress]::HostToNetworkOrder([int]$StreamId)), 0, $frame, 1, 4)
    [Array]::Copy($Payload, 0, $frame, 5, $Payload.Length)
    return $frame
}

function Send-Frame {
    param([Net.WebSockets.ClientWebSocket]$WebSocket, [byte[]]$Frame, [Threading.CancellationToken]$Token)
    [void]$WebSocket.SendAsync([ArraySegment[byte]]::new($Frame), [Net.WebSockets.WebSocketMessageType]::Binary, $true, $Token).GetAwaiter().GetResult()
}

function Start-StreamRead {
    param([Net.Sockets.TcpClient]$Client, [Threading.CancellationToken]$Token)
    $buffer = New-Object byte[] $MaxData
    return [PSCustomObject]@{
        Buffer = $buffer
        Task = $Client.GetStream().ReadAsync($buffer, 0, $buffer.Length, $Token)
    }
}

$config = Get-Config $ConfigPath
$serverUri = [Uri]$config.server_url
if ($serverUri.Scheme -ne 'wss') { throw 'server_url must use wss://' }
$delay = 1
while ($true) {
    $socket = [Net.WebSockets.ClientWebSocket]::new()
    $streams = @{}
    $cts = [Threading.CancellationTokenSource]::new()
    $connectedAt = [DateTime]::MinValue
    try {
        $socket.Options.AddSubProtocol('ragtime-tunnel-v1')
        $socket.Options.SetRequestHeader('Authorization', "Bearer $($config.credential)")
        $socket.Options.SetRequestHeader('X-Ragtime-Pairing-ID', [string]$config.pairing_id)
        $socket.ConnectAsync($serverUri, $cts.Token).GetAwaiter().GetResult()
        if ($socket.SubProtocol -ne 'ragtime-tunnel-v1') { throw 'Server did not select ragtime-tunnel-v1' }
        $connectedAt = [DateTime]::UtcNow
        $lastReceived = $connectedAt
        $receiveBuffer = New-Object byte[] 65541
        $receiveMessage = New-Object IO.MemoryStream
        $receiveTask = $socket.ReceiveAsync([ArraySegment[byte]]::new($receiveBuffer), $cts.Token)
        while ($socket.State -eq [Net.WebSockets.WebSocketState]::Open) {
            $waits = @($receiveTask) + @($streams.Values | ForEach-Object {
                if (-not $_.Connected) { $_.ConnectTask }
                else {
                    if (-not $_.LocalEof) { $_.Read.Task }
                    if ($null -ne $_.WriteTask) { $_.WriteTask }
                }
            })
            $queuedWriteReady = @($streams.Values | Where-Object {
                $_.Connected -and $null -eq $_.WriteTask -and $_.Writes.Count -gt 0
            }).Count -gt 0
            if (-not $queuedWriteReady) {
                [Threading.Tasks.Task]::WaitAny([Threading.Tasks.Task[]]$waits, 1000) | Out-Null
            }
            foreach ($streamId in @($streams.Keys)) {
                $entry = $streams[$streamId]
                if (-not $entry.Connected) {
                    if (-not $entry.ConnectTask.IsCompleted) { continue }
                    try { $entry.ConnectTask.GetAwaiter().GetResult() } catch {
                        $entry.Client.Close(); $streams.Remove($streamId); Send-Frame $socket (New-Frame $Close $streamId) $cts.Token; continue
                    }
                    $entry.Connected = $true; $entry.Read = Start-StreamRead $entry.Client $cts.Token
                }
                if ($null -ne $entry.WriteTask -and $entry.WriteTask.IsCompleted) {
                    try { $entry.WriteTask.GetAwaiter().GetResult() } catch {
                        $entry.Client.Close(); $streams.Remove($streamId); Send-Frame $socket (New-Frame $Close $streamId) $cts.Token; continue
                    }
                    $entry.WriteTask = $null
                }
                if ($entry.RemoteClosed -and $entry.Writes.Count -eq 0 -and $null -eq $entry.WriteTask) {
                    try { $entry.Client.Client.Shutdown([Net.Sockets.SocketShutdown]::Send) } catch [Net.Sockets.SocketException] { }
                }
                if ($null -eq $entry.WriteTask -and $entry.Writes.Count -gt 0) {
                    $next = $entry.Writes.Dequeue(); $entry.PendingBytes -= $next.Length
                    $entry.WriteTask = $entry.Client.GetStream().WriteAsync($next, 0, $next.Length, $cts.Token)
                }
                if ($entry.LocalEof -or -not $entry.Read.Task.IsCompleted) { continue }
                try { $count = $entry.Read.Task.GetAwaiter().GetResult() } catch {
                    $entry.Client.Close(); $streams.Remove($streamId)
                    Send-Frame $socket (New-Frame $Close $streamId) $cts.Token
                    continue
                }
                if ($count -eq 0) {
                    $entry.LocalEof = $true
                    $entry.Read = $null
                    Send-Frame $socket (New-Frame $Close $streamId) $cts.Token
                    if ($entry.RemoteClosed) { $entry.Client.Close(); $streams.Remove($streamId) }
                } else {
                    $payload = New-Object byte[] $count
                    [Array]::Copy($entry.Read.Buffer, 0, $payload, 0, $count)
                    Send-Frame $socket (New-Frame $Data $streamId $payload) $cts.Token
                    $entry.Read = Start-StreamRead $entry.Client $cts.Token
                }
            }
            if (-not $receiveTask.IsCompleted) {
                if (([DateTime]::UtcNow - $lastReceived).TotalSeconds -gt 75) { throw 'WebSocket peer was silent for 75 seconds' }
                continue
            }
            $result = $receiveTask.GetAwaiter().GetResult()
            $lastReceived = [DateTime]::UtcNow
            if ($result.MessageType -eq [Net.WebSockets.WebSocketMessageType]::Close) { throw 'WebSocket closed' }
            if ($result.MessageType -ne [Net.WebSockets.WebSocketMessageType]::Binary) { throw 'Tunnel requires binary WebSocket messages' }
            if ($receiveMessage.Length + $result.Count -gt 1048576) { throw 'WebSocket message too large' }
            $receiveMessage.Write($receiveBuffer, 0, $result.Count)
            $receiveTask = $socket.ReceiveAsync([ArraySegment[byte]]::new($receiveBuffer), $cts.Token)
            if (-not $result.EndOfMessage) { continue }
            $message = $receiveMessage.ToArray(); $receiveMessage.SetLength(0)
            if ($message.Length -lt 5) { throw 'Short tunnel frame' }
            $type = $message[0]
            $streamId = [uint32](($message[1] -shl 24) -bor ($message[2] -shl 16) -bor ($message[3] -shl 8) -bor $message[4])
            if ($type -eq $Ping) { Send-Frame $socket (New-Frame $Pong 0) $cts.Token; continue }
            if ($type -eq $Pong) { continue }
            if ($type -eq $Open) {
                if ($streams.Count -ge 32 -or $streams.ContainsKey($streamId)) { Send-Frame $socket (New-Frame $Close $streamId) $cts.Token; continue }
                $client = [Net.Sockets.TcpClient]::new()
                $streams[$streamId] = [PSCustomObject]@{ Client = $client; ConnectTask = $client.ConnectAsync('127.0.0.1', [int]$config.sshd_port); Connected = $false; Read = $null; WriteTask = $null; Writes = [Collections.Generic.Queue[byte[]]]::new(); PendingBytes = 0; RemoteClosed = $false; LocalEof = $false }
                continue
            }
            if (-not $streams.ContainsKey($streamId)) { continue }
            if ($type -eq $Data) {
                $payloadLength = $message.Length - 5
                if ($payloadLength -gt $MaxData) { throw 'Tunnel payload exceeds 64 KiB' }
                if ($streams[$streamId].RemoteClosed) { throw 'Data received after directional close' }
                if ($streams[$streamId].PendingBytes + $payloadLength -gt 1048576) {
                    $streams[$streamId].Client.Close(); $streams.Remove($streamId)
                    Send-Frame $socket (New-Frame $Close $streamId) $cts.Token
                }
                else {
                    $copy = New-Object byte[] $payloadLength
                    [Array]::Copy($message, 5, $copy, 0, $payloadLength)
                    $streams[$streamId].Writes.Enqueue($copy); $streams[$streamId].PendingBytes += $payloadLength
                }
            } elseif ($type -eq $Close) {
                $streams[$streamId].RemoteClosed = $true
                if ($streams[$streamId].Writes.Count -eq 0 -and $null -eq $streams[$streamId].WriteTask) {
                    try { $streams[$streamId].Client.Client.Shutdown([Net.Sockets.SocketShutdown]::Send) } catch [Net.Sockets.SocketException] { }
                }
                if ($streams[$streamId].LocalEof) { $streams[$streamId].Client.Close(); $streams.Remove($streamId) }
            }
        }
    } catch {
        [Console]::Error.WriteLine("Ragtime tunnel disconnected: $($_.Exception.Message)")
    } finally {
        foreach ($entry in @($streams.Values)) { $entry.Client.Close() }
        $cts.Cancel(); $socket.Dispose(); $cts.Dispose()
    }
    if ($connectedAt -ne [DateTime]::MinValue -and ([DateTime]::UtcNow - $connectedAt).TotalSeconds -ge 60) { $delay = 1 }
    Start-Sleep -Seconds ([Math]::Min(60, $delay) * (0.8 + (Get-Random -Minimum 0 -Maximum 400) / 1000))
    $delay = [Math]::Min(60, $delay * 2)
}
