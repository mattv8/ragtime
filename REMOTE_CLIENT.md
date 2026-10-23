# Connect a remote machine

Ragtime's remote tools use SSH to execute commands. Quick connect prepares a target machine and pairs it with Ragtime; Manual connection lets you supply an existing SSH configuration. Choose **Windows Remote** for PowerShell execution on Windows, or **SSH Shell** for a POSIX shell.

## Prerequisites

- An administrator account in Ragtime to configure tools.
- Permission to authorize an SSH key for the target account. Installing OpenSSH or a system startup task also requires administrator/root privileges on that machine.
- A working network path: Ragtime can reach the target's SSH port, or the target can reach Ragtime's HTTPS endpoint. An offline installer cannot create connectivity when neither direction works.
- For online pairing and tunnels, HTTPS certificates trusted by the target. For a private certificate authority, install its trust certificate before downloading an installer. Do not disable TLS validation.
- A single Ragtime API worker. Tunnel endpoints live in that process; multi-worker startup is rejected rather than routing SSH to a different worker's connection.

## Quick connect

1. Add an SSH Shell or Windows Remote tool and open **Quick connect** in the Connection step.
2. Select the operating system of the **target machine**, even if your browser runs on a different OS.
3. Generate and copy the installation command. Run it in a terminal on the target, under the account Ragtime should use for SSH.
4. Follow the installer's messages. The installer may need elevation to install or start OpenSSH. Existing machine-wide authentication policies still apply.
5. Wait for Ragtime to verify the connection. Check the reported hostname, account, and connection method before proceeding.
6. Set execution options and access permissions, then save the tool.

The installation command contains a short-lived enrollment credential. Treat it as a secret. Regenerate an expired command in the wizard rather than sharing a previously used command.

### Connection methods

- **Direct:** Ragtime connects to the target's SSH port. The target does not need a persistent tunnel process.
- **Tunnel:** A target-side process opens an outbound HTTPS websocket connection to Ragtime and forwards SSH traffic to its local SSH server. No additional public Ragtime port is required. The reverse proxy must permit websocket upgrades and long-lived connections.
- **Offline:** Transfer the downloaded installer and public-key instructions to the target, then paste the installer's result back into the wizard. Ragtime must still reach the target directly. A published GitHub installer is not required for this download-and-transfer workflow.

## Windows accounts and persistence

The target account controls what remote commands can access. Domain policy, OpenSSH account support, and authentication requirements remain authoritative; an installer cannot bypass them.

Windows OpenSSH commonly stores administrator keys in a shared `administrators_authorized_keys` file. When the wizard reports this configuration, the installed public key may authorize other administrator accounts too. Ragtime uses the exact enrolled account, but the shared key-file policy belongs to Windows OpenSSH.

An elevated tunnel installation can run a startup task as SYSTEM while SSH still logs in as the enrolled account. A non-elevated user-logon task only provides connectivity while that user is logged in. Authentication through enterprise proxies may require additional deployment configuration.

## Troubleshooting

### Expired enrollment

Regenerate the command in the wizard. Do not edit the token or attempt to reuse it for another machine.

### Unreachable target

Check the reported SSH port and account. Direct connections need a reachable route from Ragtime. Tunnels need outbound HTTPS access from the target, a running tunnel process, and websocket forwarding through the proxy. Reaching GitHub alone is insufficient.

### TLS trust

Provision the correct private CA trust on the target, or use a publicly trusted certificate. A certificate warning should not be solved by bypassing verification.

### Host-key mismatch

Verify whether the target was rebuilt or its SSH host keys changed. Re-pair only after confirming the identity of the intended machine. Ragtime must not silently accept a changed pinned key.

### Authentication policy

Key-plus-password or interactive authentication requirements can prevent unattended setup. Keep the policy intact and use the supported manual credentials or an appropriate account policy. Installing a public key does not override server authentication requirements.

### Tunnel offline

Check the target's service/task state and outbound HTTPS connectivity. Reconnection restores the transport; it does not replay interrupted commands. Check command outcomes before retrying operations that may have changed data.

## Remove access

Revoke the pairing or delete the tool in Ragtime to stop its managed transport. Use the installer's uninstall option on the target to remove the installation's tagged key, task/service, and local configuration. Revocation cannot remove a key file on a machine that is offline. Shared OpenSSH installations and unrelated keys must remain intact.

## Execution boundaries

Read-only validation is a guardrail, not an operating-system sandbox. A working directory is an execution location, not filesystem containment. Use the target account's operating-system permissions to enforce access boundaries.

## Platform verification

Automated unit tests do not prove Windows domain policy, macOS Remote Login permissions, or every Linux service manager works in a particular environment. Validate enrollment, a command, reconnect after reboot, and uninstall on representative machines before a broad rollout.
