# SSH file transfer tool

`ssh_transfer` copies files using saved SSH connections. It is available to chat
and MCP callers with access to an SSH tool. File contents travel through Ragtime;
the source and destination servers do not need credentials for each other.

If a visible SSH connection is named `transfer` (whose existing shell tool is
already `ssh_transfer`), Ragtime preserves that shell tool and omits the file
transfer tool to avoid duplicate names. Rename that connection to enable both.

## Endpoints

| Endpoint | Meaning |
| --- | --- |
| `ssh://<name>/absolute/path` | A configured SSH connection visible to the caller. Use its normalized name, without the shell tool's `ssh_` prefix. |
| `inline` | Content supplied in the request or returned in the result. |
| `workspace:/relative/path` | A file in an authorized User Space workspace. |

At least one endpoint must use SSH. An ambiguous connection name is rejected.
Workspace endpoints require an authenticated user and workspace access; a
route password or client-credentials token alone does not grant workspace access.

## Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `source` | Required | Source endpoint. |
| `destination` | Required | Destination endpoint. |
| `content` | None | File content when the source is `inline`. |
| `encoding` | `text` | `text` (UTF-8) or `base64` for inline content. |
| `overwrite` | `false` | Permit replacement of an existing destination file. |
| `recursive` | `false` | Copy an SSH directory to another SSH directory. |
| `reason` | `SSH file transfer` | Description of the operation. |
| `timeout` | 300 | Requested time budget in seconds, from 1 to 300; each endpoint's configured ceiling still applies. |
| `workspace_id` | None | Workspace identifier; chat uses its active workspace. |

## Constraints

- Destination SSH connections require the existing **allow write** setting.
- Connection-specific content-protection policies apply to the transfer.
  Review subagents cannot use the tool, and scoped workers' workspace writes
  must stay inside their declared file scope.
- Each SSH path must stay inside its configured working directory, when set.
  Paths containing `..` or symlink components are rejected, including symlinks
  in the configured working directory itself or its ancestors.
  Working directories must be absolute paths; shell variables and `~` are not
  expanded by SFTP.
- Transfers copy files; they never delete the source.
- SSH-to-SSH files are limited to 50 MiB each. Recursive copies visit at most
  500 entries, including directories and skipped entries.
- Inline and workspace files are limited to 1 MiB. Workspace transfers support
  single UTF-8 text files, not binary files or directories.
- Recursive copies skip symlinks and special files. Files copied before a later
  failure remain at the destination.
- Recursive copies merge into the destination directory. Existing files fail
  individually unless `overwrite` is enabled. Regular-file permissions and
  modification times are preserved where supported; directory permissions and
  special permission bits are not preserved.
- Recursive copies cannot target the source directory or a directory inside it
  or one of its ancestors on the same configured SSH host and port. Different
  DNS names for the same host cannot be reliably identified as the same endpoint.
- New inline/workspace uploads use mode `0600`. Overwriting an existing file
  from those sources preserves its ordinary permission bits. SSH sources use
  the source file's ordinary mode and modification time where supported.
- Transfers use the SSH account's SFTP permissions. Shell command prefixes,
  including `sudo`, do not apply.
- Connections reuse the existing [SSH host-key policy](README.md#ssh-connections).
  Cancellation interrupts active connections; operating-system hostname
  resolution can outlast the requested timeout.
- Overwrite requires the server's POSIX rename extension. Default no-clobber
  publication relies on the server honoring standard SFTP rename semantics.
  Path checks cannot eliminate concurrent changes by other remote processes.
- A lost connection during publication can leave the outcome uncertain; check
  the destination before retrying. Failed cleanup can leave a hidden
  `.ragtime-transfer-*` staging directory alongside the destination. Remove it
  only after confirming the transfer is no longer running.

## Examples

These examples assume visible SSH connections named `web01` and `archive`.

Copy a server file:

```json
{
  "source": "ssh://web01/var/log/app.log",
  "destination": "ssh://archive/backups/app.log"
}
```

Upload UTF-8 content:

```json
{
  "source": "inline",
  "destination": "ssh://web01/srv/app/message.txt",
  "content": "Hello\n"
}
```

Download binary content as base64:

```json
{
  "source": "ssh://web01/srv/app/icon.png",
  "destination": "inline",
  "encoding": "base64"
}
```

Results include `status` (`ok`, `rejected`, or `transfer_failed`), byte and file
counts, errors, and skipped entries. Inline downloads also include `content`
and `encoding`. A failure can represent a partial recursive copy; inspect the
counts and errors before retrying.
