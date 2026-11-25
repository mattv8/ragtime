# Ragtime development entry point

Scope: any change to Ragtime's own code, tests, migrations, Docker setup or docs.
Read this first. It routes to the maintained guides instead of repeating them.

- Architecture, developer workflow, Prisma and migrations, security guardrails
  and house style: `.github/instructions/ragtime.instructions.md`. GitHub Copilot
  in VS Code applies each file in that folder to the files its `applyTo` glob
  matches; other agents read them from here.
- Path-scoped guides, matched by the `applyTo` glob at the top of each file:

  | Files you change | Guide |
  |---|---|
  | `ragtime/frontend/**` | `.github/instructions/frontend.instructions.md` |
  | `tests/**` | `.github/instructions/testing.instructions.md` |
  | `ragtime/userspace/**`, `runtime/**` | `.github/instructions/userspace-spec.instructions.md` |
  | Any; start here for runtime failures | `.github/instructions/debugging.instructions.md` |

- The other notes in `.agents/` each cover one domain, named in the note's
  title or Scope line. Read every note that covers your change. For frontend
  work that includes `ragtime-ux-style-guide.md`.
- Agents that use Ragtime's tools, rather than change its code, get their
  guidance from the `/mcp` server; see `mcp-byoh-vs-custom-routes.md`.
