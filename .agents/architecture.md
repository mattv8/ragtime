# Ragtime development entry point

Scope: any change to Ragtime's own code, tests, migrations, Docker setup or docs.
Read this first. It routes to the maintained guides instead of repeating them.

- Architecture, developer workflow, Prisma and migrations, security guardrails
  and house style: `.github/instructions/ragtime.instructions.md`. GitHub Copilot
  loads that folder automatically; other agents read it from here.
- Path-scoped guides, matched by the `applyTo` glob at the top of each file:

  | Files you change | Guide |
  |---|---|
  | `ragtime/frontend/**` | `.github/instructions/frontend.instructions.md` |
  | `tests/**` | `.github/instructions/testing.instructions.md` |
  | `ragtime/userspace/**`, `runtime/**` | `.github/instructions/userspace-spec.instructions.md` |
  | Any; start here for runtime failures | `.github/instructions/debugging.instructions.md` |

- The other notes in `.agents/` each cover one domain. Read every note whose
  Scope matches your change.
- Agents that use Ragtime's tools, rather than change its code, get their
  guidance from the `/mcp` server; see `mcp-byoh-vs-custom-routes.md`.
