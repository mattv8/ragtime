# SolidWorks essentials

Every call goes to this tool with `operation` and optional `arguments`, for example
`{"operation": "features.list", "arguments": {"document": "doc-1"}}`. `timeout` (seconds) is optional.
Call `status` first to see what this tool allows, then `describe` for argument schemas. Lengths are
metres, masses kilograms, angles radians. Convert for the user and quote the unit.

## Load the matching skill before you act

| Task | Skill |
|---|---|
| Find a file or a model | `solidworks-find-model` |
| Broken mates, missing parts, rebuild errors | `solidworks-repair-assembly` |
| Build or change a model | `solidworks-scripts` |
| Vault state, get latest, check-out, check-in | `solidworks-pdm` |

If a skill named here is not listed, that capability is off. Check `status` (its `ragtime` object)
and tell the user.

## When a call is refused

Read `error.code`. For a capability that is switched off, read `error.details.admin_action`
(what an admin can change) and `error.details.still_available` (what still works). Tell the user
both, then do what is still available. Do not retry a refusal in a loop.
