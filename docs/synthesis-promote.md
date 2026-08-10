# Synthesis promote (groups, membership, artifacts)

Operator notes for air-gapped and on-prem installs that use tool synthesis.
**Promote** means pointing a group's ``active.json`` at a staged artifact —
this is not Nomad/container deploy (`cat-agent deploy`).

## Membership file

Group membership is **not** under the writable workspace. Default path:

```
/etc/cat-agent/groups.json
```

Override for tests or non-Unix hosts with `CAT_AGENT_MEMBERSHIP_PATH` or
`cat-agent synth … --membership /path/to/groups.json`.

Expected permissions:

- owned by root (or an operator account not shared with end users)
- mode `0644` (readable by the agent process; **not** writable by group or other)

The library refuses to load a membership file that is group- or world-writable.
Do not place `groups.json` inside `<workspace>/` — that tree must be writable for
artifact staging, and a user with shell access could otherwise grant themselves
any group.

Roleful format (preferred):

```json
{
  "builder":  {"finance": ["member"]},
  "approver": {"finance": ["member", "promoter"]},
  "lead":     {"finance": ["member", "promoter", "sharer"]}
}
```

| Role | May |
|------|-----|
| `member` | synthesise into staging (`synth run`) |
| `promoter` | `promote` / `demote` / `gc` / `migrate` within the group |
| `sharer` | `share` / `unshare` / `adopt` across groups |

Roles are per group. Legacy flat lists (`{"alice": ["finance"]}`) still load as
`["member"]` and log a one-time migration warning.

## Content-addressed layout

```
<workspace>/groups/<group>/artifacts/<tool_name>/<impl_sha256[:12]>/
<workspace>/groups/<group>/active.json          # {tool_name: version} and adopted keys
<workspace>/groups/<group>/staging.json         # {tool_name: version}
<workspace>/groups/<group>/shares.json          # publisher offers
<workspace>/groups/<group>/adoptions.json       # consumer decisions
<workspace>/groups/<group>/settings.json        # e.g. {"auto_adopt_org_tools": false}
```

- Artifact directories are immutable; re-synthesis creates a new version directory.
- `promote` / `demote` only change pointer files (`active.json`).
- Rollback: `cat-agent synth promote <tool> --version <sha12>`.
- Garbage collection: `cat-agent synth gc --group G --keep N` (never removes a
  version referenced by this or any other group's `active.json`).
- Migrate legacy flat `staging/` + `active/` dirs:
  `cat-agent synth migrate --group G`.

Legacy flat `generated_tools/` is never auto-migrated; operators choose which
group owns those tools.

## Cross-group sharing (share → adopt)

Sharing is two-sided. The publisher offers; the consumer pins a version.

```bash
cat-agent synth share   validate_iban --with ops,legal   # sharer in finance
cat-agent synth adopt   finance/validate_iban --version <sha12>  # sharer in ops
cat-agent synth unshare validate_iban --with ops --reason 'bug'
```

- Adopted tools keep the owner-qualified registry name (`generated_finance_…`).
- Consumers hold a pointer only — artifacts stay in the owning group.
- Adoption always pins a content hash; publisher re-promote does not move it.
- `--with org` offers to every group without auto-install. Set
  `auto_adopt_org_tools: true` in that group's `settings.json` (default false)
  to auto-adopt org-shared tools at load time.
- After `unshare`, the next consumer `load_generated_tools` fails loudly with
  tool, owning group, and recorded reason.

## Runtime enable / demote

Generated tools load into the optional registry. After promote (or adopt), call
`enable_optional_tools(...)` before agents use them. `demote` disables the tool
in-process via `disable_tools`; if unload fails, the CLI prints that a
**restart is required**.

Agent construction should pass `principal=` (and `workspace=` when using
adoptions) and resolve tools with `tools_for_principal(principal, workspace=…)`.

Runnable offline demos live under
[`examples/synthesis/promote/`](../examples/synthesis/promote/).
