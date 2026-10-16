# Skills

Agent-facing skills for this repository live here, one directory per skill with a `SKILL.md`
carrying YAML front matter (`name`, `description`). They are kept in the repo rather than only in an
operator's `~/.claude/skills/` so that every agent working with BindPnyx gets them, and so that a
correction lands once instead of on one machine.

An operator exposes them to a local agent by symlinking or copying a skill directory into
`~/.claude/skills/<name>`, and should prefer a symlink so the repo stays the single source of truth.

| skill | what it is for |
|---|---|
| `arcrefine-binder-refinement` | refining an existing binder with ArcRefine (`structural-carryover`) without losing its binding mode, and the measured reasons to distrust the result |
| `multi-agent-bus-coordination` | coordinating several agents over a shared filesystem with `scripts/agent_bus.py`, and the measured reason posting is not the same as being heard |

Measured claims in a skill carry their n and the target they were measured on. Private campaign data
- per-run score logs, candidate sequences, operator install paths - stays out of this repository; a
skill may say that such a record exists and what it showed, not reproduce it.
