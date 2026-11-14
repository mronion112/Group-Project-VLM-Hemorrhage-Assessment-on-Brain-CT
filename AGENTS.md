# AI Agent Workflow & Instructions

- Read `.cursor/` and `.claude/` folders if present.
- Reference `CONTEXT.md` and root `README.md` for project architecture and domain knowledge.
- Reference `docs/INTERFACE_CONTRACT.md` for strict data schema requirements across blocks.
- Adhere strictly to the rules in `.agents/rules/`:
  - Code Style: [code-style.md](.agents/rules/code-style.md)
  - Architecture Rules: [architecture-rules.md](.agents/rules/architecture-rules.md)
  - Karpathy Guidelines: [karpathy-rules.md](.agents/rules/karpathy-rules.md)
- Never bypass tests, validation scripts, or silent contract changes.
