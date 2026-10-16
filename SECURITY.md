# Security

- No secrets are hardcoded; API keys come from environment variables
  (`SPARKFORGE_OPENAI_KEY`, etc.).
- The `run` tool executes shell commands; do not expose the serving endpoint to
  untrusted networks without sandboxing.
- Report vulnerabilities privately to the maintainers.
