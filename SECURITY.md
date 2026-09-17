# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | Security fixes only |
| 0.2.x | Security fixes and enhancements |

## Reporting a vulnerability

**Do not open a public GitHub issue for security problems.**

Use GitHub's private vulnerability reporting on this repository:
**Security → Report a vulnerability**. Reports stay private while triaged,
and credit is given at the reporter's discretion.

Please include:

1. Affected version(s) and, where relevant, the affected surface (client
   adapters, PII scanning, policy engine, store backends, export, CLI).
2. A minimal reproduction or proof of concept.
3. Your assessment of impact and severity — even if uncertain.

## What is in scope

- HMAC signing or verification bypasses (`verify_record`, signature checks).
- Records that persist PII that the scanner claims to have redacted.
- Policy enforcement bypasses (a BLOCK decision that does not stop the
  provider call or is not persisted).
- Injection through policy/PII YAML configuration files.
- Path traversal or credential exposure in the CLI or export packs.

## What is explicitly out of scope

- PII scanner recall: the scanner is documented as **best-effort** and is not
  a certified DLP control. Missed detections are expected limitations, not
  vulnerabilities — unless a specific documented pattern demonstrably fails
  to fire on its documented inputs.
- Issues in an application built on top of ai-stamp that stem from that
  application's own configuration.

## Response expectations

- Acknowledgment within 5 business days.
- A triage decision and remediation plan within 30 days.
- Coordinated disclosure after a fix ships; we will agree on a timeline with
  you.

## Safe handling in this repository

Maintainers and contributors: never commit real secrets, API keys, or real
PII in tests, examples, or fixtures. CI treats warnings as errors and audits
dependencies with pip-audit on every change.
