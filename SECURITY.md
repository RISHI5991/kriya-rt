# Security Policy

## Supported Versions

Security updates and critical bug fixes are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

---

## Reporting a Vulnerability

The KriyaRT engineering team takes the security of our runtime and users seriously.

If you believe you have discovered a security vulnerability in KriyaRT (including memory safety violations, out-of-bounds GPU buffer accesses, denial-of-service in the HTTP server, or malicious checkpoint ingestion exploits):

1. **Do NOT open a public GitHub issue.**
2. Send a detailed report via email to **[security@kriya-rt.dev](mailto:security@kriya-rt.dev)** or submit a private security advisory through GitHub's [Advisories](https://github.com/RISHI5991/kriya-rt/security/advisories) interface.
3. Include the following details where possible:
   - Steps to reproduce the issue.
   - Operating system and macOS version (`sw_vers`).
   - Hardware configuration (M1, M2, M3, M4, RAM size).
   - Sample inputs, prompts, or checkpoint manifests.
   - Any crash logs or stack traces (`RUST_BACKTRACE=1`).

---

## Response Timeline

* **Initial Acknowledgement:** Within 48 hours of receipt.
* **Assessment & Severity Determination:** Within 5 business days.
* **Remediation & Patch Release:** Coordinated with the reporter before public disclosure.

---

## Core Security & Isolation Principles

KriyaRT is architected with fail-closed security invariants:
* **Fail-Closed Tensor Validation:** All tensor shapes, byte offsets, and data types are verified prior to execution. Missing or corrupted weights immediately abort execution rather than falling back to uninitialized memory.
* **Bounded Unified Memory Arenas:** GPU buffers are bounded at startup to prevent buffer overflows across the Apple Silicon Unified Memory bus.
* **No Dynamic Code Execution:** Unlike Python-based inference frameworks, KriyaRT never executes `eval()` or unvetted python serialization payloads (`pickle`, `torch.load`). Only raw safetensors arrays and JSON configuration descriptors are parsed.
