---
name: scientific-validation
description: Use for changes to numerical methods, statistical assumptions, validation targets, scientific figures, or interpretation of scientific results.
---

# Scientific Validation

- Identify the authority source: protocol, paper, supplement, `docs/SPEC.md`, `docs/DECISIONS.md`, or tests.
- Preserve raw/reference data and write derived outputs separately.
- When numerical behavior or scientific assumptions change, verify affected invariants, reference examples, edge cases, and units; add or update tests where coverage is missing.
- Compare against known targets when available; document acceptable tolerance and why it is appropriate.
- For figures, optimize for the comparison the reader should make and keep labels readable at manuscript size.
- Record any assumption or divergence from the source in `docs/DECISIONS.md` or an ADR.
