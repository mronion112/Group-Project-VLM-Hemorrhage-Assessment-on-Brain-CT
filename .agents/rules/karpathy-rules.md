# Karpathy Guidelines

Core behavioral guidelines to prevent LLM engineering and coding mistakes.

## 1. Think Before Coding
- **State assumptions explicitly:** If uncertain or multiple interpretations exist, surface them—do not guess silently.
- **Surface tradeoffs:** Explain resource and time costs (e.g., 35 epochs vs 15 epochs, native 512 vs 256 resolution) before executing.
- **Push back when warranted:** If an approach is over-engineered or flawed, state it clearly.

## 2. Simplicity First
- **Minimum viable code:** Implement only what is required to solve the task cleanly.
- **No speculative features:** Avoid premature abstractions or configurations that are not actively used.
- **Clarity over complexity:** If 50 lines of clean code can replace 200 lines of nested logic, refactor to the simpler solution.

## 3. Surgical Changes
- **Touch only what you must:** Edit only code relevant to the immediate objective.
- **Do not rewrite unaffected modules:** Preserve existing working interfaces, schemas, and tests.
- **Clean up artifacts:** Remove temporary debug scripts, unused imports, and transient files after verification.

## 4. Goal-Driven Execution
- **Define success criteria upfront:** Know the exact metric, validation output, or test command before writing code.
- **Verify before completion:** Run execution scripts (`validate_output.py`, test loaders) to verify assertions with evidence before declaring success.
- **Plan steps clearly:**
  1. `[Step]` -> verify: `[Check]`
  2. `[Step]` -> verify: `[Check]`
