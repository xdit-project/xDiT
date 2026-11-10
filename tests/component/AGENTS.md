## Specification list

## Document handling rules

**Definitions** for this section:

- "code" means not only code itself, but also a comment, or a docstring. In certain contexts it also means a file that contains the code.
- "document" - any markdown (`*.md`) file found inside of this directory tree, excluding `AGENTS.md`.
- "applicable document" - a `README.md` file in the same directory as the code in question (when its content applies to the code), or any document in the specification (see the list of specifications above) that covers the functionality of the code in question.

**Rules**:

Modifications to "documents" ARE PROTECTED and require EXPLICIT UNAMBIGUOUS sanction from the user for each modification on per-message basis.

"Documents" must stay synchronized with the "code" they refer to, or that implements them, and vice-versa: "code" must stay synchronized with its "applicable document". To achieve that:

1. Any new code or modification of a "code" must be coherent with its "applicable document". If you need to make a new "code" or "code" modification that contradicts to its "applicable document", you MUST inform the user and ASK what should you do: modify the specific "applicable document", or change the "code", so it adheres to the "applicable document". It doesn't matter if the user directly requested the modification, or you need it to serve the user's task, you MUST ASK first, before editing the "code" or the "applicable document".

2. If you notice a "code" that contradicts to what its "applicable document" say, you MUST inform the user and ASK them for guidance on how to resolve the found discrepancy, and wait for the answer.



