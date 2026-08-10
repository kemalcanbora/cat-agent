"""Synthesis defence layers — threat model and measured baselines.

This system is no longer self-evident. Each layer is individually
counter-intuitive. Without this document a future contributor will "fix"
``_is_trivial_literal`` to stop treating booleans as trivial, or raise the
mutation threshold to 1.0 — both of which break working behaviour.

Anyone changing these layers should re-run the reference baselines at the
bottom and explain any movement in the PR.

## Attack classes

| Attack | Caught by |
|---|---|
| ``if x == "..."`` chain | ``check_input_equality_chain`` |
| dict lookup keyed on inputs | ``check_literal_lookup`` (AST) |
| lookup + heuristic fallback | ``check_literal_lookup`` (AST) |
| encoded / obfuscated lookup table | holdout split |
| under-generalised shape check (A4) | intake lint → insensitivity question → work stage |

## Layer limits

### Code mutation (`cat_agent/synthesis/mutation.py` — AST mutants)

Code mutation **cannot** catch under-generalisation. It explores the
neighbourhood of the implementation that was produced. A shape check such as
``len(iban) == 26 and iban[:2] == "TR"`` is a local optimum: every one-edit
neighbour fails some example, so the kill ratio is perfect (6/6). The correct
algorithm (ISO 13616 mod-97) is not a neighbour in edit distance, so mutation
never looks there.

Mutation answers *"do the examples exercise this code?"*, not *"do the
examples pin down a behaviour?"*. Do **not** raise the threshold to 1.0 to
compensate — that only rejects correct tools that have equivalent mutants
(for example appending junk after base64 ``=`` padding, which Python ignores).

### ``_is_trivial_literal`` (`cat_agent/synthesis/overfit.py`)

Booleans are treated as trivial **on purpose**. Making ``True`` / ``False``
non-trivial would flag every legitimate validator that contains those tokens.
Boolean-returning tools are covered by ``check_literal_lookup`` and the holdout
instead. The docstring on the helper says this explicitly — leave it alone.

### Holdout split (`ToolSpec.split_examples`)

The holdout is drawn from the user's own examples. It cannot be better than
they are. A weak set whose negatives are all "obviously" invalid (wrong length,
wrong prefix) will never expose a shape check. This is why intake lint exists.

### Intake lint (`cat_agent/synthesis/spec_quality.py`)

Runs before any code is generated. ``negatives_far_from_positives`` warns when
a boolean spec has no near-miss negative (edit distance ≤ 2 sharing a length
with a positive). ``unused_parameter`` is **info**, not warn: holding a
parameter constant while varying others is a common legitimate pattern, and a
noisy warn would cause ``allow_weak_spec=False`` users to disable the whole
lint.

### Input sensitivity (same ``mutation.py`` module — input axis)

Perturbs **inputs**, keeps code fixed. Only **positive** (truthy expected)
examples are probed — perturbing a negative produces variants that both a
shape check and a correct implementation reject, which dilutes the denominator
without signal.

Substitution rule for strings longer than 8 characters: leave the first 4
characters untouched (structural prefix); replace each remaining character with
every other glyph in its class (digits → 9 alternatives).

Input sensitivity is a **question**, not a rejection. Many correct tools are
legitimately insensitive to single-character edits ("is this string
non-empty"). When the user confirms the behaviour is intended, the tool ships
and ``verification.warnings_overridden`` records who was asked, what was asked,
and what they answered.

### Not yet covered: N-version differential testing

The A4 chain depends on a human answering the insensitivity question. For
community-submitted tools there is no such human, so N-version differential
testing must land before any tool marketplace opens.

## Manifest ``verification`` block (schema v2)

Written by ``artifacts.write_artifacts``. Older manifests without the key still
load (``verification`` normalises to ``null``).

```json
"verification": {
  "code_mutation": {
    "killed": 10,
    "total": 12,
    "threshold": 0.8
  },
  "input_sensitivity": [
    {"param": "iban", "changed": 0, "variants": 198},
    {"param": "iban", "changed": 0, "variants": 198}
  ],
  "spec_warnings": [
    {"code": "negatives_far_from_positives", "severity": "warn"}
  ],
  "warnings_overridden": false,
  "holdout_size": 1
}
```

The two scores are only meaningful **together**. Code mutation at 6/6 with
input sensitivity at 0/396 means the code is fully exercised by the examples
but the behaviour is not pinned down. Either number alone misleads.

## Reference baselines (re-run when changing these layers)

Fixture: IBAN validation (ISO 13616 mod-97). Implementation A4 is
``len(iban) == 26 and iban[:2] == "TR"``.

| Measurement | Weak examples | Strong examples |
|---|---|---|
| Code mutation score (A4) | **6/6 = 1.000** | **6/6 = 1.000** |
| Input sensitivity (A4, substitutions) | **0/396** | **0/657** |
| ``lint_spec`` warn | ``negatives_far_from_positives`` | none |

Per-example substitution denominators (digit body, indices ≥ 4):

| IBAN length | Positions | Count |
|---|---|---|
| 26 (TR…) | 22 × 9 | **198** |
| 22 (DE…) | 18 × 9 | **162** |
| 15 (NO…) | 11 × 9 | **99** |

Weak total: 2 × 198 = **396**.
Strong total: 396 + 162 + 99 = **657** (not 558 — that figure omits the
Norwegian positive that the strong fixture includes).
