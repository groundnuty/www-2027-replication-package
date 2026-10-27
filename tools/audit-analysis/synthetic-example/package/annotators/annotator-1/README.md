# Audit annotation package (annotator-1)

You have 24 items (Form A: 10, Form B: 8, Form C: 6), in your own order in `items.md` (the same items, every row of
each answer, in `items.jsonl`). Code each item in a copy of `sheet.csv`, one row per item: fill the columns of the
item's form and leave the "-" columns alone. Code on your own: don't discuss items with the other annotator until both
sheets are done. Disagreements are settled afterwards by discussion.

## What each item shows

- the question, as the agent was given it;
- the gold query and the gold answer;
- the agent's final query and answer. The final query is the last query the agent ran whose result contained its
  submitted answer under the registered selection rule; where none did, or the agent ran no query, the item says so.
- **Form B** also shows the message a checker produced for that query.
- **Form C** also shows the ontology text that was in the agent's prompt, and asks you to quote, as the witness, the
  statement the wrong answer follows from.

The items are shuffled. Each item has an id (such as `T001`); use it to find the item's row in
the sheet.

## Codebook (one primary code per field)

**Form A**
1. **Error type** (`error_type`):
   - `aggregation`: count against count-distinct, min against max, a wrong grouping;
   - `constant`: right type, wrong entity;
   - `filter_scope`: a condition missing or extra, or the wrong graph or scope;
   - `path`: the wrong relation path, or a missing or extra hop;
   - `shape`: wrong or missing columns, duplicates;
   - `gold_or_question`: the gold is wrong or the question is ambiguous;
   - `scorer`: the answer is equivalent but scored wrong;
   - `other`, with a short note (`error_type_note`).
2. **Structurally detectable?** (`structurally_detectable`: `yes`, `no` or `unsure`): could a vocabulary, class,
   domain/range or disjointness check over a **correct** schema flag this query in principle?

**Form B**
1. **Cause** (`cause`):
   - `undeclared_term`: the schema lacks a term the data uses;
   - `closed_world`: a membership the data doesn't assert;
   - `data_clash`: the data logically contradicts the schema;
   - `schema_too_narrow`: the axiom contradicts sensible use;
   - `checker_defect`: a rule misapplied;
   - `other`, with a note (`cause_note`).
2. **Was the agent's answer really correct?** (`answer_really_correct`: `yes`, `no` or `unsure`). This catches scorer
   errors.

**Form C**
1. **Does the wrong answer follow from a statement in the ontology shown in the prompt?** (`follows_from_ontology`:
   `yes`, `no` or `unsure`). If `yes`, quote the statement (`witness`); a `yes` without a quoted witness is coded
   `unsure`.
2. **If yes, is that statement wrong or misleading for this data?** (`statement_wrong_or_misleading`: `yes`, `no` or
   `unsure`)

## The sheet's columns

### Form A: an answer scored wrong

- **`error_type`**: Error type (`aggregation`, `constant`, `filter_scope`, `path`, `shape`, `gold_or_question`, `scorer`, `other`); `other` needs a short note in `error_type_note`
- **`structurally_detectable`**: Structurally detectable? (`yes`, `no`, `unsure`)

### Form B: an answer scored correct, with a checker's message

- **`cause`**: Cause (`undeclared_term`, `closed_world`, `data_clash`, `schema_too_narrow`, `checker_defect`, `other`); `other` needs a short note in `cause_note`
- **`answer_really_correct`**: Was the agent's answer really correct? (`yes`, `no`, `unsure`)

### Form C: an answer scored wrong, with the ontology text from the agent's prompt

- **`follows_from_ontology`**: Does the wrong answer follow from a statement in the ontology shown in the prompt? (`yes`, `no`, `unsure`)
- **`witness`**: The quoted statement (the witness) (free text)
- **`statement_wrong_or_misleading`**: If yes, is that statement wrong or misleading for this data? (`yes`, `no`, `unsure`)

`comment` is free text for anything else.
