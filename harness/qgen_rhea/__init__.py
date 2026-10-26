"""Template question generator: questions with gold SPARQL and gold answers.

``templates`` holds the seven question templates, ``sampling`` and ``split``
the deterministic selection and the dev/test split, and ``generate`` runs the
templates against a store and writes ``questions.jsonl`` and ``manifest.json``.
"""

from harness.qgen_rhea.base import AnswerColumn, Candidate, QuestionTemplate, Variant
from harness.qgen_rhea.templates import TEMPLATES, TEMPLATES_BY_ID

__all__ = ["AnswerColumn", "Candidate", "QuestionTemplate", "TEMPLATES", "TEMPLATES_BY_ID", "Variant"]
