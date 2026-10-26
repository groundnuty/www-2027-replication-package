"""Exceptions shared by the agent modules."""

from __future__ import annotations


class HarnessAbort(Exception):
    """An error the agent loop must not turn into a recorded failure (it propagates)."""


class InternalToolError(Exception):
    """A tool failed in a way that is not a tool error for the model (for example the
    SPARQL client raised something other than SparqlError). The attempt fails as
    ``other``; ``message`` is what the record stores and what replay re-raises."""

    def __init__(self, message: str, *, client_called: bool = False):
        super().__init__(message)
        self.message = message
        self.client_called = client_called
