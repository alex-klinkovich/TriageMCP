"""Domain exception hierarchy for TriageMCP."""

from __future__ import annotations


class TriageError(Exception):
    """Base class for every TriageMCP-specific error."""


class TransientLLMError(TriageError):
    """A retryable failure talking to the LLM (timeout, rate limit, connection, 5xx)."""


class AgentError(TriageError):
    """The agent could not produce a valid verdict for an alert."""


class MaxIterationsError(AgentError):
    """The agent hit its iteration cap without submitting a valid verdict."""


class ModelRefusedToSubmitError(AgentError):
    """The model ended its turn without calling submit_triage, even after a nudge."""


class AgentTimeoutError(AgentError):
    """The per-alert time budget was exceeded."""
