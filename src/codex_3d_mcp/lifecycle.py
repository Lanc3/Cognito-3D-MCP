"""Serialize job mutations against worker transitions and shutdown."""

from functools import wraps

from .errors import Codex3DError


def manager_operation(method):
    """Keep submitted inputs and state changes atomic with respect to close()."""

    @wraps(method)
    def operation(self, *args, **kwargs):
        with self._lock:
            if self._closing:
                raise Codex3DError("The generation manager is shutting down.", code="SERVER_CLOSED")
            return method(self, *args, **kwargs)

    return operation
