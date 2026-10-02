"""Errors exposed by the 3D MCP server."""


class Codex3DError(RuntimeError):
    """Base class for expected, user-actionable failures."""

    code = "CODEX_3D_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class InvalidImageError(Codex3DError):
    code = "INVALID_IMAGE"


class InvalidPathError(Codex3DError):
    code = "INVALID_PATH"


class JobNotFoundError(Codex3DError):
    code = "JOB_NOT_FOUND"


class ModelUnavailableError(Codex3DError):
    code = "MODEL_UNAVAILABLE"


class GenerationCancelled(Codex3DError):
    code = "CANCELLED"
