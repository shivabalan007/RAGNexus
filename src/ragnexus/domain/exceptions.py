class RAGNexusException(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: str = "INTERNAL_ERROR",
    ):
        self.message = message
        self.status_code = status_code
        self.code = code
        super().__init__(message)


class ResourceNotFoundError(RAGNexusException):
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, 404, "RESOURCE_NOT_FOUND")


class ConflictError(RAGNexusException):
    def __init__(self, message: str = "Resource conflict"):
        super().__init__(message, 409, "RESOURCE_CONFLICT")


class ValidationError(RAGNexusException):
    def __init__(self, message: str = "Invalid request"):
        super().__init__(message, 422, "VALIDATION_ERROR")


class ExternalServiceError(RAGNexusException):
    def __init__(self, message: str = "External service unavailable"):
        super().__init__(message, 502, "EXTERNAL_SERVICE_ERROR")