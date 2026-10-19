"""MLForge Core Exceptions.

Explicit domain-specific exceptions for dataset validation, configuration errors,
and data leakage detection.
"""

class MLForgeError(Exception):
    """Base exception for all MLForge errors."""
    pass


class DatasetValidationError(MLForgeError):
    """Raised when an input dataset fails schema, type, or integrity validation."""
    pass


class InsufficientClassInstancesError(DatasetValidationError):
    """Raised when one or more classes have fewer instances than the required fold count."""
    def __init__(self, class_label: str, count: int, required_folds: int):
        self.class_label = str(class_label)
        self.count = count
        self.required_folds = required_folds
        message = (
            f"Class '{self.class_label}' has {self.count} instance(s), which is fewer "
            f"than the configured {self.required_folds} cross-validation folds. "
            f"Stratified cross-validation requires at least {self.required_folds} samples per class."
        )
        super().__init__(message)


class DataLeakageError(MLForgeError):
    """Raised when data leakage or illegal preprocessing across splits is detected."""
    pass


class ConfigurationError(MLForgeError):
    """Raised when evaluation parameters or model configurations are invalid."""
    pass


class IncompatibleEvaluationPairError(MLForgeError):
    """Raised when two model evaluations cannot be paired due to incompatible datasets, splits, or folds."""
    pass


class AcceptanceError(MLForgeError):
    """Raised when an acceptance policy is invalid or cannot be parsed."""
    pass


class ArtifactError(MLForgeError):
    """Base exception for artifact serialization and loading errors."""
    pass


class ArtifactSerializationError(ArtifactError):
    """Raised when evaluation artifacts cannot be safely serialized."""
    pass


class ArtifactLoadError(ArtifactError):
    """Raised when artifacts cannot be loaded or fail schema/integrity validation."""
    pass


class ArtifactSecurityError(ArtifactError):
    """Raised when an untrusted or insecure artifact operation is attempted."""
    pass


class AuditBundleError(ArtifactError):
    """Base exception for audit bundle packaging, extraction, and verification errors."""
    pass


class AuditBundleCorruptError(AuditBundleError):
    """Raised when an audit bundle archive is malformed, truncated, or unreadable as a ZIP."""
    pass


class AuditBundleSecurityError(AuditBundleError):
    """Raised when an audit bundle contains unsafe paths, path traversal, or duplicate entries."""
    pass


class AuditBundleTamperError(AuditBundleError):
    """Raised when an audit bundle member fails checksum verification or contains unauthorized files."""
    pass


class ReleaseGateError(MLForgeError):
    """Base exception for release-gate evaluation failures or configuration issues."""
    pass


class CertificateError(MLForgeError):
    """Base exception for certificate issuance and signing errors."""
    pass


class CertificateSecurityError(CertificateError):
    """Raised when certificate signing or verification encounters security violations."""
    pass


class CertificateVerificationError(CertificateError):
    """Raised when certificate verification detects signature or integrity tampering."""
    pass


class PersistenceError(MLForgeError):
    """Base exception for SQLite persistence operations."""
    pass


class DuplicateRunError(PersistenceError):
    """Raised when attempting to create a run record with an existing run_id."""
    pass


class RunNotFoundError(PersistenceError):
    """Raised when a requested run_id is not found in the persistence store."""
    pass


class LifecycleTransitionError(PersistenceError):
    """Raised when an illegal run lifecycle state transition is attempted."""
    pass
