"""MLForge Application Layer.

Contains artifact serialization and lifecycle management.
"""

from mlforge.application.artifacts import (
    CURRENT_SCHEMA_VERSION,
    load_fitted_model,
    load_run_artifacts,
    serialize_run_artifacts,
)
from mlforge.application.packaging import (
    CURRENT_BUNDLE_SCHEMA_VERSION,
    build_audit_bundle,
    extract_audit_bundle,
    verify_audit_bundle,
)
from mlforge.application.release_certificates import (
    CERTIFICATE_SCHEMA_VERSION,
    canonicalize_certificate_payload,
    deserialize_public_key,
    generate_release_keypair,
    issue_release_certificate,
    serialize_public_key,
    verify_release_certificate,
)
from mlforge.application.persistence import (
    DEFAULT_DB_PATH,
    SQLiteRunRepository,
    execute_managed_evaluation,
)
from mlforge.application.reporting import (
    REPORT_SCHEMA_VERSION,
    generate_html_report,
    load_report_evidence,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "CURRENT_BUNDLE_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "CERTIFICATE_SCHEMA_VERSION",
    "DEFAULT_DB_PATH",
    "serialize_run_artifacts",
    "load_run_artifacts",
    "load_fitted_model",
    "build_audit_bundle",
    "verify_audit_bundle",
    "extract_audit_bundle",
    "generate_html_report",
    "load_report_evidence",
    "generate_release_keypair",
    "serialize_public_key",
    "deserialize_public_key",
    "canonicalize_certificate_payload",
    "issue_release_certificate",
    "verify_release_certificate",
    "SQLiteRunRepository",
    "execute_managed_evaluation",
]
