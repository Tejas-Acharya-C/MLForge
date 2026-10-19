"""MLForge Audit Bundle Packaging & Verification Module (Milestone 4.1).

Provides portable, deterministic, tamper-evident ZIP packaging for evaluation run
artifacts and fitted models, complete with SHA-256 cryptographic manifests, atomic
writes, zip-slip path traversal defenses, and safe verification without model deserialization.

SECURITY WARNING
----------------
1. Bundle verification confirms internal cryptographic consistency with the packaged manifest.
   It does NOT prove external provenance or trustworthiness of the archive contents.
2. Model files (.joblib) inside the bundle are executable Python objects. Never deserialize
   models from untrusted bundles. Verification and extraction in this module NEVER call
   joblib.load or pickle.load.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import posixpath
from typing import Any, Dict, List, Optional, Set, Union
import zipfile

from mlforge.core.contracts import (
    AuditBundleFileEntry,
    AuditBundleManifest,
    AuditBundleVerificationResult,
)
from mlforge.core.exceptions import (
    AuditBundleCorruptError,
    AuditBundleError,
    AuditBundleSecurityError,
    AuditBundleTamperError,
)

CURRENT_BUNDLE_SCHEMA_VERSION = "1.0.0"
FIXED_DETERMINISTIC_DATETIME = (2026, 1, 1, 0, 0, 0)

# Resource limits for audit bundle verification and extraction
# Designed to accommodate tabular classification models (typically < 100 MB)
# while defending against decompression bombs, zip-slip, and memory exhaustion.
DEFAULT_MAX_ARCHIVE_SIZE_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB max zip file size
DEFAULT_MAX_MEMBER_SIZE_BYTES: int = 500 * 1024 * 1024        # 500 MB max uncompressed member
DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB max total uncompressed
DEFAULT_MAX_COMPRESSION_RATIO: float = 100.0                   # Max 100:1 ratio for members > 1MB
CHUNK_SIZE_BYTES: int = 64 * 1024                              # 64 KB streaming buffer for checksum & write


def normalize_bundle_path(entry_path: str) -> str:
    """Normalize an archive member path to canonical POSIX relative format.

    Parameters
    ----------
    entry_path : str
        Raw archive entry path.

    Returns
    -------
    str
        Normalized relative path using forward slashes.

    Raises
    ------
    AuditBundleSecurityError
        If path contains null bytes, absolute prefixes, drive specifiers, or path traversal.
    """
    if not isinstance(entry_path, str):
        raise AuditBundleSecurityError(f"Archive entry path must be a string, got {type(entry_path).__name__}.")

    if "\0" in entry_path:
        raise AuditBundleSecurityError(f"Archive entry path '{entry_path}' contains forbidden null bytes.")

    raw = entry_path.strip()
    if not raw:
        raise AuditBundleSecurityError("Archive entry path cannot be empty or pure whitespace.")

    # Reject Windows drive specifiers (e.g. C: or C:\)
    if ":" in raw:
        raise AuditBundleSecurityError(
            f"Archive entry path '{entry_path}' contains forbidden drive or protocol separator."
        )

    # Reject absolute paths on POSIX or Windows
    if raw.startswith("/") or raw.startswith("\\"):
        raise AuditBundleSecurityError(
            f"Archive entry path '{entry_path}' must not be an absolute path."
        )

    # Standardize to forward slashes and collapse redundant separators
    standardized = raw.replace("\\", "/")
    normalized = posixpath.normpath(standardized)

    # Check for path traversal escaping archive root
    if normalized in (".", "..") or normalized.startswith("../") or "/../" in normalized:
        raise AuditBundleSecurityError(
            f"Archive entry path '{entry_path}' attempts path traversal outside archive root."
        )

    if normalized.startswith("/"):
        raise AuditBundleSecurityError(
            f"Normalized path '{normalized}' must not be an absolute path."
        )

    return normalized


def validate_archive_member_safety(
    entry_path: str,
    seen_paths: Optional[Set[str]] = None,
) -> str:
    """Validate archive member path against path-traversal, absolute paths, and duplicate normalized entries.

    Parameters
    ----------
    entry_path : str
        Raw archive entry path.
    seen_paths : set of str, optional
        Set of previously observed normalized member paths within the same archive.

    Returns
    -------
    str
        Normalized member path.

    Raises
    ------
    AuditBundleSecurityError
        If entry path violates security constraints or resolves to a duplicate normalized entry.
    """
    normalized = normalize_bundle_path(entry_path)

    if seen_paths is not None:
        if normalized in seen_paths:
            raise AuditBundleSecurityError(
                f"Archive contains duplicate normalized member path: '{entry_path}' (normalized: '{normalized}')."
            )
        seen_paths.add(normalized)

    return normalized


def build_audit_bundle(
    run_dir: Union[str, Path],
    output_bundle_path: Optional[Union[str, Path]] = None,
    bundle_id: Optional[str] = None,
    deterministic: bool = True,
) -> Path:
    """Build a tamper-evident, portable ZIP audit bundle from a serialized run directory.

    Atomically packages all evidence files, manifests, and fitted models listed in the
    run manifest, creating an overarching bundle_manifest.json with SHA-256 digests.

    Parameters
    ----------
    run_dir : str or Path
        Directory containing serialized run artifacts (must contain manifest.json).
    output_bundle_path : str or Path, optional
        Destination .zip file path. If None, saved adjacent to run_dir as '{run_id}_audit_bundle.zip'.
    bundle_id : str, optional
        Unique bundle identifier. If None, derived from run_id.
    deterministic : bool, default=True
        If True, normalizes file timestamps and file permissions in ZIP headers for reproducible builds.

    Returns
    -------
    Path
        Absolute path to the created audit bundle archive.

    Raises
    ------
    AuditBundleError
        If run_dir is invalid, manifest is missing/malformed, indexed files are missing/tampered,
        or packaging fails.
    """
    src_dir = Path(run_dir).resolve()
    if not src_dir.exists() or not src_dir.is_dir():
        raise AuditBundleError(f"Run directory '{src_dir}' does not exist or is not a directory.")

    manifest_path = src_dir / "manifest.json"
    if not manifest_path.exists():
        raise AuditBundleError(f"Run directory '{src_dir}' is missing required 'manifest.json'.")

    # Read and parse run manifest
    try:
        with open(manifest_path, "rb") as f:
            manifest_raw = f.read()
        manifest_dict = json.loads(manifest_raw.decode("utf-8"))
    except Exception as exc:
        raise AuditBundleError(f"Failed to read or parse manifest in '{manifest_path}': {exc}") from exc

    run_id = manifest_dict.get("run_id")
    if not run_id:
        raise AuditBundleError(f"Manifest '{manifest_path}' is missing 'run_id'.")

    active_bundle_id = bundle_id.strip() if (bundle_id and isinstance(bundle_id, str)) else f"bundle_{run_id}"

    # Verify that all files listed in run manifest exist and match their declared checksums
    file_checksums = manifest_dict.get("file_checksums", {})
    bundle_files: List[AuditBundleFileEntry] = []

    # 1. Index manifest.json itself
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    bundle_files.append(
        AuditBundleFileEntry(
            path="manifest.json",
            sha256=manifest_sha256,
            size_bytes=len(manifest_raw),
            artifact_type="manifest",
        )
    )

    # 2. Index all declared run files
    for rel_path, expected_hash in sorted(file_checksums.items()):
        norm_rel = normalize_bundle_path(rel_path)
        member_file = src_dir / norm_rel
        if not member_file.exists():
            raise AuditBundleError(
                f"Cannot build bundle: indexed run file '{norm_rel}' not found in '{src_dir}'."
            )

        hasher = hashlib.sha256()
        file_size = 0
        try:
            with open(member_file, "rb") as f:
                while True:
                    chunk = f.read(CHUNK_SIZE_BYTES)
                    if not chunk:
                        break
                    hasher.update(chunk)
                    file_size += len(chunk)
        except Exception as exc:
            raise AuditBundleError(f"Failed to read file '{member_file}': {exc}") from exc

        actual_hash = hasher.hexdigest()
        if actual_hash != expected_hash:
            raise AuditBundleTamperError(
                f"Source file '{norm_rel}' checksum mismatch before packaging "
                f"(expected {expected_hash}, computed {actual_hash}). Aborting bundle build."
            )

        # Categorize artifact type
        if norm_rel.startswith("models/"):
            art_type = "model"
        elif norm_rel == "evaluation.json":
            art_type = "evaluation"
        elif norm_rel == "decision.json":
            art_type = "decision"
        elif norm_rel == "policy.json":
            art_type = "policy"
        else:
            art_type = "evidence"

        bundle_files.append(
            AuditBundleFileEntry(
                path=norm_rel,
                sha256=actual_hash,
                size_bytes=file_size,
                artifact_type=art_type,
            )
        )

    # Sort files deterministically by normalized path
    bundle_files.sort(key=lambda e: e.path)

    # Determine bundle creation timestamp
    created_at = (
        manifest_dict.get("created_at")
        if (deterministic and manifest_dict.get("created_at"))
        else datetime.now(timezone.utc).isoformat()
    )

    # Construct AuditBundleManifest
    bundle_manifest = AuditBundleManifest(
        schema_version=CURRENT_BUNDLE_SCHEMA_VERSION,
        bundle_id=active_bundle_id,
        run_id=run_id,
        created_at=created_at,
        files=bundle_files,
        dataset_checksum=manifest_dict.get("dataset_checksum"),
        split_strategy=manifest_dict.get("split_strategy"),
        num_folds=manifest_dict.get("num_folds"),
        candidate_model_type=manifest_dict.get("candidate_model_type"),
        baseline_model_type=manifest_dict.get("baseline_model_type"),
        policy_id=manifest_dict.get("policy_id"),
        policy_version=manifest_dict.get("policy_version"),
        decision_outcome=manifest_dict.get("decision_outcome"),
        has_models=bool(manifest_dict.get("has_models", False)),
        has_robustness=bool(manifest_dict.get("has_robustness", False)),
        software_versions=manifest_dict.get("software_versions", {}),
        metadata={"source_run_id": run_id},
        model_provenance=manifest_dict.get("model_provenance"),
        candidate_model_fold_index=manifest_dict.get("candidate_model_fold_index"),
        baseline_model_fold_index=manifest_dict.get("baseline_model_fold_index"),
    )

    bundle_manifest_bytes = json.dumps(
        bundle_manifest.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
    ).encode("utf-8")

    # Resolve target output path
    if output_bundle_path is not None:
        target_path = Path(output_bundle_path).resolve()
    else:
        target_path = (src_dir.parent / f"{run_id}_audit_bundle.zip").resolve()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp_{os.getpid()}")

    try:
        with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 1. Write bundle_manifest.json first
            zinfo_manifest = zipfile.ZipInfo(
                filename="bundle_manifest.json",
                date_time=FIXED_DETERMINISTIC_DATETIME if deterministic else (2026, 1, 1, 0, 0, 0),
            )
            zinfo_manifest.external_attr = 0o644 << 16
            zinfo_manifest.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zinfo_manifest, bundle_manifest_bytes)

            # 2. Write all bundle members in sorted order
            for entry in bundle_files:
                member_file = src_dir / entry.path
                with open(member_file, "rb") as f:
                    file_bytes = f.read()

                zinfo = zipfile.ZipInfo(
                    filename=entry.path,
                    date_time=FIXED_DETERMINISTIC_DATETIME if deterministic else (2026, 1, 1, 0, 0, 0),
                )
                zinfo.external_attr = 0o644 << 16
                zinfo.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(zinfo, file_bytes)

        # Atomic rename to final destination
        os.replace(tmp_path, target_path)
        return target_path
    except Exception as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        if isinstance(exc, AuditBundleError):
            raise
        raise AuditBundleError(f"Failed to create audit bundle at '{target_path}': {exc}") from exc


def verify_audit_bundle(
    bundle_path: Union[str, Path],
    raise_on_error: bool = True,
    max_archive_size_bytes: int = DEFAULT_MAX_ARCHIVE_SIZE_BYTES,
    max_member_size_bytes: int = DEFAULT_MAX_MEMBER_SIZE_BYTES,
    max_total_uncompressed_bytes: int = DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE_BYTES,
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
) -> AuditBundleVerificationResult:
    """Verify cryptographic integrity, manifest conformity, and resource/path safety of an audit bundle.

    SECURITY GUARANTEES
    -------------------
    1. NEVER deserializes models (no joblib.load or pickle.load).
    2. NEVER extracts files to arbitrary locations during verification.
    3. Detects path traversal, absolute paths, and duplicate normalized archive entries.
    4. Detects byte tampering, missing files, or unauthorized archive entries.
    5. Rejects encrypted members and symbolic links.
    6. Enforces configurable limits on archive size, member uncompressed size, total uncompressed
       size, and member compression ratio.
    7. Streams member hashing in bounded chunks (64 KB) to avoid large memory allocations.

    Parameters
    ----------
    bundle_path : str or Path
        Path to the .zip audit bundle archive.
    raise_on_error : bool, default=True
        If True, raises AuditBundleCorruptError, AuditBundleSecurityError, or AuditBundleTamperError.
        If False, returns an AuditBundleVerificationResult with is_valid=False and error details.
    max_archive_size_bytes : int, default=2 GB
        Maximum allowed archive file size on disk.
    max_member_size_bytes : int, default=500 MB
        Maximum allowed uncompressed size for any single member.
    max_total_uncompressed_bytes : int, default=2 GB
        Maximum allowed total uncompressed size across all members.
    max_compression_ratio : float, default=100.0
        Maximum allowed compression ratio for members larger than 1 MB.

    Returns
    -------
    AuditBundleVerificationResult
        Structured verification report.

    Raises
    ------
    AuditBundleCorruptError
        If archive is unreadable, truncated, or not a valid ZIP.
    AuditBundleSecurityError
        If archive contains path traversal, symlinks, encryption, or exceeds resource limits.
    AuditBundleTamperError
        If member checksums mismatch manifest or unexpected files exist.
    """
    path = Path(bundle_path)
    errors: List[str] = []
    warnings: List[str] = []

    if not path.exists() or not path.is_file():
        err = f"Bundle file '{path}' does not exist or is not a file."
        if raise_on_error:
            raise AuditBundleCorruptError(err)
        return AuditBundleVerificationResult(
            is_valid=False,
            errors=[err],
            warnings=warnings,
        )

    try:
        stat_size = path.stat().st_size
    except OSError as exc:
        err = f"Failed to inspect bundle file '{path}': {exc}"
        if raise_on_error:
            raise AuditBundleCorruptError(err) from exc
        return AuditBundleVerificationResult(
            is_valid=False,
            errors=[err],
            warnings=warnings,
        )

    if stat_size > max_archive_size_bytes:
        err = (
            f"Audit bundle archive '{path}' size ({stat_size} bytes) exceeds "
            f"maximum permitted limit ({max_archive_size_bytes} bytes)."
        )
        if raise_on_error:
            raise AuditBundleSecurityError(err)
        return AuditBundleVerificationResult(
            is_valid=False,
            errors=[err],
            warnings=warnings,
        )

    try:
        zf = zipfile.ZipFile(path, mode="r")
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, OSError) as exc:
        err = f"Bundle file '{path}' is not a valid or readable ZIP archive: {exc}"
        if raise_on_error:
            raise AuditBundleCorruptError(err) from exc
        return AuditBundleVerificationResult(
            is_valid=False,
            errors=[err],
            warnings=warnings,
        )

    with zf:
        # 1. Validate all member paths for security, duplicates, encryption, and symlinks first
        seen_paths: Set[str] = set()
        member_names = zf.namelist()
        total_declared_uncompressed = 0

        for zinfo in zf.infolist():
            raw_name = zinfo.filename

            # Check encryption flag
            if (zinfo.flag_bits & 0x1) != 0:
                err = f"Archive member '{raw_name}' is encrypted. Encrypted audit bundles are unsupported."
                if raise_on_error:
                    raise AuditBundleSecurityError(err)
                errors.append(err)

            # Check symbolic link
            if ((zinfo.external_attr >> 16) & 0o170000) == 0o120000:
                err = f"Archive member '{raw_name}' is a symbolic link. Symbolic links are forbidden in audit bundles."
                if raise_on_error:
                    raise AuditBundleSecurityError(err)
                errors.append(err)

            # Validate path safety & duplicates
            try:
                validate_archive_member_safety(raw_name, seen_paths=seen_paths)
            except AuditBundleSecurityError as exc:
                if raise_on_error:
                    raise
                errors.append(str(exc))

            # Validate member uncompressed size limit
            if zinfo.file_size > max_member_size_bytes:
                err = (
                    f"Archive member '{raw_name}' uncompressed size ({zinfo.file_size} bytes) "
                    f"exceeds maximum permitted limit ({max_member_size_bytes} bytes)."
                )
                if raise_on_error:
                    raise AuditBundleSecurityError(err)
                errors.append(err)

            # Check suspicious compression ratio (for members with non-trivial uncompressed size)
            if zinfo.file_size > 1024 * 1024 and zinfo.compress_size > 0:
                ratio = zinfo.file_size / zinfo.compress_size
                if ratio > max_compression_ratio:
                    err = (
                        f"Archive member '{raw_name}' has suspicious compression ratio {ratio:.1f}:1 "
                        f"(exceeds maximum permitted ratio {max_compression_ratio}:1)."
                    )
                    if raise_on_error:
                        raise AuditBundleSecurityError(err)
                    errors.append(err)

            total_declared_uncompressed += zinfo.file_size
            if total_declared_uncompressed > max_total_uncompressed_bytes:
                err = (
                    f"Total uncompressed archive size ({total_declared_uncompressed} bytes) "
                    f"exceeds maximum permitted limit ({max_total_uncompressed_bytes} bytes)."
                )
                if raise_on_error:
                    raise AuditBundleSecurityError(err)
                errors.append(err)

        if errors:
            return AuditBundleVerificationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
            )

        # 2. Check ZIP structural integrity (CRC test)
        try:
            bad_member = zf.testzip()
            if bad_member is not None:
                err = f"ZIP CRC checksum failure on member '{bad_member}'."
                if raise_on_error:
                    raise AuditBundleCorruptError(err)
                return AuditBundleVerificationResult(
                    is_valid=False,
                    errors=[err],
                    warnings=warnings,
                )
        except Exception as exc:
            err = f"Failed ZIP integrity test: {exc}"
            if raise_on_error:
                raise AuditBundleCorruptError(err) from exc
            return AuditBundleVerificationResult(
                is_valid=False,
                errors=[err],
                warnings=warnings,
            )

        # 2. Check presence of bundle_manifest.json
        if "bundle_manifest.json" not in member_names:
            err = "Archive missing required 'bundle_manifest.json'."
            if raise_on_error:
                raise AuditBundleTamperError(err)
            return AuditBundleVerificationResult(
                is_valid=False,
                errors=[err],
                warnings=warnings,
            )

        # 3. Read and parse bundle_manifest.json
        try:
            with zf.open("bundle_manifest.json") as mf:
                manifest_bytes = mf.read(CHUNK_SIZE_BYTES * 16)
            manifest_dict = json.loads(manifest_bytes.decode("utf-8"))
            bundle_manifest = AuditBundleManifest.model_validate(manifest_dict)
        except Exception as exc:
            err = f"Invalid or malformed 'bundle_manifest.json': {exc}"
            if raise_on_error:
                raise AuditBundleCorruptError(err) from exc
            return AuditBundleVerificationResult(
                is_valid=False,
                errors=[err],
                warnings=warnings,
            )

        # 4. Compare declared file inventory vs archive members
        declared_entries = {entry.path: entry for entry in bundle_manifest.files}
        actual_members = {normalize_bundle_path(name) for name in member_names if name != "bundle_manifest.json"}

        # Check for missing members
        missing_members = set(declared_entries.keys()) - actual_members
        if missing_members:
            err = f"Archive is missing declared member files: {sorted(missing_members)}"
            if raise_on_error:
                raise AuditBundleTamperError(err)
            errors.append(err)

        # Check for unauthorized / unindexed members
        unexpected_members = actual_members - set(declared_entries.keys())
        if unexpected_members:
            err = f"Archive contains unindexed unexpected files not declared in manifest: {sorted(unexpected_members)}"
            if raise_on_error:
                raise AuditBundleTamperError(err)
            errors.append(err)

        verified_files: List[str] = []
        cumulative_streamed_bytes = 0

        # 5. Verify SHA-256 digests and byte sizes of all declared entries using bounded streaming
        for path_name, entry in sorted(declared_entries.items()):
            if path_name not in actual_members:
                continue

            hasher = hashlib.sha256()
            member_bytes_count = 0
            try:
                with zf.open(path_name) as member_file:
                    while True:
                        chunk = member_file.read(CHUNK_SIZE_BYTES)
                        if not chunk:
                            break
                        member_bytes_count += len(chunk)
                        cumulative_streamed_bytes += len(chunk)
                        if member_bytes_count > max_member_size_bytes:
                            err = (
                                f"Member '{path_name}' exceeded maximum permitted uncompressed size "
                                f"({max_member_size_bytes} bytes) during streaming."
                            )
                            if raise_on_error:
                                raise AuditBundleSecurityError(err)
                            errors.append(err)
                            break
                        if cumulative_streamed_bytes > max_total_uncompressed_bytes:
                            err = (
                                f"Cumulative uncompressed bytes exceeded limit "
                                f"({max_total_uncompressed_bytes} bytes) during verification."
                            )
                            if raise_on_error:
                                raise AuditBundleSecurityError(err)
                            errors.append(err)
                            break
                        hasher.update(chunk)
            except Exception as exc:
                if isinstance(exc, AuditBundleError):
                    raise
                err = f"Could not read member '{path_name}': {exc}"
                if raise_on_error:
                    raise AuditBundleTamperError(err) from exc
                errors.append(err)
                continue

            # Verify size
            if member_bytes_count != entry.size_bytes:
                err = (
                    f"Size mismatch for member '{path_name}': expected {entry.size_bytes} bytes, "
                    f"found {member_bytes_count} bytes."
                )
                if raise_on_error:
                    raise AuditBundleTamperError(err)
                errors.append(err)
                continue

            # Verify SHA-256
            actual_sha256 = hasher.hexdigest()
            if actual_sha256 != entry.sha256:
                err = (
                    f"Integrity check failed for member '{path_name}': checksum mismatch "
                    f"(expected {entry.sha256}, computed {actual_sha256}). Member bytes have been altered or corrupted."
                )
                if raise_on_error:
                    raise AuditBundleTamperError(err)
                errors.append(err)
                continue

            verified_files.append(path_name)

    is_valid = len(errors) == 0
    return AuditBundleVerificationResult(
        is_valid=is_valid,
        bundle_id=bundle_manifest.bundle_id,
        run_id=bundle_manifest.run_id,
        verified_files=verified_files,
        errors=errors,
        warnings=warnings,
        manifest=bundle_manifest if is_valid else None,
    )


def extract_audit_bundle(
    bundle_path: Union[str, Path],
    destination_dir: Union[str, Path],
    verify_first: bool = True,
    max_archive_size_bytes: int = DEFAULT_MAX_ARCHIVE_SIZE_BYTES,
    max_member_size_bytes: int = DEFAULT_MAX_MEMBER_SIZE_BYTES,
    max_total_uncompressed_bytes: int = DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE_BYTES,
) -> Path:
    """Safely extract an audit bundle into a destination directory with zip-slip path confinement.

    SECURITY GUARANTEES
    -------------------
    1. Validates all archive member paths against path-traversal before and during extraction,
       even when verify_first=False.
    2. Strictly confines extracted files within destination_dir.
    3. Rejects encrypted archive members and symbolic links.
    4. Enforces uncompressed member size and cumulative extraction limits.
    5. Streams member extraction in bounded chunks (64 KB).
    6. NEVER deserializes or executes joblib/pickle models.

    Parameters
    ----------
    bundle_path : str or Path
        Path to the .zip audit bundle.
    destination_dir : str or Path
        Target directory for extracted files.
    verify_first : bool, default=True
        If True, runs verify_audit_bundle before extracting any files.
    max_archive_size_bytes : int, default=2 GB
        Maximum allowed archive file size on disk.
    max_member_size_bytes : int, default=500 MB
        Maximum allowed uncompressed size for any single member.
    max_total_uncompressed_bytes : int, default=2 GB
        Maximum allowed cumulative uncompressed size.

    Returns
    -------
    Path
        Absolute path to destination directory.

    Raises
    ------
    AuditBundleCorruptError
        If archive is unreadable or non-existent.
    AuditBundleSecurityError
        If an entry escapes the destination directory, is a symlink/encrypted, or exceeds limits.
    AuditBundleTamperError
        If verification fails when verify_first=True.
    """
    bpath = Path(bundle_path)
    if not bpath.exists() or not bpath.is_file():
        raise AuditBundleCorruptError(f"Bundle file '{bpath}' does not exist or is not a file.")

    stat_size = bpath.stat().st_size
    if stat_size > max_archive_size_bytes:
        raise AuditBundleSecurityError(
            f"Audit bundle archive '{bpath}' size ({stat_size} bytes) exceeds limit ({max_archive_size_bytes} bytes)."
        )

    if verify_first:
        verify_audit_bundle(
            bundle_path,
            raise_on_error=True,
            max_archive_size_bytes=max_archive_size_bytes,
            max_member_size_bytes=max_member_size_bytes,
            max_total_uncompressed_bytes=max_total_uncompressed_bytes,
        )

    dest = Path(destination_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    cumulative_extracted = 0
    try:
        with zipfile.ZipFile(bundle_path, mode="r") as zf:
            for zinfo in zf.infolist():
                # Check encrypted
                if (zinfo.flag_bits & 0x1) != 0:
                    raise AuditBundleSecurityError(
                        f"Archive member '{zinfo.filename}' is encrypted. Encrypted audit bundles are unsupported."
                    )
                # Check symlink
                if ((zinfo.external_attr >> 16) & 0o170000) == 0o120000:
                    raise AuditBundleSecurityError(
                        f"Archive member '{zinfo.filename}' is a symbolic link. Symbolic links are forbidden in audit bundles."
                    )

                normalized = normalize_bundle_path(zinfo.filename)
                target_path = (dest / normalized).resolve()

                # Strict path confinement defense
                try:
                    target_path.relative_to(dest)
                except ValueError:
                    raise AuditBundleSecurityError(
                        f"Extraction path traversal detected: '{zinfo.filename}' escapes destination '{dest}'."
                    )

                if zinfo.file_size > max_member_size_bytes:
                    raise AuditBundleSecurityError(
                        f"Archive member '{zinfo.filename}' uncompressed size ({zinfo.file_size} bytes) "
                        f"exceeds limit ({max_member_size_bytes} bytes)."
                    )

                if zinfo.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    member_extracted = 0
                    with zf.open(zinfo) as source, open(target_path, "wb") as target:
                        while True:
                            chunk = source.read(CHUNK_SIZE_BYTES)
                            if not chunk:
                                break
                            member_extracted += len(chunk)
                            cumulative_extracted += len(chunk)
                            if member_extracted > max_member_size_bytes:
                                raise AuditBundleSecurityError(
                                    f"Member '{zinfo.filename}' exceeded uncompressed size limit during extraction."
                                )
                            if cumulative_extracted > max_total_uncompressed_bytes:
                                raise AuditBundleSecurityError(
                                    f"Total extracted size exceeded limit ({max_total_uncompressed_bytes} bytes)."
                                )
                            target.write(chunk)
    except (zipfile.BadZipFile, EOFError) as exc:
        raise AuditBundleCorruptError(
            f"Corrupted or invalid ZIP archive '{bundle_path}': {exc}"
        ) from exc

    return dest
