"""MLForge Command Line Interface (Milestone 4.5).

Provides offline CLI subcommands for verifying release certificates and audit bundles:
- python -m mlforge verify-release <target> --public-key <hex>
- python -m mlforge verify-bundle <bundle_path>

SECURITY & AUDIT GUARANTEES
---------------------------
1. Offline Execution: Zero network calls, telemetry, or remote services.
2. Non-Executable Verification: NEVER calls joblib.load or pickle.load.
3. Fail-Closed Trust: Requires explicit trusted key material for release authorization.
4. Strict Exit Codes:
   - 0: Verified & authorized.
   - 1: Verification failed / Certificate rejected / Gate blocked.
   - 2: Usage error / Invalid arguments / Missing files.
5. Sanitized Diagnostics: Diagnostics never print private keys or sensitive credentials.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Union
import zipfile

from cryptography.hazmat.primitives.asymmetric import ed25519

from mlforge.application.packaging import (
    DEFAULT_MAX_ARCHIVE_SIZE_BYTES,
    DEFAULT_MAX_COMPRESSION_RATIO,
    DEFAULT_MAX_MEMBER_SIZE_BYTES,
    DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE_BYTES,
    verify_audit_bundle,
)
from mlforge.application.release_certificates import (
    deserialize_public_key,
    verify_release_certificate,
)
from mlforge.core.contracts import (
    AuditBundleVerificationResult,
    CertificateVerificationResult,
)


def _load_trusted_keys(
    public_key_hex: Optional[str] = None,
    trusted_keys_file: Optional[str] = None,
) -> Dict[str, ed25519.Ed25519PublicKey]:
    """Parse and validate trusted public key material.

    Returns
    -------
    Dict[str, Ed25519PublicKey]
        Mapping of key_id (first 16 hex chars of SHA-256 of raw public key) to Ed25519PublicKey.

    Raises
    ------
    ValueError
        If key format is invalid or file cannot be parsed.
    """
    trusted_keys: Dict[str, ed25519.Ed25519PublicKey] = {}

    if public_key_hex:
        clean_hex = public_key_hex.strip()
        if len(clean_hex) != 64:
            raise ValueError(
                f"Invalid --public-key length: expected 64 hex characters (32 bytes), got {len(clean_hex)}."
            )
        try:
            pub_key = deserialize_public_key(clean_hex)
        except Exception as exc:
            raise ValueError(f"Failed to parse --public-key: {exc}") from exc

        raw_pub = pub_key.public_bytes_raw()
        key_id = hashlib.sha256(raw_pub).hexdigest()[:16]
        trusted_keys[key_id] = pub_key

    if trusted_keys_file:
        kf_path = Path(trusted_keys_file).resolve()
        if not kf_path.exists() or not kf_path.is_file():
            raise ValueError(f"Trusted keys file not found: '{trusted_keys_file}'.")

        try:
            with open(kf_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise ValueError(f"Failed to parse JSON from trusted keys file '{trusted_keys_file}': {exc}") from exc

        if isinstance(data, dict):
            for kid, hex_val in data.items():
                if not isinstance(hex_val, str) or len(hex_val.strip()) != 64:
                    raise ValueError(f"Invalid public key hex for key '{kid}' in trusted keys file.")
                pub_key = deserialize_public_key(hex_val.strip())
                derived_id = hashlib.sha256(pub_key.public_bytes_raw()).hexdigest()[:16]
                trusted_keys[kid] = pub_key
                trusted_keys[derived_id] = pub_key
        elif isinstance(data, list):
            for idx, hex_val in enumerate(data):
                if not isinstance(hex_val, str) or len(hex_val.strip()) != 64:
                    raise ValueError(f"Invalid public key hex at index {idx} in trusted keys file.")
                pub_key = deserialize_public_key(hex_val.strip())
                derived_id = hashlib.sha256(pub_key.public_bytes_raw()).hexdigest()[:16]
                trusted_keys[derived_id] = pub_key
        else:
            raise ValueError("Trusted keys file must contain a JSON object or array of hex keys.")

    return trusted_keys


def run_verify_bundle(args: argparse.Namespace) -> int:
    """Execute verify-bundle subcommand.

    Returns
    -------
    int
        0 on valid bundle, 1 on verification failure, 2 on usage/file error.
    """
    bpath = Path(args.bundle_path)
    if not bpath.exists():
        sys.stderr.write(f"Error: Bundle file not found: '{args.bundle_path}'\n")
        return 2
    if not bpath.is_file():
        sys.stderr.write(f"Error: Path is not a file: '{args.bundle_path}'\n")
        return 2

    res: AuditBundleVerificationResult = verify_audit_bundle(
        bundle_path=bpath,
        raise_on_error=False,
        max_archive_size_bytes=args.max_archive_size,
        max_member_size_bytes=args.max_member_size,
        max_total_uncompressed_bytes=args.max_total_uncompressed,
        max_compression_ratio=args.max_compression_ratio,
    )

    if res.is_valid:
        if not args.quiet:
            print(f"[VALID] Audit bundle '{bpath.name}' verified successfully.")
            if res.run_id:
                print(f"  Run ID: {res.run_id}")
            print(f"  Verified Files: {len(res.verified_files)}")
            if res.warnings:
                for w in res.warnings:
                    print(f"  Warning: {w}")
        return 0
    else:
        if not args.quiet:
            sys.stderr.write(f"[INVALID] Audit bundle verification failed for '{bpath.name}'.\n")
            for err in res.errors:
                sys.stderr.write(f"  - Error: {err}\n")
            for w in res.warnings:
                sys.stderr.write(f"  - Warning: {w}\n")
        return 1


def run_verify_release(args: argparse.Namespace) -> int:
    """Execute verify-release subcommand.

    Returns
    -------
    int
        0 on authorized release, 1 on verification rejection, 2 on usage/file error.
    """
    # 1. Parse and validate trust configuration
    if not args.public_key and not args.trusted_keys_file:
        sys.stderr.write(
            "Error: Missing required trust material. Either --public-key or --trusted-keys-file must be specified.\n"
        )
        return 2

    try:
        trusted_keys = _load_trusted_keys(
            public_key_hex=args.public_key,
            trusted_keys_file=args.trusted_keys_file,
        )
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2

    # 2. Inspect target path
    target_path = Path(args.target).resolve()
    if not target_path.exists():
        sys.stderr.write(f"Error: Target path not found: '{args.target}'\n")
        return 2

    certificate_source: Union[str, Path]
    expected_source: Optional[Union[str, Path]] = None

    if target_path.is_file() and target_path.suffix.lower() == ".zip":
        expected_source = target_path
        if args.certificate:
            cert_p = Path(args.certificate).resolve()
            if not cert_p.exists() or not cert_p.is_file():
                sys.stderr.write(f"Error: Specified certificate file not found: '{args.certificate}'\n")
                return 2
            certificate_source = cert_p
        else:
            # Discover certificate inside ZIP bundle
            try:
                with zipfile.ZipFile(target_path, "r") as zf:
                    namelist = zf.namelist()
                    cert_name: Optional[str] = None
                    for candidate in [
                        "release/release_certificate.json",
                        "release_certificate.json",
                        "certificate.json",
                    ]:
                        if candidate in namelist:
                            cert_name = candidate
                            break

                    if cert_name is None:
                        sys.stderr.write(
                            f"Error: No release certificate found inside audit bundle '{target_path.name}'. "
                            "Specify certificate location via --certificate.\n"
                        )
                        return 1

                    certificate_source = zf.read(cert_name).decode("utf-8")
            except Exception as exc:
                sys.stderr.write(f"Error: Failed to inspect audit bundle '{target_path.name}': {exc}\n")
                return 1

    elif target_path.is_dir():
        expected_source = target_path
        if args.certificate:
            cert_p = Path(args.certificate).resolve()
            if not cert_p.exists() or not cert_p.is_file():
                sys.stderr.write(f"Error: Specified certificate file not found: '{args.certificate}'\n")
                return 2
            certificate_source = cert_p
        else:
            # Discover certificate in directory
            discovered: Optional[Path] = None
            for cand in [
                target_path / "release" / "release_certificate.json",
                target_path / "release_certificate.json",
                target_path / "certificate.json",
            ]:
                if cand.exists() and cand.is_file():
                    discovered = cand
                    break

            if discovered is None:
                sys.stderr.write(
                    f"Error: No release certificate found in directory '{target_path}'. "
                    "Specify certificate location via --certificate.\n"
                )
                return 1
            certificate_source = discovered

    elif target_path.is_file() and target_path.suffix.lower() == ".json":
        certificate_source = target_path
        if args.expected_source:
            exp_p = Path(args.expected_source).resolve()
            if not exp_p.exists():
                sys.stderr.write(f"Error: Expected source not found: '{args.expected_source}'\n")
                return 2
            expected_source = exp_p
        else:
            # Fail closed: certificate alone cannot verify physical evidence without expected_source
            expected_source = None
    else:
        sys.stderr.write(
            f"Error: Unsupported target '{args.target}'. Expected a run directory, .zip bundle, or .json certificate.\n"
        )
        return 2

    # Override expected_source if explicitly provided
    if args.expected_source:
        exp_p = Path(args.expected_source).resolve()
        if not exp_p.exists():
            sys.stderr.write(f"Error: Explicit expected source not found: '{args.expected_source}'\n")
            return 2
        expected_source = exp_p

    # 3. Perform Verification
    res: CertificateVerificationResult = verify_release_certificate(
        certificate=certificate_source,
        trusted_keys=trusted_keys,
        expected_source=expected_source,
    )

    if res.release_authorized:
        if not args.quiet:
            print(f"[AUTHORIZED] Release authorized for run '{res.run_id}'.")
            print(f"  Certificate ID: {res.certificate_id}")
            print(f"  Signer Key ID:  {res.key_id} (Trusted)")
            print("  Tiers:")
            print("    - Payload Integrity:    VALID")
            print("    - Signature Validity:   VALID")
            print("    - Signer Trust:         TRUSTED")
            print("    - Evidence Consistency: CONSISTENT")
            if res.warnings:
                for w in res.warnings:
                    print(f"  Warning: {w}")
        return 0
    else:
        if not args.quiet:
            sys.stderr.write(f"[REJECTED] Release NOT authorized for run '{res.run_id or 'unknown'}'.\n")
            print_status = lambda cond, t_str, f_str: t_str if cond else f_str
            sys.stderr.write("  Tiers:\n")
            sys.stderr.write(f"    - Payload Integrity:    {print_status(res.payload_integrity, 'VALID', 'FAILED')}\n")
            sys.stderr.write(f"    - Signature Validity:   {print_status(res.signature_validity, 'VALID', 'FAILED')}\n")
            sys.stderr.write(f"    - Signer Trust:         {print_status(res.signer_trust, 'TRUSTED', 'UNTRUSTED')}\n")
            sys.stderr.write(f"    - Evidence Consistency: {print_status(res.evidence_consistency, 'CONSISTENT', 'FAILED')}\n")
            if res.errors:
                sys.stderr.write("  Errors:\n")
                for err in res.errors:
                    sys.stderr.write(f"    - {err}\n")
            if res.warnings:
                sys.stderr.write("  Warnings:\n")
                for w in res.warnings:
                    sys.stderr.write(f"    - {w}\n")
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="mlforge",
        description="MLForge AI Model Evaluation & Reliability Lab CLI",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available subcommands",
    )

    # 1. verify-bundle subcommand
    bundle_p = subparsers.add_parser(
        "verify-bundle",
        help="Verify cryptographic integrity, manifest conformity, and resource bounds of an audit bundle ZIP.",
    )
    bundle_p.add_argument(
        "bundle_path",
        type=str,
        help="Path to the .zip audit bundle archive to verify.",
    )
    bundle_p.add_argument(
        "--max-archive-size",
        type=int,
        default=DEFAULT_MAX_ARCHIVE_SIZE_BYTES,
        help="Maximum permitted archive file size in bytes (default: 2 GB).",
    )
    bundle_p.add_argument(
        "--max-member-size",
        type=int,
        default=DEFAULT_MAX_MEMBER_SIZE_BYTES,
        help="Maximum permitted uncompressed size for any single member in bytes (default: 500 MB).",
    )
    bundle_p.add_argument(
        "--max-total-uncompressed",
        type=int,
        default=DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE_BYTES,
        help="Maximum permitted cumulative uncompressed size in bytes (default: 2 GB).",
    )
    bundle_p.add_argument(
        "--max-compression-ratio",
        type=float,
        default=DEFAULT_MAX_COMPRESSION_RATIO,
        help="Maximum permitted compression ratio for members > 1 MB (default: 100.0).",
    )
    bundle_p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose diagnostic output.",
    )

    # 2. verify-release subcommand
    release_p = subparsers.add_parser(
        "verify-release",
        help="Verify release certificate signature, signer trust, and evidence consistency.",
    )
    release_p.add_argument(
        "target",
        type=str,
        help="Path to release_certificate.json, evaluation run directory, or .zip audit bundle.",
    )
    release_p.add_argument(
        "--public-key",
        type=str,
        default=None,
        help="Hexadecimal Ed25519 public key (64 hex characters) of the trusted signer.",
    )
    release_p.add_argument(
        "--trusted-keys-file",
        type=str,
        default=None,
        help="Path to JSON file containing trusted public keys (mapping of key_id to hex key, or list of hex keys).",
    )
    release_p.add_argument(
        "--expected-source",
        type=str,
        default=None,
        help="Path to evaluation run directory or .zip audit bundle to verify physical evidence against certificate claims. "
             "Defaults to target if target is already a directory or .zip bundle.",
    )
    release_p.add_argument(
        "--certificate",
        type=str,
        default=None,
        help="Path to release certificate JSON file if target is a directory or audit bundle.",
    )
    release_p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose diagnostic output.",
    )

    # 3. serve subcommand
    serve_p = subparsers.add_parser(
        "serve",
        help="Start the MLForge local REST API server.",
    )
    serve_p.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind socket host (default: 127.0.0.1).",
    )
    serve_p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind socket port (default: 8000).",
    )
    serve_p.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development.",
    )

    return parser


def run_serve(args: argparse.Namespace) -> int:
    """Start local uvicorn server for MLForge."""
    try:
        import uvicorn
    except ImportError:
        sys.stderr.write("Error: uvicorn is required to start the API server. Install with `pip install uvicorn`.\n")
        return 2

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        sys.stderr.write(
            f"[SECURITY WARNING] Binding to non-loopback address '{args.host}'. "
            "MLForge is designed for local development and lacks multi-tenant isolation. "
            "Exposing to a LAN or public network is not recommended.\n"
        )

    sys.stdout.write(f"Starting MLForge Laboratory API on http://{args.host}:{args.port}\n")
    uvicorn.run("mlforge.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entrypoint.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments. Defaults to sys.argv[1:].

    Returns
    -------
    int
        POSIX process exit code (0 = success, 1 = verification failed, 2 = usage error).
    """
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return 2

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    if args.command == "verify-bundle":
        return run_verify_bundle(args)
    elif args.command == "verify-release":
        return run_verify_release(args)
    elif args.command == "serve":
        return run_serve(args)
    else:
        parser.print_help()
        return 2
