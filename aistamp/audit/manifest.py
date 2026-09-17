"""Export manifests: reproducibility metadata for audit exports.

An export manifest records which pattern and policy files were in effect
(identified by SHA-256 content digests, not just paths) alongside the
package version that produced the export. An auditor can later confirm
that an exported evidence file was produced from a known set of inputs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version() -> str:
    """Installed distribution version, falling back to the source version."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("ai-stamp")
    except PackageNotFoundError:
        from aistamp import __version__

        return __version__


@dataclass(frozen=True)
class FileDigest:
    """A file identity for reproducibility: path plus content digest."""

    path: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ExportManifest:
    """Reproducibility metadata attached to an export."""

    manifest_version: int = MANIFEST_VERSION
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    aistamp_version: str = field(default_factory=package_version)
    pattern_files: tuple[FileDigest, ...] = ()
    policy_file: FileDigest | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "generated_at": self.generated_at.isoformat(),
            "aistamp_version": self.aistamp_version,
            "pattern_files": [f.to_dict() for f in self.pattern_files],
            "policy_file": self.policy_file.to_dict() if self.policy_file else None,
        }


def build_export_manifest(
    pattern_files: Sequence[str | Path] = (),
    policy_file: str | Path | None = None,
    generated_at: datetime | None = None,
) -> ExportManifest:
    """Digest the given pattern/policy files into an ExportManifest.

    Raises FileNotFoundError for any path that does not exist, so a
    manifest is never produced for inputs that were not actually read.
    """
    digests: list[FileDigest] = []
    for raw in pattern_files:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Pattern file not found: {p}")
        digests.append(FileDigest(path=str(p), sha256=_digest_file(p)))

    policy_digest: FileDigest | None = None
    if policy_file is not None:
        p = Path(policy_file)
        if not p.exists():
            raise FileNotFoundError(f"Policy file not found: {p}")
        policy_digest = FileDigest(path=str(p), sha256=_digest_file(p))

    return ExportManifest(
        generated_at=generated_at or datetime.now(timezone.utc),
        pattern_files=tuple(digests),
        policy_file=policy_digest,
    )
