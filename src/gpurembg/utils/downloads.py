import hashlib
import os
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(
    url: str,
    destination: Path,
    expected_sha256: Optional[str] = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """
    Download a remote file with streaming and optional checksum verification.

    Parameters
    ----------
    url: str
        Remote URL to download.
    destination: Path
        Local destination path.
    expected_sha256: Optional[str]
        Optional SHA-256 hex digest. When provided the file is verified after download.
    chunk_size: int
        Streaming chunk size in bytes. Defaults to 1 MiB.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Short-circuit if we already have a verified file.
    if destination.exists() and expected_sha256:
        if sha256_file(destination) == expected_sha256.lower():
            return destination

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    progress = tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {os.path.basename(destination)}",
    )

    tmp_path = destination.with_suffix(destination.suffix + ".tmp")

    with tmp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            handle.write(chunk)
            progress.update(len(chunk))

    progress.close()

    tmp_path.replace(destination)

    if expected_sha256 and sha256_file(destination) != expected_sha256.lower():
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {destination}. Expected {expected_sha256}."
        )

    return destination
