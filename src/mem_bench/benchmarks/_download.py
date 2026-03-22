"""Helper for downloading benchmark data from HuggingFace Hub."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mem_bench"


def download_benchmark(
    repo_id: str,
    filename: str,
    *,
    cache_dir: Path | str | None = None,
) -> Path:
    """Download a file from a HuggingFace Hub dataset repository.

    Uses ``huggingface_hub`` when available, falling back to a plain
    ``requests`` download so that the heavy HF dependency is optional.

    Args:
        repo_id: HuggingFace dataset repository, e.g. ``"xiaowu0162/longmemeval-cleaned"``.
        filename: File path within the repository, e.g. ``"longmemeval_oracle.json"``.
        cache_dir: Local directory for caching downloaded files.
            Defaults to ``~/.cache/mem_bench/``.

    Returns:
        Path to the downloaded (or cached) file on disk.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    local_path = cache_dir / repo_id.replace("/", "--") / filename

    if local_path.exists():
        logger.debug("Using cached file: %s", local_path)
        return local_path

    # Attempt 1: huggingface_hub
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(cache_dir / "_hf_cache"),
        )
        downloaded = Path(downloaded)
        # Copy/symlink into our own cache layout for consistency
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            try:
                local_path.symlink_to(downloaded)
            except OSError:
                import shutil

                shutil.copy2(downloaded, local_path)
        logger.info("Downloaded via huggingface_hub: %s", local_path)
        return local_path

    except ImportError:
        logger.debug("huggingface_hub not installed, falling back to requests")
    except Exception as exc:
        logger.warning("huggingface_hub download failed (%s), falling back to requests", exc)

    # Attempt 2: plain requests
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Neither huggingface_hub nor requests is installed. "
            "Install at least one: pip install huggingface_hub  OR  pip install requests"
        ) from exc

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    logger.info("Downloading %s via HTTP ...", url)

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MiB
            f.write(chunk)

    logger.info("Saved to %s", local_path)
    return local_path
