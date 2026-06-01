"""Install pinned Helm and kubectl binaries into a project-local tool directory."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import tarfile
import tempfile
import urllib.request
from pathlib import Path


HELM_VERSION = "v3.15.4"
KUBECTL_VERSION = "v1.30.4"


def _linux_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "amd64"
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    raise RuntimeError(f"unsupported Linux architecture: {machine}")


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _install_helm(bin_dir: Path, arch: str) -> None:
    archive_name = f"helm-{HELM_VERSION}-linux-{arch}.tar.gz"
    url = f"https://get.helm.sh/{archive_name}"
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        archive = tmpdir / archive_name
        _download(url, archive)
        with tarfile.open(archive) as handle:
            handle.extractall(tmpdir)
        shutil.copy2(tmpdir / f"linux-{arch}" / "helm", bin_dir / "helm")
    _make_executable(bin_dir / "helm")


def _install_kubectl(bin_dir: Path, arch: str) -> None:
    url = (
        f"https://dl.k8s.io/release/{KUBECTL_VERSION}/bin/linux/{arch}/kubectl"
    )
    _download(url, bin_dir / "kubectl")
    _make_executable(bin_dir / "kubectl")


def install(bin_dir: Path) -> None:
    if platform.system().lower() != "linux":
        raise RuntimeError("this installer supports Linux developer/CI hosts only")
    arch = _linux_arch()
    bin_dir.mkdir(parents=True, exist_ok=True)
    _install_helm(bin_dir, arch)
    _install_kubectl(bin_dir, arch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", default=".tools/bin")
    args = parser.parse_args()
    install(Path(args.bin_dir))
    checks = {
        "helm": "version --short",
        "kubectl": "version --client=true",
    }
    for name, command in checks.items():
        path = Path(args.bin_dir) / name
        os.system(f"{path} {command} >/dev/null")


if __name__ == "__main__":
    main()
