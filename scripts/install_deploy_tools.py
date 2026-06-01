"""Install pinned Helm and kubectl binaries into a project-local tool directory."""

from __future__ import annotations

import argparse
import hashlib
import platform
import stat
import subprocess
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


def _parse_checksum(path: Path) -> str:
    tokens = path.read_text().split()
    if not tokens:
        raise RuntimeError(f"checksum file is empty: {path}")
    return tokens[0]


def _verify_sha256(path: Path, expected: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"checksum mismatch for {path.name}: expected {expected}, got {actual}"
        )


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _install_helm(bin_dir: Path, arch: str) -> None:
    archive_name = f"helm-{HELM_VERSION}-linux-{arch}.tar.gz"
    url = f"https://get.helm.sh/{archive_name}"
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        archive = tmpdir / archive_name
        checksum = tmpdir / f"{archive_name}.sha256sum"
        _download(url, archive)
        _download(f"{url}.sha256sum", checksum)
        _verify_sha256(archive, _parse_checksum(checksum))
        with tarfile.open(archive) as handle:
            member_name = f"linux-{arch}/helm"
            try:
                member = handle.extractfile(member_name)
            except KeyError as exc:
                raise RuntimeError(f"missing Helm archive member: {member_name}") from exc
            if member is None:
                raise RuntimeError(f"Helm archive member is not a file: {member_name}")
            with member:
                (bin_dir / "helm").write_bytes(member.read())
    _make_executable(bin_dir / "helm")


def _install_kubectl(bin_dir: Path, arch: str) -> None:
    url = (
        f"https://dl.k8s.io/release/{KUBECTL_VERSION}/bin/linux/{arch}/kubectl"
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        binary = tmpdir / "kubectl"
        checksum = tmpdir / "kubectl.sha256"
        _download(url, binary)
        _download(f"{url}.sha256", checksum)
        _verify_sha256(binary, _parse_checksum(checksum))
        (bin_dir / "kubectl").write_bytes(binary.read_bytes())
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
        subprocess.run(
            [str(path), *command.split()],
            check=True,
            stdout=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    main()
