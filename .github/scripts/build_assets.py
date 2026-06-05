from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlretrieve


docaligner_model_path = "models/docaligner-fastvit_sa24.onnx"
docaligner_license_path = "models/LICENSE-docaligner"
uvdoc_model_path = "models/uvdoc-best-model.onnx"
uvdoc_license_path = "models/LICENSE-uvdoc"
models_readme_path = "models/README.md"

windows_platform = "windows-directml"
linux_platform = "linux-tensorrt-cuda"

windows_ort_readme_fallback = (
    "Windows ONNX Runtime DirectML files are extracted from "
    "Microsoft.ML.OnnxRuntime.DirectML and Microsoft.AI.DirectML NuGet packages.\n"
)
linux_ort_readme_fallback = (
    "Linux ONNX Runtime GPU files are extracted from the official "
    "Microsoft ONNX Runtime GitHub Release asset.\n"
)


@dataclass(frozen=True)
class ZipMember:
    archive_path: str
    output_path: str
    required: bool = True


@dataclass(frozen=True)
class TarMember:
    archive_suffix: str
    output_path: str
    required: bool = True


windows_onnxruntime_members = (
    ZipMember("runtimes/win-x64/native/onnxruntime.dll", "onnxruntime/windows/onnxruntime.dll"),
    ZipMember(
        "runtimes/win-x64/native/onnxruntime_providers_shared.dll",
        "onnxruntime/windows/onnxruntime_providers_shared.dll",
    ),
    ZipMember("README.md", "onnxruntime/windows/README.md", required=False),
    ZipMember("LICENSE", "onnxruntime/windows/LICENSE-onnxruntime", required=False),
    ZipMember(
        "ThirdPartyNotices.txt",
        "onnxruntime/windows/ThirdPartyNotices-onnxruntime.txt",
        required=False,
    ),
)
windows_directml_members = (
    ZipMember("bin/x64-win/DirectML.dll", "onnxruntime/windows/DirectML.dll"),
    ZipMember("LICENSE", "onnxruntime/windows/LICENSE-directml", required=False),
)
linux_onnxruntime_members = (
    TarMember("lib/libonnxruntime.so", "onnxruntime/linux/libonnxruntime.so"),
    TarMember("lib/libonnxruntime.so.1", "onnxruntime/linux/libonnxruntime.so.1"),
    TarMember("lib/libonnxruntime.so.1.24.4", "onnxruntime/linux/libonnxruntime.so.1.24.4"),
    TarMember(
        "lib/libonnxruntime_providers_shared.so",
        "onnxruntime/linux/libonnxruntime_providers_shared.so",
    ),
    TarMember(
        "lib/libonnxruntime_providers_cuda.so",
        "onnxruntime/linux/libonnxruntime_providers_cuda.so",
    ),
    TarMember(
        "lib/libonnxruntime_providers_tensorrt.so",
        "onnxruntime/linux/libonnxruntime_providers_tensorrt.so",
    ),
    TarMember("LICENSE", "onnxruntime/linux/LICENSE-onnxruntime", required=False),
    TarMember(
        "ThirdPartyNotices.txt",
        "onnxruntime/linux/ThirdPartyNotices-onnxruntime.txt",
        required=False,
    ),
    TarMember("Privacy.md", "onnxruntime/linux/Privacy.md", required=False),
    TarMember("README.md", "onnxruntime/linux/README.md", required=False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scanner asset packages.")
    parser.add_argument("--asset-tag", required=True)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--dist-dir", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object.")
    return value


def require_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def source_url(value: object, name: str) -> str:
    if isinstance(value, str):
        return require_str(value, name)
    source = require_mapping(value, name)
    return require_str(source.get("url"), f"{name}.url")


def get_source_url(sources: dict, *keys: str) -> str:
    value: object = sources
    path_parts = []
    for key in keys:
        path_parts.append(key)
        value = require_mapping(value, ".".join(path_parts[:-1]) or "sources").get(key)
    return source_url(value, ".".join(("sources", *keys)))


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_source(source_urls: dict[str, str], output_path: str, source_url_value: str) -> None:
    source_urls[output_path] = source_url_value


def download_package(url: str, upstream: Path, fallback_name: str) -> Path:
    parsed = urlparse(url)
    basename = Path(unquote(parsed.path)).name or fallback_name
    package_path = upstream / basename
    download(url, package_path)
    return package_path


def is_google_drive_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc.lower().endswith("drive.google.com")


def download_google_drive_or_url(url: str, path: Path) -> None:
    if is_google_drive_url(url):
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "gdown", url, "-O", str(path)],
            check=True,
        )
        return
    download(url, path)


def download_to_package(
    url: str,
    package_root: Path,
    output_path: str,
    source_urls: dict[str, str],
) -> None:
    download(url, package_root / output_path)
    register_source(source_urls, output_path, url)


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def zip_member(zf: zipfile.ZipFile, requested: str) -> str:
    requested_lower = requested.lower()
    for name in zf.namelist():
        if name.lower() == requested_lower:
            return name
    raise RuntimeError(f"Missing expected zip member: {requested}")


def copy_zip_member(archive: Path, member: ZipMember, package_root: Path) -> bool:
    with zipfile.ZipFile(archive) as zf:
        try:
            matched_member = zip_member(zf, member.archive_path)
        except RuntimeError:
            if member.required:
                raise
            return False
        destination = package_root / member.output_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(zf.read(matched_member))
        return True


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tf.extractall(destination)


def find_extracted_path(root: Path, suffix: str) -> Path:
    normalized_suffix = suffix.replace("\\", "/")
    for path in root.rglob("*"):
        if path.is_file() and path.as_posix().endswith(normalized_suffix):
            return path
    raise RuntimeError(f"Missing expected extracted file: {suffix}")


def copy_extracted_member(extract_root: Path, member: TarMember, package_root: Path) -> bool:
    try:
        source = find_extracted_path(extract_root, member.archive_suffix)
    except RuntimeError:
        if member.required:
            raise
        return False
    destination = package_root / member.output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination, follow_symlinks=True)
    return True


def write_asset_manifest(
    package_root: Path,
    platform_target: str,
    asset_tag: str,
    source_urls: dict[str, str],
) -> None:
    entries = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative_path = path.relative_to(package_root).as_posix()
        source_url_value = source_urls.get(relative_path)
        if not source_url_value:
            raise RuntimeError(f"No sourceUrl mapping for {relative_path}")
        entries.append(
            {
                "path": relative_path,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "sourceUrl": source_url_value,
            }
        )

    manifest = {
        "schemaVersion": 1,
        "assetVersion": asset_tag,
        "platformTarget": platform_target,
        "files": entries,
    }
    write_text(
        package_root / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
    )


def package_zip(package_root: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(package_root).as_posix())


def package_tar_gz(package_root: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as tf:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                tf.add(path, arcname=path.relative_to(package_root).as_posix())


def build_common_assets(sources: dict, common: Path, upstream: Path) -> dict[str, str]:
    common_source_urls: dict[str, str] = {}
    models_dir = common / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    docaligner_model_url = get_source_url(sources, "docaligner", "model")
    download_google_drive_or_url(docaligner_model_url, common / docaligner_model_path)
    register_source(common_source_urls, docaligner_model_path, docaligner_model_url)

    download_to_package(
        get_source_url(sources, "docaligner", "license"),
        common,
        docaligner_license_path,
        common_source_urls,
    )
    download_to_package(
        get_source_url(sources, "uvdoc", "license"),
        common,
        uvdoc_license_path,
        common_source_urls,
    )
    download_to_package(
        get_source_url(sources, "uvdoc", "readme"),
        common,
        models_readme_path,
        common_source_urls,
    )

    uvdoc_model_url = get_source_url(sources, "uvdoc", "model")
    uvdoc_weights_path = download_package(uvdoc_model_url, upstream, "uvdoc-best_model.pkl")

    import torch
    from uvdoc_model import UVDocnet

    loaded_weights = torch.load(
        uvdoc_weights_path,
        map_location="cpu",
        weights_only=False,
    )
    model = UVDocnet(num_filter=32, kernel_size=5)
    model.load_state_dict(loaded_weights["model_state"])
    model.eval()
    dummy = torch.zeros(1, 3, 488, 712, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        common / uvdoc_model_path,
        input_names=["image"],
        output_names=["grid2d", "grid3d"],
        opset_version=16,
        external_data=False,
    )
    register_source(common_source_urls, uvdoc_model_path, uvdoc_model_url)

    return common_source_urls


def build_windows_assets(
    sources: dict,
    common: Path,
    work: Path,
    upstream: Path,
    common_source_urls: dict[str, str],
) -> tuple[Path, dict[str, str]]:
    package_root = work / windows_platform
    copy_tree(common, package_root)
    source_urls = dict(common_source_urls)

    onnxruntime_url = get_source_url(sources, "onnxruntime", "windows")
    onnxruntime_package = download_package(
        onnxruntime_url,
        upstream,
        "microsoft.ml.onnxruntime.directml.nupkg",
    )
    for member in windows_onnxruntime_members:
        if copy_zip_member(onnxruntime_package, member, package_root):
            register_source(source_urls, member.output_path, onnxruntime_url)

    directml_url = get_source_url(sources, "directml")
    directml_package = download_package(directml_url, upstream, "microsoft.ai.directml.nupkg")
    for member in windows_directml_members:
        if copy_zip_member(directml_package, member, package_root):
            register_source(source_urls, member.output_path, directml_url)

    windows_readme = package_root / "onnxruntime" / "windows" / "README.md"
    if not windows_readme.exists():
        output_path = "onnxruntime/windows/README.md"
        write_text(windows_readme, windows_ort_readme_fallback)
        register_source(source_urls, output_path, onnxruntime_url)

    return package_root, source_urls


def build_linux_assets(
    sources: dict,
    common: Path,
    work: Path,
    upstream: Path,
    common_source_urls: dict[str, str],
) -> tuple[Path, dict[str, str]]:
    onnxruntime_url = get_source_url(sources, "onnxruntime", "linux")
    archive_path = download_package(onnxruntime_url, upstream, "onnxruntime-linux-x64-gpu.tgz")

    extract_root = upstream / "linux-ort"
    safe_extract_tar(archive_path, extract_root)

    package_root = work / linux_platform
    copy_tree(common, package_root)
    source_urls = dict(common_source_urls)
    for member in linux_onnxruntime_members:
        if copy_extracted_member(extract_root, member, package_root):
            register_source(source_urls, member.output_path, onnxruntime_url)

    linux_readme = package_root / "onnxruntime" / "linux" / "README.md"
    if not linux_readme.exists():
        output_path = "onnxruntime/linux/README.md"
        write_text(linux_readme, linux_ort_readme_fallback)
        register_source(source_urls, output_path, onnxruntime_url)

    return package_root, source_urls


def verify_archive_roots(dist: Path, asset_tag: str) -> None:
    windows_archive = dist / f"{windows_platform}-{asset_tag}.zip"
    linux_archive = dist / f"{linux_platform}-{asset_tag}.tar.gz"
    with zipfile.ZipFile(windows_archive) as zf:
        if "manifest.json" not in zf.namelist():
            raise RuntimeError("Windows package is missing root manifest.json.")
    with tarfile.open(linux_archive) as tf:
        if "manifest.json" not in tf.getnames():
            raise RuntimeError("Linux package is missing root manifest.json.")


def main() -> None:
    args = parse_args()
    source_manifest = load_json(args.source_manifest)
    if source_manifest.get("schemaVersion") != 1:
        raise RuntimeError("Unsupported asset source manifest schemaVersion.")

    sources = require_mapping(source_manifest.get("sources"), "sources")
    root = Path.cwd()
    work = (root / args.work_dir).resolve()
    common = work / "common"
    upstream = work / "upstream"
    dist = (root / args.dist_dir).resolve()
    reset_dir(work)
    reset_dir(common)
    reset_dir(upstream)
    reset_dir(dist)

    common_source_urls = build_common_assets(sources, common, upstream)
    windows_root, windows_source_urls = build_windows_assets(
        sources,
        common,
        work,
        upstream,
        common_source_urls,
    )
    linux_root, linux_source_urls = build_linux_assets(
        sources,
        common,
        work,
        upstream,
        common_source_urls,
    )

    write_asset_manifest(
        windows_root,
        windows_platform,
        args.asset_tag,
        windows_source_urls,
    )
    write_asset_manifest(
        linux_root,
        linux_platform,
        args.asset_tag,
        linux_source_urls,
    )
    package_zip(windows_root, dist / f"{windows_platform}-{args.asset_tag}.zip")
    package_tar_gz(linux_root, dist / f"{linux_platform}-{args.asset_tag}.tar.gz")
    verify_archive_roots(dist, args.asset_tag)

    print("Built scanner asset packages:")
    for archive in sorted(dist.iterdir()):
        print(f"- {archive.name} {archive.stat().st_size} bytes")


if __name__ == "__main__":
    main()
