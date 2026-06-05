from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


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


def require_list(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array.")
    return value


def require_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")
    return value


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


def install_python_packages(manifest: dict) -> None:
    python_config = require_mapping(manifest.get("python"), "python")
    packages = require_list(python_config.get("packages"), "python.packages")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
    )
    for package in packages:
        command = [sys.executable, "-m", "pip", "install"]
        if isinstance(package, str):
            command.append(package)
        else:
            package_config = require_mapping(package, "python.packages[]")
            index_url = package_config.get("indexUrl")
            if index_url is not None:
                command.extend(["--index-url", require_str(index_url, "indexUrl")])
            command.append(require_str(package_config.get("name"), "name"))
        subprocess.run(command, check=True)


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


def copy_zip_member(archive: Path, member: dict, package_root: Path) -> bool:
    archive_path = require_str(member.get("archivePath"), "archivePath")
    output_path = require_str(member.get("outputPath"), "outputPath")
    required = bool(member.get("required", True))
    with zipfile.ZipFile(archive) as zf:
        try:
            matched_member = zip_member(zf, archive_path)
        except RuntimeError:
            if required:
                raise
            return False
        destination = package_root / output_path
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


def copy_extracted_member(extract_root: Path, member: dict, package_root: Path) -> bool:
    archive_suffix = require_str(member.get("archiveSuffix"), "archiveSuffix")
    output_path = require_str(member.get("outputPath"), "outputPath")
    required = bool(member.get("required", True))
    try:
        source = find_extracted_path(extract_root, archive_suffix)
    except RuntimeError:
        if required:
            raise
        return False
    destination = package_root / output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination, follow_symlinks=True)
    return True


def register_source(source_urls: dict[str, str], output_path: str, source_url: str) -> None:
    source_urls[output_path] = source_url


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
        source_url = source_urls.get(relative_path)
        if not source_url:
            raise RuntimeError(f"No sourceUrl mapping for {relative_path}")
        entries.append(
            {
                "path": relative_path,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "sourceUrl": source_url,
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

    docaligner = require_mapping(sources.get("docaligner"), "sources.docaligner")
    docaligner_model = require_mapping(docaligner.get("model"), "docaligner.model")
    docaligner_model_path = require_str(docaligner_model.get("outputPath"), "outputPath")
    docaligner_model_url = require_str(docaligner_model.get("downloadUrl"), "downloadUrl")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gdown",
            "--id",
            require_str(docaligner_model.get("fileId"), "fileId"),
            "-O",
            str(common / docaligner_model_path),
        ],
        check=True,
    )
    register_source(common_source_urls, docaligner_model_path, docaligner_model_url)

    docaligner_license = require_mapping(docaligner.get("license"), "docaligner.license")
    download_source_file(docaligner_license, common, common_source_urls)

    uvdoc = require_mapping(sources.get("uvdoc"), "sources.uvdoc")
    uvdoc_license = require_mapping(uvdoc.get("license"), "uvdoc.license")
    download_source_file(uvdoc_license, common, common_source_urls)

    checkpoint = require_mapping(uvdoc.get("checkpoint"), "uvdoc.checkpoint")
    model_source = require_mapping(uvdoc.get("modelSource"), "uvdoc.modelSource")
    checkpoint_url = require_str(checkpoint.get("downloadUrl"), "downloadUrl")
    checkpoint_path = upstream / require_str(checkpoint.get("cacheFile"), "cacheFile")
    model_source_path = upstream / require_str(model_source.get("cacheFile"), "cacheFile")
    download(checkpoint_url, checkpoint_path)
    download(
        require_str(model_source.get("downloadUrl"), "downloadUrl"),
        model_source_path,
    )

    sys.path.insert(0, str(upstream))
    import torch
    from model import UVDocnet

    loaded_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = UVDocnet(num_filter=32, kernel_size=5)
    model.load_state_dict(loaded_checkpoint["model_state"])
    model.eval()
    dummy = torch.zeros(1, 3, 488, 712, dtype=torch.float32)
    uvdoc_output_path = require_str(checkpoint.get("outputPath"), "outputPath")
    torch.onnx.export(
        model,
        dummy,
        common / uvdoc_output_path,
        input_names=["image"],
        output_names=["grid2d", "grid3d"],
        opset_version=16,
    )
    register_source(common_source_urls, uvdoc_output_path, checkpoint_url)

    models_readme = require_mapping(sources.get("modelsReadme"), "sources.modelsReadme")
    readme_path = require_str(models_readme.get("outputPath"), "modelsReadme.outputPath")
    readme_lines = require_list(models_readme.get("lines"), "modelsReadme.lines")
    write_text(common / readme_path, "\n".join(str(line) for line in readme_lines) + "\n")
    register_source(
        common_source_urls,
        readme_path,
        require_str(models_readme.get("sourceUrl"), "modelsReadme.sourceUrl"),
    )

    return common_source_urls


def download_source_file(config: dict, root: Path, source_urls: dict[str, str]) -> None:
    output_path = require_str(config.get("outputPath"), "outputPath")
    source_url = require_str(config.get("downloadUrl"), "downloadUrl")
    download(source_url, root / output_path)
    register_source(source_urls, output_path, source_url)


def build_windows_assets(
    sources: dict,
    common: Path,
    work: Path,
    upstream: Path,
    common_source_urls: dict[str, str],
) -> tuple[Path, dict[str, str]]:
    windows = require_mapping(sources.get("windows"), "sources.windows")
    package_root = work / "windows-directml"
    copy_tree(common, package_root)
    source_urls = dict(common_source_urls)

    for package_key in ["onnxRuntimeDirectML", "directML"]:
        package = require_mapping(windows.get(package_key), f"windows.{package_key}")
        package_url = require_str(package.get("packageUrl"), f"{package_key}.packageUrl")
        package_path = upstream / require_str(package.get("cacheFile"), f"{package_key}.cacheFile")
        download(package_url, package_path)
        for member in require_list(package.get("members"), f"{package_key}.members"):
            member_config = require_mapping(member, f"{package_key}.members[]")
            copied = copy_zip_member(package_path, member_config, package_root)
            if copied:
                register_source(
                    source_urls,
                    require_str(member_config.get("outputPath"), "outputPath"),
                    package_url,
                )

    windows_readme = package_root / "onnxruntime" / "windows" / "README.md"
    if not windows_readme.exists():
        onnx_package = require_mapping(
            windows.get("onnxRuntimeDirectML"),
            "windows.onnxRuntimeDirectML",
        )
        fallback_lines = require_list(onnx_package.get("fallbackReadme"), "fallbackReadme")
        output_path = "onnxruntime/windows/README.md"
        write_text(windows_readme, " ".join(str(line) for line in fallback_lines) + "\n")
        register_source(
            source_urls,
            output_path,
            require_str(onnx_package.get("packageUrl"), "packageUrl"),
        )

    return package_root, source_urls


def build_linux_assets(
    sources: dict,
    common: Path,
    work: Path,
    upstream: Path,
    common_source_urls: dict[str, str],
) -> tuple[Path, dict[str, str]]:
    linux = require_mapping(sources.get("linux"), "sources.linux")
    linux_ort = require_mapping(linux.get("onnxRuntimeGpu"), "linux.onnxRuntimeGpu")
    archive_url = require_str(linux_ort.get("archiveUrl"), "archiveUrl")
    archive_path = upstream / require_str(linux_ort.get("cacheFile"), "cacheFile")
    download(archive_url, archive_path)

    expected_sha256 = require_str(linux_ort.get("sha256"), "sha256").lower()
    actual_sha256 = sha256_file(archive_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Linux ONNX Runtime archive sha256 mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )

    extract_root = upstream / "linux-ort"
    safe_extract_tar(archive_path, extract_root)

    package_root = work / "linux-tensorrt-cuda"
    copy_tree(common, package_root)
    source_urls = dict(common_source_urls)
    for member in require_list(linux_ort.get("members"), "linux.onnxRuntimeGpu.members"):
        member_config = require_mapping(member, "linux.onnxRuntimeGpu.members[]")
        copied = copy_extracted_member(extract_root, member_config, package_root)
        if copied:
            register_source(
                source_urls,
                require_str(member_config.get("outputPath"), "outputPath"),
                archive_url,
            )

    linux_readme = package_root / "onnxruntime" / "linux" / "README.md"
    if not linux_readme.exists():
        fallback_lines = require_list(linux_ort.get("fallbackReadme"), "fallbackReadme")
        output_path = "onnxruntime/linux/README.md"
        write_text(linux_readme, " ".join(str(line) for line in fallback_lines) + "\n")
        register_source(source_urls, output_path, archive_url)

    return package_root, source_urls


def verify_archive_roots(dist: Path, asset_tag: str) -> None:
    windows_archive = dist / f"windows-directml-{asset_tag}.zip"
    linux_archive = dist / f"linux-tensorrt-cuda-{asset_tag}.tar.gz"
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

    install_python_packages(source_manifest)

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
        "windows-directml",
        args.asset_tag,
        windows_source_urls,
    )
    write_asset_manifest(
        linux_root,
        "linux-tensorrt-cuda",
        args.asset_tag,
        linux_source_urls,
    )
    package_zip(windows_root, dist / f"windows-directml-{args.asset_tag}.zip")
    package_tar_gz(linux_root, dist / f"linux-tensorrt-cuda-{args.asset_tag}.tar.gz")
    verify_archive_roots(dist, args.asset_tag)

    print("Built scanner asset packages:")
    for archive in sorted(dist.iterdir()):
        print(f"- {archive.name} {archive.stat().st_size} bytes")


if __name__ == "__main__":
    main()
