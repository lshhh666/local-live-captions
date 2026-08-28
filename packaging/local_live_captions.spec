from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files


project_root = Path(SPEC).resolve().parents[1]
site_packages = project_root / ".venv" / "Lib" / "site-packages"

nvidia_binaries = []
required_cuda_dlls = {
    "cublas": {"cublas64_12.dll", "cublasLt64_12.dll"},
    "cudnn": {"cudnn64_9.dll", "cudnn_graph64_9.dll", "cudnn_ops64_9.dll"},
    "cuda_nvrtc": {"nvrtc64_120_0.dll", "nvrtc-builtins64_129.dll"},
}
missing_cuda = []
for component in ("cublas", "cudnn", "cuda_nvrtc"):
    source = site_packages / "nvidia" / component / "bin"
    available = {path.name for path in source.glob("*.dll")} if source.is_dir() else set()
    missing_cuda.extend(
        f"{component}/{name}" for name in sorted(required_cuda_dlls[component] - available)
    )
    nvidia_binaries.extend(
        (str(path), f"nvidia/{component}/bin") for path in source.glob("*.dll")
    )
if missing_cuda:
    raise SystemExit("缺少便携版 CUDA DLL：" + ", ".join(missing_cuda))

common = dict(
    pathex=[str(project_root / "py")],
    hiddenimports=["pyaudiowpatch", "faster_whisper", "ctranslate2", "tokenizers"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["imageio_ffmpeg", "pip", "setuptools", "unittest"],
    noarchive=False,
    optimize=1,
)

launcher_analysis = Analysis(
    [str(project_root / "packaging" / "launcher_entry.py")],
    binaries=[],
    datas=[],
    **common,
)
worker_analysis = Analysis(
    [str(project_root / "packaging" / "worker_entry.py")],
    binaries=nvidia_binaries,
    datas=collect_data_files("faster_whisper"),
    **common,
)

launcher_pyz = PYZ(launcher_analysis.pure)
worker_pyz = PYZ(worker_analysis.pure)

launcher_exe = EXE(
    launcher_pyz,
    launcher_analysis.scripts,
    [],
    exclude_binaries=True,
    name="本地实时字幕",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
worker_exe = EXE(
    worker_pyz,
    worker_analysis.scripts,
    [],
    exclude_binaries=True,
    name="caption-worker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

bundle = COLLECT(
    launcher_exe,
    worker_exe,
    launcher_analysis.binaries,
    launcher_analysis.datas,
    worker_analysis.binaries,
    worker_analysis.datas,
    strip=False,
    upx=True,
    name="本地实时字幕-便携版",
)
