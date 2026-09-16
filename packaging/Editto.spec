# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


ROOT = Path(SPECPATH).parent.parent
PACKAGING = ROOT / "packaging"
BIN_DIR = PACKAGING / "vendor" / "bin"

binaries = []
for name in ("ffmpeg.exe", "ffprobe.exe"):
    candidate = BIN_DIR / name
    if candidate.exists():
        binaries.append((str(candidate), "resources/bin"))

a = Analysis(
    [str(PACKAGING / "launcher.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=[
        (str(ROOT / "models"), "resources/models"),
        (str(ROOT / "LICENSE"), "."),
        (str(ROOT / "MODEL_LICENSE.md"), "."),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "scipy", "matplotlib", "pandas", "pytest", "torchvision", "torchaudio",
        "torchtext", "onnxruntime", "tensorflow", "tensorboard", "PIL",
        "win32com", "setuptools",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Editto",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Editto",
)
