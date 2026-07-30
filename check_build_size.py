"""Report dist/installer size for TCS and check the 100 MB budget.

Run via CHECK_BUILD_SIZE.bat. Not part of the app package - a build/dev tool only.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIMIT_BYTES = 100 * 1024 * 1024

DIST_DIR = ROOT / "dist" / "TCS"
INSTALLER_PATH = ROOT / "release" / "TCS_Setup.exe"


def human(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def top_files(path: Path, count: int = 30):
    files = [(f.stat().st_size, f) for f in path.rglob("*") if f.is_file()]
    files.sort(key=lambda x: x[0], reverse=True)
    return files[:count]


def main() -> int:
    print("=" * 60)
    print("TCS")
    print("=" * 60)

    any_found = False
    ok = True

    if DIST_DIR.exists():
        any_found = True
        size = folder_size(DIST_DIR)
        print(f"dist folder ({DIST_DIR.name}): {human(size)}")
        print(f"Top 30 file terbesar di {DIST_DIR.name}:")
        for i, (fsize, fpath) in enumerate(top_files(DIST_DIR), start=1):
            print(f"  {i:2d}. {human(fsize):>10}  {fpath.relative_to(DIST_DIR)}")
    else:
        print(f"[SKIP] {DIST_DIR} tidak ditemukan.")

    if INSTALLER_PATH.exists():
        any_found = True
        isize = INSTALLER_PATH.stat().st_size
        ok = isize < LIMIT_BYTES
        status = "PASS" if ok else "FAIL"
        print(f"\nInstaller: {INSTALLER_PATH.name} = {human(isize)}")
        print(f"{status}: Installer {'<' if ok else '>='} 100 MB")
        if not ok:
            print("Rekomendasi: cek daftar file terbesar di atas. Kandidat umum penyebab besar:")
            print("  - duplikasi GDAL dari fiona.libs + rasterio.libs (pertimbangkan pyshp)")
            print("  - PyQt6 Qt Quick/QML/WebEngine/Designer/Pdf plugin (pastikan tidak pakai collect_all)")
            print("  - icon .png berukuran besar di assets/icons (pertimbangkan kompres/resize)")
    else:
        print(f"\n[SKIP] {INSTALLER_PATH} tidak ditemukan. Build installer dulu.")

    if not any_found:
        print("[FAILED] Tidak ada dist/ atau installer yang ditemukan sama sekali.")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
