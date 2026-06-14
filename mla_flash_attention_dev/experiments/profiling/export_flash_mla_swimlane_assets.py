from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = REPO_ROOT / "mla_flash_attention_dev" / "docs"
DEFAULT_SOURCE_MARKDOWN = DOCS_DIR / "flash-mla-wh-profile-swimlane-diagrams.md"
DEFAULT_ASSET_DIR = DOCS_DIR / "assets" / "flash-mla-wh-profile-swimlanes"
MERMAID_CLI = ["npx", "--yes", "@mermaid-js/mermaid-cli@8.13.10"]


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def extract_diagrams(markdown: str, source_markdown: Path) -> list[tuple[str, str]]:
    diagrams: list[tuple[str, str]] = []
    current_heading = "diagram"
    current_case = None
    in_block = False
    block_lines: list[str] = []

    for line in markdown.splitlines():
        if line.startswith("## "):
            current_heading = line[3:].strip()
            backtick_match = re.search(r"`([^`]+)`", current_heading)
            current_case = backtick_match.group(1) if backtick_match else None

        if line.strip() == "```mermaid":
            in_block = True
            block_lines = []
            continue

        if in_block and line.strip() == "```":
            name = current_case or slugify(current_heading)
            diagrams.append((name, "\n".join(block_lines).strip() + "\n"))
            in_block = False
            block_lines = []
            continue

        if in_block:
            block_lines.append(line)

    if not diagrams:
        raise RuntimeError(f"No mermaid blocks found in {source_markdown}")

    return diagrams


def render_mermaid(
    input_path: Path,
    output_path: Path,
    background: str,
    puppeteer_config: Path,
    width: int,
    scale: int | None = None,
) -> None:
    command = MERMAID_CLI + [
        "-p",
        str(puppeteer_config),
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-b",
        background,
        "-w",
        str(width),
    ]
    if scale is not None:
        command += ["-s", str(scale)]
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Mermaid swimlane diagrams to SVG/PNG assets.")
    parser.add_argument(
        "--source-markdown",
        default=str(DEFAULT_SOURCE_MARKDOWN.relative_to(REPO_ROOT)),
        help="Markdown file containing mermaid code fences.",
    )
    parser.add_argument(
        "--asset-dir",
        default=str(DEFAULT_ASSET_DIR.relative_to(REPO_ROOT)),
        help="Output asset directory for generated SVG/PNG/source files.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2400,
        help="Mermaid canvas width.",
    )
    args = parser.parse_args()

    source_markdown = resolve_repo_path(args.source_markdown)
    asset_dir = resolve_repo_path(args.asset_dir)
    source_dir = asset_dir / "src"
    manifest_path = asset_dir / "manifest.json"
    puppeteer_config = asset_dir / "puppeteer-config.json"

    markdown = source_markdown.read_text(encoding="utf-8")
    diagrams = extract_diagrams(markdown, source_markdown)

    asset_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    puppeteer_config.write_text(
        json.dumps({"args": ["--no-sandbox", "--disable-setuid-sandbox"]}, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest: list[dict[str, str]] = []
    for name, mermaid_source in diagrams:
        source_path = source_dir / f"{name}.mmd"
        svg_path = asset_dir / f"{name}.svg"
        png_path = asset_dir / f"{name}.png"

        source_path.write_text(mermaid_source, encoding="utf-8")
        render_mermaid(source_path, svg_path, "transparent", puppeteer_config, args.width)
        render_mermaid(source_path, png_path, "white", puppeteer_config, args.width, scale=2)

        manifest.append(
            {
                "name": name,
                "source": str(source_path.relative_to(REPO_ROOT)),
                "svg": str(svg_path.relative_to(REPO_ROOT)),
                "png": str(png_path.relative_to(REPO_ROOT)),
            }
        )
        print(f"rendered {name}")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
