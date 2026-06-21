#!/usr/bin/env python3
"""Render Markdown reports and slides to DOCX/PPTX.

The renderer keeps the source Markdown untouched. It converts Mermaid and
PlantUML fenced blocks to images in a temporary build directory, rewrites a
temporary Markdown copy, and delegates the final document conversion to
Pandoc.

External tools:
  - pandoc (required)
  - mmdc / Mermaid CLI (required when Mermaid blocks are present)
  - plantuml, or java + PLANTUML_JAR (required when PlantUML blocks are present)
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


FENCED_BLOCK_RE = re.compile(
    r"(?P<fence>```+|~~~+)[ \t]*(?P<language>[^\r\n]*)\r?\n"
    r"(?P<body>.*?)\r?\n(?P=fence)[ \t]*",
    re.DOTALL,
)
HTML_IMAGE_RE = re.compile(r"<img\s+(?P<attrs>[^>]+?)\s*/?>", re.IGNORECASE)
HTML_ATTR_RE = re.compile(
    r"(?P<name>[\w:-]+)\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    re.DOTALL,
)


class RenderError(RuntimeError):
    """Raised when a conversion dependency or document build fails."""


@dataclass(frozen=True)
class DiagramToolchain:
    mmdc: str | None
    plantuml: str | None
    java: str | None
    plantuml_jar: Path | None


def executable(name: str) -> str | None:
    return shutil.which(name)


def resolve_source(explicit: str | None, candidates: Sequence[Path], label: str) -> Path:
    if explicit:
        source = Path(explicit).expanduser().resolve()
        if not source.is_file():
            raise RenderError(f"{label} source does not exist: {source}")
        return source

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise RenderError(f"Could not find {label} source. Checked: {searched}")


def detect_toolchain() -> DiagramToolchain:
    jar_value = os.environ.get("PLANTUML_JAR", "").strip()
    jar = Path(jar_value).expanduser().resolve() if jar_value else None
    if jar is not None and not jar.is_file():
        jar = None
    return DiagramToolchain(
        mmdc=executable("mmdc"),
        plantuml=executable("plantuml"),
        java=executable("java"),
        plantuml_jar=jar,
    )


def run(command: Sequence[str], *, cwd: Path | None = None, stdin: bytes | None = None) -> bytes:
    printable = " ".join(str(part) for part in command)
    print(f"[render] {printable}")
    completed = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RenderError(f"Command failed ({completed.returncode}): {printable}\n{stderr}")
    return completed.stdout


def diagram_kind(language: str, body: str) -> str | None:
    normalized = language.strip().lower().split(maxsplit=1)[0] if language.strip() else ""
    if normalized in {"mermaid", "mmd"}:
        return "mermaid"
    if normalized in {"plantuml", "puml"} or body.lstrip().startswith("@startuml"):
        return "plantuml"
    return None


def render_mermaid(
    body: str,
    destination: Path,
    toolchain: DiagramToolchain,
    image_format: str,
) -> None:
    if not toolchain.mmdc:
        raise RenderError(
            "A Mermaid block was found, but 'mmdc' is unavailable. "
            "Install @mermaid-js/mermaid-cli and ensure mmdc is on PATH."
        )
    source = destination.with_suffix(".mmd")
    source.write_text(body.rstrip() + "\n", encoding="utf-8")
    command = [
        toolchain.mmdc,
        "--input",
        str(source),
        "--output",
        str(destination),
        "--backgroundColor",
        "transparent",
        "--scale",
        "2",
    ]
    if image_format == "png":
        command.extend(["--width", "1600"])
    run(command)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RenderError(f"Mermaid did not produce an image: {destination}")


def render_plantuml(
    body: str,
    destination: Path,
    toolchain: DiagramToolchain,
    image_format: str,
) -> None:
    output_flag = f"-t{image_format}"
    if toolchain.plantuml:
        command = [toolchain.plantuml, output_flag, "-pipe"]
    elif toolchain.java and toolchain.plantuml_jar:
        command = [
            toolchain.java,
            "-Djava.awt.headless=true",
            "-jar",
            str(toolchain.plantuml_jar),
            output_flag,
            "-pipe",
        ]
    else:
        raise RenderError(
            "A PlantUML block was found, but PlantUML is unavailable. Install the "
            "'plantuml' command, or set PLANTUML_JAR and provide Java on PATH."
        )
    rendered = run(command, stdin=(body.rstrip() + "\n").encode("utf-8"))
    destination.write_bytes(rendered)
    if destination.stat().st_size == 0:
        raise RenderError(f"PlantUML did not produce an image: {destination}")


def normalize_html_images(markdown: str) -> str:
    """Convert simple HTML img tags into Pandoc image syntax.

    SLIDES.md uses HTML to size images side by side. Pandoc handles image
    attributes more reliably when they use its Markdown extension syntax.
    """

    def replace_image(match: re.Match[str]) -> str:
        attrs = {
            item.group("name").lower(): html.unescape(item.group("value"))
            for item in HTML_ATTR_RE.finditer(match.group("attrs"))
        }
        source = attrs.get("src")
        if not source:
            return match.group(0)
        alt = attrs.get("alt", "Image").replace("]", "\\]")
        width = attrs.get("width", "").strip()
        width_attr = f"{{width={width}}}" if width else ""
        return f"![{alt}]({source}){width_attr}"

    markdown = HTML_IMAGE_RE.sub(replace_image, markdown)
    markdown = re.sub(r"</?p(?:\s+[^>]*)?>", "", markdown, flags=re.IGNORECASE)
    return markdown


def preprocess_markdown(
    source: Path,
    temporary_markdown: Path,
    diagrams_dir: Path,
    toolchain: DiagramToolchain,
    image_format: str,
) -> int:
    original = source.read_text(encoding="utf-8-sig")
    diagram_count = 0
    diagrams_dir.mkdir(parents=True, exist_ok=True)

    def replace_block(match: re.Match[str]) -> str:
        nonlocal diagram_count
        language = match.group("language")
        body = match.group("body")
        kind = diagram_kind(language, body)
        if kind is None:
            return match.group(0)

        diagram_count += 1
        destination = diagrams_dir / f"{source.stem.lower()}-{diagram_count:02d}-{kind}.{image_format}"
        print(f"[render] diagram {diagram_count}: {kind} -> {destination}")
        if kind == "mermaid":
            render_mermaid(body, destination, toolchain, image_format)
        else:
            render_plantuml(body, destination, toolchain, image_format)

        # Absolute POSIX-style paths work with Pandoc on Linux, macOS, and Windows.
        return f"![{kind.title()} diagram]({destination.resolve().as_posix()})"

    processed = FENCED_BLOCK_RE.sub(replace_block, original)
    processed = normalize_html_images(processed)
    temporary_markdown.parent.mkdir(parents=True, exist_ok=True)
    temporary_markdown.write_text(processed, encoding="utf-8")
    return diagram_count


def pandoc_resource_path(source: Path, project_root: Path) -> str:
    paths = [source.parent.resolve(), project_root.resolve()]
    return os.pathsep.join(str(path) for path in paths)


def build_with_pandoc(
    source: Path,
    temporary_markdown: Path,
    destination: Path,
    project_root: Path,
    reference: Path | None,
    is_slides: bool,
) -> None:
    pandoc = executable("pandoc")
    if not pandoc:
        raise RenderError("Pandoc is required but was not found on PATH.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        pandoc,
        str(temporary_markdown),
        "--from",
        "gfm+raw_html+link_attributes",
        "--standalone",
        "--resource-path",
        pandoc_resource_path(source, project_root),
        "--output",
        str(destination),
    ]
    if is_slides:
        command.extend(["--slide-level", "1"])
    if reference:
        if not reference.is_file():
            raise RenderError(f"Reference template does not exist: {reference}")
        command.extend(["--reference-doc", str(reference.resolve())])
    run(command, cwd=project_root)
    validate_office_file(destination)


def validate_office_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RenderError(f"Output file was not created: {path}")
    if not zipfile.is_zipfile(path):
        raise RenderError(f"Output is not a valid OOXML ZIP package: {path}")
    required = "word/document.xml" if path.suffix.lower() == ".docx" else "ppt/presentation.xml"
    with zipfile.ZipFile(path) as archive:
        if required not in archive.namelist():
            raise RenderError(f"Output does not contain {required}: {path}")


def print_dependency_status(toolchain: DiagramToolchain) -> bool:
    pandoc = executable("pandoc")
    plantuml_ready = bool(toolchain.plantuml or (toolchain.java and toolchain.plantuml_jar))
    rows = [
        ("pandoc", pandoc or "MISSING"),
        ("mmdc", toolchain.mmdc or "MISSING"),
        ("plantuml", toolchain.plantuml or "MISSING"),
        ("java", toolchain.java or "MISSING"),
        ("PLANTUML_JAR", str(toolchain.plantuml_jar) if toolchain.plantuml_jar else "NOT SET"),
    ]
    for name, value in rows:
        print(f"{name:14} {value}")
    print(f"PlantUML ready: {'yes' if plantuml_ready else 'no'}")
    return bool(pandoc)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Render PAPER.md to DOCX and SLIDES.md to PPTX after converting diagrams to images."
    )
    result.add_argument("--paper", help="Path to PAPER.md (defaults to docs/PAPER.md, then PAPER.md)")
    result.add_argument("--slides", help="Path to SLIDES.md (defaults to docs/SLIDES.md, then SLIDES.md)")
    result.add_argument("--output-dir", default="output", help="Destination directory (default: output)")
    result.add_argument(
        "--target",
        choices=("all", "paper", "slides"),
        default="all",
        help="Generate both files or only one target",
    )
    result.add_argument("--reference-docx", help="Optional Pandoc reference DOCX")
    result.add_argument("--reference-pptx", help="Optional Pandoc reference PPTX")
    result.add_argument(
        "--diagram-format",
        choices=("svg", "png"),
        default="png",
        help="Diagram image format (default: png for broad Office compatibility)",
    )
    result.add_argument("--keep-temp", action="store_true", help="Keep rewritten Markdown and diagram files")
    result.add_argument("--check", action="store_true", help="Only report external dependency availability")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    toolchain = detect_toolchain()

    if args.check:
        return 0 if print_dependency_status(toolchain) else 1

    output_dir = (project_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    temporary_context: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_temp:
        build_dir = output_dir / "render-work"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
    else:
        temporary_context = tempfile.TemporaryDirectory(prefix="hycu-render-")
        build_dir = Path(temporary_context.name)

    try:
        if args.target in {"all", "paper"}:
            paper = resolve_source(
                args.paper,
                (project_root / "docs" / "PAPER.md", project_root / "PAPER.md"),
                "paper",
            )
            paper_temp = build_dir / "paper" / "PAPER.rendered.md"
            count = preprocess_markdown(
                paper,
                paper_temp,
                build_dir / "paper" / "diagrams",
                toolchain,
                args.diagram_format,
            )
            print(f"[render] paper diagrams rendered: {count}")
            build_with_pandoc(
                paper,
                paper_temp,
                output_dir / "PAPER.docx",
                project_root,
                Path(args.reference_docx).expanduser().resolve() if args.reference_docx else None,
                is_slides=False,
            )

        if args.target in {"all", "slides"}:
            slides = resolve_source(
                args.slides,
                (project_root / "docs" / "SLIDES.md", project_root / "SLIDES.md"),
                "slides",
            )
            slides_temp = build_dir / "slides" / "SLIDES.rendered.md"
            count = preprocess_markdown(
                slides,
                slides_temp,
                build_dir / "slides" / "diagrams",
                toolchain,
                args.diagram_format,
            )
            print(f"[render] slide diagrams rendered: {count}")
            build_with_pandoc(
                slides,
                slides_temp,
                output_dir / "SLIDES.pptx",
                project_root,
                Path(args.reference_pptx).expanduser().resolve() if args.reference_pptx else None,
                is_slides=True,
            )

        print(f"[render] completed. Output directory: {output_dir}")
        return 0
    finally:
        if temporary_context is not None:
            temporary_context.cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RenderError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

