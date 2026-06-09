"""File + directory connectors. Reads text/markdown/code/html/pdf.

  pdf  -> needs the docs extra: pip install "engram[docs]"  (pypdf)
  html -> uses beautifulsoup4 if present, else a regex tag-strip
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

from engram.connectors.base import Connector, Document

TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".csv", ".log", ".tsv",
             ".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".cc", ".cpp",
             ".h", ".hpp", ".rb", ".php", ".sh", ".sql", ".json", ".yaml", ".yml", ".toml"}
RICH_EXTS = {".html", ".htm", ".pdf"}
SUPPORTED = TEXT_EXTS | RICH_EXTS


def _strip_html(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    except ImportError:
        return re.sub(r"<[^>]+>", " ", html)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise ImportError('PDF support needs the docs extra: pip install "engram[docs]"') from e
    return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)


def read_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return _strip_html(raw) if ext in (".html", ".htm") else raw


class FileConnector(Connector):
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def documents(self) -> Iterator[Document]:
        t = read_text(self.path)
        if t and t.strip():
            yield Document(text=t, metadata={"source": str(self.path), "name": self.path.name})


class DirectoryConnector(Connector):
    def __init__(self, path: str | os.PathLike, recursive: bool = True, exts: set | None = None):
        self.path = Path(path)
        self.recursive = recursive
        self.exts = exts or SUPPORTED

    def documents(self) -> Iterator[Document]:
        it = self.path.rglob("*") if self.recursive else self.path.glob("*")
        for p in sorted(it):
            if p.is_file() and p.suffix.lower() in self.exts:
                try:
                    t = read_text(p)
                except Exception:                       # skip unreadable files
                    continue
                if t and t.strip():
                    yield Document(text=t, metadata={"source": str(p), "name": p.name})


def to_connector(source) -> Connector:
    """Accept a Connector, or a path (file or directory)."""
    if isinstance(source, Connector):
        return source
    p = Path(source)
    return DirectoryConnector(p) if p.is_dir() else FileConnector(p)
