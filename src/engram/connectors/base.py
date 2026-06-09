"""Connector framework (FR-C17) — pluggable knowledge sources.

A Connector yields Documents (text + provenance metadata) from a source. The
core ingests them into RAG (and optionally queues consolidation). Implement this
interface to add a source (files, Confluence, a DB, a code repo, ...).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)   # e.g. {"source": "...", "name": "..."}


class Connector(ABC):
    @abstractmethod
    def documents(self) -> Iterator[Document]:
        """Yield Documents from the source."""
