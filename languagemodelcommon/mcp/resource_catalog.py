"""Resource catalog with BM25 Okapi search for MCP resource discovery.

Provides a searchable index of MCP resources that supports ranked retrieval
by keyword relevance. Used by the meta-discovery tools (search_resources,
read_resource).

Supports lazy resolution: servers can be registered with metadata only,
and their resources are fetched on-demand when a search matches the
server's category.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from mcp.types import Resource as MCPResource, ResourceTemplate as MCPResourceTemplate

from languagemodelcommon.configs.schemas.config_schema import AgentConfig
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)


@runtime_checkable
class ResourceResolverProtocol(Protocol):
    """Callback to lazily fetch resources from an MCP server."""

    async def resolve_resources(
        self,
        agent_config: AgentConfig,
    ) -> tuple[list[MCPResource], list[MCPResourceTemplate]]: ...


@dataclass
class ResourceServerRegistration:
    """An MCP server registered in the catalog, possibly not yet resolved."""

    server_name: str
    category: str | None
    agent_config: AgentConfig
    resolved: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class ResourceCatalogEntry:
    """A single resource in the catalog."""

    server_name: str
    resource: MCPResource
    category: str | None
    agent_config: AgentConfig


@dataclass
class ResourceTemplateCatalogEntry:
    """A single resource template in the catalog."""

    server_name: str
    template: MCPResourceTemplate
    category: str | None
    agent_config: AgentConfig


@dataclass
class _BM25Index:
    """In-memory BM25 Okapi index over resource documents."""

    k1: float = 1.5
    b: float = 0.75
    corpus: list[list[str]] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    avgdl: float = 0.0
    n_docs: int = 0
    inverted_index: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)

    def build(self, corpus: list[list[str]]) -> None:
        self.corpus = corpus
        self.n_docs = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.n_docs if self.n_docs > 0 else 0.0

        self.inverted_index = {}
        for doc_idx, doc in enumerate(corpus):
            term_freqs: dict[str, int] = {}
            for term in doc:
                term_freqs[term] = term_freqs.get(term, 0) + 1
            for term, freq in term_freqs.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, freq))

        self.idf = {}
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(
        self, query_tokens: list[str], top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Return (doc_index, score) pairs sorted by descending score."""
        scores: dict[int, float] = {}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            idf = self.idf[token]
            for doc_idx, tf in self.inverted_index[token]:
                dl = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score = idf * numerator / denominator
                scores[doc_idx] = scores.get(doc_idx, 0.0) + score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


_TOKENIZE_RE = re.compile(r"[_\-\s/.,;:]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace, underscores, hyphens, and punctuation."""
    return [t for t in _TOKENIZE_RE.split(text.lower()) if t]


def _build_resource_document(resource: MCPResource) -> list[str]:
    """Build a searchable token list from a resource's metadata."""
    parts: list[str] = []
    parts.extend(_tokenize(resource.name))
    if resource.description:
        parts.extend(_tokenize(resource.description))
    parts.extend(_tokenize(str(resource.uri)))
    if resource.mimeType:
        parts.extend(_tokenize(resource.mimeType))
    return parts


def _build_template_document(template: MCPResourceTemplate) -> list[str]:
    """Build a searchable token list from a resource template's metadata."""
    parts: list[str] = []
    parts.extend(_tokenize(template.name))
    if template.description:
        parts.extend(_tokenize(template.description))
    parts.extend(_tokenize(template.uriTemplate))
    if template.mimeType:
        parts.extend(_tokenize(template.mimeType))
    return parts


def _format_resource_schema(resource: MCPResource) -> dict[str, Any]:
    """Format a resource's metadata for search result output."""
    result: dict[str, Any] = {
        "name": resource.name,
        "uri": str(resource.uri),
        "description": resource.description or "",
    }
    if resource.mimeType:
        result["mimeType"] = resource.mimeType
    return result


def _format_template_schema(template: MCPResourceTemplate) -> dict[str, Any]:
    """Format a resource template's metadata for search result output."""
    result: dict[str, Any] = {
        "name": template.name,
        "uriTemplate": template.uriTemplate,
        "description": template.description or "",
        "type": "template",
    }
    if template.mimeType:
        result["mimeType"] = template.mimeType
    return result


class ResourceCatalog:
    """Searchable catalog of MCP resources with BM25 Okapi ranking.

    Supports two modes:
    - **Eager:** Call ``add_resources`` to populate immediately.
    - **Lazy:** Call ``register_server`` to record metadata only, then
      ``resolve_server`` (or let ``SearchResourcesTool`` do it) to fetch
      resources on-demand when a search matches the server's category.
    """

    def __init__(self) -> None:
        self._entries: list[ResourceCatalogEntry] = []
        self._template_entries: list[ResourceTemplateCatalogEntry] = []
        self._index: _BM25Index | None = None
        self._entries_by_uri: dict[str, ResourceCatalogEntry] = {}
        self._servers: dict[str, ResourceServerRegistration] = {}

    def register_server(
        self,
        *,
        server_name: str,
        category: str | None,
        agent_config: AgentConfig,
    ) -> None:
        """Register a server for lazy resolution (no MCP call yet)."""
        self._servers[server_name] = ResourceServerRegistration(
            server_name=server_name,
            category=category,
            agent_config=agent_config,
        )
        logger.info(
            "Registered resource server %s (category=%s) for lazy resolution",
            server_name,
            category,
        )

    async def resolve_server(
        self,
        server_name: str,
        resolver: ResourceResolverProtocol,
    ) -> None:
        """Resolve a registered server by fetching its resources via the resolver."""
        registration = self._servers.get(server_name)
        if registration is None or registration.resolved:
            return

        async with registration._lock:
            if registration.resolved:
                return

            logger.info("Resolving resources for server %s", server_name)
            resources, templates = await resolver.resolve_resources(
                agent_config=registration.agent_config,
            )
            self.add_resources(
                server_name=server_name,
                category=registration.category,
                resources=resources,
                templates=templates,
                agent_config=registration.agent_config,
            )
            registration.resolved = True
            logger.info(
                "Resolved %d resources and %d templates for server %s",
                len(resources),
                len(templates),
                server_name,
            )

    def get_unresolved_servers(
        self, category: str | None = None
    ) -> list[ResourceServerRegistration]:
        """Return unresolved server registrations, optionally filtered by category."""
        unresolved = [s for s in self._servers.values() if not s.resolved]
        if category is None:
            return unresolved
        return [
            s
            for s in unresolved
            if (s.category and category.lower() in s.category.lower())
            or category.lower() in s.server_name.lower()
        ]

    def add_resources(
        self,
        *,
        server_name: str,
        category: str | None,
        resources: list[MCPResource],
        templates: list[MCPResourceTemplate],
        agent_config: AgentConfig,
    ) -> None:
        """Add resources and templates from an MCP server to the catalog."""
        for resource in resources:
            entry = ResourceCatalogEntry(
                server_name=server_name,
                resource=resource,
                category=category,
                agent_config=agent_config,
            )
            self._entries.append(entry)
            self._entries_by_uri[str(resource.uri)] = entry

        for template in templates:
            self._template_entries.append(
                ResourceTemplateCatalogEntry(
                    server_name=server_name,
                    template=template,
                    category=category,
                    agent_config=agent_config,
                )
            )

        self._index = None
        logger.info(
            "Added %d resources and %d templates from %s to catalog (total: %d resources, %d templates)",
            len(resources),
            len(templates),
            server_name,
            len(self._entries),
            len(self._template_entries),
        )

    def _ensure_index(self) -> _BM25Index:
        """Lazily build or return the BM25 index over resources and templates."""
        if self._index is None:
            corpus: list[list[str]] = []
            corpus.extend(
                _build_resource_document(entry.resource) for entry in self._entries
            )
            corpus.extend(
                _build_template_document(entry.template)
                for entry in self._template_entries
            )
            self._index = _BM25Index()
            self._index.build(corpus)
            logger.info(
                "Built BM25 index over %d resources and %d templates",
                len(self._entries),
                len(self._template_entries),
            )
        return self._index

    def _get_result_at_index(self, idx: int) -> dict[str, Any]:
        """Get a formatted result from a combined index position."""
        if idx < len(self._entries):
            entry = self._entries[idx]
            return {
                **_format_resource_schema(entry.resource),
                "server_name": entry.server_name,
                "category": entry.category,
            }
        template_idx = idx - len(self._entries)
        entry_t = self._template_entries[template_idx]
        return {
            **_format_template_schema(entry_t.template),
            "server_name": entry_t.server_name,
            "category": entry_t.category,
        }

    def search(
        self,
        query: str,
        category: str | None = None,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for resources matching the query."""
        total = len(self._entries) + len(self._template_entries)
        if total == 0:
            return []

        if category:
            filtered_resources = [
                e
                for e in self._entries
                if (e.category and category.lower() in e.category.lower())
                or category.lower() in e.server_name.lower()
            ]
            filtered_templates = [
                e
                for e in self._template_entries
                if (e.category and category.lower() in e.category.lower())
                or category.lower() in e.server_name.lower()
            ]
            if not filtered_resources and not filtered_templates:
                return []

            corpus: list[list[str]] = []
            corpus.extend(
                _build_resource_document(e.resource) for e in filtered_resources
            )
            corpus.extend(
                _build_template_document(e.template) for e in filtered_templates
            )
            index = _BM25Index()
            index.build(corpus)
            query_tokens = _tokenize(query)
            ranked = index.search(query_tokens, top_k=max_results)

            results: list[dict[str, Any]] = []
            for idx, _score in ranked:
                if idx < len(filtered_resources):
                    e_r = filtered_resources[idx]
                    results.append(
                        {
                            **_format_resource_schema(e_r.resource),
                            "server_name": e_r.server_name,
                            "category": e_r.category,
                        }
                    )
                else:
                    t_idx = idx - len(filtered_resources)
                    e_t = filtered_templates[t_idx]
                    results.append(
                        {
                            **_format_template_schema(e_t.template),
                            "server_name": e_t.server_name,
                            "category": e_t.category,
                        }
                    )
            return results

        index = self._ensure_index()
        query_tokens = _tokenize(query)
        ranked = index.search(query_tokens, top_k=max_results)
        return [self._get_result_at_index(idx) for idx, _score in ranked]

    def get_resource(self, uri: str) -> ResourceCatalogEntry | None:
        """Look up a resource by exact URI."""
        return self._entries_by_uri.get(uri)

    def get_resource_agent_config(self, uri: str) -> AgentConfig | None:
        """Look up the agent config for a resource by URI.

        Falls back to checking template entries if no exact URI match,
        matching by server name from the URI prefix.
        """
        entry = self._entries_by_uri.get(uri)
        if entry is not None:
            return entry.agent_config
        # For templated URIs, find the first template entry whose server
        # has a matching URI prefix pattern.
        for t_entry in self._template_entries:
            if uri.startswith(t_entry.template.uriTemplate.split("{")[0]):
                return t_entry.agent_config
        return None

    def get_categories(self) -> list[dict[str, Any]]:
        """Get a summary of resource categories for the system prompt."""
        categories: dict[str, dict[str, Any]] = {}

        for reg in self._servers.values():
            if reg.server_name not in categories:
                categories[reg.server_name] = {
                    "name": reg.server_name,
                    "description": reg.category or reg.server_name,
                    "resource_count": 0,
                    "template_count": 0,
                    "resolved": reg.resolved,
                }

        for entry in self._entries:
            key = entry.server_name
            if key not in categories:
                categories[key] = {
                    "name": key,
                    "description": entry.category or entry.server_name,
                    "resource_count": 0,
                    "template_count": 0,
                    "resolved": True,
                }
            categories[key]["resource_count"] += 1
            categories[key]["resolved"] = True

        for entry in self._template_entries:
            key = entry.server_name
            if key not in categories:
                categories[key] = {
                    "name": key,
                    "description": entry.category or entry.server_name,
                    "resource_count": 0,
                    "template_count": 0,
                    "resolved": True,
                }
            categories[key]["template_count"] += 1
            categories[key]["resolved"] = True

        return list(categories.values())

    def list_resources(self, category: str | None = None) -> list[dict[str, Any]]:
        """List all resources, optionally filtered by category."""
        results: list[dict[str, Any]] = []

        entries = self._entries
        if category:
            entries = [
                e
                for e in entries
                if (e.category and category.lower() in e.category.lower())
                or category.lower() in e.server_name.lower()
            ]
        for e in entries:
            results.append(
                {
                    **_format_resource_schema(e.resource),
                    "server_name": e.server_name,
                    "category": e.category,
                }
            )

        templates = self._template_entries
        if category:
            templates = [
                e
                for e in templates
                if (e.category and category.lower() in e.category.lower())
                or category.lower() in e.server_name.lower()
            ]
        for e in templates:
            results.append(
                {
                    **_format_template_schema(e.template),
                    "server_name": e.server_name,
                    "category": e.category,
                }
            )

        return results

    @property
    def resource_count(self) -> int:
        return len(self._entries) + len(self._template_entries)