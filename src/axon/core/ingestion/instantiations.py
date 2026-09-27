"""Phase 8: Instantiation detection for Axon.

Resolves object instantiation calls (JS/TS `new` expressions and Python class
calls) to target Class nodes, creating INSTANTIATES relationships with confidence
scores.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.calls import (
    _CALL_BLOCKLIST,
    _build_import_cache,
    resolve_call,
)
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.resolved import ResolvedEdge
from axon.core.ingestion.symbol_lookup import (
    FileSymbolIndex,
    build_file_symbol_index,
    build_name_index,
    find_containing_symbol,
)

logger = logging.getLogger(__name__)

_CONTAINER_LABELS: tuple[NodeLabel, ...] = (
    NodeLabel.FUNCTION,
    NodeLabel.METHOD,
    NodeLabel.CLASS,
)


def resolve_file_instantiations(
    fpd: FileParseData,
    class_index: dict[str, list[str]],
    file_sym_index: FileSymbolIndex,
    graph: KnowledgeGraph,
) -> list[ResolvedEdge]:
    """Resolve class instantiations for a single file into ResolvedEdge objects."""
    edges: list[ResolvedEdge] = []
    seen: set[str] = set()
    import_cache = _build_import_cache(fpd.file_path, graph)
    is_js_ts = fpd.language in ("javascript", "typescript", "tsx")

    for call in fpd.parse_result.calls:
        # MAJOR 4: Consult _CALL_BLOCKLIST
        if call.name in _CALL_BLOCKLIST and call.receiver not in ("self", "this"):
            continue

        # CRITICAL 1 & 2: is_new gate
        # - For JS/TS files, ONLY calls with is_new == True are constructor calls. Skip if is_new is False.
        # - For Python files, calls have is_new == False. Skip if is_new is True.
        if is_js_ts:
            if not getattr(call, "is_new", False):
                continue
        else:
            if getattr(call, "is_new", False):
                continue

        target_id, confidence = resolve_call(
            call,
            fpd.file_path,
            class_index,
            graph,
            import_cache=import_cache,
        )

        if target_id is None:
            continue

        target_node = graph.get_node(target_id)
        if target_node is None or target_node.label != NodeLabel.CLASS:
            continue

        # MAJOR 5: Log ambiguous class matches
        candidates = class_index.get(call.name, [])
        if len(candidates) > 1:
            logger.warning(
                "Ambiguous class match for '%s' in %s at line %d; candidates: %s (selected: %s)",
                call.name,
                fpd.file_path,
                call.line,
                candidates,
                target_id,
            )

        source_id = find_containing_symbol(
            call.line, fpd.file_path, file_sym_index
        )
        if source_id is None:
            source_id = generate_id(NodeLabel.FILE, fpd.file_path)

        rel_id = f"instantiates:{source_id}->{target_id}"
        if rel_id in seen:
            continue
        seen.add(rel_id)

        edges.append(
            ResolvedEdge(
                rel_id=rel_id,
                rel_type=RelType.INSTANTIATES,
                source=source_id,
                target=target_id,
                properties={
                    "confidence": confidence,
                    "line": call.line,
                },
            )
        )

    return edges


def process_instantiations(
    parse_data: list[FileParseData],
    graph: KnowledgeGraph,
    name_index: dict[str, list[str]] | None = None,
    *,
    parallel: bool = False,
    collect: bool = False,
) -> list[ResolvedEdge] | None:
    """Resolve object instantiations and create INSTANTIATES relationships in the graph."""
    if name_index is not None:
        class_index: dict[str, list[str]] = {}
        for name, ids in name_index.items():
            filtered = [
                nid
                for nid in ids
                if (n := graph.get_node(nid)) is not None
                and n.label == NodeLabel.CLASS
            ]
            if filtered:
                class_index[name] = filtered
    else:
        class_index = build_name_index(graph, (NodeLabel.CLASS,))

    file_sym_index = build_file_symbol_index(graph, _CONTAINER_LABELS)

    per_file_edges: list[list[ResolvedEdge]] = []
    # MAJOR 3: Error boundary around per-file resolution
    if parallel and len(parse_data) > 1:
        workers = min(os.cpu_count() or 4, 8, len(parse_data))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    resolve_file_instantiations,
                    fpd,
                    class_index,
                    file_sym_index,
                    graph,
                )
                for fpd in parse_data
            ]
            for fpd, future in zip(parse_data, futures):
                try:
                    per_file_edges.append(future.result())
                except Exception as exc:
                    logger.warning(
                        "Error resolving instantiations in %s: %s",
                        fpd.file_path,
                        exc,
                        exc_info=True,
                    )
    else:
        for fpd in parse_data:
            try:
                per_file_edges.append(
                    resolve_file_instantiations(fpd, class_index, file_sym_index, graph)
                )
            except Exception as exc:
                logger.warning(
                    "Error resolving instantiations in %s: %s",
                    fpd.file_path,
                    exc,
                    exc_info=True,
                )

    seen: set[str] = set()
    deduped: list[ResolvedEdge] = []
    for file_edges in per_file_edges:
        for edge in file_edges:
            if edge.rel_id not in seen:
                seen.add(edge.rel_id)
                deduped.append(edge)

    if collect:
        return deduped

    for edge in deduped:
        graph.add_relationship(
            GraphRelationship(
                id=edge.rel_id,
                type=edge.rel_type,
                source=edge.source,
                target=edge.target,
                properties=edge.properties,
            )
        )
    return None
