"""Render a graph run as a Mermaid flowchart from observability events.

This handler listens for `node.start` / `node.end` events emitted by the graph
engine and reconstructs the DAG path that was actually taken (nodes plus the
`next` edge recorded on each `node.end`). Call `to_mermaid()` after a run to get
the diagram text, or pass `path=...` to write it automatically on `run.end`.

It is dependency-free; the output renders anywhere Mermaid is supported
(GitHub, many IDEs/markdown viewers, https://mermaid.live).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from cat_agent.observability.events import EventEnvelope

START = '__start__'
END = '__end__'


class MermaidExporter:
    """Collect graph node events and emit a Mermaid `flowchart`.

    Args:
        path: If set, the diagram is written to this file on every `run.end`.
        direction: Mermaid flow direction (e.g. 'TD', 'LR').
    """

    def __init__(self, path: Optional[str] = None, direction: str = 'TD') -> None:
        self.path = path
        self.direction = direction
        self._nodes: List[str] = []
        self._node_types: dict = {}
        self._edges: List[Tuple[str, str]] = []
        self._seen_edges: set = set()

    def on_event(self, event: EventEnvelope) -> None:
        et = event.event_type
        if et == 'run.start':
            self.reset()
        elif et == 'node.start':
            self._add_node(event.payload.get('node'), event.payload.get('node_type'))
        elif et == 'node.end':
            node = event.payload.get('node')
            nxt = event.payload.get('next')
            self._add_node(node, event.payload.get('node_type'))
            if not self._edges and node:
                # First executed node is the graph entry point.
                self._add_edge(START, node)
            if node and nxt:
                self._add_edge(node, nxt)
        elif et == 'run.end' and self.path:
            self.write(self.path)

    def reset(self) -> None:
        self._nodes.clear()
        self._node_types.clear()
        self._edges.clear()
        self._seen_edges.clear()

    def _add_node(self, node: Optional[str], node_type: Optional[str]) -> None:
        if not node or node in self._nodes:
            return
        self._nodes.append(node)
        if node_type:
            self._node_types[node] = node_type

    def _add_edge(self, src: str, dst: str) -> None:
        key = (src, dst)
        if key in self._seen_edges:
            return
        self._seen_edges.add(key)
        self._edges.append(key)

    @staticmethod
    def _render_node(node: str, node_type: Optional[str]) -> str:
        if node == START:
            return f'{START}([start])'
        if node == END:
            return f'{END}([end])'
        label = node if not node_type else f'{node}<br/><i>{node_type}</i>'
        return f'{node}["{label}"]'

    def to_mermaid(self) -> str:
        lines = [f'flowchart {self.direction}']
        rendered: set = set()
        for src, dst in self._edges:
            for n in (src, dst):
                if n not in rendered:
                    rendered.add(n)
            lines.append(f'    {self._render_node(src, self._node_types.get(src))} '
                         f'--> {self._render_node(dst, self._node_types.get(dst))}')
        # Include any isolated nodes that never appeared in an edge.
        for n in self._nodes:
            if n not in rendered:
                lines.append(f'    {self._render_node(n, self._node_types.get(n))}')
        return '\n'.join(lines)

    def write(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_mermaid() + '\n')
