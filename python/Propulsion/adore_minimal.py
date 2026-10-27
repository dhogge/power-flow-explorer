"""Convert ADORE port connections into GPSA architecture matrices."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import numpy as np


NODE_ORDER = ("batt", "eng1", "fuel", "gen", "motor1", "prop1", "thrust")
SOURCES = ("batt", "fuel")
SINK = "thrust"
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "architectures/powertrain.adore"
)
CACHE_PATH = Path(__file__).resolve().parents[1] / ".adsg-cache"


def _edges(*items):
    return frozenset(items)


# These aliases make the current seven topologies readable. They do not limit
# which topologies ADORE may generate or which ones this adapter accepts.
ARCHITECTURE_EDGES = {
    "conventional": _edges(
        ("fuel", "eng1"),
        ("eng1", "prop1"),
        ("prop1", "thrust")),

    "battery_electric": _edges(
        ("batt", "motor1"),
        ("motor1", "prop1"),
        ("prop1", "thrust")
    ),
    "turboelectric": _edges(
        ("fuel", "eng1"),
        ("eng1", "gen"),
        ("gen", "motor1"),
        ("motor1", "prop1"),
        ("prop1", "thrust"),
    ),
    "series_hybrid": _edges(
        ("fuel", "eng1"),
        ("batt", "motor1"),
        ("eng1", "gen"),
        ("gen", "motor1"),
        ("motor1", "prop1"),
        ("prop1", "thrust"),
    ),
    "parallel_hybrid": _edges(
        ("fuel", "eng1"),
        ("batt", "motor1"),
        ("eng1", "prop1"),
        ("motor1", "prop1"),
        ("prop1", "thrust"),
    ),
    "mechanical_electric_parallel": _edges(
        ("fuel", "eng1"),
        ("eng1", "gen"),
        ("eng1", "prop1"),
        ("gen", "motor1"),
        ("motor1", "prop1"),
        ("prop1", "thrust"),
    ),
    "series_parallel_hybrid": _edges(
        ("fuel", "eng1"),
        ("batt", "motor1"),
        ("eng1", "gen"),
        ("eng1", "prop1"),
        ("gen", "motor1"),
        ("motor1", "prop1"),
        ("prop1", "thrust"),
    ),
}
NAME_BY_EDGES = {edges: name for name, edges in ARCHITECTURE_EDGES.items()}
VERSION_1_NAMES = ("conventional", "parallel_hybrid", "series_hybrid")


@dataclass(frozen=True)
class ArchitectureRecord:
    """A topology with a read-only matrix and one representative ADORE design.

    Serialization validates the matrix, not source-to-sink or physical feasibility.
    """

    name: str
    node_order: tuple[str, ...]
    adjacency: np.ndarray
    design_vector: tuple[float, ...] = ()
    activeness: tuple[bool, ...] = ()
    component_refs: tuple[str, ...] = ()

    def __post_init__(self):
        node_order = tuple(self.node_order)
        adjacency = np.asarray(self.adjacency)
        if len(node_order) != len(set(node_order)):
            raise ValueError("node_order contains duplicates")
        if adjacency.shape != (len(node_order), len(node_order)):
            raise ValueError("adjacency must be square and match node_order")
        if not np.all((adjacency == 0) | (adjacency == 1)):
            raise ValueError("adjacency must be binary")
        component_refs = tuple(self.component_refs) or node_order
        if len(component_refs) != len(node_order):
            raise ValueError("component_refs must match node_order")
        adjacency = adjacency.astype(np.int8, copy=True)
        adjacency.setflags(write=False)
        object.__setattr__(self, "node_order", node_order)
        object.__setattr__(self, "adjacency", adjacency)
        object.__setattr__(self, "design_vector", tuple(self.design_vector))
        object.__setattr__(self, "activeness", tuple(self.activeness))
        object.__setattr__(self, "component_refs", component_refs)

    @property
    def edges(self):
        return frozenset(
            (self.node_order[i], self.node_order[j])
            for i, j in np.argwhere(self.adjacency == 1)
        )

    def to_dict(self):
        return {
            "name": self.name,
            "node_order": list(self.node_order),
            "adjacency": self.adjacency.tolist(),
            "design_vector": list(self.design_vector),
            "activeness": list(self.activeness),
            "component_refs": list(self.component_refs),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            node_order=tuple(data["node_order"]),
            adjacency=data["adjacency"],
            design_vector=tuple(data.get("design_vector", ())),
            activeness=tuple(data.get("activeness", ())),
            component_refs=tuple(data.get("component_refs", ())),
        )


def _reachable_nodes(starts, edges):
    # ponytail: O(V*E) is fine for small graphs; index outgoing edges if this grows.
    pending = list(starts)
    visited = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend(destination for source, destination in edges if source == node)
    return visited


def _is_acyclic(nodes, edges):
    sorter = TopologicalSorter({node: set() for node in nodes})
    for source, destination in edges:
        sorter.add(destination, source)
    try:
        sorter.prepare()
    except CycleError:
        return False
    return True


def _invariant(edges, types):
    """Summarize a graph without instance names; relabelings share this value."""

    outgoing = Counter(source for source, _ in edges)
    incoming = Counter(destination for _, destination in edges)
    nodes = {node for edge in edges for node in edge}
    return (
        tuple(sorted((types[node], outgoing[node], incoming[node]) for node in nodes)),
        tuple(sorted((types[source], types[destination]) for source, destination in edges)),
    )


def _same_architecture(first, second, types):
    """Whether a type-preserving instance relabeling maps first onto second."""

    return find_isomorphism(first, second, types) is not None


def find_isomorphism(first, second, types):
    """A type-preserving relabeling {first node: second node} mapping first onto second, or None.

    Callers compare invariants first (or accept a search that fails slowly), so
    equal node profiles and edge counts make a complete edge-consistent mapping
    an isomorphism.
    """

    def profiles(edges):
        outgoing = Counter(source for source, _ in edges)
        incoming = Counter(destination for _, destination in edges)
        nodes = {node for edge in edges for node in edge}
        return {node: (types[node], outgoing[node], incoming[node]) for node in nodes}

    first_profile, second_profile = profiles(first), profiles(second)
    if sorted(first_profile.values()) != sorted(second_profile.values()):
        return None
    order = sorted(first_profile, key=lambda node: (first_profile[node], node))
    mapping, used = {}, set()

    def extend(position):
        if position == len(order):
            return True
        node = order[position]
        for candidate in sorted(second_profile):
            if candidate in used or second_profile[candidate] != first_profile[node]:
                continue
            if all(
                ((node, mapped) in first) == ((candidate, image) in second)
                and ((mapped, node) in first) == ((image, candidate) in second)
                for mapped, image in mapping.items()
            ):
                mapping[node] = candidate
                used.add(candidate)
                if extend(position + 1):
                    return True
                del mapping[node]
                used.remove(candidate)
        return False

    return dict(mapping) if extend(0) else None


def _distinct_architectures(edge_sets, types):
    """Keep the first edge set of each class of instance relabelings, in order."""

    kept = {}
    for edges in edge_sets:
        bucket = kept.setdefault(_invariant(edges, types), [])
        if not any(_same_architecture(edges, other, types) for other in bucket):
            bucket.append(edges)
    unique = {edges for bucket in kept.values() for edges in bucket}
    return [edges for edges in edge_sets if edges in unique]


class InvalidTopology(ValueError):
    """An ADORE candidate fails the adapter's complete-power-path requirement."""


def _validate_topology(edges, active_nodes, node_order, sources, sink):
    known_nodes = set(node_order)
    unknown_nodes = set(active_nodes) - known_nodes
    if unknown_nodes:
        raise ValueError(f"node_order is missing ADORE nodes: {sorted(unknown_nodes)}")

    sinks = {sink} if isinstance(sink, str) else set(sink)
    if not sinks or not sinks <= set(active_nodes):
        raise InvalidTopology(f"required sink is missing: {sink}")
    active_sources = set(sources) & set(active_nodes)
    if not active_sources:
        raise InvalidTopology("architecture has no active source")
    reachable_from_sources = _reachable_nodes(active_sources, edges)
    if not sinks <= reachable_from_sources:
        raise InvalidTopology("architecture has no source-to-sink path")
    if not _is_acyclic(active_nodes, edges):
        raise InvalidTopology("architecture contains a cycle")

    reaches_sink = _reachable_nodes(sinks, {(end, start) for start, end in edges})
    disconnected = set(active_nodes) - (reachable_from_sources & reaches_sink)
    if disconnected:
        raise InvalidTopology(
            f"components are not on a complete power path: {sorted(disconnected)}"
        )


def _instance_name(component, instance):
    if instance.index == 0:
        return component.ref
    return f"{component.ref}__{instance.index}"


def _component_instances(system, prefix=""):
    """Flatten actual instances, retaining their type and full subsystem identity."""

    for component in system.components:
        if not component.ref:
            raise ValueError("ADORE components require a ref for GPSA node names")
        for instance in component.instances:
            yield prefix + _instance_name(component, instance), component.ref, instance
    for child in system.systems:
        if not child.ref:
            raise ValueError("ADORE subsystems require a ref for GPSA node names")
        child_prefix = f"{prefix}{child.ref}__{child.index}__"
        yield from _component_instances(child, child_prefix)


def _extract_topology(architecture):
    """Read directed edges from generated ADORE output-port target IDs."""

    input_owners = {}
    output_ports = []
    active_nodes = set()

    for node, _, instance in _component_instances(architecture.system):
        if node in active_nodes:
            raise ValueError(f"duplicate ADORE component-instance name: {node}")
        active_nodes.add(node)
        for input_port in instance.input_ports:
            if input_port.id in input_owners:
                raise ValueError(f"duplicate ADORE input-port ID: {input_port.id}")
            input_owners[input_port.id] = node
        output_ports.extend((node, port) for port in instance.output_ports)

    edges = set()
    for source, output_port in output_ports:
        for target_id in output_port.target_ids:
            if target_id not in input_owners:
                raise ValueError(f"ADORE connection target not found: {target_id}")
            edges.add((source, input_owners[target_id]))

    return frozenset(edges), frozenset(active_nodes)


def _make_record(name, edges, node_order, design_vector, activeness, refs=None):
    index = {node: i for i, node in enumerate(node_order)}
    adjacency = np.zeros((len(node_order), len(node_order)), dtype=np.int8)
    for source, destination in edges:
        adjacency[index[source], index[destination]] = 1
    return ArchitectureRecord(
        name=name,
        node_order=node_order,
        adjacency=adjacency,
        design_vector=tuple(float(value) for value in design_vector),
        activeness=tuple(bool(value) for value in activeness),
        component_refs=tuple(refs[node] for node in node_order) if refs else (),
    )


def _drop_qois(system):
    for element in (system, *system.functions, *system.components):
        element.qois.clear()
    for component in system.components:
        component.instance_qois.clear()
    for child in system.systems:
        _drop_qois(child)


def _translator(model_path, keep_qois=True):
    """Load an ADORE model, keeping encoder caches in the repository."""

    try:
        import adore as ad
        from adsg_core.optimization.assign_enc import cache as adsg_cache
    except ImportError as error:
        raise ImportError(
            "ADORE is required to generate propulsion architectures"
        ) from error

    adsg_cache.user_cache_dir = lambda _: str(CACHE_PATH)
    manager = ad.GraphApiManager.from_file(str(model_path))
    if not keep_qois:
        # Topology needs no QoIs; each adds a graph node that every candidate pays for.
        _drop_qois(manager.project.design_space.system)
    manager.sync()
    return manager.get_design_problem_translator()


def valid_architectures(
    version=2,
    model_path=DEFAULT_MODEL_PATH,
    node_order=None,
    sources=SOURCES,
    sink=SINK,
    max_candidates=200_000,
):
    """Enumerate unique component-edge sets; version 1 selects the three parity cases.

    ADORE enforces component, port, and choice constraints. This adapter filters
    cycles and incomplete power paths. Malformed identities/connections raise.
    ``sink`` is one component ref or a tuple of them.
    Port arrangements and function choices sharing an edge set share one record.
    Edge sets that differ only by relabeling instances of the same component
    type (e.g. which of nine motors draw on the battery rather than the
    generator) are one architecture; the first in the sorted order is kept.
    ``max_candidates`` bounds exact enumeration; it never truncates the result.
    """

    if version not in (1, 2):
        raise ValueError("version must be 1 or 2")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    node_order = tuple(node_order) if node_order is not None else None
    sources = tuple(sources)
    sinks = (sink,) if isinstance(sink, str) else tuple(sink)
    if node_order is not None and len(node_order) != len(set(node_order)):
        raise ValueError("node_order contains duplicates")

    translator = _translator(model_path, keep_qois=False)
    candidate_count = translator.graph_processor.get_n_valid_designs()
    if candidate_count > max_candidates:
        raise RuntimeError(
            f"ADORE estimates {candidate_count:,} candidates, exceeding "
            f"max_candidates={max_candidates:,}; tighten the model constraints "
            "or explicitly raise the limit"
        )
    result = translator.graph_processor.get_all_discrete_x()
    if result is None:
        raise RuntimeError(
            "ADORE cannot enumerate this model exactly with its current encoder; "
            "tighten the bounds or constraints. No partial design space was returned."
        )
    vectors, _ = result

    topologies = {}
    component_refs = {}
    for vector in vectors:
        architecture, corrected, activeness = translator.get_architecture(vector)
        # adsg_core caches about 1 MB of graphs per design and never evicts them.
        translator.graph_processor._graph_cache.clear()
        edges, active_nodes = _extract_topology(architecture)
        refs = {
            name: ref for name, ref, _ in _component_instances(architecture.system)
        }
        active_sources = [
            name for name, ref in refs.items() if ref in sources or name in sources
        ]
        active_sinks = [
            name for name, ref in refs.items() if ref in sinks or name in sinks
        ]
        try:
            _validate_topology(
                edges,
                active_nodes,
                active_nodes if node_order is None else node_order,
                active_sources,
                active_sinks,
            )
        except InvalidTopology:
            continue
        component_refs.update(refs)
        topologies.setdefault(edges, (corrected, activeness))

    if node_order is None:
        node_order = tuple(sorted(component_refs))
    else:
        # Explicit orders may reserve absent nodes, as in the original fixtures.
        component_refs = {node: component_refs.get(node, node) for node in node_order}

    known_order = {
        edges: index for index, edges in enumerate(ARCHITECTURE_EDGES.values())
    }
    sorted_edges = sorted(
        topologies,
        key=lambda edges: (known_order.get(edges, len(known_order)), sorted(edges)),
    )
    # Named topologies sort first, so a relabeled copy never displaces one.
    sorted_edges = _distinct_architectures(sorted_edges, component_refs)
    records = []
    unnamed = 0
    for edges in sorted_edges:
        name = NAME_BY_EDGES.get(edges)
        if name is None:
            unnamed += 1
            name = f"topology_{unnamed:03d}"
        corrected, activeness = topologies[edges]
        records.append(
            _make_record(name, edges, node_order, corrected, activeness, component_refs)
        )

    if version == 2:
        return tuple(records)

    by_name = {record.name: record for record in records}
    missing = set(VERSION_1_NAMES) - by_name.keys()
    if missing:
        raise RuntimeError(
            f"ADORE did not generate Version 1 architectures: {sorted(missing)}"
        )
    return tuple(by_name[name] for name in VERSION_1_NAMES)
