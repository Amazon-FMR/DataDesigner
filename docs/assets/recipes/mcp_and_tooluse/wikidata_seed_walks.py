# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "duckdb",
#     "numpy",
#     "pyarrow",
# ]
# ///
"""Generate search-agent seeds at scale by random walks over a local Wikidata store.

Produces the seed format that ``search_agent.py`` consumes::

    {"seed_entity", "final_answer_entity", "readable_path", "num_hops_in_graph", "ground_truth"}

Input is the DuckDB projection of the Wikidata JSON dump built by
``search_synthetic_data/qlever/`` (tables: ``entity``, ``edge``, ``year``,
``quantity``, ``coord``). A first pass projects that store into a much smaller
walkable graph, cached in its own DuckDB file; subsequent runs reuse the cache.

Walk policy, in rough order of how much it matters to the riddles built downstream:

* **Seeds are stratified by entity type.** Uniform sampling over the graph yields
  40% humans and 30% French communes; the buckets below give films, companies,
  landmarks, artworks and the rest a fixed share each.
* **Hops prefer near-functional edges.** A clue like "the university this person
  attended" only pins one node if that (entity, property) pair has few values, so
  edge weight falls off with the fan-out of the pair and pairs above
  ``MAX_FANOUT`` are dropped entirely.
* **Paths stay off the highway.** Containment and adjacency hops are capped at
  ``MAX_GEO_HOPS`` and no intermediate may be a household name, which is what
  stops every walk from draining into "the United States -> capital -> ...".
* **The closing hop is chosen under its own policy.** It decides what kind of
  thing the answer is, so it is selected separately, with a tighter share cap and
  a requirement that the answer label be unambiguous enough to grade by string
  match.

Usage::

    python wikidata_seed_walks.py --output seeds_30k.jsonl --num-records 30000
    python wikidata_seed_walks.py --output seeds.jsonl --num-records 500 --rebuild-graph
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa

DEFAULT_STORE = Path("/opt/dlami/nvme/cyongzho/wikidata/wikidata.duckdb")
DEFAULT_GRAPH = Path("/opt/dlami/nvme/cyongzho/seedgen/walkgraph.duckdb")

# ---------------------------------------------------------------------------
# relations: pid -> (readable phrase, sampling weight, may be the closing hop)
# ---------------------------------------------------------------------------

RELATIONS: dict[str, tuple[str, float, bool]] = {
    # creation / authorship
    "P50": ("written by", 1.0, True),
    "P57": ("directed by", 1.2, True),
    "P58": ("screenplay written by", 0.8, True),
    "P86": ("music composed by", 0.9, True),
    "P170": ("created by", 1.0, True),
    "P84": ("designed by the architect", 1.0, True),
    "P1040": ("edited by", 0.5, True),
    "P162": ("produced by", 0.7, True),
    "P272": ("produced by the company", 0.8, True),
    "P123": ("published by", 0.8, True),
    "P175": ("performed by", 0.9, True),
    "P264": ("released on the record label", 0.8, True),
    "P449": ("originally aired on", 0.8, True),
    "P179": ("part of the series", 0.7, True),
    "P144": ("based on", 0.9, True),
    "P161": ("features the actor", 0.7, True),
    "P800": ("known for the work", 0.9, True),
    "P112": ("founded by", 1.0, True),
    "P169": ("chief executive officer", 1.2, True),
    "P488": ("chaired by", 0.7, True),
    "P286": ("head coach", 0.7, True),
    "P664": ("organized by", 0.6, True),
    # people
    "P22": ("father", 1.0, True),
    "P25": ("mother", 1.0, True),
    "P26": ("spouse", 1.0, True),
    "P40": ("child", 0.9, True),
    "P3373": ("sibling", 0.8, True),
    "P1038": ("relative", 0.5, True),
    "P451": ("partner", 0.5, True),
    "P184": ("doctoral advisor", 0.9, True),
    "P185": ("doctoral student", 0.8, True),
    "P802": ("taught the student", 0.7, True),
    "P1066": ("studied under", 0.9, True),
    "P737": ("influenced by", 0.9, True),
    # affiliation
    "P69": ("educated at", 1.4, True),
    "P108": ("employed by", 1.1, True),
    "P54": ("played for", 1.0, True),
    "P102": ("member of the political party", 0.7, True),
    "P463": ("member of", 0.7, False),
    "P1416": ("affiliated with", 0.6, True),
    "P118": ("competed in the league", 0.6, True),
    "P127": ("owned by", 0.9, True),
    "P749": ("parent organization", 0.9, True),
    "P355": ("has the subsidiary", 0.7, True),
    "P1830": ("owner of", 0.3, True),
    # places
    "P19": ("born in", 1.4, True),
    "P20": ("died in", 1.0, True),
    "P119": ("buried at", 0.9, True),
    "P551": ("resided in", 0.7, True),
    "P937": ("worked in", 0.7, True),
    "P131": ("located in", 0.9, False),
    "P17": ("located in the country", 0.7, False),
    "P159": ("headquartered in", 1.0, True),
    "P276": ("located at", 0.8, True),
    "P740": ("formed in", 0.8, True),
    "P495": ("originated in the country", 0.5, False),
    "P36": ("capital", 0.6, True),
    "P1376": ("capital of", 0.7, False),
    "P47": ("shares a border with", 0.6, False),
    "P138": ("named after", 0.9, True),
    "P206": ("located next to the body of water", 0.7, False),
    "P403": ("flows into", 0.7, True),
    "P115": ("home venue", 0.9, True),
    "P361": ("part of", 0.6, False),
    # events / honours: usable mid-path, weak as an answer
    "P166": ("received the award", 0.9, False),
    "P1346": ("won by", 0.7, True),
    "P1344": ("participated in", 0.6, False),
    "P607": ("fought in the conflict", 0.6, False),
    "P39": ("held the position", 0.5, False),
    "P1027": ("conferred by", 0.8, True),
}

# Containment and adjacency hops chain forever ("located in" x4) and drift towards
# countries, so a path may use at most MAX_GEO_HOPS of them.
GEO_CONTAINER_PIDS = frozenset({"P131", "P17", "P361", "P47", "P495", "P36", "P1376", "P206", "P403"})

# ---------------------------------------------------------------------------
# graph projection
# ---------------------------------------------------------------------------

# Descriptions and labels that mark a page as not an entity worth asking about.
BAD_DESCRIPTION = r"wikimedia|wikipedia|wikinews|disambiguation|category of|template for"
BAD_LABEL = (
    r"^(list of |lists of |category:|template:|index of |outline of |timeline of |"
    r"glossary of |bibliography of |history of |portal:|draft:)"
)
# A qid used this many times as a P31 value is a class, not an entity.
CLASS_INSTANCE_THRESHOLD = 40
GRAPH_MIN_SITELINKS = 2

# ---------------------------------------------------------------------------
# seed buckets: ordered rules matched against an entity's P31 class labels
# ---------------------------------------------------------------------------

SEED_BUCKETS: list[tuple[str, str, int, int]] = [
    # (bucket, pattern over class labels, relative share, min sitelinks)
    # The floors trade seed recognisability against pool size: at 30k records the
    # small buckets (art, school, event) are drained outright, so they sit near the
    # bottom of what still counts as a documented entity.
    ("person", r"\b(human)\b", 18, 20),
    ("film", r"\b(film|feature film|animated film|documentary film)\b", 9, 12),
    ("tv", r"\b(television series|television program|anime television series|web series)\b", 4, 10),
    ("book", r"\b(literary work|novel|book|written work|play|poem|short story|manga|comic)\b", 5, 10),
    ("music_release", r"\b(album|single|studio album|song|musical work|composition|opera|symphony)\b", 5, 10),
    ("music_group", r"\b(musical group|band|rock band|musical ensemble|orchestra|duo)\b", 4, 10),
    (
        "company",
        r"\b(business|enterprise|public company|company|brand|corporation|bank|airline|publisher|"
        r"record label|automobile manufacturer)\b",
        9,
        10,
    ),
    (
        "school",
        r"\b(university|college|public university|private university|higher education institution|"
        r"school|research institute|academy|business school)\b",
        5,
        10,
    ),
    (
        "sports_team",
        r"\b(football club|association football club|sports team|baseball team|basketball team|"
        r"ice hockey team|rugby team|national association football team)\b",
        5,
        10,
    ),
    ("city", r"\b(city|big city|town|capital|human settlement|metropolis|county seat|city or town|borough)\b", 9, 20),
    (
        "country",
        r"\b(country|sovereign state|state|federal state|province|region|department|county|island country)\b",
        3,
        30,
    ),
    (
        "landmark",
        r"\b(museum|art museum|building|church building|cathedral|castle|palace|monument|memorial|"
        r"library|theatre|opera house|skyscraper|bridge|lighthouse|archaeological site|temple|"
        r"mosque|synagogue|stadium|hotel)\b",
        8,
        10,
    ),
    (
        "nature",
        r"\b(river|mountain|lake|island|national park|volcano|waterfall|desert|glacier|forest|bay|"
        r"cave|mountain range|protected area)\b",
        5,
        12,
    ),
    (
        "event",
        r"\b(battle|war|treaty|revolution|conflict|siege|expedition|earthquake|conference|summit|"
        r"festival|exhibition|championship|tournament)\b",
        4,
        12,
    ),
    ("art", r"\b(painting|sculpture|artwork|photograph|mural|fresco|drawing|statue)\b", 3, 10),
    (
        "tech",
        r"\b(programming language|software|video game|operating system|spacecraft|aircraft|"
        r"automobile model|weapon|invention|scientific theory|chemical compound|space telescope|"
        r"particle accelerator|rocket|satellite)\b",
        5,
        10,
    ),
    (
        "org",
        r"\b(organization|nonprofit organization|political party|international organization|"
        r"government agency|trade union|learned society|award|prize|newspaper|magazine|journal|"
        r"radio station|television channel|hospital|observatory|laboratory|zoo|botanical garden|"
        r"opera company)\b",
        6,
        10,
    ),
]

# Classes that make a bad seed regardless of anything else.
SEED_CLASS_DENY = re.compile(
    r"\b(commune|municipality|delegated commune|olympic delegation|sports season|"
    r"season|recurring sporting event|sporting event edition|village|hamlet|"
    r"civil parish|census-designated place|ward|parish|urban area|"
    r"administrative territorial entity|railway station|metro station|road|"
    r"highway|street|list|taxon|gene|protein|chemical element|number|"
    r"calendar year|decade|century|surname|given name|family name|"
    r"unincorporated community|neighborhood|quarter|locality|"
    r"electoral district|constituency|legislative term|election)\b",
    re.I,
)
SEED_MIN_SITELINKS = 8
SEED_MIN_OUT_DEGREE = 3
SEED_OVERSAMPLE = 3.5

# ---------------------------------------------------------------------------
# walk policy
# ---------------------------------------------------------------------------

GENERIC_ANSWER = re.compile(
    r"^(the |a |an )?(list|people|men|women|human|person|technology|science|art|music|"
    r"history|culture|society|government|language|literature|film|television|sport|"
    r"unknown|none|other|various|male|female|city|town|village|country|state)$",
    re.I,
)
NOT_A_NAME = re.compile(r"^[\d\W]+$")

ANSWER_MIN_SITELINKS = 8
ANSWER_MAX_SITELINKS = 200
HOP_MIN_SITELINKS = 3
# Above this a node is a household name; a riddle routed through "the United States",
# or answered by "Washington, D.C.", is guessable without searching. The seed itself
# is exempt, since the opening entity is meant to be recognisable.
HOP_MAX_SITELINKS = 200
MAX_FANOUT = 6
MAX_SAME_PID = 2
MAX_GEO_HOPS = 2
MAX_ANSWER_REUSE = 3
# A runner-up sharing the answer's label must be this much less notable, or the
# answer string is ambiguous ("Wellington" the city vs. the village).
ANSWER_LABEL_DOMINANCE = 0.6
MIN_HOPS = 3
# Every walk runs out to MAX_HOPS and then picks where to close, so the hop count is
# set by this prior rather than by a target length drawn up front. The riddle prompt
# downstream asks for 4-8 meaningful hops and drops the weak ones, so aim long.
MAX_HOPS = 6
HOP_COUNT_PRIOR = {3: 0.08, 4: 0.25, 5: 0.32, 6: 0.35}
# Soft share caps, in the spirit of the pipeline's DiversityTracker: once enough
# hops are recorded, any relation running above its share is deprioritised.
DIVERSITY_MIN_HOPS = 2_000
DIVERSITY_SHARE_CAP = 0.07
DIVERSITY_PENALTY = 0.15
TERMINAL_MIN_WALKS = 400
TERMINAL_SHARE_CAP = 0.06
TERMINAL_PENALTY = 0.04
# Down-weighting the closing hop is not enough on its own: at a late cut point the
# only admissible closing relation is often "named after", so an over-quota closing
# hop is also mostly rejected outright and the seed is walked again.
TERMINAL_REJECT_KEEP = 0.25


def build_graph(store: Path, out: Path) -> None:
    """Project the full store into the small walkable graph, cached in its own file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = duckdb.connect(str(out))
    con.execute("PRAGMA threads=64")
    con.execute(f"ATTACH '{store}' AS wd (READ_ONLY)")

    t0 = time.time()
    con.execute(
        f"""
        CREATE TABLE class_qid AS
        SELECT DISTINCT src AS qid FROM wd.edge WHERE pid = 'P279'
        UNION
        SELECT dst AS qid FROM wd.edge WHERE pid = 'P31' AND truthy
        GROUP BY dst HAVING count(*) >= {CLASS_INSTANCE_THRESHOLD}
        """
    )
    con.execute(
        f"""
        CREATE TABLE node AS
        SELECT e.qid, e.label, e.description, e.enwiki, e.sitelinks
        FROM wd.entity e
        WHERE e.enwiki IS NOT NULL
          AND e.label IS NOT NULL
          AND length(e.label) BETWEEN 2 AND 90
          AND coalesce(e.sitelinks, 0) >= {GRAPH_MIN_SITELINKS}
          AND NOT regexp_matches(lower(e.label), '{BAD_LABEL}')
          AND (e.description IS NULL OR NOT regexp_matches(lower(e.description), '{BAD_DESCRIPTION}'))
          AND e.qid NOT IN (SELECT qid FROM class_qid)
        """
    )
    pids = ", ".join(f"'{p}'" for p in RELATIONS)
    con.execute(
        f"""
        CREATE TABLE gedge AS
        SELECT DISTINCT e.src, e.pid, e.dst
        FROM wd.edge e
        WHERE e.truthy AND e.pid IN ({pids})
          AND e.src <> e.dst
          AND e.src IN (SELECT qid FROM node)
          AND e.dst IN (SELECT qid FROM node)
        """
    )
    con.execute(
        """
        CREATE TABLE node_type AS
        SELECT e.src AS qid, e.dst AS class_qid
        FROM wd.edge e
        WHERE e.pid = 'P31' AND e.truthy AND e.src IN (SELECT qid FROM node)
        """
    )
    con.execute(
        """
        CREATE TABLE class_label AS
        SELECT qid, label FROM wd.entity WHERE qid IN (SELECT DISTINCT class_qid FROM node_type)
        """
    )
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("node", "gedge")}
    con.close()
    print(f"built graph: {counts['node']:,} nodes, {counts['gedge']:,} edges ({time.time() - t0:.0f}s)")


def col_np(table: pa.Table, name: str) -> np.ndarray:
    """Materialise one Arrow column as a plain numpy array."""
    return table[name].combine_chunks().to_numpy(zero_copy_only=False)


def hub_factor(sitelinks: np.ndarray) -> np.ndarray:
    """Down-weight walking into famous nodes without forbidding it."""
    return 1.0 / (1.0 + np.maximum(sitelinks.astype(np.float32) - 40.0, 0.0) / 25.0)


class Graph:
    """CSR view of the projected graph, plus the node attributes walks need."""

    def __init__(self, db: Path) -> None:
        con = duckdb.connect(str(db), read_only=True)
        con.execute("PRAGMA threads=64")
        t0 = time.time()

        nodes = con.execute("SELECT qid, label, sitelinks FROM node ORDER BY qid").fetch_arrow_table()
        self.qid = col_np(nodes, "qid")
        self.label: list[str] = nodes["label"].to_pylist()
        self.sitelinks = col_np(nodes, "sitelinks").astype(np.int32)

        pids = ", ".join(f"'{p}'" for p in RELATIONS)
        edges = con.execute(
            f"""
            WITH f AS (
                SELECT src, pid, dst, count(*) OVER (PARTITION BY src, pid) AS fanout
                FROM gedge WHERE pid IN ({pids})
            )
            SELECT src, pid, dst, fanout FROM f WHERE fanout <= {MAX_FANOUT} ORDER BY src
            """
        ).fetch_arrow_table()
        src = np.searchsorted(self.qid, col_np(edges, "src"))
        dst = np.searchsorted(self.qid, col_np(edges, "dst"))
        fanout = col_np(edges, "fanout").astype(np.float32)

        self.pids = list(RELATIONS)
        self.phrase = [RELATIONS[p][0] for p in self.pids]
        self.rel_weight = np.array([RELATIONS[p][1] for p in self.pids], dtype=np.float32)
        self.terminal_ok = np.array([RELATIONS[p][2] for p in self.pids], dtype=bool)
        self.is_geo = np.array([p in GEO_CONTAINER_PIDS for p in self.pids], dtype=bool)
        code_of = {p: i for i, p in enumerate(self.pids)}
        pid_code = np.array([code_of[p] for p in edges["pid"].to_pylist()], dtype=np.int8)

        order = np.argsort(src, kind="stable")
        self.e_dst = dst[order]
        self.e_pid = pid_code[order]
        self.e_weight = self.rel_weight[self.e_pid] / fanout[order] ** 1.5 * hub_factor(self.sitelinks[self.e_dst])
        self.indptr = np.searchsorted(src[order], np.arange(len(self.qid) + 1))

        # Two nodes may carry the same label ("Spirit Lake" the town and the lake);
        # a path that appears to revisit a node reads as broken, so labels are
        # compared by code rather than by qid.
        codes: dict[str, int] = {}
        self.label_code = np.empty(len(self.qid), dtype=np.int32)
        for i, lab in enumerate(self.label):
            self.label_code[i] = codes.setdefault(lab.lower(), len(codes))

        named = np.array(
            [
                bool(lab) and lab[0].isupper() and not NOT_A_NAME.match(lab) and not GENERIC_ANSWER.match(lab)
                for lab in self.label
            ]
        )
        self.unambiguous = self._unambiguous(con)
        self.hop_ok = (self.sitelinks >= HOP_MIN_SITELINKS) & (self.sitelinks <= HOP_MAX_SITELINKS)
        self.answer_ok = (
            named
            & self.unambiguous
            & (self.sitelinks >= ANSWER_MIN_SITELINKS)
            & (self.sitelinks <= ANSWER_MAX_SITELINKS)
        )
        con.close()
        print(
            f"loaded {len(self.qid):,} nodes, {len(self.e_dst):,} edges, "
            f"{int(self.answer_ok.sum()):,} answer-eligible nodes ({time.time() - t0:.0f}s)"
        )

    def _unambiguous(self, con: duckdb.DuckDBPyConnection) -> np.ndarray:
        """Nodes that own their label outright, so a string match can grade them."""
        rows = con.execute(
            f"""
            SELECT qid FROM (
                SELECT qid,
                       row_number() OVER (PARTITION BY lower(label) ORDER BY sitelinks DESC) AS rk,
                       max(sitelinks) OVER (PARTITION BY lower(label)) AS best,
                       coalesce(nth_value(sitelinks, 2) OVER (
                           PARTITION BY lower(label) ORDER BY sitelinks DESC
                           ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                       ), 0) AS runner_up
                FROM node
            )
            WHERE rk = 1 AND runner_up < {ANSWER_LABEL_DOMINANCE} * best
            """
        ).fetch_arrow_table()
        flags = np.zeros(len(self.qid), dtype=bool)
        flags[np.searchsorted(self.qid, col_np(rows, "qid"))] = True
        return flags

    def out_edges(self, node: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, hi = self.indptr[node], self.indptr[node + 1]
        return self.e_dst[lo:hi], self.e_pid[lo:hi], self.e_weight[lo:hi]


def sample_seeds(db: Path, graph: Graph, n: int, rng: np.random.Generator) -> list[int]:
    """Pick seed nodes, stratified over the entity-type buckets."""
    con = duckdb.connect(str(db), read_only=True)
    con.execute("PRAGMA threads=64")
    rows = con.execute(
        f"""
        WITH deg AS (SELECT src AS qid, count(*) AS d FROM gedge GROUP BY src),
        cls AS (
            SELECT t.qid, lower(string_agg(cl.label, '; ')) AS classes
            FROM node_type t JOIN class_label cl ON cl.qid = t.class_qid
            GROUP BY t.qid
        )
        SELECT n.qid, n.sitelinks, coalesce(cls.classes, '') AS classes
        FROM node n JOIN deg ON deg.qid = n.qid LEFT JOIN cls ON cls.qid = n.qid
        WHERE n.sitelinks >= {SEED_MIN_SITELINKS} AND deg.d >= {SEED_MIN_OUT_DEGREE}
        """
    ).fetch_arrow_table()
    con.close()

    qids = col_np(rows, "qid")
    sitelinks = col_np(rows, "sitelinks")
    classes = rows["classes"].to_pylist()
    rules = [(bucket, re.compile(pat, re.I), share, min_sl) for bucket, pat, share, min_sl in SEED_BUCKETS]
    pools: dict[str, list[tuple[int, int]]] = {bucket: [] for bucket, *_ in SEED_BUCKETS}
    keep = graph.unambiguous[np.searchsorted(graph.qid, qids)]
    for qid, sl, cl, ok in zip(qids, sitelinks, classes, keep):
        if not ok or not cl or SEED_CLASS_DENY.search(cl):
            continue
        for bucket, pat, _share, min_sl in rules:
            if sl >= min_sl and pat.search(cl):
                pools[bucket].append((int(qid), int(sl)))
                break

    total_share = sum(share for _, _, share, _ in SEED_BUCKETS)
    seeds: list[int] = []
    for bucket, _pat, share, _min_sl in rules:
        pool = pools[bucket]
        want = min(round(n * SEED_OVERSAMPLE * share / total_share), len(pool))
        print(f"  {bucket:<13} pool={len(pool):>7,} sampled={want:>6,}")
        if not want:
            continue
        idx = np.searchsorted(graph.qid, np.array([q for q, _ in pool]))
        # Lean towards the more notable end of each bucket without collapsing onto
        # the same few hundred household names.
        p = np.log1p(np.array([sl for _, sl in pool], dtype=np.float64)) ** 2
        p /= p.sum()
        seeds.extend(int(i) for i in rng.choice(idx, size=want, replace=False, p=p))
    rng.shuffle(seeds)
    return seeds


class Walker:
    """Random walks under the diversity, uniqueness and notability caps."""

    def __init__(self, graph: Graph, rng: np.random.Generator) -> None:
        self.g = graph
        self.rng = rng
        self.pid_hits: Counter[int] = Counter()
        self.terminal_hits: Counter[int] = Counter()
        self.answer_uses: Counter[int] = Counter()
        self.total_hops = 0
        self.total_walks = 0

    def _share_penalty(self, hits: Counter[int], total: int, floor: int, cap: float, penalty: float) -> np.ndarray:
        pen = np.ones(len(self.g.pids), dtype=np.float32)
        if total >= floor:
            for code, seen in hits.items():
                if seen / total > cap:
                    pen[code] = penalty
        return pen

    def _admissible(
        self, dst: np.ndarray, pid: np.ndarray, visited: set[int], label_codes: set[int], pids: list[int], geo_hops: int
    ) -> np.ndarray:
        """Mask of out-edges a walk in this state is allowed to take."""
        g = self.g
        keep = g.hop_ok[dst] & ~np.isin(dst, list(visited)) & ~np.isin(g.label_code[dst], list(label_codes))
        for code, seen in Counter(pids).items():
            if seen >= MAX_SAME_PID:
                keep &= pid != code
        if geo_hops >= MAX_GEO_HOPS:
            keep &= ~g.is_geo[pid]
        return keep

    def _choose(self, weight: np.ndarray) -> int | None:
        total = weight.sum()
        if total <= 0:
            return None
        return int(self.rng.choice(len(weight), p=weight / total))

    def walk(self, seed: int) -> tuple[list[int], list[int]] | None:
        """Walk out to MAX_HOPS, then close under the terminal policy."""
        g = self.g
        hop_pen = self._share_penalty(
            self.pid_hits, self.total_hops, DIVERSITY_MIN_HOPS, DIVERSITY_SHARE_CAP, DIVERSITY_PENALTY
        )
        nodes = [seed]
        pids: list[int] = []
        visited = {seed}
        label_codes = {int(g.label_code[seed])}
        geo_hops = 0

        for _ in range(MAX_HOPS - 1):
            dst, pid, w = g.out_edges(nodes[-1])
            if len(dst) == 0:
                break
            keep = self._admissible(dst, pid, visited, label_codes, pids, geo_hops)
            if not keep.any():
                break
            pick = self._choose(w[keep] * hop_pen[pid[keep]])
            if pick is None:
                break
            nodes.append(int(dst[keep][pick]))
            pids.append(int(pid[keep][pick]))
            visited.add(nodes[-1])
            label_codes.add(int(g.label_code[nodes[-1]]))
            geo_hops += int(g.is_geo[pids[-1]])

        return self._close(nodes, pids, visited, label_codes, geo_hops)

    def _close(
        self, nodes: list[int], pids: list[int], visited: set[int], label_codes: set[int], geo_hops: int
    ) -> tuple[list[int], list[int]] | None:
        """Add the closing hop, which may branch off any node the walk passed through.

        The closing hop fixes what kind of thing the answer is, so it is never
        inherited from an intermediate choice. Candidates are pooled over every
        admissible ending position rather than taken from the far end first: a walk
        that only reaches "named after" at its tip usually offers a person or an
        organisation one node back, which is what gives the share cap something to
        choose between.
        """
        g = self.g
        term_pen = self._share_penalty(
            self.terminal_hits, self.total_walks, TERMINAL_MIN_WALKS, TERMINAL_SHARE_CAP, TERMINAL_PENALTY
        )
        cuts: list[int] = []
        dsts: list[np.ndarray] = []
        cand_pids: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for cut in range(MIN_HOPS - 1, len(nodes)):
            prefix_nodes, prefix_pids = nodes[: cut + 1], pids[:cut]
            dst, pid, w = g.out_edges(prefix_nodes[-1])
            if len(dst) == 0:
                continue
            keep = self._admissible(
                dst,
                pid,
                set(prefix_nodes),
                {int(g.label_code[i]) for i in prefix_nodes},
                prefix_pids,
                sum(int(g.is_geo[p]) for p in prefix_pids),
            )
            keep &= g.answer_ok[dst] & g.terminal_ok[pid]
            # a path needs at least two distinct relations to read as a chain
            if len(set(prefix_pids)) < 2:
                keep &= ~np.isin(pid, prefix_pids)
            if not keep.any():
                continue
            reuse = np.array([self.answer_uses[int(d)] for d in dst[keep]], dtype=np.float32)
            weight = w[keep] * term_pen[pid[keep]] * (reuse < MAX_ANSWER_REUSE)
            if weight.sum() <= 0:
                continue
            # Normalise within the cut, then scale by the prior for the hop count it
            # produces; otherwise short paths win purely by having more branches.
            weight = weight / weight.sum() * HOP_COUNT_PRIOR.get(cut + 1, 0.0)
            cuts.append(cut)
            dsts.append(dst[keep])
            cand_pids.append(pid[keep])
            weights.append(weight)

        if not cuts:
            return None
        offsets = np.cumsum([0, *(len(w) for w in weights)])
        pick = self._choose(np.concatenate(weights))
        if pick is None:
            return None
        slot = int(np.searchsorted(offsets, pick, side="right")) - 1
        local = pick - offsets[slot]
        cut = cuts[slot]
        return (
            [*nodes[: cut + 1], int(dsts[slot][local])],
            [*pids[:cut], int(cand_pids[slot][local])],
        )

    def over_terminal_share(self, pid: int) -> bool:
        if self.total_walks < TERMINAL_MIN_WALKS:
            return False
        return self.terminal_hits[pid] / self.total_walks > TERMINAL_SHARE_CAP

    def record(self, nodes: list[int], pids: list[int]) -> None:
        self.pid_hits.update(pids)
        self.terminal_hits[pids[-1]] += 1
        self.answer_uses[nodes[-1]] += 1
        self.total_hops += len(pids)
        self.total_walks += 1


def render(graph: Graph, nodes: list[int], pids: list[int]) -> dict[str, Any]:
    lines = [f"START ENTITY: {graph.label[nodes[0]]}"]
    for pid, node in zip(pids, nodes[1:]):
        lines.append(f"  -> [{graph.phrase[pid]}]")
        lines.append(f"  NODE: {graph.label[node]}")
    answer = graph.label[nodes[-1]]
    return {
        "seed_entity": graph.label[nodes[0]],
        "final_answer_entity": answer,
        "readable_path": "\n".join(lines),
        "num_hops_in_graph": len(pids),
        "ground_truth": answer,
    }


def generate(graph: Graph, seeds: list[int], n: int, attempts: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    walker = Walker(graph, rng)
    seen: set[tuple[int, int]] = set()
    records: list[dict[str, Any]] = []
    t0 = time.time()
    for seed in seeds:
        if len(records) >= n:
            break
        for _ in range(attempts):
            result = walker.walk(seed)
            if result is None:
                continue
            nodes, pids = result
            if walker.over_terminal_share(pids[-1]) and rng.random() > TERMINAL_REJECT_KEEP:
                continue
            key = (int(graph.label_code[nodes[0]]), int(graph.label_code[nodes[-1]]))
            if key in seen:
                continue
            seen.add(key)
            walker.record(nodes, pids)
            records.append(render(graph, nodes, pids))
            if len(records) % 5_000 == 0:
                print(f"  {len(records):,} records ({time.time() - t0:.0f}s)")
            break

    print(f"generated {len(records):,} records from {len(seeds):,} seeds ({time.time() - t0:.0f}s)")
    print("  hops:", dict(sorted(Counter(r["num_hops_in_graph"] for r in records).items())))
    print("  distinct answers:", len({r["ground_truth"] for r in records}))
    top = walker.terminal_hits.most_common(8)
    print("  closing hops:", [(graph.phrase[c], f"{k / max(walker.total_walks, 1):.1%}") for c, k in top])
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE, help="DuckDB projection of the Wikidata dump")
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH, help="cache file for the walkable graph")
    ap.add_argument("--rebuild-graph", action="store_true", help="rebuild the cache even if it exists")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-records", type=int, default=30_000)
    ap.add_argument("--attempts-per-seed", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    if args.rebuild_graph or not args.graph.exists():
        build_graph(args.store, args.graph)

    rng = np.random.default_rng(args.seed)
    graph = Graph(args.graph)
    seeds = sample_seeds(args.graph, graph, args.num_records, rng)
    records = generate(graph, seeds, args.num_records, args.attempts_per_seed, rng)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records):,} records to {args.output}")


if __name__ == "__main__":
    main()
