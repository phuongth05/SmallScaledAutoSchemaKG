"""Deterministic research interventions. No model calls and no gold-aware retrieval.

Only explicitly named oracle/scoring/split helpers may inspect support labels.
Post-hoc pruning reduces retrieval/index size, not sunk graph construction cost.
"""
from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from hotpotqa_benchmark import digest, graph_variant


def validate_arms(arms):
    names = set()
    allowed = {"name", "method", "variant", "filter", "stages", "candidate_policy",
               "factual_quota", "concept_weight", "concept_cap", "pruning"}
    for arm in arms:
        name = arm.get("name", "")
        if not re.fullmatch(r"[a-z0-9_]+", name) or name in names:
            raise ValueError("Arm names must be unique safe lowercase identifiers")
        names.add(name)
        if set(arm) - allowed:
            raise ValueError(f"Unknown arm fields: {set(arm) - allowed}")
        if arm.get("method") not in {"bm25", "dense", "graph", "oracle", "no_context"}:
            raise ValueError("Unknown method")
        if arm.get("variant", "full") not in {"entity", "entity_event", "full"}:
            raise ValueError("Unknown graph variant")
        if arm.get("candidate_policy", "all") not in {"all", "quota", "factual"}:
            raise ValueError("Unknown candidate policy")
        if arm.get("pruning", "specificity") not in {"specificity", "random_matched"}:
            raise ValueError("Unknown pruning policy")
        if not isinstance(arm.get("filter", True), bool):
            raise ValueError("filter must be boolean")
        if not set(arm.get("stages", ["diagnostic", "qa"])) <= {"diagnostic", "qa"}:
            raise ValueError("Unknown stage")
        for key in ("concept_cap", "factual_quota"):
            if key in arm and (type(arm[key]) is not int or arm[key] < 0):
                raise ValueError(f"{key} must be a nonnegative integer")
        weight = arm.get("concept_weight", 1.0)
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("concept_weight must be finite and positive")
    if not arms:
        raise ValueError("Empty experiment suite")
    return arms


def split_questions(manifest, dev_size=100, seed=42):
    """Proportional type stratification; disjoint QIDs, NOT support-document groups."""
    if dev_size <= 0:
        raise ValueError("dev_size must be positive")
    groups = defaultdict(list)
    for q in manifest:
        groups[q.get("type", "unknown")].append(str(q["id"]))
    total = len(manifest)
    target = min(dev_size, total)
    counts = {k: len(v) * target // total for k, v in groups.items()}
    remainder_order = sorted(groups, key=lambda k: (-(len(groups[k]) * target % total), k))
    for k in remainder_order[:target - sum(counts.values())]:
        counts[k] += 1
    dev, test = [], []
    for kind in sorted(groups):
        values = sorted(groups[kind])
        random.Random(digest([seed, kind])).shuffle(values)
        dev.extend(values[:counts[kind]])
        test.extend(values[counts[kind]:])
    # Mix types before selecting a smoke subset; independent of manifest ordering.
    random.Random(seed).shuffle(dev)
    random.Random(seed + 1).shuffle(test)
    return {"seed": seed, "dev_size_requested": dev_size, "dev": dev, "test": test,
            "smoke": dev[:3], "unit": "question_id", "document_disjoint": False}


def is_concept_edge(graph, edge):
    return any(graph.nodes[n].get("type") == "concept" for n in edge[:2])


def prepare_graph(graph, arm, seed=42):
    g = graph_variant(graph, arm.get("variant", "full"))
    edges = list(g.edges)
    concept_edges = [e for e in edges if is_concept_edge(g, e)]
    groups = defaultdict(list)
    neighbors = defaultdict(set)
    for e in concept_edges:
        for node in e[:2]:
            if g.nodes[node]["type"] == "concept":
                neighbors[node].update(n for n in e[:2] if g.nodes[n]["type"] != "concept")
    for e in concept_edges:
        nonconcept = [n for n in e[:2] if g.nodes[n]["type"] != "concept"]
        concepts = [n for n in e[:2] if g.nodes[n]["type"] == "concept"]
        if len(nonconcept) == len(concepts) == 1:
            groups[nonconcept[0]].append((concepts[0], e))
    remove = set()
    cap = arm.get("concept_cap")
    if cap is not None:
        for node, entries in groups.items():
            # Rare concepts first; deterministic ID tie-break. A hypothesis, not a quality label.
            concepts = sorted({c for c, _ in entries}, key=lambda c: (len(neighbors[c]), str(c)))
            retained = set(concepts[:cap])
            remove.update(e for c, e in entries if c not in retained)
        if arm.get("pruning") == "random_matched":
            pool = sorted(concept_edges, key=repr)
            remove = set(random.Random(seed).sample(pool, len(remove)))
    g.remove_edges_from(remove)
    # Keep every entity/event/passage, including isolated passages. Remove only unused concepts.
    g.remove_nodes_from([n for n, d in list(g.nodes(data=True))
                         if d["type"] == "concept" and g.degree(n) == 0])
    for e in g.edges:
        if is_concept_edge(g, e):
            g.edges[e]["weight"] = float(g.edges[e].get("weight", 1)) * arm.get("concept_weight", 1.0)
    return g, {"nodes": len(g), "edges": g.number_of_edges(),
               "concept_edges_before": len(concept_edges), "concept_edges_removed": len(remove),
               "concept_nodes": sum(d["type"] == "concept" for _, d in g.nodes(data=True)),
               "passages": sum(d["type"] == "passage" for _, d in g.nodes(data=True))}


def candidate_selector(data, policy="all", factual_quota=20):
    if policy == "all":
        return None  # Keep the baseline's exact FAISS behavior, including ties.
    concept = np.array([is_concept_edge(data["KG"], e) for e in data["edge_list"]])

    def select(scores, top_n):
        ranked = np.argsort(-np.asarray(scores), kind="stable").tolist()
        factual = [i for i in ranked if not concept[i]]
        if policy == "factual":
            return factual[:top_n]  # No concept backfill: clean factual-seed ablation.
        if policy != "quota" or not 0 <= factual_quota <= top_n:
            raise ValueError("Require quota in [0, top_edges]")
        concepts = [i for i in ranked if concept[i]]
        selected = factual[:factual_quota] + concepts[:top_n - factual_quota]
        # If one group is undersupplied, fill from remaining globally ranked edges.
        used = set(selected)
        for i in ranked:
            if len(selected) >= top_n:
                break
            if i not in used:
                selected.append(i)
        return sorted(selected, key=lambda i: (-float(scores[i]), i))
    return select


def tokens(text):
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold(), flags=re.UNICODE)


class BM25:
    """Okapi BM25 with positive log IDF, Unicode word/syllable tokens, no stopwords."""
    def __init__(self, texts, k1=1.5, b=0.75):
        self.ids = list(texts)
        self.k1, self.b = k1, b
        self.lengths = np.zeros(len(texts))
        self.postings = defaultdict(list)
        for i, text in enumerate(texts.values()):
            counts = Counter(tokens(text))
            self.lengths[i] = sum(counts.values())
            for term, count in counts.items():
                self.postings[term].append((i, count))
        self.average = max(float(self.lengths.mean()), 1)

    def retrieve(self, question, k):
        scores = np.zeros(len(self.ids))
        for term in set(tokens(question)):
            posting = self.postings.get(term, [])
            idf = math.log(1 + (len(self.ids) - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, tf in posting:
                scores[i] += idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.average))
        return [self.ids[i] for i in np.argsort(-scores, kind="stable")[:k]]


def support_documents(sample, docs):
    if sample.get("gold_document_ids"):
        return set(map(str, sample["gold_document_ids"]))
    titles = {str(t) for t, _ in sample["supporting_facts"]}
    return {d for d, row in docs.items() if str(row["metadata"]["title"]) in titles}


def oracle_passages(sample, docs, passage_docs, k):
    gold = support_documents(sample, docs)
    # One chunk per supporting document first, then remaining supporting chunks.
    remaining, chosen = set(gold), []
    for p, ids in passage_docs.items():
        if remaining & set(ids):
            chosen.append(p)
            remaining -= set(ids)
    if remaining:
        raise ValueError("Oracle cannot find all supporting documents")
    if len(chosen) > k:
        raise ValueError("Oracle evidence exceeds passage budget")
    chosen.extend(p for p, ids in passage_docs.items() if set(ids) & gold and p not in chosen)
    return chosen[:k]


def metrics_at_k(ids, sample, docs, passage_docs, ks=(2, 5, 10)):
    # Reuse official title/ID mapping, with rank cutoff supplied explicitly.
    result = {}
    for k in ks:
        gold = support_documents(sample, docs)
        if not gold:
            raise ValueError("Supporting document labels cannot be resolved")
        # English baseline scores by titles (including duplicate titles), VN by exact IDs.
        if sample.get("gold_document_ids"):
            got = {d for p in ids[:k] for d in passage_docs[p]}
            recall, complete = len(got & gold) / len(gold), int(gold <= got)
        else:
            # Compare title sets, not document IDs, for English HotpotQA.
            target = {str(t) for t, _ in sample["supporting_facts"]}
            got = {str(docs[d]["metadata"]["title"]) for p in ids[:k] for d in passage_docs[p]}
            recall, complete = len(got & target) / len(target), int(target <= got)
        result[f"support_recall@{k}"] = recall
        result[f"all_support@{k}"] = complete
    return result


def edge_passages(graph, edges, passage_docs):
    found = set()
    for edge in edges:
        for n in edge[:2]:
            # Direct provenance only, not recursive concept reachability or gold-guided paths.
            if graph.nodes[n].get("type") == "passage":
                found.add(n)
            else:
                for p in set(graph.predecessors(n)) | set(graph.successors(n)):
                    if p in passage_docs:
                        found.add(p)
    return sorted(found)


def audit_template(graph, passage_docs, per_type=25, seed=42):
    groups = defaultdict(list)
    for e in graph.edges:
        types = [graph.nodes[n]["type"] for n in e[:2]]
        if "passage" in types:
            continue
        kind = "concept" if "concept" in types else "-".join(sorted(types))
        groups[kind].append(e)
    rows = []
    for kind, edges in sorted(groups.items()):
        chosen = random.Random(digest([seed, kind])).sample(sorted(edges, key=repr), min(per_type, len(edges)))
        for edge in chosen:
            passages = edge_passages(graph, [edge], passage_docs)
            rows.append({"audit_id": digest(list(edge)), "stratum": kind, "edge": list(edge),
                         "triple": [graph.nodes[edge[0]]["id"], graph.edges[edge]["relation"], graph.nodes[edge[1]]["id"]],
                         "candidate_evidence": [{"passage_id": p, "text": graph.nodes[p]["id"]} for p in passages],
                         "reviewer_1": {"supported": None, "error_type": None, "notes": ""},
                         "reviewer_2": {"supported": None, "error_type": None, "notes": ""}})
    return {"seed": seed, "stratum_population": {k: len(v) for k, v in groups.items()},
            "note": "Evidence candidates are NOT proof. Judge against source; do not use QA test answers. Stratified sample is not population-weighted.",
            "rows": rows}


def paired_bootstrap(baseline, treatment, seed=42, repetitions=2000):
    if set(baseline) != set(treatment) or not baseline:
        raise ValueError("Paired comparison requires identical, nonempty question IDs")
    delta = np.array([treatment[k] - baseline[k] for k in sorted(baseline)])
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(delta, size=len(delta), replace=True).mean()) for _ in range(repetitions)]
    return {"n": len(delta), "delta": float(delta.mean()),
            "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
            "note": "Paired question bootstrap; exploratory, no multiple-comparison or shared-document correction."}
