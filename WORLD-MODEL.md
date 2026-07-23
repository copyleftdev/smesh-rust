# WORLD-MODEL — Signed Organizational World Models via SMESH × AION

**Status:** Design accepted, scaffold in `smesh-world/`
**Depends on:** `smesh-core` (signal field, reputation, trust), `smesh-agent` (LLM backends, tool contracts), [aion-context](https://github.com/aion-context/aion-context) (signed graph substrate)

---

## 1. Problem

An agent without a structured model of the organization it serves does not say
"I don't know your PTO policy." It **confabulates context**: it imports the
nearest-neighbor organization from pretraining and speaks fluent Generic Corp —
wrong verbiage, wrong policy, wrong org chart, delivered with full confidence.

The fix is not more prompt. It is converting "fill in the blank" into
"look up the node" — and making the *absence* of a node itself signed and
queryable (**verifiable negative space**). An agent grounded in a signed world
model can refuse to invent what the world does not contain.

## 2. Thesis

> **SMESH is how the world model earns its edges. AION is how the world model
> proves its identity. The revision loop is how it stays alive. The human is
> the merge authority.**

- **AION = the truth plane.** Content-addressed nodes (idempotent ingestion),
  Ed25519-signed revision chain (an agent verifies it loaded *this org's*
  world, untampered), append-only history (replay any decision against the
  world *as it was*), deny/allow (retraction without deletion).
- **SMESH = the refinery.** Expert agents emit candidate nodes/edges as
  signals. Reinforcement = independent corroboration. Decay = unsupported
  claims fade. Contested edges escalate to a human instead of being resolved
  by the mesh.
- **Governing principle — separation of powers:** extractors propose,
  verifiers judge, one non-LLM curator signs. No expert holds two of those
  powers.

## 3. Revision loop

```
raw corpus ──► Tier 0 intake ──► CDM docs ──► Tier 1 experts ──► candidate signals
                                                                       │ reinforce / decay
                                                                       ▼
                                                    Tier 2 verification (adversarial)
                                                                       │
                                                                       ▼
                                       staged changeset  (rev N → proposed rev N+1)
                                                                       │
                                                                       ▼
                                    HUMAN RATIFICATION (dashboard, tiered lanes)
                                                                       │
                                                                       ▼
                              Curator signs ratified changeset ──► world rev N+1
                                                                       ▲
                          new docs / drift ────────────────────────────┘
```

Nothing reaches the signed graph without passing `Ratified`. The Curator's
signing key is only ever applied to a changeset carrying a human ratification
record.

## 4. Expert roster

### Tier 0 — Intake ("Registrar of Record")

Any artifact in → **Canonical Document Model (CDM)** out. Tier 1 never sees a
raw format; adding a format touches Tier 0 only.

| Concern | Design |
|---|---|
| Identity | `doc_id = BLAKE3(bytes)` — content-addressed, idempotent re-ingestion |
| Detection | Deterministic sniffing first; LLM triage only for ambiguous blobs |
| Provenance | **Dual anchoring**: extractor citations anchor to CDM spans; every CDM span maps back to native coordinates (PDF page/rect, email message-id + line, markdown heading path). The dashboard highlights the *original* artifact |
| Metadata | Email headers are native provenance — `From:` + `Date:` yield attribution and temporal edges for free |

Tooling (mostly assembled from the existing portfolio): `scry` for format
triage, AkamaiForms PDF kernel as the PDF adapter, `whatthediff` for
cross-format structural unification and near-duplicate detection, Prévue's
pdf-extract path as text fallback. New work: `.eml`/mbox adapter.

### Tier 1 — Extraction (read CDM → emit candidate signals)

| Expert | Lens | Tools |
|---|---|---|
| **Lexicon** | Org verbiage: defined terms, acronyms, canonical names — department-scoped | `corpus_read`, `concordance`, `emit_candidate(Term)` |
| **Policy** | Normative statements (must/shall/may), effective dates, supersession | `corpus_read`, `emit_candidate(PolicyRule)` |
| **Structure** | Org chart, roles, systems, ownership | `corpus_read`, `emit_candidate(Entity\|Relation)` |
| **Process** | Workflows, lifecycles, temporal ordering | `corpus_read`, `emit_candidate(ProcessStep)` |

### Tier 2 — Verification (adversarial; may corroborate/refute, **cannot emit facts**)

| Expert | Lens | Tools |
|---|---|---|
| **Grounding Auditor** | Does the cited span *entail* the claim? Runs a **different model family** than the extractor — always | `span_fetch`, `verdict(corroborate\|refute)` |
| **Contradiction Sentinel** | Candidate deltas vs current rev: `aion_contradictions` + typed-edge constraints (one `reports_to` per person per timeslice; `supersedes` acyclic) | `graph_read`, `aion_contradictions`, `escalate` |

### Tier 3 — Stewardship

| Expert | Lens | Tools |
|---|---|---|
| **Ontologist** | Owns the edge vocabulary (rev-0 ontology node). Only role that may propose new edge kinds — via human-approved escalation only | `graph_read`, `ontology_map`, `propose_edge_kind` |
| **Curator** | Sole AION writer. Batches consensus survivors, runs final gates, signs. **Deliberately not an LLM** — signing authority is code | `aion_annotate`, `aion_link`, `aion_snapshot` |

## 5. Hard rules

1. **Provenance is enforced at the tool layer.** `emit_candidate`
   schema-requires `{doc_id, span, quote}`. A citation-free candidate is
   rejected by the tool contract before it becomes a signal.
2. **Capability-scoped tools per role.** Extractors: corpus read only.
   Verifiers: graph read, no corpus write. Curator alone writes AION.
3. **Reputation-weighted amplitude.** Auditor verdicts feed
   `smesh-core::reputation`; a repeatedly-refuted extractor emits weaker
   signals.
4. **Model diversity is a requirement.** Extraction: `moonshotai/kimi-k3`
   (1M ctx — whole-corpus passes, no chunking). Verification: a different
   family (Claude, or minimally `kimi-k2-thinking`). Same-model verification
   lets correlated hallucinations survive.
5. **Consensus thresholds are per-edge-kind.** `defines_term` with one
   corroboration may pass; `governed_by` needs ≥2 independent corroborations
   and zero contradictions.

## 6. Human ratification protocol

The mesh never publishes — it stages. A staged changeset is a PR against the
world; the dashboard is the review screen.

### Delta lifecycle

```
Draft ──► Staged ──┬── approve ──► Ratified ──► Signed (rev N+1)
                   ├── edit ─────► HumanAmended ──► Ratified (human version wins)
                   ├── reject ───► Killed (+reason → expert reputation hit)
                   └── defer ────► Parked (decays back into the field)
```

### Reviewer identity

Each reviewer holds an Ed25519 identity; ratification is a signed act recorded
in the graph. Every edge answers four questions forever: what it says, where
it came from (span), which experts corroborated it, **which human approved it
and when**.

### Provenance classes

- `corpus_derived` — extracted, span-cited, mesh-corroborated.
- `human_attested` — no document exists; the provenance *is* the human's
  signature. Highest-trust class. The dashboard is the tribal-knowledge
  capture instrument: reviewers author facts the corpus never contained.

Human edits to staged candidates are labeled error signals — fed back into
expert reputation, retained as future tuning data.

### Attention lanes (ratification is total; attention is tiered)

| Lane | Contents | Interaction |
|---|---|---|
| **Green** | High-consensus, exact-quote, low-risk (terms, org structure) | Summarized batch, spot-checkable |
| **Amber** | Normative/policy edges; single corroborator | Individually surfaced with evidence |
| **Red** | Contested edges, contradictions, ontology changes | Mandatory individual decision |

Dashboard write path is a thin Axum layer over the prevue-api registry
pattern (browse, drill into cited evidence) plus Solv's cited-cards queue UX.

## 7. Showcase corpus: Meridian Mutual

**Author the gold graph first; render documents from it.** The answer key is
exact by construction; validation is scored, not vibes.

Meridian Mutual: fictional regional insurance company. Departments: Claims,
Clinical Policy, Finance, HR, IT/Security, Legal/Compliance. Formats:
markdown handbooks, PDF policy manuals, `.eml` threads, YAML
config-as-policy, CSV org roster, wiki export, scanned-memo PDFs.

### Planted-defect manifest

| Planted challenge | Exercises |
|---|---|
| "Claim" means different things in Legal / Claims / Finance | Lexicon + department-scoped terms |
| Two departments with contradictory rules | Contradiction Sentinel → red lane → human |
| Policy PDF superseded by a later email memo | Cross-format supersession chains |
| Same policy as PDF and markdown, slightly divergent | Consensus + divergence detection |
| Rule existing only in an email thread | Tribal-knowledge analog; email provenance |
| Question the corpus deliberately never answers | **Confabulation trap** — must yield negative space |
| Attached vendor policy document | Attribution scoping — external ≠ Meridian |
| Generic-corp boilerplate contradicting Meridian's quirky rule | The founding failure mode |

### Scorecard (also the permanent regression gate)

- **Edge precision / recall** vs gold graph
- **Contradiction detection rate** (caught / planted)
- **Confabulation rate** — edges in no document with no human signature. Target: **zero**. Headline number for the thesis
- **Provenance integrity** — cross-model audit that every `corpus_derived` span entails its claim
- **Escalation quality** — red-lane items reached the human; green-lane items deserved green

## 8. Scaffold map (`smesh-world/`)

| Module | Contents |
|---|---|
| `role.rs` | `WorldRole` taxonomy (Tiers 0–3), capability sets, per-role tool inventories, model assignment |
| `cdm.rs` | Canonical Document Model: content-addressed docs, blocks, dual-anchor spans |
| `ontology.rs` | Typed edge vocabulary + per-edge-kind consensus thresholds + structural constraints |
| `candidate.rs` | Candidate node/edge emissions with schema-mandatory citations |
| `delta.rs` | Changeset lifecycle state machine; type-state ratification (unsigned changeset cannot reach `Signed`) |
| `corpus.rs` | Gold graph, planted-defect manifest, scorecard types for Meridian |

## 9. Phasing

- **Phase 0 (this scaffold):** types + state machine + gates green.
- **Phase 1:** Tier 0 intake adapters (markdown, PDF via AkamaiForms path, `.eml`) → CDM fixtures.
- **Phase 2:** Meridian gold graph + deterministic renderer + planted defects.
- **Phase 3:** Wire Tier 1/2 experts over `smesh-agent` (K3 extract, cross-model verify), run the field, emit first staged changeset.
- **Phase 4:** Ratification dashboard (Axum; prevue-api pattern) + Curator signing into AION.
- **Phase 5:** Scorecard harness; Meridian becomes the regression gate.
