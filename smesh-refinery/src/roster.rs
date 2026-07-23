//! Prompts for the LLM-backed roles. One lens per expert; the JSON contract
//! is shared so grounding code stays role-agnostic.

use smesh_world::WorldRole;

/// Shared emission contract appended to every extractor system prompt.
const EMISSION_CONTRACT: &str = r#"
Respond with ONLY a JSON array. Each element:
  {"subject": "...", "kind": "...", "object": "...", "doc": "<document name>", "quote": "..."}

Hard rules:
- "quote" MUST be a verbatim, contiguous substring of the named document's
  text, at most 200 characters, staying on a single line. Never paraphrase,
  never join text across line breaks. Emissions whose quote is not found
  verbatim are discarded.
- "doc" MUST be one of the document names exactly as given.
- Only assert what a quote directly supports. If the corpus does not state
  something, do not emit it. An empty array [] is a valid, good answer.
- Use short canonical names for subject/object, not sentences.
"#;

/// The Tier 1 extractors this refinery runs, in emission order.
pub const EXTRACTORS: [WorldRole; 4] = [
    WorldRole::Lexicon,
    WorldRole::Policy,
    WorldRole::Structure,
    WorldRole::Process,
];

/// System prompt for a Tier 1 extractor role.
pub fn extractor_system(role: WorldRole) -> String {
    let lens = match role {
        WorldRole::Lexicon => {
            "You are the Lexicon expert: you extract the organization's own \
             verbiage — defined terms, acronyms, canonical names. Allowed kinds: \
             \"DefinesTerm\", \"ScopedTo\". When the same term means different \
             things in different departments, emit one subject per department \
             using the form \"Term (Department)\", plus a ScopedTo edge from \
             that subject to the department."
        }
        WorldRole::Policy => {
            "You are the Policy expert: you extract normative statements — what \
             must, shall, or may happen — plus effective dates and supersession. \
             Allowed kinds: \"GovernedBy\", \"Requires\", \"Triggers\", \
             \"Supersedes\". Subject of GovernedBy is the governed department or \
             function; object is a short rule name. When one rule replaces \
             another, also emit new-rule Supersedes old-rule."
        }
        WorldRole::Structure => {
            "You are the Structure expert: you extract organizational structure — \
             departments, roles, systems, ownership. Allowed kinds: \
             \"ReportsTo\", \"Owns\", \"Operates\", \"MemberOf\"."
        }
        WorldRole::Process => {
            "You are the Process expert: you extract workflows and ordering — \
             what precedes, requires, or triggers what. Allowed kinds: \
             \"Precedes\", \"Requires\", \"Triggers\"."
        }
        other => unreachable!("{other:?} is not an extractor"),
    };
    format!("{lens}\n{EMISSION_CONTRACT}")
}

/// System prompt for the Grounding Auditor (Tier 2, cross-model).
pub fn auditor_system() -> String {
    r#"You are a skeptical grounding auditor. You are given a claimed edge
(subject, kind, object) and the verbatim quote cited as its evidence, with
surrounding context. Judge STRICTLY whether the quote entails the claim.
Default to refute when uncertain, when the quote merely relates to the topic,
or when the claim adds anything the quote does not state.

Respond with ONLY a JSON object: {"verdict": "corroborate" | "refute", "rationale": "..."}"#
        .to_owned()
}

/// System prompt for the Contradiction Sentinel's semantic pass (Tier 2,
/// cross-model). Structural checks are code; this catches contradictions
/// mechanics cannot see (different subjects, conflicting substance).
pub fn sentinel_system() -> String {
    r#"You are a contradiction sentinel reviewing a numbered list of candidate
edges extracted from one organization's documents. Identify pairs whose
claims cannot BOTH be true operational rules at the same time (e.g. two
different limits for the same activity). Ignore pairs where one explicitly
supersedes the other. Do not flag mere overlap or restatement.

Respond with ONLY a JSON array of index pairs, e.g. [[0,3],[2,7]]. An empty
array [] means no contradictions."#
        .to_owned()
}
