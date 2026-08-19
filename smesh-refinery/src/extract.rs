//! Tier 1 extraction and the confabulation firewall.
//!
//! An emission becomes a `CandidateEdge` only if its quote is found verbatim
//! in the cited document — `Citation::grounded` does the rest. Everything
//! else lands in `rejected`, counted and attributable, never silently gone.

use crate::packet::NamedDoc;
use crate::roster;
use crate::{extract_json, Oracle, RefineryError};
use serde::Deserialize;
use smesh_world::corpus::canon;
use smesh_world::role::ModelPolicy;
use smesh_world::{CandidateEdge, CdmSpan, Citation, EdgeKind, WorldRole};

/// Shortest quote that can count as grounding a claim.
///
/// Not a magic number so much as a floor: below this, a match against the
/// document says more about the document than about the claim.
const MIN_QUOTE_CHARS: usize = 8;

#[derive(Debug, Clone, Deserialize)]
pub struct Emission {
    pub subject: String,
    pub kind: String,
    pub object: String,
    pub doc: String,
    pub quote: String,
}

#[derive(Debug)]
pub struct RejectedEmission {
    pub role: WorldRole,
    pub emission: Emission,
    pub reason: String,
}

/// An expert that produced nothing usable at all.
///
/// Distinct from a rejected emission: there is no emission to attribute, so
/// recording it as one would put words in the model's mouth. Kept because this
/// module's contract is that nothing is silently gone.
#[derive(Debug, Clone)]
pub struct ExpertFailure {
    pub role: WorldRole,
    pub reason: String,
}

#[derive(Debug, Default)]
pub struct ExtractionOutcome {
    pub candidates: Vec<CandidateEdge>,
    pub rejected: Vec<RejectedEmission>,
    /// Experts whose response could not be parsed at all.
    pub failures: Vec<ExpertFailure>,
}

pub(crate) fn kind_from_str(s: &str) -> Option<EdgeKind> {
    let normalized: String = s.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    EdgeKind::ALL
        .into_iter()
        .find(|k| format!("{k:?}").eq_ignore_ascii_case(&normalized))
}

fn extractor_model(role: WorldRole) -> &'static str {
    match role.model_policy() {
        ModelPolicy::Fixed(m) => m,
        other => unreachable!("extractor {role:?} has policy {other:?}"),
    }
}

/// Run every Tier 1 extractor over the packet and ground each emission.
pub async fn run_extractors(
    oracle: &dyn Oracle,
    docs: &[NamedDoc],
    doc_block: &str,
) -> Result<ExtractionOutcome, RefineryError> {
    let mut outcome = ExtractionOutcome::default();
    for role in roster::EXTRACTORS {
        let response = oracle
            .complete(
                extractor_model(role),
                &roster::extractor_system(role),
                doc_block,
            )
            .await?;
        // One expert returning malformed JSON used to abort the entire run and
        // discard every other expert's work. A model producing something
        // unparseable is ordinary, not exceptional: record it as that expert
        // failing and carry on with the rest.
        let emissions: Result<Vec<Emission>, _> = extract_json(&response)
            .map_err(|e| e.to_string())
            .and_then(|v| serde_json::from_value::<Vec<Emission>>(v).map_err(|e| e.to_string()));

        match emissions {
            Ok(emissions) => {
                for emission in emissions {
                    ground(role, emission, docs, &mut outcome);
                }
            }
            Err(e) => outcome.failures.push(ExpertFailure {
                role,
                reason: format!("unusable output: {e}"),
            }),
        }
    }
    Ok(outcome)
}

/// The firewall. Grounding failures are recorded, not raised: one bad
/// emission must never abort a run.
pub fn ground(
    role: WorldRole,
    emission: Emission,
    docs: &[NamedDoc],
    outcome: &mut ExtractionOutcome,
) {
    let reject = |reason: String, outcome: &mut ExtractionOutcome, emission: Emission| {
        outcome.rejected.push(RejectedEmission {
            role,
            emission,
            reason,
        });
    };

    let Some(kind) = kind_from_str(&emission.kind) else {
        return reject(
            format!("unknown kind {:?}", emission.kind),
            outcome,
            emission,
        );
    };
    let Some(named) = docs.iter().find(|d| d.name == emission.doc) else {
        return reject(
            format!("unknown document {:?}", emission.doc),
            outcome,
            emission,
        );
    };
    // An empty quote is found at offset zero in every document, so it passes
    // "appears verbatim" trivially and yields a citation that points at nothing.
    // The firewall exists to stop exactly that: a claim carrying evidence that
    // is not evidence. A floor on length is applied for the same reason — a
    // one-character quote matches by accident, not by grounding.
    let quote = emission.quote.trim();
    if quote.chars().count() < MIN_QUOTE_CHARS {
        return reject(
            format!(
                "quote is {} characters; needs at least {MIN_QUOTE_CHARS} to ground anything",
                quote.chars().count()
            ),
            outcome,
            emission,
        );
    }

    let Some(start) = named.doc.canonical_text.find(&emission.quote) else {
        return reject(
            "quote not found verbatim in document".into(),
            outcome,
            emission,
        );
    };
    let span = CdmSpan::new(start, start + emission.quote.len());
    let citation = match Citation::grounded(&named.doc, span) {
        Ok(c) => c,
        Err(e) => return reject(format!("citation rejected: {e}"), outcome, emission),
    };

    let (subj, obj) = (canon(&emission.subject), canon(&emission.object));
    if let Some(existing) = outcome
        .candidates
        .iter_mut()
        .find(|c| c.kind == kind && canon(&c.subject) == subj && canon(&c.object) == obj)
    {
        if let smesh_world::ProvenanceClass::CorpusDerived { citations } = &mut existing.provenance
        {
            if !citations.contains(&citation) {
                citations.push(citation);
            }
        }
        return;
    }

    match CandidateEdge::emit(
        role,
        emission.subject.clone(),
        kind,
        emission.object.clone(),
        vec![citation],
    ) {
        Ok(candidate) => outcome.candidates.push(candidate),
        Err(e) => reject(format!("emit rejected: {e}"), outcome, emission),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packet::ingest_all;

    fn docs() -> Vec<NamedDoc> {
        ingest_all(&[(
            "handbook.md".to_owned(),
            b"# Handbook\n\nClaims must be filed within 30 days.\n".to_vec(),
        )])
        .unwrap()
    }

    fn emission(quote: &str) -> Emission {
        Emission {
            subject: "Claims".into(),
            kind: "GovernedBy".into(),
            object: "30-day filing window".into(),
            doc: "handbook.md".into(),
            quote: quote.into(),
        }
    }

    #[test]
    fn verbatim_quotes_ground_into_candidates() {
        let docs = docs();
        let mut outcome = ExtractionOutcome::default();
        ground(
            WorldRole::Policy,
            emission("Claims must be filed within 30 days."),
            &docs,
            &mut outcome,
        );
        assert_eq!(outcome.candidates.len(), 1);
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn hallucinated_quotes_are_rejected_and_counted() {
        let docs = docs();
        let mut outcome = ExtractionOutcome::default();
        ground(
            WorldRole::Policy,
            emission("Claims must be filed within 45 days."),
            &docs,
            &mut outcome,
        );
        assert!(outcome.candidates.is_empty());
        assert_eq!(outcome.rejected.len(), 1);
        assert!(outcome.rejected[0].reason.contains("not found verbatim"));
    }

    #[test]
    fn duplicate_emissions_merge_citations_instead_of_duplicating() {
        let docs = docs();
        let mut outcome = ExtractionOutcome::default();
        let quote = "Claims must be filed within 30 days.";
        ground(WorldRole::Policy, emission(quote), &docs, &mut outcome);
        ground(WorldRole::Lexicon, emission(quote), &docs, &mut outcome);
        assert_eq!(outcome.candidates.len(), 1);
        match &outcome.candidates[0].provenance {
            smesh_world::ProvenanceClass::CorpusDerived { citations } => {
                assert_eq!(citations.len(), 1);
            }
            other => panic!("unexpected provenance {other:?}"),
        }
    }

    #[test]
    fn unknown_kind_and_unknown_doc_are_rejected() {
        let docs = docs();
        let mut outcome = ExtractionOutcome::default();
        let mut bad_kind = emission("Claims must be filed within 30 days.");
        bad_kind.kind = "Blesses".into();
        ground(WorldRole::Policy, bad_kind, &docs, &mut outcome);
        let mut bad_doc = emission("Claims must be filed within 30 days.");
        bad_doc.doc = "ghost.md".into();
        ground(WorldRole::Policy, bad_doc, &docs, &mut outcome);
        assert!(outcome.candidates.is_empty());
        assert_eq!(outcome.rejected.len(), 2);
    }

    #[test]
    fn case_variant_emissions_merge_into_one_candidate() {
        let docs = docs();
        let mut outcome = ExtractionOutcome::default();
        let quote = "Claims must be filed within 30 days.";
        ground(WorldRole::Policy, emission(quote), &docs, &mut outcome);
        let mut variant = emission(quote);
        variant.subject = "CLAIMS".into();
        variant.object = "30-Day Filing Window".into();
        ground(WorldRole::Policy, variant, &docs, &mut outcome);
        assert_eq!(outcome.candidates.len(), 1);
    }

    #[test]
    fn kind_parsing_accepts_case_and_snake_variants() {
        assert_eq!(kind_from_str("GovernedBy"), Some(EdgeKind::GovernedBy));
        assert_eq!(kind_from_str("governed_by"), Some(EdgeKind::GovernedBy));
        assert_eq!(kind_from_str("DEFINES_TERM"), Some(EdgeKind::DefinesTerm));
        assert_eq!(kind_from_str("Blesses"), None);
    }
}
