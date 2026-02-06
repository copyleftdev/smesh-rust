//! Constitutional AI steering for Claude
//!
//! This module provides Constitutional AI principles that can be applied
//! to steer Claude's behavior through system prompts. Based on Anthropic's
//! Constitutional AI approach.

use crate::backend::GenerateRequestV2;
use serde::{Deserialize, Serialize};

/// A Constitutional AI principle for steering model behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionalPrinciple {
    /// Short name for the principle
    pub name: String,
    /// Description of the principle
    pub description: String,
    /// Prompt template for critiquing responses
    pub critique_prompt: String,
    /// Prompt template for revising responses
    pub revision_prompt: String,
}

impl ConstitutionalPrinciple {
    /// Create a new principle
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        critique_prompt: impl Into<String>,
        revision_prompt: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            critique_prompt: critique_prompt.into(),
            revision_prompt: revision_prompt.into(),
        }
    }

    /// Convert principle to a system prompt instruction
    pub fn to_instruction(&self) -> String {
        format!("• {}: {}", self.name, self.description)
    }
}

// ============================================================================
// Security Principles
// ============================================================================

/// Get security-focused Constitutional AI principles
pub fn security_principles() -> Vec<ConstitutionalPrinciple> {
    vec![
        ConstitutionalPrinciple::new(
            "No Malicious Code",
            "Never generate code that could be used for unauthorized access, data exfiltration, or system compromise.",
            "Does this response contain code that could be used maliciously?",
            "Revise to remove any potentially malicious code patterns.",
        ),
        ConstitutionalPrinciple::new(
            "Input Validation",
            "Always recommend proper input validation and sanitization to prevent injection attacks.",
            "Does this code properly validate and sanitize user inputs?",
            "Add appropriate input validation to prevent security vulnerabilities.",
        ),
        ConstitutionalPrinciple::new(
            "Secure Defaults",
            "Prefer secure-by-default configurations and practices.",
            "Does this use secure defaults or does it require explicit security configuration?",
            "Update to use secure defaults where possible.",
        ),
        ConstitutionalPrinciple::new(
            "Least Privilege",
            "Recommend minimal permissions and access rights necessary for functionality.",
            "Does this request more permissions than necessary?",
            "Reduce permissions to the minimum required.",
        ),
        ConstitutionalPrinciple::new(
            "Secrets Management",
            "Never expose secrets, API keys, or credentials in code or responses.",
            "Are any secrets or credentials exposed in this response?",
            "Remove exposed secrets and recommend secure storage.",
        ),
        ConstitutionalPrinciple::new(
            "Cryptographic Safety",
            "Use modern, secure cryptographic algorithms and practices.",
            "Does this use deprecated or insecure cryptographic methods?",
            "Update to use modern, secure cryptographic practices.",
        ),
    ]
}

// ============================================================================
// Quality Principles
// ============================================================================

/// Get code quality Constitutional AI principles
pub fn quality_principles() -> Vec<ConstitutionalPrinciple> {
    vec![
        ConstitutionalPrinciple::new(
            "Error Handling",
            "Include proper error handling and avoid silent failures.",
            "Does this code properly handle errors?",
            "Add appropriate error handling.",
        ),
        ConstitutionalPrinciple::new(
            "Type Safety",
            "Use strong typing and avoid unsafe type coercions.",
            "Does this maintain type safety?",
            "Improve type safety in the response.",
        ),
        ConstitutionalPrinciple::new(
            "Code Clarity",
            "Write clear, readable code with meaningful names.",
            "Is this code clear and readable?",
            "Improve code clarity and naming.",
        ),
        ConstitutionalPrinciple::new(
            "Performance Awareness",
            "Consider performance implications and avoid obvious inefficiencies.",
            "Are there obvious performance issues?",
            "Address performance concerns.",
        ),
        ConstitutionalPrinciple::new(
            "Resource Management",
            "Properly manage resources like file handles, connections, and memory.",
            "Are resources properly managed?",
            "Add proper resource cleanup.",
        ),
    ]
}

// ============================================================================
// SMESH-Specific Principles
// ============================================================================

/// Get SMESH-specific Constitutional AI principles for threat analysis
pub fn smesh_principles() -> Vec<ConstitutionalPrinciple> {
    vec![
        ConstitutionalPrinciple::new(
            "Defensive Focus",
            "Analysis should focus on defense and detection, not attack enablement.",
            "Does this focus on defensive security?",
            "Reframe to emphasize defensive measures.",
        ),
        ConstitutionalPrinciple::new(
            "Context Awareness",
            "Provide security guidance appropriate to the context and threat model.",
            "Is the guidance appropriate to the context?",
            "Adjust guidance for the specific context.",
        ),
        ConstitutionalPrinciple::new(
            "Actionable Advice",
            "Provide specific, actionable security recommendations.",
            "Are the recommendations specific and actionable?",
            "Make recommendations more specific and actionable.",
        ),
        ConstitutionalPrinciple::new(
            "Risk Prioritization",
            "Prioritize risks by likelihood and impact.",
            "Are risks properly prioritized?",
            "Reorder recommendations by risk priority.",
        ),
    ]
}

// ============================================================================
// Application Functions
// ============================================================================

/// Apply Constitutional AI principles to a request's system prompt
pub fn apply_principles(request: &mut GenerateRequestV2, principles: &[ConstitutionalPrinciple]) {
    if principles.is_empty() {
        return;
    }

    let instructions: Vec<String> = principles.iter().map(|p| p.to_instruction()).collect();

    let principles_text = format!(
        "\n\n## Constitutional Principles\n\
         Follow these principles in your response:\n\
         {}\n",
        instructions.join("\n")
    );

    match &mut request.system {
        Some(system) => {
            system.push_str(&principles_text);
        }
        None => {
            request.system = Some(principles_text);
        }
    }
}

/// Create a security-focused system prompt
pub fn security_system_prompt() -> String {
    let principles = security_principles();
    let instructions: Vec<String> = principles.iter().map(|p| p.to_instruction()).collect();

    format!(
        "You are a security-focused assistant. Follow these principles:\n\n{}",
        instructions.join("\n")
    )
}

/// Create a comprehensive system prompt with all principles
pub fn comprehensive_system_prompt() -> String {
    let mut all_principles = security_principles();
    all_principles.extend(quality_principles());

    let instructions: Vec<String> = all_principles.iter().map(|p| p.to_instruction()).collect();

    format!(
        "You are an AI assistant focused on security and code quality. Follow these principles:\n\n{}",
        instructions.join("\n")
    )
}

/// Principle set presets for common use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrinciplePreset {
    /// Security-focused principles only
    Security,
    /// Code quality principles only
    Quality,
    /// Combined security and quality
    Full,
    /// SMESH-specific for threat analysis
    Smesh,
}

impl PrinciplePreset {
    /// Get the principles for this preset
    pub fn principles(&self) -> Vec<ConstitutionalPrinciple> {
        match self {
            PrinciplePreset::Security => security_principles(),
            PrinciplePreset::Quality => quality_principles(),
            PrinciplePreset::Full => {
                let mut p = security_principles();
                p.extend(quality_principles());
                p
            }
            PrinciplePreset::Smesh => {
                let mut p = security_principles();
                p.extend(smesh_principles());
                p
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Message;

    #[test]
    fn test_principle_creation() {
        let principle = ConstitutionalPrinciple::new(
            "Test",
            "A test principle",
            "Is this a test?",
            "Make it more testy.",
        );

        assert_eq!(principle.name, "Test");
        assert!(principle.to_instruction().contains("Test"));
    }

    #[test]
    fn test_security_principles() {
        let principles = security_principles();
        assert!(!principles.is_empty());
        assert!(principles.iter().any(|p| p.name.contains("Malicious")));
    }

    #[test]
    fn test_quality_principles() {
        let principles = quality_principles();
        assert!(!principles.is_empty());
        assert!(principles.iter().any(|p| p.name.contains("Error")));
    }

    #[test]
    fn test_apply_principles() {
        let mut request = GenerateRequestV2::simple("Hello");
        let principles = vec![ConstitutionalPrinciple::new(
            "Test",
            "Be good",
            "Is it good?",
            "Make it good.",
        )];

        apply_principles(&mut request, &principles);

        assert!(request.system.is_some());
        assert!(request.system.as_ref().unwrap().contains("Test"));
    }

    #[test]
    fn test_apply_principles_with_existing_system() {
        let mut request =
            GenerateRequestV2::new(vec![Message::user("Hello")]).with_system("You are helpful.");

        let principles = vec![ConstitutionalPrinciple::new(
            "Test",
            "Be good",
            "Is it good?",
            "Make it good.",
        )];

        apply_principles(&mut request, &principles);

        let system = request.system.unwrap();
        assert!(system.contains("You are helpful"));
        assert!(system.contains("Test"));
    }

    #[test]
    fn test_principle_preset() {
        let security = PrinciplePreset::Security.principles();
        let quality = PrinciplePreset::Quality.principles();
        let full = PrinciplePreset::Full.principles();

        assert!(full.len() > security.len());
        assert!(full.len() > quality.len());
        assert_eq!(full.len(), security.len() + quality.len());
    }

    #[test]
    fn test_security_system_prompt() {
        let prompt = security_system_prompt();
        assert!(prompt.contains("security"));
        assert!(prompt.contains("No Malicious Code"));
    }
}
