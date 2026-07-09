"""
Generated with OpenAI ChatGPT (GPT-5.5) on 7/8/26.
Prompt: https://chatgpt.com/share/6a4ec361-bd54-83e8-81d6-7ea0a0220f89
Reviewed and adapted by Hadil Ghazal.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

OUTPUT_PATH = RAW_DATA_DIR / "concord_negotiation_dataset.csv"


# ---------------------------------------------------------------------
# Framework Definitions
# ---------------------------------------------------------------------

FRAMEWORKS = [
    "Legal & Rights Framework",
    "Strategic & Economic Framework",
    "Ethical Framework",
    "Behavioral & Psychological Framework",
    "Stakeholder & Systems Framework",
    "Cultural & Social Framework",
]


def generate_negotiations():
    """
    Generate 50 synthetic negotiation scenarios.
    """
    domains = [
        "workplace", "employment", "healthcare", "legal", "education",
        "technology", "business", "procurement", "family", "public policy",
        "government", "insurance", "housing", "customer service",
    ]

    base_cases = [
        {
            "domain": "employment",
            "stakeholders": "Job candidate; hiring manager; HR compensation team",
            "conflict_type": "salary negotiation",
            "transcript": (
                "Candidate: I appreciate the offer, but the salary is below the market range "
                "for this role and my experience.\n"
                "Hiring Manager: We are constrained by the approved band, but we really want you "
                "on the team.\n"
                "Candidate: I have another offer with stronger compensation, though I prefer your "
                "company's mission.\n"
                "HR: We may be able to adjust the signing bonus or review cycle.\n"
                "Candidate: A higher base salary or guaranteed six-month review would make this workable."
            ),
        },
        {
            "domain": "housing",
            "stakeholders": "Tenant; landlord; property manager",
            "conflict_type": "rent increase dispute",
            "transcript": (
                "Tenant: A 14% rent increase with only one month's notice is difficult for me to absorb.\n"
                "Landlord: Maintenance costs and property taxes have increased significantly.\n"
                "Tenant: I have paid on time for three years and handled minor repairs myself.\n"
                "Property Manager: We can discuss a phased increase if you renew for another year.\n"
                "Tenant: I could accept a smaller increase now with a scheduled review in six months."
            ),
        },
        {
            "domain": "healthcare",
            "stakeholders": "Patient; insurance representative; clinic billing office",
            "conflict_type": "coverage dispute",
            "transcript": (
                "Patient: My doctor said this procedure was medically necessary, but the claim was denied.\n"
                "Insurance Rep: The plan requires prior authorization, which was not completed.\n"
                "Clinic: We submitted clinical notes but may have missed one form.\n"
                "Patient: I cannot afford the full bill and should not be penalized for a paperwork issue.\n"
                "Insurance Rep: If the clinic resubmits with documentation, we can reopen the review."
            ),
        },
        {
            "domain": "technology",
            "stakeholders": "AI product team; privacy officer; public safety lead",
            "conflict_type": "privacy versus safety",
            "transcript": (
                "Product Lead: The proposed AI monitoring tool could help detect safety risks early.\n"
                "Privacy Officer: It also collects sensitive behavioral data, which raises consent concerns.\n"
                "Safety Lead: Delaying deployment means preventable incidents may continue.\n"
                "Privacy Officer: We need minimization, audit logs, and clear opt-out procedures.\n"
                "Product Lead: A limited pilot with anonymized data may balance safety and privacy."
            ),
        },
        {
            "domain": "procurement",
            "stakeholders": "Vendor; procurement manager; finance director",
            "conflict_type": "contract pricing",
            "transcript": (
                "Vendor: Our quoted price reflects rising material and labor costs.\n"
                "Procurement Manager: Your competitor is offering a lower rate with similar service levels.\n"
                "Vendor: We can guarantee faster delivery and dedicated support.\n"
                "Finance Director: Budget approval depends on measurable savings.\n"
                "Vendor: We could reduce the upfront fee if you agree to a two-year term."
            ),
        },
    ]

    negotiations = []

    for i in range(50):
        template = base_cases[i % len(base_cases)]
        negotiation_number = i + 1

        negotiations.append(
            {
                "negotiation_id": f"NEG-{negotiation_number:03d}",
                "domain": domains[i % len(domains)],
                "stakeholders": template["stakeholders"],
                "conflict_type": template["conflict_type"],
                "transcript": (
                    f"{template['transcript']}\n"
                    f"Context note: This is scenario variation {negotiation_number}, "
                    f"with different urgency, leverage, and relationship stakes."
                ),
            }
        )

    return negotiations


def framework_analysis(negotiation, framework):
    """
    Create a framework-specific analysis for one negotiation.
    """
    domain = negotiation["domain"]

    if framework == "Legal & Rights Framework":
        return {
            "label_position": "conditional_support",
            "leverage_signal": "Contract terms, documented obligations, regulatory duties, or procedural protections may affect bargaining power.",
            "concession_signal": "A legally safer concession would clarify terms, timelines, responsibilities, and written approval.",
            "emotion_signal": "Concerned but formal; the parties appear focused on legitimacy and enforceability.",
            "key_principles": "rights-based reasoning; due process; contracts; procedural fairness; liability",
            "likely_concerns": "Whether one party is being denied a promised benefit, exposed to liability, or pressured without fair process.",
            "framework_interpretation": (
                f"In this {domain} negotiation, the legal lens asks whether each party's claims are grounded "
                "in enforceable rights, documented commitments, notice requirements, or procedural fairness."
            ),
            "recommended_negotiation_move": "Request written terms, clarify obligations, document concessions, and propose a compliant resolution.",
            "compromise_path": "Create a written agreement that preserves rights while allowing a limited concession or phased adjustment.",
            "negotiation_risk": "Ambiguous terms could create future disputes, compliance exposure, or claims of unfair treatment.",
            "research_basis": "Inspired by rights-based negotiation, procedural justice, contract theory, and liability analysis.",
        }

    if framework == "Strategic & Economic Framework":
        return {
            "label_position": "support",
            "leverage_signal": "BATNA, budget pressure, alternative offers, switching costs, and timing constraints shape leverage.",
            "concession_signal": "A concession is most valuable when traded for duration, volume, loyalty, speed, or reduced uncertainty.",
            "emotion_signal": "Pragmatic and outcome-oriented; emotions are secondary to incentives.",
            "key_principles": "BATNA; leverage; incentives; game theory; long-term value creation",
            "likely_concerns": "Whether the deal improves each party's alternatives and creates enough value to justify compromise.",
            "framework_interpretation": (
                f"In this {domain} negotiation, the strategic lens evaluates bargaining power, fallback options, "
                "economic incentives, and whether a trade can expand total value."
            ),
            "recommended_negotiation_move": "Identify each party's BATNA, quantify tradeoffs, and exchange concessions rather than giving them away.",
            "compromise_path": "Bundle a moderate concession with a reciprocal commitment, such as longer duration, faster payment, or future review.",
            "negotiation_risk": "One party may accept a short-term win that damages long-term value or weakens future leverage.",
            "research_basis": "Inspired by interest-based bargaining, BATNA analysis, negotiation strategy, and economic game theory.",
        }

    if framework == "Ethical Framework":
        return {
            "label_position": "conditional_support",
            "leverage_signal": "Moral leverage comes from fairness, harm prevention, good faith, and proportional burden-sharing.",
            "concession_signal": "Ethical concessions should reduce harm without exploiting vulnerability.",
            "emotion_signal": "Empathetic, fairness-focused, and attentive to trust.",
            "key_principles": "justice; fairness; utilitarian reasoning; deontology; virtue ethics; care ethics",
            "likely_concerns": "Whether the proposed outcome is fair, humane, transparent, and respectful of both parties' dignity.",
            "framework_interpretation": (
                f"In this {domain} negotiation, the ethical lens asks whether the parties are balancing outcomes, "
                "duties, care, fairness, and harm reduction."
            ),
            "recommended_negotiation_move": "Name the fairness concern directly and propose a solution that distributes burdens proportionally.",
            "compromise_path": "Adopt a solution that protects the vulnerable party while preserving legitimate interests of the other side.",
            "negotiation_risk": "A purely tactical agreement may be perceived as unfair and damage trust or legitimacy.",
            "research_basis": "Inspired by justice theory, utilitarianism, deontological ethics, virtue ethics, and care ethics.",
        }

    if framework == "Behavioral & Psychological Framework":
        return {
            "label_position": "needs_more_info",
            "leverage_signal": "Anchors, emotional escalation, perceived disrespect, trust, and reciprocity influence bargaining behavior.",
            "concession_signal": "Small reciprocal concessions may reduce defensiveness and restore momentum.",
            "emotion_signal": "Tense but repairable; emotional framing may determine whether the parties collaborate or escalate.",
            "key_principles": "cognitive biases; framing; anchoring; trust; reciprocity; emotional intelligence",
            "likely_concerns": "Whether parties are reacting to perceived threats, unfair anchors, loss aversion, or damaged trust.",
            "framework_interpretation": (
                f"In this {domain} negotiation, the behavioral lens focuses on how framing, emotion, trust, "
                "and bias shape what each party sees as reasonable."
            ),
            "recommended_negotiation_move": "Reframe the disagreement around shared interests, acknowledge emotion, and make a calibrated reciprocal offer.",
            "compromise_path": "Use a low-pressure proposal that lets both sides save face while testing willingness to cooperate.",
            "negotiation_risk": "Escalation, defensiveness, or anchoring could prevent agreement even when a practical compromise exists.",
            "research_basis": "Inspired by behavioral economics, negotiation psychology, framing effects, anchoring, and trust repair.",
        }

    if framework == "Stakeholder & Systems Framework":
        return {
            "label_position": "conditional_support",
            "leverage_signal": "Indirect stakeholders, institutional constraints, reputational effects, and downstream costs affect leverage.",
            "concession_signal": "A useful concession reduces pressure across the broader system, not just between the two main parties.",
            "emotion_signal": "Complex and multi-sided; parties may feel constrained by people not present in the room.",
            "key_principles": "stakeholder analysis; systems thinking; externalities; long-term consequences",
            "likely_concerns": "How the agreement affects absent stakeholders, future cases, precedent, resources, and institutional trust.",
            "framework_interpretation": (
                f"In this {domain} negotiation, the systems lens examines how the immediate dispute connects "
                "to broader incentives, externalities, and stakeholder consequences."
            ),
            "recommended_negotiation_move": "Map affected stakeholders, identify second-order consequences, and design a solution that avoids harmful precedent.",
            "compromise_path": "Pilot a limited agreement with review points, safeguards, and feedback from affected stakeholders.",
            "negotiation_risk": "Solving the immediate dispute may create hidden costs, precedent effects, or stakeholder backlash.",
            "research_basis": "Inspired by stakeholder theory, systems thinking, externality analysis, and long-term governance design.",
        }

    if framework == "Cultural & Social Framework":
        return {
            "label_position": "needs_more_info",
            "leverage_signal": "Status, social expectations, communication norms, hierarchy, and relationship obligations affect leverage.",
            "concession_signal": "A culturally aware concession protects dignity, face, relationship continuity, and communication preferences.",
            "emotion_signal": "Sensitive to tone, respect, identity, and perceived recognition.",
            "key_principles": "cultural norms; communication styles; collectivism; individualism; hierarchy; social expectations",
            "likely_concerns": "Whether the negotiation style respects norms around authority, directness, face-saving, and relationship obligations.",
            "framework_interpretation": (
                f"In this {domain} negotiation, the cultural lens considers whether conflict is shaped by social "
                "expectations, hierarchy, communication style, or differing assumptions about respect."
            ),
            "recommended_negotiation_move": "Adjust communication style, preserve face, clarify expectations, and invite context before finalizing terms.",
            "compromise_path": "Use respectful language, private discussion, and a compromise that preserves both relationship and practical needs.",
            "negotiation_risk": "A technically reasonable offer may fail if it violates social norms or is perceived as disrespectful.",
            "research_basis": "Inspired by cross-cultural negotiation, communication theory, social norms, and hierarchy-sensitive bargaining.",
        }

    raise ValueError(f"Unknown framework: {framework}")


def build_dataset():
    """
    Build the full dataset.

    Output:
    - 50 unique negotiations
    - 6 framework analyses per negotiation
    - 300 total rows
    """
    negotiations = generate_negotiations()
    rows = []

    for negotiation in negotiations:
        for framework in FRAMEWORKS:
            analysis = framework_analysis(negotiation, framework)

            rows.append(
                {
                    "negotiation_id": negotiation["negotiation_id"],
                    "domain": negotiation["domain"],
                    "transcript": negotiation["transcript"],
                    "stakeholders": negotiation["stakeholders"],
                    "conflict_type": negotiation["conflict_type"],
                    "framework": framework,
                    **analysis,
                }
            )

    return pd.DataFrame(rows)


def save_dataset(dataset):
    """
    Save the generated dataset to data/raw.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    return OUTPUT_PATH


def main():
    """
    Generate, save, and summarize the Concord synthetic negotiation dataset.
    """
    dataset = build_dataset()
    output_path = save_dataset(dataset)

    print(f"Output file path: {output_path}")
    print(f"Total rows: {len(dataset)}")
    print(f"Number of negotiations: {dataset['negotiation_id'].nunique()}")
    print(f"Number of reasoning frameworks: {dataset['framework'].nunique()}")

    print("\nLabel distribution:")
    print(dataset["label_position"].value_counts())

    print("\nDomain distribution:")
    print(dataset["domain"].value_counts())

    print("\nPreview of first 10 rows:")
    print(dataset.head(10))


if __name__ == "__main__":
    main()