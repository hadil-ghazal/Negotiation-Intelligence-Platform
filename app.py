# Concord web app
#
# Original code developed by Hadil Ghazal on 7/8/26.
# Enhanced with Claude (Sonnet 5, Anthropic) on 7/8/26 to improve UI/UX
# interpretability for non-technical users. Specific enhancements:
#   1. Plain-language verdict translation — raw model labels (e.g.
#      "conditional_support") are mapped to human-readable headlines
#      (e.g. "Proceed, with conditions") via get_position_style().
#   2. Confidence reframed as a qualitative signal-strength tier
#      (Strong / Moderate / Preliminary) rather than a raw percentage,
#      via get_confidence_tier(); the raw percentage is retained as a
#      secondary detail.
#   3. Replaced the markdown-style "Executive Summary" block with a
#      structured verdict card (build_verdict()) plus a visual
#      consensus meter showing per-framework agreement/disagreement.
#   4. Color-semantic system (green/amber/slate/red) applied consistently
#      across the verdict card, framework cards, and confidence chart.
#   5. CSS/layout polish to the framework cards and verdict card for
#      visual hierarchy and SaaS-style presentation.
# All modeling, data pipeline, and core application logic remain
# original work by Hadil Ghazal.

#imports
from pathlib import Path
import os
import pickle
import tempfile
 
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
 
 
PROJECT_ROOT = Path(__file__).resolve().parent
 
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "concord_negotiation_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
 
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
 
 
FRAMEWORKS = [
    "Legal & Rights Framework",
    "Strategic & Economic Framework",
    "Ethical Framework",
    "Behavioral & Psychological Framework",
    "Stakeholder & Systems Framework",
    "Cultural & Social Framework",
]
 
 
FRAMEWORK_ICONS = {
    "Legal & Rights Framework": "⚖️",
    "Strategic & Economic Framework": "📊",
    "Ethical Framework": "🧭",
    "Behavioral & Psychological Framework": "🧠",
    "Stakeholder & Systems Framework": "🕸️",
    "Cultural & Social Framework": "🌐",
}
 
# Translates raw model labels into plain-language verdicts a non-technical
# reader can act on immediately. "tone" drives color semantics everywhere
# (cards, badges, consensus meter) so the same 4 colors mean the same thing
# throughout the whole app.
POSITION_STYLES = {
    "support": {"headline": "Green light to proceed", "tone": "positive"},
    "conditional_support": {"headline": "Proceed, with conditions", "tone": "caution"},
    "needs_more_info": {"headline": "Not enough to decide yet", "tone": "neutral"},
    "oppose": {"headline": "Recommend against", "tone": "negative"},
    "reject": {"headline": "Recommend against", "tone": "negative"},
}
 
TONE_META = {
    "positive": {"color": "#059669", "bg": "#ecfdf5", "border": "#a7f3d0", "icon": "✓"},
    "caution": {"color": "#b45309", "bg": "#fffbeb", "border": "#fde68a", "icon": "◐"},
    "neutral": {"color": "#475569", "bg": "#f1f5f9", "border": "#e2e8f0", "icon": "?"},
    "negative": {"color": "#b91c1c", "bg": "#fef2f2", "border": "#fecaca", "icon": "✗"},
}
 
 
def get_position_style(raw_label: str) -> dict:
    """Looking up plainlanguage copy for a raw model label
    Falls bakc to a title cased version of the raw label with a neutral
    tone if the lable encoder ever produces something we havent mapped,
    so the UI never breaks or shows an unhandled internal value
    """
    key = str(raw_label).strip().lower()
    style = POSITION_STYLES.get(key)
    if style is None:
        style = {"headline": str(raw_label).replace("_", " ").title(), "tone": "neutral"}
    return {**style, **TONE_META[style["tone"]]}
 
 
def get_confidence_tier(confidence: float) -> dict:
    """Converting a raw softmax confidence score into a plainlang signal
    strength, with a short actionable hint : "moderate confidence" means
    nothing to most people on its own, but "worth a second look tells you
    what to actually do with that information
    """
    if confidence >= 0.75:
        return {"label": "Strong signal", "dots": 3, "hint": ""}
    if confidence >= 0.5:
        return {"label": "Moderate signal", "dots": 2, "hint": "worth a second look"}
    return {"label": "Preliminary read", "dots": 1, "hint": "verify manually before relying on this"}
 
 
# Concrete, actionable guidance for what each framework's verdict actually
# means to *do*, keyed by [framework][tone]. This replaces abstract lens
# descriptions ("focuses on procedural fairness") with a direct answer to
# "so what should I do" for that framework + predicted position combo.
#
# NOTE ON SCOPE: the underlying classifier predicts one label per
# (transcript, framework) pair. It has no concept of "party" in its
# training data or label space, so it cannot produce a genuinely distinct
# prediction for each side of the negotiation. The two dicts below
# (decision-maker vs. requesting-party) are both hand-authored templates
# applied to the SAME predicted label/tone — they demonstrate what a real
# per-party view could look like, but they are not independently modeled.
# See the "requester_perspective" toggle in the UI, which is labeled as a
# template demo for this reason. Real per-party analysis would require
# either party-labeled training data or a generative model call grounded
# in the actual transcript (see Future Work).
FRAMEWORK_ACTION_GUIDANCE = {
    "Legal & Rights Framework": {
        "positive": "You have no legal or contractual barrier here — you can approve this without added risk.",
        "caution": "Nothing blocks you legally, but put the terms in writing so you and the other party both have an enforceable record of exactly what was agreed.",
        "neutral": "Before you decide, check what any existing policy, contract, or past precedent actually requires — that's what should settle this for you.",
        "negative": "Approving this as-is likely puts you in conflict with existing policy or precedent — get it reviewed before you move forward.",
    },
    "Strategic & Economic Framework": {
        "positive": "The trade-off favors you saying yes — your cost of agreeing is lower than your cost of the friction from saying no.",
        "caution": "It's a reasonable trade for you, but attach conditions that protect your leverage — offer a trial period or a review date, not an open-ended yes.",
        "neutral": "You don't have enough on the cost/benefit side yet — ask the other party for specifics (numbers, timeline, alternatives) before you commit.",
        "negative": "The trade-off doesn't favor you agreeing right now — the cost or precedent risk outweighs the benefit to you.",
    },
    "Ethical Framework": {
        "positive": "Granting this treats both you and the other party fairly and doesn't create an unfair advantage either way.",
        "caution": "It's fair for you to grant in principle, but only if you offer it consistently to others in the same position going forward.",
        "neutral": "Whether this is fair for you to grant depends on a detail you don't have yet — whether others in the same situation would get the same treatment from you.",
        "negative": "Granting this as requested would have you treating similarly-situated people inconsistently — fix that before you agree.",
    },
    "Behavioral & Psychological Framework": {
        "positive": "You saying yes here builds trust and isn't likely to trigger resentment from others you manage or work with.",
        "caution": "You can say yes, but communicate your reasoning openly so your decision doesn't read as favoritism to everyone else watching.",
        "neutral": "You don't have a clear read on the trust stakes yet — talk to the other party directly before you decide; an email won't surface it.",
        "negative": "The way this is currently framed risks you damaging trust or morale — slow down before you respond.",
    },
    "Stakeholder & Systems Framework": {
        "positive": "This decision is contained for you — unlikely to ripple out to other teams or set a precedent you'll have to manage later.",
        "caution": "Fine for you to approve, but expect others to ask you for the same thing — decide your policy on that now, not after the second request.",
        "neutral": "You don't yet know who else this decision touches — map out the other people or teams affected before you commit.",
        "negative": "Approving this sets a precedent that's likely to create problems for you with other teams or stakeholders down the line.",
    },
    "Cultural & Social Framework": {
        "positive": "This aligns with how you'd normally handle similar requests — no friction expected on your end.",
        "caution": "Fine for you to grant, but be deliberate about how you communicate it — your tone and framing matter as much as the decision.",
        "neutral": "How this lands for you depends on norms you haven't confirmed yet — check how you've handled similar requests before.",
        "negative": "This cuts against the norms you'd normally expect and is likely to be read by others as unusual or unfair if you grant it.",
    },
}
 
# Same predicted label/tone, reframed as advice to the party who INITIATED
# the ask, rather than the party deciding on it. Hand-authored template,
# not a second model output — see the note above FRAMEWORK_ACTION_GUIDANCE.
FRAMEWORK_ACTION_GUIDANCE_REQUESTER = {
    "Legal & Rights Framework": {
        "positive": "There's no legal barrier stopping you from asking for this — you're on solid ground pushing for it.",
        "caution": "You can reasonably push for this, but be ready to put your ask in writing so there's a clear record of exactly what you're requesting and why.",
        "neutral": "Before you push further, find out what policy or precedent actually says — that's what will decide whether your ask holds up.",
        "negative": "Your ask likely runs into existing policy or precedent — expect pushback, and come with a stronger justification.",
    },
    "Strategic & Economic Framework": {
        "positive": "The trade-off favors you here — what you're asking for costs the other side less than saying no would cost them in friction.",
        "caution": "Your ask is reasonable, but offer something in return — a trial period or a concession — to make it easier for the other side to say yes.",
        "neutral": "You haven't given the other side enough to evaluate the trade-off — be ready with specifics (numbers, timeline, alternatives) when they ask.",
        "negative": "Right now the trade-off doesn't clearly favor your ask — expect resistance unless you can show more upside for the other side.",
    },
    "Ethical Framework": {
        "positive": "What you're asking for is fair to both sides and doesn't put you ahead unfairly.",
        "caution": "Your ask is fair in principle, but be prepared to accept the same standard being applied to others in your position.",
        "neutral": "Whether your ask is fair depends on how others in your position have been treated — that's worth raising directly.",
        "negative": "Your ask, as framed, could look like you're asking for inconsistent treatment — reframe it before pushing further.",
    },
    "Behavioral & Psychological Framework": {
        "positive": "Asking for this is unlikely to damage trust — it should land as a reasonable request.",
        "caution": "Your ask is reasonable, but how you communicate it matters — explain your reasoning so it doesn't come across as entitled.",
        "neutral": "You don't have a clear read on how this will land emotionally — have a direct conversation instead of relying on email.",
        "negative": "The way your ask is currently framed risks damaging trust — consider softening it or explaining your reasoning more.",
    },
    "Stakeholder & Systems Framework": {
        "positive": "Your ask is unlikely to create ripple effects for other people or teams.",
        "caution": "Your ask is reasonable, but expect the other side to think about precedent — be ready to explain why your situation is specific.",
        "neutral": "You don't know who else this decision affects on the other side — anticipate that question before you push further.",
        "negative": "Your ask could create a precedent problem for the other side — expect that to be a real objection, not just an excuse.",
    },
    "Cultural & Social Framework": {
        "positive": "Your ask aligns with how similar requests are normally handled — it shouldn't feel unusual.",
        "caution": "Your ask is reasonable, but be thoughtful about tone — how you frame it matters as much as what you're asking for.",
        "neutral": "How your ask lands depends on norms you haven't confirmed — it may be worth asking how similar requests have been handled before.",
        "negative": "Your ask cuts against expected norms here — be ready for it to be read as unusual, and consider reframing it.",
    },
}
 
PERSPECTIVE_GUIDANCE = {
    "decision_maker": FRAMEWORK_ACTION_GUIDANCE,
    "requester": FRAMEWORK_ACTION_GUIDANCE_REQUESTER,
}
 
PERSPECTIVE_LABELS = {
    "decision_maker": "Deciding party",
    "requester": "Requesting party",
}
 
 
def get_action_guidance(framework: str, tone: str, perspective: str = "decision_maker") -> str:
    """Concrete 'what this means for you' sentence for a framework+tone pair,
    for the given perspective ("decision_maker" or "requester"). Falls back
    to a generic-but-still-direct sentence for any framework/tone
    combination not explicitly authored above"""
    fallback = {
        "decision_maker": {
            "positive": "This framework doesn't flag a problem for you — no changes needed from this angle.",
            "caution": "This framework says it's workable for you, but only with clear conditions attached.",
            "neutral": "This framework doesn't give you enough to go on yet — get more specifics before you decide.",
            "negative": "This framework flags a real concern for you — address it before you move forward.",
        },
        "requester": {
            "positive": "This framework doesn't flag a problem with your ask.",
            "caution": "This framework says your ask is workable, but be ready to compromise on the details.",
            "neutral": "This framework doesn't give you enough to strengthen your ask yet — gather more specifics.",
            "negative": "This framework flags a real weakness in your ask — expect it to be challenged.",
        },
    }
    guidance = PERSPECTIVE_GUIDANCE.get(perspective, FRAMEWORK_ACTION_GUIDANCE)
    return guidance.get(framework, {}).get(tone, fallback[perspective][tone])
 
 
def load_pickle(path: Path):
    with open(path, "rb") as file:
        return pickle.load(file)
 
 
def build_model_text(transcript: str, framework: str) -> str:
    return (
        f"Framework: {framework}\n\n"
        f"Negotiation Transcript: {transcript}\n\n"
        f"Framework Interpretation: assess this negotiation through the {framework}.\n\n"
        f"Recommended Move: identify whether support, conditional support, or more information is needed\n\n"
        f"Compromise Path: Recommend a practical path toward resolution"
    )
 
 
def train_fallback_model():
# Fallback for deployment, local trained elements prefered but if render 
#... doesnt have them, then this model is fit from the synthetic raw dataset for demo preservation
    df = pd.read_csv(RAW_DATA_PATH)
 
    texts = (
        "Framework: " + df["framework"].astype(str)
        + "\n\nNegotiation Transcript: " + df["transcript"].astype(str)
        + "\n\nFramework Interpretation: " + df["framework_interpretation"].astype(str)
        + "\n\nRecommended Move: " + df["recommended_negotiation_move"].astype(str)
        + "\n\nCompromise Path: " + df["compromise_path"].astype(str)
    )
 
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["label_position"])
 
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
 
    features = vectorizer.fit_transform(texts)
 
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
 
    model.fit(features, labels)
    return model, vectorizer, label_encoder
 
 
def load_model_assets():
    if LOGISTIC_MODEL_PATH.exists() and VECTORIZER_PATH.exists() and LABEL_ENCODER_PATH.exists():
        return (
            load_pickle(LOGISTIC_MODEL_PATH),
            load_pickle(VECTORIZER_PATH),
            load_pickle(LABEL_ENCODER_PATH),
        )
 
    return train_fallback_model()
 
 
MODEL, VECTORIZER, LABEL_ENCODER = load_model_assets()
 
 
def create_confidence_chart(results_df: pd.DataFrame):
    bar_colors = [get_position_style(pos)["color"] for pos in results_df["predicted_position"]]
    headlines = [get_position_style(pos)["headline"] for pos in results_df["predicted_position"]]
 
    fig = go.Figure(
        data=[
            go.Bar(
                x=results_df["framework"],
                y=results_df["confidence"],
                text=[f"{h}<br>{c:.0%}" for h, c in zip(headlines, results_df["confidence"])],
                textposition="outside",
                marker_color=bar_colors,
                hovertext=headlines,
                hoverinfo="text+y",
            )
        ]
    )
 
    fig.update_layout(
        title="How strongly each framework holds its position",
        xaxis_title=None,
        yaxis_title="Signal strength",
        yaxis=dict(range=[0, 1.15], tickformat=".0%"),
        height=420,
        margin=dict(l=40, r=40, t=60, b=120),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
    )
 
    return fig


def get_framework_scenario_guidance(framework: str, transcript: str, position: str) -> str:
    #Return scenario-specific guidance for each framework card
    scenario = detect_scenario(transcript)

    guidance = {
        "remote_work": {
            "Legal & Rights Framework": "Check whether remote-work approvals are governed by policy, precedent, disability accommodation rules, or manager discretion.",
            "Strategic & Economic Framework": "Evaluate productivity, retention risk, collaboration cost, and whether a trial period protects both sides.",
            "Ethical Framework": "Make sure similarly situated employees would be treated consistently.",
            "Behavioral & Psychological Framework": "Explain the decision clearly so it does not create resentment or perceived favoritism.",
            "Stakeholder & Systems Framework": "Consider team coverage, meeting availability, and whether this becomes a broader workplace precedent.",
            "Cultural & Social Framework": "Frame the decision around norms of flexibility, trust, and accountability.",
        },
        "vendor_contract": {
            "Legal & Rights Framework": "Review contract terms, renewal rights, SLA language, and whether outages justify credits or renegotiation.",
            "Strategic & Economic Framework": "Compare renewal cost against switching cost, outage impact, and vendor leverage.",
            "Ethical Framework": "Balance fair pricing with accountability for prior service failures.",
            "Behavioral & Psychological Framework": "Use the outage history to reset trust without making the negotiation purely punitive.",
            "Stakeholder & Systems Framework": "Consider downstream users, operations teams, and business continuity risk.",
            "Cultural & Social Framework": "Set expectations for responsiveness, transparency, and partnership norms.",
        },
        "family_care": {
            "Legal & Rights Framework": "Clarify any formal care, financial, or medical decision responsibilities.",
            "Strategic & Economic Framework": "Balance time contributions, financial contributions, opportunity cost, and burnout risk.",
            "Ethical Framework": "Make the arrangement fair to both siblings and centered on the parent’s wellbeing.",
            "Behavioral & Psychological Framework": "Address resentment directly before assigning duties.",
            "Stakeholder & Systems Framework": "Include backup care, medical needs, schedules, and other family members affected.",
            "Cultural & Social Framework": "Acknowledge family expectations, gender norms, birth-order assumptions, or cultural duties that may shape the conflict.",
        },
        "general": {
            "Legal & Rights Framework": "Clarify rules, obligations, and enforceable expectations.",
            "Strategic & Economic Framework": "Identify leverage, tradeoffs, costs, and incentives.",
            "Ethical Framework": "Check whether the outcome is fair and harm-reducing.",
            "Behavioral & Psychological Framework": "Watch for trust, emotion, resentment, or escalation risk.",
            "Stakeholder & Systems Framework": "Map who else is affected by the agreement.",
            "Cultural & Social Framework": "Account for norms, expectations, and communication style.",
        },
    }

    return guidance.get(scenario, guidance["general"]).get(
        framework,
        "Use this framework to clarify what risk or compromise condition matters most."
    )




#def create_framework_cards(results_df: pd.DataFrame, perspective: str = "decision_maker") -> str:
def create_framework_cards(results_df: pd.DataFrame, transcript: str, perspective: str = "decision_maker") -> str:
    cards = ""
    for _, row in results_df.iterrows():
        framework = row["framework"]
        position = row["predicted_position"]
        confidence = row["confidence"]
 
        style = get_position_style(position)
        tier = get_confidence_tier(confidence)
        icon = FRAMEWORK_ICONS.get(framework, "•")
        action_text = get_action_guidance(framework, style["tone"], perspective)
        scenario_text = get_framework_scenario_guidance(framework, transcript, position)
        dots = "".join(
            f'<span class="dot {"filled" if i < tier["dots"] else ""}"></span>'
            for i in range(3)
        )
        hint = f" — {tier['hint']}" if tier["hint"] else ""
 
        cards += f"""
        <div class="framework-card" style="--accent:{style['color']};">
            <div class="framework-card-head">
                <span class="framework-icon">{icon}</span>
                <h3>{framework}</h3>
            </div>
            <div class="verdict-badge" style="color:{style['color']}; background:{style['bg']}; border-color:{style['border']};">
                <span>{style['icon']}</span> {style['headline']}
            </div>
            <p class="why-line">{action_text}</p>
            <p class="scenario-line"><b>Scenario read:</b> {scenario_text}</p>
            <div class="confidence-footer">
                <span class="dots">{dots}</span>
                <span class="tier-label">{tier['label']}{hint}</span>
                <span class="raw-confidence">({confidence:.0%})</span>
            </div>
        </div>
        """
    return cards
 
 

def detect_scenario(transcript: str) -> str:
    """Detect the negotiation domain from transcript keywords."""
    text = transcript.lower()

    if any(word in text for word in ["remote", "work from home", "office", "employee", "manager"]):
        return "remote_work"
    if any(word in text for word in ["vendor", "contract", "renewal", "price", "service outage"]):
        return "vendor_contract"
    if any(word in text for word in ["sibling", "parent", "care", "family", "aging"]):
        return "family_care"
    if any(word in text for word in ["salary", "raise", "promotion", "compensation"]):
        return "compensation"
    if any(word in text for word in ["rent", "tenant", "landlord", "housing", "lease"]):
        return "housing"

    return "general"



def get_dynamic_compromise(transcript: str, top_position: str) -> str:
    """Create a transcript-aware compromise recommendation."""
    scenario = detect_scenario(transcript)

    recommendations = {
        "remote_work": {
            "support": "Approve the remote-work request with written expectations for availability, productivity, communication, and review timing.",
            "conditional_support": "Offer a 60–90 day remote-work trial with measurable productivity expectations, core collaboration hours, and a fairness policy for similar requests.",
            "needs_more_info": "Request more detail on schedule, team coverage, productivity tracking, and whether similar employees would receive the same option.",
        },
        "vendor_contract": {
            "support": "Proceed with renewal, but document service expectations, pricing terms, and escalation rights.",
            "conditional_support": "Approve renewal only if the vendor accepts service credits, performance guarantees, or phased pricing tied to uptime improvements.",
            "needs_more_info": "Request outage history, SLA performance, pricing justification, and alternative vendor comparisons before renewing.",
        },
        "family_care": {
            "support": "Move toward agreement by formally dividing caregiving and financial responsibilities.",
            "conditional_support": "Create a shared care plan with defined weekly tasks, financial contributions, backup coverage, and a monthly family check-in.",
            "needs_more_info": "Clarify actual care hours, financial capacity, medical needs, and each sibling’s availability before assigning responsibilities.",
        },
        "compensation": {
            "support": "Proceed with the compensation adjustment while documenting rationale and expectations.",
            "conditional_support": "Tie compensation change to role scope, performance milestones, market comparison, or a scheduled review date.",
            "needs_more_info": "Gather role benchmarks, performance evidence, budget constraints, and promotion criteria before deciding.",
        },
        "housing": {
            "support": "Proceed with agreement while documenting payment terms, maintenance duties, and communication expectations.",
            "conditional_support": "Move forward only with clear written terms around rent, repairs, timing, and consequences if obligations are missed.",
            "needs_more_info": "Clarify lease terms, payment history, repair obligations, and legal responsibilities before finalizing.",
        },
        "general": {
            "support": "Proceed toward agreement while documenting roles, timing, and expectations.",
            "conditional_support": "Move forward with conditions, safeguards, and a review point before final acceptance.",
            "needs_more_info": "Pause final agreement until the parties clarify unresolved facts, risks, and obligations.",
        },
    }

    return recommendations[scenario].get(top_position, recommendations[scenario]["conditional_support"])




def build_verdict(results_df: pd.DataFrame, perspective: str = "decision_maker") -> dict:
    """Build the plain-language verdict shown at the top of the results.
 
    Names which frameworks hold which position (not just a count), and
    turns any dissenting frameworks into a concrete checklist of what to
    resolve — "the rest see it differently" tells you nothing actionable,
    "Ethical and Stakeholder frameworks flag X — resolve before finalizing"
    does. `perspective` controls whether the checklist/bottom line is
    phrased for the deciding party or the requesting party (see
    PERSPECTIVE_GUIDANCE — both are templates over the same prediction).
    """
    position_counts = results_df["predicted_position"].value_counts()
    top_position = position_counts.idxmax()
    total = len(results_df)
    top_style = get_position_style(top_position)
 
    # Group frameworks by their predicted position, in descending group
    # size, so the breakdown reads "majority first, then each dissent."
    groups = []
    for position, group_df in results_df.groupby("predicted_position"):
        groups.append((len(group_df), position, group_df["framework"].tolist()))
    groups.sort(key=lambda g: -g[0])
 
    def framework_list_text(names):
        short = [n.split(" & ")[0].split(" Framework")[0] for n in names]
        if len(short) == 1:
            return short[0]
        return ", ".join(short[:-1]) + f" and {short[-1]}"
 
    breakdown_parts = []
    for count, position, names in groups:
        pos_style = get_position_style(position)
        breakdown_parts.append(
            f"{framework_list_text(names)} "
            f"{'says' if len(names) == 1 else 'say'} "
            f'\u201c{pos_style["headline"].lower()}\u201d ({count}/{total})'
        )
    breakdown_line = "; ".join(breakdown_parts) + "."
 
    # Concrete checklist: for every group that disagrees with the majority,
    # surface that framework's specific concern as an action item, phrased
    # for whichever party is currently selected in the UI.
    checklist = []
    for count, position, names in groups:
        if position == top_position:
            continue
        pos_style = get_position_style(position)
        for name in names:
            checklist.append(get_action_guidance(name, pos_style["tone"], perspective))
 
    if not checklist:
        bottom_line = f"Bottom line: {get_action_guidance(results_df.iloc[0]['framework'], top_style['tone'], perspective)}"
    else:
        bottom_line = (
            f"Bottom line: you can move forward on {top_style['headline'].lower()}, "
            f"but resolve these before finalizing:"
        )
 
    # Build one consensus-meter segment per framework, colored by that
    # framework's own predicted position, in framework order (stable, not
    # sorted) so the meter always reads left-to-right the same way.
    segments = ""
    for _, row in results_df.iterrows():
        seg_style = get_position_style(row["predicted_position"])
        segments += (
            f'<span class="segment" style="background:{seg_style["color"]};" '
            f'title="{row["framework"]}: {seg_style["headline"]}"></span>'
        )
 
    return {
        "headline": top_style["headline"],
        "color": top_style["color"],
        "bg": top_style["bg"],
        "border": top_style["border"],
        "icon": top_style["icon"],
        "breakdown_line": breakdown_line,
        "bottom_line": bottom_line,
        "checklist": checklist,
        "segments_html": segments,
    }
 
 
def verdict_to_report_text(verdict: dict, results_df: pd.DataFrame) -> str:
    """Plain-text/markdown version of the verdict and used only for the
    downloadable report file - NOT the on-screen UI """
    lines = [
        f"**Overall verdict:** {verdict['headline']}",
        "",
        verdict["breakdown_line"],
        "",
        verdict["bottom_line"],
    ]
    for item in verdict["checklist"]:
        lines.append(f"- {item}")
    return "\n".join(lines)
 
 
def create_report_file(verdict: dict, results_df: pd.DataFrame) -> str:
    report = "# Concord Negotiation Analysis Report\n\n"
    report += verdict_to_report_text(verdict, results_df) + "\n\n"
    report += "## Framework Results\n\n"
    report += results_df.to_markdown(index=False)
 
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w")
    temp_file.write(report)
    temp_file.close()
 
    return temp_file.name
 
 
def analyze_negotiation(transcript: str, perspective_label: str = "Deciding party"):
    perspective = "requester" if perspective_label == "Requesting party" else "decision_maker"
 
    if not transcript or len(transcript.strip()) < 25:
        return (
            "Please enter a longer negotiation transcript.",
            pd.DataFrame(),
            None,
            None,
        )
 
    rows = []
 
    for framework in FRAMEWORKS:
        model_text = build_model_text(transcript, framework)
        features = VECTORIZER.transform([model_text])
 
        prediction = MODEL.predict(features)[0]
        probabilities = MODEL.predict_proba(features)[0]
 
        predicted_label = LABEL_ENCODER.inverse_transform([prediction])[0]
        confidence = float(max(probabilities))
 
        rows.append(
            {
                "framework": framework,
                "predicted_position": predicted_label,
                "confidence": confidence,
            }
        )
 
    results_df = pd.DataFrame(rows)
    verdict = build_verdict(results_df, perspective)
    verdict["dynamic_compromise"] = get_dynamic_compromise(transcript, results_df["predicted_position"].mode()[0])
    #cards_html = create_framework_cards(results_df, perspective)
    #V2 fix
    cards_html = create_framework_cards(results_df, transcript, perspective)
    chart = create_confidence_chart(results_df)
    report_file = create_report_file(verdict, results_df)
 
    checklist_html = ""
    if verdict["checklist"]:
        items = "".join(f"<li>{item}</li>" for item in verdict["checklist"])
        checklist_html = f'<ul class="verdict-checklist">{items}</ul>'
 
    perspective_banner = ""
    if perspective == "requester":
        perspective_banner = (
            '<div class="perspective-banner">'
            "<b>Demo view — templated, not model-generated:</b> this reframes the same "
            "framework predictions as advice to the requesting party. The underlying model "
            "predicts one label per framework and has no separate party-level output; a "
            "real per-party analysis would need party-labeled training data or a generative "
            "model grounded in the transcript (see Future Work)."
            "</div>"
        )
 
    full_html = f"""
    {perspective_banner}
    <div class="verdict-card" style="--accent:{verdict['color']};">
        <span class="verdict-eyebrow">Overall verdict — {PERSPECTIVE_LABELS[perspective]} view</span>
        <h2 style="color:{verdict['color']};">{verdict['icon']} {verdict['headline']}</h2>
        <p class="verdict-line">{verdict['breakdown_line']}</p>
        <div class="consensus-meter">{verdict['segments_html']}</div>
        <div class="verdict-next-step">
            <p class="bottom-line-text"><b>{verdict['bottom_line']}</b></p>
            <p><b>Scenario-specific compromise:</b> {verdict['dynamic_compromise']}</p>
            {checklist_html}
        </div>
    </div>
 
    <h2 class="section-title">Framework-by-framework breakdown</h2>
    <div class="framework-grid">
        {cards_html}
    </div>
    """
 
    return full_html, results_df, chart, report_file
 
 
 
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
 
* {
    font-family: 'Inter', sans-serif;
}
 
body {
    background:
        radial-gradient(circle at top left, rgba(99,102,241,0.20), transparent 32%),
        linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #f9fafb 100%);
}
 
.gradio-container {
    max-width: 1240px !important;
    margin: auto !important;
}
 
.hero {
    background:
        linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.96)),
        radial-gradient(circle at top right, rgba(129,140,248,0.45), transparent 35%);
    color: white !important;
    padding: 42px;
    border-radius: 28px;
    margin: 20px 0 28px 0;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.28);
}
 
.hero * {
    color: white !important;
}
 
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.20);
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 18px;
    color: rgba(255,255,255,0.85) !important;
}
 
.hero h1 {
    font-size: 58px;
    line-height: 1;
    margin: 0 0 12px 0;
    letter-spacing: -0.05em;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff;
}
 
.hero p {
    font-size: 18px;
    color: #dbeafe !important;
    max-width: 850px;
    line-height: 1.55;
}
 
.panel {
    background: rgba(255,255,255,0.90);
    border: 1px solid rgba(226,232,240,0.95);
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
}
 
/* --- Verdict card: the single most important thing on the results side --- */
.verdict-card {
    background: #ffffff;
    padding: 30px 32px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    border-left: 5px solid var(--accent, #475569);
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    margin-bottom: 22px;
}
 
.verdict-eyebrow {
    display: inline-block;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 8px;
}
 
.verdict-card h2 {
    margin: 0 0 10px 0;
    font-size: 30px;
    letter-spacing: -0.03em;
}
 
.verdict-line {
    color: #1e293b;
    font-size: 16px;
    line-height: 1.6;
    margin: 0 0 18px 0;
}
 
.consensus-meter {
    display: flex;
    gap: 4px;
    height: 10px;
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 18px;
}
 
.consensus-meter .segment {
    flex: 1;
    display: block;
    height: 100%;
}
 
.verdict-next-step {
    font-size: 14.5px;
    color: #334155;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 14px 16px;
    margin: 0;
    line-height: 1.55;
}
 
.verdict-next-step .bottom-line-text {
    margin: 0 0 4px 0;
    color: #0f172a;
}
 
.verdict-checklist {
    margin: 8px 0 0 0;
    padding-left: 20px;
}
 
.verdict-checklist li {
    margin-bottom: 6px;
    color: #334155;
}
 
.section-title {
    font-size: 20px;
    letter-spacing: -0.02em;
    color: #0f172a;
    margin: 6px 0 14px 2px;
}
 
.perspective-banner {
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    color: #3730a3;
    font-size: 13px;
    line-height: 1.5;
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 16px;
}
 
.framework-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 18px;
    margin-top: 14px;
}
 
.framework-card {
    background: #ffffff;
    padding: 20px 22px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    border-left: 4px solid var(--accent, #cbd5e1);
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
}
 
.framework-card-head {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}
 
.framework-icon {
    font-size: 18px;
    line-height: 1;
}
 
.framework-card h3 {
    margin: 0;
    color: #0f172a;
    font-size: 16px;
    letter-spacing: -0.01em;
}
 
.verdict-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid;
    padding: 7px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 12px;
}
 
.why-line {
    color: #475569;
    font-size: 14px;
    line-height: 1.55;
    margin: 0 0 14px 0;
}
 
.confidence-footer {
    display: flex;
    align-items: center;
    gap: 8px;
    padding-top: 10px;
    border-top: 1px dashed #e2e8f0;
}
 
.confidence-footer .dots {
    display: inline-flex;
    gap: 3px;
}
 
.confidence-footer .dot {
    width: 6px;
    height: 6px;
    border-radius: 999px;
    background: #e2e8f0;
}
 
.confidence-footer .dot.filled {
    background: #94a3b8;
}
 
.confidence-footer .tier-label {
    font-size: 12.5px;
    font-weight: 600;
    color: #64748b;
}
 
.confidence-footer .raw-confidence {
    font-size: 12px;
    color: #94a3b8;
    margin-left: auto;
}
 
button.primary {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 800 !important;
    box-shadow: 0 12px 28px rgba(79,70,229,0.32) !important;
}
 
textarea {
    border-radius: 18px !important;
    border: 1px solid #cbd5e1 !important;
}
 
@media (max-width: 900px) {
    .framework-grid {
        grid-template-columns: 1fr;
    }
 
    .hero h1 {
        font-size: 42px;
    }
}
"""
 
 
 
 
with gr.Blocks(css=CSS, title="Concord") as demo:
    gr.HTML(
        """
        <div class="hero">
            <div class="hero-badge">AI Negotiation Intelligence</div>
            <h1>Concord</h1>
            <p>
                Analyze one negotiation across six reasoning frameworks to surface
                consensus, disagreement, risk, and compromise opportunities.
            </p>
        </div>
        """
    )
 
    # Pre-declare the output components (render=False) so gr.Examples can
    # wire them as outputs before they're visually placed in the right
    # column below. This keeps the auto-run-on-example-click behavior
    # while preserving the original two-column layout.
    output_html = gr.HTML(render=False)
    results_table = gr.Dataframe(label="Framework Prediction Table", render=False)
    confidence_chart = gr.Plot(label="Confidence Dashboard", render=False)
    report_download = gr.File(label="Download Markdown Report", render=False)
 
    with gr.Row():
        with gr.Column(scale=1):
            transcript_input = gr.Textbox(
                label="Negotiation Transcript",
                placeholder="Paste a negotiation transcript here...",
                lines=14,
            )
 
            perspective_radio = gr.Radio(
                choices=["Deciding party", "Requesting party"],
                value="Deciding party",
                label="View guidance for",
                info="Demo toggle: reframes the same predictions using pre-written templates — see the note below the results.",
            )
 
            analyze_button = gr.Button("Analyze Negotiation", variant="primary")
 
            gr.Examples(
                examples=[
                    ["An employee asks to work remotely three days per week after relocating farther from the office. The manager is concerned about team availability, fairness to other employees, and maintaining productivity."],
                    ["A software vendor requests a price increase during contract renewal. The client argues that service outages from the prior quarter should offset any increase."],
                    ["Two siblings are negotiating care responsibilities for an aging parent. One sibling provides more daily support while the other contributes more financially."],
                ],
                inputs=transcript_input,
                outputs=[output_html, results_table, confidence_chart, report_download],
                fn=analyze_negotiation,
                run_on_click=True,
                cache_examples=False,
            )
 
        with gr.Column(scale=1):
            output_html.render()
            results_table.render()
            confidence_chart.render()
            report_download.render()
 
    analyze_button.click(
        fn=analyze_negotiation,
        inputs=[transcript_input, perspective_radio],
        outputs=[
            output_html,
            results_table,
            confidence_chart,
            report_download,
        ],
    )
 
    perspective_radio.change(
        fn=analyze_negotiation,
        inputs=[transcript_input, perspective_radio],
        outputs=[
            output_html,
            results_table,
            confidence_chart,
            report_download,
        ],
    )
 
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)