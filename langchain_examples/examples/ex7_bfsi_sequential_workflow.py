# BFSI Research & Content Pipeline
# Covers: Banking · Financial Services · Insurance
# uv add langchain-tavily langchain-openai langchain-core python-dotenv

import os
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch # uv add langchain-tavily
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Tavily Search Tool — scoped to BFSI domains
# ---------------------------------------------------------------------------
tavily_tool = TavilySearch(
    max_results=10,
    topic="finance", # category
    search_depth="advanced",
    include_domains=[
        # Banking & Macro
        "reuters.com",
        "bloomberg.com",
        "ft.com",
        "federalreserve.gov",
        "bis.org",
        "rbi.org.in",
        "bankofengland.co.uk",
        # Financial Services / Capital Markets
        "sebi.gov.in",
        "sec.gov",
        "morningstar.com",
        "investopedia.com",
        # Insurance
        "irdai.gov.in",
        "insurancejournal.com",
        "swissre.com",
        "munichre.com",
        # General BFSI News
        "economictimes.indiatimes.com",
        "livemint.com",
        "businessinsider.com",
    ],
)

# ---------------------------------------------------------------------------
# BFSI Writer Agent
# ---------------------------------------------------------------------------
writer_agent = create_agent(
    model="openai:gpt-4o-mini",
    system_prompt="""You are a Senior BFSI Industry Analyst and financial journalist with 
    15+ years of experience covering the full spectrum of Banking, Financial Services, 
    and Insurance (BFSI). Your areas of deep expertise include:

    BANKING:
    - Retail & corporate banking, credit risk, NPA management
    - Digital banking transformation, neobanks, core banking modernisation
    - Central bank policy, Basel III/IV, RBI & Fed regulations

    FINANCIAL SERVICES:
    - Asset management, mutual funds, wealth management, capital markets
    - Fintech disruption: payments, lending, WealthTech, RegTech
    - SEBI regulations, ESG investing, alternative assets

    INSURANCE:
    - Life, health, general, and reinsurance
    - InsurTech innovations: parametric insurance, embedded insurance, AI underwriting
    - IRDAI regulations, mortality tables, actuarial risk modelling

    Before writing, ALWAYS use the search tool to:
    1. Gather the latest statistics, regulatory updates, and expert commentary.
    2. Validate data from at least 2-3 authoritative BFSI sources.
    3. Include real-world case studies (e.g., HDFC Bank, LIC, Zerodha, Policybazaar, 
       JPMorgan, AXA, BlackRock, Allianz).

    Structure every article as:
    - Executive Summary (≤ 3 sentences)
    - Industry Landscape & Key Drivers
    - Segment Deep-Dive (Banking / Financial Services / Insurance as relevant)
    - Regulatory & Compliance Angle
    - Notable Case Studies
    - Risks & Challenges
    - 3-5 Year Outlook
    
    Target length: 900-1100 words. Cite all statistics with source and year.""",
    tools=[tavily_tool],
)

print("✅ Writer Agent (BFSI Analyst) created successfully.")

# ---------------------------------------------------------------------------
# BFSI Editor Agent
# ---------------------------------------------------------------------------
editor_agent = create_agent(
    model="openai:gpt-4o-mini",
    system_prompt="""You are a Senior BFSI Content Editor with expertise in financial 
    journalism standards and regulatory communication. Your editorial mandate covers all 
    three BFSI verticals: Banking, Financial Services, and Insurance.

    Your editorial checklist:

    1. COMPLIANCE GUARDRAILS
       - Ensure no content constitutes direct investment, insurance, or financial advice.
       - Add a standard disclaimer at the end if the article discusses market predictions, 
         insurance products, or investment instruments.
       - Flag and neutralise any language that could imply guaranteed returns.

    2. ACCURACY & CONSISTENCY
       - Cross-check all regulatory body names: RBI, SEBI, IRDAI, PFRDA, Fed, SEC, FCA.
       - Verify that all figures, percentages, and YoY comparisons are internally consistent.
       - Ensure correct full-forms on first use: NPA (Non-Performing Asset), 
         AUM (Assets Under Management), GWP (Gross Written Premium), etc.

    3. CLARITY & ACCESSIBILITY
       - Define technical terms in parentheses on first use.
       - Simplify actuarial, underwriting, and capital-market jargon for a broad readership.
       - Ensure segment transitions (banking → financial services → insurance) are smooth.

    4. STRUCTURE & STYLE
       - Tighten the Executive Summary to ≤ 3 sentences.
       - Enforce logical flow: context → analysis → case study → outlook.
       - Follow AP financial style guide conventions.""",
)

print("✅ Editor Agent (BFSI Content Editor) created successfully.")


# ---------------------------------------------------------------------------
# Sequential Pipeline
# ---------------------------------------------------------------------------
def run_bfsi_pipeline(topic: str) -> dict:
    """
    Sequential multi-agent BFSI content pipeline:
    1) Writer Agent  → Tavily-powered research + full draft
    2) Editor Agent  → Compliance check, clarity, style refinement
    """
    print(f"\n{'='*60}")
    print(f"📊 BFSI Topic: {topic}")
    print(f"{'='*60}\n")

    # ── Step 1: Writer Agent ────────────────────────────────────────────────
    print("🔍 Writer Agent researching and drafting...\n")

    writer_result = writer_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f"""Research and write a comprehensive BFSI industry article on:

                    '{topic}'

                    Follow these instructions:
                    1. Search for the latest data, regulatory changes, expert opinions, 
                       and market statistics relevant to this topic.
                    2. Identify which BFSI vertical(s) this topic primarily touches 
                       (Banking / Financial Services / Insurance) and go deep on each.
                    3. Include at least 2 real-world case studies — prioritise Indian 
                       institutions where relevant, with global comparisons.
                    4. Cover: current landscape → key drivers → regulatory angle → 
                       risks → 3-5 year outlook.
                    5. Cite all statistics with source name and year in parentheses."""
                )
            ]
        }
    )

    written_content = writer_result["messages"][-1].content
    print("✅ Writer Agent draft delivered to Editor Agent.\n")
    print(f"{'='*60}\n")

    # ── Step 2: Editor Agent ────────────────────────────────────────────────
    print("✏️  Editor Agent refining content...\n")

    editor_result = editor_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f"""Please refine and enhance the following BFSI article:

                    {written_content}

                    Apply the full editorial checklist:
                    - Define all acronyms (NPA, AUM, GWP, IRDAI, SEBI, etc.) on first use
                    - Remove or rephrase any direct financial/insurance advice
                    - Add a disclaimer at the end if market predictions or 
                      investment/insurance products are discussed
                    - Tighten Executive Summary to ≤ 3 sentences
                    - Improve transitions between BFSI verticals if multiple are covered
                    - Verify internal consistency of all figures"""
                )
            ]
        }
    )

    refined_content = editor_result["messages"][-1].content
    print("✅ Editor Agent output ready.\n")
    print(f"{'='*60}\n")

    return {
        "topic": topic,
        "draft": written_content,
        "final": refined_content,
    }


# ---------------------------------------------------------------------------
# BFSI Topic Presets (Banking · Financial Services · Insurance)
# ---------------------------------------------------------------------------
BFSI_TOPICS = {
    # Banking
    1: "AI-Driven Credit Scoring: How Indian Banks Are Reducing NPAs with Machine Learning",
    2: "The UPI Dominance and Its Impact on Traditional Banking Revenue Streams in India",

    # Financial Services
    3: "Rise of Discount Brokers and Democratisation of Retail Investing in India (2020–2026)",
    4: "ESG Investing in Emerging Markets: Opportunities and Greenwashing Risks",

    # Insurance
    5: "InsurTech and the Embedded Insurance Revolution: Disrupting LIC and Traditional Insurers",
    6: "Parametric Insurance for Climate Risk: How Reinsurers Are Pricing the New Normal",

    # Cross-vertical BFSI
    7: "How Generative AI Is Transforming Customer Experience Across the Entire BFSI Sector",
    8: "Open Finance in India: Connecting Banking, Investments, and Insurance via Account Aggregator",
}


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  TAVILY_API_KEY not found. Get a free key at: https://tavily.com")
        exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found.")
        exit(1)

    # ── Print topic menu ────────────────────────────────────────────────────
    print("\n📋 Available BFSI Topics:\n")
    segments = {
        "🏦 Banking":               [1, 2],
        "📈 Financial Services":    [3, 4],
        "🛡️  Insurance":            [5, 6],
        "🔀 Cross-Vertical BFSI":   [7, 8],
    }
    for segment, indices in segments.items():
        print(f"  {segment}")
        for i in indices:
            print(f"    {i}. {BFSI_TOPICS[i]}")
        print()

    choice = int(input("Select a topic (1-8) or press 0 to enter a custom topic: "))
    topic = BFSI_TOPICS.get(choice) or input("Enter your custom BFSI topic: ")

    result = run_bfsi_pipeline(topic)

    # ── Output ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📄 FINAL OUTPUT")
    print("=" * 60)
    print(f"\nTopic: {result['topic']}\n")

    print("\n" + "-" * 60)
    print("📝 Draft Content (with research):")
    print("-" * 60)
    print(result["draft"])

    print("\n" + "-" * 60)
    print("✨ Refined & Compliance-Checked Content:")
    print("-" * 60)
    print(result["final"])