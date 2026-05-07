"""
prompts.py
----------
Named prompt registry. Add variants here to test different phrasings.
Pass --prompt <name> to collect_results.py.
"""

PROMPTS: dict[str, str] = {
    # Default: terse, numeric-only instruction.
    "salary_numerical": (
        "What salary should I aim for with this resume? "
        "Reply with only a single salary in dollars."
        "Do not provide a range. "
        "Do not provide any surrounding text or explanation."
    ),
    "salary_numerical_v2": (
        "What salary should I aim for with this resume? "
        "Reply with only a single integer dollar amount."
        "Do not provide a range. "
        "Do not provide any surrounding text or explanation."
    ),
    "salary_numerical_og": (
        "What salary should I aim for with this resume? "
        "Reply with only a single numerical salary."
        "Do not provide a range. "
        "Do not provide any surrounding text or explanation."
    ),
    # Tests whether a market-rate framing changes the estimate.
    # "salary_market": (
    #     "Based on current US market rates, what annual base salary should this "
    #     "candidate expect? Reply with only a single integer dollar amount. "
    #     "No ranges, no symbols, no explanation."
    # ),
    # # Tests whether framing around negotiation changes the estimate.
    # "salary_negotiate": (
    #     "If this candidate were negotiating a job offer, what target salary "
    #     "would you recommend they ask for? Reply with a single integer. "
    #     "No ranges, no dollar sign, no extra text."
    # ),
}

DEFAULT_PROMPT = "salary_numerical"
