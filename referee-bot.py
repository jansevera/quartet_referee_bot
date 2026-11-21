"""
Quartet referee bot
===================

Homework: a Python script that calls an LLM API, lets the LLM use a tool
(a calculation function), and sends the tool result back to the LLM.

The bot is a referee for an EV quartet card game (32 cards, sets A-H).
You ask it something like:

    "Compare Tesla Model 3 and Kia EV6 on range, who wins?"

and this happens:

    1. The script sends your question + a list of available tools to the LLM.
    2. The LLM answers with a "tool call", e.g. compare_cards("Tesla Model 3", "Kia EV6", "range").
    3. The script runs that Python function on the card data (data/cars.json).
    4. The script sends the function result back to the LLM.
    5. The LLM writes the final answer for you (steps 2-4 can repeat).

The LLM runs locally in Unsloth Studio, which offers an OpenAI-compatible
API, so we can use the normal `openai` Python package.

Run:  python referee-bot.py "Compare Tesla Model 3 and Kia EV6 on range, who wins?"
(see README.md for setup)
"""

import argparse
import json
import os
import sys
import unicodedata
from typing import Any
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Read settings (API key, base URL, model) from the .env file next to this script.
load_dotenv()

DEFAULT_MODEL = "unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL"
DEFAULT_BASE_URL = "http://127.0.0.1:8888/v1"
CARS_FILE = Path(__file__).parent / "data" / "cars.json"

# The playable stats on a card. For each one: a readable label, the unit,
# and whether the HIGHER or the LOWER value wins ("high card wins",
# but a faster 0-100 km/h time is a smaller number).
STATS = {
    "range":   {"label": "Range",          "unit": "km",        "higher_wins": True},
    "acc":     {"label": "0-100 km/h",     "unit": "s",         "higher_wins": False},
    "speed":   {"label": "Top speed",      "unit": "km/h",      "higher_wins": True},
    "battery": {"label": "Battery",        "unit": "kWh",       "higher_wins": True},
    # Not printed on the card, calculated from battery and range.
    "efficiency": {"label": "Consumption", "unit": "kWh/100 km", "higher_wins": False},
}


# ---------------------------------------------------------------------------
# Card data
# ---------------------------------------------------------------------------

def load_cars(path: Path = CARS_FILE) -> list[dict]:
    """Load all 32 cards from the JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


CARS = load_cars()


def normalize(text: str) -> str:
    """Lowercase and remove accents, so 'skoda enyaq' matches 'Škoda Enyaq'."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.lower().split())


def find_car(name: str) -> dict | None:
    """Find a card by its id ('A1') or by (part of) its name ('model 3')."""
    wanted = normalize(name)
    # 1) exact card id, e.g. "B3"
    for car in CARS:
        if car["car_id"].lower() == wanted:
            return car
    # 2) exact name, e.g. "Kia EV6"
    for car in CARS:
        if normalize(car["car_name"]) == wanted:
            return car
    # 3) the name contains the text, e.g. "ev6" or "model 3"
    for car in CARS:
        if wanted in normalize(car["car_name"]):
            return car
    # 4) the text contains the name, e.g. "the tesla model 3 card"
    for car in CARS:
        if normalize(car["car_name"]) in wanted:
            return car
    return None


def stat_value(car: dict, stat: str) -> float:
    """Value of one stat for a card (efficiency is calculated)."""
    if stat == "efficiency":
        return kwh_per_100km(car["battery"], car["range"])["kwh_per_100km"]
    return car[stat]


# ---------------------------------------------------------------------------
# Tools: plain Python functions the LLM is allowed to call
# ---------------------------------------------------------------------------

def get_card(name: str) -> dict:
    """Return everything printed on one card."""
    car = find_car(name)
    if car is None:
        return {"error": f"No card found for '{name}'."}
    return car


def compare_cards(car_a: str, car_b: str, stat: str) -> dict:
    """Compare two cards on one stat and say who wins (or if it's a tie)."""
    if stat not in STATS:
        return {"error": f"Unknown stat '{stat}'. Use one of: {', '.join(STATS)}."}
    a, b = find_car(car_a), find_car(car_b)
    if a is None or b is None:
        missing = car_a if a is None else car_b
        return {"error": f"No card found for '{missing}'."}

    rule = STATS[stat]
    value_a, value_b = stat_value(a, stat), stat_value(b, stat)

    if value_a == value_b:
        winner = "tie"
    elif (value_a > value_b) == rule["higher_wins"]:
        winner = a["car_name"]
    else:
        winner = b["car_name"]

    return {
        "stat": rule["label"],
        "unit": rule["unit"],
        "rule": "higher wins" if rule["higher_wins"] else "lower wins",
        a["car_name"]: value_a,
        b["car_name"]: value_b,
        "winner": winner,
        "difference": round(abs(value_a - value_b), 2),
    }


def best_in_set(set_letter: str, stat: str) -> dict:
    """Find the strongest card of one set (A-H) for one stat."""
    if stat not in STATS:
        return {"error": f"Unknown stat '{stat}'. Use one of: {', '.join(STATS)}."}
    letter = set_letter.strip().upper()[:1]
    cards = [car for car in CARS if car["car_id"].startswith(letter)]
    if not cards:
        return {"error": f"No set '{set_letter}'. Sets are A-H."}

    rule = STATS[stat]
    # Sort so that the best card comes first.
    ranked = sorted(cards, key=lambda car: stat_value(car, stat), reverse=rule["higher_wins"])
    return {
        "set": letter,
        "stat": rule["label"],
        "unit": rule["unit"],
        "rule": "higher wins" if rule["higher_wins"] else "lower wins",
        "best": ranked[0]["car_name"],
        "ranking": [{"car": car["car_name"], "value": stat_value(car, stat)} for car in ranked],
    }


def rank_in_deck(stat: str, position: int | None = None, car: str | None = None) -> dict:
    """Rank all 32 cards on one stat, from best to worst.

    - position=1 is the best card, 2 the second best, ...
      negative numbers count from the end: -1 is the worst, -2 the second worst.
    - car="Kia EV6" tells you where that card is in the ranking.
    - with neither, you get the whole ranking.
    Cards with the same value share a place (e.g. two cards in 1st place).
    """
    if stat not in STATS:
        return {"error": f"Unknown stat '{stat}'. Use one of: {', '.join(STATS)}."}

    rule = STATS[stat]
    ranked = sorted(CARS, key=lambda c: stat_value(c, stat), reverse=rule["higher_wins"])

    # Give every card its place. Equal values get the same place, the next
    # different value skips ahead (1, 2, 2, 4, ...) like in sports results.
    ranking = []
    for index, c in enumerate(ranked):
        value = stat_value(c, stat)
        if index > 0 and value == ranking[-1]["value"]:
            place = ranking[-1]["place"]
        else:
            place = index + 1
        ranking.append({"place": place, "car": c["car_name"], "card": c["car_id"], "value": value})

    result = {
        "stat": rule["label"],
        "unit": rule["unit"],
        "rule": "higher wins" if rule["higher_wins"] else "lower wins",
        "cards_in_deck": len(ranking),
    }

    if car:
        found = find_car(car)
        if found is None:
            return {"error": f"No card found for '{car}'."}
        result["card"] = next(r for r in ranking if r["card"] == found["car_id"])
    if position is not None:
        if position == 0 or abs(position) > len(ranking):
            return {"error": f"Position must be 1..{len(ranking)} or -1..-{len(ranking)} (-1 = worst)."}
        index = position - 1 if position > 0 else position  # Python lists also use -1 for the last item
        result["at_position"] = ranking[index]
    if not car and position is None:
        result["ranking"] = ranking
    return result


def kwh_per_100km(battery: float, range_km: float) -> dict:
    """Calculation tool: energy use in kWh per 100 km = battery / range * 100."""
    if range_km <= 0:
        return {"error": "Range must be more than 0 km."}
    return {"kwh_per_100km": round(battery / range_km * 100, 2)}


# Map the tool names the LLM sees to the real Python functions.
TOOL_FUNCTIONS = {
    "get_card": get_card,
    "compare_cards": compare_cards,
    "best_in_set": best_in_set,
    "rank_in_deck": rank_in_deck,
    "kwh_per_100km": kwh_per_100km,
}

# Descriptions of the tools in the OpenAI "function calling" format.
# The LLM only sees these descriptions, never the Python code.
STAT_PARAM = {
    "type": "string",
    "enum": list(STATS),
    "description": "range (km), acc (0-100 km/h in s), speed (top speed km/h), "
                   "battery (kWh) or efficiency (kWh/100 km)",
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_card",
            "description": "Get all stats printed on one quartet card.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Car name like 'Kia EV6' or card id like 'B3'"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_cards",
            "description": "Compare two cards on one stat and return the winner by the game rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "car_a": {"type": "string", "description": "First car name or card id"},
                    "car_b": {"type": "string", "description": "Second car name or card id"},
                    "stat": STAT_PARAM,
                },
                "required": ["car_a", "car_b", "stat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "best_in_set",
            "description": "Find the best card in one set (A-H) for one stat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "set_letter": {"type": "string", "description": "Set letter A-H"},
                    "stat": STAT_PARAM,
                },
                "required": ["set_letter", "stat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_in_deck",
            "description": "Rank all 32 cards of the whole deck on one stat from best to worst. "
                           "Use position to get the best (1), 2nd best (2), ... or worst (-1), 2nd worst (-2). "
                           "Use car to find where a card ranks. With neither, returns the full ranking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stat": STAT_PARAM,
                    "position": {"type": "integer", "description": "1 = best, 2 = 2nd best, -1 = worst, -2 = 2nd worst"},
                    "car": {"type": "string", "description": "Car name or card id to find its place"},
                },
                "required": ["stat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kwh_per_100km",
            "description": "Calculate energy consumption in kWh/100 km from battery size and range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "battery": {"type": "number", "description": "Battery capacity in kWh"},
                    "range_km": {"type": "number", "description": "Range in km"},
                },
                "required": ["battery", "range_km"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are the referee of an electric car quartet card game. "
    "There are 32 cards in sets A-H. Each card has: range (km), acc (0-100 km/h in seconds), "
    "speed (top speed in km/h) and battery (kWh). Higher value wins, except acc, "
    "where the lower (faster) time wins. Consumption (efficiency, kWh/100 km) also wins when lower. "
    "Never guess card values: always use the tools to look them up or calculate. "
    "Answer briefly and name the winner."
)


def run_tool(name: str, arguments: str) -> str:
    """Run the tool the LLM asked for and return the result as a JSON string."""
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return json.dumps({"error": f"Unknown tool '{name}'."})
    try:
        args = json.loads(arguments or "{}")
        result = function(**args)
    except (TypeError, ValueError) as error:  # bad JSON or wrong arguments from the LLM
        result = {"error": f"Bad arguments for {name}: {error}"}
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# The LLM conversation loop
# ---------------------------------------------------------------------------

def ask_referee(client: OpenAI, model: str, question: str, max_rounds: int = 5) -> str:
    """Ask the LLM, run any tools it calls, send results back, return the final answer."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(max_rounds):
        # 1) Call the LLM API with the conversation so far and the tool list.
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        reply = response.choices[0].message

        # 2) No tool call means the LLM has written its final answer.
        if not reply.tool_calls:
            return reply.content or ""

        # Keep the LLM's tool request in the history, so it knows what it asked for.
        messages.append({
            "role": "assistant",
            "content": reply.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in reply.tool_calls
            ],
        })

        # 3) Run every tool the LLM asked for and 4) send each result back.
        for call in reply.tool_calls:
            result = run_tool(call.function.name, call.function.arguments)
            print(f"  [tool] {call.function.name}({call.function.arguments}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        # The loop continues: the LLM now sees the results and answers (or calls more tools).

    return "The referee gave up: too many tool calls without a final answer."


def print_all_cars() -> None:
    """Print every card in the deck as a table."""
    print(f"{'Card':<5}{'Car':<30}{'Range':>8}{'0-100':>8}{'Speed':>8}{'Battery':>10}")
    print(f"{'':<5}{'':<30}{'km':>8}{'s':>8}{'km/h':>8}{'kWh':>10}")
    for car in CARS:
        print(f"{car['car_id']:<5}{car['car_name']:<30}{car['range']:>8}"
              f"{car['acc']:>8}{car['speed']:>8}{car['battery']:>10}")
    print(f"\n{len(CARS)} cards")


def main() -> int:
    parser = argparse.ArgumentParser(description="EV quartet referee bot (LLM + tools).")
    parser.add_argument("question", nargs="*", help="Your question. Leave empty for interactive mode.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
                        help=f"OpenAI-compatible API URL (default {DEFAULT_BASE_URL})")
    parser.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL),
                        help=f"Model name (default {DEFAULT_MODEL})")
    parser.add_argument("--list-models", action="store_true", help="Print the models the server offers and exit.")
    parser.add_argument("--all-cars", action="store_true", help="Print all available cars and exit.")
    args = parser.parse_args()

    # --all-cars only reads the card file, so it works without the LLM server.
    if args.all_cars:
        print_all_cars()
        return 0

    # Unsloth Studio needs an API key (Settings > API > Create, starts with sk-unsloth-).
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UNSLOTH_STUDIO_AUTH_TOKEN")
    if not api_key:
        print("Missing API key: put OPENAI_API_KEY=sk-unsloth-... into the .env file.")
        return 1
    client = OpenAI(base_url=args.base_url, api_key=api_key)

    if args.list_models:
        for model in client.models.list():
            print(model.id)
        return 0

    # One question from the command line ...
    if args.question:
        print(ask_referee(client, args.model, " ".join(args.question)))
        return 0

    # ... or a small chat loop.
    print("Quartet referee. Ask about the cards (empty line to quit).")
    while True:
        question = input("\nYou: ").strip()
        if not question:
            return 0
        print("Referee:", ask_referee(client, args.model, question))


if __name__ == "__main__":
    sys.exit(main())
