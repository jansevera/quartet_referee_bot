# Quartet Referee Bot

A referee for the EV quartet card game. The script asks a local LLM a question,
the LLM calls a tool (a Python function that looks up or calculates card stats),
the script runs the tool and sends the result back to the LLM, which then writes
the final answer.

## Tools the LLM can use

| Tool | What it does |
|---|---|
| `get_card(name)` | All stats of one card (by name like `Kia EV6` or id like `B3`) |
| `compare_cards(car_a, car_b, stat)` | Who wins on a stat (higher wins, but 0-100 time and consumption: lower wins) |
| `best_in_set(set_letter, stat)` | Best card of set A-H, with the full ranking |
| `rank_in_deck(stat, position, car)` | Ranks all 32 cards best → worst: `position` 1 = best, 2 = 2nd best, -1 = worst; `car` = where that card ranks; neither = full ranking |
| `kwh_per_100km(battery, range_km)` | Calculates consumption = battery / range × 100 |

Stats: `range` (km), `acc` (0-100 km/h, s), `speed` (km/h), `battery` (kWh), `efficiency` (kWh/100 km, calculated).
Card data is in `data/cars.json`.

## Setup

1. Install the packages:
   ```
   pip install openai python-dotenv
   ```
2. Start Unsloth Studio and load a model that supports tool calling.
3. Create an API key in Studio: avatar (bottom left) → Settings → API → Create.
4. Put it into `.env` (optional lines in brackets):
   ```
   OPENAI_API_KEY=sk-unsloth-...
   [OPENAI_BASE_URL=http://127.0.0.1:8888/v1]
   [MODEL=unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL]
   ```

## Run

```
python referee-bot.py "Compare Tesla Model 3 and Kia EV6 on range, who wins?"
python referee-bot.py "Which car in set G is the most efficient?"
python referee-bot.py "Which card has the 2nd worst range in the whole deck?"
python referee-bot.py              # interactive mode, empty line quits
python referee-bot.py --list-models  # shows model names the server offers
python referee-bot.py --all-cars     # prints all 32 cards (no LLM needed)
```

`--base-url` and `--model` override the `.env` values. Every tool call is printed
as `[tool] name(arguments) -> result`, so you can see the LLM → tool → LLM round trip.

## If the LLM never calls a tool

Some Unsloth Studio versions did not pass `tools` through to GGUF models
(issue unslothai/unsloth#4999). If the answer comes back without any `[tool]` line,
update Unsloth, or run `llama-server --jinja` with the same model and point
`--base-url` at it.
