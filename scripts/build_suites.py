"""Build the committed evaluation suites under ``suites/``.

Sources and licences:

* ``gsm8k_200.jsonl``: 200 items sampled with a fixed seed from the test split of GSM8K
  (https://huggingface.co/datasets/openai/gsm8k, MIT licence). The script does not download the
  dataset. Pass the official ``test.jsonl`` with ``--gsm8k-jsonl``. Gold = the number after ``####``.
* ``longctx_synth.jsonl``: synthetic. Filler text from the built-in word list (``day0.corpus``),
  with 10 to 30 embedded facts per document. Generated here, so no external data is used.
* ``chat_open_100.jsonl``: 100 original prompts written for this project (see ``CHAT_OPEN_PROMPTS``).
  Gold labels are ``null``. No text was copied from any dataset or website.

The build is deterministic. The same seed and inputs produce byte-identical files. It is not needed
at run time.

Usage:
    python scripts/build_suites.py [--gsm8k-jsonl PATH] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from day0.corpus import filler_sentence

SEED = 1234
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "suites"
GSM8K_TEMPLATE = (
    "Solve the following problem. Show your reasoning briefly, then finish with "
    "'The answer is N'.\n\n{question}"
)
GSM8K_SAMPLE = 200
LONGCTX_LENGTHS = (2000, 4000, 8000, 16000, 32000)  # target tokens
LONGCTX_PER_LENGTH = 20
CHARS_PER_TOKEN = 4
PROJECT_NAMES: tuple[str, ...] = (
    "Falcon",
    "Harbor",
    "Lantern",
    "Meridian",
    "Basalt",
    "Cobalt",
    "Dune",
    "Ember",
    "Fjord",
    "Glacier",
    "Hollow",
    "Indigo",
    "Juniper",
    "Kestrel",
    "Lumen",
    "Mosaic",
    "Nimbus",
    "Obsidian",
    "Prism",
    "Quarry",
    "Relay",
    "Summit",
    "Tundra",
    "Umbra",
    "Vertex",
    "Willow",
    "Xenon",
    "Yarrow",
    "Zephyr",
    "Aspen",
    "Brook",
    "Cedar",
    "Delta",
    "Echo",
    "Forge",
    "Grove",
    "Haven",
    "Iris",
)


CHAT_OPEN_PROMPTS: tuple[tuple[str, str], ...] = (
    # Code explanation (15)
    (
        "code_explain",
        "Explain what this Python function does, step by step:\n\ndef f(xs):\n    return sorted(set(x for x in xs if x % 2 == 0))[:3]",
    ),
    (
        "code_explain",
        "What does this JavaScript snippet print and why?\n\nconst a = [3, 1, 2];\nconsole.log([...a].sort().reverse()[0]);",
    ),
    (
        "code_explain",
        "Explain the difference between `is` and `==` in Python with a short example.",
    ),
    (
        "code_explain",
        "Describe what a Python generator is and when you would prefer it over a list.",
    ),
    (
        "code_explain",
        "Walk through this SQL query and say what it returns:\n\nSELECT dept, COUNT(*) AS n FROM staff WHERE active = 1 GROUP BY dept HAVING COUNT(*) > 3;",
    ),
    ("code_explain", "Explain what a race condition is, using a bank account example."),
    (
        "code_explain",
        "What is the purpose of a context manager in Python? Give the basic shape of one.",
    ),
    ("code_explain", "Explain big-O notation to someone who only knows loops. Use two examples."),
    ("code_explain", "What does `git rebase -i` do, in plain language?"),
    ("code_explain", "Explain how a hash map handles collisions with chaining."),
    (
        "code_explain",
        "Explain what this regular expression matches: ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$",
    ),
    ("code_explain", "Describe what a closure is in JavaScript and show a tiny counter example."),
    (
        "code_explain",
        "Explain why floating-point addition is not associative, with a concrete example.",
    ),
    ("code_explain", "What is dependency injection? Explain it with a small Python class example."),
    ("code_explain", "Explain what an HTTP 304 response means and how caching uses it."),
    # Short code generation (15)
    (
        "code_generate",
        "Write a Python function that returns the n-th Fibonacci number iteratively.",
    ),
    (
        "code_generate",
        "Write a bash one-liner that counts the lines in all .txt files in the current directory.",
    ),
    (
        "code_generate",
        "Write a SQL query that returns the three most recent orders per customer from an orders(customer_id, created_at, id) table.",
    ),
    (
        "code_generate",
        "Write a JavaScript function that debounces another function by a given delay in milliseconds.",
    ),
    (
        "code_generate",
        "Write a Python snippet that reads a CSV file and prints the average of its 'price' column.",
    ),
    ("code_generate", "Write a function in Go that reverses a slice of integers in place."),
    (
        "code_generate",
        "Write a Python function that checks whether a string is a palindrome, ignoring case and spaces.",
    ),
    (
        "code_generate",
        "Write a regular expression that matches a US ZIP code, with an optional four-digit extension.",
    ),
    (
        "code_generate",
        "Write a Python class Stack with push, pop, and peek methods that raises an error on an empty pop.",
    ),
    ("code_generate", "Write a shell command that finds files larger than 100 MB under /var/log."),
    ("code_generate", "Write a Python function that merges two sorted lists into one sorted list."),
    (
        "code_generate",
        "Write a TypeScript type for a user object with an id, an optional email, and a list of roles.",
    ),
    (
        "code_generate",
        "Write a Python decorator that prints how long the decorated function took to run.",
    ),
    (
        "code_generate",
        "Write a SQL statement that creates a table products with an id, a unique sku, and a non-negative price.",
    ),
    (
        "code_generate",
        "Write a Python function that converts a Roman numeral string to an integer.",
    ),
    # Summarizing a provided paragraph (15)
    (
        "summarize",
        "Summarize the following paragraph in two sentences.\n\nThe workshop had been quiet since the spring. Mara returned with a box of old radios and a plan: restore three of them before the autumn fair. She cleaned the dust from the dials, replaced the brittle capacitors, and learned that patience mattered more than skill. By August, two radios played clear music and one only hissed, which she kept as a reminder.",
    ),
    (
        "summarize",
        "Summarize the following text in one sentence.\n\nCity planners proposed widening the river path to reduce flooding and to give cyclists a safer route. Residents raised concerns about lost parking near the market, and the council agreed to study alternatives before voting next year.",
    ),
    (
        "summarize",
        "Give a short summary of this paragraph.\n\nSmall farms often depend on a few crops. When prices fall, they struggle to pay for seed and equipment. Cooperatives let them share machines and sell together, which lowers costs and improves bargaining power, although members must agree on decisions and accept some loss of independence.",
    ),
    (
        "summarize",
        "Summarize this paragraph for a ten-year-old.\n\nThe immune system is a team of cells that find and remove germs. Some cells raise an alarm, others attack invaders, and a few remember them so the body can respond faster next time. This memory is why a vaccine can prepare the body without causing the illness.",
    ),
    (
        "summarize",
        "Summarize the main point of this paragraph.\n\nThe library extended its hours during exam weeks, and attendance rose by a third. Staff reported that students stayed longer and borrowed more reference books, but noise complaints from nearby flats increased, prompting new quiet zones on the upper floor.",
    ),
    (
        "summarize",
        "Summarize the paragraph below in three bullet points.\n\nA kitchen garden needs sunlight, regular watering, and healthy soil. Compost improves soil structure and feeds microorganisms. Rotating crops each season reduces pests and keeps nutrients balanced, so the same bed can produce for many years.",
    ),
    (
        "summarize",
        "Summarize this in one sentence.\n\nThe airline cancelled forty flights after a software failure in its scheduling system. Engineers restored service within six hours, but many passengers missed connections, and the company promised refunds and an audit of its update process.",
    ),
    (
        "summarize",
        "Summarize the following paragraph.\n\nMemory in computers is organized in a hierarchy. Registers are fastest but smallest, caches are larger and slower, and main memory is larger still. Programs run fastest when the data they need is already close to the processor, which is why access patterns matter so much.",
    ),
    (
        "summarize",
        "Give a one-sentence summary.\n\nThe choir rehearsed twice a week in a church hall with poor acoustics. The conductor moved the rehearsals to the school gym, where the sound carried well, and attendance improved because the new room was easier to reach by bus.",
    ),
    (
        "summarize",
        "Summarize this paragraph in plain language.\n\nInterest rates set the price of borrowing money. When they rise, mortgages and business loans become more expensive, households spend less, and demand cools. When they fall, borrowing becomes cheaper and spending tends to pick up.",
    ),
    (
        "summarize",
        "Summarize the following passage in two sentences.\n\nThe trail crosses three ridges and a dry riverbed. Hikers should carry at least two litres of water, because the only source is a spring halfway along and it is seasonal. Early starts help avoid the heat of the afternoon.",
    ),
    (
        "summarize",
        "Summarize this paragraph in one short sentence.\n\nThe bakery switched to a smaller menu to reduce waste. Bread sold out earlier in the day, and customers praised the focus, although a few regulars missed the seasonal pastries that were dropped.",
    ),
    (
        "summarize",
        "Summarize the text below.\n\nLanguage learners often progress faster with spaced practice than with long single sessions. Reviewing vocabulary a day after learning it, then a week later, strengthens recall. Short daily goals are easier to keep than occasional intensive study.",
    ),
    (
        "summarize",
        "Summarize the following in exactly one sentence.\n\nThe bridge was closed for repairs for eleven weeks. Commuters used ferries and extra bus routes, which added travel time but kept businesses open. The repaired deck is expected to last several decades.",
    ),
    (
        "summarize",
        "Give a brief summary of this paragraph.\n\nOcean temperatures influence weather patterns far from the coast. Warm currents can make winters milder, while cold upwellings can cool summers. Scientists track these changes with buoys, satellites, and ships that collect measurements over many years.",
    ),
    # Finance and markets concepts (15)
    ("finance", "Explain the difference between a bond's price and its yield."),
    (
        "finance",
        "What is diversification, and why does it reduce risk without necessarily reducing returns?",
    ),
    (
        "finance",
        "Explain what a stock split is and whether it changes a company's market value directly.",
    ),
    (
        "finance",
        "What does it mean when a market is described as 'liquid'? Give one example of an illiquid asset.",
    ),
    ("finance", "Explain how an index fund differs from an actively managed fund."),
    ("finance", "What is a P/E ratio, and what can it not tell you about a company?"),
    ("finance", "Explain the concept of opportunity cost with a simple personal finance example."),
    ("finance", "What is inflation, and how does it affect the real value of a fixed salary?"),
    ("finance", "Explain what a short sale is and what risk the seller takes."),
    (
        "finance",
        "What is compound interest? Show how a 5% annual rate grows over three years on 1,000.",
    ),
    ("finance", "Explain the difference between a market order and a limit order."),
    ("finance", "What does a credit spread measure, in general terms?"),
    ("finance", "Explain what it means for a currency to appreciate against another."),
    ("finance", "What is the difference between revenue and profit for a business?"),
    ("finance", "Explain what a dividend yield is and how it is calculated."),
    # Unit conversions (15)
    ("units", "Convert 72 degrees Fahrenheit to Celsius and show the formula you used."),
    ("units", "How many millilitres are in 2.5 US gallons?"),
    ("units", "Convert 10 miles to kilometres. Round to one decimal place."),
    ("units", "How many seconds are in 3 hours and 25 minutes?"),
    ("units", "Convert 5 kilograms to pounds. Use 1 kilogram = 2.2046 pounds."),
    ("units", "How many square feet are in one square metre?"),
    ("units", "Convert 250 grams of flour to ounces."),
    ("units", "A car uses 6 litres per 100 km. How many miles per US gallon is that?"),
    ("units", "Convert 3.5 hours to minutes."),
    ("units", "How many bytes are in 2 gibibytes?"),
    ("units", "Convert 1 atmosphere to kilopascals."),
    (
        "units",
        "A recipe needs 3/4 cup of milk. How many millilitres is that? Use 1 cup = 236.6 mL.",
    ),
    ("units", "Convert 40 degrees Celsius to Kelvin."),
    ("units", "How many feet are in 1.5 kilometres?"),
    ("units", "Convert 15 nautical miles to statute miles."),
    # Instruction following (25)
    (
        "instruction",
        "Answer in exactly three bullet points: why do people learn a second language?",
    ),
    ("instruction", "Write a haiku about rain on a tin roof. Do not use the letter 'z'."),
    ("instruction", "List five fruits in alphabetical order, one per line, with nothing else."),
    (
        "instruction",
        "Respond only with the word 'yes' or the word 'no': is the Pacific the largest ocean?",
    ),
    (
        "instruction",
        "Write a two-sentence product description for a reusable water bottle. Include the word 'durable'.",
    ),
    (
        "instruction",
        "Give a title for a blog post about urban beekeeping. The title must be under six words.",
    ),
    (
        "instruction",
        "Write an email of at most 60 words asking a colleague to move a meeting to Thursday.",
    ),
    ("instruction", "Explain photosynthesis in under 40 words."),
    (
        "instruction",
        "Rewrite this sentence in the passive voice: The committee approved the budget.",
    ),
    ("instruction", "Produce a JSON object with keys 'city' and 'country' for Lisbon."),
    ("instruction", "Write a limerick about a cat who loves boxes. Keep the rhyme scheme AABBA."),
    (
        "instruction",
        "Give three tips for better sleep. Number them and keep each tip to one sentence.",
    ),
    (
        "instruction",
        "Translate 'good morning, how are you?' into Spanish and then back into English.",
    ),
    ("instruction", "Answer in lowercase only: what is the capital of Canada?"),
    (
        "instruction",
        "Write a short apology message to a customer whose order arrived late. Do not mention money.",
    ),
    (
        "instruction",
        "List the planets of the solar system in order from the sun, separated by commas.",
    ),
    (
        "instruction",
        "Write a dialogue of four lines between a barista and a customer who wants decaf.",
    ),
    ("instruction", "Describe a sunset using exactly five adjectives, and no other adjectives."),
    (
        "instruction",
        "Write a headline for a news story about a new bridge opening early. Use no more than eight words.",
    ),
    ("instruction", "Summarize the rules of chess in three sentences for a beginner."),
    ("instruction", "Name four tools a carpenter uses, in a numbered list."),
    ("instruction", "Write a motivational sentence that ends with a question mark."),
    (
        "instruction",
        "Explain the difference between weather and climate in two sentences, the second starting with 'In short'.",
    ),
    ("instruction", "Give a one-word answer: what colour do you get by mixing blue and yellow?"),
    (
        "instruction",
        "Write a recipe title and a list of exactly three ingredients for a tomato soup.",
    ),
)


def _gsm8k_items(source: Path, rng: np.random.Generator) -> list[dict[str, object]]:
    """Sample GSM8K items from the official test split.

    Args:
        source: The official ``test.jsonl`` with ``question`` and ``answer`` fields.
        rng: Seeded generator.

    Returns:
        ``GSM8K_SAMPLE`` items with ids ``gsm8k-0001`` onwards.

    Raises:
        ValueError: If the file has fewer items than the sample size.
    """
    rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(rows) < GSM8K_SAMPLE:
        raise ValueError(f"{source} has {len(rows)} rows; need at least {GSM8K_SAMPLE}")
    chosen = sorted(rng.choice(len(rows), size=GSM8K_SAMPLE, replace=False).tolist())
    items: list[dict[str, object]] = []
    for number, index in enumerate(chosen, start=1):
        row = rows[index]
        gold = str(row["answer"]).split("####")[-1].strip().replace(",", "")
        items.append(
            {
                "id": f"gsm8k-{number:04d}",
                "messages": [
                    {"role": "user", "content": GSM8K_TEMPLATE.format(question=row["question"])}
                ],
                "gold": gold,
                "answer_type": "number",
                "tags": ["math"],
            }
        )
    return items


def _longctx_document(
    rng: np.random.Generator, target_chars: int, names: list[str], facts: dict[str, int]
) -> str:
    """Build one document of filler sentences with embedded facts.

    Args:
        rng: Seeded generator.
        target_chars: Minimum document length in characters.
        names: Project names that carry facts in this document.
        facts: Project name to access code.

    Returns:
        The document text.
    """
    sentences: list[str] = []
    total = 0
    while total < target_chars:
        sentence = filler_sentence(rng)
        sentences.append(sentence)
        total += len(sentence) + 1
    for name in names:
        position = int(rng.integers(0, len(sentences) + 1))
        sentences.insert(position, f"The access code for project {name} is {facts[name]}.")
    return " ".join(sentences)


def _longctx_items(rng: np.random.Generator) -> list[dict[str, object]]:
    """Generate the synthetic long-context retrieval items.

    Args:
        rng: Seeded generator.

    Returns:
        ``len(LONGCTX_LENGTHS) * LONGCTX_PER_LENGTH`` items.
    """
    items: list[dict[str, object]] = []
    counter = 0
    for target_tokens in LONGCTX_LENGTHS:
        for _ in range(LONGCTX_PER_LENGTH):
            counter += 1
            fact_count = int(rng.integers(10, 31))
            names = [
                PROJECT_NAMES[i]
                for i in rng.choice(len(PROJECT_NAMES), size=fact_count, replace=False).tolist()
            ]
            facts = {name: int(rng.integers(10000, 100000)) for name in names}
            document = _longctx_document(rng, target_tokens * CHARS_PER_TOKEN, names, facts)
            if counter % 2 == 1:
                name = names[int(rng.integers(0, len(names)))]
                question = (
                    f"What is the access code for project {name}? Reply with the number only."
                )
                gold = str(facts[name])
                kind = "single"
            else:
                first, second = (names[i] for i in rng.choice(len(names), size=2, replace=False))
                question = (
                    f"What is the sum of the access codes for project {first} and "
                    f"project {second}? Reply with the number only."
                )
                gold = str(facts[first] + facts[second])
                kind = "sum"
            items.append(
                {
                    "id": f"longctx-{counter:04d}",
                    "messages": [{"role": "user", "content": f"{document}\n\n{question}"}],
                    "gold": gold,
                    "answer_type": "number",
                    "tags": [kind, f"target-{target_tokens}"],
                }
            )
    return items


def _chat_open_items() -> list[dict[str, object]]:
    """Build the 100 open-ended prompts.

    Returns:
        Items with ``gold`` set to ``null``.
    """
    return [
        {
            "id": f"chat-open-{number:04d}",
            "messages": [{"role": "user", "content": prompt}],
            "gold": None,
            "answer_type": None,
            "tags": [category],
        }
        for number, (category, prompt) in enumerate(CHAT_OPEN_PROMPTS, start=1)
    ]


def write_jsonl(path: Path, items: list[dict[str, object]]) -> None:
    """Write items as JSONL with sorted keys and UTF-8 encoding.

    Args:
        path: Destination file.
        items: Items to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Parse arguments and write every suite that can be built here."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gsm8k-jsonl",
        type=Path,
        default=None,
        help="Official GSM8K test.jsonl. Without it, gsm8k_200.jsonl is skipped.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output: Path = args.output_dir

    rng = np.random.default_rng(SEED)
    if args.gsm8k_jsonl is not None:
        gsm8k = _gsm8k_items(args.gsm8k_jsonl, np.random.default_rng(SEED))
        write_jsonl(output / "gsm8k_200.jsonl", gsm8k)
        print(f"wrote {len(gsm8k)} items to {output / 'gsm8k_200.jsonl'}")
    else:
        print("skipped gsm8k_200.jsonl: pass --gsm8k-jsonl PATH with the official test split")

    longctx = _longctx_items(rng)
    write_jsonl(output / "longctx_synth.jsonl", longctx)
    print(f"wrote {len(longctx)} items to {output / 'longctx_synth.jsonl'}")

    chat = _chat_open_items()
    write_jsonl(output / "chat_open_100.jsonl", chat)
    print(f"wrote {len(chat)} items to {output / 'chat_open_100.jsonl'}")


if __name__ == "__main__":
    main()
