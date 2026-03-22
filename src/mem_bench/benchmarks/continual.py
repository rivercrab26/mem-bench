"""Continual Learning benchmark.

Tests whether a memory system can learn from corrections and improve over time.
Self-contained with 20 built-in test sequences (no external dataset needed).

Each sequence simulates multi-round interactions where information is corrected,
preferences evolve, or facts are superseded. The benchmark evaluates whether the
memory system correctly tracks the latest state after all rounds.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from mem_bench.core.benchmark import BenchmarkSample
from mem_bench.core.types import IngestItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Raw benchmark data — 20 sequences
# ---------------------------------------------------------------------------

_SEQUENCES: list[dict[str, Any]] = [
    # ======================================================================
    # correction  (10 sequences) — Can the system incorporate explicit corrections?
    # ======================================================================
    {
        "sample_id": "cl-cor-01",
        "question_type": "correction",
        "question": "What is the user's preferred meeting time?",
        "reference_answer": "The user prefers meetings at 2 PM. They initially said 10 AM but later corrected it to 2 PM.",
        "ground_truth_doc_ids": ["cl-cor-01-r2"],
        "rounds": [
            {
                "doc_id": "cl-cor-01-r1",
                "round": 1,
                "content": (
                    "user: I prefer to schedule meetings at 10 AM, that's when I'm most productive.\n"
                    "assistant: Got it, I'll note that you prefer 10 AM meetings."
                ),
            },
            {
                "doc_id": "cl-cor-01-r2",
                "round": 2,
                "content": (
                    "user: Actually, I was wrong about the meeting time. I've realized I'm more focused in the afternoon. Let's change my preferred meeting time to 2 PM.\n"
                    "assistant: Updated! Your preferred meeting time is now 2 PM."
                ),
            },
            {
                "doc_id": "cl-cor-01-r3",
                "round": 3,
                "content": (
                    "user: Can you remind me what schedule preferences you have for me?\n"
                    "assistant: Sure, you prefer meetings at 2 PM."
                ),
            },
        ],
    },
    {
        "sample_id": "cl-cor-02",
        "question_type": "correction",
        "question": "What programming language is the user learning?",
        "reference_answer": "The user is learning Kotlin. They initially said they were learning Swift but corrected it to Kotlin.",
        "ground_truth_doc_ids": ["cl-cor-02-r2"],
        "rounds": [
            {
                "doc_id": "cl-cor-02-r1",
                "round": 1,
                "content": (
                    "user: I just started learning Swift this week. Excited to build iOS apps!\n"
                    "assistant: That's great! Swift is a powerful language for iOS development."
                ),
            },
            {
                "doc_id": "cl-cor-02-r2",
                "round": 2,
                "content": (
                    "user: Sorry, I misspoke earlier. I'm actually learning Kotlin, not Swift. I want to build Android apps.\n"
                    "assistant: No worries! Kotlin is the recommended language for Android development."
                ),
            },
            {
                "doc_id": "cl-cor-02-r3",
                "round": 3,
                "content": (
                    "user: I found a good Kotlin tutorial on YouTube.\n"
                    "assistant: Nice, Kotlin resources are plentiful these days."
                ),
            },
        ],
    },
    {
        "sample_id": "cl-cor-03",
        "question_type": "correction",
        "question": "How many people are on the user's team?",
        "reference_answer": "There are 8 people on the user's team. The user initially said 6 but corrected it to 8 after two new members joined.",
        "ground_truth_doc_ids": ["cl-cor-03-r2"],
        "rounds": [
            {
                "doc_id": "cl-cor-03-r1",
                "round": 1,
                "content": (
                    "user: My team has 6 people including me. We're a small backend team.\n"
                    "assistant: A team of 6 is a nice size for a backend team."
                ),
            },
            {
                "doc_id": "cl-cor-03-r2",
                "round": 2,
                "content": (
                    "user: Update on the team size: we just hired two new members, so now we're 8 people.\n"
                    "assistant: Great, the team is growing! 8 people is still manageable."
                ),
            },
            {
                "doc_id": "cl-cor-03-r3",
                "round": 3,
                "content": (
                    "user: We had our first full team standup today with everyone.\n"
                    "assistant: How did it go with the full team of 8?"
                ),
            },
        ],
    },
    {
        "sample_id": "cl-cor-04",
        "question_type": "correction",
        "question": "What is the name of the user's project?",
        "reference_answer": "The user's project is called 'Phoenix'. It was originally called 'Atlas' but was renamed to 'Phoenix'.",
        "ground_truth_doc_ids": ["cl-cor-04-r2"],
        "rounds": [
            {
                "doc_id": "cl-cor-04-r1",
                "round": 1,
                "content": (
                    "user: I'm working on a new project called Atlas. It's a data pipeline system.\n"
                    "assistant: Atlas sounds like an interesting project. Data pipelines are crucial."
                ),
            },
            {
                "doc_id": "cl-cor-04-r2",
                "round": 2,
                "content": (
                    "user: We renamed the project from Atlas to Phoenix. Management thought the name was better.\n"
                    "assistant: Phoenix is a nice name! The project itself is still the data pipeline system?"
                ),
            },
            {
                "doc_id": "cl-cor-04-r3",
                "round": 3,
                "content": (
                    "user: Yes, Phoenix is the same data pipeline project, just a new name.\n"
                    "assistant: Got it, I'll remember it as Phoenix."
                ),
            },
        ],
    },
    {
        "sample_id": "cl-cor-05",
        "question_type": "correction",
        "question": "What database does the user's application use?",
        "reference_answer": "The user's application uses MongoDB. They initially said PostgreSQL but corrected it after the team decided to switch to MongoDB.",
        "ground_truth_doc_ids": ["cl-cor-05-r2"],
        "rounds": [
            {
                "doc_id": "cl-cor-05-r1",
                "round": 1,
                "content": (
                    "user: Our application uses PostgreSQL as the primary database.\n"
                    "assistant: PostgreSQL is a solid choice, great for relational data."
                ),
            },
            {
                "doc_id": "cl-cor-05-r2",
                "round": 2,
                "content": (
                    "user: Actually, the team decided to switch to MongoDB because our data model is mostly document-based. We've already migrated.\n"
                    "assistant: MongoDB makes sense for document-based data. Migration can be tricky."
                ),
            },
            {
                "doc_id": "cl-cor-05-r3",
                "round": 3,
                "content": (
                    "user: The MongoDB migration went smoothly. Query performance is much better now.\n"
                    "assistant: That's great to hear!"
                ),
            },
        ],
    },
    {
        "sample_id": "cl-cor-06",
        "question_type": "correction",
        "question": "What is the user's work schedule?",
        "reference_answer": "The user works a 4-day week (Monday to Thursday). They initially said they work 9-to-5 Monday to Friday, but corrected it after switching to a compressed schedule.",
        "ground_truth_doc_ids": ["cl-cor-06-r2"],
        "rounds": [
            {
                "doc_id": "cl-cor-06-r1",
                "round": 1,
                "content": (
                    "user: I work a standard 9-to-5, Monday to Friday.\n"
                    "assistant: That's a typical schedule. Do you like it?"
                ),
            },
            {
                "doc_id": "cl-cor-06-r2",
                "round": 2,
                "content": (
                    "user: Big news — my company approved a compressed work week! I now work Monday through Thursday, 10-hour days, and have Fridays off.\n"
                    "assistant: That's amazing! A three-day weekend every week sounds wonderful."
                ),
            },
            {
                "doc_id": "cl-cor-06-r3",
                "round": 3,
                "content": (
                    "user: Loving the long weekends. I've been using Fridays for personal projects.\n"
                    "assistant: That's a great use of the extra day off."
                ),
            },
        ],
    },
    {
        "sample_id": "cl-cor-07",
        "question_type": "correction",
        "question": "What is the user's email client?",
        "reference_answer": "The user uses Fastmail. They initially said Gmail but switched to Fastmail for privacy reasons.",
        "ground_truth_doc_ids": ["cl-cor-07-r2"],
        "rounds": [
            {
                "doc_id": "cl-cor-07-r1",
                "round": 1,
                "content": (
                    "user: I use Gmail for all my personal email.\n"
                    "assistant: Gmail is widely used and has great integration with Google services."
                ),
            },
            {
                "doc_id": "cl-cor-07-r2",
                "round": 2,
                "content": (
                    "user: I migrated all my email to Fastmail. Privacy concerns made me leave Gmail. Please update your records.\n"
                    "assistant: Noted! Fastmail is a great privacy-focused alternative."
                ),
            },
            {
                "doc_id": "cl-cor-07-r3",
                "round": 3,
                "content": (
                    "user: Fastmail's custom domain support is really nice.\n"
                    "assistant: Yes, it's one of their best features."
                ),
            },
        ],
    },
    {
        "sample_id": "cl-cor-08",
        "question_type": "correction",
        "question": "What is the user's target for their savings goal?",
        "reference_answer": "The user's savings goal target is $15,000. They initially set it at $10,000 but revised it upward to $15,000.",
        "ground_truth_doc_ids": ["cl-cor-08-r2"],
        "rounds": [
            {
                "doc_id": "cl-cor-08-r1",
                "round": 1,
                "content": (
                    "user: I've set a savings goal of $10,000 for this year.\n"
                    "assistant: That's a solid goal! Are you tracking your progress?"
                ),
            },
            {
                "doc_id": "cl-cor-08-r2",
                "round": 2,
                "content": (
                    "user: I'm ahead of schedule on savings, so I'm revising my goal upward to $15,000.\n"
                    "assistant: Impressive! Increasing your target when ahead is smart financial planning."
                ),
            },
            {
                "doc_id": "cl-cor-08-r3",
                "round": 3,
                "content": (
                    "user: I'm now about 60% of the way to my savings target.\n"
                    "assistant: That means you've saved about $9,000 toward your $15,000 goal. Great progress!"
                ),
            },
        ],
    },
    {
        "sample_id": "cl-cor-09",
        "question_type": "correction",
        "question": "What version of Python does the user's project require?",
        "reference_answer": "The user's project requires Python 3.12. They initially said 3.10 but corrected it to 3.12 after updating the requirements.",
        "ground_truth_doc_ids": ["cl-cor-09-r2"],
        "rounds": [
            {
                "doc_id": "cl-cor-09-r1",
                "round": 1,
                "content": (
                    "user: Our project requires Python 3.10 minimum.\n"
                    "assistant: Python 3.10 has pattern matching, which is nice."
                ),
            },
            {
                "doc_id": "cl-cor-09-r2",
                "round": 2,
                "content": (
                    "user: We bumped the minimum Python version to 3.12. We need the new typing features.\n"
                    "assistant: Python 3.12 has great type annotation improvements."
                ),
            },
            {
                "doc_id": "cl-cor-09-r3",
                "round": 3,
                "content": (
                    "user: CI is now running on Python 3.12 and 3.13.\n"
                    "assistant: Good to keep up with the latest versions."
                ),
            },
        ],
    },
    {
        "sample_id": "cl-cor-10",
        "question_type": "correction",
        "question": "What is the deadline for the user's project?",
        "reference_answer": "The project deadline is March 31st. It was originally February 15th but was pushed back to March 31st.",
        "ground_truth_doc_ids": ["cl-cor-10-r2"],
        "rounds": [
            {
                "doc_id": "cl-cor-10-r1",
                "round": 1,
                "content": (
                    "user: The project deadline is February 15th. We need to ship by then.\n"
                    "assistant: That's coming up soon. Is the team on track?"
                ),
            },
            {
                "doc_id": "cl-cor-10-r2",
                "round": 2,
                "content": (
                    "user: The deadline has been pushed back to March 31st. Stakeholders agreed we needed more time for testing.\n"
                    "assistant: That extra time for testing will pay off in quality."
                ),
            },
            {
                "doc_id": "cl-cor-10-r3",
                "round": 3,
                "content": (
                    "user: Testing is going well, we should be ready well before the new deadline.\n"
                    "assistant: Great, finishing early is always nice."
                ),
            },
        ],
    },
    # ======================================================================
    # preference_evolution  (5 sequences) — User preferences change over time
    # ======================================================================
    {
        "sample_id": "cl-pev-01",
        "question_type": "preference_evolution",
        "question": "What kind of coffee does the user prefer now?",
        "reference_answer": "The user now prefers oat milk lattes. They originally liked black coffee, then switched to regular lattes, and most recently prefer oat milk lattes.",
        "ground_truth_doc_ids": ["cl-pev-01-r3"],
        "rounds": [
            {
                "doc_id": "cl-pev-01-r1",
                "round": 1,
                "content": (
                    "user: I'm a black coffee person. No sugar, no cream. Just pure coffee.\n"
                    "assistant: A purist! Black coffee is a great way to taste the beans."
                ),
            },
            {
                "doc_id": "cl-pev-01-r2",
                "round": 2,
                "content": (
                    "user: I've been getting into lattes lately. A good latte is so comforting.\n"
                    "assistant: That's a shift from black coffee! What changed?\n"
                    "user: I think the cold weather made me crave something creamier."
                ),
            },
            {
                "doc_id": "cl-pev-01-r3",
                "round": 3,
                "content": (
                    "user: I've discovered oat milk lattes and I'm obsessed. It's all I order now.\n"
                    "assistant: Oat milk has a great creamy texture for lattes."
                ),
            },
        ],
    },
    {
        "sample_id": "cl-pev-02",
        "question_type": "preference_evolution",
        "question": "What is the user's current exercise routine?",
        "reference_answer": "The user currently does yoga 4 times a week. They started with running, moved to weightlifting, and most recently switched to yoga.",
        "ground_truth_doc_ids": ["cl-pev-02-r3"],
        "rounds": [
            {
                "doc_id": "cl-pev-02-r1",
                "round": 1,
                "content": (
                    "user: I run 5K every morning. It's the best way to start the day.\n"
                    "assistant: Running is great cardio! Do you run every day?\n"
                    "user: About 5 times a week."
                ),
            },
            {
                "doc_id": "cl-pev-02-r2",
                "round": 2,
                "content": (
                    "user: I hurt my knee running, so I've switched to weightlifting. Going to the gym 4 times a week now.\n"
                    "assistant: Weightlifting is easier on the joints. Hope your knee recovers."
                ),
            },
            {
                "doc_id": "cl-pev-02-r3",
                "round": 3,
                "content": (
                    "user: I've gotten really into yoga. I do it 4 times a week now. It's done wonders for my flexibility and my knee.\n"
                    "assistant: Yoga is excellent for recovery and flexibility."
                ),
            },
        ],
    },
    {
        "sample_id": "cl-pev-03",
        "question_type": "preference_evolution",
        "question": "What is the user's preferred way to read books?",
        "reference_answer": "The user now prefers audiobooks. They started with physical books, then moved to Kindle, and most recently switched to audiobooks for commute listening.",
        "ground_truth_doc_ids": ["cl-pev-03-r3"],
        "rounds": [
            {
                "doc_id": "cl-pev-03-r1",
                "round": 1,
                "content": (
                    "user: Nothing beats a physical book. I love the feel of paper.\n"
                    "assistant: Physical books have a charm that's hard to replicate digitally."
                ),
            },
            {
                "doc_id": "cl-pev-03-r2",
                "round": 2,
                "content": (
                    "user: I got a Kindle and I'm reading so much more now. It's so convenient to carry.\n"
                    "assistant: Kindles are great for avid readers. How do you like the screen?\n"
                    "user: The e-ink screen is easy on the eyes."
                ),
            },
            {
                "doc_id": "cl-pev-03-r3",
                "round": 3,
                "content": (
                    "user: I've become an audiobook convert. I listen during my commute and finish 2-3 books a month now.\n"
                    "assistant: Audiobooks are amazing for making use of commute time."
                ),
            },
        ],
    },
    {
        "sample_id": "cl-pev-04",
        "question_type": "preference_evolution",
        "question": "What type of music does the user listen to now?",
        "reference_answer": "The user now listens to jazz. They started with pop, moved to electronic/EDM, and most recently got into jazz.",
        "ground_truth_doc_ids": ["cl-pev-04-r3"],
        "rounds": [
            {
                "doc_id": "cl-pev-04-r1",
                "round": 1,
                "content": (
                    "user: I mostly listen to pop music. Taylor Swift and Dua Lipa are my favorites.\n"
                    "assistant: Pop music is always fun. Do you have a favorite album?"
                ),
            },
            {
                "doc_id": "cl-pev-04-r2",
                "round": 2,
                "content": (
                    "user: I've been getting into electronic music lately. EDM and house really get me going during workouts.\n"
                    "assistant: Electronic music is perfect for high-energy activities."
                ),
            },
            {
                "doc_id": "cl-pev-04-r3",
                "round": 3,
                "content": (
                    "user: These days I'm all about jazz. Miles Davis and Coltrane. So sophisticated and relaxing.\n"
                    "assistant: Jazz is timeless. Miles Davis' Kind of Blue is a masterpiece."
                ),
            },
        ],
    },
    {
        "sample_id": "cl-pev-05",
        "question_type": "preference_evolution",
        "question": "What is the user's current preferred text editor for notes?",
        "reference_answer": "The user currently uses Obsidian for notes. They started with Apple Notes, moved to Notion, and most recently switched to Obsidian for local-first markdown.",
        "ground_truth_doc_ids": ["cl-pev-05-r3"],
        "rounds": [
            {
                "doc_id": "cl-pev-05-r1",
                "round": 1,
                "content": (
                    "user: I use Apple Notes for everything. Simple and syncs across devices.\n"
                    "assistant: Apple Notes is great for quick capture."
                ),
            },
            {
                "doc_id": "cl-pev-05-r2",
                "round": 2,
                "content": (
                    "user: Moved to Notion. The databases and templates are amazing for organizing.\n"
                    "assistant: Notion is very flexible. Using it for personal or work?\n"
                    "user: Both actually."
                ),
            },
            {
                "doc_id": "cl-pev-05-r3",
                "round": 3,
                "content": (
                    "user: Switched to Obsidian. I love that my notes are just local markdown files. No vendor lock-in.\n"
                    "assistant: Obsidian's local-first approach and plugin ecosystem are fantastic."
                ),
            },
        ],
    },
    # ======================================================================
    # fact_supersession  (5 sequences) — New facts replace old ones
    # ======================================================================
    {
        "sample_id": "cl-fs-01",
        "question_type": "fact_supersession",
        "question": "Where does the user live?",
        "reference_answer": "The user lives in Austin, Texas. They previously lived in Seattle, then moved to Denver, and most recently relocated to Austin.",
        "ground_truth_doc_ids": ["cl-fs-01-r3"],
        "rounds": [
            {
                "doc_id": "cl-fs-01-r1",
                "round": 1,
                "content": (
                    "user: I live in Seattle. Love the coffee culture here but the rain gets old.\n"
                    "assistant: Seattle's coffee scene is legendary! The rain is a known trade-off."
                ),
            },
            {
                "doc_id": "cl-fs-01-r2",
                "round": 2,
                "content": (
                    "user: I moved to Denver last month! Enjoying the sunshine and mountain views.\n"
                    "assistant: Denver is beautiful. The outdoor activities there are incredible."
                ),
            },
            {
                "doc_id": "cl-fs-01-r3",
                "round": 3,
                "content": (
                    "user: Another move — I'm now in Austin, Texas. Got a great job offer I couldn't refuse.\n"
                    "assistant: Austin's tech scene is booming! Welcome to Texas."
                ),
            },
        ],
    },
    {
        "sample_id": "cl-fs-02",
        "question_type": "fact_supersession",
        "question": "What car does the user drive?",
        "reference_answer": "The user drives a Tesla Model 3. They previously had a Honda Civic, then a Toyota RAV4, and most recently got a Tesla Model 3.",
        "ground_truth_doc_ids": ["cl-fs-02-r3"],
        "rounds": [
            {
                "doc_id": "cl-fs-02-r1",
                "round": 1,
                "content": (
                    "user: My Honda Civic has been super reliable. 150K miles and still running great.\n"
                    "assistant: Honda Civics are known for their reliability."
                ),
            },
            {
                "doc_id": "cl-fs-02-r2",
                "round": 2,
                "content": (
                    "user: Traded in the Civic for a Toyota RAV4. Needed more space for camping gear.\n"
                    "assistant: The RAV4 is great for outdoor adventures."
                ),
            },
            {
                "doc_id": "cl-fs-02-r3",
                "round": 3,
                "content": (
                    "user: Just picked up a Tesla Model 3! Going electric. The RAV4 went to my sister.\n"
                    "assistant: Congrats on going electric! How's the driving experience?"
                ),
            },
        ],
    },
    {
        "sample_id": "cl-fs-03",
        "question_type": "fact_supersession",
        "question": "Who is the user's manager at work?",
        "reference_answer": "The user's current manager is David Chen. They previously reported to Sarah, then briefly to Mike, and now to David Chen after a reorg.",
        "ground_truth_doc_ids": ["cl-fs-03-r3"],
        "rounds": [
            {
                "doc_id": "cl-fs-03-r1",
                "round": 1,
                "content": (
                    "user: My manager Sarah is really supportive. She lets me work independently.\n"
                    "assistant: Having a supportive manager makes a huge difference."
                ),
            },
            {
                "doc_id": "cl-fs-03-r2",
                "round": 2,
                "content": (
                    "user: Sarah left the company. I'm now reporting to Mike temporarily.\n"
                    "assistant: Transitions can be challenging. How's working with Mike?"
                ),
            },
            {
                "doc_id": "cl-fs-03-r3",
                "round": 3,
                "content": (
                    "user: Org restructure happened. My new permanent manager is David Chen. He seems great.\n"
                    "assistant: Hope the reorg brings positive changes. David sounds promising."
                ),
            },
        ],
    },
    {
        "sample_id": "cl-fs-04",
        "question_type": "fact_supersession",
        "question": "What operating system does the user use on their personal laptop?",
        "reference_answer": "The user uses Linux (Fedora) on their personal laptop. They used Windows, then macOS, and most recently switched to Linux.",
        "ground_truth_doc_ids": ["cl-fs-04-r3"],
        "rounds": [
            {
                "doc_id": "cl-fs-04-r1",
                "round": 1,
                "content": (
                    "user: I use Windows 11 on my personal laptop. It works fine for gaming and development.\n"
                    "assistant: Windows 11 has improved a lot for developers with WSL2."
                ),
            },
            {
                "doc_id": "cl-fs-04-r2",
                "round": 2,
                "content": (
                    "user: Got a MacBook Air for personal use. macOS is so smooth.\n"
                    "assistant: The M-series MacBooks are fantastic. The battery life is incredible."
                ),
            },
            {
                "doc_id": "cl-fs-04-r3",
                "round": 3,
                "content": (
                    "user: Installed Fedora Linux on my laptop. I want full control over my system and love the customizability.\n"
                    "assistant: Fedora is a great choice — cutting edge but stable."
                ),
            },
        ],
    },
    {
        "sample_id": "cl-fs-05",
        "question_type": "fact_supersession",
        "question": "What is the user's primary programming project?",
        "reference_answer": "The user's primary project is an open-source CLI tool for database migrations. They previously worked on a personal blog, then a weather app, and most recently the CLI tool.",
        "ground_truth_doc_ids": ["cl-fs-05-r3"],
        "rounds": [
            {
                "doc_id": "cl-fs-05-r1",
                "round": 1,
                "content": (
                    "user: I'm building a personal blog with Next.js. It's my main side project.\n"
                    "assistant: Next.js is great for blogs. Are you using MDX for content?"
                ),
            },
            {
                "doc_id": "cl-fs-05-r2",
                "round": 2,
                "content": (
                    "user: Finished the blog, moved on to a new project — a weather app using Rust and a public API.\n"
                    "assistant: Rust for a weather app sounds like a fun learning exercise."
                ),
            },
            {
                "doc_id": "cl-fs-05-r3",
                "round": 3,
                "content": (
                    "user: Weather app is done. My new main project is an open-source CLI tool for database migrations. Already got 50 stars on GitHub!\n"
                    "assistant: That's awesome! Database migration tools are always in demand."
                ),
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class ContinualLearningBenchmark:
    """Benchmark for testing continual learning and correction tracking.

    20 test sequences across 3 categories:
    - correction (10): explicit user corrections
    - preference_evolution (5): preferences that change over time
    - fact_supersession (5): new facts replacing old ones

    Each ``BenchmarkSample`` represents one complete multi-round sequence.
    ``ingest_items`` contain all rounds (tagged with round number in metadata),
    ``question`` is the final query, and ``reference_answer`` is the expected
    answer after all rounds.

    Implements the ``Benchmark`` protocol defined in
    ``mem_bench.core.benchmark``.
    """

    def __init__(self) -> None:
        self._samples: list[BenchmarkSample] = []

    # -- Protocol properties --------------------------------------------------

    @property
    def name(self) -> str:
        return "continual"

    @property
    def version(self) -> str:
        return "1.0"

    # -- Protocol methods -----------------------------------------------------

    def load(self, *, split: str = "test", limit: int | None = None) -> None:
        """Load the built-in dataset.

        Args:
            split: Only ``"test"`` is supported.
            limit: Maximum number of samples. ``None`` or ``0`` means all.
        """
        if split != "test":
            raise ValueError(
                f"Unknown split {split!r}. ContinualLearningBenchmark only has 'test'."
            )

        samples: list[BenchmarkSample] = []
        for seq in _SEQUENCES:
            ingest_items: list[IngestItem] = []
            for rnd in seq["rounds"]:
                ingest_items.append(
                    IngestItem(
                        content=rnd["content"],
                        document_id=rnd["doc_id"],
                        metadata={
                            "round": rnd["round"],
                            "sequence_id": seq["sample_id"],
                        },
                    )
                )

            samples.append(
                BenchmarkSample(
                    sample_id=seq["sample_id"],
                    question=seq["question"],
                    reference_answer=seq["reference_answer"],
                    question_type=seq["question_type"],
                    ingest_items=ingest_items,
                    ground_truth_doc_ids=seq["ground_truth_doc_ids"],
                    metadata={"total_rounds": len(seq["rounds"])},
                )
            )

        if limit and limit > 0:
            samples = samples[:limit]

        self._samples = samples
        logger.info("Loaded %d continual learning samples", len(self._samples))

    def __iter__(self) -> Iterator[BenchmarkSample]:
        for sample in self._samples:
            yield sample

    def __len__(self) -> int:
        return len(self._samples)
