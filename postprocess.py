"""
Phonetic post-processing for ASR output.

Uses double metaphone to correct sound-alike errors against a vocabulary
derived from the PROMPT config (e.g., "cloud code" → "Claude Code").

Matches full phrases, not individual words.
"""

from __future__ import annotations

import re
from functools import lru_cache
from itertools import product

from metaphone import doublemetaphone


def collapse_repeats(text: str, max_repeats: int = 3) -> str:
    """Collapse a Whisper repetition loop down to a single occurrence.

    Whisper sometimes gets stuck and emits the same word or short phrase dozens
    of times ("the cat the cat the cat ..."). A unit (one or two words) repeated
    more than ``max_repeats`` times in a row is almost certainly a hallucination
    loop, not real speech, so we keep one copy and drop the rest. Comparison is
    case-insensitive; genuine short repeats (e.g. "no no") stay untouched because
    they fall under the threshold.
    """
    if not text:
        return text

    tokens = text.split()
    for unit in (1, 2):
        out: list[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            chunk = tokens[i : i + unit]
            if len(chunk) < unit:
                out.extend(tokens[i:])
                break
            lowered = [t.lower() for t in chunk]
            reps = 1
            j = i + unit
            while j + unit <= n and [t.lower() for t in tokens[j : j + unit]] == lowered:
                reps += 1
                j += unit
            if reps > max_repeats:
                out.extend(chunk)  # collapse the loop to one occurrence
            else:
                out.extend(tokens[i:j])  # not a loop, keep verbatim
            i = j
        tokens = out

    return " ".join(tokens)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def parse_vocabulary(prompt: str) -> list[str]:
    """
    Parse PROMPT string into vocabulary terms (phrases).

    "Claude Code, WezTerm, PyTorch" → ["Claude Code", "WezTerm", "PyTorch"]
    """
    if not prompt:
        return []

    terms = []
    for term in prompt.split(","):
        term = term.strip()
        if term:
            terms.append(term)
    return terms


@lru_cache(maxsize=1024)
def get_phonetic_codes(word: str) -> tuple[str, str]:
    """Get double metaphone codes for a word (cached)."""
    return doublemetaphone(word.lower())


def phrase_to_phonetic_keys(phrase: str) -> set[str]:
    """
    Convert phrase to all possible phonetic keys.

    Uses both primary and secondary metaphone codes for each word,
    generating all combinations.
    """
    words = phrase.split()
    if not words:
        return set()

    # Get all non-empty codes for each word
    word_codes = []
    for word in words:
        codes = [c for c in get_phonetic_codes(word) if c]
        if codes:
            word_codes.append(codes)
        else:
            # Word has no phonetic code, skip it
            pass

    if not word_codes:
        return set()

    # Generate all combinations
    keys = set()
    for combo in product(*word_codes):
        keys.add("".join(combo))

    return keys


def phrases_sound_alike(phrase1: str, phrase2: str) -> bool:
    """Check if two phrases sound alike (any phonetic key matches)."""
    keys1 = phrase_to_phonetic_keys(phrase1)
    keys2 = phrase_to_phonetic_keys(phrase2)
    return bool(keys1 & keys2)


def find_phonetic_match(phrase_keys: set[str], phonetic_index: dict[str, str], max_distance: int = 1) -> str | None:
    """
    Find a matching term from the phonetic index.

    First tries exact match, then fuzzy match within max_distance.
    """
    # Exact match first
    for key in phrase_keys:
        if key in phonetic_index:
            return phonetic_index[key]

    # Fuzzy match on phonetic codes
    for phrase_key in phrase_keys:
        for index_key, term in phonetic_index.items():
            # Only fuzzy match if lengths are similar
            if abs(len(phrase_key) - len(index_key)) <= max_distance:
                if levenshtein_distance(phrase_key, index_key) <= max_distance:
                    return term

    return None


def correct_text(text: str, vocab: list[str]) -> str:
    """
    Correct phrases in text that sound like vocabulary terms.

    Matches longest phrases first. Only replaces if phrase sounds alike.
    Handles compound words by trying multiple word groupings.
    Uses fuzzy matching on phonetic codes to handle slight variations.
    """
    if not vocab or not text:
        return text

    # Build phonetic index: phonetic_key → original term
    # Include all possible keys for each term
    phonetic_index: dict[str, str] = {}
    for term in vocab:
        for key in phrase_to_phonetic_keys(term):
            if key not in phonetic_index:
                phonetic_index[key] = term

    # Sort vocab by word count descending (match longer phrases first)
    # Also consider compound words as potentially multi-word
    max_words = max(
        max(len(term.split()), 2)  # At least try 2-word matches for compounds
        for term in vocab
    )

    result = text

    # Try matching from longest to shortest phrase lengths
    for num_words in range(max_words, 0, -1):
        word_pattern = r"[a-zA-Z']+"
        if num_words == 1:
            pattern = word_pattern
        else:
            pattern = word_pattern + (r"\s+" + word_pattern) * (num_words - 1)

        def make_replacer(phonetic_idx: dict[str, str]):
            def replace_if_match(match: re.Match) -> str:
                phrase = match.group(0)
                phrase_keys = phrase_to_phonetic_keys(phrase)
                replacement = find_phonetic_match(phrase_keys, phonetic_idx)
                if replacement:
                    return replacement
                return phrase
            return replace_if_match

        result = re.sub(pattern, make_replacer(phonetic_index), result)

    return result
