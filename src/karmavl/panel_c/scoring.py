"""Deterministic response scoring frozen for Panel C."""

from __future__ import annotations

import ast
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any


LABEL_RE = re.compile(r"^(?:final\s+answer|answer)\s*:\s*", re.IGNORECASE)
OPTION_RE = re.compile(r"^([A-H])\.\s+(.+?)\s*$")
LETTER_RE = re.compile(r"^(?:option\s*)?([A-H])(?:[.)])?$", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?\s*%?")


def first_answer_line(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    line = next((row.strip() for row in text.splitlines() if row.strip()), "")
    line = LABEL_RE.sub("", line, count=1).strip()
    while len(line) >= 2 and (line[0], line[-1]) in {(chr(34), chr(34)), ("'", "'"), ("“", "”")}:
        line = line[1:-1].strip()
    return line


def _plain(value: Any) -> str:
    return " ".join(first_answer_line(value).casefold().split())


def _short(value: Any) -> str:
    text = _plain(value)
    normalized: list[str] = []
    for index, char in enumerate(text):
        if char == "." and 0 < index < len(text) - 1 and text[index - 1].isdigit() and text[index + 1].isdigit():
            normalized.append(char)
        elif char == "," and 0 < index < len(text) - 1 and text[index - 1].isdigit() and text[index + 1].isdigit():
            continue
        elif unicodedata.category(char).startswith("P"):
            normalized.append(" ")
        else:
            normalized.append(char)
    return " ".join(word for word in "".join(normalized).split() if word not in {"a", "an", "the"})


def _references(reference: Any) -> list[Any]:
    if isinstance(reference, (list, tuple)):
        return list(reference)
    text = str(reference).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    return [reference]


def _options(prompt: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in unicodedata.normalize("NFKC", prompt).splitlines():
        match = OPTION_RE.match(line.strip())
        if match:
            found[match.group(1)] = match.group(2)
    return found


def _letter(value: Any, options: dict[str, str], *, reference: bool) -> str | None:
    text = first_answer_line(value)
    match = LETTER_RE.fullmatch(text)
    if match and match.group(1).upper() in options:
        return match.group(1).upper()
    normalized = _short(text)
    exact = [letter for letter, option in options.items() if _short(option) == normalized]
    if len(exact) == 1:
        return exact[0]
    if reference and re.fullmatch(r"\d+", text):
        index = int(text)
        letters = sorted(options)
        if index == 0 and letters:
            return letters[0]
        if index == len(letters) and letters:
            return letters[-1]
    return None


def _number(value: Any) -> tuple[Decimal, bool] | None:
    match = NUMBER_RE.search(first_answer_line(value))
    if not match:
        return None
    token = match.group(0).strip()
    percent = token.endswith("%")
    if percent:
        token = token[:-1].strip()
    try:
        number = Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None
    return (number, percent) if number.is_finite() else None


def score_response(response: Any, reference: Any, scorer: str, prompt: str) -> bool:
    """Apply the frozen scorer family; terminal/empty responses are incorrect."""

    if not first_answer_line(response):
        return False
    references = _references(reference)
    if scorer == "yes_no":
        match = re.match(r"[a-z]+", _plain(response))
        token = match.group(0) if match else ""
        return token in {"yes", "no"} and any(token == _plain(value) for value in references)
    if scorer == "numeric_short":
        predicted = _number(response)
        if predicted is None:
            return False
        for value in references:
            truth = _number(value)
            if truth is None or predicted[1] != truth[1]:
                continue
            predicted_number, truth_number = predicted[0], truth[0]
            if predicted[1]:
                predicted_number /= Decimal(100)
                truth_number /= Decimal(100)
            tolerance = Decimal("0.000001") * max(Decimal(1), abs(truth_number))
            if abs(predicted_number - truth_number) <= tolerance:
                return True
        return False
    if scorer == "normalized_short":
        return any(_short(response) == _short(value) for value in references)
    if scorer == "numeric_or_normalized_short":
        numeric = [value for value in references if NUMBER_RE.fullmatch(first_answer_line(value))]
        return score_response(response, numeric, "numeric_short", prompt) if numeric else score_response(
            response, references, "normalized_short", prompt
        )
    if scorer == "multiple_choice_or_normalized_short":
        options = _options(prompt)
        predicted = _letter(response, options, reference=False) if options else None
        truths = {_letter(value, options, reference=True) for value in references} if options else set()
        if predicted is not None and predicted in truths:
            return True
        return any(_short(response) == _short(value) for value in references)
    raise ValueError(f"unknown frozen scorer: {scorer}")
