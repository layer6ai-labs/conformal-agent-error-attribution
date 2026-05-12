import re
from typing import List, Dict, Any, Optional


def _extract_probability(text: str) -> float:
    # Try to parse a numeric probability like 0.73, 73%, etc.
    # Fallback: map keywords to probabilities.
    try:
        # percentage
        m = re.search(r"(\d{1,3})\s*%", text)
        if m:
            val = float(m.group(1)) / 100.0
            return max(0.0, min(1.0, val))
        # decimal
        m = re.search(r"\b(0?\.\d+|1(?:\.0+)?)\b", text)
        if m:
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
    except Exception:
        pass
    
    # Default fallback if no probability found
    return 0.5


def _normalize_history(history: List[Any]):
    norm = []
    for i, entry in enumerate(history):
        if isinstance(entry, str):
            if ": " in entry:
                parts = entry.split(": ", 1)
                norm.append({"name": parts[0].strip(), "content": parts[1].strip()})
            else:
                norm.append({"name": f"Agent_{i}", "content": entry})
        elif isinstance(entry, dict):
            name = entry.get("name") or entry.get("role") or f"Agent_{i}"
            content = entry.get("content", "")
            norm.append({"name": name, "content": content})
    return norm