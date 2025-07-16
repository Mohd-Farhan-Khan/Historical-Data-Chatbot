# utils/timeline_extractor.py

import re
from typing import List, Dict

def extract_timeline(text: str, window: int = 30) -> List[Dict[str, str]]:
    """
    Extracts (year, event) pairs using regex and surrounding context.

    Parameters:
        text (str): Raw input text from the PDF.
        window (int): Number of words to include around the year.

    Returns:
        List[Dict[str, str]]: List of dicts with 'year' and 'event' keys.
    """
    pattern = r"\b(1[5-9][0-9]{2}|20[0-2][0-9])\b"  # Matches years from 1500–2029
    words = text.split()
    timeline = []
    seen_years = set()

    for i, word in enumerate(words):
        if re.fullmatch(pattern, word):
            # Skip if we've already processed this year
            if word in seen_years:
                continue
            
            seen_years.add(word)
            start = max(i - window, 0)
            end = min(i + window, len(words))
            context = " ".join(words[start:end])
            timeline.append({
                "year": word,
                "event": context.strip()
            })

    return timeline
