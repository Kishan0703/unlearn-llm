"""Step 2: Anchor terms extraction and generic translation.

The paper uses GPT-4 for entity extraction. Here we provide:
1. A default Harry Potter anchor dictionary (from the paper's examples)
2. A frequency-based extraction method as an alternative
3. A function to translate text blocks using the anchor dictionary
"""

import re
from collections import Counter


from .constants import DEFAULT_HP_ANCHORS


def extract_anchors_by_frequency(text: str, top_k: int = 100) -> list[str]:
    """Extract potential anchor terms by finding capitalized words with high frequency.

    This is a heuristic alternative to GPT-4-based extraction.
    Returns a list of candidate terms sorted by frequency.
    """
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    counter = Counter(words)
    # Filter out common English words
    common = {
        "The", "A", "An", "In", "On", "At", "To", "For", "Of", "And", "But",
        "Or", "Not", "Is", "Was", "Are", "Were", "Been", "Have", "Has", "Had",
        "Do", "Does", "Did", "Will", "Would", "Could", "Should", "May", "Might",
        "He", "She", "It", "They", "We", "You", "His", "Her", "My", "Your",
        "Their", "Our", "This", "That", "These", "Those", "What", "Which", "Who",
        "When", "Where", "How", "Why", "If", "Then", "So", "As", "With", "From",
        "By", "About", "Into", "Through", "After", "Before", "Between", "Under",
        "Over", "Again", "Further", "Once", "Here", "There", "All", "Each",
        "Every", "Both", "Few", "More", "Most", "Other", "Some", "Such",
        "Only", "Own", "Same", "Than", "Too", "Very", "Just", "Because",
        "Now", "Also", "No", "Yes", "Mr", "Mrs", "Miss", "Professor",
        "Dad", "Mom", "Mother", "Father", "Brother", "Sister", "Uncle",
        "Aunt", "Night", "Day", "Morning", "Evening", "Monday", "Tuesday",
        "Christmas", "Halloween", "Easter", "Summer", "Winter", "Spring",
        "Autumn", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "English", "French", "Great", "Good", "Bad", "New", "Old", "Big",
        "Small", "Long", "Short", "High", "Low", "Right", "Left", "Next",
        "Last", "First", "Second", "Third", "Never", "Always", "Still",
        "Much", "Many", "Well", "Back", "Away", "Down", "Up", "Out",
        "Off", "Over", "Under", "Upon", "Nothing", "Something", "Everything",
        "Anything", "One", "Two", "Three", "Four", "Five", "Six", "Seven",
        "Eight", "Nine", "Ten", "Hundred", "Thousand", "Million",
    }
    filtered = {w: c for w, c in counter.items() if w not in common and c >= 3}
    return sorted(filtered.keys(), key=lambda w: filtered[w], reverse=True)[:top_k]


def translate_text(text: str, anchor_dict: dict[str, str]) -> str:
    """Replace anchor terms in text with their generic translations.

    Uses word-boundary matching to avoid partial replacements.
    """
    result = text
    # Sort by length (longest first) to avoid partial matches
    for anchor in sorted(anchor_dict.keys(), key=len, reverse=True):
        # Match with word boundaries, case-sensitive
        pattern = r'\b' + re.escape(anchor) + r'\b'
        result = re.sub(pattern, anchor_dict[anchor], result)
    return result


def get_anchor_dict(config) -> dict[str, str]:
    """Get the anchor dictionary, either from config or the default HP one."""
    if config.anchor_dict:
        return config.anchor_dict
    return DEFAULT_HP_ANCHORS
