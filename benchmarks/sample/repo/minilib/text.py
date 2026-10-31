"""Text utilities."""
import re


def slugify(title):
    """Lowercase, keep alphanumerics, join words with single hyphens."""
    words = re.findall(r"[a-z0-9]+", title.lower())
    return "-".join(words)


def word_count(text):
    return len([w for w in text.split() if w])


def truncate(text, limit, suffix="..."):
    """Cut text to at most `limit` characters including the suffix."""
    if len(text) <= limit:
        return text
    keep = limit - len(suffix)
    if keep < 1:
        return suffix[:limit]
    return text[:keep] + suffix


def is_palindrome(text):
    cleaned = [c for c in text.lower() if c.isalnum()]
    return cleaned == cleaned[::-1] and len(cleaned) > 0
