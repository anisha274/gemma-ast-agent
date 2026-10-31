import pytest

from minilib import shapes, stats, text


def test_mean():
    assert stats.mean([1, 2, 3, 4]) == 2.5
    with pytest.raises(ValueError):
        stats.mean([])


def test_median_odd_even():
    assert stats.median([3, 1, 2]) == 2
    assert stats.median([4, 1, 3, 2]) == 2.5
    with pytest.raises(ValueError):
        stats.median([])


def test_variance():
    assert stats.variance([2, 4, 4, 4, 5, 5, 7, 9]) == 4


def test_clamp():
    assert stats.clamp(5, 0, 10) == 5
    assert stats.clamp(-1, 0, 10) == 0
    assert stats.clamp(11, 0, 10) == 10
    assert stats.clamp(0, 0, 10) == 0
    with pytest.raises(ValueError):
        stats.clamp(1, 5, 0)


def test_rounded_mean():
    assert stats.rounded_mean([1, 2, 2]) == 1.67


def test_slugify():
    assert text.slugify("Hello, World! 2026") == "hello-world-2026"


def test_word_count():
    assert text.word_count("  one two   three ") == 3


def test_truncate():
    assert text.truncate("short", 10) == "short"
    assert text.truncate("abcdefghij", 10) == "abcdefghij"
    assert text.truncate("abcdefghijk", 8) == "abcde..."
    assert text.truncate("abcdef", 2) == ".."


def test_palindrome():
    assert text.is_palindrome("A man, a plan, a canal: Panama")
    assert not text.is_palindrome("abc")
    assert not text.is_palindrome("")


def test_rectangle():
    r = shapes.Rectangle(0, 0, 4, 2)
    assert r.area() == 8
    assert r.perimeter() == 12
    assert r.contains(4, 2) and r.contains(0, 0)
    assert not r.contains(5, 1)
    with pytest.raises(ValueError):
        shapes.Rectangle(0, 0, -1, 1)


def test_overlaps():
    a = shapes.Rectangle(0, 0, 2, 2)
    assert a.overlaps(shapes.Rectangle(1, 1, 2, 2))
    assert a.overlaps(shapes.Rectangle(2, 0, 1, 1))
    assert not a.overlaps(shapes.Rectangle(3, 3, 1, 1))


def test_total_area():
    assert shapes.total_area([shapes.Rectangle(0, 0, 1, 1), shapes.Rectangle(0, 0, 2, 3)]) == 7
