"""Axis-aligned geometry."""


class Rectangle:
    """Rectangle with its lower-left corner at (x, y)."""

    def __init__(self, x, y, width, height):
        if width < 0 or height < 0:
            raise ValueError("width and height must be non-negative")
        self.x, self.y = x, y
        self.width, self.height = width, height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

    def contains(self, px, py):
        """True if the point lies inside or on the border."""
        inside_x = self.x <= px <= self.x + self.width
        inside_y = self.y <= py <= self.y + self.height
        return inside_x and inside_y

    def overlaps(self, other):
        if self.x + self.width < other.x or other.x + other.width < self.x:
            return False
        if self.y + self.height < other.y or other.y + other.height < self.y:
            return False
        return True


def total_area(rects):
    return sum(r.area() for r in rects)
