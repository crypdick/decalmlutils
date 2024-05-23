def convex_hull(points, include_lower=False):
    """
    Computes the upper convex hull of a set of 2D points.

    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    Code from:
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain

    Args:
        points: an iterable sequence of (x, y) pairs representing the points.

    Returns:
        A list of vertices of the upper convex hull in counter-clockwise order,
        starting from the vertex with the lexicographically smallest coordinates.

    Usage:
    >>> # Example: convex hull of a 10-by-10 grid
    >>> convex_hull([(i // 10, i % 10) for i in range(100)], include_lower=True)
    [(0, 0), (9, 0), (9, 9), (0, 9)]
    >>> convex_hull([(i // 10, i % 10) for i in range(100)], include_lower=False)
    [(9, 9), (0, 9), (0, 0)]
    """
    # Sort the points (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    if include_lower:
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Concatenation of the lower and upper hulls gives the convex hull.
        # Last point of each list is omitted because it is repeated at the beginning of the other list.
        return lower[:-1] + upper[:-1]

    return upper
