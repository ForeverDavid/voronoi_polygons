import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


def finite_voronoi(vor, scale=1):
    """
    finite_voronoi(vor, scale=1)

    Construct finite boundaries for Voronoi diagram.

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, 2)
        Coordinates of points to construct a convex hull from
    scale : float, optional
        Scale of extension lines. Some points may be outside their associated
        region if scale < 1

    Attributes
    ----------
    points : ndarray of double, shape (npoints, 2)
        Coordinates of input points.
    vertices : ndarray of double, shape (nvertices, 2)
        Coordinates of the Voronoi vertices.
    ridge_points : ndarray of ints, shape (nridges, 2)
        Indices of the points between which each Voronoi ridge lies.
    ridge_vertices : list of list of ints, shape (nridges, *)
        Indices of the Voronoi vertices forming each Voronoi ridge.
    regions : list of list of ints, shape (nregions, *)
        Indices of the Voronoi vertices forming each Voronoi region.
    point_region : list of ints, shape (npoints)
        Index of the Voronoi region for each input point.
    """

    vor = Voronoi(points)

    # Calculate size of extension lines
    ptp_bound = vor.points.ptp(axis=0).max() * scale

    # Count number of new vertices required
    nv_existing = len(vor.vertices)
    nv_required = len([v for v in vor.ridge_vertices if v[0] < 0])
    new_vertices = list(range(nv_existing, nv_existing + nv_required))
    outer_vertices = []

    center = vor.points.mean(axis=0)
    for idx, pointidx, simplex in zip(
            range(len(vor.ridge_points)), vor.ridge_points,
            vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):

            # Find finite end Voronoi vertex
            i = simplex[simplex >= 0][0]

            # Calculate tangent and normal
            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            # Draw line
            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * ptp_bound
            outer_vertices.append(list(pointidx))

            # Add start point to ridge vertices
            vor.ridge_vertices[idx][0] = len(vor.vertices)
            vor.vertices = np.vstack([vor.vertices, far_point])

    # Assign points to outer vertices
    outer_points = list(set(
        [item for sublist in outer_vertices for item in sublist]))

    for op in outer_points:
        p = [nv for nv, ov in zip(new_vertices, outer_vertices) if op in ov]
        vor.ridge_vertices = np.vstack([vor.ridge_vertices, p])
        vor.ridge_points = np.vstack([vor.ridge_points, [op, -1]])

    # Cycle through points
    for point_idx in range(len(vor.points)):
        region_idx = vor.point_region[point_idx]
        segs = vor.ridge_vertices[(vor.ridge_points == point_idx).any(axis=1)]

        # Remove first point from segment, to force correct direction
        segs[0, 0] = segs[0, 1]

        # Put segments in correct order
        for i in range(1, len(segs)):
            idx = np.where([np.any(np.in1d(segs[i - 1], s)) for s in segs[i:]
                            ])[0][-1]
            segs[i:] = np.roll(segs[i:], -idx, axis=0)

        # Extract vertices from segments
        pts = segs.flatten()
        _, i = np.unique(pts, return_index=True)
        vor.regions[region_idx] = pts[np.sort(i)]

    return vor

# Generate voroni diagrams
points = np.random.random([10, 2])
vor_a = Voronoi(points)
vor_b = finite_voronoi(points, scale=1)

# Get limits of vorionoi diagram
x = vor_b.vertices[:, 0]
y = vor_b.vertices[:, 1]

vor = [vor_a, vor_b]
lab = ['Infinite voronoi diagram', 'Finite voronoi diagram']

fig, ax = plt.subplots(1, 2)
for i in range(len(vor)):
    voronoi_plot_2d(vor[i], ax=ax[i])
    ax[i].set_title(lab[i])
    ax[i].axis((x.min(), x.max(), y.min(), y.max()))
plt.show()
