import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

points = [
[-2, 0],
[2, 0],
[0, 3],
[0, 1.5],
[0, -2],
[-2, -1.5],
[2, -1.5],
 ]

points = np.array(points)
vor = Voronoi(points)

ax = plt.axes()

#    ax.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], 'r-')

ptp_bound = vor.points.ptp(axis=0).max()

# Count number of new vertices required
nv_existing = len(vor.vertices)
nv_required = len([v for v in vor.ridge_vertices if v[0] < 0])
new_vertices = list(range(nv_existing, nv_existing + nv_required))

outer_vertices = []

center = vor.points.mean(axis=0)
for idx, pointidx, simplex in zip(range(len(vor.ridge_points)), vor.ridge_points, vor.ridge_vertices):
    simplex = np.asarray(simplex)
    if np.all(simplex >= 0):
       ax.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], '-')
    else:
        # First point
        i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
        t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
        t /= np.linalg.norm(t)
        n = np.array([-t[1], t[0]])  # normal

        midpoint = vor.points[pointidx].mean(axis=0)
        direction = np.sign(np.dot(midpoint - center, n)) * n
        far_point = vor.vertices[i] + direction * ptp_bound

        ax.plot([vor.vertices[i,0], far_point[0]],
                [vor.vertices[i,1], far_point[1]], 'k--')
                
        ax.text(far_point[0], far_point[1], pointidx)

        outer_vertices.append(list(pointidx))
        
        # Add start point to ridge vertices
        vor.ridge_vertices[idx][0] = len(vor.vertices)
        vor.vertices = np.vstack([vor.vertices, far_point])

# Assign points to outer vertices
outer_points = list(set([item for sublist in outer_vertices for item in sublist]))

for op in outer_points:
    p = [nv for nv, ov in zip(new_vertices, outer_vertices) if op in ov]
    vor.ridge_vertices = np.vstack(
        [vor.ridge_vertices, p])
    vor.ridge_points = np.vstack(
        [vor.ridge_points, [op, -1]])


ax.plot(vor.points[:,0], vor.points[:,1], '.')

for i in range(len(vor.vertices)):
    ax.plot(vor.vertices[i, 0], vor.vertices[i,1], 'wo')
    ax.text(vor.vertices[i, 0], vor.vertices[i,1]-1, i, color='r')

for i in range(len(vor.points)):
    ax.text(vor.points[i, 0], vor.points[i,1], i)


ax = plt.axes()
voronoi_plot_2d(vor, ax=ax)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

plt.show()

# attributes to add
# vor.point_region     
# vor.points           not needed
# vor.regions          
# vor.ridge_dict       
# vor.ridge_points     done
# vor.ridge_vertices   done
# vor.vertices         done




