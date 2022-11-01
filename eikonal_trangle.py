import matplotlib
#matplotlib.use('TkAgg')
import colorcet as cc
import matplotlib.animation as animation
import matplotlib.pyplot as plt


import meshpy.triangle as triangle
import numpy as np
import numpy.linalg as la


def round_trip_connect(start, end):
    return [(i, i + 1) for i in range(start, end)] + [(end, start)]


points = [(1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1), (1, 0)]
facets = round_trip_connect(0, len(points) - 1)

circ_start = len(points)
points.extend(
    (3 * np.cos(angle), 3 * np.sin(angle))
    for angle in np.linspace(0, 2 * np.pi, 30, endpoint=False)
)

facets.extend(round_trip_connect(circ_start, len(points) - 1))

def needs_refinement(vertices, area):
    bary = np.sum(np.array(vertices), axis=0) / 3
    max_area = 0.01 + (la.norm(bary, np.inf) - 1) * 0.01
    return bool(area > max_area)

info = triangle.MeshInfo()
info.set_points(points)
info.set_holes([(0, 0)])
info.set_facets(facets)

mesh = triangle.build(info, refinement_func=needs_refinement)

mesh_points = np.array(mesh.points)
mesh_tris = np.array(mesh.elements)

MM = np.matrix([[1,4],[4,1]])
# plt.figure()
# plt.triplot(*mesh_points.T, triangles=mesh_tris)
# plt.show()

def find_triangle(x):
    adj_tris = []
    for item in mesh_tris:
        if x in item:
            adj_tris.append(item)
    return adj_tris

def find_adjacent(x):
    adj = [x]
    for item in find_triangle(x):
        for ele in item:
            if not ele in adj:
                adj.append(ele)
    adj.pop(0)
    return adj

def norm(V, M=np.matrix([[1,0.4],[0.4,1]])):
    V = np.array([V])
    return np.sqrt(np.dot(V,np.dot(M, V.T)))[0,0]

def inner(U, V, M=np.matrix([[1,0.4],[0.4,1]])):
    U, V = np.array([U]), np.array([V])
    return np.dot(U, np.dot(M, V.T))[0,0]

def put_in_list(inlist, Q, index):
    for item in index:
        inlist[item] = 1
        Q.append(item)

def solve_eikonal(x, u, f):
    tris = find_triangle(x)
    umin = np.Inf
    for tri in tris:
        temp = solve_local(x, tri, u, f)
        if temp < umin:
            umin = temp
    return umin

def solve_local(x, tri, u, f, rec=0):
    yz = np.array([item for item in tri if not item == x])
    ab = np.array([mesh_points[yz[0]], mesh_points[yz[1]]]) - np.array(mesh_points[x])
    ## Calculate angle
    gamma = np.arccos(inner(ab[0], ab[1]) / (norm(ab[0]) * norm(ab[1])))
    if gamma > np.pi / 2:
        for item in mesh_tris:
            if yz[0] in item and yz[1] in item and x not in item and rec<10:
                tri0 = item.copy()
                tri0[tri0 == yz[0]] = x
                tri1 = item.copy()
                tri1[tri1 == yz[1]] = x
                return np.min([solve_local(x, tri0, u, f, rec+1), solve_local(x, tri1, u, f, rec+1)])
        return solve_acute(ab, yz, u, f)
    else:
        return solve_acute(ab, yz, u, f)


def solve_acute(ab, yz, u, f):
    if u[yz[1]] == np.Inf and u[yz[1]] == np.Inf:
        Delta = 0
    else:
        Delta = (u[yz[1]] - u[yz[0]]) / norm(ab[0] - ab[1])
    alpha = inner(ab[0], ab[0] - ab[1]) / (norm(ab[0]) * norm(ab[0] - ab[1]))
    beta = inner(ab[1], ab[1] - ab[0]) / (norm(ab[1]) * norm(ab[1] - ab[0]))
    if alpha < Delta:
        return u[yz[0]] + f * norm(ab[0])
    elif - beta > Delta:
        return u[yz[1]] + f * norm(ab[1])
    else:
        return u[yz[0]] + np.cos(np.arccos(Delta) - np.arccos(alpha)) * f * norm(ab[0])


epsilon = 10 ** -4
resolution = len(mesh_points)
inlist = np.zeros(resolution)
speed = 1
u = np.ones(resolution) * np.Inf
source = 0
u[source] = 0
Q = []
put_in_list(inlist, Q, find_adjacent(source))

plt.figure(figsize=(11, 9))

def plot():
    plt.triplot(*mesh_points.T, triangles=mesh_tris, c='k',
                linewidth=0.5, zorder=1)
    mask = np.isfinite(u)
    plt.tricontourf(mesh_points[:,0], mesh_points[:,1], u, triangles=mesh_tris, levels=10)
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

i = 0
while len(Q) > 0:
    temp = Q.pop(0)
    p = u[temp]
    q = solve_eikonal(temp, u, speed)
    u[temp] = q
    if np.abs(p - q) < epsilon:
        neighbors = find_adjacent(temp)
        for item in neighbors:
            if inlist[item] == 0:
                p = u[item]
                q = solve_eikonal(item, u, speed)
                if p > q:
                    u[item] = q
                    put_in_list(inlist, Q, [item])
        inlist[temp] = 0
    else:
        put_in_list(inlist, Q, [temp])
    print(len(Q))
    i = i + 1
plot()