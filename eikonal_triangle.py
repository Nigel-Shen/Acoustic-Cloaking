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
    max_area = 0.01 + (la.norm(bary, np.inf) - 1) * 0.05
    return bool(area > max_area)

info = triangle.MeshInfo()
info.set_points(points)
info.set_holes([(0, 0)])
info.set_facets(facets)

mesh = triangle.build(info, refinement_func=needs_refinement)
plt.figure(figsize=(11, 9))

# def plot():
#     plt.triplot(*mesh_points.T, triangles=mesh_tris, c='k',
#                 linewidth=0.5, zorder=1)
#     mask = np.isfinite(u)
#     plt.tricontourf(mesh_points[:,0], mesh_points[:,1], u, triangles=mesh_tris, levels=10)
#     plt.colorbar()
#     plt.xlabel('$x$')
#     plt.ylabel('$y$')
#     plt.gca().set_aspect('equal')
#     plt.tight_layout()
#     plt.show()
def norm(V, M=np.matrix([[1,0],[0,1]])):
    V = np.array([V])
    return np.sqrt(np.dot(V,np.dot(M, V.T)))[0,0]

def inner(U, V, M=np.matrix([[1,0],[0,1]])):
    U, V = np.array([U]), np.array([V])
    return np.dot(U, np.dot(M, V.T))[0,0]


class Eikonal_solver:

    def __init__(self, points, elements):
        self.mesh_points = points # np.array(mesh.points)
        self.mesh_tris = elements # np.array(mesh.elements)
        self.vm = np.tensordot(np.ones(len(self.mesh_points)), np.eye(2), axes=0)
        self.tri_min = np.zeros(len(self.mesh_points))
        self.epsilon = 10 ** -4
        self.resolution = len(self.mesh_points)
        self.inlist = np.zeros(self.resolution)
        self.u = np.ones(self.resolution) * np.Inf
        self.source = 0
        self.u[self.source] = 0
        self.Q = []

# plt.figure()
# plt.triplot(*mesh_points.T, triangles=mesh_tris)
# plt.show()

    def find_triangle(self,x, indexed=False):
        adj_tris = []
        for i in range(len(self.mesh_tris)):
            if x in self.mesh_tris[i]:
                if indexed:
                    adj_tris.append(i)
                else:
                    adj_tris.append(self.mesh_tris[i])
        return adj_tris

    def find_adjacent(self, x):
        adj = [x]
        for item in self.find_triangle(x):
            for ele in item:
                if not ele in adj:
                    adj.append(ele)
        adj.pop(0)
        return adj

    def put_in_list(self, index):
        for item in index:
            self.inlist[item] = 1
            self.Q.append(item)

    def solve_eikonal(self, x, u):
        tris = self.find_triangle(x, True)
        tri_min = None
        umin = np.Inf
        for i in range(len(tris)):
            tri = self.mesh_tris[tris[i]]
            temp = self.solve_local(x, tri, u)
            if temp < umin:
                umin = temp
                tri_min = tris[i]
        return umin, tri_min

    def solve_local(self, x, tri, u, rec=0):
        yz = np.array([item for item in tri if not item == x])
        ab = np.array([self.mesh_points[yz[0]], self.mesh_points[yz[1]]]) - np.array(self.mesh_points[x])
        ## Calculate angle
        mm=self.vm[x]
        gamma = np.arccos(inner(ab[0], ab[1], M=mm) / (norm(ab[0], M=mm) * norm(ab[1], M=mm)))
        if gamma > np.pi / 2:
            for item in self.mesh_tris:
                if yz[0] in item and yz[1] in item and x not in item and rec<10:
                    tri0 = item.copy()
                    tri0[tri0 == yz[0]] = x
                    tri1 = item.copy()
                    tri1[tri1 == yz[1]] = x
                    return np.min([self.solve_local(x, tri0, u, rec+1), self.solve_local(x, tri1, u, rec+1)])
            return self.solve_acute(ab, yz, u, mm)
        else:
            return self.solve_acute(ab, yz, u, mm)


    def solve_acute(self, ab, yz, u, mm):
        if u[yz[1]] == np.Inf and u[yz[1]] == np.Inf:
            Delta = 0
        else:
            Delta = (u[yz[1]] - u[yz[0]]) / norm(ab[0] - ab[1], M=mm)
        alpha = inner(ab[0], ab[0] - ab[1], M=mm) / (norm(ab[0], M=mm) * norm(ab[0] - ab[1], M=mm))
        beta = inner(ab[1], ab[1] - ab[0], M=mm) / (norm(ab[1], M=mm) * norm(ab[1] - ab[0], M=mm))
        if alpha < Delta:
            return u[yz[0]] + norm(ab[0], M=mm)
        elif - beta > Delta:
            return u[yz[1]] + norm(ab[1], M=mm)
        else:
            return u[yz[0]] + np.cos(np.arccos(Delta) - np.arccos(alpha)) * norm(ab[0], M=mm)


    def find_u(self):
        self.put_in_list(self.find_adjacent(self.source))
        i = 0
        while len(self.Q) > 0:
            temp = self.Q.pop(0)
            p = self.u[temp]
            q, tm = self.solve_eikonal(temp, self.u)
            self.u[temp] = q
            self.tri_min[temp] = tm
            if np.abs(p - q) < self.epsilon:
                neighbors = self.find_adjacent(temp)
                for item in neighbors:
                    if self.inlist[item] == 0:
                        p = self.u[item]
                        q, tm = self.solve_eikonal(item, self.u)
                        if p > q:
                            self.u[item] = q
                            self.put_in_list([item])
                            self.tri_min[item] = tm
                self.inlist[temp] = 0
            else:
                self.put_in_list([temp])
            print(len(self.Q))
            i = i + 1
        self.tri_min.astype(int)
        
    def get_path(self, x, points=[], triangles=[]):
        i = int(self.tri_min[x])
        tri = self.mesh_tris[i]
        print(tri)
        if self.source in tri:
            points = tri
            triangles.append(i)
            return points, triangles
        else:
            points.append(x)
            triangles.append(i)
            for item in tri:
                if not item in points:
                    recpoints, rectriangles = self.get_path(item, points, triangles)
                    points.extend(recpoints)
                    triangles.extend(rectriangles)
            points=list(set(points))
            triangles=list(set(triangles))
            return points, triangles
            


solver = Eikonal_solver(np.array(mesh.points), np.array(mesh.elements))
solver.find_u()
points, triangles = solver.get_path(20)
print(len(points), len(solver.mesh_tris))
len()
#plot()
