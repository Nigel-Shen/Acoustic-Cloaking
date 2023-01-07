import matplotlib
#matplotlib.use('TkAgg')
#import jax.numpy as jnp
#from jax import grad, jit, vmap
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

def norm(V, M=np.array([[1,0],[0,1]])):
    V = np.array([V])
    return np.sqrt(np.dot(V,np.dot(M, V.T)))[0,0]

def inner(U, V, M=np.array([[1,0],[0,1]])):
    U, V = np.array([U]), np.array([V])
    return np.dot(U, np.dot(M, V.T))[0,0]

#vm = np.tensordot(np.ones(len(self.mesh_points)), np.eye(2), axes=0)
class Eikonal_solver:

    def __init__(self, points, elements, subsolver=None, source=0):
        self.mesh_points = points # np.array(mesh.points)
        self.mesh_tris = elements # np.array(mesh.elements)
        self.tri_min = np.zeros([len(self.mesh_points),3])
        self.epsilon = 10 ** -4
        self.vm = np.ones([len(self.mesh_points), 3])
        self.vm[:, 2] = 0 # Symmetric matrices have three variables
        self.resolution = len(self.mesh_points)
        self.inlist = np.zeros(self.resolution)
        self.u = np.ones(self.resolution) * np.Inf
        self.solve_order = []
        self.source = source
        self.u[self.source] = 0
        self.subsolver = subsolver
        self.Q = []

    def plot(self):
        plt.triplot(*self.mesh_points.T, triangles=self.mesh_tris, c='k',
                    linewidth=0.5, zorder=1)
        mask = np.isfinite(self.u)
        plt.tricontourf(self.mesh_points[:,0], self.mesh_points[:,1], self.u, triangles=self.mesh_tris, levels=10)
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()

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

    def solve_eikonal(self, x, u, vmx):
        tris = self.find_triangle(x, True)
        tri_min = None
        umin = np.Inf
        for i in range(len(tris)):
            tri = self.mesh_tris[tris[i]]
            temp, tri = self.solve_local(x, tri, u, vmx)
            if temp < umin:
                umin = temp
                tri_min = tri
        return umin, tri_min

    def solve_local(self, x, tri, u, vmx, rec=0):
        yz = np.array([item for item in tri if not item == x])
        ab = np.array([self.mesh_points[yz[0]], self.mesh_points[yz[1]]]) - np.array(self.mesh_points[x])
        ## Calculate angle
        mm=np.matrix([[vmx[0],vmx[2]],[vmx[2], vmx[1]]])
        gamma = np.arccos(inner(ab[0], ab[1], M=mm) / (norm(ab[0], M=mm) * norm(ab[1], M=mm)))
        if gamma > np.pi / 2:
            for item in self.mesh_tris:
                if yz[0] in item and yz[1] in item and x not in item and rec<10:
                    tri0 = item.copy()
                    tri0[tri0 == yz[0]] = x
                    tri1 = item.copy()
                    tri1[tri1 == yz[1]] = x
                    value0, tri0 = self.solve_local(x, tri0, u, vmx, rec+1)
                    value1, tri1 = self.solve_local(x, tri1, u, vmx, rec+1)
                    if value0 <= value1:
                        return value0, tri0
                    else:
                        return value1, tri1
            return self.solve_acute(ab, yz, u, mm), tri
        else:
            return self.solve_acute(ab, yz, u, mm), tri


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

    def initialize(self):
        self.u = np.ones(self.resolution) * np.Inf
        self.u[self.source] = 0
        self.put_in_list(self.find_adjacent(self.source))

    def find_u(self, vm):
        self.initialize()
        i = 0
        while len(self.Q) > 0:
            temp = self.Q.pop(0)
            p = self.u[temp]
            q, tm = self.solve_eikonal(temp, self.u, vm[temp,:])
            self.u[temp] = q
            self.tri_min[temp,:] = tm
            self.solve_order.append(temp)
            if np.abs(p - q) < self.epsilon:
                neighbors = self.find_adjacent(temp)
                for item in neighbors:
                    if self.inlist[item] == 0:
                        p = self.u[item]
                        q, tm = self.solve_eikonal(item, self.u, vm[item,:])
                        if p > q:
                            self.u[item] = q
                            self.put_in_list([item])
                            self.tri_min[item] = tm
                self.inlist[temp] = 0
            else:
                self.put_in_list([temp])
            i = i + 1
        self.tri_min.astype(int)
    
    def jax_find_u(self, vm):
        self.initialize()
        for pt in self.solve_order:
            vmpt = vm[pt,:]
            mm=np.matrix([[vmpt[0],vmpt[2]],[vmpt[2], vmpt[1]]])
            tri = self.tri_min[pt]
            yz = np.array([item for item in tri if not item == pt]).astype(int)
            ab = np.array([self.mesh_points[yz[0]], self.mesh_points[yz[1]]]) - np.array(self.mesh_points[pt])
            self.u[pt] = self.solve_acute(ab, yz, self.u, mm)
        return self.u[self.subsolver]

    def fim_solver(self):
        self.find_u(self.vm)
        
    def get_path(self, x, points=[], triangles=[]):
        if x==self.source:
            return None
        tri = int(self.tri_min[x])
        print(tri)
        if self.source in tri:
            points = tri
            triangles.append(tri)
            return points, triangles
        else:
            points.append(x)
            triangles.append(tri)
            for item in tri:
                if not item in points:
                    recpoints, rectriangles = self.get_path(item, points, triangles)
                    points.extend(recpoints)
                    triangles.extend(rectriangles)
            points=list(set(points))
            triangles=list(set(triangles))
            return points, triangles
    
    # Incorrect result in autodiff, giving all zeros----------------------------
    # def find_derivatives(self, x):
    #     pts, trs = self.get_path(x) # Find domain of dependence
    #     subsolver = Eikonal_solver(self.mesh_points[pts], trs, subsolver=x) # Create a solver on the domain of dependence
    #     subsolver.vm = self.vm[pts]
    #     dm = grad(subsolver.jax_find_u)(subsolver.vm) # Take the gradient by autodiff
    #     #subsolver.plot_derivative(dm, 1)
    #     return dm
    ## Ends Here-----------------------------------

    def plot_derivative(self, dm, comp=0):
        id = np.arange(len(self.mesh_points))*3+comp
        derivative = np.array(dm)[id.astype(int)]
        plt.triplot(*self.mesh_points.T, triangles=self.mesh_tris, c='k',
                    linewidth=0.5, zorder=1)
        mask = np.isfinite(self.u)
        plt.tricontourf(self.mesh_points[:,0], self.mesh_points[:,1], derivative, triangles=self.mesh_tris, levels=10)
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def find_jacobian(self, xobs):
        j=[]
        for x in xobs:
            j.append(self.find_derivatives(x))
        return j
                


solver = Eikonal_solver(np.array(mesh.points), np.array(mesh.elements))
solver.find_u(solver.vm)
#solver.plot()
solver.jax_find_u(solver.vm)
solver.plot()
#points, triangles = solver.get_path(100)
#print(len(points), len(solver.mesh_tris))
print(np.max(solver.find_jacobian([20])))
#plot()
