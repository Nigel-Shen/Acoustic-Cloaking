from inspect import EndOfBlock
import matplotlib
import time
#matplotlib.use('TkAgg')
import jax.numpy as jnp
from jax import grad, jit, vmap
import colorcet as cc
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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
    max_area = 0.01 + (la.norm(bary, np.inf) - 1) * 0.03
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

def jax_norm(V, M=jnp.array([[1,0],[0,1]])):
    V = jnp.array([V])
    return jnp.sqrt(jnp.dot(V,jnp.dot(M, V.T)))[0,0]

def jax_inner(U, V, M=jnp.array([[1,0],[0,1]])):
    U, V = jnp.array([U]), jnp.array([V])
    return jnp.dot(U, jnp.dot(M, V.T))[0,0]

def find_plane_wave_bc(dir, grid_points, boundary=np.arange(5, 35).tolist(), normalize=True):
    if normalize:
        dir = dir / np.linalg.norm(dir)
    bc = []
    for point in boundary:
        bc.append(np.inner(grid_points[point], dir))
    bc = bc - np.min(bc)
    return bc


#vm = np.tensordot(np.ones(len(self.mesh_points)), np.eye(2), axes=0)
class Eikonal_solver:
    def __init__(self, points, elements, **kwargs):
        self.mesh_points = points # np.array(mesh.points)
        self.N = points.shape[0]
        self.mesh_tris = elements # np.array(mesh.elements)
        self.tri_min = np.ones([len(self.mesh_points),3]) * np.Inf
        self.epsilon = 10 ** -4
        self.vm = np.ones([len(self.mesh_points), 3])
        self.vm[:, 2] = 0 # Symmetric matrices have three variables
        self.resolution = len(self.mesh_points)
        self.inlist = np.zeros(self.resolution)
        self.u = np.ones(self.resolution) * np.Inf
        self.solve_order = []
        self.source = None#np.arange(5, 35).tolist()
        self.init_value = None #3 - self.mesh_points[self.source, 0]
        self.subsolver = None
        self.Q = []
        self.xobs = np.arange(5, 35, dtype=int)
        self.__dict__.update(kwargs)
        self.make_compatible()

    def make_compatible(self):
        if not self.source == None:
            new_source=[]
            new_init_value=[]
            for i in range(len(self.source)):
                if self.init_value[i]<3:
                    new_source.append(self.source[i])
                    new_init_value.append(self.init_value[i])
            self.source = new_source
            self.init_value = new_init_value

    def set_condition(self, source, value):
        self.source = source
        self.init_value = value
        self.make_compatible()

    def plot(self, saveas=False):
        plt.figure(figsize=(11, 9))
        plt.triplot(*self.mesh_points.T, triangles=self.mesh_tris, c='k',
                    linewidth=0.5, zorder=1)
        mask = np.isfinite(self.u)
        #plt.scatter(*self.mesh_points[self.source], s=50, facecolors='cyan', edgecolors='black', zorder=3)
        plt.tricontourf(self.mesh_points[:,0], self.mesh_points[:,1], self.u, triangles=self.mesh_tris, levels=10)
        plt.colorbar()
        plt.xlabel('')
        plt.ylabel('')
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        if saveas==False:
            plt.show()
        else:
            plt.savefig('%s.png' % saveas, dpi=300)


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

    def jax_solve_acute(self, ab, yz, u, mm):
        if u.take(yz[1]) == np.Inf and u.take(yz[1]) == np.Inf:
            Delta = 0
        else:
            Delta = (u.take(yz[1]) - u.take(yz[0])) / jax_norm(ab[0] - ab[1], M=mm)
        alpha = jax_inner(ab[0], ab[0] - ab[1], M=mm) / (jax_norm(ab[0], M=mm) * jax_norm(ab[0] - ab[1], M=mm))
        beta = jax_inner(ab[1], ab[1] - ab[0], M=mm) / (jax_norm(ab[1], M=mm) * jax_norm(ab[1] - ab[0], M=mm))
        if alpha < Delta:
            return u.take(yz[0]) + jax_norm(ab[0], M=mm)
        elif - beta > Delta:
            return u.take(yz[1]) + jax_norm(ab[1], M=mm)
        else:
            return u.take(yz[0]) + jnp.cos(jnp.arccos(Delta) - jnp.arccos(alpha)) * jax_norm(ab[0], M=mm)

    def initialize(self):
        self.u = np.ones(self.resolution) * np.Inf
        print(self.source,self.init_value)
        self.u[self.source] = self.init_value
        for x in self.source:
            self.put_in_list(self.find_adjacent(x))

    def find_u(self, vm):
        self.initialize()
        i = 0
        while len(self.Q) > 0:
            temp = self.Q.pop(0)
            p = self.u[temp]
            q, tm = self.solve_eikonal(temp, self.u, vm[temp,:])
            if p > q:
                self.u[temp] = q
                self.tri_min[temp,:] = tm
                if not (tm in self.mesh_tris or tm is None):
                    self.mesh_tris = np.append(self.mesh_tris, [tm], axis=0)
                if np.abs(p - q) < self.epsilon:
                    neighbors = self.find_adjacent(temp)
                    for item in neighbors:
                        if self.inlist[item] == 0:
                            p = self.u[item]
                            q, tm = self.solve_eikonal(item, self.u, vm[item,:])
                            if p > q:
                                self.u[item] = q
                                #if item in self.source:
                                    #self.source.remove(item)
                                self.put_in_list([item])
                                self.tri_min[item] = tm
                    self.inlist[temp] = 0
                else:
                    self.put_in_list([temp])
            else:
                neighbors = self.find_adjacent(temp)
                for item in neighbors:
                    if self.inlist[item] == 0:
                        p = self.u[item]
                        q, tm = self.solve_eikonal(item, self.u, vm[item,:])
                        if p > q:
                            self.u[item] = q
                            #if item in self.source:
                                #self.source.remove(item)
                            self.put_in_list([item])
                            self.tri_min[item] = tm
                self.inlist[temp] = 0
        self.solve_order = np.argsort(self.u)
        self.tri_min.astype(int)

    def jax_find_u(self, vm):
        self.initialize()
        self.u = jnp.array(self.u)
        for pt in self.solve_order:
            if pt not in self.source:
                vmpt = vm[pt,:]
                mm=jnp.array([[vmpt[0],vmpt[2]],[vmpt[2], vmpt[1]]])
                tri = self.tri_min[pt]
                yz = jnp.array([item for item in tri if not item == pt]).astype(int)
                ab = jnp.array([self.mesh_points[yz[0]], self.mesh_points[yz[1]]]) - jnp.array(self.mesh_points[pt])
                val=self.jax_solve_acute(ab, yz, self.u, mm)
                self.u = self.u.at[pt].set(val)
        self.u=jnp.asarray(self.u)
        return self.u[self.subsolver]

    def fim_solver(self):
        self.find_u(self.vm)

    def get_path(self, x, points=[], triangles=[]):
        if x in self.source:
            return None
        tri = self.tri_min[x].astype(int)
        end = False
        for item in tri:
            if item in self.source:
                end = True
                break
        if end:
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
            triangles=np.unique(triangles, axis=0)
            return points, triangles

    def find_derivatives(self, x):
        pts, trs = self.get_path(x) # Find domain of dependence
        subsolver = Eikonal_solver(self.mesh_points, trs, subsolver=x) # Create a solver on the domain of dependence
        subsolver.vm = self.vm
        subsolver.tri_min=self.tri_min
        subsolver.source = solver.source[:]
        subsolver.init_value = solver.init_value[:]
        xy, x_ind, y_ind = np.intersect1d(self.solve_order, pts, return_indices=True)
        subsolver.solve_order = np.array(self.solve_order)[np.sort(x_ind)]
        dm = grad(subsolver.jax_find_u)(subsolver.vm) # Take the gradient by autodiff
        return dm.flatten().tolist()

    def plot_derivative(self, i, comp=0):
        plt.figure(figsize=(11, 9))
        Df = np.array(self.find_derivatives(i)).reshape((460, 3))

        vmax = np.abs(Df).max()
        vmin = - vmax
        levels = np.linspace(vmin, vmax, 11)

        I_nnz = np.where((Df != 0).any(1))

        ax = plt.gca()
        ax.set_aspect('equal')

        plt.tricontourf(*self.mesh_points.T, self.mesh_tris, Df[:, comp], levels=levels, zorder=1, cmap='PiYG')
        plt.colorbar()
        plt.triplot(*self.mesh_points.T, self.mesh_tris, c='black', linewidth=0.5, zorder=2)
        plt.scatter(*self.mesh_points[i], s=50, facecolors='orange', edgecolors='black', zorder=3)
        plt.scatter(*self.mesh_points[self.source], s=50, facecolors='green', edgecolors='black', zorder=3)
        plt.scatter(*self.mesh_points[I_nnz].T, s=50, alpha=0.5, facecolors='black', edgecolors='white', zorder=3)
        plt.tight_layout()
        plt.show()

    def plot_velocity(self):
        plt.figure(figsize=(9, 9))
        ax = plt.gca()
        vmax = np.abs(self.vm).max()
        vmin = - vmax
        levels = np.linspace(vmin, vmax, 11)
        for i in range(460):
            M = np.array([[self.vm[i, 0],self.vm[i, 2]],[self.vm[i, 2], self.vm[i, 1]]])
            e, v = np.linalg.eig(M)
            angle = np.angle(v[0, 0] + v[0, 1]*1j, deg=True)
            ellipse = Ellipse(xy=self.mesh_points[i], width=e[0]/10, height=e[1]/10, facecolor='green', alpha=np.min(e)/np.max(e), angle=angle)
            ax.add_patch(ellipse)
        #plt.colorbar()
        ax.set_aspect('equal')
        plt.triplot(*self.mesh_points.T, self.mesh_tris, c='black', linewidth=0.5, zorder=2)
        plt.tight_layout()
        plt.show()


    def find_jacobian(self):
        j=[]
        for x in self.xobs:
            if x in self.source:
                j.append(np.zeros([self.N * 3]).tolist())
            else:
                j.append(self.find_derivatives(x))
            print('found')
        self.subsolver = None
        return np.array(j)

    def uni_direction_design(self, dir, boundary=np.arange(5, 35).tolist(), max_iter=10):
        ## Using Gauss-Newton Method to find the optimal velocity matrices
        count = 0
        tau = find_plane_wave_bc(dir, self.mesh_points)
        self.set_condition(boundary, tau)
        while count < max_iter:
            print("iteration #: ", count)
            self.solve_order = []
            self.fim_solver()
            res = np.array([tau - self.u[self.xobs]]).T
            print(np.sum(res**2))
            if np.max(np.abs(res)) < 10**-2:
                break
            J = self.find_jacobian()
            self.vm = self.vm + np.dot(np.linalg.pinv(J), res).reshape((self.N, 3))
            count = count + 1

    def multi_direction_design(self, dirs, boundary=np.arange(5, 35).tolist(), max_iter=10):
        ## Using Gauss-Newton Method to find the optimal velocity matrices
        count = 0
        tau = []
        for dir in dirs:
            tau.append(find_plane_wave_bc(dir, self.mesh_points, boundary))
        while count < max_iter:
            res = []
            J = []
            for i in range(len(dirs)):
                self.set_condition(boundary, tau[i])
                print("iteration #: ", count)
                self.solve_order = []
                self.fim_solver()
                res.append(np.array([tau[i] - self.u[self.xobs]]).T)
                J.append(self.find_jacobian())
            res = np.vstack(res)
            J = np.vstack(J)
            self.vm = self.vm + np.dot(np.linalg.pinv(J), res).reshape((self.N, 3))
            R = np.max(np.abs(res))
            if R < 10**-2:
                break
            print(R)
            count = count + 1

dirs = [[0, -1], [1,-1], [1,0], [1,1], [0,1], [-1,1], [-1, 0], [-1, -1]]
solver = Eikonal_solver(np.array(mesh.points), np.array(mesh.elements))
solver.multi_direction_design(dirs)
#solver.find_u(solver.vm)
#print(solver.source)
#solver.plot(saveas='plane_wave')
#print(solver.source)
#solver.jax_find_u(solver.vm)
#solver.plot()
#points, triangles = solver.get_path(100)
#print(len(points), len(solver.mesh_tris))
t = time.time()
#solver.find_velocity()
#solver.plot_velocity()
#solver.plot_derivative(20)
print("Finding jacobian costs", time.time()-t)
#plot()
