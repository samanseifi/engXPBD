import numpy as np
import meshio
import os
import matplotlib.pyplot as plt
from mpi4py import MPI
import csv

class MeshGenerator:
    @staticmethod
    def create_quadrilateral_mesh(Lx, Hy, nx, ny):
        x = np.linspace(0, Lx, nx + 1)
        y = np.linspace(0, Hy, ny + 1)
        xv, yv = np.meshgrid(x, y)
        coords = np.column_stack([xv.flatten(), yv.flatten()])

        elems = []
        for i in range(ny):
            for j in range(nx):
                n0 = i * (nx + 1) + j
                n1 = n0 + 1
                n2 = n0 + (nx + 1) + 1
                n3 = n0 + (nx + 1)
                elems.append([n0, n1, n2, n3])
        elems = np.array(elems, dtype=int)
        
        return coords, elems


class Material:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))


class BoundaryConditions:
    def __init__(self, coords, Lx):
        self.coords = coords
        self.Lx = Lx

    def get_fixed_dofs(self):
        eps = 1e-9
        fixed_nodes = np.where(self.coords[:, 0] < eps)[0]
        fixed_dofs = []
        for fn in fixed_nodes:
            fixed_dofs.extend([2 * fn, 2 * fn + 1])
        return fixed_dofs

    def get_right_edge_nodes(self):
        eps = 1e-9
        return np.where(np.abs(self.coords[:, 0] - self.Lx) < eps)[0]


class ExplicitSolver:
    def __init__(self, mesh, material, boundary_conditions, thickness, rho):
        self.coords, self.elems = mesh
        self.material = material
        self.boundary_conditions = boundary_conditions
        self.thickness = thickness
        self.rho = rho
        self.num_nodes = self.coords.shape[0]

        self.fixed_dofs = boundary_conditions.get_fixed_dofs()
        self.right_nodes = boundary_conditions.get_right_edge_nodes()

        self.u = np.zeros(2 * self.num_nodes)
        self.v = np.zeros(2 * self.num_nodes)
        self.f_ext = np.zeros(2 * self.num_nodes)

        self.M_lumped = self.compute_lumped_mass()
        self.M_inv = 1.0 / self.M_lumped

    def compute_lumped_mass(self):
        M_lumped = np.zeros(2 * self.num_nodes)
        for quad in self.elems:
            X = self.coords[quad]
            area = 0.5 * abs(np.cross(X[1] - X[0], X[3] - X[0])) + 0.5 * abs(np.cross(X[2] - X[1], X[3] - X[1]))
            m_elem = self.rho * area * self.thickness
            m_node = m_elem / 4.0
            for nd in quad:
                M_lumped[2 * nd] += m_node
                M_lumped[2 * nd + 1] += m_node
        return M_lumped

    def apply_gravity(self, g):
        for i in range(self.num_nodes):
            self.f_ext[2 * i + 1] = -self.M_lumped[2 * i + 1] * g

    def run_simulation(self, dt, nsteps, save_interval, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        kinetic_energies = []
        internal_energies = []
        time_values = []

        for step in range(nsteps):
            f_int = self.compute_internal_forces()
            f_net = self.f_ext - f_int
            f_net[self.fixed_dofs] = 0.0

            a = f_net * self.M_inv
            self.v += a * dt
            self.u += self.v * dt

            self.u[self.fixed_dofs] = 0.0
            self.v[self.fixed_dofs] = 0.0

            current_time = step * dt
            time_values.append(current_time)

            if step % save_interval == 0 or step == nsteps - 1:
                self.save_results(output_dir, step)

        self.plot_energy(time_values, kinetic_energies, internal_energies)

    def compute_internal_forces(self):
        f_int = np.zeros(2 * self.num_nodes)
        for quad in self.elems:
            X_ref = self.coords[quad]
            X_def = X_ref + self.u.reshape(-1, 2)[quad]
            f_elem = self.compute_element_force(X_ref, X_def)
            dof_map = [2 * node for node in quad] + [2 * node + 1 for node in quad]
            for i, global_idx in enumerate(dof_map):
                f_int[global_idx] += f_elem[i]
        return f_int

    def compute_element_force(self, X_ref, X_def):
        mu = self.material.mu
        lam = self.material.lam

        # Gaussian quadrature points and weights
        gauss_pts = np.array([[-1 / np.sqrt(3), -1 / np.sqrt(3)],
                              [1 / np.sqrt(3), -1 / np.sqrt(3)],
                              [1 / np.sqrt(3), 1 / np.sqrt(3)],
                              [-1 / np.sqrt(3), 1 / np.sqrt(3)]]);
        weights = np.ones(4)

        f_elem = np.zeros(8)

        for xi, w in zip(gauss_pts, weights):
            dN_dxi = np.array([
                [-(1 - xi[1]), -(1 - xi[0])],
                [(1 - xi[1]), -(1 + xi[0])],
                [(1 + xi[1]), (1 + xi[0])],
                [-(1 + xi[1]), (1 - xi[0])]
            ]) * 0.25

            J = dN_dxi.T @ X_ref
            invJ = np.linalg.inv(J)
            dN_dx = dN_dxi @ invJ

            F = X_def.T @ dN_dx
            J_det = np.linalg.det(F)

            if J_det <= 0:
                raise ValueError("Non-positive Jacobian determinant detected.")

            P = mu * (F - np.linalg.inv(F).T) + lam * np.log(J_det) * np.linalg.inv(F).T

            for i in range(4):
                force_contrib = (dN_dx[i, 0] * P[0, :] + dN_dx[i, 1] * P[1, :])
                f_elem[2 * i:2 * i + 2] += force_contrib * w * J_det

        return f_elem

    def save_results(self, output_dir, step):
        filename = os.path.join(output_dir, f"step_{step:04d}.vtu")
        points_3d = np.column_stack([self.coords, np.zeros(self.coords.shape[0])])
        disp_3d = np.column_stack([self.u.reshape(-1, 2), np.zeros(self.coords.shape[0])])
        cells = [("quad", self.elems)]
        point_data = {"displacement": disp_3d}

        mesh = meshio.Mesh(points=points_3d, cells=cells, point_data=point_data)
        mesh.write(filename)
        print(f"Saved results to {filename}")

    def plot_energy(self, time_values, kinetic_energies, internal_energies):
        plt.figure()
        plt.plot(time_values, kinetic_energies, label="Kinetic Energy")
        plt.plot(time_values, internal_energies, label="Internal Energy")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Simulation parameters
    Lx, Hy = 4.0, 1.0
    nx, ny = 40, 10
    E, nu = 1e8, 0.3
    rho, thickness = 1e3, 1.0

    dt = 1e-5
    nsteps = 1000
    save_interval = 100
    output_dir = "results"

    # Create instances
    mesh = MeshGenerator.create_quadrilateral_mesh(Lx, Hy, nx, ny)
    material = Material(E, nu)
    bc = BoundaryConditions(mesh[0], Lx)
    solver = ExplicitSolver(mesh, material, bc, thickness, rho)

    # Apply gravity and run simulation
    solver.apply_gravity(g=-9.81)
    solver.run_simulation(dt, nsteps, save_interval, output_dir)
