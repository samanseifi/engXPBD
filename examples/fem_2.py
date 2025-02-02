import csv
import numpy as np
import meshio
import os
import time

import matplotlib.pyplot as plt
import netCDF4 as nc

from numba import cuda, njit, prange


# ===========================================================
#                  Simple Mesh Generation
# ===========================================================
def create_quadrilateral_mesh(Lx=4.0, Hy=1.0, nx=80, ny=20):
    """
    Generate a structured 2D quadrilateral mesh for the rectangle [0,Lx] x [0,Hy].
    
    Parameters:
    ----------
    Lx, Hy : float
        Dimensions of the rectangle in x and y directions.
    nx, ny : int
        Number of elements along x and y directions.

    Returns:
    -------
    coords : (N,2) ndarray
        Node coordinates.
    elems : (M,4) ndarray
        Quad element connectivity (zero-based node indices).
    """
    x = np.linspace(-Lx/2, Lx/2, nx + 1)
    y = np.linspace(-Hy/2, Hy/2, ny + 1)
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

import numpy as np
import netCDF4 as nc

def load_exo_file(filename):
    """
    Loads an Exodus (.exo) file and returns node coordinates, element connectivity, node sets, and side sets.

    Args:
        filename (str): Path to the Exodus file.

    Returns:
        tuple: (coords, elems, nodesets, sidesets)
            - coords (np.ndarray): Shape (num_nodes, 2) or (num_nodes, 3), depending on input.
            - elems (np.ndarray): Shape (num_elems, nodes_per_element), with int dtype.
            - nodesets (dict): {nodeset_name: np.ndarray of shape (num_nodes_in_set,)}
            - sidesets (dict): {sideset_name: list of (node1, node2) edges}
    """
    # Open the Exodus file using netCDF4
    dataset = nc.Dataset(filename, mode="r")

    print(dataset)
    
    # Get node coordinates
    x = dataset.variables["coordx"][:]
    y = dataset.variables["coordy"][:]
    z = dataset.variables.get("coordz", None)  # Some files might not have z-coordinates

    if z is None or np.all(z == 0):  # If z-coordinates are missing or all zero, assume 2D
        coords = np.column_stack((x, y))
    else:
        coords = np.column_stack((x, y, z))

    # Get element connectivity
    elems_list = []
    num_blocks = dataset.dimensions["num_el_blk"].size if "num_el_blk" in dataset.dimensions else 0

    for i in range(1, num_blocks + 1):
        block_var_name = f"connect{i}"  # Exodus uses names like "connect1", "connect2", etc.
        if block_var_name in dataset.variables:
            connectivity = dataset.variables[block_var_name][:] - 1  # Convert to 0-based indexing
            elems_list.append(connectivity)

    elems = np.vstack(elems_list).astype(int) if elems_list else np.array([], dtype=int)

    # Get node sets
    nodesets = {} 

    num_nodesets = sum(1 for key in dataset.variables.keys() if key.startswith("node_ns"))
    
    for i in range(1, num_nodesets + 1):
        nodeset_var_name = f"node_ns{i}"  # Exodus uses names like "node_ns1", "node_ns2", etc.
        if nodeset_var_name in dataset.variables:
            nodes = dataset.variables[nodeset_var_name][:] - 1  # Convert to 0-based indexing
            nodesets[f"nodeset_{i}"] = nodes.astype(int)

    # Extract side sets with correct edges
    sidesets = {}
    
    num_sidesets = sum(1 for key in dataset.variables.keys() if key.startswith("elem_ss"))

    for i in range(1, num_sidesets + 1):
        sideset_name = f"sideset_{i}"

        # Get element indices and side indices
        elem_ss = dataset.variables[f"elem_ss{i}"][:] - 1  # Convert to zero-based indexing
        side_ss = dataset.variables[f"side_ss{i}"][:] - 1  # Convert to zero-based indexing

        edge_list = set()

        for elem_index, face_id in zip(elem_ss, side_ss):
            elem_nodes = elems[elem_index]

            # Identify the correct edge based on face_id
            if len(elem_nodes) == 3:  # Triangular elements
                edges = [(elem_nodes[0], elem_nodes[1]),
                         (elem_nodes[1], elem_nodes[2]),
                         (elem_nodes[2], elem_nodes[0])]
            elif len(elem_nodes) == 4:  # Quadrilateral elements
                edges = [(elem_nodes[0], elem_nodes[1]),
                         (elem_nodes[1], elem_nodes[2]),
                         (elem_nodes[2], elem_nodes[3]),
                         (elem_nodes[3], elem_nodes[0])]
            else:
                raise ValueError("Unsupported element type")

            # Select the correct boundary edge
            if 0 <= face_id < len(edges):
                edge_list.add(tuple(sorted(edges[face_id])))  # Store sorted to avoid duplicate order

        sidesets[sideset_name] = sorted(edge_list)

    dataset.close()

    # Unmask the arrays
    coords = np.asarray(coords)
    elems = np.asarray(elems)     

    return coords, elems, nodesets, sidesets

# ===========================================================
#                  Output-Related Functions
# ===========================================================
def compute_element_stress_strain_tri3(X_ref_elem, X_def_elem, E, nu):
    """
    Compute full stress tensor, strain tensor, and von Mises stress for a linear triangular element at its centroid.

    Args:
        X_ref_elem (np.ndarray): Reference coordinates of the 3 triangle nodes (3,2).
        X_def_elem (np.ndarray): Deformed coordinates of the 3 triangle nodes (3,2).
        E (float): Young's modulus.
        nu (float): Poisson's ratio.

    Returns:
        stress_tensor (np.ndarray): 3x3 Cauchy stress tensor.
        strain_tensor (np.ndarray): 3x3 Green-Lagrange strain tensor.
        von_mises (float): Von Mises stress.
    """
    
    # Material constants for plane-strain
    mu = E / (2.0 * (1.0 + nu))
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    # Extract reference coordinates of the triangle
    X1, X2, X3 = X_ref_elem  # Shape: (3,2)

    # Compute the area of the triangle
    A = 0.5 * np.linalg.det(np.array([
        [1, X1[0], X1[1]],
        [1, X2[0], X2[1]],
        [1, X3[0], X3[1]]
    ]))

    if A <= 0:
        raise ValueError("Invalid element: Triangle area must be positive.")

    # Compute shape function gradients in reference coordinates (constant for linear triangles)
    dN_dx_ref = np.array([
        [X2[1] - X3[1], X3[1] - X1[1], X1[1] - X2[1]],
        [X3[0] - X2[0], X1[0] - X3[0], X2[0] - X1[0]]
    ]) / (2 * A)  # Shape: (2,3)

    # Compute deformation gradient
    F = X_def_elem.T @ dN_dx_ref.T  # (2,3) @ (3,2) = (2,2)

    # Compute Green-Lagrange strain tensor: E = 0.5 * (F^T F - I)
    C = F.T @ F  # Right Cauchy-Green deformation tensor (2,2)
    strain_matrix_2d = 0.5 * (C - np.eye(2))  # (2,2)

    # Expand strain tensor to 3x3 for plane-strain case
    strain_tensor = np.array([
        [strain_matrix_2d[0, 0], strain_matrix_2d[0, 1], 0],
        [strain_matrix_2d[1, 0], strain_matrix_2d[1, 1], 0],
        [0, 0, 0]
    ])

    # Compute 1st Piola-Kirchhoff stress using Neo-Hookean model
    J = np.linalg.det(F)  # Jacobian determinant
    if J <= 0:
        raise ValueError("Negative Jacobian detected. Element is inverted.")

    FinvT = np.linalg.inv(F).T
    P = mu * (F - FinvT) + lam * np.log(J) * FinvT  # First Piola-Kirchhoff stress

    # Convert 1st PK stress to Cauchy stress: sigma = (1/J) * P * F^T
    cauchy_stress_2d = (1.0 / J) * P @ F.T  # (2,2)

    # Expand to 3x3 stress tensor for plane strain
    stress_tensor = np.array([
        [cauchy_stress_2d[0, 0], cauchy_stress_2d[0, 1], 0],
        [cauchy_stress_2d[1, 0], cauchy_stress_2d[1, 1], 0],
        [0, 0, 0]
    ])

    # Compute von Mises stress for plane strain
    s_xx = cauchy_stress_2d[0, 0]
    s_yy = cauchy_stress_2d[1, 1]
    s_xy = cauchy_stress_2d[0, 1]
    von_mises = np.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3 * s_xy**2)

    return stress_tensor, strain_tensor, von_mises

def compute_element_stress_strain_quad(X_ref_elem, X_def_elem, E, nu):
    """
    Compute full stress tensor, strain tensor, and von Mises stress for a quadrilateral element at its centroid.
    """
    # Material constants for plane-strain
    mu = E / (2.0 * (1.0 + nu))
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    # Shape function derivatives at centroid (xi=0, eta=0)
    xi = 0.0
    eta = 0.0
    dN_dxi = np.array([
        [-(1 - eta), -(1 - xi)],
        [ (1 - eta), -(1 + xi)],
        [ (1 + eta),  (1 + xi)],
        [-(1 + eta),  (1 - xi)]
    ]) * 0.25
    
    # Jacobian at centroid
    Jmat = dN_dxi.T @ X_ref_elem
    detJ = np.linalg.det(Jmat)
    invJ = np.linalg.inv(Jmat)
    dN_dx = dN_dxi @ invJ  # (4,2)
    
    # Deformation gradient at centroid
    F = X_def_elem.T @ dN_dx  # (2,2)
    
    # Compute Green-Lagrange strain tensor: E = 0.5*(F^T F - I)
    C = F.T @ F
    strain_matrix_2d = 0.5 * (C - np.eye(2))
    # Expand to 3x3 tensor for plane-strain
    strain_tensor = np.array([[strain_matrix_2d[0,0], strain_matrix_2d[0,1], 0],
                              [strain_matrix_2d[1,0], strain_matrix_2d[1,1], 0],
                              [0,                    0,                    0]])
    
    # Compute 1st Piola-Kirchhoff stress using Neo-Hookean model
    J = np.linalg.det(F)
    FinvT = np.linalg.inv(F).T
    P = mu * (F - FinvT) + lam * np.log(J) * FinvT
    
    # Convert 1st PK stress to Cauchy stress: sigma = (1/J) * P * F^T
    cauchy_stress_2d = (1.0/J) * P @ F.T
    
    # Expand to 3x3 tensor for plane-strain; σ_zz set to 0 for simplicity
    stress_tensor = np.array([[cauchy_stress_2d[0,0], cauchy_stress_2d[0,1], 0],
                              [cauchy_stress_2d[1,0], cauchy_stress_2d[1,1], 0],
                              [0,                    0,                    0]])
    
    # Compute von Mises stress for plane strain
    s_xx = cauchy_stress_2d[0,0]
    s_yy = cauchy_stress_2d[1,1]
    s_xy = cauchy_stress_2d[0,1]
    von_mises = np.sqrt(s_xx**2 - s_xx*s_yy + s_yy**2 + 3*s_xy**2)
    
    return stress_tensor, strain_tensor, von_mises

def compute_kinetic_energy(v, M_lumped):
    ke = 0.0
    num_nodes = len(M_lumped) // 2
    for i in range(num_nodes):
        vx = v[2*i]
        vy = v[2*i + 1]
        m = M_lumped[2*i]  # assume diagonal lumped mass matrix entries for both directions are equal
        ke += 0.5 * m * (vx**2 + vy**2)
    return ke

def compute_internal_energy(elems, coords, u, E, nu, thickness, element_type: str):
    total_internal_energy = 0.0
    for elem in elems:
        # Get reference and deformed coordinates for the element
        X_ref_elem = coords[elem]
        # Extract current displacements for the nodes in this element
        # Assuming u is a flat array with 2 dofs per node
        dof_indices = []
        for node in elem:
            dof_indices.extend([2*node, 2*node+1])
        u_elem = u[dof_indices].reshape(len(elem), 2)
        X_def_elem = X_ref_elem + u_elem

        # Compute stress and strain at the element's centroid
        if element_type == "quad":
            stress_tensor, strain_tensor, _ = compute_element_stress_strain_quad(X_ref_elem, X_def_elem, E, nu)
            
            # Compute area for the quadrilateral element (approximate as two triangles)
            X = X_ref_elem
            cross_val1 = np.cross(X[1]-X[0], X[3]-X[0])
            cross_val2 = np.cross(X[2]-X[1], X[3]-X[1])
            area = 0.5 * (abs(cross_val1) + abs(cross_val2))
            volume = area * thickness            
            
        elif element_type == "tri":
            stress_tensor, strain_tensor, _ = compute_element_stress_strain_tri3(X_ref_elem, X_def_elem, E, nu)

            # Compute area for the triangular element
            X = X_ref_elem
            area = 0.5 * abs(np.cross(X[1] - X[0], X[2] - X[0]))
            volume = area * thickness            

        # Compute internal energy for this element: 0.5 * sigma:epsilon * volume
        strain_energy = 0.5 * np.tensordot(stress_tensor, strain_tensor) * volume
        total_internal_energy += strain_energy
    return total_internal_energy

def compute_hourglass_energy(elems, coords, u, thickness):
    total_hourglass_energy = 0.0
    for elem in elems:
        # Get reference and deformed coordinates for the element
        X_ref_elem = coords[elem]
        dof_indices = []
        for node in elem:
            dof_indices.extend([2*node, 2*node+1])
        u_elem = u[dof_indices].reshape(len(elem), 2)
        X_def_elem = X_ref_elem + u_elem

        # Compute hourglass energy (simplified version)
        # This is a placeholder for the actual hourglass energy computation
        # which would depend on the specific hourglass control scheme used.
        hourglass_energy = np.sum((X_def_elem - X_ref_elem)**2) * thickness
        total_hourglass_energy += hourglass_energy
    return total_hourglass_energy

# ===========================================================
#              Plain Strain Neo-Hookean Models
# ===========================================================
@njit
def plane_strain_neo_hookean_t3p0(
    X_ref: np.ndarray,
    X_def: np.ndarray,
    mu: float,
    kappa: float
) -> np.ndarray:
    """
    Computes the internal force vector for a linear triangle (T3) element using a P0 stabilization.
    
    Args:
        X_ref (np.ndarray): Reference configuration (3 nodes, 2D coordinates).
        X_def (np.ndarray): Deformed configuration (3 nodes, 2D coordinates).
        mu (float): Shear modulus.
        kappa (float): Bulk modulus.

    Returns:
        np.ndarray: Internal force vector (size 6, corresponding to 3 nodes).
    """

    # Compute shape function gradients (constant in T3 elements)
    X1, X2, X3 = X_ref  # Triangle vertices in reference config
    A = 0.5 * np.linalg.det(np.array([[1, X1[0], X1[1]],
                                       [1, X2[0], X2[1]],
                                       [1, X3[0], X3[1]]]))  # Element area
    
    if A <= 0:
        raise ValueError("Invalid element: Area must be positive.")

    # Shape function derivatives in reference configuration
    dN_dx_ref = np.array([
        [X2[1] - X3[1], X3[1] - X1[1], X1[1] - X2[1]],
        [X3[0] - X2[0], X1[0] - X3[0], X2[0] - X1[0]]
    ]) / (2 * A)  # Shape: (2,3)

    # TODO: figure out why this is needed!

    
    F = dN_dx_ref @ X_def  # Correct: (2,3) @ (3,2) = (2,2)

    # Compute isochoric deformation gradient
    J = np.linalg.det(F)  # Jacobian determinant
    if J <= 0:
        raise ValueError("Negative Jacobian detected. Element is inverted.")
    
    F_iso = (J ** (-1 / 2)) * F  # Isochoric part
    P_iso = mu * (F_iso - np.linalg.inv(F_iso).T)  # Deviatoric stress
    
    # Volumetric stress component
    p = kappa * np.log(J)  # Volumetric pressure
    P_vol = p * np.linalg.inv(F).T  # Volumetric stress

    # Total first Piola-Kirchhoff stress tensor
    P = P_iso + P_vol

    # Compute internal force vector
    f_elem = np.zeros(6)
    for i in range(3):  # Loop over nodes
        force_contrib = (dN_dx_ref[0, i] * P[0, :] + dN_dx_ref[1, i] * P[1, :])
        f_elem[2 * i:2 * i + 2] += force_contrib * (2 * A)  # Multiply by element area

    return f_elem

@njit
def plane_strain_neo_hookean_q1p0(
    X_ref: np.ndarray,
    X_def: np.ndarray,
    mu: float,
    kappa: float
) -> np.ndarray:
    """
    Reimplementation of a 2D internal force vector for a quadrilateral element
    using the Q1P0 includes volumetric locking mitigation via volume-based 
    corrections (B-bar method).
    """

    # Gaussian quadrature points and weights for 2x2 integration
    gauss_pts = np.array([[-1 / np.sqrt(3), -1 / np.sqrt(3)],
                          [1 / np.sqrt(3), -1 / np.sqrt(3)],
                          [1 / np.sqrt(3), 1 / np.sqrt(3)],
                          [-1 / np.sqrt(3), 1 / np.sqrt(3)]])
    weights = np.ones(4)

    # Initialize force vector and volume-related quantities
    f_elem = np.zeros(8)
    element_volume = 0.0
    reference_volume = 0.0
    mean_gradient = np.zeros((2, 4))  # Mean shape function gradients (B-bar)

    # First pass: compute element volumes and mean gradients
    detJ_list = []
    dN_dx_list = []

    for gp, w in zip(gauss_pts, weights):
        xi, eta = gp

        # Shape function derivatives w.r.t xi and eta
        dN_dxi = np.array([
            [-(1 - eta), -(1 - xi)],
            [(1 - eta), -(1 + xi)],
            [(1 + eta), (1 + xi)],
            [-(1 + eta), (1 - xi)]
        ]) * 0.25
        
        # Compute Jacobian matrix and determinant
        Jmat = dN_dxi.T @ X_ref
        detJ = np.linalg.det(Jmat)
        invJ = np.linalg.inv(Jmat)
        dN_dx = dN_dxi @ invJ  # Gradients w.r.t x and y

        # Update volumes
        reference_volume += w * detJ
        element_volume += w * np.linalg.det(dN_dx.T @ X_def)

        # Accumulate mean gradient for B-bar correction
        mean_gradient += w * detJ * dN_dx.T

        # Store for the second pass
        detJ_list.append(detJ)
        dN_dx_list.append(dN_dx)

    # Normalize mean gradient by the element volume
    mean_gradient /= element_volume

    # Second pass: compute internal forces
    for idx, (gp, w) in enumerate(zip(gauss_pts, weights)):
        dN_dx = dN_dx_list[idx]
        detJ = detJ_list[idx]

        # Deformation gradient
        F = X_def.T @ dN_dx

        # Compute isochoric part of deformation gradient
        J = np.linalg.det(F)
        F_iso = (J ** (-1 / 2)) * F  # Remove volumetric effects

        # Deviatoric (isochoric) stress
        P_iso = mu * (F_iso - np.linalg.inv(F_iso).T)

        # Volumetric stress correction using mean gradient
        p = kappa * np.log(J)
        P_vol = p * np.linalg.inv(F).T

        # Total first Piola-Kirchhoff stress tensor
        P = P_iso + P_vol

        # Compute and accumulate internal force contributions
        for i in range(4):
            force_contrib = (dN_dx[i, 0] * P[0, :] + dN_dx[i, 1] * P[1, :])
            f_elem[2 * i:2 * i + 2] += force_contrib * detJ * w

    return f_elem

# ===========================================================
#                  Explicit Solver
# ===========================================================
@njit(parallel=True)
def compute_lumped_mass(coords, elems, rho, thickness, element_type):
    num_nodes = coords.shape[0]
    M_lumped = np.zeros(2 * num_nodes)
    
    def safe_cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    for elem in elems:
        X = coords[elem]  # Coordinates of element nodes
        
        if element_type == "quad":
            area = 0.5 * abs(safe_cross(X[1] - X[0], X[3] - X[0])) + \
                   0.5 * abs(safe_cross(X[2] - X[1], X[3] - X[1]))
            m_elem = rho * area * thickness
            m_node = m_elem / 4.0  # Lump equally to 4 nodes
        
        elif element_type == "tri":
            area = 0.5 * abs(safe_cross(X[1] - X[0], X[2] - X[0]))
            m_elem = rho * area * thickness
            m_node = m_elem / 3.0  # Lump equally to 3 nodes
        
        else:
            raise ValueError(f"Unsupported element type: {element_type}")
        
        for nd in elem:
            M_lumped[2*nd]   += m_node
            M_lumped[2*nd+1] += m_node
    
    return M_lumped

def register_fixed_dofs(fixed_nodes, dofs, fixed_dofs):
    for node in fixed_nodes:
        if dofs == [0]:
            fixed_dofs.append(2 * node)     # x-direction
        elif dofs == [1]:
            fixed_dofs.append(2 * node + 1)
        elif dofs == [0, 1]:
            fixed_dofs.append(2 * node)
            fixed_dofs.append(2 * node + 1) # y-direction
            
def apply_gravity(f_ext, M_lumped, g):
    for i in range(len(M_lumped)):
        f_ext[i] = - M_lumped[i] * g
        
def apply_forces(f_ext, node, f):
    f_ext[2*node] += f[0]
    f_ext[2*node + 1] += f[1]


# ==================================
# 🚀 GPU Implementation with CUDA
# ==================================
@cuda.jit
def compute_internal_forces_gpu_kernel(elems, coords, u, mu, kappa, element_type, f_int):
    """
    GPU Kernel for internal force computation.
    Each thread processes one element.
    """
    i = cuda.grid(1)  # Get thread index
    if i >= elems.shape[0]:
        return  # Ensure we stay in bounds

    elem = elems[i]
    X_ref = cuda.local.array((4, 2), dtype=np.float64)  # Local memory
    X_def = cuda.local.array((4, 2), dtype=np.float64)  
    f_elem = cuda.local.array(8, dtype=np.float64)
    dof_map = cuda.local.array(8, dtype=np.int32)

    # Load element data
    for j in range(len(elem)):
        node = elem[j]
        X_ref[j, 0] = coords[node, 0]
        X_ref[j, 1] = coords[node, 1]
        dof_map[2*j] = 2 * node
        dof_map[2*j+1] = 2 * node + 1

    # Compute deformed configuration
    for j in range(len(elem)):
        X_def[j, 0] = X_ref[j, 0] + u[dof_map[2*j]]
        X_def[j, 1] = X_ref[j, 1] + u[dof_map[2*j+1]]

    # Compute element forces
    if element_type == "quad":
        f_elem = plane_strain_neo_hookean_q1p0(X_ref, X_def, mu, kappa)
    elif element_type == "tri":
        f_elem = plane_strain_neo_hookean_t3p0(X_ref, X_def, mu, kappa)
    else:
        return

    # Assemble into global force vector (atomic for safety)
    for j in range(len(dof_map)):
        cuda.atomic.add(f_int, dof_map[j], f_elem[j])

def compute_internal_forces_gpu(elems, coords, u, mu, kappa, element_type):
    """Launch GPU kernel and return results"""
    d_elems = cuda.to_device(elems)
    d_coords = cuda.to_device(coords)
    d_u = cuda.to_device(u)
    d_f_int = cuda.to_device(np.zeros_like(u))  # Output array

    threads_per_block = 128
    blocks_per_grid = (elems.shape[0] + threads_per_block - 1) // threads_per_block

    compute_internal_forces_gpu_kernel[blocks_per_grid, threads_per_block](
        d_elems, d_coords, d_u, mu, kappa, element_type, d_f_int
    )

    return d_f_int.copy_to_host()
  
# ==================================
# ⚡ CPU Parallel Implementation
# ==================================
@njit(parallel=True)
def compute_internal_forces_cpu(elems, coords, u, mu, kappa, element_type):
    """CPU-parallelized internal force computation."""
    num_nodes = coords.shape[0]
    f_int = np.zeros(2 * num_nodes)

    for e in prange(elems.shape[0]):  # <-- Parallelized loop
        elem = elems[e]
        X_ref = coords[elem]
        dof_map = np.zeros(2 * len(elem), dtype=np.int32)

        for i in range(len(elem)):
            dof_map[2*i] = 2 * elem[i]
            dof_map[2*i+1] = 2 * elem[i] + 1

        X_def = X_ref + u[dof_map].reshape(len(elem), 2)

        if element_type == "quad":
            f_elem = plane_strain_neo_hookean_q1p0(X_ref, X_def, mu, kappa)
        elif element_type == "tri":                            
            f_elem = plane_strain_neo_hookean_t3p0(X_ref, X_def, mu, kappa)
        else:
            continue  

        for i_loc in range(len(dof_map)):
            f_int[dof_map[i_loc]] += f_elem[i_loc]  # Parallel-safe update

    return f_int

def explicit_solver(dt, mu, kappa, rho, coords, elems, element_type, fixed_nodes_and_dofs, displacements, nsteps=200000, save_interval=100):
    """
    2D explicit dynamic FEM for a cantilever beam with Neo-Hookean (plane-strain)
    material, supporting multiple element types (quadrilateral, triangular, etc.).
    """
    time_values = []
    kinetic_energies = []
    internal_energies = []
    hourglass_energies = []

    num_nodes = coords.shape[0]

    thickness = 1.0  # Plane strain assumption
    M_lumped = compute_lumped_mass(coords, elems, rho, thickness, element_type)
    M_inv = 1.0 / M_lumped

    # Register Fixed Boundary Conditions
    fixed_dofs = []
    for fixed_nodes, dofs in fixed_nodes_and_dofs:
        register_fixed_dofs(fixed_nodes, dofs, fixed_dofs)
    
    # External Force => Gravity in -y 
    f_ext = np.zeros(2*num_nodes)
    apply_gravity(f_ext, M_lumped, g=9.81)

    # Explicit Time Integration
    u = np.zeros(2*num_nodes)
    v = np.zeros(2*num_nodes)
    out_dir = f"output_neo_hookean_{element_type}"
    os.makedirs(out_dir, exist_ok=True)
    
    if cuda.is_available():
        print("🚀 Running on GPU!")        
    else:
        print("⚡ Running on CPU (parallel mode)")        

    for step in range(nsteps):
        f_int = np.zeros(2*num_nodes)

        if cuda.is_available():
            f_int = compute_internal_forces_gpu(elems, coords, u, mu, kappa, element_type)
        else:            
            f_int = compute_internal_forces_cpu(elems, coords, u, mu, kappa, element_type)
        
        f_net = f_ext - f_int
        f_net[fixed_dofs] = 0.0
        
        damping_coefficient = 0.5
        damping_force = -damping_coefficient * M_lumped * v
        f_net += damping_force
        
        a = f_net * M_inv
        v += a * dt
        u += v * dt
        
        u[fixed_dofs] = 0.0
        v[fixed_dofs] = 0.0
        
        # Apply prescribed displacements
        for disp_nodes, prescribed_disp_vec in displacements:
            for node in disp_nodes:
                if prescribed_disp_vec[0] is not None:
                    u[2*node] = step * prescribed_disp_vec[0]
                    v[2*node] = prescribed_disp_vec[0] / dt
                if prescribed_disp_vec[1] is not None:
                    u[2*node + 1] = step * prescribed_disp_vec[1]
                    v[2*node + 1] = prescribed_disp_vec[1] / dt
        
        nu = kappa / (2 * mu) - 1.0 / 3.0
        E = 2 * mu * (1 + nu)
        
        ke = compute_kinetic_energy(v, M_lumped)
        ie = compute_internal_energy(elems, coords, u, E, nu, thickness, element_type)
        he = compute_hourglass_energy(elems, coords, u, thickness)
        
        kinetic_energies.append(ke)
        internal_energies.append(ie)
        hourglass_energies.append(he)
        time_values.append(step * dt)
        
        if step % save_interval == 0 or step == nsteps-1:
            disp_2d = u.reshape(-1,2)
            fname = os.path.join(out_dir, f"beam_step_{step:04d}.vtu")
            export_vtu_2d_with_nodal_stress_strain(coords, elems, disp_2d, fname, E, nu, element_type)
            print(f"Step {step}/{nsteps} => wrote {fname}")

    save_energies("energies.csv", time_values, kinetic_energies, internal_energies, hourglass_energies)
    plot_energies("energy_plot.pdf", out_dir, time_values, kinetic_energies, internal_energies, hourglass_energies)
    
    print("Done. Results in", out_dir)

# ===========================================================
#                 Visualization Functions
# ===========================================================
def save_energies(csv_filename: str, time_values: list, kinetic_energies: list, internal_energies: list, hourglass_energies: list):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time', 'kinetic_energy', 'internal_energy', 'hourglass_energy'])
        for t, ke, ie, he in zip(time_values, kinetic_energies, internal_energies, hourglass_energies):
            writer.writerow([t, ke, ie, he])
    
    print(f"Energy data saved to {csv_filename}")
    
def plot_energies(plot_filename: str, out_dir: str, time_values: list, kinetic_energies: list, internal_energies: list, hourglass_energies: list):
    plt.figure(figsize=(8, 6))
    plt.plot(time_values, kinetic_energies, label='Kinetic Energy')
    plt.plot(time_values, internal_energies, label='Internal Energy')
    plt.plot(time_values, hourglass_energies, label='Hourglass Energy')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    print(f"Energy plot saved to {plot_filename}")

def export_vtu_2d_with_nodal_stress_strain(coords, elems, disp, filename, E, nu, element_type):

    # Update deformed coordinates based on displacement
    coords_def = coords + disp
    num_nodes = coords.shape[0]

    # Initialize arrays for nodal stress, strain, von Mises stress, and counters
    nodal_stress = np.zeros((num_nodes, 3, 3))
    nodal_strain = np.zeros((num_nodes, 3, 3))
    nodal_von_mises = np.zeros(num_nodes)
    count = np.zeros(num_nodes)  # to track contributions per node

    # Loop over elements to accumulate stress and strain contributions at nodes
    for elem in elems:
        X_ref_elem = coords[elem]       # reference positions of the element
        X_def_elem = coords_def[elem]   # deformed positions of the element

        # Compute stress and strain tensors at the element's centroid
        if element_type == "quad":
            stress_tensor, strain_tensor, von_mises = compute_element_stress_strain_quad(X_ref_elem, X_def_elem, E, nu)
        elif element_type == "tri":            
            stress_tensor, strain_tensor, von_mises = compute_element_stress_strain_tri3(X_ref_elem, X_def_elem, E, nu)
        else:
            raise ValueError(f"Unsupported element type: {element_type}")

        # Accumulate stress, strain, and von Mises stress at each node of the element
        for node in elem:
            nodal_stress[node] += stress_tensor
            nodal_strain[node] += strain_tensor
            nodal_von_mises[node] += von_mises
            count[node] += 1

    # Average stresses, strains, and von Mises stress at nodes
    for i in range(num_nodes):
        if count[i] > 0:
            nodal_stress[i] /= count[i]
            nodal_strain[i] /= count[i]
            nodal_von_mises[i] /= count[i]

    # Prepare meshio input for exporting
    points_3d = np.column_stack([coords, np.zeros(num_nodes)])  # Convert 2D coords to 3D
    disp_3d = np.column_stack([disp, np.zeros(num_nodes)])  # Convert 2D displacements to 3D

    # Reshape tensor fields for meshio: each point gets a flat array of 9 components
    stress_flat = nodal_stress.reshape(num_nodes, 9)
    strain_flat = nodal_strain.reshape(num_nodes, 9)

    point_data = {
        "displacement": disp_3d,
        "stress": stress_flat,
        "strain": strain_flat,
        "von_mises_stress": nodal_von_mises
    }

    if element_type == "quad":
        cells = [("quad", elems)]
    elif element_type == "tri":
        cells = [("triangle", elems)]
    else:
        raise ValueError(f"Unsupported element type: {element_type}")

    msh = meshio.Mesh(points=points_3d, cells=cells, point_data=point_data)
    msh.write(filename)
    print("Saved:", filename)

    
if __name__ == "__main__": 
    # Simulation parameters
    dt = 1e-5     
    nsteps = 200000  
    save_interval = 1000
    
    # Material properties
    nu = 0.4999 # Poisson's ratio
    E = 1e8    # Young's modulus
    mu = E / (2 * (1 + nu))  # shear modulus
    kappa = E / (3 * (1 - 2 * nu))  # bulk modulus    
    rho = 1e3        # density
    print("Material parameters:")
    print(f"mu = {mu}, kappa = {kappa}")        
    
    element_type = "tri"    
    
    # load a mesh
    coords, elems, nodesets, sidesets = load_exo_file("data/cube_mesh_tri_cubit.e")    
    
    # print num of dofs
    print("Number of nodes:", coords.shape[0])
    print("Number of elements:", elems.shape[0])
    print("Number of dofs:", 2 * coords.shape[0])    
    
    # Define boundary conditions
    fixed_node_set_1 = [nodesets["nodeset_1"], [0]]
    fixed_node_set_3 = [nodesets["nodeset_3"], [1]]    
    fixed_nodes_and_dofs = [fixed_node_set_1, fixed_node_set_3]
    
    # Define the displaceemnts for the nodes
    disp_node_set_1 = [nodesets["nodeset_2"], [-0.00001, None]]
    displacements = [disp_node_set_1]
    
    print (sidesets)
        
    # Simulate!   
    star_time = time.time()
    # explicit_solver(dt, mu, kappa, rho, coords, elems, element_type, fixed_nodes_and_dofs, displacements, nsteps, save_interval)
    end_time = time.time()
    
    print("Elapsed time:", end_time - star_time, "seconds")