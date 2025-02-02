"""
2D Neo-Hookean Cantilever Beam (Plane-Strain) via Explicit Dynamic FEM with Linear Quad Elements
--------------------------------------------------------------------------------------------
 - Domain: [0, Lx] x [0, Hy] (rectangular beam)
 - Mesh:   Structured grid of linear quadrilateral elements
 - BC:     Left edge x=0 fixed in x,y
 - Load:   Gravity downward
 - Material: Compressible Neo-Hookean, plane-strain
 - Method:  Explicit (no global stiffness), internal force from 1st PK stress
 - Output:  Nodal displacements in .vtu for visualization
"""
import csv
import numpy as np
import meshio
import os
import matplotlib.pyplot as plt

from mpi4py import MPI


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

# ===========================================================
#                  Output-Related Functions
# ===========================================================
def compute_element_stress_strain(X_ref_elem, X_def_elem, E, nu):
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

def compute_internal_energy(elems, coords, u, E, nu, thickness):
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
        stress_tensor, strain_tensor, _ = compute_element_stress_strain(X_ref_elem, X_def_elem, E, nu)

        # Compute area for the quadrilateral element (approximate as two triangles)
        X = X_ref_elem
        cross_val1 = np.cross(X[1]-X[0], X[3]-X[0])
        cross_val2 = np.cross(X[2]-X[1], X[3]-X[1])
        area = 0.5 * (abs(cross_val1) + abs(cross_val2))
        volume = area * thickness

        # Compute internal energy for this element: 0.5 * sigma:epsilon * volume
        strain_energy = 0.5 * np.tensordot(stress_tensor, strain_tensor) * volume
        total_internal_energy += strain_energy
    return total_internal_energy


# ===========================================================
#              Plain Strain Neo-Hookean Models
# ===========================================================
def plane_strain_neo_hookean_force_quad_bbar_incompressible(
    X_ref: np.ndarray,
    X_def: np.ndarray,
    E: float,
    nu: float
) -> np.ndarray:
    """
    Compute the 2D internal force vector for a quadrilateral element
    with incompressible Neo-Hookean material in plane-strain conditions.
    Implements the B-bar method to mitigate volumetric locking.
    """

    # Material properties: enforcing near-incompressibility
    mu = E / (2.0 * (1.0 + nu))  # Shear modulus
    kappa = mu * 1000  # Large bulk modulus to enforce incompressibility

    # Gaussian quadrature points and weights for 2x2 integration
    gauss_pts = np.array([[-1/np.sqrt(3), -1/np.sqrt(3)],
                          [ 1/np.sqrt(3), -1/np.sqrt(3)],
                          [ 1/np.sqrt(3),  1/np.sqrt(3)],
                          [-1/np.sqrt(3),  1/np.sqrt(3)]])
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    f_elem = np.zeros(8)
    lnJ_vals = []  # Store ln(J) at each Gauss point
    detJ_list = []
    F_list = []
    dN_dx_list = []

    # First pass: compute F and ln(J) at each Gauss point
    for gp, w in zip(gauss_pts, weights):
        xi, eta = gp

        # Shape function derivatives with respect to xi, eta
        dN_dxi = np.array([
            [-(1 - eta), -(1 - xi)],
            [ (1 - eta), -(1 + xi)],
            [ (1 + eta),  (1 + xi)],
            [-(1 + eta),  (1 - xi)]
        ]) * 0.25  # Derivatives wrt xi and eta

        # Compute the Jacobian matrix and its determinant
        Jmat = dN_dxi.T @ X_ref
        detJ = np.linalg.det(Jmat)
        invJ = np.linalg.inv(Jmat)
        dN_dx = dN_dxi @ invJ  # (4,2)

        # Compute the deformation gradient at this Gauss point
        F = X_def.T @ dN_dx  # (2,2)
        J = np.linalg.det(F)

        # Store quantities for B-bar correction
        lnJ_vals.append(np.log(J))
        detJ_list.append(detJ)
        F_list.append(F)
        dN_dx_list.append(dN_dx)

    # Compute the weighted average of ln(J) over the element (B-bar method)
    avg_lnJ = sum(lnJ * detJ * w for lnJ, detJ, w in zip(lnJ_vals, detJ_list, weights)) / sum(detJ_list)

    # Second pass: compute forces using B-bar corrected volumetric strain
    for idx, (gp, w) in enumerate(zip(gauss_pts, weights)):
        xi, eta = gp
        F = F_list[idx]
        dN_dx = dN_dx_list[idx]
        detJ = detJ_list[idx]

        # Compute isochoric deformation gradient
        J = np.linalg.det(F)
        F_iso = (J ** (-1/2)) * F  # Remove volumetric effects

        # Deviatoric stress (isochoric part)
        P_iso = mu * (F_iso - np.linalg.inv(F_iso).T)

        # Volumetric stress correction using B-bar averaged strain
        p = kappa * (avg_lnJ)  # Enforcing incompressibility via penalty
        P_vol = p * np.linalg.inv(F).T

        # Total first Piola-Kirchhoff stress tensor
        P = P_iso + P_vol

        # Compute and accumulate internal force contributions
        for i in range(4):
            force_contrib = (dN_dx[i, 0] * P[0, :] + dN_dx[i, 1] * P[1, :])
            f_elem[2*i:2*i+2] += force_contrib * detJ * w

    return f_elem

def plane_strain_neo_hookean_q1p0(
    X_ref: np.ndarray,
    X_def: np.ndarray,
    E: float,
    nu: float
) -> np.ndarray:
    """
    Reimplementation of a 2D internal force vector for a quadrilateral element
    using the Q1P0 formulation inspired by the C++ code. Includes volumetric
    locking mitigation via volume-based corrections (B-bar method).
    """
    # Material properties
    mu = E / (2.0 * (1.0 + nu))  # Shear modulus
    kappa = mu * 1000  # Large bulk modulus to enforce near-incompressibility

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

def plane_strain_ogden_force_quad_bbar(
    X_ref: np.ndarray,
    X_def: np.ndarray,
    mu_params: list,
    alpha_params: list
) -> np.ndarray:
    """
    Compute the 2D internal force vector for a quadrilateral element
    with incompressible Ogden hyperelastic material in plane-strain conditions.
    Implements the B-bar method to mitigate volumetric locking.
    
    Args:
        X_ref (np.ndarray): Reference configuration (4 nodes x 2 coordinates)
        X_def (np.ndarray): Deformed configuration (4 nodes x 2 coordinates)
        mu_params (list): Ogden model shear modulus parameters [mu1, mu2, ...]
        alpha_params (list): Ogden model exponent parameters [alpha1, alpha2, ...]

    Returns:
        np.ndarray: Internal force vector (8 elements)
    """

    # Number of Gauss quadrature points for 2x2 integration
    gauss_pts = np.array([[-1/np.sqrt(3), -1/np.sqrt(3)],
                          [ 1/np.sqrt(3), -1/np.sqrt(3)],
                          [ 1/np.sqrt(3),  1/np.sqrt(3)],
                          [-1/np.sqrt(3),  1/np.sqrt(3)]])
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    f_elem = np.zeros(8)
    lnJ_vals = []
    detJ_list = []
    F_list = []
    dN_dx_list = []

    # First pass: Compute deformation gradient and log(J) at Gauss points
    for gp, w in zip(gauss_pts, weights):
        xi, eta = gp

        # Shape function derivatives with respect to xi, eta
        dN_dxi = np.array([
            [-(1 - eta), -(1 - xi)],
            [ (1 - eta), -(1 + xi)],
            [ (1 + eta),  (1 + xi)],
            [-(1 + eta),  (1 - xi)]
        ]) * 0.25

        # Compute the Jacobian matrix and determinant
        Jmat = dN_dxi.T @ X_ref
        detJ = np.linalg.det(Jmat)
        invJ = np.linalg.inv(Jmat)
        dN_dx = dN_dxi @ invJ  # (4,2)

        # Compute the deformation gradient at this Gauss point
        F = X_def.T @ dN_dx  # (2,2)
        J = np.linalg.det(F)

        # Store quantities for B-bar correction
        lnJ_vals.append(np.log(J))
        detJ_list.append(detJ)
        F_list.append(F)
        dN_dx_list.append(dN_dx)

    # Compute the weighted average of ln(J) over the element (B-bar method)
    avg_lnJ = sum(lnJ * detJ * w for lnJ, detJ, w in zip(lnJ_vals, detJ_list, weights)) / sum(detJ_list)

    # Second pass: compute forces using B-bar corrected volumetric strain
    for idx, (gp, w) in enumerate(zip(gauss_pts, weights)):
        xi, eta = gp
        F = F_list[idx]
        dN_dx = dN_dx_list[idx]
        detJ = detJ_list[idx]

        # Ensure incompressibility by using isochoric deformation gradient
        J = np.linalg.det(F)
        F_iso = (J ** (-1/2)) * F  # Remove volumetric effects

        # Compute principal stretches (singular values of F_iso)
        U, S, Vt = np.linalg.svd(F_iso)
        lambda1, lambda2 = S[0], S[1]
        lambda3 = 1.0 / (lambda1 * lambda2)  # Incompressibility condition

        # Compute 1st Piola-Kirchhoff stress tensor using Ogden model
        P_iso = np.zeros((2, 2))
        for mu, alpha in zip(mu_params, alpha_params):
            P_iso += mu * (lambda1 ** (alpha - 1) - lambda3 ** (-alpha - 1)) * np.outer(U[:, 0], U[:, 0])
            P_iso += mu * (lambda2 ** (alpha - 1) - lambda3 ** (-alpha - 1)) * np.outer(U[:, 1], U[:, 1])

        # Volumetric stress correction using B-bar averaged strain
        p = avg_lnJ  # Average volumetric strain
        P_vol = p * np.linalg.inv(F).T  # Pressure term due to incompressibility

        # Total first Piola-Kirchhoff stress tensor
        P = P_iso + P_vol

        # Compute internal force contributions
        for i in range(4):
            force_contrib = (dN_dx[i, 0] * P[0, :] + dN_dx[i, 1] * P[1, :])
            f_elem[2*i:2*i+2] += force_contrib * detJ * w

    return f_elem



def plane_strain_neo_hookean_force_quad_bbar(
    X_ref: np.ndarray,
    X_def: np.ndarray,
    E: float,
    nu: float
) -> np.ndarray:
    """
    Compute the 2D internal force vector for a single quadrilateral element
    using the B-bar method with compressible Neo-Hookean material in plane-strain.
    """
    # Material constants (Lamé parameters) for plane-strain
    mu = E / (2.0 * (1.0 + nu))
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # Gaussian quadrature points and weights for 2x2 integration
    gauss_pts = np.array([[-1/np.sqrt(3), -1/np.sqrt(3)],
                          [ 1/np.sqrt(3), -1/np.sqrt(3)],
                          [ 1/np.sqrt(3),  1/np.sqrt(3)],
                          [-1/np.sqrt(3),  1/np.sqrt(3)]])
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    f_elem = np.zeros(8)
    lnJ_vals = []      # To store ln(J) at each Gauss point
    detJ_global = []   # To store det(J) at each Gauss point
    F_list = []        # To store deformation gradients
    dN_dx_list = []    # To store shape function derivatives at each GP
    detJ_list = []     # To store determinants of Jacobian at each GP

    # First pass: compute F, ln(J), and gather integration data at each Gauss point
    for gp, w in zip(gauss_pts, weights):
        xi, eta = gp

        # Shape functions and their derivatives with respect to xi, eta
        N = np.array([
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta)
        ])

        dN_dxi = np.array([
            [-(1 - eta), -(1 - xi)],
            [ (1 - eta), -(1 + xi)],
            [ (1 + eta),  (1 + xi)],
            [-(1 + eta),  (1 - xi)]
        ]) * 0.25  # Derivatives of shape functions wrt xi and eta

        # Compute the Jacobian matrix with respect to the reference configuration
        Jmat = dN_dxi.T @ X_ref  # (2,2)
        detJ = np.linalg.det(Jmat)
        invJ = np.linalg.inv(Jmat)
        dN_dx = dN_dxi @ invJ  # (4,2)

        # Deformation gradient at this GP
        F = X_def.T @ dN_dx  # (2,2)

        # Store necessary quantities for B-bar computation
        lnJ_vals.append(np.log(np.linalg.det(F)))
        detJ_list.append(detJ)
        F_list.append(F)
        dN_dx_list.append(dN_dx)

    # Compute the weighted average of ln(J) over the element
    weighted_sum = 0.0
    total_weight = 0.0
    for lnJ, detJ, w in zip(lnJ_vals, detJ_list, weights):
        weighted_sum += lnJ * detJ * w
        total_weight += detJ * w
    avg_lnJ = weighted_sum / total_weight if total_weight != 0 else 0.0

    # Second pass: compute forces using averaged volumetric strain
    for idx, (gp, w) in enumerate(zip(gauss_pts, weights)):
        xi, eta = gp
        F = F_list[idx]
        dN_dx = dN_dx_list[idx]
        detJ = detJ_list[idx]

        # Use averaged volumetric strain for stress computation
        # Compute deviatoric part of the stress
        J_local = np.linalg.det(F)
        # Deviatoric part: remove volumetric portion from F
        F_bar = (J_local**(-0.5)) * F  # 2D equivalent scaling for isochoric part

        # Compute inverse transpose of F_bar
        FinvT_bar = np.linalg.inv(F_bar).T

        # Modified stress using B-bar: use avg_lnJ for volumetric part
        P = mu * (F - np.linalg.inv(F).T) + lam * avg_lnJ * np.linalg.inv(F).T

        # Accumulate internal force contributions
        for i in range(4):
            # Compute contribution for node i at this Gauss point
            # Combining the derivatives with the stress tensor
            force_contrib = (dN_dx[i, 0] * P[0, :] + dN_dx[i, 1] * P[1, :])
            f_elem[2*i:2*i+2] += force_contrib * detJ * w * 0.25

    return f_elem

def plane_strain_neo_hookean_force_quad(
    X_ref: np.ndarray,
    X_def: np.ndarray,
    E: float,
    nu: float
) -> np.ndarray:
    """
    Compute the 2D internal force vector for a single quadrilateral element
    using compressible Neo-Hookean material in plane-strain.

    Parameters
    ----------
    X_ref : (4,2) ndarray
        Reference (undeformed) quad coords (nodes 0,1,2,3).
    X_def : (4,2) ndarray
        Current (deformed) quad coords for the same 4 nodes.
    E, nu : float
        Young's modulus and Poisson ratio.

    Returns
    -------
    f_elem : (8,) ndarray
        The internal force contribution on these 4 nodes,
        in the order [fx0, fy0, fx1, fy1, fx2, fy2, fx3, fy3].
    """
    # Material constants (Lamé parameters) for plane-strain
    mu = E / (2.0 * (1.0 + nu))
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # Gaussian quadrature points and weights for 2x2 integration
    gauss_pts = np.array([[-1/np.sqrt(3), -1/np.sqrt(3)],
                          [ 1/np.sqrt(3), -1/np.sqrt(3)],
                          [ 1/np.sqrt(3),  1/np.sqrt(3)],
                          [-1/np.sqrt(3),  1/np.sqrt(3)]])
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    f_elem = np.zeros(8)

    for gp, w in zip(gauss_pts, weights):
        xi, eta = gp

        # Shape functions for bilinear quad
        N = np.array([
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta)
        ])  # (4,)

        dN_dxi = np.array([
            [-0.25 * (1 - eta), -0.25 * (1 - xi)],
            [ 0.25 * (1 - eta), -0.25 * (1 + xi)],
            [ 0.25 * (1 + eta),  0.25 * (1 + xi)],
            [-0.25 * (1 + eta),  0.25 * (1 - xi)]
        ])  # (4,2)

        # Compute Jacobian matrix
        J = dN_dxi.T @ X_ref  # (2,2)

        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError("Jacobian determinant is non-positive.")

        invJ = np.linalg.inv(J)

        # Gradients of shape functions w.r. to reference coordinates
        dN_dx = dN_dxi @ invJ  # (4,2)

        # Deformation gradient F
        F = X_def.T @ dN_dx  # (2,2)

        # Compute determinant
        J_det = np.linalg.det(F)

        # 1st Piola-Kirchhoff stress for compressible Neo-Hookean
        F_invT = np.linalg.inv(F).T
        P = mu * (F - F_invT) + lam * np.log(J_det) * F_invT  # (2,2)

        # B matrix
        B = dN_dx  # (4,2)

        # Internal force contribution: sum over nodes of B^T * P * weight * detJ * (1/4)
        # The factor (1/4) comes from averaging the stress over the element
        for i in range(4):
            f_elem[2*i:2*i+2] += (B[i, 0] * P[0, :] + B[i, 1] * P[1, :]) * detJ * w * 0.25

    return f_elem


# ===========================================================
#                  Explicit Solver
# ===========================================================
def main_explicit_cantilever_neo_hookean_quad():
    """
    2D explicit dynamic FEM for a cantilever beam with Neo-Hookean (plane-strain)
    material using linear quadrilateral elements. The internal force per element is
    computed from the 1st Piola-Kirchhoff stress using Gaussian quadrature.
    """
    time_values = []
    kinetic_energies = []
    internal_energies = []
    # Optionally for other energies:
    # hourglass_energies = []
    # damping_energies = []


    # -----------------------------------------------------------------
    # 1) Mesh the domain [0, Lx] x [0, Hy] with quadrilateral elements
    Lx, Hy = 10.0, 1.0
    nx, ny = 50, 5  # number of elements in x and y
    coords, elems = create_quadrilateral_mesh(Lx, Hy, nx, ny)
    num_nodes = coords.shape[0]
    num_elems = elems.shape[0]
    print(f"Mesh created: {num_nodes} nodes, {num_elems} quadrilaterals")

    # -----------------------------------------------------------------
    # 2) Material parameters, plane-strain Neo-Hookean
    E   = 1e8   # Young's modulus
    nu  = 0.4999   # Poisson ratio
    rho = 1e3   # density
    thickness = 1.0  # for the out-of-plane dimension (plane strain)

    # -----------------------------------------------------------------
    # 3) Build lumped mass (no stiffness!)
    M_lumped = np.zeros(2*num_nodes)
    for quad in elems:
        X = coords[quad]  # (4,2)
        # Compute area using two triangles
        area = 0.5 * abs(np.cross(X[1] - X[0], X[3] - X[0])) + \
               0.5 * abs(np.cross(X[2] - X[1], X[3] - X[1]))
        m_elem = rho * area * thickness
        # Lump equally to 4 nodes
        m_node = m_elem / 4.0
        for nd in quad:
            M_lumped[2*nd]   += m_node
            M_lumped[2*nd+1] += m_node

    # Inverse mass for acceleration
    M_inv = 1.0 / M_lumped

    # -----------------------------------------------------------------
    # 4) Boundary Conditions
    #    - Fix left edge (x=0) => zero displacement & velocity
    eps = 1e-9
    fixed_nodes = np.where(coords[:,0] < eps)[0]  # all nodes with x=0
    fixed_dofs = []
    for fn in fixed_nodes:
        # fixed_dofs += [2*fn, 2*fn+1]
        fixed_dofs += [2*fn] # fix only x-direction
        
    #    - Identify right edge nodes for prescribed displacement
    right_nodes = np.where(np.abs(coords[:,0] - Lx) < eps)[0]  # nodes at x = Lx
    
    # bottom nodes fixed in y-direction
    bottom_nodes = np.where(coords[:,1] < eps)[0]
    for bn in bottom_nodes:
        fixed_dofs.append(2*bn+1)
    
    # fix right nodes in y-direction
    # for rn in right_nodes:
    #     fixed_dofs.append(2*rn+1)            

    # Define prescribed displacement value (for demonstration, a constant value)
    # This can be a function of time for dynamic loading.
    prescribed_disp_x = -0.0001  # example displacement in x-direction at the right edge

    # -----------------------------------------------------------------
    # 5) External Force => Gravity in -y
    f_ext = np.zeros(2*num_nodes)
    g = 0
    for i in range(num_nodes):
        # Apply f=mass*g on the y-DOF (negative for downward)
        f_ext[2*i+1] = - M_lumped[2*i+1] * g

    # -----------------------------------------------------------------
    # 6) Explicit Time Integration
    u = np.zeros(2*num_nodes)  # displacement
    v = np.zeros(2*num_nodes)  # velocity

    dt = 1e-5     # time step (might need tuning for stability)
    nsteps = 2000  # total steps
    save_interval = 100

    out_dir = "output_neo_hookean_quad_2"
    os.makedirs(out_dir, exist_ok=True)

    for step in range(nsteps):
        # (a) Compute internal force
        f_int = np.zeros(2*num_nodes)
        for quad in elems:
            X_ref = coords[quad]  # (4,2)
            # Current coords = reference + displacement
            X_def = X_ref + u[[2*quad[0], 2*quad[0]+1,
                               2*quad[1], 2*quad[1]+1,
                               2*quad[2], 2*quad[2]+1,
                               2*quad[3], 2*quad[3]+1]].reshape(4,2)
            
            f_elem = plane_strain_neo_hookean_force_quad_bbar_incompressible(X_ref, X_def, E, nu)

            # Scatter into global f_int
            dof_map = [2*quad[0], 2*quad[0]+1,
                       2*quad[1], 2*quad[1]+1,
                       2*quad[2], 2*quad[2]+1,
                       2*quad[3], 2*quad[3]+1]
            for i_loc, i_glob in enumerate(dof_map):
                f_int[i_glob] += f_elem[i_loc]

        # (b) Net force
        f_net = f_ext - f_int

        # (c) Apply BC => zero net force at fixed dofs => no acceleration
        f_net[fixed_dofs] = 0.0

        
        # (d) Add damping force proportional to velocity:
        damping_coefficient = 0.5
        damping_force = -damping_coefficient * M_lumped * v  # element-wise multiplication
        f_net += damping_force  # Adjust net force with damping

        # (e) Acceleration
        a = f_net * M_inv  # element-wise multiply

        # (f) Update velocity, displacement
        v += a * dt
        u += v * dt

        # Re-enforce BC in displacement & velocity
        u[fixed_dofs] = 0.0
        v[fixed_dofs] = 0.0
        
        # ---- New Block: Enforce prescribed displacement at right edge ----
        for rn in right_nodes:
            # Set the x-displacement to the prescribed value.
            # You can extend this to y-displacement or time-dependent profiles as needed.
            u[2*rn] = step*prescribed_disp_x  
            # Optionally, reset velocity at right edge nodes if the displacement is static.
            v[2*rn] = prescribed_disp_x / dt
        # -------------------------------------------------------------------

        # Compute energies at this time step
        ke = compute_kinetic_energy(v, M_lumped)
        ie = compute_internal_energy(elems, coords, u, E, nu, thickness)

        current_time = step * dt
        
        kinetic_energies.append(ke)
        internal_energies.append(ie)
        time_values.append(current_time)

        # (g) Save results occasionally
        if step % save_interval == 0 or step == nsteps-1:
            disp_2d = u.reshape(-1,2)
            fname = os.path.join(out_dir, f"beam_step_{step:04d}.vtu")
            export_vtu_2d_with_nodal_stress_strain(coords, elems, disp_2d, fname, E, nu)
            print(f"Step {step}/{nsteps} => wrote {fname}")

    csv_filename = "energies.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time', 'kinetic_energy', 'internal_energy'])
        for t, ke, ie in zip(time_values, kinetic_energies, internal_energies):
            writer.writerow([t, ke, ie])
    
    print(f"Energy data saved to {csv_filename}")

    # Plot and save the energies to a PDF file
    plt.figure(figsize=(8, 6))
    plt.plot(time_values, kinetic_energies, label='Kinetic Energy')
    plt.plot(time_values, internal_energies, label='Internal Energy')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('energy_plot.pdf')
    plt.close()
    print("Energy plot saved to energy_plot.pdf")

    print("Done. Results in", out_dir)


# ===========================================================
#                 Visualization Functions
# ===========================================================
def export_vtu_2d(coords, elems, disp, filename, E, nu):
    
    # Update deformed coordinates based on displacement
    coords_def = coords + disp
    
    cells = [("quad", elems)]
    
    # Initialize arrays for tensor fields and von Mises stress
    num_elems = len(elems)
    stress_tensors = np.zeros((num_elems, 3, 3))
    strain_tensors = np.zeros((num_elems, 3, 3))
    von_mises_vals = np.zeros(num_elems)
    
    for idx, elem in enumerate(elems):
        X_ref_elem = coords[elem]       # reference positions (4,2)
        X_def_elem = coords_def[elem]   # deformed positions (4,2)
        
        stress_tensor, strain_tensor, von_mises = compute_element_stress_strain(X_ref_elem, X_def_elem, E, nu)
        stress_tensors[idx, :, :] = stress_tensor
        strain_tensors[idx, :, :] = strain_tensor
        von_mises_vals[idx] = von_mises
    
    # Prepare meshio input
    points_3d = np.column_stack([coords, np.zeros(len(coords))])
    disp_3d = np.column_stack([disp, np.zeros(len(disp))])
    
    point_data = {"displacement": disp_3d}
    
    # Store tensor fields and von Mises stress as cell data
    cell_data = {
        "stress": [stress_tensors.reshape(num_elems, 9)],  # Flatten tensors for meshio
        "strain": [strain_tensors.reshape(num_elems, 9)],
        "von_mises": [von_mises_vals]
    }
    
    msh = meshio.Mesh(points=points_3d, cells=cells, point_data=point_data, cell_data=cell_data)
    msh.write(filename)
    print("Saved:", filename)

def export_vtu_2d_with_nodal_stress_strain(coords, elems, disp, filename, E, nu):

    # Update deformed coordinates based on displacement
    coords_def = coords + disp
    num_nodes = coords.shape[0]

    # Initialize arrays for nodal stress and strain tensors and counters
    nodal_stress = np.zeros((num_nodes, 3, 3))
    nodal_strain = np.zeros((num_nodes, 3, 3))
    count = np.zeros(num_nodes)  # to track contributions per node

    # Loop over elements to accumulate stress and strain contributions at nodes
    for elem in elems:
        X_ref_elem = coords[elem]       # reference positions of the element
        X_def_elem = coords_def[elem]   # deformed positions of the element

        # Compute stress and strain tensors at the element's centroid
        stress_tensor, strain_tensor, _ = compute_element_stress_strain(X_ref_elem, X_def_elem, E, nu)

        # Accumulate stress and strain tensors at each node of the element
        for node in elem:
            nodal_stress[node] += stress_tensor
            nodal_strain[node] += strain_tensor
            count[node] += 1

    # Average stresses and strains at nodes
    for i in range(num_nodes):
        if count[i] > 0:
            nodal_stress[i] /= count[i]
            nodal_strain[i] /= count[i]

    # Prepare meshio input for exporting
    points_3d = np.column_stack([coords, np.zeros(num_nodes)])
    disp_3d = np.column_stack([disp, np.zeros(num_nodes)])

    # Reshape tensor fields for meshio: each point gets a flat array of 9 components
    stress_flat = nodal_stress.reshape(num_nodes, 9)
    strain_flat = nodal_strain.reshape(num_nodes, 9)

    point_data = {
        "displacement": disp_3d,
        "stress": stress_flat,
        "strain": strain_flat
    }

    cells = [("quad", elems)]

    msh = meshio.Mesh(points=points_3d, cells=cells, point_data=point_data)
    msh.write(filename)
    print("Saved:", filename)


if __name__ == "__main__":
    main_explicit_cantilever_neo_hookean_quad()
