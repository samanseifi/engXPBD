import numpy as np
from engxpbd.mesh import Mesh
from engxpbd.simulator import PhysicsSimulator
from engxpbd.constraints.points_constraints import PointConstraints
from engxpbd.constraints.volume_constraints import SolidElasticity, VolumeConstraint
from engxpbd.boundary_conditions import FixedPoint


if __name__ == "__main__":
    filepath = "data/cuboid_tetrahedral_mesh_1order.msh"  # Replace with your Gmsh file path
    mesh = Mesh(filepath)
    
    # mesh.add_perturbation(magnitude=0.01)
            
    # Print a summary of the mesh
    mesh_summary = f"Mesh Summary:\nNumber of nodes: {len(mesh.node_coords)}\n" + \
                          "\n".join([f"{k}: {len(v)} elements" for k, v in mesh.connectivity.items()])
    print(mesh_summary)

    # Initialize the physics simulator
    simulator = PhysicsSimulator(mesh, dt=0.01)
  
    # Fix the four corners (example indices, adjust as necessary)
    # fixed_nodes = [0, 12, 2, 29, 49, 31, 4, 22, 6]  # Replace with actual corner node indices
    fixed_nodes = [0,17,18,2,54,136,138,58,55,137,139,59,4,39,40,6]
    bc_nodes = [3,29,28,1,56,134,132,52,57,135,133,53,7,51,50,5]
    # fixed_nodes = [0,27,28,29,30,2,104,436,440,444,448,112,105,437,441,445,449,113,106,438,442,446,450,114,107,439,443,447,451,115,4,73,74,75,76,6]
    simulator.set_fixed_nodes(fixed_nodes)
    simulator.set_bc_nodes(bc_nodes)
    
    # Define constraint for solid model
    constraints = []
    for cell in mesh.get_connectivity("tetra"):
        constraints.append(lambda pos, c=cell, m=simulator.masses, f=simulator.fixed_nodes, b=bc_nodes:
                           SolidElasticity.saint_venant_constraint(pos, mesh.node_coords_init, *c, 1e8, 0.49, 0.0, simulator.dt, m, f, b))

    # fixed nodes
    constraints.append(lambda pos, pos_init=mesh.node_coords_init, fixed_ids=fixed_nodes, dof=None:
                           PointConstraints.fix_points(pos, pos_init, fixed_ids, dof))
    
    constraints.append(lambda pos, pos_init=mesh.node_coords_init, fixed_ids=bc_nodes, dof=1:        
                           PointConstraints.fix_points(pos, pos_init, fixed_ids, dof))
    
    constraints.append(lambda pos, pos_init=mesh.node_coords_init, fixed_ids=bc_nodes, dof=2:
                           PointConstraints.fix_points(pos, pos_init, fixed_ids, dof))
    
    # constraints.append(lambda pos, r=mesh.node_coords_init, n=bc_nodes:
                            # PointConstraints.apply_displacement(pos, r, n, np.array([0.0, 0.0, -0.01])))

    # add volume constraint
    # for cell in mesh.get_connectivity("tetra"):
    #     constraints.append(lambda pos, c=cell, m=simulator.masses, f=simulator.fixed_nodes, b=bc_nodes:
    #                        VolumeConstraint.volume_constraint(pos, mesh.node_coords_init, *c, 1.0, 0.0, simulator.dt, m, f, b))
    
    
    # higher-order constraints
    # constraints = []
    
    # shape_gradients = np.array([
    #     [-1, -1, -1],
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ])
    # quadrature_weights = np.array([1/6, 1/6, 1/6, 1/6])
    
    # for cell in mesh_reader.get_connectivity("tetra"):
    #     constraints.append(lambda pos, c=cell, r=simulator.rest_lengths, m=simulator.masses, f=simulator.fixed_nodes, s=shape_gradients, q=quadrature_weights:
    #                        SolidSaintVenantElasticity.saint_venant_constraint_high_order(pos, mesh_reader.node_coords_init, c, 1e6, 0.3, 0.0, simulator.dt, m, s, q, f))
    

    # Simulate over 100 steps and save each step
    simulator.simulate(300, constraints, save_path="solid_simultiom")
    print("Simulation complete. Mesh files saved for each step.")