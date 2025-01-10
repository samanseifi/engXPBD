import numpy as np
from engxpbd.mesh import Mesh
from engxpbd.simulator import PhysicsSimulator
from engxpbd.constraints.points_constraints import PointConstraints
from engxpbd.constraints.volume_constraints import SolidElasticity, VolumeConstraint

if __name__ == "__main__":
    filepath = "data/cuboid_tetrahedral_mesh_1order.msh"  # Replace with your Gmsh file path
    mesh = Mesh(filepath)
    dt = 0.008
                
    # Print a summary of the mesh
    mesh_summary = f"Mesh Summary:\nNumber of nodes: {len(mesh.node_coords)}\n" + \
                          "\n".join([f"{k}: {len(v)} elements" for k, v in mesh.connectivity.items()])
    print(mesh_summary)

    # Initialize the physics simulator
    simulator = PhysicsSimulator(mesh, dt=dt, num_constraints_iterations=1000)
  
    # Fix the four corners (example indices, adjust as necessary)
    # fixed_nodes = [0, 12, 2, 29, 49, 31, 4, 22, 6]  # Replace with actual corner node indices
    fixed_nodes = [0,17,18,2,54,136,138,58,55,137,139,59,4,39,40,6]
    bc_nodes = [3,29,28,1,56,134,132,52,57,135,133,53,7,51,50,5]
    # fixed_nodes = [0,27,28,29,30,2,104,436,440,444,448,112,105,437,441,445,449,113,106,438,442,446,450,114,107,439,443,447,451,115,4,73,74,75,76,6]
    
    simulator.set_fixed_nodes(fixed_nodes)
    simulator.set_bc_nodes(bc_nodes)
    
    # Initialize SolidElasticity
    elasticity = SolidElasticity(
        youngs_modulus=1e6,
        poisson_ratio=0.49,
        dt=dt,
        masses=simulator.masses,
        fixed_nodes=fixed_nodes,
        bc_nodes=bc_nodes
    )

    # Initialize Lagrange multipliers
    elasticity.initialize_lagrange_multipliers(mesh.get_connectivity("tetra"))
    
    constraints = []
    for cell in mesh.get_connectivity("tetra"):
        constraints.append(lambda pos, c=cell:
                        elasticity.neo_hookean_constraint(pos, mesh.node_coords_init, *c))
    # fixed nodes
    constraints.append(lambda pos, pos_init=mesh.node_coords_init, fixed_ids=fixed_nodes, dof=None:
                           PointConstraints.fix_points(pos, pos_init, fixed_ids, dof))
    
    constraints.append(lambda pos, pos_init=mesh.node_coords_init, fixed_ids=bc_nodes, dof=1:        
                           PointConstraints.fix_points(pos, pos_init, fixed_ids, dof))
    
    constraints.append(lambda pos, pos_init=mesh.node_coords_init, fixed_ids=bc_nodes, dof=2:
                           PointConstraints.fix_points(pos, pos_init, fixed_ids, dof))

    # volume constraint
    volumeconstraint = VolumeConstraint(                
        dt=dt,
        masses=simulator.masses,
        fixed_nodes=fixed_nodes,
        bc_nodes=bc_nodes
    )
    
    volumeconstraint.initialize_lagrange_multipliers(mesh.get_connectivity("tetra"))
    
    for cell in mesh.get_connectivity("tetra"):
        constraints.append(lambda pos, c=cell:
                        volumeconstraint.incompressibility(pos, mesh.node_coords_init, *c))
    

    # Simulate over 100 steps and save each step
    simulator.simulate(150, constraints, elasticity_instance=elasticity, save_path="solid_simultiom")
    print("Simulation complete. Mesh files saved for each step.")