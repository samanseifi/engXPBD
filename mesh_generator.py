import gmsh

class GeometryMesher3D:
    def __init__(self):
        """
        Initialize the mesher for 3D tetrahedral meshes.
        """
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

    def generate_cuboid(self, length=1.0, width=1.0, height=1.0):
        """
        Generate a cuboid geometry.

        Args:
            length (float): Length of the cuboid.
            width (float): Width of the cuboid.
            height (float): Height of the cuboid.
        """
        gmsh.model.add("Cuboid")
        gmsh.model.occ.addBox(0, 0, 0, length, width, height)
        gmsh.model.occ.synchronize()

    def mesh_geometry(self, element_size=0.1):
        """
        Mesh the generated geometry with tetrahedral elements.

        Args:
            element_size (float): Desired mesh element size.
        """
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), element_size)
        gmsh.model.mesh.setTransfiniteSurface(1) # making structure mesh
        gmsh.model.mesh.setTransfiniteSurface(2) # making structure mesh
        gmsh.model.mesh.setTransfiniteSurface(3) # making structure mesh
        gmsh.model.mesh.setTransfiniteSurface(4) # making structure mesh

        gmsh.model.mesh.setTransfiniteSurface(5) # making structure mesh
        gmsh.model.mesh.setTransfiniteSurface(6) # making structure mesh
        
        # gmsh.option.setNumber("Mesh.ElementOrder", 2)  # Set elements to second order
        # gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Use Delaunay meshing for 3D
        # gmsh.option.setNumber("Mesh.Optimize", 1)  # Enable optimization of the mesh
        gmsh.model.mesh.generate(3)

    def write_mesh(self, filename="tetrahedral_mesh.msh"):
        """
        Write the generated tetrahedral mesh to a file.

        Args:
            filename (str): Output file name for the mesh.
        """
        gmsh.write(filename)

    def finalize(self):
        """
        Finalize and clean up the Gmsh API.
        """
        gmsh.finalize()


# Example Usage
if __name__ == "__main__":
    mesher = GeometryMesher3D()

    # Generate geometry (cuboid in this case)
    length = 1.0
    width = 1.0
    height = 4.0
    mesher.generate_cuboid(length=length, width=width, height=height)

    # Mesh the geometry
    element_size = 0.4
    mesher.mesh_geometry(element_size=element_size)

    # Write the mesh to a file
    mesher.write_mesh("cuboid_tetrahedral_mesh_1order.msh")

    # Finalize the mesher
    mesher.finalize()
