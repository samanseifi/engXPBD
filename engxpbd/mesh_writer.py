import meshio

class MeshWriter:
    @staticmethod
    def save_mesh(mesh_reader, output_filepath, file_format="vtk"):
        """
        Save the mesh to a file.

        Parameters:
        - output_filepath (str): Path to save the mesh.
        - file_format (str): Format to save the mesh (default is VTK).
        """
        meshio.write_points_cells(
            output_filepath,
            mesh_reader.node_coords,
            mesh_reader.mesh.cells,
            file_format=file_format
        )