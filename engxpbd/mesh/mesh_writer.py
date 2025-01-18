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
       
        if hasattr(mesh_reader, "triangles"):
            cells = [("triangle", mesh_reader.triangles)]
        else:
            cells = mesh_reader.mesh.cells  # Use the meshio cells directly      
        
        meshio.write_points_cells(
            output_filepath,
            mesh_reader.node_coords,
            cells,
            file_format=file_format
        )

    @staticmethod
    def save_mesh_every_n_steps(mesh_reader, output_directory, step, interval, file_format="vtk"):
        """
        Save the mesh at specified intervals.

        Parameters:
        - mesh_reader: Mesh reader object containing the mesh data.
        - output_directory (str): Directory to save the mesh files.
        - step (int): Current simulation step.
        - interval (int): Interval at which to save the mesh.
        - file_format (str): Format to save the mesh (default is VTK).
        """
        if step % interval == 0:
            output_filepath = f"{output_directory}/mesh_step_{step}.{file_format}"
            MeshWriter.save_mesh(mesh_reader, output_filepath, file_format=file_format)