import sys
import dolfin_adjoint as da
import dolfin as df


def read_mesh(mesh_path, comm=df.MPI.comm_world):
    mesh = da.Mesh()
    with df.HDF5File(comm, f"{mesh_path}.h5", "r") as hdf:
        hdf.read(mesh, "/mesh", False)
        dim = mesh.geometry().dim()
        bndry = df.MeshFunction("size_t", mesh, dim - 1, 0)
        hdf.read(bndry, "/boundaries")
    mesh.init()
    return mesh, bndry, dim

def inspect_mesh(meshname, latex=False):
    mesh_path = f"/usr/work/jarolimova/assimilation/data/{meshname}"
    mesh, bndry, dim = read_mesh(mesh_path)
    p1p1_dofs = 4*mesh.num_vertices()
    mini_dofs = 4*mesh.num_vertices()+3*mesh.num_cells()
    th_dofs = 4*mesh.num_vertices()+3*mesh.num_facets()
    print(f"MESH: {meshname}")
    if latex:
        print("LateX tabular (vertices, faces, cells, p1p1dofs, minidofs, thdofs):")
        print(f"{mesh.num_vertices()} & {mesh.num_facets()} & {mesh.num_cells()}")
    else:
        print("Vertices, Facets (3D -> faces or 2D -> edges), Cells:")
        print(mesh.num_vertices(),",", mesh.num_facets(), ",", mesh.num_cells(), "\n")

        print(f"P1P1 dofs: {p1p1_dofs}")
        print(f"MINI dofs: {mini_dofs}")
        print(f"TH dofs:   {th_dofs}")
    print(50*"=")

if __name__ == "__main__":
    input = sys.argv
    if len(input) > 1:
        print(input[1:])
        for meshname in input[1:]:
            inspect_mesh(meshname)
    else:
        print("Provide meshnames of meshes that are to be inspected.")