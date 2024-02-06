import os
import sys
import json
import numpy as np
from typing import Optional
from shutil import copy
import dolfin as df

from vmtk_functions import (
    read_surface,
    get_mesh,
    write_mesh,
)


def vmtk_volume_meshing(surface_path: str, meshname: Optional[str] = None):
    """generate volume mesh from a surface mesh

    Args:
        surface_path: path to the surface mesh
        meshname: name of the volume mesh
    """
    surface = read_surface(surface_path)
    folder_path = os.path.dirname(surface_path)
    if meshname is None:
        basename = os.path.basename(surface_path)
        meshname, _ = os.path.splitext(basename)
    print("meshing")
    mesh = get_mesh(surface)
    write_mesh(mesh, f"{folder_path}/vmtk_mesh.vtu")
    write_mesh(mesh, f"{folder_path}/{meshname}.xml")
    print(f"Number of mesh tetrahedra:  {mesh.GetNumberOfCells()}")
    print(f"Number of points: {mesh.GetNumberOfPoints()}")
    return mesh


def read_json(path):
    with open(path, "r") as js:
        data = json.load(js)
        return data


def test_point(x, cut, tol):
    """test if point belongs to a given boundary

    Args:
        x: point being tested
        cut: dictionary containing information about the boundary
        tol: tolerance for distance of the point from the plane
    """
    dist = x - np.array(cut["point"])
    if (
        np.linalg.norm(dist) <= 1.5 * cut["radius"]
        and abs(np.dot(dist, np.array(cut["normal"]))) <= tol
    ):
        return True
    else:
        return False


def mark_boundaries(marks, mesh, json_path, tol):
    """create mesh function containing marks of the boundaries

    Args:
        marks: dictionary containing the boundary markings 
        mesh: dolfin mesh
        json_path: path to the json cuts file
        tol: tolerance for testing whether a point belongs to a boundary
    """
    dim = mesh.topology().dim()
    bndry = df.MeshFunction("size_t", mesh, dim - 1, 0)
    mesh.init()
    cuts = read_json(json_path)
    for f in df.facets(mesh):
        bndry[f] = 0
        if f.exterior():
            bndry[f] = marks["wall"]
            x = f.midpoint().array()[:dim]
            if test_point(x, cuts["in"], tol):
                bndry[f] = marks["in"]
            else:
                for i, out in enumerate(cuts["outs"]):
                    if test_point(x, out, tol):
                        bndry[f] = marks["out"] + i
    return bndry


def write_mesh_h5(folder, name, tol):
    """Mark boundaries on the mesh and save it in h5 format

    Args:
        folder: folder containing the xml mesh
        name: name of the mesh
        tol: tolerance for the boundary markings
    """
    mesh_path = f"{folder}/{name}"
    mesh = df.Mesh(f"{mesh_path}/{name}.xml")
    print("marking boundaries ...")
    bd = mark_boundaries(
        {"wall": 1, "in": 2, "out": 3},
        mesh,
        f"{mesh_path}/{name}_cuts.json",
        tol,
    )
    print("saving marks and h5 mesh ...")
    with df.XDMFFile(mesh.mpi_comm(), f"{mesh_path}/{name}_bnd_marks.xdmf") as meshfile:
        meshfile.write(bd)
    with df.HDF5File(mesh.mpi_comm(), f"{mesh_path}/{name}.h5", "w") as hdf:
        hdf.write(mesh, "/mesh")
        hdf.write(bd, "/boundaries")


def copy_to_data(folder, meshname, datafolder="data"):
    """copy meshes with markings and cuts json file to folder data

    Args:
        folder: path to the folder containing the mesh
        meshname: name of the mesh
        datafolder: path to the folder containing assimilation data
    """
    copy(os.path.join(folder, meshname, meshname + ".h5"), datafolder)
    copy(os.path.join(folder, meshname, meshname + "_cuts.json"), datafolder)
    return


if __name__ == "__main__":
    folder, meshname = sys.argv[1:]
    vmtk_volume_meshing(os.path.join(folder, meshname, meshname + ".stl"))
    write_mesh_h5(folder, meshname, 0.0001)
    copy_to_data(folder, meshname)
