import os
import json
import gmsh
import meshio
import numpy as np
from math import pi, sin, cos
from typing import List

from vmtk_functions import (
    read_surface,
    clip_surface,
    CuttingEquation,
    add_scalar_function_array,
    surface_connectivity,
    remesh_surface,
    cap_surface,
    write_surface,
)


def make_tube_mesh(
    meshname, radius1, radius2, length, edgelength, bend=0.0, extensions=0.0, shift=0.0
):
    """generate surface mesh of a tube in .msh format together with .json file with information on the boundaries

    Args:
        meshname: name of the mesh saved to the folder synthetic_aorta
        radius1: radius of the tube in z=0
        radius2: radius of tube in z=length
        length: length of the mesh in z direction
        edgelength: length of the triangle elements
        bend: shift of the end of the tube (z=length) in y direction (default = 0.0)
        extension: the length of the cylindrical extensions to be added to the geometry (default = 0.0)
        shift: shift of the whole geometry in x direction (default = 0.0)
    """
    folder = "synthetic_aorta"
    if bend == 0.0:
        folder += "/straight_aorta"
    else:
        folder += f"/bent{bend}_aorta"
    if not os.path.exists(f"{folder}/{meshname}"):
        os.makedirs(f"{folder}/{meshname}")
    linspace = np.linspace(-extensions, length + extensions, num=80)
    tube_kwargs = {
        "radius1": radius1,
        "radius2": radius2,
        "length": length,
        "bend": bend,
        "shift": shift,
    }
    generate_geometry(
        f"{folder}/{meshname}/{meshname}",
        edgelength,
        linspace,
        tube_centerline,
        tube_kwargs,
    )
    generate_bc_json(
        [shift, 0.0, 0.0 - extensions],
        [0.0, 0.0, 1.0],
        radius1,
        [shift, bend, length + extensions],
        [0.0, 0.0, 1.0],
        radius2,
        f"{folder}/{meshname}/{meshname}_cuts",
    )


def tube_centerline(z, radius1=0.01, radius2=0.01, length=0.1, bend=0.0, shift=0.0):
    """definition of the geometry for each z based on given radii, length, bend and shift

    Args:
        z: z coordinate
        r1: radius at inlet
        r2: radius at outlet
        length: length of the narrowing part of geometry
        bend: how much does the tube bent (how much is the outlet offset from inlet)
        shift: how much is the tube shifted in the x direction

    Returns:
        x, y, z: x, y, z coords of the centerline at given z coordinate
        r: radius at given z coordinate
        t: tangent at the given z coordinate
    """
    x = shift
    if z <= 0.0:
        r = radius1
        y = 0.0
        t = [0.0, 0.0, 1.0]
    elif z >= length:
        r = radius2
        y = bend
        t = [0.0, 0.0, 1.0]
    else:
        y = bend * 0.5 * (1 - cos(pi * z / length))
        r = radius1 + z / length * (radius2 - radius1)
        t = [0.0, 0.0, 1.0]
    return (x, y, z, r, t)


def generate_geometry(
    name, edgelength, linspace, centerline_function, centerline_kwargs
):
    """generating surface geometry of a bent tube in .msh format

    Args:
        name: path to the file where the surface mesh is supposed to be saved
        edgelength: size of the tetrahedra making up the surface mesh
        linspace: a list of values determining the circle sections
        centerline_function: function defining the shape of the centerline of the geometry
        centerline_kwargs: dictionary containing arguments for centerline_function
    """
    gmsh.initialize()
    engine = gmsh.model.occ
    slices = []
    for a in linspace:
        x, y, z, r, t = centerline_function(a, **centerline_kwargs)
        c = engine.addCircle(x, y, z, r, zAxis=t)
        s = engine.addCurveLoop([c])
        slices.append(c)
    engine.synchronize()
    engine.removeAllDuplicates()
    engine.addThruSections(slices, makeSolid=True)
    engine.synchronize()

    # MARKING OF BOUNDARIES
    # collect entities
    all_surfaces = gmsh.model.getEntities(dim=2)
    all_volumes = gmsh.model.getEntities(dim=3)
    # collect tags
    surface_tags = [surface[1] for surface in all_surfaces]
    volume_tags = [volume[1] for volume in all_volumes]
    print("surface_tags: ", surface_tags)
    # mark volume
    gmsh.model.addPhysicalGroup(dim=3, tags=volume_tags, tag=0)
    for tag, a in zip([2, 3], [linspace[0], linspace[-1]]):
        x, y, z, r, _ = centerline_function(a, **centerline_kwargs)
        coord_min = [cp - 1.1 * r for cp in [x, y, z]]
        coord_max = [cp + 1.1 * r for cp in [x, y, z]]
        box_ents = gmsh.model.getEntitiesInBoundingBox(*coord_min, *coord_max, dim=2)
        tags = [surface[1] for surface in box_ents]
        gmsh.model.addPhysicalGroup(dim=2, tags=tags, tag=tag)
        for t in tags:
            surface_tags.remove(t)
    gmsh.model.addPhysicalGroup(dim=2, tags=surface_tags, tag=1)

    if edgelength is not None:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", edgelength)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", edgelength)
    gmsh.model.mesh.generate(3)
    gmsh.write(f"{name}.msh")
    gmsh.finalize()
    surface = meshio.read(f"{name}.msh")
    surface.write(f"{name}.stl")


def make_arch_mesh(meshname, R, r, edgelength, extensions=0.0, shift=0.0):
    """generate surface mesh of a tube in .msh format together with .json file with information on the boundaries

    Args:
        meshname: name of the mesh saved to the folder /usr/work/jarolimova/synthetic_aorta
        edgelength: length of the triangle elements
        extension: the length of the cylindrical extensions to be added to the geometry (default = 0.0)
        shift: shift of the whole geometry in x direction (default = 0.0)
    """
    folder = "synthetic_aorta/aortic_arch"
    if not os.path.exists(f"{folder}/{meshname}"):
        os.makedirs(f"{folder}/{meshname}")

    gmsh.initialize()
    engine = gmsh.model.occ

    # define a spline curve:
    npts_arc = 40
    npts_ext = round(npts_arc * extensions / (pi * R))
    tangents = []
    points = []
    for j in range(npts_ext):
        x = shift
        y = -R
        z = extensions * (1 - j / npts_ext)
        engine.addPoint(x, y, z, tag=1000 + j)
        points.append(1000 + j)
        tangents = tangents + [0.0, 0.0, -1.0]

    for i in range(npts_arc):
        alpha = i * pi / npts_arc
        x = shift
        y = -R * cos(alpha)
        z = -R * sin(alpha)
        engine.addPoint(x, y, z, tag=1000 + npts_ext + i)
        points.append(1000 + npts_ext + i)
        tangents = tangents + [0.0, sin(alpha), -cos(alpha)]

    for j in range(npts_ext):
        x = shift
        y = R
        z = (j + 1) * extensions / npts_ext
        engine.addPoint(x, y, z, tag=1000 + npts_ext + npts_arc + j)
        points.append(1000 + npts_ext + npts_arc + j)
        tangents = tangents + [0.0, 0.0, 1.0]

    engine.addSpline(pointTags=points, tag=1000, tangents=tangents)
    engine.addWire([1000], 1000)
    disktag = engine.addDisk(xc=shift, yc=-R, zc=0.0, rx=r, ry=r)
    gmsh.model.occ.addPipe([(2, disktag)], wireTag=1000, trihedron="DiscreteTrihedron")
    gmsh.model.occ.remove([(2, disktag)])

    gmsh.model.occ.synchronize()
    if edgelength is not None:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", edgelength)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", edgelength)
    gmsh.model.mesh.generate(3)
    pth = f"{folder}/{meshname}/{meshname}"
    gmsh.write(pth + ".msh")
    gmsh.finalize()
    surface = meshio.read(pth + ".msh")
    surface.write(pth + ".stl")
    generate_bc_json(
        [shift, -R, extensions],
        tangents[:3],
        r,
        [shift, R, extensions],
        tangents[-3:],
        r,
        f"{folder}/{meshname}/{meshname}_cuts",
    )


def generate_bc_json(
    in_point: List[float],
    in_normal: List[float],
    in_radius: float,
    out_point: List[float],
    out_normal: List[float],
    out_radius: float,
    filename="",
):
    """save the boundary data into a json file to a specified location

    Args:
        in_point: centerpoint of the inlet
        in_normal: normal vector at the inlet
        in_radius: radius of the inlet
        out_point: centerpoint of the outlet
        out_normal: normal vector at the outlet
        out_radius: radius of the outlet
        filename: path where the json file is supposed to be saved
    """
    data = {
        "in": {"point": in_point, "normal": in_normal, "radius": in_radius},
        "outs": [{"point": out_point, "normal": out_normal, "radius": out_radius}],
    }
    with open(f"{filename}.json", "w") as jsnfile:
        json.dump(data, jsnfile, indent=4)
    return


def make_shorter_geometries(
    meshname_original: str,
    edgelengths: List[float],
    meshname_new: str,
    in_point: List[float],
    in_normal: List[float],
    in_radius: float,
    out_point: List[float],
    out_normal: List[float],
    out_radius: float,
    folder: str = "",
):
    """clip and remesh geometries to create shorter segments

    Args:
        meshname_original: meshname of the original long geometry
        edgelengths: edge length for remeshing
        meshname_new: name of the short segment
        in_point, in_normal: point and normal specifying the cutting plane at the inlet
        in_radius: radius at the inlet
        out_point, out_normal: point and normal specifying the cutting plane at the outlet
        out_radius: radius at the outlet
        folder: folder containing the meshes
    """
    surface = read_surface(
        os.path.join(folder, meshname_original, f"{meshname_original}.stl")
    )
    cutting_function = CuttingEquation(
        [np.array(in_point), np.array(out_point)],
        [np.array(in_normal), np.array(out_normal)],
        [in_radius, out_radius],
    )
    add_scalar_function_array(surface, cutting_function, "cutting_function")
    clip = clip_surface(surface, name="cutting_function")
    clip_clean = surface_connectivity(clip)
    for edgelength in edgelengths:
        meshname = meshname_new + "_" + str(edgelength).split(".")[1]
        remeshed = remesh_surface(clip_clean, edgelength)
        capped = cap_surface(remeshed)
        final_surface = remesh_surface(capped, exclude_ids=[1], edgelength=edgelength)
        if not os.path.exists(os.path.join(folder, meshname)):
            os.mkdir(os.path.join(folder, meshname))
        generate_bc_json(
            in_point,
            in_normal,
            in_radius,
            out_point,
            out_normal,
            out_radius,
            filename=os.path.join(folder, meshname, f"{meshname}_cuts"),
        )
        write_surface(final_surface, os.path.join(folder, meshname, f"{meshname}.stl"))


if __name__ == "__main__":
    make_tube_mesh(
        "straight_aorta_0075_ext", 0.009, 0.008, 0.1, 0.00075, bend=0.0, extensions=0.01
    )
    make_tube_mesh(
        "straight_aorta_015", 0.009, 0.008, 0.1, 0.0015, bend=0.0, extensions=0.0
    )

    make_tube_mesh(
        "bent_aorta_longer_0075_ext",
        0.01,
        0.008,
        0.12,
        0.00075,
        bend=0.02,
        extensions=0.01,
    )
    make_shorter_geometries(
        "bent_aorta_longer_0075_ext",
        [0.0015, 0.001, 0.0008],
        "clipped_bent_aorta",
        [-7.862483125092954e-06, 0.004623919005535854, 0.037992030806386204],
        [-0.0001309424446212845, 0.21339243296098317, 0.9769665564434991],
        0.009155102025032301,
        [2.8540521366794204e-08, 0.019999385809515966, 0.12835433822976447],
        [2.2632408109427948e-05, -0.0005076523870084613, -0.9999998708884057],
        0.007990414534129629,
        "synthetic_aorta/bent0.02_aorta",
    )

    make_arch_mesh("arch_0075_ext", 0.025, 0.01, 0.00075, shift=0.0, extensions=0.02)
    make_shorter_geometries(
        "arch_0075_ext",
        [0.0015, 0.001, 0.0008],
        "clipped_arch",
        [2.8654308144578553e-07, -0.019944701327024524, -0.014927441204754288],
        [-1.5110618672700407e-05, 0.6007359278542483, -0.799447524704883],
        0.009997947998847878,
        [-1.89923517281413e-06, 0.02500333697599219, 0.008058588569670469],
        [0.00041399988381598184, 0.0008279304577516354, -0.9999995715675348],
        0.010001930154207877,
        "synthetic_aorta/aortic_arch",
    )
