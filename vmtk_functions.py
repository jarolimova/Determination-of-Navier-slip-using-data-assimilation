import numpy as np
from typing import List, Callable
from vmtk import vmtkscripts
from vtk import vtkPolyData, vtkDataSet, vtkUnstructuredGrid
from vtk.util.numpy_support import numpy_to_vtk


def read_surface(spath: str) -> vtkPolyData:
    sreader = vmtkscripts.vmtkSurfaceReader()
    sreader.InputFileName = spath
    sreader.Execute()
    return sreader.Surface


class CuttingEquation:
    """Function R^3 -> R whose zeroth levelset are local planes each define using point, normal and radius

    ni = normals, pi = points, ri = radii:
    f(x) = ni.(x-pi)  if |x-pi| <= ri for some i
         = 10       otherwise

    Args:
        points: list of points (numpy arrays) in R^3 defining the planes
        normals: list of normals (numpy arrays) in R^3 defining the planes
        radii: list of floats - each define the largest sphere (with center in point) inscribed to the artery
        radius_factor: float scales radii so that the sphere contains the cut plane
    """

    def __init__(
        self,
        points: List[np.ndarray],
        normals: List[np.ndarray],
        radii: List[float],
        radius_factor: float = 1.5,
    ):
        assert (
            len(points) == len(normals) == len(radii)
        ), "Points, normals and radii must have equal length"
        self.normals = normals
        self.points = points
        self.radii = [radius_factor * radius for radius in radii]

    def __call__(self, x: np.ndarray) -> float:
        value = 10.0
        x = np.array(x)
        for i in range(len(self.points)):
            distance = np.linalg.norm(x - self.points[i])
            if self.radii[i] is None or distance <= self.radii[i]:
                d = np.dot(self.points[i], self.normals[i])
                value = np.dot(self.normals[i], x) - d
        return value


def add_scalar_function_array(
    obj: vtkDataSet, function: Callable, name: str
) -> vtkDataSet:
    """add scalar function array to a vtkDataSet object

    vtkDataSet - vtkPolyData/vtkUstructuredGrid/...

    Args:
        function: scalar function which can evaluate on space coordinates (array of 3 values)
        name: name of the PointDataArray (as displayed in Paraview)
    """
    points = np.array(obj.GetPoints().GetData())
    arr = np.array([function(point) for point in points])
    array = numpy_to_vtk(arr, deep=True)
    array.SetName(name)
    dataset = obj.GetPointData()
    dataset.AddArray(array)
    dataset.Modified()
    return obj


def clip_surface(
    surface: vtkPolyData, name: str = "", value: float = 0.0
) -> vtkPolyData:
    """Clip surface using array values or interactively

    Args:
        name: name of the clip array
            "" -> interactive clipping
    """
    clipper = vmtkscripts.vmtkSurfaceClipper()
    clipper.Surface = surface
    if name != "":
        clipper.Interactive = 0
        clipper.ClipArrayName = name
        clipper.ClipValue = value
    clipper.Execute()
    return clipper.Surface


def surface_connectivity(surface: vtkPolyData) -> vtkPolyData:
    """Extract largest connected part of the surface"""
    conn = vmtkscripts.vmtkSurfaceConnectivity()
    conn.Surface = surface
    conn.Execute()
    return conn.Surface


def remesh_surface(
    surface: vtkPolyData,
    edgelength: float = 1.0,
    edgelength_array: str = "",
    factor: float = 1.0,
    minedge: float = 0.0,
    maxedge: float = 1e16,
    exclude_ids: List[int] = [],
    preserve_edges: bool = False,
) -> vtkPolyData:
    """Remesh surface with quality triangles of the given size

    Args:
        edgelength_array: used (when specified) to determine size of elements based on their position
            overrides uniform edgelength specified by edgelength parameter!
        factor: usually used for scaling of the sizearray
        preserve_edges: whether the edges should be remeshed or not - creates overlapped triangles if capped afterwards!!!
    """
    # possibility to add more parameters!
    remesh = vmtkscripts.vmtkSurfaceRemeshing()
    remesh.Surface = surface
    if edgelength_array == "":
        remesh.ElementSizeMode = "edgelength"
        remesh.TargetEdgeLength = edgelength
    else:
        remesh.ElementSizeMode = "edgelengtharray"
        remesh.TargetEdgeLengthArrayName = edgelength_array
    remesh.TargetEdgeLengthFactor = factor
    remesh.MaxEdgeLength = maxedge
    remesh.MinEdgeLength = minedge
    # remesh.InternalAngleTolerance = 0.2
    if exclude_ids != []:
        remesh.ExcludeEntityIds = exclude_ids
        remesh.CellEntityIdsArrayName = "CellEntityIds"
    if preserve_edges:
        remesh.PreserveBoundaryEdges = 1
    remesh.Execute()
    return remesh.Surface


def cap_surface(surface: vtkPolyData) -> vtkPolyData:
    """Add cap to the holes of the surface"""
    capper = vmtkscripts.vmtkSurfaceCapper()
    capper.Surface = surface
    capper.Interactive = 0
    capper.Method = "simple"
    capper.Execute()
    return capper.Surface


def write_surface(surface: vtkPolyData, filename: str):
    """Save surface to file"""
    swriter = vmtkscripts.vmtkSurfaceWriter()
    swriter.Surface = surface
    swriter.OutputFileName = filename
    swriter.Execute()
    return


def get_mesh(
    surface: vtkPolyData,
    remesh_caps: bool = True,
    cap_edgelength: float = 1.0,
) -> vtkUnstructuredGrid:
    """Generate mesh from surface

    Surface -> Mesh
    """
    mesher = vmtkscripts.vmtkMeshGenerator()
    mesher.Surface = surface
    mesher.TargetEdgeLength = cap_edgelength
    if remesh_caps:
        mesher.RemeshCapsOnly = 1
    else:
        mesher.SkipRemeshing = 1
    mesher.Execute()
    return mesher.Mesh


def write_mesh(mesh: vtkUnstructuredGrid, filename: str):
    """Save mesh to a file"""
    mwriter = vmtkscripts.vmtkMeshWriter()
    mwriter.Mesh = mesh
    mwriter.OutputFileName = filename
    mwriter.Compressed = 0
    mwriter.Execute()
    return
