from petsc4py import PETSc
import numpy as np
from abc import ABC
import dolfin as df

__all__ = [
    "P1P1",
    "TaylorHood",
    "MiniElement",
    "string_to_element",
]


class NSElement(ABC):
    """Abstract base class for the finite element objects"""

    def __init__(self, mesh: df.Mesh, bndry: df.MeshFunction, *args, **kwargs):
        """Initialize the finite element object

        Args:
            mesh: FEM computational mesh (df.Mesh object)
            bndry: mesh function with boundary marks
        """
        self.mesh = mesh
        self.bndry = bndry
        self.c = mesh.ufl_cell()
        self.ds = df.Measure("ds", subdomain_data=bndry, domain=mesh)
        self.dS = df.Measure("dS", subdomain_data=bndry, domain=mesh)
        self.dx = df.Measure("dx", domain=mesh)
        self.stable = True
        self.W = None

    def setup(self):
        """Setup self.V and self.P as the appropriate velocity and pressure function spaces as well as some PetSc magic.
        Should be run after initialization.
        """
        self.V = self.W.sub(0)
        self.P = self.W.sub(1)
        self.vdofs = np.array(self.W.sub(0).dofmap().dofs())
        self.pdofs = np.array(self.W.sub(1).dofmap().dofs())

        ## Setup PETSc field decomposition
        ## Create PetscSection ##
        self.section = PETSc.Section().create()
        self.section.setNumFields(2)
        self.section.setChart(0, self.W.dim())
        offset = np.min(self.W.dofmap().dofs())

        self.section.setFieldName(0, "velocity")
        self.section.setFieldComponents(0, 1)
        self.section.setFieldName(1, "pressure")
        self.section.setFieldComponents(1, 1)

        ## Assign dof to PetscSection ##
        for i in range(self.mesh.geometry().dim()):
            pts = np.array(self.W.sub(0).sub(i).dofmap().dofs())
            for p in np.nditer(pts):  # , flags=['zerosize_ok']):
                self.section.setDof(p - offset, 1)
                self.section.setFieldDof(p - offset, 0, 1)

        pts = np.array(self.W.sub(1).dofmap().dofs())
        for p in np.nditer(pts):
            self.section.setDof(p - offset, 1)
            self.section.setFieldDof(p - offset, 1, 1)

        ## Create DM and assign PetscSection ##
        self.section.setUp()
        self.dm = PETSc.DMShell().create()
        self.dm.setDefaultSection(self.section)
        self.dm.setUp()

    def split(self, w: df.Function, test: bool = True):
        """Split function from mixed space W into two functions from v and p.
        If test is set to True, create test functions v_ and p_ as well.
        The split functions should be used for definition of weak formulation.

        Args:
            w: dolfin function from space W to be split into v and p.
            test: switch determining whether to create test functions or not.
        """
        t = df.split(w)
        v = t[0]
        p = t[1]
        if test:
            t_ = df.TestFunctions(self.W)
            v_ = t_[0]
            p_ = t_[1]
            return (v, p, v_, p_)
        else:
            return (v, p)

    def extract(self, w: df.Function):
        """Extract v and p components from w for to be able to save them.

        Args: dolfin function from space W from which v and p are to be extracted.
        """
        t = w.split(True)
        v = t[0]
        p = t[1]
        return (v, p)

    def stab(self, w: df.Function):
        w_ = df.TestFunction(w.function_space())
        return df.Constant(0.0) * df.inner(w, w_) * self.dx


class P1P1(NSElement):
    """P1P1 element - piecewise linear velocities and pressures
    not stable element (does not satisfy the inf-sup condition)
    need stabilization to work properly: usually supg, ip in v and p works as well
    number of dofs: v_dofs = dim * #points, p_dofs = #points
    """

    def __init__(self, mesh: df.Mesh, bndry: df.MeshFunction, *args, **kwargs):
        super().__init__(mesh, bndry)
        V = df.VectorElement("CG", self.c, 1)
        P = df.FiniteElement("CG", self.c, 1)
        R = df.FiniteElement("R", self.c, 0)
        E = df.MixedElement([V, P])
        self.W = df.FunctionSpace(mesh, E)
        self.stable = False


class TaylorHood(NSElement):
    """Taylor Hood element - piecewise quadratic in velocities and piecewise linear in pressures
    stable element (satisfy inf-sup condition)
    standart element for incompressible Navier Stokes
    number of dofs: v_dofs = dim * (#points + #faces), p_dofs = #points
    """

    def __init__(self, mesh: df.Mesh, bndry: df.MeshFunction, *args, **kwargs):
        super().__init__(mesh, bndry)
        V = df.VectorElement("CG", self.c, 2)
        P = df.FiniteElement("CG", self.c, 1)
        E = df.MixedElement([V, P])
        self.W = df.FunctionSpace(mesh, E)


class MiniElement(NSElement):
    """Mini Element - piecewise linear + bubble in velocities and piecewise linear in pressures
    stable element (satisfy inf-sup condition)
    standart element for incompressible Navier Stokes
    number of dofs: v_dofs = dim * (#points + #cells), p_dofs = #points
    """

    def __init__(self, mesh: df.Mesh, bndry: df.MeshFunction, *args, **kwargs):
        super().__init__(mesh, bndry)

        d = self.mesh.geometry().dim()
        U = df.FiniteElement("CG", self.c, 1)
        B = df.FiniteElement("Bubble", self.c, d + 1)
        P = df.FiniteElement("CG", self.c, 1)
        V = df.VectorElement(df.NodalEnrichedElement(U, B))
        E = df.MixedElement([V, P])
        self.W = df.FunctionSpace(mesh, E)


string_to_element = {
    "p1p1": P1P1,
    "th": TaylorHood,
    "mini": MiniElement,
}
