from petsc4py import PETSc
import numpy as np
import dolfin as df
import dolfin_adjoint as da

print = PETSc.Sys.Print
opts = PETSc.Options()


def my_solve(A, x, b):
    ksp = PETSc.KSP().create()
    ksp.setType("preonly")
    pc = PETSc.PC().create()
    pc.setType("lu")
    pc.setFactorSolverType("superlu_dist")
    ksp.setPC(pc)

    ksp.setOperators(da.as_backend_type(A).mat())
    ksp.setTolerances(rtol=1e-20, atol=1e-10, max_it=1000)

    print("Solving with:", ksp.getType(), pc.getType(), flush=True)
    # Solve
    bb = da.as_backend_type(b).vec()
    xx = da.as_backend_type(x).vec()
    ksp.solve(bb, xx)
    print(
        "Converged reason:",
        ksp.getConvergedReason(),
        " in ",
        ksp.getIterationNumber(),
        "iterations",
        flush=True,
    )


def normalize_vector(n):
    # really normalize the vector n at each dof - assumes 3D fenics vector function on input
    V = n.function_space()
    nx_dofs = V.sub(0).dofmap().dofs()
    ny_dofs = V.sub(1).dofmap().dofs()
    nz_dofs = V.sub(2).dofmap().dofs()

    nv = da.as_backend_type(n.vector()).vec()

    nx = nv[nx_dofs]
    ny = nv[ny_dofs]
    nz = nv[nz_dofs]
    dn = np.sqrt(nx * nx + ny * ny + nz * nz)

    nx = np.divide(nx, dn, where=(dn > 0.0))
    ny = np.divide(ny, dn, where=(dn > 0.0))
    nz = np.divide(nz, dn, where=(dn > 0.0))
    n.vector().update_ghost_values()


# compute normal vector on boundary marked by id, by projection of FacetNormal to FE space
def make_normal_projection(mesh, bndry, id=None, type="CG1"):
    if type == "DG0":
        degree = 1
        ve = df.VectorElement("CR", mesh.ufl_cell(), 1)
        e = df.FiniteElement("CR", mesh.ufl_cell(), 1)
    elif type == "CG1":
        degree = 1
        ve = df.VectorElement("CG", mesh.ufl_cell(), 1)
        e = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    else:
        raise ValueError("Invalid normal type.")

    V = df.FunctionSpace(mesh, e)
    VV = df.FunctionSpace(mesh, ve)

    n = df.FacetNormal(mesh)
    ds = df.Measure("ds", subdomain_data=bndry, domain=mesh)

    u, v = df.TrialFunction(V), df.TestFunction(V)
    nn = df.Function(V)
    vn = df.Function(VV)

    a = u * v * ds(id) + da.Constant(0.0) * df.avg(u) * df.avg(v) * df.dS
    A = da.assemble(a, keep_diagonal=True)
    A.ident_zeros()

    ksp = PETSc.KSP().create()
    ksp.setType("preonly")
    pc = PETSc.PC().create()
    pc.setType("lu")
    pc.setFactorSolverType("superlu_dist")
    ksp.setPC(pc)

    ksp.setOperators(da.as_backend_type(A).mat())
    ksp.setTolerances(rtol=1e-20, atol=1e-10, max_it=1000)

    # Solve
    # make it faster by computing each component separately (reuse mumps LU factors)
    for i in [0, 1, 2]:
        L = df.inner(n[i], v) * ds(id)
        b = da.assemble(L)
        nn.vector().zero()

        bb = da.as_backend_type(b).vec()
        xx = da.as_backend_type(nn.vector()).vec()
        ksp.solve(bb, xx)
        nn.vector().update_ghost_values()

        dofs = VV.sub(i).dofmap().dofs()
        df.as_backend_type(vn.vector()).vec()[dofs] = df.as_backend_type(
            nn.vector()
        ).vec()
        vn.vector().update_ghost_values()

    vn.vector().apply("insert")
    fvn = da.interpolate(vn, df.VectorFunctionSpace(mesh, "CG", 1))
    return fvn


# compute normal vector on the boundary taged with id, using distance function
def make_normal_distance(mesh, bndry, id=None, type="CG1"):
    print(f"Computing {type} normal.... {id}")
    if type == "DG0":
        degree = 1
        ve = df.VectorElement("CR", mesh.ufl_cell(), 1)
        e = df.FiniteElement("CR", mesh.ufl_cell(), 1)
    elif type == "CG1":
        degree = 1
        ve = df.VectorElement("CG", mesh.ufl_cell(), 1)
        e = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    elif type == "CG2":
        degree = 2
        ve = df.VectorElement("CG", mesh.ufl_cell(), 2)
        e = df.FiniteElement("CG", mesh.ufl_cell(), 2)
    else:
        exit()

    # compute the distance function d
    d = distance_eikonal(mesh, bndry, id, degree=degree)

    V = df.FunctionSpace(mesh, e)
    VV = df.FunctionSpace(mesh, ve)

    n = df.FacetNormal(mesh)
    ds = df.Measure("ds", subdomain_data=bndry, domain=mesh)

    u, v = df.TrialFunction(V), df.TestFunction(V)
    nn = da.Function(V)
    vn = da.Function(VV)

    a = df.inner(u, v) * da.dx
    A = da.assemble(a)

    ksp = PETSc.KSP().create()
    ksp.setType("preonly")
    pc = PETSc.PC().create()
    pc.setType("lu")
    pc.setFactorSolverType("superlu_dist")
    ksp.setPC(pc)

    ksp.setOperators(da.as_backend_type(A).mat())
    ksp.setTolerances(rtol=1e-20, atol=1e-10, max_it=1000)
    ksp.setFromOptions()

    print("Solving with:", ksp.getType(), pc.getType(), flush=True)
    # Solve
    # make it faster by computing each component separately (reuse mumps LU factors)
    for i in [0, 1, 2]:
        L = df.inner(-df.grad(d)[i], v) * da.dx
        b = da.assemble(L)
        nn.vector().zero()

        bb = da.as_backend_type(b).vec()
        xx = da.as_backend_type(nn.vector()).vec()
        ksp.solve(bb, xx)
        print(
            "Converged reason:",
            ksp.getConvergedReason(),
            " in ",
            ksp.getIterationNumber(),
            "iterations",
            flush=True,
        )

        dofs = VV.sub(i).dofmap().dofs()
        vn.vector().vec()[dofs] = nn.vector().vec()

    normalize_vector(vn)
    print(f"done.")
    return vn


# compute distance function from facets marked with tag, represent the result as CG(degree) function
def distance_eikonal(mesh, bndry, tag, degree=1):
    """Solve Eikonal equation to get distance from tagged facets"""
    element = df.FiniteElement("CG", mesh.ufl_cell(), degree)
    V = df.FunctionSpace(mesh, element)
    bc = da.DirichletBC(V, da.Constant(0.0), bndry, tag)

    u, v = df.TrialFunction(V), df.TestFunction(V)
    d = da.Function(V)
    f = da.Constant(1.0)

    # Smooth initial guess
    a = df.inner(df.grad(u), df.grad(v)) * da.dx
    L = df.inner(f, v) * da.dx
    A, b = da.assemble_system(a, L, bc)
    my_solve(A, d.vector(), b)

    # Eikonal equation with stabilization
    eps = da.Constant(1e-1 * mesh.hmin())
    F = (df.sqrt(df.inner(df.grad(d), df.grad(d))) - f) * v * da.dx + eps * df.inner(
        df.grad(d), df.grad(v)
    ) * da.dx

    solver_parameters = {
        "nonlinear_solver": "snes",
        "snes_solver": {
            "maximum_iterations": 200,
            "absolute_tolerance": 1e-8,
            "relative_tolerance": 1e-20,
            "linear_solver": "superlu_dist",
            "preconditioner": "lu",
            "method": "newtontr",
        },
    }
    da.solve(F == 0, d, bc, solver_parameters=solver_parameters)
    return d
