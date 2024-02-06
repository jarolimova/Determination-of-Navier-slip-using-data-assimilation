import dolfin as df
import dolfin_adjoint as da
import generate_normal
import ufl

beta = 1e3
marks = {"wall": 1, "in": 2, "out": 3}


def T(p: df.Function, v: df.Function, mu: df.Constant):
    """Cauchy stress tensor for incompressible fluid
    mu constant -> generates Navier Stokes equations

    Args:
        p: pressure field
        v: velocity field
        mu: dynamic viscosity
    """
    dim = v.geometric_dimension()
    I = df.Identity(dim)
    return -p * I + 2 * mu * df.sym(df.grad(v))


def nitsche_bc(
    eq,
    n,
    ds: df.Measure,
    FE,
    w: df.Function,
    mu: df.Constant,
    nitsche_penalty,
    bc_type: str = "nonsym",
):
    """create penalization terms to enforce the boundary condition defined by the equation eq

    Args:
        eq: equation which is supposed to be fulfilled in weak sense
        n: normal to the boundary on which we are enforcing the boundary condition
        ds: measure on the boundary where the equation is defined
        FE: finite element object
        w: function containing the unknowns
        mu: dynamic viscosity - df.Constant or df.Function
        nitsche_penalty: defined by beta * mu / edgelen
        bc_type: type of the nitsche bc condition sym or nonsym, default: nonsym
    """
    (u, p) = FE.split(w, test=False)
    w_ = df.TestFunction(w.function_space())
    penalty = 0.5 * nitsche_penalty * df.derivative(df.inner(eq, eq), w, w_) * ds
    if bc_type == "nonsym":
        bcpart = (
            df.inner(T(p, u, mu) * n, df.derivative(eq, w, w_)) * ds
            - df.inner(df.derivative(T(p, u, mu) * n, w, w_), eq) * ds
        )
    elif bc_type == "sym":
        bcpart = (
            df.inner(T(p, u, mu) * n, df.derivative(eq, w, w_)) * ds
            + df.inner(df.derivative(T(p, u, mu) * n, w, w_), eq) * ds
        )
    else:
        raise ValueError("Invalid nitsche_type value.")
    return -bcpart + penalty


def ip_stabilization(FE, w, edgelen, rho, alpha_v, alpha_p):
    """ip stabilization terms

    Args:
        FE: finite element object
        w: mixed space function from W = (V, P)
        edgelen: edge length of the mesh
        rho: density
        alpha_v: velocity stabilization weight
        alpha_p: pressure stabilization weight
    """
    a = edgelen * edgelen
    dS = FE.dS(metadata={"quadrature_degree": 4})
    (u, p, v, q) = FE.split(w)
    penalty_v = (
        alpha_v
        * rho
        * df.avg(a)
        * df.inner(df.jump(df.grad(u)), df.jump(df.grad(v)))
        * dS
    )
    penalty_p = (
        alpha_p
        / rho
        * df.avg(a)
        * df.inner(df.jump(df.grad(p)), df.jump(df.grad(q)))
        * dS
    )
    return penalty_v + penalty_p


def weak_formulation(
    theta,
    mesh,
    bndry,
    FE,
    s,
    slip_penalty,
    rho=1050.0,
    mu=3.8955e-3,
    stab_v=0.0,
    stab_p=0.0,
    normal_way="facet",
    linearization_method="newton",
):
    stab = True if stab_v.values()[0] > 0.0 else False
    (u, p, v, q) = FE.split(s)
    ds = df.Measure(
        "ds", subdomain_data=bndry, domain=mesh, metadata={"quadrature_degree": 4}
    )
    dx = df.Measure("dx", domain=mesh, metadata={"quadrature_degree": 4})

    n = dict()
    for i in ["wall", "in", "out"]:
        if normal_way == "proj":
            with da.stop_annotating():
                n[i] = generate_normal.make_normal_projection(
                    mesh, bndry, id=marks[i], type="CG1"
                )
        elif normal_way == "facet":
            n[i] = df.FacetNormal(mesh)
        else:
            raise ValueError("Invalid --normal_way input.")

    u_n = lambda v, n: df.inner(v, n) * n
    u_t = lambda v, n: v - df.inner(v, n) * n
    edgelen = df.MPI.min(mesh.mpi_comm(), mesh.hmin())
    nitsche_penalty = beta * mu / edgelen

    s0 = df.Function(FE.W)
    (u0, p0) = FE.split(s0, test=False)
    # Create and solve constraint system

    F = +df.inner(T(p, u, mu), df.grad(v)) * dx + df.inner(q, df.div(u)) * dx
    F += rho * df.inner(df.grad(u) * u0, v) * dx

    if slip_penalty.use_theta:
        if float(theta) < 1.0:
            F += (
                slip_penalty.theta_to_penalty(theta)
                * df.inner(u_t(u, n["wall"]), u_t(v, n["wall"]))
                * ds(marks["wall"])
            )
        else:
            F += nitsche_bc(
                u_t(u, n["wall"]),
                n["wall"],
                ds(marks["wall"]),
                FE,
                s,
                mu,
                nitsche_penalty,
            )
    else:
        if slip_penalty.penalty_to_theta(float(theta)) < 1.0:
            F += (
                theta
                * df.inner(u_t(u, n["wall"]), u_t(v, n["wall"]))
                * ds(marks["wall"])
            )
        else:
            F += nitsche_bc(
                u_t(u, n["wall"]),
                n["wall"],
                ds(marks["wall"]),
                FE,
                s,
                mu,
                nitsche_penalty,
            )

    F += nitsche_bc(
        u_n(u, n["wall"]), n["wall"], ds(marks["wall"]), FE, s, mu, nitsche_penalty
    )
    F += nitsche_bc(
        u_t(u, n["out"]), n["out"], ds(marks["out"]), FE, s, mu, nitsche_penalty
    )
    if stab:
        F += ip_stabilization(FE, s, edgelen, rho, stab_v, stab_p)
    F += (
        -rho
        * 0.5
        * df.conditional(df.gt(df.inner(u, n["out"]), 0.0), 0.0, 1.0)
        * df.inner(u, n["out"])
        * df.inner(df.inner(u, n["out"]), df.inner(v, n["out"]))
        * ds(marks["out"])
    )

    if linearization_method == "picard":
        J = df.derivative(F, s)
        F = ufl.replace(F, {s0: s})
        J = ufl.replace(J, {s0: s})
    else:  # newton
        F = ufl.replace(F, {s0: s})
        J = df.derivative(F, s)

    return F, J
