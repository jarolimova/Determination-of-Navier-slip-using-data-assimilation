import argparse
from petsc4py import PETSc
import dolfin as df
from fem_utils import finite_elements
from inlet_bc import init_analytic
from weak_formulation import weak_formulation, marks
from solver_settings import solver_setup
from optimization_setup import SlipPenalty
from read_mesh import read_mesh

print = PETSc.Sys.Print

df.parameters["std_out_all_processes"] = False
df.parameters["ghost_mode"] = "shared_facet"


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("meshname", type=str, help="name of the mesh to be used")
    parser.add_argument(
        "theta",
        type=float,
        help="theta value (Navier slip parameter) from interval [0, 1]",
    )
    parser.add_argument(
        "--element",
        type=str,
        default="mini",
        help="finite element to be used",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="theta",
        help="name of data case (they can be different in velocity, rho, nu and amount of noise)",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=0.1,
        help="velocity of the flow (default: 0.1 m/s)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="slip weight gamma: theta/gamma(1-theta)",
    )
    parser.add_argument(
        "--rho", type=float, default=1050.0, help="fluid density (default: 1050 kg/m^3)"
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=3.71e-6,
        help="kinematic viscosity (default: 3.71e-6 m^2/s)",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="where to save the data",
    )
    args = parser.parse_args()
    return args


def simulate_flow(
    theta: float,
    mesh,
    bndry,
    FE,
    meshname: str,
    gamma: float = 0.25,
    vel: float = 0.8,
    nu: float = 3.71e-6,
    rho: float = 1050.0,
):
    if not FE.stable:
        stab_v = df.Constant(0.01)
        stab_p = df.Constant(0.01)
    else:
        stab_v = df.Constant(0.0)
        stab_p = df.Constant(0.0)
    W = FE.W
    V, Q = W.split()
    V_collapse = V.collapse()
    u_in = df.Function(V_collapse)
    s = df.Function(W)
    const_theta = df.Constant(theta)
    mu = nu * rho

    # Boundary conditions
    in_nitsche = False
    u_in = init_analytic(
        u_in,
        velocity=vel,
        theta=const_theta,
        mu=mu,
        gamma=gamma,
        meshname=meshname,
        data_folder=args.data_folder,
    )
    inflow = df.DirichletBC(W.sub(0), u_in, bndry, marks["in"])
    bcs = [] if in_nitsche else [inflow]
    slip_penalty = SlipPenalty(gamma=gamma)

    F, Jac = weak_formulation(
        const_theta,
        mesh,
        bndry,
        FE,
        s,
        slip_penalty,
        rho=rho,
        mu=mu,
        stab_v=stab_v,
        stab_p=stab_p,
        normal_way="proj",
        linearization_method="picard",
    )
    if Jac is None:
        Jac = df.derivative(F, s)
    problem = df.NonlinearVariationalProblem(F, s, bcs, Jac)
    solver = df.NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["nonlinear_solver"] = "snes"
    solver_setup(prm)
    solver.solve()
    velocity, press = s.split(deepcopy=True)
    return velocity, press


def write_data(velocity, pressure, filepath: str, comm=df.MPI.comm_world):
    velocity.rename("v", "v")
    pressure.rename("p", "p")
    with df.XDMFFile(comm, f"{filepath}_vel.xdmf") as xdmffile:
        xdmffile.write(velocity)
    with df.XDMFFile(comm, f"{filepath}_press.xdmf") as xdmffile:
        xdmffile.write(pressure)
    with df.HDF5File(comm, f"{filepath}_velh5.h5", "w") as h5file:
        h5file.write(velocity, "v")
    with df.HDF5File(comm, f"{filepath}_pressh5.h5", "w") as h5file:
        h5file.write(pressure, "p")


if __name__ == "__main__":
    args = get_args()
    meshpath = f"{args.data_folder}/{args.meshname}"
    comm = df.MPI.comm_world
    mesh, bndry, _ = read_mesh(meshpath, comm=comm)

    ns_element = finite_elements.string_to_element[args.element]
    FE = ns_element(mesh, bndry)
    FE.setup()
    velocity, press = simulate_flow(
        args.theta,
        mesh,
        bndry,
        FE,
        meshname=args.meshname,
        vel=args.velocity,
        nu=args.nu,
        rho=args.rho,
        gamma=args.gamma,
    )
    name = args.name
    filepath = f"{args.data_folder}/{args.meshname}/{args.element}/{name}{int(round(args.theta*1000))}"
    if args.gamma != 0.25:
        filepath += f"_gamma{args.gamma}"
    write_data(velocity, press, filepath=filepath)
