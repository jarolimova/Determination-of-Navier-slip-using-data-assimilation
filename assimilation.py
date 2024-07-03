import os
import csv
import time
from datetime import datetime
from petsc4py import PETSc
import dolfin as df
import dolfin_adjoint as da
from pyadjoint.placeholder import Placeholder

from read_mesh import read_mesh
from solver_settings import solver_setup
import finite_elements
from weak_formulation import weak_formulation, marks
from input_args import get_args, prepare_folder, prepare_files
import inlet_bc
from optimization_setup import ErrorFunctional, Controls, SlipPenalty

print = PETSc.Sys.Print

df.parameters["form_compiler"]["quadrature_degree"] = 4
df.parameters["std_out_all_processes"] = False
df.parameters["ghost_mode"] = "shared_facet"

iteration = 0

nu = da.Constant(3.71e-6)
rho = da.Constant(1050.0)
mu = da.Constant(nu * rho)


def forward(
    theta,
    u_in,
    mesh,
    bndry,
    FE,
    stab_v,
    stab_p,
    slip_penalty,
    normal_way="proj",
    nonlinear_solver="snes",
    picard_weight=0.0,
    init_from_prev=True,
    mu=mu,
    rho=rho,
    annotate=False,
):
    # initialize s and s_init
    s = da.Function(FE.W, name="State", annotate=annotate)
    s_init = da.Function(FE.W, name="Init", annotate=annotate)
    placeholder = Placeholder(s_init)
    # set intial s from the previous solve
    if init_from_prev:
        s.assign(s_init, annotate=annotate)
    # boundary conditions
    inflow = da.DirichletBC(FE.W.sub(0), u_in, bndry, marks["in"], annotate=annotate)
    bcs = [inflow]
    # weak formulation and solve
    F, Jac = weak_formulation(
        theta,
        mesh,
        bndry,
        FE,
        s,
        slip_penalty,
        rho=rho,
        mu=mu,
        stab_v=da.Constant(stab_v, annotate=annotate),
        stab_p=da.Constant(stab_p, annotate=annotate),
        normal_way=normal_way,
        picard_weight=picard_weight,
    )
    problem = da.NonlinearVariationalProblem(F, s, bcs, Jac)
    solver = da.NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["nonlinear_solver"] = nonlinear_solver
    solver_setup(prm)
    solver.solve(annotate=annotate)
    # remember the value for next solve
    placeholder.set_value(s)
    return s


def eval_cb_post(j, m):
    theta, u_in = controls.extract_values(m)
    if theta is None:
        theta = args.init_theta
    global iteration
    st = s_control.tape_value()
    s_control.update(st)
    uu, pp = st.split()
    with da.stop_annotating():
        jreg = error_functional.Jreg(u_in, theta, velocity_avg)
        err = j - jreg
        jalpha = error_functional.J_alpha(u_in)
        jbeta = error_functional.J_beta(u_in, theta, velocity_avg)
    if write_results:
        if df.MPI.rank(df.MPI.comm_world) == 0:
            output_log(iteration, float(theta), err, jreg, j, jalpha, jbeta)
        uu.rename("v", "v")
        pp.rename("p", "p")
        uu_file.write(uu, iteration)
        pp_file.write(pp, iteration)
    if args.picard == iteration:
        picard_control.update(da.Constant(0.0))
        print(f"Picard -> Newton (at the end of iteration {iteration})")
    iteration += 1


def output_log(iteration, theta, jdata, jreg, j, jalpha=0.0, jbeta=0.0):
    print(
        f"iter: {iteration}, theta = {theta}, Jdata = {jdata}, Jreg = {jreg}, J = {j}, J_alpha = {jalpha}, J_beta = {jbeta})"
    )
    with open(output_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([iteration, theta, j, jdata, jreg, jalpha, jbeta])


if __name__ == "__main__":
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"Datetime: {now}")
    initial_time = time.time()
    # setup folder structure based on the args
    args = get_args()
    foldername, output_file = prepare_folder(args)
    print(f"Settings: {args.__dict__}")

    # prepare xdmf and csv files to write to
    write_results = False if args.no_results else True
    if write_results:
        uu_file, pp_file = prepare_files(foldername, output_file)

    # Read mesh
    comm = df.MPI.comm_world
    mesh_path = os.path.join(args.data_folder, args.meshname)
    mesh, bndry, dim = read_mesh(mesh_path, comm)
    with df.XDMFFile(f"{foldername}/bnd_marks.xdmf") as f:
        f.write(bndry)

    # Prepare spaces and initial data for forward
    ns_element = finite_elements.string_to_element[args.element]
    FE = ns_element(mesh, bndry)
    FE.setup()
    W = FE.W
    V_collapse = FE.V.collapse()

    data = da.Function(V_collapse)
    with df.HDF5File(
        comm, f"{mesh_path}/{args.element}/{args.dataname}h5.h5", "r"
    ) as h5file:
        h5file.read(data, "data")
    with da.stop_annotating():
        area_in = da.assemble(da.Constant(1.0) * FE.ds(marks["in"]))
        velocity_avg = (
            -da.assemble(df.inner(data, df.FacetNormal(mesh)) * FE.ds(marks["in"]))
            / area_in
        )

    slip_penalty = SlipPenalty(gamma=args.gamma_star)
    theta = da.Constant(args.init_theta)
    u_in = da.Function(V_collapse, name="Control")
    with da.stop_annotating():
        init_function = getattr(inlet_bc, f"init_{args.init_vin}")
        u_in = init_function(
            u_in,
            data=data,
            epsilon=args.diffusion,
            velocity=velocity_avg,
            theta=args.init_theta,
            mu=mu,
            gamma=args.gamma_star,
            meshname=args.meshname,
            data_folder=args.data_folder,
        )
        with df.XDMFFile(f"{foldername}/vin_init.xdmf") as xdmf:
            xdmf.write(u_in)

    picard_weight = da.Constant(1.0 if args.picard > 0 else 0.0)

    s = forward(
        theta,
        u_in,
        mesh,
        bndry,
        FE,
        args.stab_v,
        args.stab_p,
        slip_penalty,
        normal_way=args.normal,
        nonlinear_solver=args.nonlinear_solver,
        picard_weight=picard_weight,
        init_from_prev=not args.init_with_zero,
        annotate=True,
    )
    u, p = FE.split(s, test=False)

    analytic_profile = inlet_bc.AnalyticProfile.from_json(
        mu, args.meshname, mesh, slip_penalty, data_folder=args.data_folder
    )
    error_functional = ErrorFunctional(
        data, FE.ds, analytic_profile, alpha=args.alpha, beta=args.beta, char_len=0.01
    )
    J = error_functional.J(u, u_in, theta, velocity_avg)

    s_control = da.Control(s)
    picard_control = da.Control(picard_weight)
    controls = Controls(
        theta if not args.no_theta else None,
        u_in if not args.no_vin else None,
        slip_penalty=slip_penalty,
    )
    m = controls.m
    Jhat = da.ReducedFunctional(J, m, eval_cb_post=eval_cb_post)

    before_optimization_time = time.time()
    # Minimize
    m_opt = da.minimize(
        Jhat,
        bounds=controls.bounds,
        method="L-BFGS-B",
        options={
            "ftol": args.ftol,
            "gtol": args.gtol,
            "disp": df.MPI.rank(df.MPI.comm_world) == 0,
        },
    )
    after_optimization_time = time.time()

    theta_opt, u_in_opt = controls.extract_values(m_opt)
    if u_in_opt is None:
        u_in_opt = u_in
    else:
        with df.HDF5File(comm, f"{foldername}/u_in_opt_h5.h5", "w") as hdf:
            hdf.write(u_in_opt, "u_in")
    if theta_opt is None:
        theta_opt = theta
    else:
        print("Optimal Theta: ", float(theta_opt))

    s_opt = forward(
        theta_opt,
        u_in_opt,
        mesh,
        bndry,
        FE,
        args.stab_v,
        args.stab_p,
        slip_penalty,
        normal_way=args.normal,
        nonlinear_solver=args.nonlinear_solver,
        picard_weight=1.0,
        init_from_prev=not args.init_with_zero,
    )

    if write_results:
        # Rename and write results
        vel, press = s_opt.split()
        vel.rename("v", "v")
        press.rename("p", "p")
        data.rename("v", "v")
        with df.XDMFFile(f"{foldername}/vel.xdmf") as xdmf:
            xdmf.write(vel)
        with df.XDMFFile(f"{foldername}/press.xdmf") as xdmf:
            xdmf.write(press)
        with df.XDMFFile(f"{foldername}/data.xdmf") as xdmf:
            xdmf.write(data)
        with df.HDF5File(comm, f"{foldername}/w_opt_h5.h5", "w") as hdf:
            hdf.write(s_opt, "w")
    end_time = time.time()

    print(f"Setup time: {round(before_optimization_time-initial_time, 3)}")
    print(
        f"Optimization time: {round(after_optimization_time-before_optimization_time, 3)}"
    )
    print(f"Finalization time: {round(end_time-after_optimization_time, 3)}")
    print(50 * "=")
    print(f"Total time: {round(end_time-initial_time, 3)}")
