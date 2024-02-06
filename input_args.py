import os
import csv
import argparse
from pathlib import Path
from dolfin import MPI, XDMFFile


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("meshname", type=str, help="name of the mesh to be used")
    parser.add_argument("dataname", type=str, help="name of the data")
    parser.add_argument(
        "--init_vin",
        type=str,
        default="analytic",
        help="u_in initial value, default: analytic, other options: zero, data",
    )
    parser.add_argument(
        "--init_theta",
        type=float,
        default=0.75,
        help="theta initial value, default: 0.75",
    )
    parser.add_argument(
        "--no_vin", action="store_true", help="turn off v_in as a control variable"
    )
    parser.add_argument(
        "--no_theta", action="store_true", help="turn off theta as a control variable"
    )
    parser.add_argument(
        "--element",
        type=str,
        default="p1p1",
        help="finite element to be used, default: p1p1, other options: mini, th",
    )
    parser.add_argument(
        "--stab_v",
        type=float,
        default=0.01,
        help="set the stabilization penalty - if alpha > 0, the stabilization is applied, default: 0.01",
    )
    parser.add_argument(
        "--stab_p",
        type=float,
        default=None,
        help="set the stabilization penalty - if alpha > 0, the stabilization is applied, default: stab_v for p1p1 element and 0.0 for all other elements",
    )
    parser.add_argument(
        "--picard",
        action="store_true",
        help="use picard iterations instead of newton",
    )
    parser.add_argument(
        "--init_with_zero",
        action="store_true",
        help="turn off using the previous optimization iteration solution for the first nonlinear solver iteration solve and set it to zero instead",
    )
    parser.add_argument(
        "--normal",
        type=str,
        default="proj",
        help=", default: proj, other options: facet",
    )
    parser.add_argument(
        "--nonlinear_solver",
        type=str,
        default="snes",
        help="nonlinear solver, default: snes, other options: newton",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default="1e-3",
        help="the regularization weight of H1 norm of inflow, default: 1e-3",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default="0.1",
        help="the regularization weight of |v-v(theta)| regularization, default: 0.1",
    )
    parser.add_argument(
        "--gamma_star",
        type=float,
        default="0.25",
        help="slip weight gamma star, default: 0.25",
    )
    parser.add_argument(
        "--diffusion",
        type=float,
        default="1e-6",
        help="amount of diffusion in case of smoothed initial condition, default: 1e-6",
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=1e-7,
        help="tolerance for termination of the minimization, f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol",
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=1e-5,
        help="tolerance for termination of the minimization, max{|proj g_i | i = 1, ..., n} <= gtol",
    )
    parser.add_argument(
        "--no_results", action="store_true", help="turn off rewritting of results"
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="results",
        help="Location of results folder",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="Location of results folder",
    )
    args = parser.parse_args()
    return args


def prepare_folder(args):
    if args.stab_p is None:
        args.stab_p = args.stab_v if args.element == "p1p1" else 0.0
    element_foldername = f"{args.element}_stab{args.stab_v}_{args.stab_p}"
    inits_foldername = f"{args.init_vin}_{args.init_theta}"
    discretization_foldername = args.normal
    if args.gamma_star != 1.0:
        discretization_foldername += f"_{args.gamma_star}"
    J_foldername = f"alpha{args.alpha}"
    if args.beta != 0.0:
        J_foldername += f"_beta{args.beta}"
    nonlinear_solver = args.nonlinear_solver
    if args.picard:
        nonlinear_solver += "_picard"
    if args.no_vin and not args.no_theta:
        optional = "only_theta"
    elif args.no_theta and not args.no_vin:
        optional = "only_uin"
    elif args.no_theta and args.no_vin:
        optional = "no_control"
    else:
        optional = ""
    foldername = folder_path(
        args.results_folder,
        args.meshname,
        args.dataname,
        element_foldername,
        inits_foldername,
        discretization_foldername,
        J_foldername,
        nonlinear_solver,
        optional=optional,
    )
    output_file = f"{foldername}/output.csv"
    return foldername, output_file


def folder_path(
    results_folder, meshname, dataname, element, init, normal, J, solver, optional=""
):
    if optional == "":
        setup = normal
    else:
        setup = f"{normal}_{optional}"
    foldername = (
        f"{results_folder}/{meshname}/{dataname}/{element}/{init}/{setup}/{J}/{solver}"
    )
    return foldername


def prepare_files(foldername, output_file):
    Path(foldername).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(foldername):
        if MPI.rank(MPI.comm_world) == 0:
            if os.path.isfile(f"{foldername}/{f}"):
                os.remove(f"{foldername}/{f}")
    uu_file = XDMFFile(f"{foldername}/u.xdmf")
    pp_file = XDMFFile(f"{foldername}/p.xdmf")
    uu_file.parameters["flush_output"] = True
    pp_file.parameters["flush_output"] = True
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "theta", "J", "Jdata", "Jreg", "Jalpha", "Jbeta"])
    return uu_file, pp_file


if __name__ == "__main__":
    args = get_args()
    foldername, output_file = prepare_folder(args)
    print(foldername)
