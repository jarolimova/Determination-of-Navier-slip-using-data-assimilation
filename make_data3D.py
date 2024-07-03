import os
import argparse
from math import ceil
import numpy as np
import dolfin as df
from typing import Callable

from fem_utils import finite_elements
from read_mesh import read_mesh

from petsc4py import PETSc

print = PETSc.Sys.Print

df.parameters["std_out_all_processes"] = False
df.parameters["ghost_mode"] = "shared_facet"


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("meshname", type=str, help="name of the mesh to be used")
    parser.add_argument(
        "--meshname_original",
        type=str,
        default="",
        help="mesh on which the flow should be simulated before interpolated to the final mesh (one including extensions for example), dafault empty string mean take the same mesh",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="theta",
        help="name of data case (they can be different in velocity, rho, nu and amount of noise)",
    )
    parser.add_argument(
        "--SNR",
        type=float,
        default=np.infty,
        help="signal to noise ratio, default: infinity (no noise)",
    )
    parser.add_argument("--pressure", action="store_true", help="save pressure")
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="gamma* scaling of the Navier slip bc",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="Location of data folder",
    )
    args = parser.parse_args()
    return args


def make_noise(shape, venc: float = 1.5, snr: float = np.infty):
    # adding noise
    np.random.seed(1111)
    amount_of_noise = venc / snr
    standard_deviation = 0.45 * venc / snr
    print(
        f"Amount of noise: {amount_of_noise}, standard deviation: {standard_deviation}"
    )
    noise = amount_of_noise * np.random.normal(size=shape, scale=standard_deviation)
    return noise


def add_noise_basic(
    data,
    venc: float,
    snr: float = np.infty,
    comm=df.MPI.comm_world,
    function_space=None,
):
    print(f"Adding basic noise with signal to noise ratio {snr}...")
    local_vector = data.vector().get_local()
    local_size = data.vector().local_size()
    noise = make_noise(local_size, venc=venc, snr=snr)
    data.vector().set_local(local_vector + noise)
    if function_space is not None:
        data.set_allow_extrapolation(True)
        data = df.interpolate(data, function_space)
    return data


def write_results(
    data_original,
    venc: float,
    add_noise: Callable = add_noise_basic,
    snr: float = np.infty,
    domain: str = "",
    name: str = "",
    **add_noise_kwargs,
):
    comm = df.MPI.comm_world
    data = data_original.copy(deepcopy=True)
    data = add_noise(data, venc, snr=snr, comm=comm, **add_noise_kwargs)
    if name != "":
        if type(data) == tuple:
            data, mri = data
            mri_comm = mri.function_space().mesh().mpi_comm()
            with df.XDMFFile(mri_comm, f"{domain}/{name}_mri.xdmf") as xdmffile:
                xdmffile.write(mri)
        data.rename("data", "data")
        with df.XDMFFile(comm, f"{domain}/{name}.xdmf") as xdmffile:
            xdmffile.write(data)
        with df.HDF5File(comm, f"{domain}/{name}h5.h5", "w") as h5file:
            h5file.write(data, "data")
        print(f"{name} succesfully saved.")
    return data


def estimate_ideal_venc(data_dict):
    linf_norms = []
    for name, velocity, pressure in data_dict.values():
        max_vel = velocity.vector().norm("linf")
        max_vel = ceil(100 * max_vel) * 0.01
        linf_norms.append(max_vel)
    return max(linf_norms)


if __name__ == "__main__":
    args = get_args()
    datafolder = args.data_folder
    meshpath = f"{datafolder}/{args.meshname}"
    comm = df.MPI.comm_world
    mesh, bndry, _ = read_mesh(meshpath, comm=comm)
    if args.meshname_original != "":
        if df.MPI.size(comm) > 1:
            raise NotImplementedError(
                "Interpolation from one mesh does not work in parallel!"
            )
        meshpath_o = os.path.join(datafolder, args.meshname_original)
        mesh_o, bndry_o, _ = read_mesh(meshpath_o, comm=comm)
        meshname_o = args.meshname_original
    else:
        mesh_o, bndry_o = mesh, bndry
        meshname_o = args.meshname

    thetas = [0.0, 0.2, 0.5, 0.8, 1.0]
    elements = [
        "p1p1",
        "mini",
    ]

    mini_element = finite_elements.string_to_element["mini"]
    FE_o = mini_element(mesh_o, bndry_o)
    FE_o.setup()
    V_o = FE_o.V.collapse()
    P_o = FE_o.P.collapse()

    data_dict = dict()
    for theta in thetas:
        velocity = df.Function(V_o)
        pressure = df.Function(P_o)
        name = args.name
        filepath = f"{datafolder}/{meshname_o}/mini/{name}{int(round(theta*1000))}"
        if args.gamma != 0.25:
            filepath += f"_gamma{args.gamma}"
        with df.HDF5File(comm, f"{filepath}_velh5.h5", "r") as h5file:
            print(f"Reading {filepath}_velh5.h5 ...")
            h5file.read(velocity, "v")
        with df.HDF5File(comm, f"{filepath}_pressh5.h5", "r") as h5file:
            print(f"Reading {filepath}_pressh5.h5 ...")
            h5file.read(pressure, "p")
        write_name = f"{name}{int(round(theta*1000))}"
        if args.gamma != 0.25:
            write_name += f"_gamma{args.gamma}"
        data_dict[theta] = (write_name, velocity, pressure)
    ideal_venc = estimate_ideal_venc(data_dict)
    for element in elements:
        for theta in thetas:
            write_name, velocity, pressure = data_dict[theta]
            print(
                f"element={element}  theta={theta} ---------------------------------------------------------"
            )
            ns_element = finite_elements.string_to_element[element]
            FE = ns_element(mesh, bndry)
            FE.setup()
            if args.pressure:
                P_collapse = FE.P.collapse()
                write_results(
                    pressure,
                    ideal_venc,
                    domain=f"{meshpath}/{element}",
                    name=f"{write_name}_pressure",
                    function_space=P_collapse,
                )
            else:
                V_collapse = FE.V.collapse()
                if args.SNR == np.infty:
                    noise = "clean"
                else:
                    noise = f"snr{args.SNR}"
                write_results(
                    velocity,
                    ideal_venc,
                    domain=f"{meshpath}/{element}",
                    name=f"{write_name}_{noise}",
                    snr=args.SNR,
                    function_space=V_collapse,
                )
