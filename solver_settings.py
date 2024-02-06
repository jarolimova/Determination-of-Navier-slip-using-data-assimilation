from petsc4py import PETSc
import dolfin as df

print = PETSc.Sys.Print


def solver_setup(prm):
    prm["snes_solver"]["linear_solver"] = "mumps"
    prm["snes_solver"]["absolute_tolerance"] = 1e-10
    prm["snes_solver"]["relative_tolerance"] = 1e-10
    prm["snes_solver"]["maximum_iterations"] = 100
    prm["snes_solver"]["method"] = "newtonls"
    prm["snes_solver"]["line_search"] = "nleqerr"
    prm["snes_solver"]["report"] = True
    prm["snes_solver"]["error_on_nonconvergence"] = True

    prm["newton_solver"]["linear_solver"] = "mumps"
    prm["newton_solver"]["absolute_tolerance"] = 1e-10
    prm["newton_solver"]["relative_tolerance"] = 1e-14
    prm["newton_solver"]["maximum_iterations"] = 50
    prm["newton_solver"]["relaxation_parameter"] = 1.0
    prm["newton_solver"]["report"] = True
    prm["newton_solver"]["error_on_nonconvergence"] = True

    df.PETScOptions.set("mat_mumps_icntl_24", 1)  # detect null pivots
    df.PETScOptions.set(
        "mat_mumps_icntl_14", 200
    )  # work array, multiple to estimate to allocate
    df.PETScOptions.set("mat_mumps_cntl_1", 1e-2)  # relative pivoting threshold
    return
