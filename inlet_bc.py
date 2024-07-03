import os
import json
from petsc4py import PETSc
import dolfin as df
import dolfin_adjoint as da

print = PETSc.Sys.Print


def init_analytic(
    u_in,
    velocity=None,
    theta=None,
    meshname=None,
    mu=0.0038955,
    gamma=1.0,
    data_folder="data",
    **kwargs,
):
    """analytical profile expression initial guess"""
    dim = u_in.function_space().element().geometric_dimension()
    if dim != 3:
        raise ValueError(f"Analytic profile implemented only for dim=3 (not {dim})")
    if velocity is None:
        raise ValueError("No value provided for velocity.")
    if theta is None:
        raise ValueError("No value provided for theta.")
    if meshname is None:
        raise ValueError("No meshaname provided.")
    jsonpath = os.path.join(data_folder, f"{meshname}_cuts.json")
    with open(jsonpath, "r") as jsnfile:
        bnd_data = json.load(jsnfile)
    radius = bnd_data["in"]["radius"]
    cp = bnd_data["in"]["point"]
    normal_in = bnd_data["in"]["normal"]
    expr = analytic_profile_expression(
        radius, mu, theta, gamma, velocity, cp, normal_in
    )
    u_in = da.interpolate(expr, u_in.function_space())
    return u_in


def init_zero(u_in, **kwargs):
    """zero initial guess"""
    u_in.vector()[:] = 0.0
    return u_in


def init_data(u_in, data=None, **kwargs):
    """data initial guess"""
    if data is None:
        raise ValueError("No data provided for the init.")
    u_in.vector()[:] = data.vector()
    return u_in


def analytic_profile_expression(radius, mu, theta, gamma, velocity, cp, normal_in):
    profile = "max((4.0*Mu*g*(1.0-t)*r + 2.0*t*(pow(r,2)-pow(x[0]-c1,2)-pow(x[1]-c2,2)-pow(x[2]-c3,2)))/( 4.0*Mu*g*(1-t)*r + t*r*r), 0.0)"
    expr = da.Expression(
        (profile + "*v*n1", profile + "*v*n2", profile + "*v*n3"),
        r=radius,
        Mu=mu,
        t=theta,
        g=gamma,
        v=velocity,
        c1=cp[0],
        c2=cp[1],
        c3=cp[2],
        n1=normal_in[0],
        n2=normal_in[1],
        n3=normal_in[2],
        degree=2,
    )
    return expr


class AnalyticProfile:
    def __init__(self, mu, R, cp, normal_in, mesh, slip_penalty):
        self.mu = mu
        self.R = R
        self.cp = cp
        self.normal_in = normal_in
        self.mesh = mesh
        self.slip_penalty = slip_penalty

    @classmethod
    def from_json(cls, mu, meshname, mesh, slip_penalty, data_folder="data"):
        jsonpath = os.path.join(data_folder, f"{meshname}_cuts.json")
        with open(jsonpath, "r") as jsnfile:
            bnd_data = json.load(jsnfile)
        radius = bnd_data["in"]["radius"]
        cp = bnd_data["in"]["point"]
        normal_in = bnd_data["in"]["normal"]
        return cls(mu, radius, cp, normal_in, mesh, slip_penalty)

    def __call__(self, theta, v_avg):
        x, y, z = df.SpatialCoordinate(self.mesh)
        mu = self.mu
        R = self.R
        cp = self.cp
        gamma = self.slip_penalty.gamma
        if self.slip_penalty.use_theta:
            profile = (
                -v_avg
                * (
                    4 * mu * gamma * (1 - theta) * R
                    + 2
                    * theta
                    * (
                        pow(R, 2)
                        - pow(x - cp[0], 2)
                        - pow(y - cp[1], 2)
                        - pow(z - cp[2], 2)
                    )
                )
                / (4 * mu * gamma * (1 - theta) * R + theta * R**2)
                * df.FacetNormal(self.mesh)
            )
        else:
            profile = (
                -v_avg
                * (
                    4 * mu * R
                    + 2
                    * theta
                    * (
                        pow(R, 2)
                        - pow(x - cp[0], 2)
                        - pow(y - cp[1], 2)
                        - pow(z - cp[2], 2)
                    )
                )
                / (4 * mu * R + theta * R**2)
                * df.FacetNormal(self.mesh)
            )
        return profile
