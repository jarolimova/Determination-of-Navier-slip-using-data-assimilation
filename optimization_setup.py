import dolfin as df
import dolfin_adjoint as da
import numpy as np

from weak_formulation import marks


class ErrorFunctional:
    def __init__(
        self,
        data,
        ds,
        analytic_profile,
        alpha=0.0,
        beta=0.0,
        char_len=0.01,
        dx=None,
    ):
        self.data = data
        self.ds = ds
        self.analytic_profile = analytic_profile
        self.mesh = data.function_space().mesh()
        self.out_normal = df.FacetNormal(self.mesh)
        if dx is None:
            dx = df.Measure("dx", domain=self.mesh, metadata={"quadrature_degree": 4})
        self.dx = dx
        self.char_len = char_len
        self.alpha = alpha
        self.beta = beta

    def J(self, u, u_in, theta, v_avg):
        J = da.assemble(
            0.5
            / pow(self.char_len, 3)
            * df.inner(u - self.data, u - self.data)
            * self.dx
        )
        J += self.Jreg(u_in, theta, v_avg)
        return J

    def Jreg(self, u_in, theta, v_avg):
        Jreg = 0.0
        Jreg += self.J_alpha(u_in)
        Jreg += self.J_beta(u_in, theta, v_avg)
        return Jreg

    def J_alpha(self, u_in):
        alpha_reg = 0.0
        if u_in is not None and self.alpha > 0.0:
            alpha_reg += self.alpha * 0.5 * h1_seminorm(u_in, self.out_normal, self.ds(marks["in"]))
        return alpha_reg

    def J_beta(self, u_in, theta, v_avg):
        if u_in is not None and self.beta > 0.0:
            profile = self.analytic_profile(theta, v_avg)
            return da.assemble(
                self.beta
                * 0.5
                / pow(self.char_len, 2)
                * df.inner(u_in - profile, u_in - profile)
                * self.ds(marks["in"])
            )
        else:
            return 0.0

class SlipPenalty:
    """Object created for computation of slip penalty on the wall
    defined using theta, penalty = theta/(gamma*(1-theta))
    theta = gamma*penalty/(1+ gamma*penalty)
    Args:
        use_theta: if set to False, penalty is used as a control variable instead of theta
        gamma: gamma parameter in the wall bc
    """

    def __init__(self, use_theta=True, gamma: float = 1.0):
        self.gamma = gamma
        self.use_theta=use_theta

    def theta_to_penalty(self, theta):
        """compute penalty using theta"""
        return theta / (self.gamma*(1.0 - theta))

    def penalty_to_theta(self, penalty):
        """compute theta using penalty"""
        return self.gamma*penalty / (1 + self.gamma*penalty)


class Controls:
    def __init__(self, theta=None, u_in=None, slip_penalty=SlipPenalty()):
        self.theta = theta
        self.u_in = u_in
        self.slip_penalty = slip_penalty
        theta_bounds = [0.0, 0.999]
        if not slip_penalty.use_theta:
            theta_bounds = [slip_penalty.theta_to_penalty(bound) for bound in theta_bounds]
        if u_in is not None:
            dim = u_in.function_space().mesh().geometric_dimension()
            lbounds = da.interpolate(
                da.Constant(tuple(dim * [-10000.0])),
                u_in.function_space(),
            )
            rbounds = da.interpolate(
                da.Constant(tuple(dim * [10000.0])),
                u_in.function_space(),
            )
            if theta is not None:
                self.m = [da.Control(theta), da.Control(u_in)]
                self.bounds = np.asarray(
                    [
                        [da.Constant(theta_bounds[0]), lbounds],
                        [da.Constant(theta_bounds[1]), rbounds],
                    ],
                    dtype="object",
                )
            else:
                self.m = da.Control(u_in)
                self.bounds = [[lbounds], [rbounds]]
        else:
            if theta is not None:
                self.m = da.Control(theta)
                self.bounds = [[da.Constant(theta_bounds[0])], [da.Constant(theta_bounds[1])]]
            else:
                raise ValueError("There are no control variables.")


    def extract_values(self, m):
        theta, u_in = (None, None)
        if self.theta is not None and self.u_in is not None:
            theta, u_in = m
        elif self.theta is not None:
            theta = m
        elif self.u_in is not None:
            u_in = m
        if not self.slip_penalty.use_theta and theta is not None:
            theta = self.slip_penalty.penalty_to_theta(theta)
        return theta, u_in


def h1_seminorm(function, n, ds):
    dim = function.function_space().mesh().geometric_dimension()
    I = df.Identity(dim)
    It = I - df.outer(n, n)
    seminorm = da.assemble(
        df.inner(df.grad(function) * It, df.grad(function) * It) * ds
    )
    return seminorm