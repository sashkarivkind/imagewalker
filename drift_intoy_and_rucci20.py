import numpy as np
import matplotlib.pyplot as plt


def brownian_motion_simulation(m=2, n=1001, d=10.0, t=1.0):
    # *****************************************************************************80
    #
    ## BROWNIAN_MOTION_SIMULATION simulates Brownian motion.
    #
    #  Discussion:
    #
    #    Thanks to Feifei Xu for pointing out a missing factor of 2 in the
    #    stepsize calculation, 08 March 2016.
    #
    #    Thanks to Joerg Peter Pfannmoeller for pointing out a missing factor
    #    of M in the stepsize calculation, 23 April 2018.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    10 June 2018
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, integer M, the spatial dimension.
    #    This defaults to 2.
    #
    #    Input, integer N, the number of time steps to take, plus 1.
    #    This defaults to 1001.
    #
    #    Input, real D, the diffusion coefficient.
    #    This defaults to 10.0.
    #
    #    Input, real T, the total time.
    #    This defaults to 1.0
    #
    #    Output, real X(M,N), the initial position at time 0.0, and
    #    the N-1 successive locations of the particle.
    #

    #
    #  Set the time step.
    #
    dt = t / float(n - 1)

    x = np.zeros([m, n])
    dx = np.zeros([m, n])

    step_size = np.sqrt(2.0 * m * d * dt) * np.random.randn(n)
    dx = np.random.randn(m, n)

    norm_dx = np.sqrt(np.sum(dx ** 2, 0))
    dx = step_size * dx / norm_dx
    x = np.cumsum(dx, axis=1)

    return x


# def brownian_motion_simulation( m = 2, n = 1001, d = 10.0, t = 1.0 ):

# #*****************************************************************************80
# #
# ## BROWNIAN_MOTION_SIMULATION simulates Brownian motion.
# #
# #  Discussion:
# #
# #    Thanks to Feifei Xu for pointing out a missing factor of 2 in the
# #    stepsize calculation, 08 March 2016.
# #
# #    Thanks to Joerg Peter Pfannmoeller for pointing out a missing factor
# #    of M in the stepsize calculation, 23 April 2018.
# #
# #  Licensing:
# #
# #    This code is distributed under the GNU LGPL license.
# #
# #  Modified:
# #
# #    10 June 2018
# #
# #  Author:
# #
# #    John Burkardt
# #
# #  Parameters:
# #
# #    Input, integer M, the spatial dimension.
# #    This defaults to 2.
# #
# #    Input, integer N, the number of time steps to take, plus 1.
# #    This defaults to 1001.
# #
# #    Input, real D, the diffusion coefficient.
# #    This defaults to 10.0.
# #
# #    Input, real T, the total time.
# #    This defaults to 1.0
# #
# #    Output, real X(M,N), the initial position at time 0.0, and
# #    the N-1 successive locations of the particle.
# #

#     #
#     #  Set the time step.
#     #
#     dt = t / float ( n - 1 )


#     x = np.zeros ( [ m, n ] )
#     dx = np.zeros ( [ m, n ] )

#     step_size = np.sqrt ( 2.0 * m * d * dt ) * np.random.randn ( n )
#     dx = np.random.randn ( m, n )

#     #
#     #  Compute the individual steps.
#     #
#     for j in range ( 1, n ):
#         #
#         #  S is the stepsize
#         #
# #         s = np.sqrt ( 2.0 * m * d * dt ) * np.random.randn ( n )

#         #
#         #  Direction is random.
#         #
#         if ( m == 1 ):
#             dx[j] = step_size[j] * np.ones ( 1 );
#         else:
# #             dx[0:m,j] = np.random.randn ( m )
#             norm_dx = np.sqrt ( np.sum ( dx[:,j] ** 2 ) )
#             for i in range ( 0, m ):
#                 dx[i,j] = step_size[j] * dx[i,j] / norm_dx

#         #
#         #  Each position is the sum of the previous steps.
#         #
#         x[0:m,j] = x[0:m,j-1] + dx[0:m,j]

#     return x


def diffusion_const_conversion(D_angle):
    #     input - D_angle: diffusion factor in units of arcmin^2*sec^(-1)
    #     output- D_receptors: diffusion factor in units of pix^2*sec^(-1)

    k = 1  # conversion factor, photoreceptor/pixel diameter in arcmin [pix*arcmin^(-1)]
    # k=1 means that one photoreceptor in the fovea spans 1 arcmin [1].

    D_receptors = k ** 2 * D_angle

    return D_receptors


def gen_drift_traj(D_arcmin=10.998, duration=0.3, N=5):
    #     D_arcmin - Diffusion constant with units of arcmin^2*sec^(-1).

    D = diffusion_const_conversion(D_arcmin)  # D- Diffusion constant with units of receptor_spacing^2*sec^(-1)
    m = 2
    n = int(duration * 1000 + 1)  # assuming original traj is sampled at 1KHz
    traj = brownian_motion_simulation(m=m, n=n, d=D, t=duration)

    # subsampling traj to produce trajectory with N points evenly distributed in time
    subsamp_traj = traj[:, range(0, n, n // (N - 1))]

    return traj, subsamp_traj


def gen_drift_traj_condition(duration=0.3, N=5, snellen=True):
    # duration- duration of the drift in seconds
    # N- the number of points the subsmapled trajectory should include
    # snellen- True: generate Snellen-condition drift, False: generate Fixation condition drift

    D_IR = 0.0
    if snellen:
        D_IR = 10.998  # see [1] source data file
    else:
        D_IR = 30.155  # see [1] source data file

    traj, traj_sb = gen_drift_traj(D_arcmin=D_IR, duration=duration, N=N)
    return traj, traj_sb
