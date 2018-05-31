cimport cython
cimport numpy as np
import numpy as np
from copy import copy 
from scipy import sparse
import scipy.sparse.linalg


from boundary_point import Boundary_point


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef  SORKineticPrepereRightSide(np.ndarray[np.double_t,ndim=1] X, np.ndarray[np.double_t,ndim=1] Y, double h, np.ndarray[np.double_t,ndim=2] omega,
                                                          np.ndarray[np.double_t,ndim=2] b, np.ndarray[np.double_t,ndim=2] v, np.ndarray[np.double_t,ndim=2] u,
                                                          np.ndarray[np.double_t,ndim=2] inner_points, double kappa1, double kappa2, double kappa3, double kappa4,
                                                          double alpha1, double alpha2, double alpha3):
    cdef unsigned int ix,iy
    cdef unsigned int Nx = X.size
    cdef unsigned int Ny = Y.size

    cpdef np.ndarray[np.double_t,ndim=2] fomega = np.zeros((Ny,Nx))
    cpdef np.ndarray[np.double_t,ndim=2] fb     = np.zeros((Ny,Nx))
    
    h = h * 2.0    

    for ix in range(Nx):
        for iy in range(Ny):
            if inner_points[iy,ix] == 1:
                fomega[iy,ix] = kappa1*(omega[iy,ix+1] - omega[iy,ix-1])/h * ( b[iy,ix+1]/omega[iy,ix+1] - b[iy,ix-1]/omega[iy,ix-1])/h + \
                                kappa1*(omega[iy-1,ix] - omega[iy+1,ix])/h * ( b[iy-1,ix]/omega[iy-1,ix] - b[iy+1,ix]/omega[iy+1,ix])/h - \
                          kappa2*omega[iy,ix]*omega[iy,ix]                                                                       - \
                          alpha1*v[iy,ix]*(omega[iy,ix+1] - omega[iy,ix-1])/h - alpha1*u[iy,ix]*(omega[iy-1,ix] - omega[iy+1,ix])/h                   
                fb[iy,ix]     = kappa3*(b[iy,ix+1]     - b[iy,ix-1])/h     * ( b[iy,ix+1]/omega[iy,ix+1] - b[iy,ix-1]/omega[iy,ix-1])/h + \
                                kappa3*(b[iy-1,ix]     - b[iy+1,ix])/h     * ( b[iy-1,ix]/omega[iy-1,ix] - b[iy+1,ix]/omega[iy+1,ix])/h - \
                         alpha3*omega[iy,ix]*b[iy,ix]                                                                           - \
                         alpha2*v[iy,ix]*(b[iy,ix+1] - b[iy,ix-1])/h - alpha2*u[iy,ix]*(b[iy-1,ix] - b[iy+1,ix])/h                     + \
                         kappa4*b[iy,ix]/omega[iy,ix]*(v[iy,ix+1] - v[iy,ix-1])/h * (v[iy,ix+1] - v[iy,ix-1])/h                 + \
                         kappa4*b[iy,ix]/omega[iy,ix]*(v[iy-1,ix] - v[iy+1,ix])/h * (v[iy-1,ix] - v[iy+1,ix])/h                 + \
                         kappa4*b[iy,ix]/omega[iy,ix]*(u[iy-1,ix] - u[iy+1,ix])/h * (u[iy-1,ix] - u[iy+1,ix])/h                 + \
                         kappa4*b[iy,ix]/omega[iy,ix]*(u[iy,ix+1] - u[iy,ix-1])/h * (u[iy,ix+1] - u[iy,ix-1])/h            
    return fomega,fb


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef  turbulent_time_step(np.ndarray[np.double_t,ndim=1] X, np.ndarray[np.double_t,ndim=1] Y, double dt, double h, np.ndarray[np.double_t,ndim=2] fomega,
                  np.ndarray[np.double_t,ndim=2] fb, np.ndarray[np.double_t,ndim=2] omega, np.ndarray[np.double_t,ndim=2] b,
                  np.ndarray[np.double_t,ndim=2] inner, double kappa1, double kappa3):


    cdef np.ndarray[np.double_t,ndim=2] omega_1 = np.zeros((Y.size,X.size))
    cdef np.ndarray[np.double_t,ndim=2] b_1     = np.zeros((Y.size,X.size))

    cpdef np.ndarray[np.double_t,ndim=2] omega_new = np.zeros((Y.size,X.size))
    cpdef np.ndarray[np.double_t,ndim=2] b_new     = np.zeros((Y.size,X.size))

    cdef np.double_t mu

    #cdef np.ndarray[np.double_t,ndim=2] A 
    #cdef np.ndarray[np.double_t,ndim=2] bomega 
    #cdef np.ndarray[np.double_t,ndim=2] bb 

    for iy in range(Y.size):
        Aomega      = np.zeros((X.size,X.size))
        Ab      = np.zeros((X.size,X.size))
        bomega = np.zeros((X.size,1))
        bb     = np.zeros((X.size,1))

        # usupelnianie ukladu rownan
        for ix in range(X.size):
            if inner[iy,ix] == 1:
                mu = dt/2/h/h * b[iy,ix] / omega[iy,ix]
                Aomega[ix,ix-1] = - mu*kappa1
                Aomega[ix,ix]   = 1 + 2 * mu*kappa1
                Aomega[ix,ix+1] = - mu*kappa1
                Ab[ix,ix-1] = - mu*kappa3
                Ab[ix,ix]   = 1 + 2 * mu*kappa3
                Ab[ix,ix+1] = - mu*kappa3
                bomega[ix]      = omega[iy,ix] + mu * kappa1 * (omega[iy-1,ix] - 2 * omega[iy,ix] +  omega[iy+1,ix]) + dt/2*fomega[iy,ix]
                bb[ix]      = b[iy,ix] + mu * kappa3 * (b[iy-1,ix] - 2 * b[iy,ix] + b[iy+1,ix]) + dt/2*fb[iy,ix]
            elif inner[iy,ix] == 2:
                if ix - 1 > 0 and inner[iy,ix-1] == 1:
                    Aomega[ix,ix-1] = - 1
                    Aomega[ix,ix]   =   1
                    Ab[ix,ix-1] = - 1
                    Ab[ix,ix]   =   1
                    bomega[ix]  =   0
                    bb[ix]      =   0
                elif ix + 1 < X.size and inner[iy,ix+1] == 1:
                    Aomega[ix,ix+1] = - 1
                    Aomega[ix,ix]   =   1
                    Ab[ix,ix+1] = - 1
                    Ab[ix,ix]   =   1
                    bomega[ix] =   0
                    bb[ix]      =   0
                elif iy + 1 < Y.size and not inner[iy+1,ix] == 0:
                    Aomega[ix,ix]   =   1
                    Ab[ix,ix]   =   1
                    bomega[ix] =   omega[iy+1,ix]
                    bb[ix]      =   b[iy+1,ix]
                elif iy - 1 >= 0  and not inner[iy-1,ix] == 0:
                    Aomega[ix,ix]   =   1
                    Ab[ix,ix]   =   1
                    bomega[ix] =   omega[iy-1,ix]
                    bb[ix]      =   b[iy-1,ix]
            else:
                Aomega[ix,ix]   =   1
                Ab[ix,ix]   =   1
                bomega[ix]      =  omega[iy,ix]
                bb[ix]      =  b[iy,ix]

        Aomega = sparse.csr_matrix(Aomega)
        Ab = sparse.csr_matrix(Ab)
        omega_1[iy,:] = scipy.sparse.linalg.spsolve(Aomega,bomega)
        b_1[iy,:]     = scipy.sparse.linalg.spsolve(Ab,bb)

    for ix in range(X.size):
        Aomega      = np.zeros((Y.size,Y.size))
        Ab      = np.zeros((Y.size,Y.size))
        bomega = np.zeros((Y.size,1))
        bb     = np.zeros((Y.size,1))

        # usupelnianie ukladu rownan
        for iy in range(Y.size):
            if inner[iy,ix] == 1:
                mu = dt/2/h/h * b[iy,ix] / omega[iy,ix]
                Aomega[iy,iy-1] = - mu*kappa1
                Aomega[iy,iy]   = 1 + 2 * mu*kappa1
                Aomega[iy,iy+1] = - mu*kappa1
                Ab[iy,iy-1] = - mu*kappa3
                Ab[iy,iy]   = 1 + 2 * mu*kappa3
                Ab[iy,iy+1] = - mu*kappa3
                bomega[iy]      = omega_1[iy,ix] + mu *kappa1 * (omega_1[iy,ix+1] - 2 * omega_1[iy,ix] + omega_1[iy,ix-1]) + dt/2*fomega[iy,ix]
                bb[iy]      = b_1[iy,ix] + mu * kappa3 * (b_1[iy,ix+1] - 2 * b_1[iy,ix] + b_1[iy,ix-1]) + dt/2*fb[iy,ix]
            elif inner[iy,ix] == 2:
                if iy - 1 > 0 and inner[iy-1,ix] == 1:
                    Aomega[iy,iy-1] = - 1
                    Aomega[iy,iy]   =   1
                    Ab[iy,iy-1] = - 1
                    Ab[iy,iy]   =   1
                    bomega[iy]      =   0
                    bb[iy]      =   0
                elif iy + 1 < Y.size and inner[iy+1,ix] == 1:
                    Aomega[iy,iy+1] = - 1
                    Aomega[iy,iy]   =   1
                    Ab[iy,iy+1] = - 1
                    Ab[iy,iy]   =   1
                    bomega[iy]      =   0
                    bb[iy]      =   0
                elif ix + 1 < X.size and not inner[iy,ix+1] == 0:
                    Aomega[iy,iy]   =   1
                    Ab[iy,iy]   =   1
                    bomega[iy] =   omega_1[iy,ix+1]
                    bb[iy]     =   b_1[iy,ix+1]
                elif ix - 1 >= 0  and not inner[iy,ix-1] == 0:
                    Aomega[iy,iy]   =   1
                    Ab[iy,iy]   =   1
                    bomega[iy] =   omega_1[iy,ix-1]
                    bb[iy]     =   b_1[iy,ix-1]
            else:
                Aomega[iy,iy]   =   1
                Ab[iy,iy]   =   1
                bomega[iy]  = omega_1[iy,ix]
                bb[iy]      = b_1[iy,ix]
        Aomega = sparse.csr_matrix(Aomega)
        Ab = sparse.csr_matrix(Ab)
        omega_new[:,ix] = scipy.sparse.linalg.spsolve(Aomega,bomega)
        b_new[:,ix] = scipy.sparse.linalg.spsolve(Ab,bb)
    return omega_new,b_new


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef  SORKinetic(np.ndarray[np.double_t,ndim=1] X, np.ndarray[np.double_t,ndim=1] Y, double h, double dt, np.ndarray[np.double_t,ndim=2] omega,
                  np.ndarray[np.double_t,ndim=2] b, np.ndarray[np.double_t,ndim=2] v, np.ndarray[np.double_t,ndim=2] u, boundary_points,
                  np.ndarray[np.double_t,ndim=2] inner_points, double kappa1, double kappa2, double kappa3, double kappa4,
                  double alpha1, double alpha2, double alpha3):



    cpdef np.ndarray[np.double_t,ndim=2] fomega
    cpdef np.ndarray[np.double_t,ndim=2] fb     
    
    # obliczanie prawej strony
    fomega,fb = SORKineticPrepereRightSide(X, Y, h, omega, b, v, u, inner_points, kappa1, kappa2, kappa3, kappa4, alpha1, alpha2, alpha3)
 
    
    return turbulent_time_step(X,  Y, dt, h, fomega, fb, omega, b, inner_points, kappa1, kappa3)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.double_t,ndim=2] SORPressureNeumann(np.ndarray[np.double_t,ndim=1] X, np.ndarray[np.double_t,ndim=1] Y, double h, np.ndarray[np.double_t,ndim=2] omega, np.ndarray[np.double_t,ndim=2] sol_prev, boundary_points,
          np.ndarray[np.double_t,ndim=2] inner_points, double delta):

    cdef int it = 0
    cdef double eta = 2.0/(1+ np.sin(np.pi * h))
    cdef double gs_sum = delta*delta + 1
    cdef double sol_gs 
    # obsadzenie warunkiem poczatkowym
    cpdef np.ndarray[np.double_t,ndim=2] sol = copy(sol_prev)

    cdef unsigned int ix,iy
    cdef unsigned int Nx = X.size
    cdef unsigned int Ny = Y.size

    cdef unsigned int IX,IY

    for ix in range(Nx):
        for iy in range(Ny):
            if inner_points[iy,ix] == 1:
                IX = ix
                IY = iy
                break

    
    while np.abs(gs_sum) > delta*delta and it < 5000:
        gs_sum = 0
        it = it + 1 
        # Update wartosci na brzegu
#        for p in boundary_points:
#            # obsluga punktow brzegowych - 1D
#            if p.ix == 0:
#                    sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy,p.ix+1]
#            elif p.ix - X.size + 1 == 0:
#                    sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy,p.ix-1]
#            elif p.iy == 0:
#                    sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy+1,p.ix]
#            elif p.iy - Y.size + 1 == 0:
#                    sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy-1,p.ix]
#            # obsluga pozostalych punktow brzegowych
#            elif X[p.ix] == p.brzeg_element.x_r:
#                sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy,p.ix+1]
#            elif X[p.ix] == p.brzeg_element.x_l:
#                sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy,p.ix-1]
#            elif Y[p.iy] == p.brzeg_element.y_u:
#                sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy-1,p.ix]
#            elif Y[p.iy] == p.brzeg_element.y_d:
#                sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy+1,p.ix]
#            else:
#                print("Warning")

        for p in boundary_points:
            # obsluga punktow brzegowych - 1D
            if p.ix == 0:
                    sol[p.iy,p.ix] =  sol[p.iy,p.ix+1]
            elif p.ix - X.size + 1 == 0:
                    sol[p.iy,p.ix] = sol[p.iy,p.ix-1]
            elif p.iy == 0:
                    sol[p.iy,p.ix] = sol[p.iy+1,p.ix]
            elif p.iy - Y.size + 1 == 0:
                    sol[p.iy,p.ix] = sol[p.iy-1,p.ix]
            # obsluga pozostalych punktow brzegowych
            elif X[p.ix] == p.brzeg_element.x_r:
                sol[p.iy,p.ix] = sol[p.iy,p.ix+1]
            elif X[p.ix] == p.brzeg_element.x_l:
                sol[p.iy,p.ix] = sol[p.iy,p.ix-1]
            elif Y[p.iy] == p.brzeg_element.y_u:
                sol[p.iy,p.ix] = sol[p.iy-1,p.ix]
            elif Y[p.iy] == p.brzeg_element.y_d:
                sol[p.iy,p.ix] = sol[p.iy+1,p.ix]
            else:
                print("Warning")


                
        # Glowna petla SOR
       
        for ix in range(Nx):
            for iy in range(Ny):
                if inner_points[iy,ix] == 1:
                    sol_gs = ( sol[iy,ix+1] + sol[iy,ix-1] + sol[iy+1,ix] + sol[iy-1,ix])/4 - h*h*omega[iy,ix]/4 - sol[iy,ix]

                    gs_sum += np.abs(sol_gs)
                    sol[iy,ix] = sol_gs*eta + sol[iy,ix]

                    
        sol = sol - sol[IY,IX]

        gs_sum *= eta*h*h
    #print(gs_sum)
    #print(it)
    return sol

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.double_t,ndim=2] calculate_poisson_equation(np.ndarray[np.double_t,ndim=1] X, np.ndarray[np.double_t,ndim=1] Y, double h, double dt, np.ndarray[np.double_t,ndim=2] u_star_y,
                                                                np.ndarray[np.double_t,ndim=2] u_star_x, np.ndarray[np.double_t,ndim=2] sol_prev,
                                                                boundary_points, np.ndarray[np.double_t,ndim=2] inner_points, double delta):

    cdef unsigned int ix,iy
    cdef unsigned int Nx = X.size
    cdef unsigned int Ny = Y.size
    cdef np.ndarray[np.double_t,ndim=2] omega = np.zeros((Ny,Nx))
        
    for ix in range(Nx):
        for iy in range(Ny):
            if inner_points[iy,ix] == 1:
                omega[iy,ix] = ( u_star_x[iy,ix+1] - u_star_x[iy,ix-1] + u_star_y[iy-1,ix] - u_star_y[iy+1,ix] )/2/h/dt


    return SORPressureNeumann(X, Y, h, omega, sol_prev, boundary_points, inner_points, delta)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef calculate_intermediate_velocity(np.ndarray[np.double_t,ndim=1] X, np.ndarray[np.double_t,ndim=1] Y, double h, double dt, double nu, np.ndarray[np.double_t,ndim=2] u,
                                      np.ndarray[np.double_t,ndim=2] v, np.ndarray[np.double_t,ndim=2] p, np.ndarray[np.double_t,ndim=2] fx, np.ndarray[np.double_t,ndim=2] fy,
                                      np.ndarray[np.double_t,ndim=2] omega, np.ndarray[np.double_t,ndim=2] b, np.ndarray[np.double_t,ndim=2] inner_points, double beta, boundary_points):


    cdef unsigned int ix,iy
    cdef unsigned int Nx = X.size
    cdef unsigned int Ny = Y.size
    
    # inicjalizacja posrednich predkosci
    cpdef np.ndarray[np.double_t,ndim=2] inter_v = np.zeros((Ny,Nx))
    cpdef np.ndarray[np.double_t,ndim=2] inter_u = np.zeros((Ny,Nx))
    

    for ix in range(Nx):
        for iy in range(Ny):
            if inner_points[iy,ix] == 1:
                inter_v[iy,ix] = (  -(v[iy,ix+1]*v[iy,ix+1] - v[iy,ix-1]*v[iy,ix-1])/h/2 - \
                                     (u[iy-1,ix]*v[iy-1,ix] - u[iy+1,ix]*v[iy+1,ix])/h/2 - \
                                beta*(p[iy,ix+1] - p[iy,ix-1])/h/2                       + \
            nu*b[iy,ix]/omega[iy,ix]*(v[iy,ix+1] - 2*v[iy,ix] + v[iy,ix-1])/h/h          + \
            nu*b[iy,ix]/omega[iy,ix]*(v[iy-1,ix] - 2*v[iy,ix] + v[iy+1,ix])/h/h          + \
                                     (v[iy,ix+1] - v[iy,ix-1])/h/2 * ( b[iy,ix+1]/omega[iy,ix+1] - b[iy,ix-1]/omega[iy,ix-1])/h/2 + \
                                     (v[iy-1,ix] - v[iy+1,ix])/h/2 * ( b[iy-1,ix]/omega[iy-1,ix] - b[iy-1,ix]/omega[iy-1,ix])/h/2 + \
                                      fx[iy,ix]                                            \
                                 )*dt  + v[iy,ix]
                inter_u[iy,ix] = (  -(u[iy-1,ix]*u[iy-1,ix] - u[iy+1,ix]*u[iy+1,ix])/h/2 - \
                                     (u[iy,ix+1]*v[iy,ix+1] - u[iy,ix-1]*v[iy,ix-1])/h/2 - \
                                beta*(p[iy-1,ix] - p[iy+1,ix])/h/2                       + \
            nu*b[iy,ix]/omega[iy,ix]*(u[iy,ix+1] - 2*u[iy,ix] + u[iy,ix-1])/h/h          + \
            nu*b[iy,ix]/omega[iy,ix]*(u[iy-1,ix] - 2*u[iy,ix] + u[iy+1,ix])/h/h          + \
                                     (u[iy,ix+1] - u[iy,ix-1])/h/2 * ( b[iy,ix+1]/omega[iy,ix+1] - b[iy,ix-1]/omega[iy,ix-1])/h/2 + \
                                     (u[iy-1,ix] - u[iy+1,ix])/h/2 * ( b[iy-1,ix]/omega[iy-1,ix] - b[iy-1,ix]/omega[iy-1,ix])/h/2 + \
                                      fy[iy,ix]                                            \
                                 )*dt  + u[iy,ix]
#    a = """
    #poprawka na brzegu
    for pp in boundary_points:
        # obsluga punktow brzegowych - 1D
        if pp.ix == 0:
            inter_u[pp.iy,pp.ix] = inter_u[pp.iy,pp.ix+1]
            inter_v[pp.iy,pp.ix] = inter_v[pp.iy,pp.ix+1]
        elif pp.ix - X.size + 1 == 0:
            inter_u[pp.iy,pp.ix] = inter_u[pp.iy,pp.ix-1]
            inter_v[pp.iy,pp.ix] = inter_v[pp.iy,pp.ix-1]
        elif pp.iy == 0:
            inter_u[pp.iy,pp.ix] = inter_u[pp.iy+1,pp.ix]
            inter_v[pp.iy,pp.ix] = inter_v[pp.iy+1,pp.ix]
        elif pp.iy - Y.size + 1 == 0:
            inter_u[pp.iy,pp.ix] = inter_u[pp.iy-1,pp.ix]
            inter_v[pp.iy,pp.ix] = inter_v[pp.iy-1,pp.ix]
        # obsluga pozostalych punktow brzegowych
        elif X[pp.ix] == pp.brzeg_element.x_r:
            inter_u[pp.iy,pp.ix] = inter_u[pp.iy,pp.ix+1]
            inter_v[pp.iy,pp.ix] = inter_v[pp.iy,pp.ix+1]
        elif X[pp.ix] == pp.brzeg_element.x_l:
            inter_u[pp.iy,pp.ix] = inter_u[pp.iy,pp.ix-1]
            inter_v[pp.iy,pp.ix] = inter_v[pp.iy,pp.ix-1]
        elif Y[pp.iy] == pp.brzeg_element.y_u:
            inter_u[pp.iy,pp.ix] = inter_u[pp.iy-1,pp.ix]
            inter_v[pp.iy,pp.ix] = inter_v[pp.iy-1,pp.ix]
        elif Y[pp.iy] == pp.brzeg_element.y_d:
            inter_u[pp.iy,pp.ix] = inter_u[pp.iy+1,pp.ix]
            inter_v[pp.iy,pp.ix] = inter_v[pp.iy+1,pp.ix]
        else:
            print("Warning")
#     """
                    

    return inter_v,inter_u
