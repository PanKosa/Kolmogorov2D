import numpy as np

from copy import copy
from siatka import Siatka

from SORy import calculate_intermediate_velocity, calculate_poisson_equation, SORKinetic


class Solver:
    
    def convert_velocity_to_matrix(self,v0,u0):
        self.u_prev = np.zeros((self.siatka.Y.size,self.siatka.X.size))
        self.v_prev = np.zeros((self.siatka.Y.size,self.siatka.X.size))

        for ix in range(self.siatka.X.size):
            for iy in range(self.siatka.Y.size):
                if self.siatka.inner_points[iy,ix] == 1:
                    self.u_prev[iy,ix] = u0(self.siatka.X[ix], self.siatka.Y[iy])
                    self.v_prev[iy,ix] = v0(self.siatka.X[ix], self.siatka.Y[iy])
                    
    def convert_omega_to_matrix(self,omega0):
        for ix in range(self.siatka.X.size):
            for iy in range(self.siatka.Y.size):
                if self.siatka.inner_points[iy,ix] == 1 or self.siatka.inner_points[iy,ix] == 2:
                    self.omega[iy,ix] = omega0(self.siatka.X[ix], self.siatka.Y[iy])
                    
    def convert_b_to_matrix(self,b0):
        for ix in range(self.siatka.X.size):
            for iy in range(self.siatka.Y.size):
                if self.siatka.inner_points[iy,ix] == 1 or self.siatka.inner_points[iy,ix] == 2:
                    self.b[iy,ix] = b0(self.siatka.X[ix], self.siatka.Y[iy])
                    
    def update_force_matrix(self):
        for ix in range(self.siatka.X.size):
            for iy in range(self.siatka.Y.size):
                if self.siatka.inner_points[iy,ix] == 1:
                    self.fx[iy,ix] = self.f_func_x(self.siatka.X[ix], self.siatka.Y[iy], self.t)
                    self.fy[iy,ix] = self.f_func_y(self.siatka.X[ix], self.siatka.Y[iy], self.t)
                    
    def calculate_pressure(self):
        pass
    
    def update_velocity(self):
        for ix in range(self.siatka.X.size):
            for iy in range(self.siatka.Y.size):
                if self.siatka.inner_points[iy,ix] == 1:
                    self.v[iy,ix] = self.u_star_x[iy,ix] - self.dt * ( self.phi[iy,ix+1] - self.phi[iy,ix-1] )/2/self.siatka.h
                    self.u[iy,ix] = self.u_star_y[iy,ix] - self.dt * ( self.phi[iy-1,ix] - self.phi[iy+1,ix] )/2/self.siatka.h
        for p in self.siatka.boundary_points:
                    self.v[p.iy,p.ix] = p.brzeg_element.boundary_v(self.siatka.X[p.ix], self.siatka.Y[p.iy], self.t)
                    self.u[p.iy,p.ix] = p.brzeg_element.boundary_u(self.siatka.X[p.ix], self.siatka.Y[p.iy], self.t)
                    
    def update_pressure(self):
        for ix in range(self.siatka.X.size):
            for iy in range(self.siatka.Y.size):
                if self.siatka.inner_points[iy,ix] == 1 and self.options['beta'] > 0:
                    self.p[iy,ix] = self.options['beta'] * self.p[iy,ix] + self.phi[iy,ix]
                else:
                    self.p[iy,ix] = self.phi[iy,ix]
            
    def __init__(self,dt,siatka,v0,u0,omega0,b0,fx,fy,options):

        print("Rozpoczynam zapis siatki...")
        self.siatka = siatka
        print("Wykonane")
        
        self.t  = 0
        self.dt = dt
        self.options = options
        
        print("Rozpoczynam inicjalizacje predkosci...") 
        self.v  = np.zeros((self.siatka.Y.size, self.siatka.X.size))
        self.u  = np.zeros((self.siatka.Y.size, self.siatka.X.size))
        self.convert_velocity_to_matrix(v0,u0)
        print("Wykonane")
        
        print("Rozpoczynam inicjalizacje cisnienia...") 
        self.p  = np.zeros((self.siatka.Y.size, self.siatka.X.size))
        self.calculate_pressure()
        print("Wykonane")
        
        print("Rozpoczynam inicjalizacje omega...") 
        self.omega  = np.zeros((self.siatka.Y.size, self.siatka.X.size))
        self.convert_omega_to_matrix(omega0)
        print("Wykonane")
        
        print("Rozpoczynam inicjalizacje b...") 
        self.b  = np.zeros((self.siatka.Y.size, self.siatka.X.size))
        self.convert_b_to_matrix(b0)
        print("Wykonane")
        
        print("Rozpoczynam inicjalizacje sily zewnetrzej...")
        self.f_func_x = fx
        self.f_func_y = fy
        self.fx  = np.zeros((self.siatka.Y.size, self.siatka.X.size))
        self.fy  = np.zeros((self.siatka.Y.size, self.siatka.X.size))
        self.update_force_matrix()
        print("Wykonane")
        
        self.phi = np.zeros((self.siatka.Y.size, self.siatka.X.size))

    def time_step(self):
        self.t += self.dt
        
        # obliczam nowe wartosci omega i b
        self.omega,self.b = SORKinetic(self.siatka.X, self.siatka.Y, self.siatka.h, self.dt, self.omega,
                                       self.b, self.v_prev, self.u_prev, self.siatka.boundary_points,
                                       self.siatka.inner_points, self.options['kappa1'], self.options['kappa2'],
                                       self.options['kappa3'], self.options['kappa4'], self.options['alpha1'],
                                       self.options['alpha2'], self.options['alpha3'])
        
        # obliczam wartosc u*
        self.u_star_x, self.u_star_y = calculate_intermediate_velocity(self.siatka.X, self.siatka.Y, self.siatka.h, self.dt,
                                                                       self.options['nu'], self.u_prev, self.v_prev, self.p,
                                                                       self.fx,
                                                                       self.fy, self.omega, self.b, self.siatka.inner_points,
                                                                       self.options['beta'], self.siatka.boundary_points)
                                                                      
        
        # rozwiazuje rownanie \nabla \phi = 1/dt \nabla \cdot u^*     
        neumann_boundary_codition = np.zeros((self.siatka.Y.size,self.siatka.X.size))
        
        self.phi = calculate_poisson_equation(self.siatka.X, self.siatka.Y, self.siatka.h, self.dt, self.u_star_y, self.u_star_x,
                                              self.phi, self.siatka.boundary_points,
                                              self.siatka.inner_points, self.options['delta'])
        
        # update predkosci na podstawie phi
        self.update_velocity()
        
        # update cisnienia na podstawie phi
        self.update_pressure()
        
        # update wartosci sily
        self.update_force_matrix()
        
        self.v_prev = copy(self.v)
        self.u_prev = copy(self.u)
        
    def solve(self, T, add_dt, save_dt, save_name):
        time_from_last_add = 0            # mierzy ile czasu minelo od ostatniego dodania rozwiazania
        time_from_last_save = 0
        
        # Tworzenie rozwiazan
        self.solution_v     = np.zeros((self.siatka.Y.size, self.siatka.X.size, round(T/add_dt)))
        self.solution_u     = np.zeros((self.siatka.Y.size, self.siatka.X.size, round(T/add_dt)))
        self.solution_p     = np.zeros((self.siatka.Y.size, self.siatka.X.size, round(T/add_dt)))
        self.solution_omega = np.zeros((self.siatka.Y.size, self.siatka.X.size, round(T/add_dt)))
        self.solution_b     = np.zeros((self.siatka.Y.size, self.siatka.X.size, round(T/add_dt)))
        self.solution_times = np.zeros((round(T/add_dt)))
        
        # Zapis w chwili poczatkowej
        itSave = 0
        self.solution_v[:,:,itSave]     = self.v_prev
        self.solution_u[:,:,itSave]     = self.u_prev
        self.solution_p[:,:,itSave]     = self.p
        self.solution_omega[:,:,itSave] = self.omega
        self.solution_b[:,:,itSave]     = self.b
        self.solution_times[itSave]     = self.t
        itSave += 1

        while self.t < T and itSave < round(T/add_dt):
            self.time_step()
            time_from_last_add  += self.dt
            time_from_last_save += self.dt
            if time_from_last_add >= add_dt:
                time_from_last_add = 0
                self.solution_v[:,:,itSave]     = self.v
                self.solution_u[:,:,itSave]     = self.u
                self.solution_p[:,:,itSave]     = self.p
                self.solution_omega[:,:,itSave] = self.omega
                self.solution_b[:,:,itSave]     = self.b
                self.solution_times[itSave]     = self.t
                itSave += 1
                print("Czas " + str(round(self.t,4)) + ".....Dodano")
            if time_from_last_save >= save_dt:
                time_from_last_save = 0
                d = dict(P=self.solution_p, V=self.solution_v, U=self.solution_u,
                         OMEGA=self.solution_omega, B=self.solution_b,
                          TIMES=self.solution_times, X=self.siatka.X, Y=self.siatka.Y)
                np.savez(save_name, **d )
                print("Czas:" + str(round(self.t,4)) + " - zapisano rozwiazanie do pliku: " + save_name + ".npz")            
                
        # Zapis do pliku rozwiazan
        d = dict(P=self.solution_p, V=self.solution_v, U=self.solution_u,
                 OMEGA=self.solution_omega, B=self.solution_b,
                 TIMES=self.solution_times, X=self.siatka.X, Y=self.siatka.Y)
        np.savez(save_name, **d )
        print("Zapisano rozwiazanie do pliku: " + save_name + ".npz")
