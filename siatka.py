import numpy as np

from boundary_point import Boundary_point

class Siatka:

    def __init__(self, x, y, h, brzeg, bounds):
        nx = round(x/h)+1
        ny = round(y/h)+1

        X = np.linspace(0,x,nx)
        Y = np.flip(np.linspace(0,y,ny),0)

        # poprawianie brzegow
        for bb in brzeg:
            bb.x_l = X[np.abs(X - bb.x_l).argmin()]
            bb.x_r = X[np.abs(X - bb.x_r).argmin()]
            bb.y_u = X[np.abs(X - bb.y_u).argmin()]
            bb.y_d = X[np.abs(X - bb.y_d).argmin()]


        inner_points = np.ones((Y.size,X.size))

        for ix in range(X.size):
            for iy in range(Y.size):
                if inner_points[iy,ix] == 1:
                    for bb in brzeg:
                        if X[ix] >= bb.x_l and X[ix] <= bb.x_r and Y[iy] >= bb.y_d and Y[iy] <= bb.y_u:
                            inner_points[iy,ix] = 0
                            break
                            
        for ix in range(X.size):
            inner_points[0, ix] = 0
            inner_points[Y.size - 1, ix] = 0

        for iy in range(Y.size):
            inner_points[iy, 0] = 0
            inner_points[iy, X.size - 1] = 0
         
        points = []
        
        for ix in range(X.size):
            for iy in range(Y.size):
                if inner_points[iy, ix] == 0:
                    for bb in bounds:
                        if X[ix] >= bb.x_l and X[ix] <= bb.x_r and Y[iy] <= bb.y_u and Y[iy] >= bb.y_d:
                            p = Boundary_point(ix, iy, bb, X, Y)
                            points.append(p)
                            inner_points[iy,ix] = 2
                            break
                    if not inner_points[iy,ix] == 2:
                        for bb in brzeg:
                            if X[ix] == bb.x_l and Y[iy] <= bb.y_u and Y[iy] >= bb.y_d:
                                p = Boundary_point(ix, iy, bb, X, Y)
                                points.append(p)
                                inner_points[iy,ix] = 2
                                break
                            elif X[ix] == bb.x_r and Y[iy] <= bb.y_u and Y[iy] >= bb.y_d:
                                p = Boundary_point(ix, iy, bb, X, Y)
                                points.append(p)
                                inner_points[iy,ix] = 2
                                break
                            elif Y[iy] == bb.y_u and X[ix] <= bb.x_r and X[ix] >= bb.x_l:
                                p = Boundary_point(ix, iy, bb, X, Y)
                                points.append(p)
                                inner_points[iy,ix] = 2
                                break
                            elif Y[iy] == bb.y_d and X[ix] <= bb.x_r and X[ix] >= bb.x_l:
                                p = Boundary_point(ix, iy, bb, X, Y)
                                points.append(p)
                                inner_points[iy,ix] = 2
                                break
                    elif inner_points[iy,ix] == 2: # ujednolicanie
                        for bb in brzeg:
                            if X[ix] == bb.x_l and Y[iy] < bb.y_u and Y[iy] > bb.y_d:
                                inner_points[iy,ix] = 0
                                break
                            elif X[ix] == bb.x_r and Y[iy] < bb.y_u and Y[iy] > bb.y_d:
                                inner_points[iy,ix] = 0
                                break
                            elif Y[iy] == bb.y_u and X[ix] < bb.x_r and X[ix] > bb.x_l:
                                inner_points[iy,ix] = 0
                                break
                            elif Y[iy] == bb.y_d and X[ix] < bb.x_r and X[ix] > bb.x_l:
                                inner_points[iy,ix] = 0
                                break
        # p new 
        points_new = []
        for p in points:
            if inner_points[p.iy,p.ix] == 2:
                points_new.append(p)
                
                

                                
        # pomocnicze do Neumanna
        normal_deriv_direction_x = np.zeros((Y.size,X.size)) 
        normal_deriv_direction_y = np.zeros((Y.size,X.size)) 
        
        for p in points:
            # obsluga punktow brzegowych - 1D
            if p.ix == 0:
                normal_deriv_direction_x[p.iy,p.ix] = 1 
                normal_deriv_direction_y[p.iy,p.ix] = 0 
                #sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy,p.ix+1]
            elif p.ix == X.size-1:
                normal_deriv_direction_x[p.iy,p.ix] = -1 
                normal_deriv_direction_y[p.iy,p.ix] = 0 
                #sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy,p.ix-1]
            if p.iy == 0:
                normal_deriv_direction_x[p.iy,p.ix] = 0 
                normal_deriv_direction_y[p.iy,p.ix] = 1 
                #sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy+1,p.ix]
            elif p.ix == X.size-1:
                normal_deriv_direction_x[p.iy,p.ix] = 0 
                normal_deriv_direction_y[p.iy,p.ix] = -1 
                #sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy-1,p.ix]
            # obsluga pozostalych punktow brzegowych
            elif X[p.ix] == p.brzeg_element.x_r:
                normal_deriv_direction_x[p.iy,p.ix] = 1 
                normal_deriv_direction_y[p.iy,p.ix] = 0 
                #sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy,p.ix+1]
            elif X[p.ix] == p.brzeg_element.x_l:
                normal_deriv_direction_x[p.iy,p.ix] = -1 
                normal_deriv_direction_y[p.iy,p.ix] = 0 
                #sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy,p.ix-1]
            elif Y[p.iy] == p.brzeg_element.y_u:
                normal_deriv_direction_x[p.iy,p.ix] = 0
                normal_deriv_direction_y[p.iy,p.ix] = -1 
                #sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy-1,p.ix]
            elif Y[p.iy] == p.brzeg_element.y_d:
                normal_deriv_direction_x[p.iy,p.ix] = 0 
                normal_deriv_direction_y[p.iy,p.ix] = 1 
                #sol[p.iy,p.ix] = normal_deriv[p.iy,p.ix]*h + sol[p.iy+1,p.ix]
            else:
                print("Warning")
        
        self.normal_deriv_direction_x = normal_deriv_direction_x.astype(int)      
        self.normal_deriv_direction_y = normal_deriv_direction_y.astype(int)                    
        self.X = X
        self.Y = Y
        self.inner_points = inner_points
        self.boundary_points = points_new
        self.h = h
