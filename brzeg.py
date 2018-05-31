



class Brzeg_protokat:
    def __init__(self, x1, x2, y1, y2, func_v, func_u):
        # polozenie prostokota
        self.x_l = x1
        self.x_r = x2
        self.y_d = y1
        self.y_u = y2
        # funkcje
        self.func_v = func_v
        self.func_u = func_u

    def boundary_v(self, x, y, t):
        # na podstawie x,y nastempuje wyszunkanie na ktorym boku znajduje sie punkt
        # oraz zwraca funkcje v z brzegu
        if x == self.x_r:
            return self.func_v[0](x,y,t)
        elif y == self.y_u:
            return self.func_v[1](x,y,t)
        elif x == self.x_l:
            return self.func_v[2](x,y,t)
        elif y == self.y_d:
            return self.func_v[3](x,y,t)
    def boundary_u(self, x, y, t):
        # na podstawie x,y nastempuje wyszunkanie na ktorym boku znajduje sie punkt
        # oraz zwraca funkcje u z brzegu
        if x == self.x_r:
            return self.func_u[0](x,y,t)
        elif y == self.y_u:
            return self.func_u[1](x,y,t)
        elif x == self.x_l:
            return self.func_u[2](x,y,t)
        elif y == self.y_d:
            return self.func_u[3](x,y,t)

    
class Brzeg_1D:
    def __init__(self, x1, x2, y1, y2, func_v, func_u):
        self.x_l = x1
        self.x_r = x2
        self.y_d = y1
        self.y_u = y2
        # funkcje
        self.func_v = func_v
        self.func_u = func_u

    def boundary_v(self, x, y, t):
        # zwraca funkcje v z brzegu
        return self.func_v(x,y,t)
    def boundary_u(self, x, y, t):
        # zwraca funkcje u z brzegu
        return self.func_u(x,y,t)
