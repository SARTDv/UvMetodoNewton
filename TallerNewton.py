import math

# Estas librerias deben estar instaladas, si no no va a funcionar
import numpy as np # type: ignore 
import sympy as sp # type: ignore 
from numpy.linalg import inv # type: ignore 

#****************************
#*         COMPUTER         *
#*         PROBLEMS         *
#****************************

# -----------------------------------------PUNTO 1-----------------------------------------
# se define la ecuacion como y = x - tan(x)
def f(x):
    return x - math.tan(x)

# Primera derivada de la funcion y'(x) = 1 - sec^2(x)
def fDerivada(x):
    return 1 - (1 / math.cos(x))**2

def newton(funcion,derivada,x0, tol=1e-7, max_iter=1000):
    x = x0
    for i in range(max_iter):
        fx = funcion(x)
        fpx = derivada(x)

        if fx == 0: return fx

        if abs(fpx) < 1e-10:  # Evitar divisiones por valores muy pequeños
            return None 
        
        # Metodo
        x_n = x - fx / fpx
        
        # Limitar el tamaño del salto para evitar cambios muy grandes
        if abs(x_n - x) > 1.0:
            x_n = x + (0.3 if x_n > x else -0.3)
        
        # Revisa que esté dentro del rango de tolerancia
        if abs(x_n - x) < tol:
            return x_n
        
        x = x_n
    
    print(f"Advertencia: El método de Newton no convergió para la estimación inicial x0 = {x0}")
    return None  


# En el problema ofrecen dos posibles raices
def punto1():
    iniciales = [4.5, 7.7]
    raices = []

    for posible in iniciales:
        try:
            raiz = newton(posible)
            raices.append(raiz)
        except ValueError as e:
            print(f"El método de Newton falla para la raiz: {posible}: {e}")

    print("Raices cerca de los valores iniciales:", raices)

# -----------------------------------------PUNTO 2-----------------------------------------
def encontrarNraices(n):
    raicesEncontradas = []
    posible = 4.5
    while len(raicesEncontradas) < n:
        try:
            raiz = newton(f,fDerivada,posible)

            # Se debe asegurar que las raices no son muy cercanas
            if not any(abs(raiz - r) < 1e-5 for r in raicesEncontradas):
                raicesEncontradas.append(raiz)

        except ValueError as e:
            e   # Genera muchos errores por la ejecucion pero no se desean ver en pantalla o consola
        
        posible += math.pi  #Se usa pi por la naturaleza de la funcion tan(x)
    
    return raicesEncontradas

# Muestra 10 raices de la funcion
def punto2():
    r = encontrarNraices(10)
    print(f"10 raices de x = tan(x) son: {r}")


# -----------------------------------------PUNTO 3-----------------------------------------

# Para la funcion f(x)=x⁻² * tan(x)
def punto3():
    def fex(x):
        return (1 / x**2) * math.tan(x)
    # Se necesita una variable simbolica para las derivadas
    x = sp.symbols('x')
    f = x**(-2) * sp.tan(x)

    # Son la primera y segunda derivada usando sympy
    fp = sp.diff(f, x) 
    fdp = sp.diff(fp, x)


    # Convierte las derivadas en metodos numericos
    fpNumerica = sp.lambdify(x, fp, 'numpy')
    fdpNumerica = sp.lambdify(x, fdp, 'numpy')

    x0 = 1.5

    minimo_positivo = newton(fpNumerica, fdpNumerica, x0)

    print(f"El punto mínimo positivo es aproximadamente ({minimo_positivo} , {fex(minimo_positivo)}")


# -----------------------------------------PUNTO 4-----------------------------------------

def punto4():
    def f(x):
        return x**3 + 3*x - 5*x**2 - 7  # Esta funcion solo tiene una raiz real

    # Derivada
    def fp(x):
        return 3*x**2 - 10*x + 3
    
    x0 = 5
    x = x0

    for i in range(10):
        fx = f(x)
        fpx = fp(x)
        
        if fpx == 0:
            print("La derivada es cero. No se puede continuar.")
            return None
        
        # Actualización de Newton
        x_n = x - fx / fpx
                
        x = x_n
    
    print(f"Despues de 10 iteraciones: x = {x_n}" )

# -----------------------------------------PUNTO 5-----------------------------------------

def punto5():
    
    def f(x):
        return 2*x**4 + 24*x**3 + 61*x**2 - 16*x + 1  # Esta funcion solo tiene una raiz real

    # Derivada
    def fp(x):
        return 8*x**3 + 72*x**2 + 122*x - 16
    
    # Aproximación inicial
    initial_guess_1 = 0.1
    initial_guess_2 = -0.1  # Para encontrar la otra raíz cercana
    initial_guess_3 = 0.2  # si las dos raices son iguales o muy cercanas lo hace por el otro lado

    raiz1 = newton(f,fp,initial_guess_1)
    raiz2 = newton(f,fp,initial_guess_2)

    if abs(raiz1-raiz2) < 1e-4:
        raiz2 = newton(f,fp,initial_guess_3)

    print(f"Primera raíz cercana a 0.1: {raiz1}")
    print(f"Segunda raíz cercana a 0.1: {raiz2}")


# -----------------------------------------PUNTO 13-----------------------------------------
def newtonNoLineal(f1,f2,iteraciones,x0,y0):
    x = sp.symbols('x')
    y = sp.symbols('y')

    # Derivadas parciales
    f1_x = sp.diff(f1, x)
    f1_y = sp.diff(f1, y)
    f2_x = sp.diff(f2, x)
    f2_y = sp.diff(f2, y)

    # Hace que las funciones sean métodos numericos (funciones)
    f1_numerica = sp.lambdify((x, y), f1, 'numpy')
    f2_numerica = sp.lambdify((x, y), f2, 'numpy')
    f1_x_numerica = sp.lambdify((x, y), f1_x, 'numpy')
    f1_y_numerica = sp.lambdify((x, y), f1_y, 'numpy')
    f2_x_numerica = sp.lambdify((x, y), f2_x, 'numpy')
    f2_y_numerica = sp.lambdify((x, y), f2_y, 'numpy')

    x_n = x0
    y_n = y0

    for i in range (iteraciones):
        J = np.array([ [f1_x_numerica(x_n,y_n), f1_y_numerica(x_n,y_n)],
                       [f2_x_numerica(x_n,y_n), f2_y_numerica(x_n,y_n)]
                       ])
        F = np.array([f1_numerica(x_n,y_n),f2_numerica(x_n,y_n)])
        vectorSln = inv(J).dot(F)
        x_n = vectorSln[0]
        y_n = vectorSln[1]

    print(f"Despues de {iteraciones} iteraciones el vector H equivale a: ({x_n} , {y_n})")

def punto13():

    # Se necesita una variable simbolica para las derivadas
    x = sp.symbols('x')
    y = sp.symbols('y')

    f1 = 1 + x**2 - y**2 + sp.exp(x) * sp.cos(y)
    f2 = 2*x*y + sp.exp(x) * sp.sin(y)

    newtonNoLineal(f1,f2,5,-1,4)

# -----------------------------------------PUNTO 14-----------------------------------------

def punto14a():
    x = sp.symbols('x')
    y = sp.symbols('y')

    f1 = 4*y**2 + 4*y + 52*x - 19
    f2 = 169*x**2 + 3*y**2 + 111*x - 10*y - 10

    newtonNoLineal(f1,f2,50,0,0)

def punto14b():
    x = sp.symbols('x')
    y = sp.symbols('y')

    f1 = x + sp.exp(-1/x) + y**3
    f2 = x**2 + 2*x*y - y**2 + sp.tan(x)

    newtonNoLineal(f1,f2,50,1,1)
   