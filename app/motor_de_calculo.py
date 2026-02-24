import numpy as np



# Constantes
R = 83.14472 # J / (mol K)
ERROR_PERMITIDO = 1e-6 # |valor_real - valor_obtenido| / valor_real


# NOTA: En los comentarios se usa 'c' para denotar propiedad de compuesto y 't' para denotar variable termodinamica



def cardano_vectorizado(a : np.ndarray, b : np.ndarray, c : np.ndarray, d : np.ndarray):
    """
    Encuentra las raíces reales de un polinomios de tercer grado usando el metodo de cardano.
    funciona con Numpy de manera vectorizada
    El polinomio tiene la forma : 
    P(x) = a*x^3 + b*x^2 + c*x + d
    
    Args:
        a, b, c, d (1, T): vectores de coeficientes del polinomio.

    Returns:
        array_soluciones (3, T) : Matriz que contiene las raices reales del polinomio.
    """
    array_soluciones = np.empty((3, a.shape[1]), dtype=np.float64) # La matriz de soluciones a devolver

    p = (3*a*c - b**2) / (3*a**2)
    q = (2*b**3 - 9*a*b*c + 27*a**2*d) / (27*a**3)
    dis = q**2 + (4*p**3) / 27
    b_entre_3a = b / (3 * a)

    # CASO 1 -> UNICA RAIZ REAL
    caso_1 = dis > 1e-12
    dis_caso = dis[caso_1]
    b_entre_3a_caso = b_entre_3a[caso_1]
    q_caso = q[caso_1]

    u = np.cbrt(
        (-q_caso + np.sqrt(dis_caso)) / 2
    )
    v = np.cbrt(
        (-q_caso - np.sqrt(dis_caso)) / 2
    )
    z_caso_1 = u + v - b_entre_3a_caso
    z_caso_1 = np.vstack([z_caso_1, np.full(z_caso_1.shape, np.nan), np.full(z_caso_1.shape, np.nan)])
    
    # CASO 2 -> Se fragmenta en otros dos casos
    caso_2 = np.isclose(dis, 0, atol=1e-12)
    p_y_q_son_cero = np.logical_and(np.isclose(p, 0, atol=1e-12), np.isclose(q, 0, atol=1e-12))

    # CASO 2_1 -> Tres soluciones reales identicas -> Det = 0 y p = 0 y q = 0
    caso_2_1 = np.logical_and(caso_2, p_y_q_son_cero)
    
    z_caso_2_1 = -b_entre_3a[caso_2_1]
    z_caso_2_1 = np.vstack([z_caso_2_1, z_caso_2_1, z_caso_2_1])

    # CASO 2_2 -> Tres soluciones reales, dos iguales -> Det = 0 y (p != 0 o q != 0)
    caso_2_2 = np.logical_and(caso_2, np.logical_not(caso_2_1))
    p_caso = p[caso_2_2]
    q_caso = q[caso_2_2]
    b_entre_3a_caso = b_entre_3a[caso_2_2]

    z1_caso_2_2 = 3 * q_caso / p_caso - b_entre_3a_caso
    z2_caso_2_2 = -3 * q_caso / (2 * p_caso) - b_entre_3a_caso
    z_caso_2_2 = np.vstack([z1_caso_2_2, z2_caso_2_2, z2_caso_2_2])
    z_caso_2_2 = np.sort(z_caso_2_2, axis=0)

    # CASO 3 -> Tres soluciones reales todas diferentes -> dis < 0
    caso_3 = dis < -1e-12
    dis_caso = dis[caso_3]
    p_caso = p[caso_3]
    q_caso = q[caso_3]
    b_entre_3a_caso = b_entre_3a[caso_3]

    theta = np.arccos(
        (3 * q_caso) / (2 * p_caso) * np.sqrt(-3 / p_caso)
    )
    z1_caso_3 = 2 * np.sqrt(-p_caso / 3) * np.cos(theta / 3) - b_entre_3a_caso
    z2_caso_3 = 2 * np.sqrt(-p_caso / 3) * np.cos((theta + 2*np.pi) / 3) - b_entre_3a_caso
    z3_caso_3 = 2 * np.sqrt(-p_caso / 3) * np.cos((theta + 4*np.pi) / 3) - b_entre_3a_caso
    z_caso_3 = np.vstack([z2_caso_3, z3_caso_3, z1_caso_3])

    # Juntar todos los casos en un solo array
    # vamos a usar los siguientes numeros para distinguir entre casos:
    # CASO_1 : 0, CASO_2_1 : 1, CASO_2_2 : 2, CASO_3 : 3

    # Creamos vector que contiene todos los casos
    casos = 1*caso_2_1 + 2*caso_2_2 + 3*caso_3
    casos = casos.ravel()

    # Dependiendo de cada caso insertamos sus raíces respectivas
    array_soluciones[:, casos == 0] = z_caso_1
    array_soluciones[:, casos == 1] = z_caso_2_1
    array_soluciones[:, casos == 2] = z_caso_2_2
    array_soluciones[:, casos == 3] = z_caso_3

    return array_soluciones


def cardano_vectorizado_2(a, b, c, d):
    """
    Encuentra las raices de un polinomio de tercer grado P(x) = ax^3 + bx^2 + cx + d de manera vecotorizada.
    Args:
        a, b, c, d (numpy.ndarray): Arrays de coeficientes del polinomio.
    Returns
        raiz_1, raiz_2, raiz_3 (tuple): tupla de np.ndarrays de las raices del polinomio.
        Donde raiz_1 < raiz_2 < raiz_3.
    Notas:
        En lugar de devolver raices complejas devuelve np.nan
    """
    # Nos asegumramos que a, b, c, d sean de la misma forma. Les aplicamos broadcasting
    a, b, c, d = np.broadcast_arrays(a, b, c, d)

    # Generamos arrays con la forma final que vamos a devolver
    raiz_1 = np.empty(a.shape, dtype=np.float64)
    raiz_2 = np.empty(a.shape, dtype=np.float64)
    raiz_3 = np.empty(a.shape, dtype=np.float64)

    # Calcuamos parametros del metodo de cardano
    p = (3*a*c - b**2)/(3*a**2)
    q = (2*b**3 - 9*a*b*c + 27*a**2*d) / (27*a**3)
    dis = q**2 + (4*p**3) / 27

    # Calculamos algunas mascaras booleanas
    p_y_q_son_cero = np.logical_and(np.isclose(p, 0, atol=1e-12), np.isclose(q, 0, atol=1e-12))
    dis_es_cero = np.isclose(dis, 0, atol=1e-12)

    # Dividimos el metodo de cardano en 4 casos
    caso_1 = dis > 1e-12
    caso_2 = np.logical_and(dis_es_cero, p_y_q_son_cero)
    caso_3 = np.logical_and(dis_es_cero, np.logical_not(p_y_q_son_cero))
    caso_4 = dis < -1e-12

    # Caso 1
    q_caso, dis_caso = q[caso_1], dis[caso_1]
    u = np.cbrt((-q_caso + np.sqrt(dis_caso)) / 2)
    v = np.cbrt((-q_caso - np.sqrt(dis_caso)) / 2)
    raiz_1[caso_1] = u + v - b[caso_1] / (3 * a[caso_1])
    raiz_2[caso_1], raiz_3[caso_1] = np.nan, np.nan

    # Caso 2
    raiz_triple = - b[caso_2] / (3*a[caso_2])
    raiz_1[caso_2], raiz_2[caso_2], raiz_3[caso_2] = raiz_triple, raiz_triple, raiz_triple

    # Caso 3
    termino_1 = np.cbrt(-q[caso_3] / 2)
    termino_2 = b[caso_3] / (3 * a[caso_3])
    raiz_unica = 2 * termino_1 -  termino_2
    raiz_doble = - termino_1 - termino_2
    # Ordenamos raices de menor a mayor
    z1 = np.where(raiz_unica < raiz_doble, raiz_unica, raiz_doble)
    z2 = np.where(raiz_doble > raiz_unica, raiz_doble, raiz_unica)
    raiz_1[caso_3], raiz_2[caso_3], raiz_3[caso_3] = z1, z2, z2

    # Caso 4
    p_caso, q_caso = p[caso_4], q[caso_4]
    theta = np.acos(3*q_caso / (2*p_caso) * np.sqrt(-3 / p_caso))
    termino_1 = 2 * np.sqrt(-p_caso / 3)
    termino_2 = b[caso_4] / (3*a[caso_4])
    raiz_1[caso_4] = termino_1 * np.cos((theta + 2*np.pi) / 3) - termino_2
    raiz_2[caso_4] = termino_1 * np.cos((theta + 4*np.pi) / 3) - termino_2
    raiz_3[caso_4] = termino_1 * np.cos(theta / 3) - termino_2

    return (raiz_1, raiz_2, raiz_3)


def resolver_newton(funcion, objetivo, suposicion, max_iter=100, tol=ERROR_PERMITIDO, derivada=None):
    """
    Intenta encontrar un arreglo X tal que: funcion(X) = objetivo usando el metodo de newton de manera vectorizada.
    Args:
        funcion: funcion que toma un arreglo y lo transofrma a otro. Las funciones de cada componente deben ser independientes
        entre si.
        inicio: Guess inicial desde donde se aplicara el algoritmo de newton
        objetivo: vector objetivo 
        derivada: derivada de la funcion (opcional)
    """
    # Si no se propociono la derivada de la funcion la definimos numericamente
    if derivada is None:
        def derivada(x):
            return (funcion(x + 1e-6) - funcion(x - 1e-6)) / 2e-6

    # Iniciamos bucle del metodo de newton.
    for _ in range(max_iter // 10):

        # Iteramos 10 veces el metodo de newton antes de checar que hayamos encontrado la solución.
        for _ in range(10):
            fx = funcion(suposicion) - objetivo
            pendiente = derivada(suposicion)

            # Evitamos divisiones por cero 
            divisiones_invalidas = np.abs(pendiente) < 1e-12
            pendiente[divisiones_invalidas] = np.nan

            # Actualizamos suposicion
            suposicion = suposicion - fx / pendiente

        # Checamos si todas las soluciones son validas.
        soluciones_validas = np.isclose(fx, 0, atol=tol)
        if np.all(soluciones_validas):
            return suposicion
        
    # En este punto significa que se itero el metodo mas que el limite, devolvemos valores que si convergen
    soluciones_invalidas = np.logical_not(soluciones_validas)
    suposicion[soluciones_invalidas] = np.nan
    return suposicion


def resolver_newton_raphson(funcion, objetivo, suposicion, max_iter=100, tol=ERROR_PERMITIDO, jacobiano=None):
    """
    Aplica el metodo de newton raphson a una funcion tal que evaluada en objetivo sea igual a suposicion.
    """
    # Si no se especifica un jacobiano de transformacion lineal, calcularlo numericamente
    if not jacobiano:
        tamaño_entradas = len(suposicion)
        tamaño_salidas = len(objetivo)
        identidad = np.identity(tamaño_entradas)

        def jacobiano(x):
            jac = np.empty((tamaño_salidas, tamaño_entradas), dtype=np.float64)

            for i in range(tamaño_entradas):
                step = 1e-6 * identidad[:, i]
                dfdx = ((funcion(x + step) - funcion(x - step)) / 2e-6)
                jac[:, i] = dfdx

            return jac
        
    # Iniciamos el bucle del metodo de Newton Raphson
    for _ in range(max_iter // 5):

        # Itermaos 5 veces antes de checar que hayamos encontrado la solucion
        for _ in range(5):
            fx = funcion(suposicion) - objetivo
            j = jacobiano(suposicion)

            sol_local = np.linalg.solve(j, -fx)
            suposicion = suposicion + sol_local

        # Checamos si encontramos la solucion
        if np.linalg.norm(funcion(suposicion) - objetivo) <= ERROR_PERMITIDO:
            return suposicion
        

def obtener_a_alfa_mezcla(y, a, alfa, k):
    """
    Calcula a_alfa_mezcla, teniendo en cuenta los factores de interacción binaria.
    Args:
        Variables Termodinamicas ---------------------
        alfa (C, T) : Valor 'alfa' proviente de la ecuación cubica de estado.

        Propiedades de Compuesto ---------------------
        y (C, 1): Composiciones del compuesto i-esimo.
        a (C, 1): Constante 'a' proviente de la ecuación cubica de estado.
        k (C, C) : Matriz del factor de interacción binaria entre el compiesto-i y el compuesto-j.

    Returns:
        a_alfa_mezcla (1, T): Vector de constante a_alfa_mezcla a las diferentes temperaturas.
    """
    # Los calculos se llevaran en los tensores de la forma:
    # (C, i, T) = (propiedad_compuesto, propiedad_compuesto, variable_termodinamica)

    # Generamos tensores aptos para broadcasting
    yi = y[:, np.newaxis]
    yj = y[np.newaxis, :]
    ai = a[:, np.newaxis]
    aj = a[np.newaxis, :]
    alfa_i = alfa[:, np.newaxis, :]
    alfa_j = alfa[np.newaxis, :, :]
    kij = k[:, :, np.newaxis]

    # Tensor de todos los elementos de a alfa de la mezcla
    a_alfa_mezcla = yi * yj * np.sqrt(ai * aj * alfa_i * alfa_j) * (1 - kij)

    # Sumamos sobre los ejes de las composiciones (C, i, T)
    a_alfa_mezcla = np.sum(np.sum(a_alfa_mezcla, axis=0), axis=0)

    # Regresamos vector con shape = (1, T)
    return a_alfa_mezcla[np.newaxis, :]


def obtener_a_mezcla(y, a, k):
    """
    Calcula a_mezcla teniendo en cuenta los factores de interacción binaria
    Args:
        Propiedades de Compuesto ---------------------
        y (C, 1): Composiciones del compuesto i-esimo.
        a (C, 1): Constante 'a' proviente de la ecuación cubica de estado.
        k (C, C) : Matriz del factor de interacción binaria entre el compiesto-i y el compuesto-j.
    Returns:
        a_mezcla (1, 1): Constante a_mezcla
    """
    yi = y
    yj = y[np.newaxis, :, 0]
    ai = a
    aj = a[np.newaxis, :, 0]
    a_mezcla = np.sum(
        yi * yj * np.sqrt(ai * aj) * (1 - k)
        )
    return a_mezcla


def peng_robinson_presion(temperatura, volumen, composiciones, pc, tc, w, k):
    """
    Calcula la presion de una mezcla utilizando la ecuacion de peng robinson
    Args:
        Variables Termodinamicas ---------------------------------------------------------
        temperatura (1, T) : vector de temperaturas en K.
        volumen (1, T) : vector de volumenes molares en mL / mol.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        w (C, 1) : vector de factores acentricos de cada compuesto de la mezcla.
        kij (C, C) : matriz de factor de interacción binaria entre compuesto-i y compuesto-j

    Returns:
        presion (1, T): vector de las presiones por cada variable termodinamica
    """

    # Calculamos constantes de la mezcla
    a = (0.45724 * R**2 * tc**2) / pc
    b = (0.0778 * R * tc) / pc
    pw = 0.37464 + 1.54226 * w - 0.26992 * w**2
    alfa = (1 + pw * (1 - np.sqrt(temperatura / tc)))**2
    b_mezcla = np.sum(composiciones * b)
    a_alfa_mezcla = obtener_a_alfa_mezcla(composiciones, a, alfa, k)
    
    # Calculamos la presion utilizando directamente la ecuacion de peng robinson
    presion = (R * temperatura) / (volumen - b_mezcla) - a_alfa_mezcla / (volumen**2 + 2 * volumen * b_mezcla - b_mezcla**2)
    return presion


def peng_robinson_temperatura(presion, volumen, composiciones, pc, tc, w, k):
    """
    Calcula la temperatura de una mezcla utilizando la ecuacion de peng robinson y el metodo iterativo de
    newton.
    Args:
        Variables Termodinamicas ---------------------------------------------------------
        presion (1, T) : vector de presion en bar.
        volumen (1, T) : vector de volumenes molares en mL / mol.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        w (C, 1) : vector de factores acentricos de cada compuesto de la mezcla.
        k (C, C) : matriz de factor de interacción binaria entre compuesto-i y compuesto-j

    Returns:
        presion (T,): vector de las presiones por cada variable termodinamica
    """

    # Calculamos constantes de la mezcla
    a = (0.45724 * R**2 * tc**2) / pc
    b = (0.0778 * R * tc) / pc
    b_mezcla = np.sum(composiciones * b)
    pw = 0.37464 + 1.54226 * w - 0.26992 * w**2
    
    # Generamos tensores aptos para broadcasting en el calculo de a_alfa_mezcla y su derivada
    # Los calculos se llevaran en el formato: shape = (C1, C2, T)
    yi = composiciones[:, np.newaxis]
    yj = composiciones[np.newaxis, :]
    ai = a[:, np.newaxis]
    aj = a[np.newaxis, :]
    kij = k[:, :, np.newaxis]

    # Calculamos constante de a_alfa_mezcla
    pepe = yi * yj * np.sqrt(ai * aj) * (1 - kij)

    # Definimos funcion para calcular la derivada de a_alfa_mezcla respecto a la temperatura
    def derivada_a_alfa_mezcla(temp):
        # Calculamos valores dependientes de la temperatura
        raiz_temperatura_reducida = np.sqrt(temp / tc)
        alfa = (1 + pw * (1 - raiz_temperatura_reducida))**2 # (C, T)
        derivada_alfa = 2 * pw * (pw * (raiz_temperatura_reducida - 1) - 1) / (tc * raiz_temperatura_reducida)

        # Aplicamos el formato de las operaciones tensoriales (C1, C2, T)
        alfa_i = alfa[:, np.newaxis]
        alfa_j = alfa[np.newaxis, :]
        derivada_alfa_j = derivada_alfa[np.newaxis, :]

        # Calculamos todos los elementos de derivada_a_alfa_mezcla
        derivada_a_alfa_mezcla = pepe * alfa_i * derivada_alfa_j / np.sqrt(alfa_i * alfa_j)

        # Sumamos sobre los ejes de los compuestos (C1, C2, T)
        return np.sum(np.sum(derivada_a_alfa_mezcla, axis=0), axis=0)
    
    # Definimos funcion para calcular la derivada de la ecuacion de peng robinson dP / dT
    def dp_dt_peng_robinson(temp):
        return R / (volumen - b_mezcla) - derivada_a_alfa_mezcla(temp) / (volumen**2 + 2 * volumen * b_mezcla - b_mezcla**2)
    
    # Definimos función de peng_robinson que solo dependa de la temperatura
    def peng_robinson(temp):
        return peng_robinson_presion(temp, volumen, composiciones, pc, tc, w, k)
    
    # Suposicion inicial del vector de temperaturas
    temperatura = presion * volumen / R

    # Resolvemos con el método de newton
    return resolver_newton(peng_robinson, presion, temperatura, derivada=dp_dt_peng_robinson)
    

def peng_robinson_volumen(presion, temperatura, composiciones, pc, tc, w, k):
    """
    Calcula los volumenes molares de una mezcla usando la ecuación de peng robinson.
    Es una funcion de mezcla.
    Args:
        Variables Termodinamicas ---------------------------------------------------------
        presion (1, T) : vector de presiones en bar. 
        temperatura (1, T) : vector de temperaturas en K.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        w (C, 1) : vector de factores acentricos de cada compuesto de la mezcla.
        kij (C, C) : matriz de factor de interacción binaria entre compuesto-i y compuesto-j

    Returns:
        volumen (3, T) : Matriz de volumenes liquido, prohibido y vapor para cada variable termodinamica
    """

    # Calculamos constantes de la mezcla
    a = (0.45724 * R**2 * tc**2) / pc
    b = (0.0778 * R * tc) / pc
    pw = 0.37464 + 1.54226 * w - 0.26992 * w**2
    alfa = (1 + pw * (1 - np.sqrt(temperatura / tc)))**2
    b_mezcla = np.sum(composiciones * b)
    a_alfa_mezcla = obtener_a_alfa_mezcla(composiciones, a, alfa, k)
    
    # Calculamos coeficientes para la forma polinomial del volumen
    termino_cubo = presion
    termino_cuadrado = b_mezcla * presion - R * temperatura
    termino_lineal = a_alfa_mezcla - 2 * R * temperatura * b_mezcla - 3 * presion * b_mezcla**2
    termino_independiente = presion * b_mezcla**3 + R * temperatura * b_mezcla**2 - a_alfa_mezcla * b_mezcla

    # Calculamos los volumenes con shape = (3, T)
    volumen = cardano_vectorizado(termino_cubo, termino_cuadrado, termino_lineal, termino_independiente)
    return volumen
    

def soave_presion(temperatura, volumen, composiciones, pc, tc, w, k):
    """
    Calcula la presion de una mezcla utilizando la ecuacion de Soave
    Args:
        Variables Termodinamicas ---------------------------------------------------------
        temperatura (1, T) : vector de temperaturas en K.
        volumen (1, T) : vector de volumenes molares en mL / mol.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        w (C, 1) : vector de factores acentricos de cada compuesto de la mezcla.
        kij (C, C) : matriz de factor de interacción binaria entre compuesto-i y compuesto-j

    Returns:
        presion (1, T): vector de las presiones por cada variable termodinamica
    """

    # Calculamos constantes de la mezcla
    a = (0.427480 * R**2 * tc**2) / pc
    b = (0.08664 * R * tc) / pc
    pw = 0.48508 + 1.55171 * w - 0.15613 * w**2
    alfa = (1 + pw * (1 - np.sqrt(temperatura / tc)))**2
    b_mezcla = np.sum(composiciones * b)
    a_alfa_mezcla = obtener_a_alfa_mezcla(composiciones, a, alfa, k)

    # Calculamos la presion usando la ecuacion de Soave
    presion = R * temperatura / (volumen - b_mezcla) - a_alfa_mezcla / (volumen * (volumen + b_mezcla))
    return presion


def soave_temperatura(presion, volumen, composiciones, pc, tc, w, k):
    """
    Calcula la temperatura de una mezcla utilizando la ecuacion de Soave y el metodo iterativo de
    newton.
    Args:
        Variables Termodinamicas ---------------------------------------------------------
        presion (1, T) : vector de presion en bar.
        volumen (1, T) : vector de volumenes molares en mL / mol.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        w (C, 1) : vector de factores acentricos de cada compuesto de la mezcla.
        k (C, C) : matriz de factor de interacción binaria entre compuesto-i y compuesto-j

    Returns:
        presion (T,): vector de las presiones por cada variable termodinamica
    """

    # Calculamos constantes de la mezcla
    a = (0.427480 * R**2 * tc**2) / pc
    b = (0.08664 * R * tc) / pc
    pw = 0.48508 + 1.55171 * w - 0.15613 * w**2
    b_mezcla = np.sum(composiciones * b)
    
    # Generamos tensores aptos para broadcasting en el calculo de a_alfa_mezcla y su derivada
    # Los calculos se llevaran en el formato: shape = (C1, C2, T)
    yi = composiciones[:, np.newaxis]
    yj = composiciones[np.newaxis, :]
    ai = a[:, np.newaxis]
    aj = a[np.newaxis, :]
    kij = k[:, :, np.newaxis]

    # Calculamos constante de a_alfa_mezcla
    pepe = yi * yj * np.sqrt(ai * aj) * (1 - kij)

    # Definimos funcion para calcular la derivada de a_alfa_mezcla respecto a la temperatura
    def derivada_a_alfa_mezcla(temp):
        # Calculamos valores dependientes de la temperatura
        raiz_temperatura_reducida = np.sqrt(temp / tc)
        alfa = (1 + pw * (1 - raiz_temperatura_reducida))**2 # (C, T)
        derivada_alfa = 2 * pw * (pw * (raiz_temperatura_reducida - 1) - 1) / (tc * raiz_temperatura_reducida)

        # Aplicamos el formato de las operaciones tensoriales (C1, C2, T)
        alfa_i = alfa[:, np.newaxis]
        alfa_j = alfa[np.newaxis, :]
        derivada_alfa_j = derivada_alfa[np.newaxis, :]

        # Calculamos todos los elementos de derivada_a_alfa_mezcla
        derivada_a_alfa_mezcla = pepe * alfa_i * derivada_alfa_j / np.sqrt(alfa_i * alfa_j)

        # Sumamos sobre los ejes de los compuestos (C1, C2, T)
        return np.sum(np.sum(derivada_a_alfa_mezcla, axis=0), axis=0)
    
    # Definimos funcion para calcular la derivada de la ecuacion de Soave dP / dT
    def dp_dt_soave(temp):
        return R / (volumen - b_mezcla) - derivada_a_alfa_mezcla(temp) / (volumen * (volumen + b_mezcla))
    
    # Definimos función de Soave que solo dependa de la temperatura
    def soave(temp):
        return soave_presion(temp, volumen, composiciones, pc, tc, w, k)
    
    # Suposicion inicial del vector de temperaturas
    temperatura = presion * volumen / R

    # Resolvemos con el método de newton
    return resolver_newton(soave, presion, temperatura, derivada=dp_dt_soave)


def soave_volumen(presion, temperatura, composiciones, pc, tc, w, k):
    """
    Calcula los volumenes molares de una mezcla usando la ecuación de Soave.
    Es una funcion de mezcla.
    Args:
        Variables Termodinamicas ---------------------------------------------------------
        presion (1, T) : vector de presiones en bar. 
        temperatura (1, T) : vector de temperaturas en K.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        w (C, 1) : vector de factores acentricos de cada compuesto de la mezcla.
        kij (C, C) : matriz de factor de interacción binaria entre compuesto-i y compuesto-j

    Returns:
        volumen (3, T) : Matriz de volumenes liquido, prohibido y vapor para cada variable termodinamica
    """

    # Calculamos constantes de la mezcla
    a = (0.427480 * R**2 * tc**2) / pc
    b = (0.08664 * R * tc) / pc
    pw = 0.48508 + 1.55171 * w - 0.15613 * w**2
    alfa = (1 + pw * (1 - np.sqrt(temperatura / tc)))**2
    b_mezcla = np.sum(composiciones * b)
    a_alfa_mezcla = obtener_a_alfa_mezcla(composiciones, a, alfa, k)
    
    # Calculamos coeficientes para la forma polinomial del volumen
    termino_cubo = presion
    termino_cuadrado = -R * temperatura
    termino_lineal = a_alfa_mezcla - R * temperatura * b_mezcla - presion * b_mezcla**2
    termino_independiente = -a_alfa_mezcla * b_mezcla

    # Calculamos los volumenes con shape = (3, T)
    volumen = cardano_vectorizado(termino_cubo, termino_cuadrado, termino_lineal, termino_independiente)
    return volumen


def van_der_waals_presion(temperatura, volumen, composiciones, pc, tc, k):
    """
    Calcula la presion de una mezcla utilizando la ecuacion de Van der Waals
    Args:
        Variables Termodinamicas ---------------------------------------------------------
        temperatura (1, T) : vector de temperaturas en K.
        volumen (1, T) : vector de volumenes molares en mL / mol.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        kij (C, C) : matriz de factor de interacción binaria entre compuesto-i y compuesto-j

    Returns:
        presion (1, T): vector de las presiones por cada variable termodinamica
    """
    # Obtenemos constantes de la ecuacion
    a = 27 * R**2 * tc**2 / (64 * pc)
    b = R * tc / (8 * pc)
    b_mezcla = np.sum(b * composiciones)
    a_mezcla = obtener_a_mezcla(composiciones, a, k)

    # Calculamos la preison usando la ecuacion de Van der Waals
    presion = R * temperatura / (volumen - b_mezcla) - a_mezcla / volumen**2
    return presion


def van_der_waals_temperatura(presion, volumen, composiciones, pc, tc, k):
    """
    Calcula la temperatura de una mezcla utilizando la ecuacion de Van der Waals.
    Args:
        Variables Termodinamicas ---------------------------------------------------------
        presion (1, T) : vector de presion en bar.
        volumen (1, T) : vector de volumenes molares en mL / mol.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        k (C, C) : matriz de factor de interacción binaria entre compuesto-i y compuesto-j

    Returns:
        presion (T,): vector de las presiones por cada variable termodinamica
    """

    # Obtenemos constantes de la ecuacion
    a = 27 * R**2 * tc**2 / (64 * pc)
    b = R * tc / (8 * pc)
    b_mezcla = np.sum(b * composiciones)
    a_mezcla = obtener_a_mezcla(composiciones, a, k)

    # Calculamos la temperatura
    temperatura = (presion + a_mezcla / volumen**2) * (volumen - b_mezcla) / R
    return temperatura
    

def van_der_waals_volumen(presion, temperatura, composiciones, pc, tc, k):
    """
    Calcula los volumenes molares de una mezcla usando la ecuación de Van der Waals.
    Es una funcion de mezcla.
    Args:
        Variables Termodinamicas ---------------------------------------------------------
        presion (1, T) : vector de presiones en bar. 
        temperatura (1, T) : vector de temperaturas en K.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        kij (C, C) : matriz de factor de interacción binaria entre compuesto-i y compuesto-j

    Returns:
        volumen (3, T) : Matriz de volumenes liquido, prohibido y vapor para cada variable termodinamica
    """

    # Obtenemos constantes de la ecuacion
    a = 27 * R**2 * tc**2 / (64 * pc)
    b = R * tc / (8 * pc)
    b_mezcla = np.sum(b * composiciones)
    a_mezcla = obtener_a_mezcla(composiciones, a, k)

    # Calculamos coeficientes para la forma polinomial del volumen
    termino_cubo = presion
    termino_cuadrado = -R * temperatura - b_mezcla * presion
    termino_lineal = a_mezcla
    termino_independiente = -a_mezcla * b_mezcla
    
    # Obtenemos raices del polinomio
    volumen = cardano_vectorizado(termino_cubo, termino_cuadrado, termino_lineal, termino_independiente)
    return volumen


def redlich_kwong_presion(temperatura, volumen, composiciones, pc, tc, k):
    """
    Calcula la presion de una mezcla utilizando la ecuacion de Redlich Kwong
    Args:
        Variables Termodinamicas ---------------------------------------------------------
        temperatura (1, T) : vector de temperaturas en K.
        volumen (1, T) : vector de volumenes molares en mL / mol.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        kij (C, C) : matriz de factor de interacción binaria entre compuesto-i y compuesto-j

    Returns:
        presion (1, T): vector de las presiones por cada variable termodinamica
    """
    # Obtenemos constantes de la ecuacion
    a = 0.42748 * R**2 * tc**2.5 / pc
    b = 0.08664 * R * tc / pc
    a_mezcla = obtener_a_mezcla(composiciones, a, k)
    b_mezcla = np.sum(composiciones * b)

    # Calculamos presion usando ecuacion de Redlich Kwong
    presion = R * temperatura / (volumen - b_mezcla) - a_mezcla / (np.sqrt(temperatura) * volumen * (volumen + b_mezcla))
    return presion

    
def redlich_kwong_temperatura(presion, volumen, composiciones, pc, tc, k):
    """
    Calcula la temperatura de una mezcla utilizando la ecuacion de Van der Waals.
    Args:
        Variables Termodinamicas ---------------------------------------------------------
        presion (1, T) : vector de presion en bar.
        volumen (1, T) : vector de volumenes molares en mL / mol.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        k (C, C) : matriz de factor de interacción binaria entre compuesto-i y compuesto-j

    Returns:
        temperatura (T,): vector de las temperaturas por cada variable termodinamica
    """

    # Obtenemos constantes de la ecuacion
    a = 0.42748 * R**2 * tc**2.5 / pc
    b = 0.08664 * R * tc / pc
    a_mezcla = obtener_a_mezcla(composiciones, a, k)
    b_mezcla = np.sum(composiciones * b)

    # Resolvemos polinomio caracteristico de la temperatura
    termino_cubo = R / (volumen - b_mezcla)
    termino_cuadrado = 0
    termino_lineal = -presion
    termino_independiente = - a_mezcla / (volumen * (volumen + b_mezcla))
    temperatura = cardano_vectorizado(termino_cubo, termino_cuadrado, termino_lineal, termino_independiente)**2

    # La temperatura se encuentra dentro de una raiz cuadrada. Algunas temperaturas son solucion para -raiz(T) y otras para raiz(T). Debemos determinar cual es cual
    presion_calculada = R * temperatura / (volumen - b_mezcla) - a_mezcla / (np.sqrt(temperatura) * volumen * (volumen + b_mezcla))
    soluciones_validas = np.isclose(presion_calculada, presion, atol=1e-12)
    soluciones_invalidas = np.logical_not(soluciones_validas)
    temperatura[soluciones_invalidas] = np.nan

    # Si en alguna de las variables termodinamicas (P, V) existe mas de una temperatura valida
    # ((seguramente fue porque el volumen o presion son negativos xd, pero aun asi se aprecia un codigo robusto))
    # Entonces quedate solamente con la mayor para evitar devolver un array de tamaño inesperado
    temperatura = np.nanmax(temperatura, axis=0) 
    # En este punto la temperatura tiene shape = (T,), devolvemos su shape a (1, T)
    temperatura = temperatura[np.newaxis, :]
    return temperatura


def redlich_kwong_volumen(presion, temperatura, composiciones, pc, tc, k):
    """
    Calcula los volumenes molares de una mezcla usando la ecuación de Redlick Kwong.
    Es una funcion de mezcla.
    Args:
        Variables Termodinamicas ---------------------------------------------------------
        presion (1, T) : vector de presiones en bar. 
        temperatura (1, T) : vector de temperaturas en K.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        kij (C, C) : matriz de factor de interacción binaria entre compuesto-i y compuesto-j

    Returns:
        volumen (3, T) : Matriz de volumenes liquido, prohibido y vapor para cada variable termodinamica
    """

    # Obtenemos constantes de la ecuacion
    a = 0.42748 * R**2 * tc**2.5 / pc
    b = 0.08664 * R * tc / pc
    a_mezcla = obtener_a_mezcla(composiciones, a, k)
    b_mezcla = np.sum(composiciones * b)

    # Calculamos coeficientes para la forma polinomial del volumen
    termino_cubo = presion
    termino_cuadrado = -R * temperatura
    termino_lineal = a_mezcla / np.sqrt(temperatura) - R * temperatura * b_mezcla - presion * b_mezcla**2
    termino_independiente = -a_mezcla * b_mezcla / np.sqrt(temperatura)

    # Obtenemos raices
    volumen = cardano_vectorizado(termino_cubo, termino_cuadrado, termino_lineal, termino_independiente)
    return volumen


def antoine_presion_vapor(temperatura, a, b, c):
    """
    FUNCION DE COMPUESTO: Calcula la presion de vapor de un compuesto utilizando la ecuacion de Antoine

    Args:
        Variables termodinamicas ------------------------------
        temperatura (1, T): Vector de temperaturas en Kelvin (K)
        Propiedades de compuesto ------------------------------
        a, b, c (C, 1): Vectores de constantes de la ecuacion de antoine

    Returns:
        presion_vapor (C, T): Matriz de la presion de vapor en bar del compuesto C a la temperatura T
    """
    # Calculos muy sencillos
    potencia = a - b / (temperatura + c - 273.15)
    presion_vapor = np.power(10, potencia)
    return presion_vapor


def antoine_temperatura(presion_vapor, a, b, c):
    """
    FUNCION DE COMPUESTO: Calcula la temperatura que genera la presion_vapor correspondiente utilizando
    la ecuacion de antoine

    Args:
        Variables termodinamicas ------------------------------
        presion_vapor (1, T): Vector de presion de vapor en Bar
        Propiedades de compuesto ------------------------------
        a, b, c (C, 1): Vectores de constantes de la ecuacion de antoine

    Returns:
        presion_vapor (C, T): Matriz de la temperatura en Kelvin (K) del compuesto C a la presion de vapor T
    """
    # Calculos sencillos
    temperatura = b / (a - np.log10(presion_vapor)) - c + 273.15
    return temperatura


def cp_gas_ideal(temperatura, a0, a1, a2, a3, a4):
    """
    Funcion de compuesto: Calcula la capacidad calorifica a presion constante.
    Cp(T) = a0 + a1/10^3 * T + a2/10^5 * T^2 + a3/10^8 * T^3 + a4/10^11 * T^4

    Args:
        Variables Termodinamicas --------------------------------------------
        temperatura (1, T): Vector de temperaturas en Kelvin (K)
        Propiedades de compuesto --------------------------------------------
        a0, a1, a2, a3, a4 (C, 1) Vectores de coeficientes del polinomio del cp
    Returns:
        cp (C, T): Capacidad calorifica del compuesto C a la temperatura T. En J / (mol * K)
    """
    # Calculamos cp adimensional
    cp = a0 + a1/10**3 * temperatura + a2/10**5 * temperatura**2 + a3/10**8 * temperatura**3 + a4/10**11 * temperatura**4
    # Colocamos el cp en unidades de J / (mol K)
    cp = cp * (R / 10)
    return cp


def cp_gas_ideal_temperatura(cp, a0, a1, a2, a3, a4):
    """
    Funcion de compuesto: Calcula la temperatura a la cual se posee la capacidad calorifica 'cp'
    Cp = a0 + a1/10^3 * T + a2/10^5 * T^2 + a3/10^8 * T^3 + a4/10^11 * T^4 

    Args:
        Variables Termodinamicas --------------------------------------------
        cp (1, T): Vector de capacidades calorificas en J / (mol K)
        Propiedades de compuesto --------------------------------------------
        a0, a1, a2, a3, a4 (C, 1) Vectores de coeficientes del polinomio del cp
    Returns:
        temperatura (C, T): temperatura del compuesto C con la capacidad calorifica T. En J / (mol * K)
    """
    # Definimos la derivada del cp para poder aplicar el metodo de Newton
    def derivada_cp(temp):
        deriv = a1/10**3 + 2*a2/10**5*temp + 3*a3/10**8*temp**2 + 4*a4/10**11*temp**3
        return deriv * (R / 10)
    
    # Definimos funcion cp(T) que solo dependa de la temperatura
    def capacidad_calorifica(temp):
        return cp_gas_ideal(temp, a0, a1, a2, a3, a4)
    
    # Guess inicial de temperatura 500 K
    temperatura = np.ones(cp.shape, dtype=np.float64) * 500

    # Aplicamos metodo de Newton
    temperatura = resolver_newton(capacidad_calorifica, cp, temperatura, derivada=derivada_cp)

    return temperatura


def integral_cp_gas_ideal(temperatura_1, temperatura_2, a0, a1, a2, a3, a4):
    """
    Funcion de compuesto: Calcula la integral del cp desde temperatura_1 hasta temperatura_2. Devuelve en unidades
    de J / mol
    Args:
        Variables termodinamicas -------------------------------
        temperatura_1 (1, T): Vector de temperatura inicial
        temperatura_2 (1, T): Vector de temperatura final
        Propiedades de compuesto -------------------------------
        a0, a1, a2, a3, a4 (C, 1): Vector de coeficientes del polinomio del cp
    Returns:
        delta_h (C, T): Matriz de valores de la integral del compuesto C a las condiciones T. En unidades
        de J / mol
    """
    # Calculamos cambio en entalpia adimensional
    delta_h = a0*(temperatura_2-temperatura_1) + a1/(2*10**3)*(temperatura_2**2-temperatura_1**2) + a2/(3*10**5)*(temperatura_2**3-temperatura_1**3) + a3/(4*10**8)*(temperatura_2**4-temperatura_1**4) + a4/(5*10**11)*(temperatura_2**5-temperatura_1**5)
    
    # Le colocamos unidades de J / mol
    delta_h = delta_h * (R / 10)
    return delta_h


def integral_cp_gas_ideal_temperatura_2(temperatura_1, delta_h, a0, a1, a2, a3, a4):
    """
    Funcion de compuesto: Calcula una temperatura tal que la integral del cp de temperatura_1 hasta esa temperatura
    sea igual a delta_h
    Args:
        Variables termodinamicas -------------------------------
        temperatura_1 (1, T): Vector de temperatura inicial en Kelvin (K)
        delta_h (1, T): Vector de cambios de entalpia en J / mol
        Propiedades de compuesto -------------------------------
        a0, a1, a2, a3, a4 (C, 1): Vector de coeficientes del polinomio del cp
    Returns:
        temperatura_2 (C, T): Matriz de la temperatura_2 en Kelvin (K). Donde C es el numero de compuestos y 
        T el numero de condiciones
    """
    # Definimos la derivada de la integral del cp
    def derivada_int_cp(temp):
        return cp_gas_ideal(temp, a0, a1, a2, a3, a4)
    
    # Definimos funcion integral_cp(T) que solo dependa de la temperatura
    def int_cp(temp):
        return integral_cp_gas_ideal(temperatura_1, temp, a0, a1, a2, a3, a4)
    
    # Suposicion inicial de temperatura_2: 500 Kelvin
    temperatura_2 = np.ones(delta_h.shape, dtype=np.float64) * 500

    # Encontramos temperatura_2 con el metodo de Newton
    temperatura_2 = resolver_newton(int_cp, delta_h, temperatura_2, derivada=derivada_int_cp)

    return temperatura_2


def obtener_a_alfa_mezcla_y_tensor_mezclado(y, a, alfa, k):
    """
    Calcula a_alfa_mezcla y el tensor de mezclado. Ambos usados en el calculo de coeficientes de
    fugacidad en una mezcla.
    El tensor de mezclado es: sqrt(ai * aj * alfai * alfaj) * (1 - kij)

    Args:
        Variables Termodinámicas ---------------------
        alfa (C, T): Valor 'alfa' proveniente de la ecuación cúbica de estado.

        Propiedades de Compuesto ---------------------
        y (C, 1): Composiciones del compuesto i-ésimo.
        a (C, 1): Constante 'a' proveniente de la ecuación cúbica de estado.
        k (C, C): Matriz del factor de interacción binaria entre el compuesto-i y compuesto-j.

    Returns:
        a_alfa_mezcla (1, T): Vector de constante a_alfa_mezcla para diferentes temperaturas.
        tensor_mezclado (C, C, T): Tensor de tercer orden para cálculo de coeficientes de fugacidad.
    """
    # Los calculos se llevaran en los tensores de la forma:
    # (C, i, T) = (propiedad_compuesto, propiedad_compuesto, variable_termodinamica)

    # Generamos tensores aptos para broadcasting
    yi = y[:, np.newaxis]
    yj = y[np.newaxis, :]
    ai = a[:, np.newaxis]
    aj = a[np.newaxis, :]
    alfa_i = alfa[:, np.newaxis, :]
    alfa_j = alfa[np.newaxis, :, :]
    kij = k[:, :, np.newaxis]

    # Tensor auxiliar 'pepe' resultara muy util para el calculo de coeficientes de fugacidad
    tensor_mezclado = np.sqrt(ai * aj * alfa_i * alfa_j) * (1 - kij)
    a_alfa_mezcla = yi * yj * tensor_mezclado

    # Sumamos sobre los ejes de las composiciones (C, i, T)
    a_alfa_mezcla = np.sum(np.sum(a_alfa_mezcla, axis=0), axis=0)

    # Regresamos a_alfa_mezcla con shape: (1, T) 
    # Regresamos tensor_mezclado con shape = (C, C, T)
    return a_alfa_mezcla[np.newaxis, :], tensor_mezclado


def coef_fugacidad_mezcla_peng_robinson(presion, temperatura, composiciones, pc, tc, w, k, volumen_num=0):
    """
    Funcion de compuesto: Devuelve el coeficiente de fugacidad de un compuesto a las dierentes condiciones
    de presion y temperatura. En una mezcla homogenea (ya sea liquido o vapor) de compuestos.

    Todos los calculos hechos aqui estan basados en el libro: Chemical, Biochemical and engineering thermodynamics
    5ta edicion de: Stanley I. Sandler. En la pagina: 440.

    Args:
        Variables Termodinamicas ---------------------------------------------------------
        presion (1, T) : vector de presion en bar.
        temperatura (1, T) : vector de temperaturas kelvin.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        w (C, 1) : vector de factores acentricos.
        k (C, C) : matriz de factor de interacción binaria entre compuesto-i y compuesto-j

        kwargs --------------------------------------------------------------------------
        volumen_num=0: Peng Robinson puede devolver un unico volumen ó un volumen de la fase liquida y 
        volumen de la fase vapor. Siempre organizadas de menor a mayor. Este argumento te permite escoger
        con que raiz te quedas.
        CUIDADO: si la ecuacion devuelve una raiz unica entonces se organizan de la siguiente manera:
        [raiz_unica, np.nan, np.nan] entonces volumen_num = 1,2 devolvera np.nan.

    Returns:
        coef_fugacidad (C, T): Devuelve el coeficiente de fugacidad del compuesto 'C' a las condiciones 'T'
        en la mezcla.
    """

    # Calculamos parametros de la ecuacion de peng robinson
    a = (0.45724 * R**2 * tc**2) / pc
    b = (0.0778 * R * tc) / pc
    pw = 0.37464 + 1.54226 * w - 0.26992 * w**2
    alfa = (1 + pw * (1 - np.sqrt(temperatura / tc)))**2
    b_mezcla = np.sum(composiciones * b)

    # Calculamos a_alfa_mezcla y tensor de mezclado.
    a_alfa_mezcla, tensor_mezclado = obtener_a_alfa_mezcla_y_tensor_mezclado(composiciones, a, alfa, k)

    # Calculamos tensor A_ij (checar libro Sandler)
    aij = presion[np.newaxis, :] / (R * temperatura[np.newaxis, :])**2 * tensor_mezclado

    # Calculamos volumenes.
    termino_cubo = presion
    termino_cuadrado = b_mezcla * presion - R * temperatura
    termino_lineal = a_alfa_mezcla - 2 * R * temperatura * b_mezcla - 3 * presion * b_mezcla**2
    termino_independiente = presion * b_mezcla**3 + R * temperatura * b_mezcla**2 - a_alfa_mezcla * b_mezcla
    volumen = cardano_vectorizado(termino_cubo, termino_cuadrado, termino_lineal, termino_independiente)

    # Nos quedamos solamente con un volumen
    volumen = volumen[volumen_num, :]

    # Calculamos factor de compresibilidad y variables de calculo
    z_mix = presion * volumen / (R * temperatura)
    bi = b * presion / (R * temperatura)
    ai = a * alfa * presion / (R * temperatura)**2
    a_mix = a_alfa_mezcla * presion / (R * temperatura)**2
    b_mix = b_mezcla * presion / (R * temperatura)

    # Calculamos el logaritmo natural del coeficiente de fugacidad
    suma_yj_aij = np.sum(composiciones[:, np.newaxis] * aij, axis=0) # Calculamos un termino de sumatoria
    log_phi = bi / b_mix * (z_mix - 1) - np.log(z_mix - b_mix) - a_mix / (2 * np.sqrt(2) * b_mix) * (2 * suma_yj_aij / a_mix - bi / b_mix) * np.log((z_mix + (1 + np.sqrt(2)) * b_mix) / (z_mix + (1 - np.sqrt(2)) * b_mix))

    # Regresamos matriz de coeficientes de fugacidad
    return np.exp(log_phi) 


def coef_fugacidad_mezcla_soave(presion, temperatura, composiciones, pc, tc, w, k, volumen_num=0):
    """
    Funcion de compuesto: Devuelve el coeficiente de fugacidad de un compuesto a las dierentes condiciones
    de presion y temperatura. En una mezcla homogenea (ya sea liquido o vapor) de compuestos.

    Los calculos se basan en la ecuacion del libro: Chemical, Biochemical and engineering thermodynamics
    5ta edicion de: Stanley I. Sandler. En la pagina: 425. La derivacion de la ecuación con la eucacion cubica
    de estado de Soave-Redlich Kwong fue hecha por mi.

    Args:
        Variables Termodinamicas ---------------------------------------------------------
        presion (1, T) : vector de presion en bar.
        temperatura (1, T) : vector de temperaturas kelvin.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        w (C, 1) : vector de factores acentricos.
        k (C, C) : matriz de factor de interacción binaria entre compuesto-i y compuesto-j

        kwargs --------------------------------------------------------------------------
        volumen_num=0: Peng Robinson puede devolver un unico volumen ó un volumen de la fase liquida y 
        volumen de la fase vapor. Siempre organizadas de menor a mayor. Este argumento te permite escoger
        con que raiz te quedas.
        CUIDADO: si la ecuacion devuelve una raiz unica entonces se organizan de la siguiente manera:
        [raiz_unica, np.nan, np.nan] entonces volumen_num = 1,2 devolvera np.nan.

    Returns:
        coef_fugacidad (C, T): Devuelve el coeficiente de fugacidad del compuesto 'C' a las condiciones 'T'
        en la mezcla.
    """
    # Calculamos constantes de la mezcla
    a = (0.427480 * R**2 * tc**2) / pc
    b = (0.08664 * R * tc) / pc
    pw = 0.48508 + 1.55171 * w - 0.15613 * w**2
    alfa = (1 + pw * (1 - np.sqrt(temperatura / tc)))**2
    b_mezcla = np.sum(composiciones * b)

    # Calculamos a_mezcla y aij (tensor de mezclado)
    a_mezcla, aij = obtener_a_alfa_mezcla_y_tensor_mezclado(composiciones, a, alfa, k)

    # Calculamos volumenes
    termino_cubo = presion
    termino_cuadrado = -R * temperatura
    termino_lineal = a_mezcla - R * temperatura * b_mezcla - presion * b_mezcla**2
    termino_independiente = -a_mezcla * b_mezcla
    volumen = cardano_vectorizado(termino_cubo, termino_cuadrado, termino_lineal, termino_independiente)

    # Nos quedamos unicamente con un volumen
    volumen = volumen[volumen_num, :]

    # Calculamos el logaritmo del coeficiente de fugacidad
    z = presion * volumen / (R * temperatura)
    log_phi = np.log(volumen / (volumen - b_mezcla)) + b / (volumen - b_mezcla) + 2*np.sum(composiciones * aij, axis=1) / (R * temperatura * b_mezcla) * np.log(volumen / (volumen + b_mezcla)) - a_mezcla * b / (R * temperatura * b_mezcla**2) * (np.log(volumen / (volumen + b_mezcla)) + b_mezcla / (volumen + b_mezcla)) - np.log(z)

    # Devolvemos coeficientes de fugacidad (C, T)
    return np.exp(log_phi)


def coef_fugacidad_puro_peng_robinson(presion, temperatura, pc, tc, w, volumen_num=0):
    """
    Funcion de compuesto: Devuelve el coeficiente de fugacidad de un compuesto a las dierentes condiciones
    de presion y temperatura. Lo calcula como substancia pura.

    Todos los calculos hechos aqui estan basados en el libro: Chemical, Biochemical and engineering thermodynamics
    5ta edicion de: Stanley I. Sandler. En la pagina: 314.

    Args:
        Variables Termodinamicas ---------------------------------------------------------
        presion (1, T) : vector de presion en bar.
        temperatura (1, T) : vector de temperaturas kelvin.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        w (C, 1) : vector de factores acentricos.

        kwargs --------------------------------------------------------------------------
        volumen_num=0: Peng Robinson puede devolver un unico volumen ó un volumen de la fase liquida y 
        volumen de la fase vapor. Siempre organizadas de menor a mayor. Este argumento te permite escoger
        con que raiz te quedas.
        CUIDADO: si la ecuacion devuelve una raiz unica entonces se organizan de la siguiente manera:
        [raiz_unica, np.nan, np.nan] entonces volumen_num = 1,2 devolvera np.nan.

    Returns:
        coef_fugacidad (C, T): Devuelve el coeficiente de fugacidad del compuesto 'C' a las condiciones 'T'
        en la mezcla.
    """
    # Calculamos parametros de la ecuacion de peng robinson
    a = (0.45724 * R**2 * tc**2) / pc
    b = (0.0778 * R * tc) / pc
    pw = 0.37464 + 1.54226 * w - 0.26992 * w**2
    alfa = (1 + pw * (1 - np.sqrt(temperatura / tc)))**2

    # Calculamos volumen de compuesto puro
    termino_cubo = presion 
    termino_cuadrado = b * presion - R * temperatura
    termino_lineal = a*alfa - 2*R*temperatura*b - 3*b**2 * presion
    termino_independiente = b**3 * presion + R*temperatura*b**2 - a*alfa*b
    volumenes = cardano_vectorizado_2(termino_cubo, termino_cuadrado, termino_lineal, termino_independiente)
    volumen = volumenes[volumen_num]

    # Calculamos coeficientes de fugacidad.
    a_mayus = a * alfa * presion / (R*temperatura)**2
    b_mayus = presion * b / (R*temperatura)
    z = presion * volumen / (R * temperatura)
    log_phi = (z - 1) - np.log(z - b_mayus) - a_mayus / (2*np.sqrt(2)*b_mayus) * np.log((z + (1+np.sqrt(2)) * b_mayus) / (z + (1-np.sqrt(2))*b_mayus))
    return np.exp(log_phi)


def coef_fugacidad_puro_soave(presion, temperatura, pc, tc, w, volumen_num=0):
    """
    Funcion de compuesto: Devuelve el coeficiente de fugacidad de un compuesto a las dierentes condiciones
    de presion y temperatura. Lo calcula como substancia pura y usando la ecuacion cubica de estado de 
    Soave

    Los calculos hechos aqui estan basados en el libro: Chemical, Biochemical and engineering thermodynamics
    5ta edicion de: Stanley I. Sandler. En la pagina: 310. La derivacion de la ecuacion fue hecha por mi.

    Args:
        Variables Termodinamicas ---------------------------------------------------------
        presion (1, T) : vector de presion en bar.
        temperatura (1, T) : vector de temperaturas kelvin.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        pc (C, 1) : vector de las presiones críticas de cada compuesto de la mezcla.
        tc (C, 1) : vector de las temperaturas críticas de cada compuesto de la mezcla.
        w (C, 1) : vector de factores acentricos.

        kwargs --------------------------------------------------------------------------
        volumen_num=0: Peng Robinson puede devolver un unico volumen ó un volumen de la fase liquida y 
        volumen de la fase vapor. Siempre organizadas de menor a mayor. Este argumento te permite escoger
        con que raiz te quedas.
        CUIDADO: si la ecuacion devuelve una raiz unica entonces se organizan de la siguiente manera:
        [raiz_unica, np.nan, np.nan] entonces volumen_num = 1,2 devolvera np.nan.

    Returns:
        coef_fugacidad (C, T): Devuelve el coeficiente de fugacidad del compuesto 'C' a las condiciones 'T'
        en la mezcla.
    """
    # Calculamos volumen utilizando la ecuacion de Soave
    a = (0.427480 * R**2 * tc**2) / pc
    b = (0.08664 * R * tc) / pc
    pw = 0.48508 + 1.55171 * w - 0.15613 * w**2
    alfa = (1 + pw * (1 - np.sqrt(temperatura / tc)))**2
    
    # Calculamos coeficientes para la forma polinomial del volumen
    termino_cubo = presion
    termino_cuadrado = -R * temperatura
    termino_lineal = a * alfa - R * temperatura * b - presion * b**2
    termino_independiente = -a * alfa * b

    # Calculamos los volumenes
    volumenes = cardano_vectorizado_2(termino_cubo, termino_cuadrado, termino_lineal, termino_independiente)

    # Nos quedamos unicamente con un volumen
    volumen = volumenes[volumen_num]

    z = presion * volumen / (R * temperatura)
    log_phi = np.log((volumen - b) / volumen) + a * alfa / (R * temperatura * b) * np.log(volumen / (volumen + b)) - np.log(z) + (z - 1)

    return np.exp(log_phi)


def coef_actividad_unifac(temperatura, composiciones, num_grupos, r_mayus, q_mayus, aij):
    """
    Funcion de compuesto: Devuelve los coeficientes de actividad de los compuestos a las diferentes temperaturas
    en una mezcla homogenea.
    En esta funcion añadimos un nuevo eje "K y S" son ejes que contiene informacion de los grupos funcionales
    presentes en la mezcla.
    En el modelo unifac la matriz aij no es igual a su transpuesta. Entonces para los calculos la correspondencia
    de ejes es la siguiente (K, S) = (i, j)
    Args:
        Variables Termodinamicas ---------------------------------------------------------
        temperatura (1, T) : vector de temperaturas kelvin.

        Propiedades de Compuesto -----------------------------------------------------------
        composiciones (C, 1) : vector de composicion del compueso c en la mezcla
        numero_grupos (C, K): Matriz del numero de ocurrencias del grupo funcional K en el compuesto C
        r_mayus (K,): Vector de parametros R de cada grupo funcional segun el modelo UNIFAC
        q_mayus (K,): Vector de parametros Q de cada grupo funcional segun el modelo UNIFAC
        aij (K, S): Vector de parametros de interaccion energetica del grupo funcional K al grupo funcional S.

    Returns:
        coef_actividad (C, T): Matriz del coeficiente de actividad del compuesto C a la temperatura T en la mezcla
    """
    # Para los calculos trabajaremos con arryas de la forma (i, T, K, S) => (compuesto, temperatura, grupo_funcional, grupo_funcional)
    # Generamos tensores aptos para broadcasting
    temperatura = temperatura[:, :, np.newaxis, np.newaxis] # (1, T, 1, 1)
    composiciones = composiciones[:, :, np.newaxis, np.newaxis] # (C, 1, 1, 1)
    num_grupos = num_grupos[:, np.newaxis, :, np.newaxis] # (C, 1, K, 1)
    r_mayus = r_mayus[np.newaxis, np.newaxis, :, np.newaxis] # (1, 1, K, 1)
    q_mayus = q_mayus[np.newaxis, np.newaxis, :, np.newaxis] # (1, 1, K, 1)

    # Calculamos el logaritmo del coeficiente de actividad combinatorial (Documentacion en wikipedia)
    r = np.sum(num_grupos * r_mayus, axis=2, keepdims=True)
    q = np.sum(num_grupos * q_mayus, axis=2, keepdims=True)
    theta_minus =  composiciones * q / np.sum(composiciones * q, axis=0, keepdims=True)
    phi = composiciones * r / np.sum(composiciones * r, axis=0, keepdims=True)
    l = 5 * (r - q) - (r - 1)
    log_actividad_combinatorial = np.log(phi/composiciones) + 5*q*np.log(theta_minus/phi) + l - phi/composiciones * np.sum(composiciones * l, axis=0, keepdims=True)

    # Comenzamos el calculo de la actividad del grupo k-esimo en toda la mezcla
    tau = np.exp(-aij / temperatura)
    x = np.sum(num_grupos * composiciones, axis=0, keepdims=True) / np.sum(num_grupos * composiciones, axis=(0, 2), keepdims=True)
    theta_mayus = q_mayus * x / np.sum(q_mayus * x, axis=2, keepdims=True)
    # Para las sumatorias, necesitamos a theta_mayus con diferentes organizaciones de ejes
    theta_mayus_2 = np.transpose(theta_mayus, (0, 1, 3, 2)).copy()
    # Dividimos el calculo de actividad en dos terminos: Qk*(1 - log(termino_1) - termino_2)
    termino_1 = np.sum(theta_mayus * tau, axis=2, keepdims=True)
    termino_2 = np.sum(theta_mayus_2 * tau / np.sum(theta_mayus * tau, axis=2, keepdims=True), axis=3, keepdims=True)
    # termino_1 y termino_2 tienen ejes (1, T, 1, S), (1, T, K, 1). Transponemos el termino_1 a (1, T, S, 1) para operar elemento a elemento en el eje K
    termino_1 = np.transpose(termino_1, (0, 1, 3, 2)).copy()
    # Caclulamos la actividad del grupo k-esimo en la mezcla
    log_actividad_mezcla = q_mayus * (1 - np.log(termino_1) - termino_2)

    # Usando el mismo orden de caculos. Ahora obtenemos la actividad del grupo k-esimo en una solucion hipotetica pura del compuesto i-esimo
    x_i = num_grupos / np.sum(num_grupos, axis=2, keepdims=True)
    theta_mayus_i = q_mayus * x_i / np.sum(q_mayus * x_i, axis=2, keepdims=True)
    theta_mayus_i_2 = np.transpose(theta_mayus_i, (0, 1, 3, 2)).copy()
    termino_1 = np.sum(theta_mayus_i * tau, axis=2, keepdims=True)
    termino_2 = np.sum(theta_mayus_i_2 * tau / np.sum(theta_mayus_i * tau, axis=2, keepdims=True), axis=3, keepdims=True)
    termino_1 = np.transpose(termino_1, (0, 1, 3, 2)).copy()
    log_actividad_i = q_mayus * (1 - np.log(termino_1) - termino_2)
    
    # Calculamos la actividad residual
    log_actividad_residual = np.sum(num_grupos * (log_actividad_mezcla - log_actividad_i), axis=2, keepdims=True)
    
    # Calculamos la actividad del compuesto i-esimo
    coef_actividad = np.exp(log_actividad_combinatorial + log_actividad_residual) # (C, T, 1, 1)

    # Devolvemos en el formato (C, T)
    return coef_actividad[:, :, 0, 0]


def entalpia_exceso_unifac(temperatura, composiciones, num_grupos, r_mayus, q_mayus, aij):
    """
    Funcion de compuesto: Devuelve la entalpia molar de exceso del compuesto C en la mezcla a la temperaura T.
    usando el modelo UNIFAC para calcular la actividad quimica.
    Args:
        Variables termodinamicas --------------------------------------
        temperatura (1, T): vector de temperaturas en kelvin
        Variables de compuesto -----------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        numero_grupos (C, K): Matriz del numero de ocurrencias del grupo funcional K en el compuesto C
        r_mayus (K,): Vector de parametros R de cada grupo funcional segun el modelo UNIFAC
        q_mayus (K,): Vector de parametros Q de cada grupo funcional segun el modelo UNIFAC
        aij (K, S): Vector de parametros de interaccion energetica del grupo funcional K al grupo funcional S.
    Returns:
        entalpia_exceso (C, T): La entalpia de exceso del compuesto C a la temperatura T en la mezcla
    """
    # Definimos funcion que calcula el ln(coeficiente_actividad_i) en funcion de la temperatura
    def log_actividad(temp):
        return np.log(coef_actividad_unifac(temp, composiciones, num_grupos, r_mayus, q_mayus, aij))
    
    # Calculamos dln(gamma_i)/dT
    dln_gamma_dt = (log_actividad(temperatura + 1e-6) - log_actividad(temperatura - 1e-6)) / 2e-6

    # Calculamos la entalpia de exceso y convertimos de bar*mL a joules
    entalpia_exceso = - R * temperatura**2 * dln_gamma_dt / 10

    return entalpia_exceso
    

def entropia_exceso_unifac(temperatura, composiciones, num_grupos, r_mayus, q_mayus, aij):
    """
    Funcion de compuesto: Devuelve la entropia molar de exceso del compuesto C en la mezcla a la temperaura T.
    usando el modelo UNIFAC para calcular la actividad quimica.
    Args:
        Variables termodinamicas --------------------------------------
        temperatura (1, T): vector de temperaturas en kelvin
        Variables de compuesto -----------------------------------------
        composiciones (C, 1) : vector de las composiciones de cada compuesto de la mezcla.
        numero_grupos (C, K): Matriz del numero de ocurrencias del grupo funcional K en el compuesto C
        r_mayus (K,): Vector de parametros R de cada grupo funcional segun el modelo UNIFAC
        q_mayus (K,): Vector de parametros Q de cada grupo funcional segun el modelo UNIFAC
        aij (K, S): Vector de parametros de interaccion energetica del grupo funcional K al grupo funcional S.
    Returns:
        entropia_exceso (C, T): La entropia de exceso del compuesto C a la temperatura T en la mezcla
    """
    # Definimos funcion que calcula el ln(coeficiente_actividad_i) en funcion de la temperatura
    def log_actividad(temp):
        return np.log(coef_actividad_unifac(temp, composiciones, num_grupos, r_mayus, q_mayus, aij))
    
    # Calculamos dln(gamma_i)/dT
    dln_gamma_dt = (log_actividad(temperatura + 1e-6) - log_actividad(temperatura - 1e-6)) / 2e-6

    # Calculamos la entropia de exceso y convertimos de bar*mL a joules
    entropia_exceso = - R * temperatura * dln_gamma_dt / 10

    return entropia_exceso

    
def correccion_poynting(presion, temperatura, a_antoine, b_antoine, c_antoine, vol_liq):
    """
    Devuelve la correccion de poynting del compuesto C a las condiciones T.
    Funcion de compuesto (C, T).
    Args:
        Variables termodinamias ----------------------
        presion (1, T): Presion en bar
        temperatura (1, T): Temperatura en kelvin
        Variables de compuesto -----------------------
        a_antoine, b_antoine, c_antoine (C, 1): Vector columna de constantes de antoine de cada compuesto
        vol_liq (C, 1): Vector columna de los volumenes de liquido de cada compuesto
    Returns:
        poynting (C, T): Factor de correccion de poynting del compuesto C a las condiciones T
    """
    # Calculamos la presion de vapor usando la ecuacion de antoine
    presion_vapor = antoine_presion_vapor(temperatura, a_antoine, b_antoine, c_antoine)

    # Calculamos correccion de poynting
    poynting = np.exp(vol_liq*(presion - presion_vapor) / (R * temperatura))

    return poynting


# FUNCION ALTAMENTE FRAGIL. PARA UN EXAMEN NAMAS
def entropia_mezcla_ideal(presion, temperatura, composiciones_liq, composiciones_vap, calidad_vap, tc, pc, w, antoine_a, antoine_b, antoine_c, a0, a1, a2, a3, a4, cpliq):
    """
    Docstring... despues
    """
    # Calculamos la presion de vapor
    t_ref = 273.15
    presion_vapor = antoine_presion_vapor(t_ref, antoine_a, antoine_b, antoine_c)

    # Calculamos cambio en el volumen a 0°C y Pvap con una ecuacion cubica de estado
    a = (0.45724 * R**2 * tc**2) / pc
    b = (0.0778 * R * tc) / pc
    pw = 0.37464 + 1.54226 * w - 0.26992 * w**2
    alfa = (1 + pw * (1 - np.sqrt(t_ref / tc)))**2
    termino_cubo = presion_vapor
    termino_cuadrado = b * presion_vapor - R * t_ref
    termino_lineal = a*alfa - 2*R*t_ref*b - 3*b**2 * presion_vapor
    termino_independiente = b**3 * presion_vapor + R*t_ref*b**2 - a*alfa*b
    volumenes = cardano_vectorizado_2(termino_cubo, termino_cuadrado, termino_lineal, termino_independiente)
    delta_v = volumenes[2] - volumenes[0]

    # Calculamos entalpia de vaporizacion
    calor_latente = (presion_vapor * antoine_b * np.log(10) * 273.15 * delta_v) / (antoine_c)**2
    calor_latente = calor_latente / 10
    entropia_vaporizacion = calor_latente / t_ref

    # Calculamos la integral de cp/T
    int_cp_t = R/10*(a0*np.log(temperatura/273.15) + a1/10**3*(temperatura-273.15) + a2/(2*10**5)*(temperatura**2-273.15**2) + a3/(3*10**8)*(temperatura**3-273.15**3) + a4/(4*10**11)*(temperatura**4-273.15**4))

    # Calculamos el termino de presion
    termino_presion = R/10 * np.log(presion / presion_vapor)

    # Calculamos entropia molar de la fase vapor
    entropia_vapor = np.sum(composiciones_vap * (entropia_vaporizacion + int_cp_t - termino_presion), axis=0, keepdims=True) - R/10 * np.sum(composiciones_vap*np.log(composiciones_vap), axis=0, keepdims=True)

    # Calculamos la entropia de liquido
    entropia_liquido = np.sum(composiciones_liq * cpliq * np.log(temperatura / 273.15), axis=0, keepdims=True) - R/10 * np.sum(composiciones_liq*np.log(composiciones_liq), axis=0, keepdims=True)

    # Calculamos la entropia molar de la mezcla
    entropia = calidad_vap * entropia_vapor + (1 - calidad_vap) * entropia_liquido

    return entropia


def entalpia_mezcla_ideal(temperatura, composiciones_liq, composiciones_vap, calidad_vap, tc, pc, w, antoine_a, antoine_b, antoine_c, a0, a1, a2, a3, a4, cpliq):
    """
    Docstring... despues
    """
    # Calculamos la presion de vapor
    t_ref = 273.15
    presion_vapor = antoine_presion_vapor(t_ref, antoine_a, antoine_b, antoine_c)

    # Calculamos cambio en el volumen a 0°C y Pvap con una ecuacion cubica de estado
    a = (0.45724 * R**2 * tc**2) / pc
    b = (0.0778 * R * tc) / pc
    pw = 0.37464 + 1.54226 * w - 0.26992 * w**2
    alfa = (1 + pw * (1 - np.sqrt(t_ref / tc)))**2
    termino_cubo = presion_vapor
    termino_cuadrado = b * presion_vapor - R * t_ref
    termino_lineal = a*alfa - 2*R*t_ref*b - 3*b**2 * presion_vapor
    termino_independiente = b**3 * presion_vapor + R*t_ref*b**2 - a*alfa*b
    volumenes = cardano_vectorizado_2(termino_cubo, termino_cuadrado, termino_lineal, termino_independiente)
    delta_v = volumenes[2] - volumenes[0]

    # Calculamos entalpia de vaporizacion
    calor_latente = (presion_vapor * antoine_b * np.log(10) * 273.15 * delta_v) / (antoine_c)**2
    calor_latente = calor_latente / 10

    # Calculamos la integral de cp
    int_cp = integral_cp_gas_ideal(t_ref, temperatura, a0, a1, a2, a3, a4)

    # Calculamos entalpia molar de la fase vapor
    entalpia_vap = np.sum(composiciones_vap * (calor_latente + int_cp), axis=0, keepdims=True)

    # Calculamos entalpia molar de la fase liquida
    entalpia_liq = np.sum(composiciones_liq * (cpliq*(temperatura - t_ref)), axis=0, keepdims=True)

    # Calculamos entlapia molar de la mezcla
    entalpia = calidad_vap * entalpia_vap + (1 - calidad_vap) * entalpia_liq

    return entalpia


def entalpia_mezcla_ideal_vapor(temperatura, composiciones, tc, pc, w, antoine_a, antoine_b, antoine_c, a0, a1, a2, a3, a4):
    """
    Docstring... despues
    """
    # Calculamos la presion de vapor
    t_ref = 273.15
    presion_vapor = antoine_presion_vapor(t_ref, antoine_a, antoine_b, antoine_c)

    # Calculamos cambio en el volumen a 0°C y Pvap con una ecuacion cubica de estado
    a = (0.45724 * R**2 * tc**2) / pc
    b = (0.0778 * R * tc) / pc
    pw = 0.37464 + 1.54226 * w - 0.26992 * w**2
    alfa = (1 + pw * (1 - np.sqrt(t_ref / tc)))**2
    termino_cubo = presion_vapor
    termino_cuadrado = b * presion_vapor - R * t_ref
    termino_lineal = a*alfa - 2*R*t_ref*b - 3*b**2 * presion_vapor
    termino_independiente = b**3 * presion_vapor + R*t_ref*b**2 - a*alfa*b
    volumenes = cardano_vectorizado_2(termino_cubo, termino_cuadrado, termino_lineal, termino_independiente)
    delta_v = volumenes[2] - volumenes[0]

    # Calculamos entalpia de vaporizacion
    calor_latente = (presion_vapor * antoine_b * np.log(10) * 273.15 * delta_v) / (antoine_c)**2
    calor_latente = calor_latente / 10

    # Calculamos la integral de cp
    int_cp = integral_cp_gas_ideal(t_ref, temperatura, a0, a1, a2, a3, a4)

    # Calculamos entalpia molar de la fase vapor
    entalpia_vap = np.sum(composiciones * (calor_latente + int_cp), axis=0, keepdims=True)

    return entalpia_vap


def integral_cp_entre_t_gas_ideal(temperatura_1, temperatura_2, a0, a1, a2, a3, a4):
    """
    Funcion de compuesto devuelve array.shape = (C, T)
    Devuelve la integral del cp polinomial dividido entre la temperatura. La integral evaluada de temperatura_1 a
    temperatura_2.
    Args:
        Variables termodinamicas -----------------------------
        temperatura_1 (1, T): Temperatura del limite inferior de la integral. (En kelvin)
        temperatrua_2 (1, T): Temperatura del limite superior de la integral. (En kelvin)
        Propiedades de compuesto -----------------------------
        a0, a1, a2, a3, a4 (C, 1): Coeficientes del cp polinomial.
    Returns:
        resultado (C, T): El valor de la integral para el compuesto C a las condiciones T
    """
    # Calculamos la integral de manera adimensional (dividida por R)
    resultado = a0*np.log(temperatura_2 / temperatura_1) + a1/10**3*(temperatura_2-temperatura_1) + a2/(2*10**5)*(temperatura_2**2-temperatura_1**2) + a3/(3*10**8)*(temperatura_2**3-temperatura_1**3) + a4/(4*10**11)*(temperatura_2**4-temperatura_1**4)

    # Convertimos resultado a unidades de J / (mol * K)
    resultado = resultado * 8.314472

    return resultado


def integral_cp_entre_t_gas_ideal_temperatura_2(temperatura_1, resultado, a0, a1, a2, a3, a4):
    """
    Devuelve la temperautra_2 que satisface que la integral del cp/T integrado de temperatura_1 a temperatura_2 sea igual a resultado.
    Args:
        Variables termodinamicas ------------------------------
        temperatura_1 (1, T): Temperatura del limite inferior de la integral. (En kelvin)
        resultado (1, T): Resultado de la integral. (En J / (mol*K))
        Propiedades de compuesto ------------------------------
        a0, a1, a2, a3, a4 (C, 1): Coeficientes del cp polinomial.
    
    Returns:
        temperatura_2 (C, T): temperatrua que satisface la ecuacion para el compuesto C a la temperatura inicial T.
    """
    # Definimos la integral de Cp/T unicamente en funcion de la temperatura 2
    def integral(temp_2):
        return integral_cp_entre_t_gas_ideal(temperatura_1, temp_2, a0, a1, a2, a3, a4)
    
    # Hacemos suposicion inicial de la temperatura 2. Suponemos un valor de 500 K
    temperatura_2 = np.ones(temperatura_1.shape, dtype=np.float64) * 500

    # Hallamos la temperatura_2 mediante el metodo de Newton
    temperatura_2 = resolver_newton(integral, resultado, temperatura_2)

    return temperatura_2


def propiedades_formacion_estandar_liquido(entalpia_form, gibbs_form, a_ant, b_ant, c_ant):
    """
    Calcula entalpia, entropia y energia libre de gibbs de formacion estandar de compuesto liquido.
    Esto lo logra a partir de estas propiedades de formacion estandar pero de compuestos como vapor.
    Se basa en la ecuación de clausis y antoine.
    deltaH_vaporizacion = b * ln(10) * T^2 * R / (T + c)^2
    Donde a, b, c son las cte de la ecuación de antoine
    Args:
        PROPIEDADES DE COMPUESTO ---------------------------
        entalpia_form (C, 1): Entalpias de formacion estandar de los compuestos como vapor a (298.15 K, 1 bar). En unidades de J / mol.
        gibbs_form (C, 1): Energía libre de gibbs de formacion estandar de los compuestos como vapor a (298.15 K, 1 bar). En unidades de J / mol.
        a_ant, b_ant, c_ant (C, 1): Constante a para la ecuacion de antoine en unidades de Kelvin Bar

    Returns:
        hf_liq (C, 1): Entalpia de formacion estandar para compuesto liquido en unidades de J / mol
        sf_liq (C, 1): Entropia de formacion estandar para compuesto liquido en unidades de J / (mol K)
        gf_liq (C, 1): Energía libre de gibbs de formación estandar para compuesto liquido en unidades de J / mol
    """
    # Calculamos entalpia de vaporizacion y presion de saturacion a 298.15 K
    calor_latente = b_ant * np.log(10) * 298.15**2 * 8.314472 / (298.15 + c_ant - 273.15)**2
    presion_vapor = antoine_presion_vapor(298.15, a_ant, b_ant, c_ant)

    # Calculamos entropia de formacion de compuesto vapor
    entropia_form = (entalpia_form - gibbs_form) / 298.15

    # Calculamos propiedades de formacion de compeusto liquido
    hf_liq = entalpia_form - calor_latente
    sf_liq = entropia_form - 8.314472*np.log(presion_vapor) - calor_latente/298.15
    gf_liq = gibbs_form + 8.314472*298.15*np.log(presion_vapor)

    return hf_liq, sf_liq, gf_liq


def correccion_gas_ideal_propiedades_reaccion(presion, temperatura, entalpia_form, gibbs_form, coef):
    """
    Calcula la entalpia, entropia y energia libre de gibbs de reacciones quimicas a condiciones especificas de presion y temperatura.
    Para esto se basa en ecuación del mosdleo de gas ideal.
    Args:
        PROPIEDADES DE COMPUESTO ------------------
        entalpia_form (C, 1): Entalpia de formacion estandar para compuesto vapor a (298.15 K, 1 bar) en unidades de J / mol
        gibbs_form (C, 1): Energía libre de gibbs de formación estandar para compuesto vapor a (298.15 K, 1 bar) en unidades de J / mol
        coef (C, R): Matriz de coeficientes estequimetricos del compuesto C en la reacción R
        CONDICIONES -------------------------------
        presion (1, T): Array de presion en bar.
        temperatura (1, T): Array de temperaturas en Kelvin
    Returns:
        hrx (C, T): Entalpia de 
    """
