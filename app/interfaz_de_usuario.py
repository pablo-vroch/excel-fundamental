import numpy as np
import json
import motor_de_calculo

# Abrimos diccionarios principales
with open('MAPA_ID.json', 'r', encoding='utf-8') as file:
    MAPA_ID = json.loads(file.read())

with open('MAPA_COLUMNAS.json', 'r', encoding='utf-8') as file:
    MAPA_COLUMNAS = json.loads(file.read())

# Abrimos base de datos de propiedades termodinamicas
PROP_TERMOF = np.load('PROP_TERMOF.npy', allow_pickle=True)

# Constantes
ERROR_PERMITIDO = 1e-6


# FUNCIONES DE APOYO (no se llaman en excel)

def normalizar_entrada(entrada : str) -> str:
    # Hacer todo minusculas, eliminar guiones '-', eliminar espacios ' '
    entrada = entrada.casefold().replace('-','').replace(' ','')
    return entrada


def obtener_id(compuesto : float | int | str):
    """
    Devuelve la fila dentro de PROP_TERMOF donde se encuentra ubicado un compuesto
    """

    # Si es un numero del tipo int, tomalo como el ID directamente
    if isinstance(compuesto, int):
        return compuesto
    
    # Si es un numero del tipo float, conviertelo a int solamente si es entero. Sino levanta un error
    if isinstance(compuesto, float):
        if compuesto % 1 != 0:
            raise TypeError(f'El ID del compuesto debe ser un número entero. El valor : {compuesto} : levanto el error')
        else:
            return int(compuesto)
    
    # Si cualquiera de los elementos del string es una letra, entonces te proporcionaron el nombre del compuesto:
    for i in compuesto:
        if i.isalpha():

            # normalizalo y buscalo en el diccionario
            compuesto_normalizado = normalizar_entrada(compuesto)
            try:
                return MAPA_ID['NOMBRE'][compuesto_normalizado]
            except KeyError:
                raise KeyError(f'{compuesto} no esta en la base de datos, revisa que este bien escrito')
    
    # Si ninguno de los caracteres es una letra, buscalo directamente ya que debe ser el CAS
    try:
        return MAPA_ID['CAS'][compuesto]
    except KeyError:
        raise KeyError(f'{compuesto} no esta en la base de datos, revisa que este bien escrito')


def obtener_columna(propiedad : str) -> int | tuple:
    # Si propiedad es un valor del tipo int entonces debe ser directamente el numero de columna
    if isinstance(propiedad, int):
        return propiedad
    
    # Si propiedad es un valor del tipo float, conviertelo a entero SOLO SI es un número entero
    if isinstance(propiedad, float):
        if propiedad % 1 != 0:
            raise TypeError(f'La columna de la propiedad debe ser un número entero. El valor : {propiedad} : levanto el error')
        else:
            return int(propiedad)

    # Normalizamos el string de entrada
    propiedad = normalizar_entrada(propiedad)

    try:
        return MAPA_COLUMNAS[propiedad] 
    except KeyError:
        raise KeyError(f'la propiedad: {propiedad} no esta en la base de datos, revisa que este bien escrita')


def aplanar(lista : list[any]) -> list[any]:
    """
    Devuelve una lista lineal de todos los elementos de una lista de listas.
    Analogo a aplastarla:
    EX: 
    entrada: [1, 2, 3, [4, 5, 6], 7]
    devuelve: [1, 2, 3, 4, 5, 6, 7]
    """

    lista_aplanada = []

    for elemento in lista:
        if isinstance(elemento, list):
            lista_aplanada.extend(aplanar(elemento))
        else:
            lista_aplanada.append(elemento)

    return lista_aplanada


def obtener_ids_lista(lista : list) -> list:
    """
    Recibe una lista de referencias a compuestos y devuelve otra lista correspondiente al numero
    de fila de cada compuesto

    IN:
    [['Water'], ['Ethanol'], ['7440-63-3']]
    OUT:
    np.array([440, 66, 468])
    """
    lista = aplanar(lista)
    id = np.array([obtener_id(i) for i in lista], dtype=np.int16)
    return id


def obtener_columnas_lista(lista : list) -> list:
    """
    Recibe una lista de referencias a propiedades termodinamicas y devuelve otra lista
    con su correspondiente numero de columna en PROP_TERMOF.
    Si la lista contiene valores 'None' simplemente los elimina.

    IN:
    [['nombre'], ['tc'], [None], ['pc']]
    OUT:
    np.array([1, 6, 7])
    """
    lista = aplanar(lista)

    # Generamos lista de numero de columnas excluyendo valores None
    columnas = list()
    for elemento in lista:

        # Excluimos elementos None
        if elemento is not None:
            col = obtener_columna(elemento)
            
            # Si la propiedad involucra varias columnas en la matriz, desempaquetalos
            if isinstance(col, list):
                columnas.extend(col)
            # Si la propiedad solo es una columna, añadela y ya esta.
            else:
                columnas.append(col)
    
    columnas = np.array(columnas, dtype=np.int16)
    return columnas


def es_vector_columna(vector):
    """
    Determina si un vector es:
    vector fila: [[1, 2, 3, 4]] => returns False
    vector columna (2): [[1],[2],[3],[4]] => returns True
    escalar (3): [[1]] => returns None
    """
    if len(vector[0]) > 1:
        return False
    
    if len(vector) > 1:
        return True
    
    return None


def determinar_orietnacion_tabla(compuestos, propiedades):
    """
    Las funciones de compuesto devuelven matrices. Un eje de propiedades y un eje de compuestos.
    Dependiendo de como orientes los compuestos y las propiedades en excel, la matriz de debera
    imprimir verticalmente (C, T) o su transpuesta (T, C). Esta funcion lo determina.
    
    Args:
        compuestos, propiedades: Vectores de compuestos y propiedades. Pueden ser listas de objetos
    Returns:
        True => Para orientacion (C, T)
        False => Para orientacion (T, C)
    """
    # Determinar la orientacion de los vectores
    compuestos_son_columna = es_vector_columna(compuestos)
    propiedades_son_columna = es_vector_columna(propiedades)

    # Caso 1. El vector de compuestos es columna, automaticamente imprimir (C, T)
    if compuestos_son_columna == True:
        return True
    
    # Caso 2. El vector de compuestos es horizontal. Imprimir (T, C)
    if compuestos_son_columna == False:
        return False
    
    # Caso 3. El vector de variables termodinamicas es vertical, imprimir (T, C)
    if propiedades_son_columna == True:
        return False

    # Caso 4. El vector de variables termodinamicas es horizontal. imprimir (C, T)
    if propiedades_son_columna == False:
        return True
    
    # Caso 5. En este punto, los compuestos y propiedades son unitarios.
    return True # Default
    

def obtener_vectores_propiedades(compuestos, propiedades):
    """
    devuelve todas las propiedades de todos los compuestos como vectores columnas de propiedades:
    Ejemplo:
    tc, pc, w = obtener_propiedades(['water', 'ethanol'], ['tc', 'pc', 'w'])
    """
    # Obtenemos vectores de las filas y columnas de los compuestos y propiedades
    filas = obtener_ids_lista(compuestos)
    columnas = obtener_columnas_lista(propiedades)

    # Generamos todas las combinaciones posibles de filas con columnas
    filas, columnas = np.meshgrid(filas, columnas, indexing='ij')

    # Generamos matriz a partir de la base de datos
    matriz_propiedades = PROP_TERMOF[filas, columnas].copy()

    # Separamos las columnas de la matriz en vectores columna de propiedades
    vectores_de_propiedades = list()
    for col in range(matriz_propiedades.shape[1]):
        vector_columna = matriz_propiedades[:, [col]].astype(np.float64)
        vectores_de_propiedades.append(vector_columna)

    # Devolvemos los vectores de propiedades desempaquetados
    return tuple(vectores_de_propiedades)


def convertir_numpy_array(*vectores, arr_columna=False):
    """
    Toma todos los vectores de variables termodinamicas y los convierte a:
    np.array con dtype=np.float64.
    Si arr_columna es True: Devuelve un vector con shape = (C, 1)
    Si arr_columna es False: Devuelve un vector con shape = (1, T)
    """
    lista_arrays = list()
    for variable in vectores:
        array = np.array(variable, dtype=np.float64).ravel()

        # Le colocamos la shape adecuada
        if arr_columna:
            array = array[:, np.newaxis]
        else:
            array = array[np.newaxis, :]

        lista_arrays.append(array)

    # Si lista_arrays solo tiene un array, devuelvelo asi solito.
    if len(lista_arrays) == 1:
        return lista_arrays[0]
    # Si lista_arrays tiene mas de un array, devuelvelos como una tupla para desempaquetar.
    else:
        return tuple(lista_arrays)


def transposicion(matriz : list) -> list:
    """
    Devuelve la transpuesta de una matriz de python.
    """
    numero_columnas = len(matriz[0])
    transpuesta = list()

    for col in range(numero_columnas):
        columna = [fila[col] for fila in matriz]
        transpuesta.append(columna)

    return transpuesta


# FUNCIONES PARA EXCEL

# Relacionadas a la base de datos
def obtener(compuestos, propiedades):
    """
    Devuelve una matriz que contiene las propiedades especificadas para los compuestos
    especificados.
    [[compuesto1propiedad1, compuesto1propiedad2] ... 
     [compuesto2propiedad1, compuesto2propiedad2]
     ...

    Args:
        compuestos (list): Vector de algun identificador de compuesto (id, cas, nombre)
        propiedades (list): Vector de algun identificador de propiedad (num de columna, nombre)

    Returns:
        Matriz de propiedades de compuestos.
    """
    # Determinamos la orientacion de la matriz a imprimir
    compuestos_son_verticales = determinar_orietnacion_tabla(compuestos, propiedades)

    # Obtenemos numero de filas y columnas de la matriz
    filas = obtener_ids_lista(compuestos)
    columnas = obtener_columnas_lista(propiedades)

    # Generamos todas las combinaciones posibles de filas con columnas
    filas, columnas = np.meshgrid(filas, columnas, indexing='ij')

    # Generamos matriz a partir de la base de datos
    matriz_propiedades = PROP_TERMOF[filas, columnas].copy()

    if compuestos_son_verticales:
        return matriz_propiedades
    else:
        return matriz_propiedades.transpose()


def modificar_datos(compuestos, propiedades, valores):
    """
    Esta funcion inserta valores definidos por el usuario en la base de datos PROP_TERMOF. No modifica el archivo de
    la base de datos. Unicamente una copia que se tiene cargada mientras se ejecuta el programa
    Args:
        compuestos (list): Vector de identificadoes de compuesto (id, cas, nombre)
        propiedades (list): Vector de identificadores de propiedades de compuesto (numero de columna, nombre)
        valores (list): Matriz del valores correspondientes de propiedades de cada compuesto que se desea agregar
        a la base de datos
    Retruns:
        (string): El string 'Datos añadidos' para que se pueda ver en que celda se llamo a la funcion
        
    """
    # Determinamos la orientacion de la matriz de valores
    orientacion_tabla = determinar_orietnacion_tabla(compuestos, propiedades)

    # Convertimos valores a np.ndarray
    valores = np.array(valores, dtype=object)

    # Si los valores tienen orientacion (PROPIEDADES, COMPUESTOS), la transponemos a (COMPUESTOS, PROPIEDADES)
    if not orientacion_tabla:
        valores = valores.transpose()

    # Obtenemos vectores del numero de filas y columnas de los compuestos y propiedades seleccionadas
    compuestos = obtener_ids_lista(compuestos)
    propiedades = obtener_columnas_lista(propiedades)

    # Colocamos compuestos como vector columna y propiedades como vector fila
    compuestos = compuestos[:, np.newaxis]
    propiedades = propiedades[np.newaxis, :]

    # Insertamos valores en la base de datos
    PROP_TERMOF[compuestos, propiedades] = valores

    return 'Datos añadidos'


# Ecuaciones cubicas de estado
def van_der_waals(compuestos, composiciones, presion=None, temperatura=None, volumen=None, raw=False):
    """
    Dependiendo de que variable no especifiques (presion, temperatura, volumen). Te la devolvera
    usando la ecuacion de van_der_waals.
    Es una funcion de mezcla, devuelve un vector.
    Args:
        compuestos (list): Vector de referencias a los compuestos de la mezcla.
        composiciones (list): Vector de las composciones correspondientes a los compuestos.
        presion (list): Vector de presiones en bar
        temperatura (list): Vector de temperatura en Kelvin
        volumen (list): Vector de volumenes molares en mL / mol

    Returns:
        presion, temperatura, volumen (list): Dependiendo de que variable no especifiques.
    """
    # Determinamos que variable es la incognita
    variables = {'p', 't', 'v'}
    variables_conocidas = set()

    if presion:
        variables_conocidas.add('p')
    if temperatura:
        variables_conocidas.add('t')
    if volumen:
        variables_conocidas.add('v')

    incognita = variables - variables_conocidas

    # Obtenemos propiedades termodinamicas de la base de datos
    tc, pc = obtener_vectores_propiedades(compuestos, ['tc', 'pc'])

    # Convertimos variables de compuestos a un array columna (C, 1)
    arr_composiciones = convertir_numpy_array(composiciones, arr_columna=True)

    # Convertimos variables termodinamicas a un array fila (1, T)
    arr_presion, arr_temperatura, arr_volumen = convertir_numpy_array(presion, temperatura, volumen, arr_columna=False)

    # Por ahora no cuento con factores de interaccion binaria en la base de datos. Asumimos cero
    k = np.zeros((len(arr_composiciones), len(arr_composiciones)), dtype=np.float64)

    # Si la incognita es la presion
    if incognita == {'p'}:
        arr_presion = motor_de_calculo.van_der_waals_presion(arr_temperatura, arr_volumen, arr_composiciones, pc, tc, k)

        # Devuelve el vector de presion vertical u horizontal dependiendo de la orientacion del vector temperatura
        if es_vector_columna(temperatura) == True:
            return arr_presion.transpose()
        else:
            return arr_presion
        
    # Si la incognita es la temperatura
    if incognita == {'t'}:
        arr_temperatura = motor_de_calculo.van_der_waals_temperatura(arr_presion, arr_volumen, arr_composiciones, pc, tc, k)
        
        # Devuelve el vector de temperaturas vertical u horizontal dependiendo de la orientacion del vector presion
        if es_vector_columna(presion) == True:
            return arr_temperatura.transpose()
        else:
            return arr_temperatura
        
    # Si la incognita es el volumen
    if incognita == {'v'}:
        arr_volumen = motor_de_calculo.van_der_waals_volumen(arr_presion, arr_temperatura, arr_composiciones, pc, tc, k)

        # Devuelve los volumenes vertical u horizontal dependiendo de la orientacion de las temperaturas
        if es_vector_columna(temperatura) == True:
            return arr_volumen.transpose()
        else:
            return arr_volumen
    
    # Si no hay incognitas
    if incognita == set():
        presion_calculada = motor_de_calculo.van_der_waals_presion(arr_temperatura, arr_volumen, arr_composiciones, pc, tc, k)

        # Calculamos el error entre la presion proporcionada y la calculada con la ecuacion
        error = np.abs(arr_presion - presion_calculada) / arr_presion

        # Si la opcion raw esta habilidata, devuelve el valor del error directamente
        if raw:
            if es_vector_columna(presion) == True:
                return error.transpose()
            else:
                return error

        # Si raw esta deshabilitado devuelve un array de booleanos dependiendo de si el error esta dentro de un margen de error
        valores_validos = error <= ERROR_PERMITIDO
        if es_vector_columna(presion) == True:
            return valores_validos.transpose()
        else:
            return valores_validos


def redlich_kwong(compuestos, composiciones, presion=None, temperatura=None, volumen=None, raw=False):
    """
    Dependiendo de que variable no especifiques (presion, temperatura, volumen). Te la devolvera
    usando la ecuacion de redlich kwong.
    Es una funcion de mezcla, devuelve un vector.
    Args:
        compuestos (list): Vector de referencias a los compuestos de la mezcla.
        composiciones (list): Vector de las composciones correspondientes a los compuestos.
        presion (list): Vector de presiones en bar
        temperatura (list): Vector de temperatura en Kelvin
        volumen (list): Vector de volumenes molares en mL / mol

    Returns:
        presion, temperatura, volumen (list): Dependiendo de que variable no especifiques.
    """
    # Determinamos que variable es la incognita
    variables = {'p', 't', 'v'}
    variables_conocidas = set()

    if presion:
        variables_conocidas.add('p')
    if temperatura:
        variables_conocidas.add('t')
    if volumen:
        variables_conocidas.add('v')

    incognita = variables - variables_conocidas

    # Obtenemos propiedades termodinamicas de la base de datos
    tc, pc = obtener_vectores_propiedades(compuestos, ['tc', 'pc'])

    # Convertimos variables de compuestos a un array columna (C, 1)
    arr_composiciones = convertir_numpy_array(composiciones, arr_columna=True)

    # Convertimos variables termodinamicas a un array fila (1, T)
    arr_presion, arr_temperatura, arr_volumen = convertir_numpy_array(presion, temperatura, volumen, arr_columna=False)

    # Por ahora no cuento con factores de interaccion binaria en la base de datos. Asumimos cero
    k = np.zeros((len(arr_composiciones), len(arr_composiciones)), dtype=np.float64)

    # Si la incognita es la presion
    if incognita == {'p'}:
        arr_presion = motor_de_calculo.redlich_kwong_presion(arr_temperatura, arr_volumen, arr_composiciones, pc, tc, k)

        # Devuelve el vector de presion vertical u horizontal dependiendo de la orientacion del vector temperatura
        if es_vector_columna(temperatura) == True:
            return arr_presion.transpose()
        else:
            return arr_presion
        
    # Si la incognita es la temperatura
    if incognita == {'t'}:
        arr_temperatura = motor_de_calculo.redlich_kwong_temperatura(arr_presion, arr_volumen, arr_composiciones, pc, tc, k)
        
        # Devuelve el vector de temperaturas vertical u horizontal dependiendo de la orientacion del vector presion
        if es_vector_columna(presion) == True:
            return arr_temperatura.transpose()
        else:
            return arr_temperatura
        
    # Si la incognita es el volumen
    if incognita == {'v'}:
        arr_volumen = motor_de_calculo.redlich_kwong_volumen(arr_presion, arr_temperatura, arr_composiciones, pc, tc, k)

        # Devuelve los volumenes vertical u horizontal dependiendo de la orientacion de las temperaturas
        if es_vector_columna(temperatura) == True:
            return arr_volumen.transpose()
        else:
            return arr_volumen
    
    # Si no hay incognitas
    if incognita == set():
        presion_calculada = motor_de_calculo.redlich_kwong_presion(arr_temperatura, arr_volumen, arr_composiciones, pc, tc, k)

        # Calculamos el error entre la presion proporcionada y la calculada con la ecuacion
        error = np.abs(arr_presion - presion_calculada) / arr_presion

        # Si la opcion raw esta habilidata, devuelve el valor del error directamente
        if raw:
            if es_vector_columna(presion) == True:
                return error.transpose()
            else:
                return error

        # Si raw esta deshabilitado devuelve un array de booleanos dependiendo de si el error esta dentro de un margen de error
        valores_validos = error <= ERROR_PERMITIDO
        if es_vector_columna(presion) == True:
            return valores_validos.transpose()
        else:
            return valores_validos


def soave(compuestos, composiciones, presion=None, temperatura=None, volumen=None, raw=False):
    """
    Dependiendo de que variable no especifiques (presion, temperatura, volumen). Te la devolvera
    usando la ecuacion de Soave.
    Es una funcion de mezcla, devuelve un vector.
    Args:
        compuestos (list): Vector de referencias a los compuestos de la mezcla.
        composiciones (list): Vector de las composciones correspondientes a los compuestos.
        presion (list): Vector de presiones en bar
        temperatura (list): Vector de temperatura en Kelvin
        volumen (list): Vector de volumenes molares en mL / mol

    Returns:
        presion, temperatura, volumen (list): Dependiendo de que variable no especifiques.
    """
    # Determinamos que variable es la incognita
    variables = {'p', 't', 'v'}
    variables_conocidas = set()

    if presion:
        variables_conocidas.add('p')
    if temperatura:
        variables_conocidas.add('t')
    if volumen:
        variables_conocidas.add('v')

    incognita = variables - variables_conocidas

    # Obtenemos propiedades termodinamicas de la base de datos
    tc, pc, w = obtener_vectores_propiedades(compuestos, ['tc', 'pc', 'w'])

    # Convertimos variables de compuestos a un array columna (C, 1)
    arr_composiciones = convertir_numpy_array(composiciones, arr_columna=True)

    # Convertimos variables termodinamicas a un array fila (1, T)
    arr_presion, arr_temperatura, arr_volumen = convertir_numpy_array(presion, temperatura, volumen, arr_columna=False)

    # Por ahora no cuento con factores de interaccion binaria en la base de datos. Asumimos cero
    k = np.zeros((len(arr_composiciones), len(arr_composiciones)), dtype=np.float64)

    # Si la incognita es la presion
    if incognita == {'p'}:
        arr_presion = motor_de_calculo.soave_presion(arr_temperatura, arr_volumen, arr_composiciones, pc, tc, w, k)

        # Devuelve el vector de presion vertical u horizontal dependiendo de la orientacion del vector temperatura
        if es_vector_columna(temperatura) == True:
            return arr_presion.transpose()
        else:
            return arr_presion
        
    # Si la incognita es la temperatura
    if incognita == {'t'}:
        arr_temperatura = motor_de_calculo.soave_temperatura(arr_presion, arr_volumen, arr_composiciones, pc, tc, w, k)
        
        # Devuelve el vector de temperaturas vertical u horizontal dependiendo de la orientacion del vector presion
        if es_vector_columna(presion) == True:
            return arr_temperatura.transpose()
        else:
            return arr_temperatura
        
    # Si la incognita es el volumen
    if incognita == {'v'}:
        arr_volumen = motor_de_calculo.soave_volumen(arr_presion, arr_temperatura, arr_composiciones, pc, tc, w, k)

        # Devuelve los volumenes vertical u horizontal dependiendo de la orientacion de las temperaturas
        if es_vector_columna(temperatura) == True:
            return arr_volumen.transpose()
        else:
            return arr_volumen
    
    # Si no hay incognitas
    if incognita == set():
        presion_calculada = motor_de_calculo.soave_presion(arr_temperatura, arr_volumen, arr_composiciones, pc, tc, w, k)

        # Calculamos el error entre la presion proporcionada y la calculada con la ecuacion
        error = np.abs(arr_presion - presion_calculada) / arr_presion

        # Si la opcion raw esta habilidata, devuelve el valor del error directamente
        if raw:
            if es_vector_columna(presion) == True:
                return error.transpose()
            else:
                return error

        # Si raw esta deshabilitado devuelve un array de booleanos dependiendo de si el error esta dentro de un margen de error
        valores_validos = error <= ERROR_PERMITIDO
        if es_vector_columna(presion) == True:
            return valores_validos.transpose()
        else:
            return valores_validos


def peng_robinson(compuestos, composiciones, presion=None, temperatura=None, volumen=None, raw=False):
    """
    Dependiendo de que variable no especifiques (presion, temperatura, volumen). Te la devolvera
    usando la ecuacion de Peng Robinson.
    Es una funcion de mezcla, devuelve un vector.
    Args:
        compuestos (list): Vector de referencias a los compuestos de la mezcla.
        composiciones (list): Vector de las composciones correspondientes a los compuestos.
        presion (list): Vector de presiones en bar
        temperatura (list): Vector de temperatura en Kelvin
        volumen (list): Vector de volumenes molares en mL / mol

    Returns:
        presion, temperatura, volumen (list): Dependiendo de que variable no especifiques.
    """
    # Determinamos que variable es la incognita
    variables = {'p', 't', 'v'}
    variables_conocidas = set()

    if presion:
        variables_conocidas.add('p')
    if temperatura:
        variables_conocidas.add('t')
    if volumen:
        variables_conocidas.add('v')

    incognita = variables - variables_conocidas

    # Obtenemos propiedades termodinamicas de la base de datos
    tc, pc, w = obtener_vectores_propiedades(compuestos, ['tc', 'pc', 'w'])

    # Convertimos variables de compuestos a un array columna (C, 1)
    arr_composiciones = convertir_numpy_array(composiciones, arr_columna=True)

    # Convertimos variables termodinamicas a un array fila (1, T)
    arr_presion, arr_temperatura, arr_volumen = convertir_numpy_array(presion, temperatura, volumen, arr_columna=False)

    # Por ahora no cuento con factores de interaccion binaria en la base de datos. Asumimos cero
    k = np.zeros((len(arr_composiciones), len(arr_composiciones)), dtype=np.float64)

    # Si la incognita es la presion
    if incognita == {'p'}:
        arr_presion = motor_de_calculo.peng_robinson_presion(arr_temperatura, arr_volumen, arr_composiciones, pc, tc, w, k)

        # Devuelve el vector de presion vertical u horizontal dependiendo de la orientacion del vector temperatura
        if es_vector_columna(temperatura) == True:
            return arr_presion.transpose()
        else:
            return arr_presion
        
    # Si la incognita es la temperatura
    if incognita == {'t'}:
        arr_temperatura = motor_de_calculo.peng_robinson_temperatura(arr_presion, arr_volumen, arr_composiciones, pc, tc, w, k)
        
        # Devuelve el vector de temperaturas vertical u horizontal dependiendo de la orientacion del vector presion
        if es_vector_columna(presion) == True:
            return arr_temperatura.transpose()
        else:
            return arr_temperatura
        
    # Si la incognita es el volumen
    if incognita == {'v'}:
        arr_volumen = motor_de_calculo.peng_robinson_volumen(arr_presion, arr_temperatura, arr_composiciones, pc, tc, w, k)

        # Devuelve los volumenes vertical u horizontal dependiendo de la orientacion de las temperaturas
        if es_vector_columna(temperatura) == True:
            return arr_volumen.transpose()
        else:
            return arr_volumen
    
    # Si no hay incognitas
    if incognita == set():
        presion_calculada = motor_de_calculo.peng_robinson_presion(arr_temperatura, arr_volumen, arr_composiciones, pc, tc, w, k)

        # Calculamos el error entre la presion proporcionada y la calculada con la ecuacion
        error = np.abs(arr_presion - presion_calculada) / arr_presion

        # Si la opcion raw esta habilidata, devuelve el valor del error directamente
        if raw:
            if es_vector_columna(presion) == True:
                return error.transpose()
            else:
                return error

        # Si raw esta deshabilitado devuelve un array de booleanos dependiendo de si el error esta dentro de un margen de error
        valores_validos = error <= ERROR_PERMITIDO
        if es_vector_columna(presion) == True:
            return valores_validos.transpose()
        else:
            return valores_validos


def antoine(compuestos, presion_vapor=None, temperatura=None, raw=False):
    """
    Dependiendo de que variable no especifiques. La calcula usando la ecuacion de antoine.
    log10(Pvap) = a - b/(temp + c).
    Es una funcion de compuesto, devuelve una matriz.
    Args:
        compuestos (list): Vector de referencias a los compuestos.
        presion_vapor (list): Vector de presiones de vapor en bar
        temperatura (list): Vector de temperatuas en kelvin
    Returns
        arr_presion, arr_temperatura (np.ndarray): Matriz de presion de vapor o temperaturas.
    """
    # Determinamos incognita
    variables = {'p','t'}
    variables_conocidas = set()
    if presion_vapor:
        variables_conocidas.add('p')
    if temperatura:
        variables_conocidas.add('t')
    incognita = variables - variables_conocidas

    # Obtenemos propiedades de la base de datos
    tmin, pmin, tmax, pmax, a, b, c = obtener_vectores_propiedades(compuestos, ['anttmin', 'antpvmin', 'anttmax', 'antpvmax', 'anta', 'antb', 'antc'])

    # Convertimos variables termodinamicas en np.ndarray fila
    arr_presion, arr_temperatura = convertir_numpy_array(presion_vapor, temperatura, arr_columna=False)

    # Si la incognita es la presion de vapor
    if incognita == {'p'}:
        arr_presion = motor_de_calculo.antoine_presion_vapor(arr_temperatura, a, b, c)

        # Si raw esta deshabilitado. Marcamos valores que no esten en el rango de la ecuacion.
        if not raw:
            temperaturas_invalidas = np.logical_or(arr_temperatura < tmin, arr_temperatura > tmax)
            presiones_invalidas = np.logical_or(arr_presion < pmin, arr_presion > pmax)
            arr_presion = arr_presion.astype(object)
            arr_presion[temperaturas_invalidas] = f'T fuera de rango'
            arr_presion[presiones_invalidas] = f'P fuera de rango'

        # Determinamos orientacion y devolvemos
        if determinar_orietnacion_tabla(compuestos, temperatura):
            return arr_presion
        else:
            return arr_presion.transpose()
        
    # Si la incognita es la temperatura
    if incognita == {'t'}:
        arr_temperatura = motor_de_calculo.antoine_temperatura(arr_presion, a, b, c)

        # Si raw esta deshabilitado. Marcamos valores que no esten en el rango de la ecuacion
        if not raw:
            temperaturas_invalidas = np.logical_or(arr_temperatura < tmin, arr_temperatura > tmax)
            presiones_invalidas = np.logical_or(arr_presion < pmin, arr_presion > pmax)
            arr_temperatura = arr_temperatura.astype(object)
            arr_temperatura[temperaturas_invalidas] = f'T fuera de rango'
            arr_temperatura[presiones_invalidas] = f'P fuera de rango'

        # Determinamos orientacion y devolvemos
        if determinar_orietnacion_tabla(compuestos, presion_vapor):
            return arr_temperatura
        else:
            return arr_temperatura.transpose()
        
    if incognita == set():
        return f'Sobrespecificado...'


def cp(compuestos, temperatura=None, capacidad_calorifica=None, raw=False):
    """
    Dependiendo de que variable no especifiques. La calcula usando la ecuacion de cp polinomial.
    Cp(T) = a0 + a1xT + a2xT^2 + a3xT^3 + a4xT^4.
    Es una funcion de compuesto, regresa una matriz.
    Args:
        compuestos (list): Vector de referencias a los compuestos.
        temperatura (list): Vector de las temperaturas en K
        capacidad_calorifica: Vector de capacidades calorificas en J / (mol * K)
    Returns:
        arr_temperatura, arr_capacidad calorifica (np.ndarray): Dependiendo de que variable no especifiques. La calculara
        y te la devolvera.

    NOTA:
        Cuando se especifican las dos variables. El programa intentara determinar si la ecuacion
        se satisface. Si este es el caso. Entonces tanto las capacidades calorificas como los compuestos
        deberan ser matrices (C, T).
    """
    # Determinamos que variable es la incognita
    variables = {'t', 'c'}
    variables_conocidas = set()
    if temperatura:
        variables_conocidas.add('t')
    if capacidad_calorifica:
        variables_conocidas.add('c')
    incognita = variables - variables_conocidas

    # Obtenemos propiedades termodinamicas de la base de datos
    tmin, tmax, a0, a1, a2, a3, a4 = obtener_vectores_propiedades(compuestos, ['cptmin', 'cptmax', 'cpa0', 'cpa1', 'cpa2', 'cpa3', 'cpa4'])

    # Convertimos variables termodinamicas a un array fila. (1, T)
    arr_temperatura, arr_cp = convertir_numpy_array(temperatura, capacidad_calorifica, arr_columna=False)

    # Si la incognita es la capacidad calorifica
    if incognita == {'c'}:
        # Calculamos cp
        arr_cp = motor_de_calculo.cp_gas_ideal(arr_temperatura, a0, a1, a2, a3, a4)

        # Si raw esta deshabilitado. Marca todas las temperaturas fuera de rango de la ecuacion
        if not raw:
            temp_debajo_rango = arr_temperatura < tmin
            temp_encima_rango = arr_temperatura > tmax
            temp_fuera_rango = np.logical_or(temp_debajo_rango, temp_encima_rango)

            arr_cp = arr_cp.astype(object)
            arr_cp[temp_fuera_rango] = f'T fuera de rango'

        # Determinamos orientacion de la matriz a imprimir y la devolvemos
        if determinar_orietnacion_tabla(compuestos, temperatura):
            return arr_cp
        else:
            return arr_cp.transpose()
        
    # Si la incognita es la temperatura
    if incognita == {'t'}:
        # Calculamos la temperatura
        arr_temperatura = motor_de_calculo.cp_gas_ideal_temperatura(arr_cp, a0, a1, a2, a3, a4)

        # Si raw esta deshabilitado. Marcamos las temperaturas que estaf fuera de rango
        if not raw:
            temp_debajo_rango = arr_temperatura < tmin
            temp_encima_rango = arr_temperatura > tmax
            temp_fuera_rango = np.logical_or(temp_debajo_rango, temp_encima_rango)

            arr_temperatura = arr_temperatura.astype(object)
            arr_temperatura[temp_fuera_rango] = f'T fuera de rango'

        # Determinamos orientacion de la matriz a imprimir y la devolvemos
        if determinar_orietnacion_tabla(compuestos, capacidad_calorifica):
            return arr_temperatura
        else:
            return arr_temperatura.transpose()

    # Si no hay incognitas, devolver error... Para checar si se satisface una ecuacion de compeusto, habria que tomar un vector y una matriz y es un rollo y ni tiene tanta utilidad...
    if incognita == set():
        return f'Sobrespecificado...'


def integral_cp(compuestos, temperatura1=None, temperatura2=None, delta_h=None, raw=False):
    """
    Evalua la integral del cp polinomial. La ecuacion tiene la forma:
    delta_h = integral(cp(T)dT, temperatura1, temperatura2)
    Es una funcion de compuesto, devuelve una matriz.
    Args:
        compuestos (list): Vector de referencias a compuestos
        temperatura1, temperatura2 (list): Vectores de rangos de temperaturas en k
    Returns:
        temperatua1, temperatura2, delta_h: Dependiendo de que variable no especifiques, la devuelve
        tal que satisfaga la ecuacion.
    """
    # Determinamos incognitas y orientacion de la tabla a imprimir al final.
    variables = {'t1', 't2', 'h'}
    variables_conocidas = set()
    if temperatura1:
        variables_conocidas.add('t1')
        orientacion_tabla = determinar_orietnacion_tabla(compuestos, temperatura1)
    if temperatura2:
        variables_conocidas.add('t2')
        orientacion_tabla = determinar_orietnacion_tabla(compuestos, temperatura2)
    if delta_h:
        variables_conocidas.add('h')
        orientacion_tabla = determinar_orietnacion_tabla(compuestos, delta_h)
    
    incognita = variables - variables_conocidas
        
    # Obtenemos propiedades termodinamicas de la base de datos
    tmin, tmax, a0, a1, a2, a3, a4 = obtener_vectores_propiedades(compuestos, ['cptmin', 'cptmax', 'cpa0', 'cpa1', 'cpa2', 'cpa3', 'cpa4'])

    # Convertimos variables termodinamicas en un np.ndarray fila
    arr_t1, arr_t2, arr_h = convertir_numpy_array(temperatura1, temperatura2, delta_h, arr_columna=False)

    # Si la incognita es delta_h
    if incognita == {'h'}:
        arr_h = motor_de_calculo.integral_cp_gas_ideal(arr_t1, arr_t2, a0, a1, a2, a3, a4)
        resultado = arr_h
        
    # Si la incognita es la temperatura 1
    if incognita == {'t1'}:
        # Usando una propiedad de las integrales de una variable, se pueden invertir los limites de integracion
        arr_t1 = motor_de_calculo.integral_cp_gas_ideal_temperatura_2(arr_t2, -arr_h, a0, a1, a2, a3, a4) 
        resultado = arr_t1 

    # Si la incognita es la temperatura 2
    if incognita == {'t2'}:
        arr_t2 = motor_de_calculo.integral_cp_gas_ideal_temperatura_2(arr_t1, arr_h, a0, a1, a2, a3, a4)
        resultado = arr_t2

    # Si raw esta deshabilitado. Marcamos valores invalidos
    if not raw:
        temp1_fuera_rango = np.logical_or(arr_t1 < tmin, arr_t1 > tmax)
        temp2_fuera_rango = np.logical_or(arr_t2 < tmin, arr_t2 > tmax)
        resultado = resultado.astype(object)
        resultado[temp1_fuera_rango] = f'T fuera de rango'
        resultado[temp2_fuera_rango] = f'T fuera de rango'

    # Devolvemos matriz en funcion de la orientacion de las variables.
    if orientacion_tabla:
         return resultado
    else:
         return resultado.transpose()


def coeficiente_fugacidad_mezcla_peng_robinson(compuestos, composiciones, presion=None, temperatura=None, raiz_volumen=3, raw=False):
    """
    Calcula el coeficiente de fugacidad de un compuesto en una mezcla homogenea usando la ecuación de peng_robinson.
    Funcion de compuesto. Devuelve una matriz
    Args:
        compuestos (list): Lista de referencias a compuestos
        composiciones (list): Lista de las composiciones de los compuestos
        temperatura (list): Lista de temperaturas en kelvin
        presion (List): Lista de presiones en bar
        raiz_volumen (int): La ecuacion de peng robinson devuelve 3 volumenes. Aqui escoges con cual operar.
        raw=False (bool): Si se activa, todos los mecanismos de prevencion de errores se desactivan
    Returns
        arr_fugacidad (np.ndarray): Devuelve una matriz del coeficiente de fugacidad de cada compuesto a las diferentes
        condiciones
    """
    # Determinamos incognita y orientacion de la tabla a devolver
    orientacion_tabla = determinar_orietnacion_tabla(compuestos, presion)

    # Obtenemos propiedades termodinamicas de la base de datos
    pc, tc, w = obtener_vectores_propiedades(compuestos, ['pc','tc','w'])

    # Convertimos variables de compuesto en un np.ndarray columna
    arr_composiciones = convertir_numpy_array(composiciones, arr_columna=True)

    # Convertimos variables termodinamicas en un np.ndarray fila
    arr_presion, arr_temperatura = convertir_numpy_array(presion, temperatura, arr_columna=False)

    # La base de datos aun no cuenta con coeficientes de interaccion binaria. Asumimos cero
    k = np.zeros((len(arr_composiciones), len(arr_composiciones)))

    # Calculamos fugacidades
    indice_volumen = int(raiz_volumen[0][0] - 1)
    arr_fugacidad = motor_de_calculo.coef_fugacidad_mezcla_peng_robinson(arr_presion, arr_temperatura, arr_composiciones, pc, tc, w, k, volumen_num=indice_volumen)

    # Devolvemos matriz
    if orientacion_tabla:
        return arr_fugacidad
    else:
        return arr_fugacidad.transpose()


def coeficiente_fugacidad_mezcla_soave(compuestos, composiciones, presion=None, temperatura=None, raiz_volumen=3, raw=False):
    """
    Calcula el coeficiente de fugacidad de un compuesto en una mezcla homogenea usando la ecuación de Soave.
    Funcion de compuesto. Devuelve una matriz
    Args:
        compuestos (list): Lista de referencias a compuestos
        composiciones (list): Lista de las composiciones de los compuestos
        temperatura (list): Lista de temperaturas en kelvin
        presion (List): Lista de presiones en bar
        raiz_volumen (int): La ecuacion de peng robinson devuelve 3 volumenes. Aqui escoges con cual operar.
        raw=False (bool): Si se activa, todos los mecanismos de prevencion de errores se desactivan
    Returns
        arr_fugacidad (np.ndarray): Devuelve una matriz del coeficiente de fugacidad de cada compuesto a las diferentes
        condiciones
    """
    # Determinamos incognita y orientacion de la tabla a devolver
    orientacion_tabla = determinar_orietnacion_tabla(compuestos, presion)

    # Obtenemos propiedades termodinamicas de la base de datos
    pc, tc, w = obtener_vectores_propiedades(compuestos, ['pc','tc','w'])

    # Convertimos variables de compuesto en un np.ndarray columna
    arr_composiciones = convertir_numpy_array(composiciones, arr_columna=True)

    # Convertimos variables termodinamicas en un np.ndarray fila
    arr_presion, arr_temperatura = convertir_numpy_array(presion, temperatura, arr_columna=False)

    # La base de datos aun no cuenta con coeficientes de interaccion binaria. Asumimos cero
    k = np.zeros((len(arr_composiciones), len(arr_composiciones)))

    # Calculamos fugacidades
    indice_volumen = int(raiz_volumen[0][0] - 1)
    arr_fugacidad = motor_de_calculo.coef_fugacidad_mezcla_soave(arr_presion, arr_temperatura, arr_composiciones, pc, tc, w, k, volumen_num=indice_volumen)

    # Devolvemos matriz
    if orientacion_tabla:
        return arr_fugacidad
    else:
        return arr_fugacidad.transpose()


def entropia_ideal(compuestos, composicion_liquida, composicion_vapor, presion, temperatura, calidad_vapor):
    # Definimos orientacion a imprimir
    orientacion_tabla = determinar_orietnacion_tabla(compuestos, presion)

    # Obtenemos propiedades termodinamicas
    pc, tc, w, antoine_a, antoine_b, antoine_c, a0, a1, a2, a3, a4, cpliq = obtener_vectores_propiedades(compuestos, ['pc','tc','w','anta','antb','antc','cpa0','cpa1','cpa2','cpa3','cpa4','cpliq'])

    # Generamos np.ndarray vector columna
    arr_xi, arr_yi = convertir_numpy_array(composicion_liquida, composicion_vapor, arr_columna=True)

    # Generamos np.ndarray vector fila
    arr_presion, arr_temp, arr_calidad = convertir_numpy_array(presion, temperatura, calidad_vapor, arr_columna=False)

    # Calculamos entropia ideal
    entropia = motor_de_calculo.entropia_mezcla_ideal(arr_presion, arr_temp, arr_xi, arr_yi, arr_calidad, tc, pc, w, antoine_a, antoine_b, antoine_c, a0, a1, a2, a3, a4, cpliq)

    # Devolvemos en orientacion
    if orientacion_tabla:
        return entropia
    else:
        return entropia.transpose()


def entalpia_ideal(compuestos, composicion_liquida, composicion_vapor, temperatura, calidad_vapor):
    # Definimos orientacion a imprimir
    orientacion_tabla = determinar_orietnacion_tabla(compuestos, temperatura)

    # Obtenemos propiedades termodinamicas
    pc, tc, w, antoine_a, antoine_b, antoine_c, a0, a1, a2, a3, a4, cpliq = obtener_vectores_propiedades(compuestos, ['pc','tc','w','anta','antb','antc','cpa0','cpa1','cpa2','cpa3','cpa4','cpliq'])

    # Generamos np.ndarray vector columna
    arr_xi, arr_yi = convertir_numpy_array(composicion_liquida, composicion_vapor, arr_columna=True)

    # Generamos np.ndarray vector fila
    arr_temp, arr_calidad = convertir_numpy_array(temperatura, calidad_vapor, arr_columna=False)

    # Calculamos entalpia ideal
    entalpia = motor_de_calculo.entalpia_mezcla_ideal(arr_temp, arr_xi, arr_yi, arr_calidad, tc, pc, w, antoine_a, antoine_b, antoine_c, a0, a1, a2, a3, a4, cpliq)

    # Devolvemos entalpia
    if orientacion_tabla:
        return entalpia
    else:
        return entalpia.transpose()


def entalpia_ideal_vapor(compuestos, composiciones, temperatura):
    """
    Calcula la entalpia de una mezcla completamente vaporizada desde el estado de referencia: Liquido saturado
    a 0°C.
    """
    # Determinamos la orientacion de la tabla a imprimir
    orientacion_tabla = determinar_orietnacion_tabla(compuestos, temperatura)

    # Obtenemos propiedades termodinamicas
    pc, tc, w, antoine_a, antoine_b, antoine_c, a0, a1, a2, a3, a4 = obtener_vectores_propiedades(compuestos, ['pc','tc','w','anta','antb','antc','cpa0','cpa1','cpa2','cpa3','cpa4'])

    # Generamos np.ndarrays de vectores columa
    arr_composiciones = convertir_numpy_array(composiciones, arr_columna=True)

    # Generamos np.ndarrays de vectores fila
    arr_temperatura = convertir_numpy_array(temperatura, arr_columna=False)

    # Calculamos la entalpia de la mezcla vaporizada ideal
    entalpia = motor_de_calculo.entalpia_mezcla_ideal_vapor(arr_temperatura, arr_composiciones, tc, pc, w, antoine_a, antoine_b, antoine_c, a0, a1, a2, a3, a4)

    # Devolvemos entalpia
    if orientacion_tabla:
        return entalpia
    else:
        return entalpia.transpose()


def coeficiente_fugacidad_puro_peng_robinson(compuestos, presion, temperatura, raiz_volumen=3, raw=False):
    """
    Calcula el coeficiente de fugacidad de los compuestos como substancia pura usando la ecuacion de peng robinson.
    Funcion de compuesto. Devuelve una matriz
    Args:
        compuestos (list): Lista de referencias a compuestos
        presion (List): Lista de presiones en bar
        temperatura (list): Lista de temperaturas en kelvin
        raiz_volumen (int): La ecuacion de peng robinson devuelve 3 volumenes. Aqui escoges con cual operar.
        raw=False (bool): Si se activa, todos los mecanismos de prevencion de errores se desactivan
    Returns
        arr_fugacidad (np.ndarray): Devuelve una matriz del coeficiente de fugacidad de cada compuesto a las diferentes
        condiciones
    """
    # Determinamos orientacion de la tabla a devolver
    orientacion_tabla = determinar_orietnacion_tabla(compuestos, presion)

    # Obtenemos propiedades termodinamicas de la base de datos
    pc, tc, w = obtener_vectores_propiedades(compuestos, ['pc','tc','w'])

    # Convertimos variables termodinamicas en vectores fila
    arr_presion, arr_temp = convertir_numpy_array(presion, temperatura, arr_columna=False)

    # Calculamos las fugacidades
    indice_volumen = int(raiz_volumen[0][0] - 1)
    arr_fugacidad = motor_de_calculo.coef_fugacidad_puro_peng_robinson(arr_presion, arr_temp, pc, tc, w, volumen_num=indice_volumen)

    # Devolvemos tabla de coeficientes de fuagicad
    if orientacion_tabla:
        return arr_fugacidad
    else:
        return arr_fugacidad.transpose()
    

def coeficiente_fugacidad_puro_soave(compuestos, presion, temperatura, raiz_volumen=3, raw=False):
    """
    Calcula el coeficiente de fugacidad de los compuestos como substancia pura usando la ecuacion de soave.
    Funcion de compuesto. Devuelve una matriz
    Args:
        compuestos (list): Lista de referencias a compuestos
        presion (List): Lista de presiones en bar
        temperatura (list): Lista de temperaturas en kelvin
        raiz_volumen (int): La ecuacion de peng robinson devuelve 3 volumenes. Aqui escoges con cual operar.
        raw=False (bool): Si se activa, todos los mecanismos de prevencion de errores se desactivan
    Returns
        arr_fugacidad (np.ndarray): Devuelve una matriz del coeficiente de fugacidad de cada compuesto a las diferentes
        condiciones
    """
    # Determinamos orientacion de la tabla a devolver
    orientacion_tabla = determinar_orietnacion_tabla(compuestos, presion)

    # Obtenemos propiedades termodinamicas de la base de datos
    pc, tc, w = obtener_vectores_propiedades(compuestos, ['pc','tc','w'])

    # Convertimos variables termodinamicas en vectores fila
    arr_presion, arr_temp = convertir_numpy_array(presion, temperatura, arr_columna=False)

    # Calculamos las fugacidades
    indice_volumen = int(raiz_volumen[0][0] - 1)
    arr_fugacidad = motor_de_calculo.coef_fugacidad_puro_soave(arr_presion, arr_temp, pc, tc, w, volumen_num=indice_volumen)

    # Devolvemos tabla de coeficientes de fuagicad
    if orientacion_tabla:
        return arr_fugacidad
    else:
        return arr_fugacidad.transpose()


def coeficiente_actividad_unifac(composiciones, temperatura, num_grupos, r_mayus, q_mayus, a_ij):
    """
    Calcula el coeficiente de actividad de cada compuesto en una mezcla usando el modelo UNIFAC.
    Es una funcion de compeusto, devuelve una matriz.
    Args:
        compuestos (list): lista de id, cas o nombre de los compuestos.
        composiciones (list): lista de las composiciones de los compeustos
        temperatura (list): lista de las temperaturas en Kelvin
        ---- aun no estan en la base de datos -----
        num_grupos (C, K): Tabla del numero de instancias del grupo funcional K en el compuesto C
        r_mayus (1, K): Constantes R unifac de cada grupo funcional
        q_mayus (1, K): Constantes Q unifac de cada grupo funcional
        a_ij (K, S): Matriz de interaccion energetica del grupo k al grupo s.
    Returns:
        coef_actividad (np.ndarray): Tabla del coeficiente de actividad de cada compuesto a cada 
        temperatura
    """
    # 1. Determinamos orientacion de la tabla a imprimir
    orientacion_tabla = determinar_orietnacion_tabla(composiciones, temperatura)

    # Convertimos constantes UNIFAC a np.ndarray
    r_mayus = np.array(r_mayus, dtype=np.float64)
    q_mayus = np.array(q_mayus, dtype=np.float64)
    v = np.array(num_grupos, dtype=np.float64)
    a_ij = np.array(a_ij, dtype=np.float64)

    # Convertimos composiciones a np.ndarray vector columna
    arr_composiciones = convertir_numpy_array(composiciones, arr_columna=True)

    # Convertimos temperaturas a np.ndarray vector fila
    arr_temp = convertir_numpy_array(temperatura, arr_columna=False)

    # Calculamos coeficientes de actividad
    coef_actividad = motor_de_calculo.coef_actividad_unifac(arr_temp, arr_composiciones, v, r_mayus, q_mayus, a_ij)

    # Devolvemos tabla de coeficientes de actividad
    if orientacion_tabla:
        return coef_actividad
    else:
        return coef_actividad.transpose()
    

def correccion_poynting(compuestos, presion, temperatura, raw=True):
    """
    Calcula el factor de correccion de poynting de cada compuesto a las diferentes condiciones
    Funcion de compuesto: Devuelve una matriz
    Args:
        compuestos (list): lista de id's, cas o nombre de los compuestos
        presion (list): lista de las presiones bar
        temperatura (list): lista de las temperaturas en kelvin
    Returns:
        poy (np.ndarray): Factor de correccion de poynting
    """
    # Determinamos la orientacion de la lista a imprimir
    orientacion_tabla = determinar_orietnacion_tabla(compuestos, presion)

    # Obtenemos propiedades termodinamias de la base de datos
    vol_liq, a_ant, b_ant, c_ant, tmin, pmin, tmax, pmax = obtener_vectores_propiedades(compuestos, ['vliq','anta','antb','antc','anttmin','antpvmin','anttmax','antpvmax'])

    # Convertimos variables termodinamicas a np.ndarray vector fila
    arr_presion, arr_temp = convertir_numpy_array(presion, temperatura, arr_columna=False)

    # Calculamos factor de poynting
    poy = motor_de_calculo.correccion_poynting(arr_presion, arr_temp, a_ant, b_ant, c_ant, vol_liq)

    # Si raw esta deshabilitado. Advertimos valores invalidos de la ecuacion de antoine
    if not raw:
        temp_invalidas = np.logical_or(arr_temp > tmax, arr_temp < tmin)
        presiones_invalidas = np.logical_or(arr_presion > pmax, arr_presion < pmin)
        poy = poy.astype(object)
        poy[temp_invalidas] = f'T fuera de rango'
        poy[presiones_invalidas] = f'P fuera de rango'

    # Devolvemos correccion de poynting
    if orientacion_tabla:
        return poy
    else:
        return poy.transpose()


def integral_cp_entre_t(compuestos, temperatura_1=None, temperatura_2=None, valor_int=None, raw=False):
    """
    Evalua la integral respecto a la temperatura del cp polinomial dividio entre la temperatura.
    Dependiendo de que variable no especifiques, esa devolverá.
    La ecuación que describe a esta funcion es la siguiente:
    valor_int = integral(Cp/T, temperatura_1, temperatura_2)
    Args:
        compuestos (list): Una lista que contiene strings de los id de loc compuestos involucrados
        temperatura_1 (list): Vector que contiene las temperaturas del limite inferior de la integral
        temperatura_2 (list): Vector que contiene las temperaturas del limite superior de la integral
        valor_int (list): Vector que contiene los valores de la integral
    Returns:
        resultado (np.ndarray): Array de numpy que contiene los valores ya sea de temperatura1, temperatura2 o valor_int que
        satisfagan la ecuacion. Devuelve la variable que no especifiques en la funcion
    """
    # Determinamos incognitas y orientacion de la tabla a imprimir al final.
    variables = {'t1', 't2', 'v'}
    variables_conocidas = set()
    if temperatura_1:
        variables_conocidas.add('t1')
        orientacion_tabla = determinar_orietnacion_tabla(compuestos, temperatura_1)
    if temperatura_2:
        variables_conocidas.add('t2')
        orientacion_tabla = determinar_orietnacion_tabla(compuestos, temperatura_2)
    if valor_int:
        variables_conocidas.add('v')
        orientacion_tabla = determinar_orietnacion_tabla(compuestos, valor_int)
    
    incognita = variables - variables_conocidas
        
    # Obtenemos propiedades termodinamicas de la base de datos
    tmin, tmax, a0, a1, a2, a3, a4 = obtener_vectores_propiedades(compuestos, ['cptmin', 'cptmax', 'cpa0', 'cpa1', 'cpa2', 'cpa3', 'cpa4'])

    # Convertimos variables termodinamicas en un np.ndarray fila
    arr_t1, arr_t2, arr_valor = convertir_numpy_array(temperatura_1, temperatura_2, valor_int, arr_columna=False)

    # Si la incognita es valor_int
    if incognita == {'v'}:
        valor_int = motor_de_calculo.integral_cp_entre_t_gas_ideal(arr_t1, arr_t2, a0, a1, a2, a3, a4)
        resultado = valor_int
        
    # Si la incognita es la temperatura 1
    if incognita == {'t1'}:
        # Usando una propiedad de las integrales de una variable, se pueden invertir los limites de integracion
        arr_t1 = motor_de_calculo.integral_cp_entre_t_gas_ideal_temperatura_2(arr_t2, -arr_valor, a0, a1, a2, a3, a4)
        resultado = arr_t1 

    # Si la incognita es la temperatura 2
    if incognita == {'t2'}:
        arr_t2 = motor_de_calculo.integral_cp_entre_t_gas_ideal_temperatura_2(arr_t1, arr_valor, a0, a1, a2, a3, a4)
        resultado = arr_t2

    # Si raw esta deshabilitado. Marcamos valores invalidos
    if not raw:
        temp1_fuera_rango = np.logical_or(arr_t1 < tmin, arr_t1 > tmax)
        temp2_fuera_rango = np.logical_or(arr_t2 < tmin, arr_t2 > tmax)
        resultado = resultado.astype(object)
        resultado[temp1_fuera_rango] = f'T fuera de rango'
        resultado[temp2_fuera_rango] = f'T fuera de rango'

    # Devolvemos matriz en funcion de la orientacion de las variables.
    if orientacion_tabla:
         return resultado
    else:
         return resultado.transpose()


