import xlwings as xw
import os

# Camiamos el directorio de trabajo, al directorio de este archivo
app_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(app_path)

# Importamos funciones de la interfaz del usuario
import interfaz_de_usuario

@xw.func
def tipo(value):
    return str(value)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('propiedades', ndim=2)
def obtener(compuestos, propiedades):
    return interfaz_de_usuario.obtener(compuestos, propiedades)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('propiedades', ndim=2)
@xw.arg('valores', ndim=2)
def modificar_datos(compuestos, propiedades, valores):
    return interfaz_de_usuario.modificar_datos(compuestos, propiedades, valores)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('composiciones', ndim=2)
@xw.arg('presion', ndim=2)
@xw.arg('temperatura', ndim=2)
@xw.arg('volumen', ndim=2)
def van_der_waals(compuestos, composiciones, presion, temperatura, volumen, raw=False):
    return interfaz_de_usuario.van_der_waals(compuestos, composiciones, presion=presion, temperatura=temperatura, volumen=volumen, raw=raw)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('composiciones', ndim=2)
@xw.arg('presion', ndim=2)
@xw.arg('temperatura', ndim=2)
@xw.arg('volumen', ndim=2)
def redlich_kwong(compuestos, composiciones, presion, temperatura, volumen, raw=False):
    return interfaz_de_usuario.redlich_kwong(compuestos, composiciones, presion=presion, temperatura=temperatura, volumen=volumen, raw=raw)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('composiciones', ndim=2)
@xw.arg('presion', ndim=2)
@xw.arg('temperatura', ndim=2)
@xw.arg('volumen', ndim=2)
def soave(compuestos, composiciones, presion, temperatura, volumen, raw=False):
    return interfaz_de_usuario.soave(compuestos, composiciones, presion=presion, temperatura=temperatura, volumen=volumen, raw=raw)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('composiciones', ndim=2)
@xw.arg('presion', ndim=2)
@xw.arg('temperatura', ndim=2)
@xw.arg('volumen', ndim=2)
def peng_robinson(compuestos, composiciones, presion, temperatura, volumen, raw=False):
    return interfaz_de_usuario.peng_robinson(compuestos, composiciones, presion=presion, temperatura=temperatura, volumen=volumen, raw=raw)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('presion', ndim=2)
@xw.arg('temperatura', ndim=2)
def antoine(compuestos, presion, temperatura, raw=False):
    return interfaz_de_usuario.antoine(compuestos, presion_vapor=presion, temperatura=temperatura, raw=raw)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('temperatura', ndim=2)
@xw.arg('capacidad_calorifica', ndim=2)
def cp(compuestos, temperatura, capacidad_calorifica, raw=False):
    return interfaz_de_usuario.cp(compuestos, temperatura=temperatura, capacidad_calorifica=capacidad_calorifica, raw=raw)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('temperatura1', ndim=2)
@xw.arg('temperatura2', ndim=2)
@xw.arg('deltah', ndim=2)
def integral_cp(compuestos, temperatura1, temperatura2, deltah, raw=False):
    return interfaz_de_usuario.integral_cp(compuestos, temperatura1=temperatura1, temperatura2=temperatura2, delta_h=deltah, raw=raw)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('composiciones', ndim=2)
@xw.arg('presion', ndim=2)
@xw.arg('temperatura', ndim=2)
@xw.arg('raiz_volumen', ndim=2)
def coef_fugacidad_mezcla_peng_robinson(compuestos, composiciones, presion, temperatura, raiz_volumen, raw=False):
    return interfaz_de_usuario.coeficiente_fugacidad_mezcla_peng_robinson(compuestos, composiciones, presion=presion, temperatura=temperatura, raiz_volumen=raiz_volumen, raw=raw)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('composiciones', ndim=2)
@xw.arg('presion', ndim=2)
@xw.arg('temperatura', ndim=2)
@xw.arg('raiz_volumen', ndim=2)
def coef_fugacidad_mezcla_soave(compuestos, composiciones, presion, temperatura, raiz_volumen, raw=False):
    return interfaz_de_usuario.coeficiente_fugacidad_mezcla_soave(compuestos, composiciones, presion=presion, temperatura=temperatura, raiz_volumen=raiz_volumen, raw=raw)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('presion', ndim=2)
@xw.arg('temperatura', ndim=2)
@xw.arg('raiz_volumen', ndim=2)
def coef_fugacidad_puro_peng_robinson(compuestos, presion, temperatura, raiz_volumen, raw=False):
    return interfaz_de_usuario.coeficiente_fugacidad_puro_peng_robinson(compuestos, presion, temperatura, raiz_volumen=raiz_volumen, raw=raw)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('presion', ndim=2)
@xw.arg('temperatura', ndim=2)
@xw.arg('raiz_volumen', ndim=2)
def coef_fugacidad_puro_soave(compuestos, presion, temperatura, raiz_volumen, raw=False):
    return interfaz_de_usuario.coeficiente_fugacidad_puro_soave(compuestos, presion, temperatura, raiz_volumen=raiz_volumen, raw=raw)

@xw.func
@xw.arg('composiciones', ndim=2)
@xw.arg('temperatura', ndim=2)
@xw.arg('grupos', ndim=2)
@xw.arg('r', ndim=1)
@xw.arg('q', ndim=1)
@xw.arg('aij', ndim=2)
def unifac(composiciones, temperatura, grupos, r, q, aij):
    return interfaz_de_usuario.coeficiente_actividad_unifac(composiciones, temperatura, grupos, r, q, aij)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('presion', ndim=2)
@xw.arg('temperatura', ndim=2)
def poynting(compuestos, presion, temperatura, raw=False):
    return interfaz_de_usuario.correccion_poynting(compuestos, presion, temperatura, raw=raw)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('composiciones_liq', ndim=2)
@xw.arg('composiciones_vap', ndim=2)
@xw.arg('presion', ndim=2)
@xw.arg('temperatura', ndim=2)
@xw.arg('calidad_vapor', ndim=2)
def entropia_ideal(compuestos, composiciones_liq, composiciones_vap, presion, temperatura, calidad_vapor):
    return interfaz_de_usuario.entropia_ideal(compuestos, composiciones_liq, composiciones_vap, presion, temperatura, calidad_vapor)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('composiciones_liq', ndim=2)
@xw.arg('composiciones_vap', ndim=2)
@xw.arg('temperatura', ndim=2)
@xw.arg('calidad_vapor', ndim=2)
def entalpia_ideal(compuestos, composiciones_liq, composiciones_vap, temperatura, calidad_vapor):
    return interfaz_de_usuario.entalpia_ideal(compuestos, composiciones_liq, composiciones_vap, temperatura, calidad_vapor)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('composiciones', ndim=2)
@xw.arg('temperatura', ndim=2)
def entalpia_ideal_vapor(compuestos, composiciones, temperatura):
    return interfaz_de_usuario.entalpia_ideal_vapor(compuestos, composiciones, temperatura)

@xw.func
@xw.arg('compuestos', ndim=2)
@xw.arg('temperatura_1', ndim=2)
@xw.arg('temperatura_2', ndim=2)
@xw.arg('valor_integral', ndim=2)
def integral_cp_entre_t(compuestos, temperatura_1, temperatura_2, valor_integral, raw=False):
    return interfaz_de_usuario.integral_cp_entre_t(compuestos, temperatura_1=temperatura_1, temperatura_2=temperatura_2, valor_int=valor_integral, raw=raw)