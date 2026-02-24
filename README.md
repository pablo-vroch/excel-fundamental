# Excel Fundamental 2.0 — Uso General

Excel Fundamental es un programa que integra modelos termodinámicos en Microsoft Excel.  
Está diseñado para evaluar estos modelos de forma vectorizada, permitiendo realizar cálculos eficientes directamente sobre tablas de Excel.

Incluye una base de datos con propiedades termodinámicas de 468 compuestos químicos.  
Los modelos pueden utilizarse directamente para los compuestos disponibles sin necesidad de ingresar manualmente sus propiedades.

---

# Instalación del programa

## 1. Instalar Python

Debes instalar Python en tu computadora. Puedes descargarlo desde el sitio oficial:

https://www.python.org/

### Detalles importantes

Durante la instalación de Python aparecerá la siguiente casilla:

![img1](docs/imagenes/img1.png)

Es importante que la actives, esto permite usar Python desde la terminal de Windows.

---

## 2. Instalar librerías de Python

Excel Fundamental realiza los cálculos numéricos mediante la librería NumPy y se conecta a Excel mediante la librería xlwings.

Estas librerías deben instalarse manualmente.

### Paso 1 — Abrir la terminal

1. Haz clic en la pestaña **Buscar** de Windows.

![img2](docs/imagenes/img2.png)

2. Escribe `Terminal`.

![img3](docs/imagenes/img3.png)

3. Haz clic en la aplicación.

---

### Paso 2 — Instalar NumPy

Escribe el siguiente comando y presiona Enter:

```bash
python -m pip install numpy
```

![img4](docs/imagenes/img4.png)

---

### Paso 3 — Instalar xlwings

Una vez termine la instalación anterior, escribe:

```bash
python -m pip install xlwings
```

![img5](docs/imagenes/img5.png)

---

## 3. Instalar Excel Fundamental

### Paso 1 — Descargar el repositorio

Ingresa al siguiente enlace:

https://github.com/pablo-vroch/excel-fundamental

Haz clic en el botón verde **Code**.

![img6](docs/imagenes/img6.png)

Luego selecciona **Download ZIP**.

![img7](docs/imagenes/img7.png)

---

### Paso 2 — Extraer los archivos

Haz clic derecho en el archivo:

`excel-fundamental-main.zip`

Selecciona la opción **Extraer todo**.

![img8](docs/imagenes/img8.png)

---

### Paso 3 — Abrir la aplicación

Ingresa a la carpeta:

`excel_fundamental_app`

![img9](docs/imagenes/img9.png)

Haz clic derecho en el archivo principal:

`excel_fundamental.xlsm`

Selecciona **Propiedades**.

![img10](docs/imagenes/img10.png)

Marca la casilla **Desbloquear** y haz clic en **Aplicar**.

![img11](docs/imagenes/img11.png)

Ahora puedes abrir el archivo normalmente.
