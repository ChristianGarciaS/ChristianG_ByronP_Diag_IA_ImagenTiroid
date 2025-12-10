# Manual de usuario

## Resumen de pasos

1. Instalacion de python 3.11 para correcto uso de dependencias
2. Ejecucion de ambiente virtual y uso de comandos para instalacion de dependencias
3. Cambio en archivo .py principal para especificacion de rutas
4. Ejecucion de comando para inicializacion de streamlit
5. Navegacion en la interfaz de usuario
6. Resultados esperados

## Instalacion de python 3.11
Se puede utilizar el siguiente enlace para descargar el instalador:
https://www.python.org/downloads/release/python-3119/?utm_source=chatgpt.com

Una vez descargado, ejecutar el .exe y asegurarse que las variables de entorno queden correctamente modificadas(Esto se puede realizar desde el mismo instalador seleccionando la casilla para configurar las variables de entorno automaticamente).

## Ejecucion de ambiente virtual y uso de comandos 
Despues de haber descargado e instalado python 3.11, se deben ejecutar los siguientes comandos en el orden especificado desde una linea de comandos.

**Nota**: El tercer comando es para la instalacion de las librerias necesarias para que el proyecto funcione correctamente, es necesario ejecutarlo en el ambiente virtual ya que este no contara con las librerias previamente instaladas en el sistema operativo.

* python -m venv venv
* venv\Scripts\activate
* pip install ipython tqdm seaborn matplotlib pandas numpy scikit-image opencv-python scikit-learn reportlab

## Cambio en archivo .py principal para especificacion de rutas
Se deben hacer modificaciones en el archivo **diagnostic_img_ia.py**, especificamente en las partes donde se definen las rutas en las que se encontraran o generaran archivos necesarios para el proyecto. Las lineas en las que se tendra que especificar la ruta local del computador que este usando el usuario son las siguientes(Se puede encontrar estas lineas con facilidad usando ctrl + F):

* path_benign = "(REEMPLAZAR POR RUTA HACIA CARPETA DE PROYECTO)/ProyFath/data/benign"

* path_malign = "(REEMPLAZAR POR RUTA HACIA CARPETA DE PROYECTO)/ProyFath/data/malignant"

* save_dir = "(REEMPLAZAR POR RUTA HACIA CARPETA DE PROYECTO)/ProyFath/modelos_entrenados_1"

* MODEL_DIR = "(REEMPLAZAR POR RUTA HACIA CARPETA DE PROYECTO)s/ProyFath/modelos_entrenados_1"

* out_pdf_path='(REEMPLAZAR POR RUTA HACIA CARPETA DE PROYECTO)/ProyFath/diagnostico_tiroides_informe_A4_landscape.pdf'

* DATASET_DIR = '(REEMPLAZAR POR RUTA HACIA CARPETA DE PROYECTO)/ProyFath/data'

**Ejemplos de como deben quedar las rutas:**

* path_benign = "C:/Users/pcgarcia/Downloads/ProyFath/data/benign"

* path_malign = "C:/Users/pcgarcia/Downloads/ProyFath/data/malignant"



## Ejecucion de comando para inicializacion de streamlit
Para inicializar la interfaz de usuario se debe ejecutar el siguiente comando:
* python -m streamlit run Login.py

En la consola deberia verse lo siguiente al ejecutar el comando

![](..../imagenes/ConsolaImg.png)

## Navegacion en la interfaz de usuario
Luego de ejecutar el comando de inicializacion de streamlit, se abrira una ventana en el navegador que redirigira al usuario a la pantalla de autenticacion.

![](../imagenes/ImgLogin.png)

El usuario debe ingresar como usuario y password el texto "admuser" para acceder al menu principal.

![](../imagenes/MenuPrincipalImg.jpg)

Una vez dentro, el usuario tendra 3 opciones: EDA, Entrenamiento y Prediccion por imagen

**EDA**: En esta pantalla se podran visualizar los resultados del EDA, estos incluyen graficos y varios procedimientos que se pueden evidenciar a traves de los logs en la misma pantalla

![](../imagenes/PantallaEDA.png)

**Entrenamiento**: En esta pantalla al pulsar el boton "Ejecutar Entrenamiento", se empezaran a entrenar los modelos que posteriormente seran guardados en el directorio "modelos_entrenados_1". Todo el proceso de entrenameinto se podra evidenciar en el apartado de logs en la pantalla cuando acabe el entrenamiento

![](../imagenes/PantallaEntrenamiento.png)

**Prediccion por imagen**: En esta pantalla se podra cargar una imagen y mediante la utilizacion de los modelos generados en la pantalla "Entrenamiento" se procedera a analizar la imagen, estos resultados se podran ver en los logs de la pantalla.

![](../imagenes/PantallaPrediccion.png)

**Importante:** En la parte superior derecha de la interfaz de usuario, al realizar una operacion, se podra observar un icono con imagenes cambiantes. Este icono representa que la operacion seleccionada se esta "ejecutando", por lo tanto es importante no relizar cambios de accion hasta que el icono desaparezca.
Una vez desaparezca los resultados se presentaran automaticamente en la pantalla, el icono se vera de la siguiente manera:

![](../imagenes/IconoDeCarga.png)


## Resultados esperados
Al final de la ejecucion de cada apartado, los resultados se deberan presentar en la misma pantalla de manera automatica, entre estos resultados se podra encontrar graficos, logs de ejecucion, imagenes generadas luego de analisis. Lo anteriormente mencionado se vera de la siguiente manera:

![](../imagenes/ResultadosEDA.png)

![](../imagenes/ResultadosEDA2.png)

![](../imagenes/ResultadosPrediccion.png)
