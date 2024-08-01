# primero en consola: pip install -r requirements.txt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import PoissonRegressor
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson
from pykml import parser
import geopandas as gpd
import shapely.speedups
shapely.speedups.enable()
import fiona
fiona.drvsupport.supported_drivers['KML'] = 'rw'
import os
import re
import streamlit as st

st.set_page_config(layout="wide")

def main():

    st.markdown('<div align="right" style="font-size:0.65em;"> Versión 1 - julio 2024 </div>', unsafe_allow_html=True)

    st.markdown('<style>description{color:blue;}</style>', unsafe_allow_html=True)
    st.title('Predicción de inscriptos en planes de MEVIR')

    st.write('Utilizando modelos de Machine Learning (ML) se realiza la predicción de la **cantidad de familias inscriptas** a un llamado.')

    ubic = '../'

    st.write('## 1. Datos de entrada:')
    datos_pipl = pd.read_csv(ubic + str('Datos/listado_PIPL_TIPO_NUM_TIPO.csv'), sep = ';', encoding = 'cp1252')
    datos_pipl.PIPL = pd.Categorical(datos_pipl.PIPL)
    datos_pipl.TIPO = pd.Categorical(datos_pipl.TIPO)
    datos_pipl.NUM_TIPO = pd.to_numeric(datos_pipl.NUM_TIPO, downcast = 'integer')
    datos_pipl.FECHA_INSCRIPCION = pd.to_datetime(datos_pipl.FECHA_INSCRIPCION, format = '%Y/%m/%d').dt.date

    # datos de entrada
    pipl = st.selectbox("Plan Integral:", np.unique(np.array(datos_pipl.PIPL)), index = None)
    agregar_pipl = st.checkbox("Agregar nuevo Plan Integral")
    if agregar_pipl:
       pipl = st.text_input("Ingresar nombre del nuevo Plan Integral:")
       agregar_fecha = True
    else:
        fecha = st.selectbox("Fecha de inscripcion:", np.unique(np.array(datos_pipl.FECHA_INSCRIPCION[datos_pipl.PIPL == pipl])), index = None)
        agregar_fecha = st.checkbox("Considerar otra fecha de inscripción")
    if agregar_fecha:
        fecha = st.date_input("Ingresar fecha de inscripción:", format = 'YYYY-MM-DD', min_value = pd.to_datetime('01/01/2015', format = '%d/%m/%Y'))
        if not agregar_pipl and fecha <= max(datos_pipl.FECHA_INSCRIPCION[datos_pipl.PIPL == pipl]):
            st.warning(str('La fecha elegida debe ser posterior al último llamado del PIPL: ') + str(max(datos_pipl.FECHA_INSCRIPCION[datos_pipl.PIPL == pipl])))
    if (agregar_pipl) & (agregar_fecha):
        tipo = st.multiselect("Tipo de llamado:", ['Vivienda Nucleada', 'Planta Urbana', 'Área Rural'])
        base = pd.DataFrame({'PIPL': pipl, 'NUM': 1, 'TIPO': tipo, 'NUM_TIPO': 1, 'FECHA_INSCRIPCION': fecha})
    if (not agregar_pipl) & (agregar_fecha):
        tipo = st.multiselect("Tipo de llamado:", ['Vivienda Nucleada', 'Planta Urbana', 'Área Rural'])
        base = pd.DataFrame({'PIPL': pipl, 'NUM': max(datos_pipl.NUM[datos_pipl.PIPL == pipl]) + 1, 'TIPO': tipo, 'NUM_TIPO': 1, 'FECHA_INSCRIPCION': fecha})
        if ('Vivienda Nucleada' in np.array(datos_pipl.TIPO[datos_pipl.PIPL == pipl])) and ('Vivienda Nucleada' in np.array(base.TIPO)):
            base.NUM_TIPO[base.TIPO == 'Vivienda Nucleada'] = max(datos_pipl.NUM[(datos_pipl.PIPL == pipl) & (datos_pipl.TIPO == 'Vivienda Nucleada')]) + 1
        if ('Planta Urbana' in np.array(datos_pipl.TIPO[datos_pipl.PIPL == pipl])) and ('Planta Urbana' in np.array(base.TIPO)):
            base.NUM_TIPO[base.TIPO == 'Planta Urbana'] = max(datos_pipl.NUM[(datos_pipl.PIPL == pipl) & (datos_pipl.TIPO == 'Planta Urbana')]) + 1
        if ('Área Rural' in np.array(datos_pipl.TIPO[datos_pipl.PIPL == pipl])) and ('Área Rural' in np.array(base.TIPO)):
            base.NUM_TIPO[base.TIPO == 'Área Rural'] = max(datos_pipl.NUM[(datos_pipl.PIPL == pipl) & (datos_pipl.TIPO == 'Área Rural')]) + 1                                                                                    
    if (not agregar_pipl) & (not agregar_fecha):
        tipo = st.multiselect("Tipo de llamado:", np.array(datos_pipl.TIPO[(datos_pipl.PIPL == pipl) & (datos_pipl.FECHA_INSCRIPCION == fecha)]))
        base = datos_pipl[(datos_pipl.PIPL == pipl) & (datos_pipl.FECHA_INSCRIPCION == fecha) & (datos_pipl.TIPO.isin(tipo))]

    # se muestran datos de entrada
    if (pipl is not None) & (fecha is not None) & (len(tipo) > 0): 
        st.dataframe(base, hide_index = True, use_container_width = True)

    st.write('## 2. Área del llamado:')

    # se carga archivo kml
    kml_file = st.file_uploader('Cargar polígono del llamado en formato KML:', type = 'kml')
        
    # si hay archivo kml cargado, se muestra el gráfico
    if kml_file is not None:
        kml = gpd.read_file(kml_file, driver = 'KML')
        kml.to_crs(32721, inplace = True)
        fig, ax = plt.subplots(1, 1, figsize = (2, 2))
        kml.plot(ax = ax, facecolor='gray')
        ax.set_axis_off()
        st.pyplot(fig, use_container_width = False)
    
    st.write('## 3. Resultados de la predicción:')

    with st.spinner('Aguarde mientras se ejecuta la predicción...'):
    
        # si se cargaron todos los datos de entrada, se ejecuta la predicción
        if (pipl is not None) & (fecha is not None) & (len(tipo) > 0) & (kml_file is not None):
            
            # ÁREA DEL LLAMADO
            area = kml.area/1000000
            base['AREA'] = np.array(area.repeat(len(base)))

            # DATOS CENSALES

            # localidades urbanas
            zonas_urb = gpd.read_file(ubic + str('/Datos/Localidades_Urbanas_INE_2011/ine_loc11_pg.shp'))
            zonas_urb = zonas_urb.set_crs(32721)

            # zonas rurales
            zonas_rur = gpd.read_file(ubic + str('/Datos/Zonas_Rurales_INE_2011/00_-_URUGUAY_-_ZONAS_RURALES_INE_2011.shp'))
            zonas_rur = zonas_rur.set_crs(32721)
            
            # cargo marcos censales para traer la cantidad de hogares por zona
            marco_rur = pd.read_csv(ubic + str('/Datos/marco_rural.csv'), encoding = 'cp1252')
            marco_urb = pd.read_csv(ubic + str('/Datos/marco_urbano.csv'), encoding = 'cp1252')

            # solamente zonas y localidades que intersectan con el kml
            zonas_rur_kml = zonas_rur.overlay(kml, how = 'intersection')
            zonas_urb_kml = zonas_urb.overlay(kml, how = 'intersection')

            # si hay zonas rurales dentro del llamado, se calculan variables correspondientes
            if len(zonas_rur_kml) > 0:

                # zonas que intersectan con kml, pero completas (sin recortar)
                zonas_rur_kml_comp = zonas_rur[zonas_rur.CODCOMP.isin(zonas_rur_kml.CODCOMP)]

                # area completa de cada zona rural, area que intersecta (nueva) y prop del área que intersecta
                zonas_rur_kml['AREA_NUEVA'] = zonas_rur_kml.area/1000000
                zonas_rur_kml_comp['AREA_COMP'] = zonas_rur_kml_comp.area/1000000
                zonas_rur_kml = zonas_rur_kml.merge(zonas_rur_kml_comp[['CODCOMP', 'AREA_COMP']], on = 'CODCOMP')
                zonas_rur_kml['PROP_AREA'] = zonas_rur_kml.apply(lambda row: row.AREA_NUEVA/row.AREA_COMP, axis = 1)

                # agrego H_PAR a zonas rurales y calculo % que corresponde considerar según prop area
                zonas_rur_kml = zonas_rur_kml.merge(marco_rur[['CODCOMP', 'H_TOT', 'H_PAR']], on = 'CODCOMP')
                zonas_rur_kml['H_TOT_VA'] = zonas_rur_kml.apply(lambda row: np.around(row.H_TOT*row.PROP_AREA), axis = 1)
                zonas_rur_kml['H_PAR_VA'] = zonas_rur_kml.apply(lambda row: np.around(row.H_PAR*row.PROP_AREA), axis = 1)

                # agrego datos a la base
                base['HOG_RUR'] = np.sum(zonas_rur_kml['H_TOT_VA']).repeat(len(base))
            else:
                base['HOG_RUR'] = 0

            # si hay zonas urbanas dentro del llamado, se calculan variables correspondientes
            if len(zonas_urb_kml) > 0:

                # zonas que intersectan con kml, pero completas (sin recortar)
                zonas_urb_kml_comp = zonas_urb[zonas_urb.CODLOC.isin(zonas_urb_kml.CODLOC)]

                # area completa de cada zona urbana, area que intersecta (nueva) y prop del área que intersecta
                zonas_urb_kml['AREA_NUEVA'] = zonas_urb_kml.area/1000000
                zonas_urb_kml_comp['AREA_COMP'] = zonas_urb_kml_comp.area/1000000
                zonas_urb_kml = zonas_urb_kml.merge(zonas_urb_kml_comp[['CODLOC', 'AREA_COMP']], on = 'CODLOC')
                zonas_urb_kml['PROP_AREA'] = zonas_urb_kml.apply(lambda row: row.AREA_NUEVA/row.AREA_COMP, axis = 1)

                # agrego H_PAR a localidades urbanas y calculo % que corresponde considerar según prop area
                zonas_urb_kml = zonas_urb_kml.merge(marco_urb[['CODLOC', 'H_TOT', 'H_PAR']], on = 'CODLOC')
                zonas_urb_kml['H_TOT_VA'] = zonas_urb_kml.apply(lambda row: np.around(row.H_TOT*row.PROP_AREA), axis = 1)
                zonas_urb_kml['H_PAR_VA'] = zonas_urb_kml.apply(lambda row: np.around(row.H_PAR*row.PROP_AREA), axis = 1)

                # agrego datos a la base
                base['HOG_URB'] = np.sum(zonas_urb_kml['H_TOT_VA']).repeat(len(base))
            else:
                base['HOG_URB'] = 0

            # agrego datos a la base
            base['HOG_TOT'] = base.apply(lambda row: row.HOG_RUR + row.HOG_URB, axis = 1)
            base['HOG_TOT_PAR'] = np.sum(zonas_rur_kml['H_PAR_VA']).repeat(len(base)) + np.sum(zonas_urb_kml['H_PAR_VA']).repeat(len(base))
            base['HOG_TOT_TIPO'] = base.apply(lambda row: row.HOG_TOT if row.TIPO == 'Vivienda Nucleada' else (row.HOG_URB if row.TIPO == 'Planta Urbana' else row.HOG_RUR), axis=1)
            base.drop(columns = ['HOG_RUR', 'HOG_URB', 'HOG_TOT'], inplace = True)

            # DATOS DE INTERVENCIÓN DE MEVIR

            # shape intervención histórica de MEVIR
            int_mevir = gpd.read_file(ubic + str('/Datos/Intervención_histórica_MEVIR/2023.12.31_-_URUGUAY_-_INTERVENCION_HISTORICA_DE_MEVIR_-_CON_DATOS_-_para_difusion.shp'))

            # intersección kml e intervenciones MEVIR
            res_int = int_mevir.overlay(kml, how = 'intersection')
            if 'COD_MEV' in res_int.columns:
                res_int.rename(columns = {"COD_MEV": "C_LOC_MEV"}, inplace = True)

            # si hay antecedentes de MEVIR, se calculan las variables correspondientes
            if len(res_int) > 0:

                # datos base de programas entregados
                bpe = pd.read_csv(ubic + str('/Datos/BPE al 31-03-2024.csv'), sep = ';', encoding = 'cp1252')
                bpe['TOT'] = bpe.apply(lambda row: row.N + row.TP + row.UP + row.AR + row.MP, axis = 1)
                bpe = bpe[bpe.TOT > 0]
                bpe = bpe[bpe.C_LOC_MEV.isin(res_int.C_LOC_MEV)]
                bpe = bpe[~bpe['PROGRAMA II'].isin(['CONVENIO', 'INC'])]
                bpe.ENTREGADA = pd.to_datetime(bpe.ENTREGADA, format = '%d/%m/%Y')
                bpe = bpe.loc[bpe.ENTREGADA < pd.unique(base.FECHA_INSCRIPCION).repeat(len(bpe))]
                if len(bpe) > 0:
                    bpe['N_TP'] = bpe.apply(lambda row: row.N + row.TP, axis = 1)
                    bpe['UP_AR'] = bpe.apply(lambda row: row.UP + row.AR + row.MP, axis = 1)
                    bpe['NOMBRE DEL PROGRAMA'] = bpe['NOMBRE DEL PROGRAMA'].str.upper()

                    # agrego datos a la base
                    base['Interv_N_TP'] = np.sum(bpe.N_TP).repeat(len(base))
                    base['Interv_UP_AR'] = np.sum(bpe.UP_AR).repeat(len(base))
                    base['Interv_PU'] = np.sum(bpe.PU).repeat(len(base))
                    base['fechaUltInt_N'] = np.array(bpe.ENTREGADA[bpe.N_TP > 0].min()).repeat(len(base))
                    base['fechaUltInt_AR'] = np.array(bpe.ENTREGADA[bpe.UP_AR > 0].min()).repeat(len(base))
                    base['fechaUltInt_PU'] = np.array(bpe.ENTREGADA[bpe.PU > 0].min()).repeat(len(base))                    
                else:
                    base['Interv_N_TP'] = np.zeros(len(base))
                    base['Interv_UP_AR'] = np.zeros(len(base))
                    base['Interv_PU'] = np.zeros(len(base))
                    base['fechaUltInt_N'] = np.array(pd.to_datetime('01/01/0001', format = '%d/%m/%Y')).repeat(len(base))
                    base['fechaUltInt_AR'] = np.array(pd.to_datetime('01/01/0001', format = '%d/%m/%Y')).repeat(len(base))
                    base['fechaUltInt_PU'] = np.array(pd.to_datetime('01/01/0001', format = '%d/%m/%Y')).repeat(len(base)) 

                base['NUM_INT_TOT'] = base.apply(lambda row: row.Interv_N_TP + row.Interv_UP_AR + row.Interv_PU, axis = 1)
                base['NUM_INT_TIPO'] = base.apply(lambda row: row.Interv_N_TP if row.TIPO == 'Vivienda Nucleada' else (row.Interv_PU if row.TIPO == 'Planta Urbana' else row.Interv_UP_AR), axis=1)
                base['FECHA_ULT_INT_TIPO'] = pd.to_datetime(base.apply(lambda row: row.fechaUltInt_N if row.TIPO == 'Vivienda Nucleada' else (row.fechaUltInt_PU if row.TIPO == 'Planta Urbana' else row.fechaUltInt_AR), axis=1), format = '%Y-%m-%d').dt.date
                base['DIAS_ULT_INT_TIPO'] = (base['FECHA_INSCRIPCION'] - base['FECHA_ULT_INT_TIPO']).dt.days
                base['AÑOS_ULT_INT_TIPO_AGRUP'] = base.apply(lambda row: 'hasta_3' if row.DIAS_ULT_INT_TIPO/365 <= 3 else 'mas_de_3', axis = 1)
                base.drop(columns = ['Interv_N_TP',	'Interv_UP_AR', 'Interv_PU', 'fechaUltInt_N', 'fechaUltInt_AR', 'fechaUltInt_PU', 'FECHA_ULT_INT_TIPO',	'DIAS_ULT_INT_TIPO'], inplace = True)
                base.AÑOS_ULT_INT_TIPO_AGRUP = pd.Categorical(base.AÑOS_ULT_INT_TIPO_AGRUP)
                base.NUM_INT_TOT = base.NUM_INT_TOT.astype(float)
                base.NUM_INT_TIPO = base.NUM_INT_TIPO.astype(float)
            else:
                base['AÑOS_ULT_INT_TIPO_AGRUP'] = pd.Categorical(np.array('mas_de_3').repeat(len(base)))
                base['NUM_INT_TOT'] = np.zeros(len(base)).astype(float)
                base['NUM_INT_TIPO'] = np.zeros(len(base)).astype(float)
            
            base.AREA = pd.to_numeric(np.around(base.AREA.astype(float)), downcast = 'integer')

            # BASES PARA PREDICCION

            # matriz para KNN
            X_KNN = base.drop(columns = ['PIPL', 'FECHA_INSCRIPCION', 'AÑOS_ULT_INT_TIPO_AGRUP', 'HOG_TOT_TIPO'], axis = 1).copy()
            # matriz para Poisson
            X_POIS = base.drop(columns = ['PIPL', 'FECHA_INSCRIPCION', 'HOG_TOT_PAR'], axis = 1).copy()
            # modleo NB no tiene pipeline, hay que hacer primero el preprocesamiento y después aplicar modelo
            X_NB = base.drop(columns = ['PIPL', 'FECHA_INSCRIPCION', 'HOG_TOT_PAR'], axis = 1).copy()

            # formateo X_NB para que quede igual que el requerido para el modelo (único que no tiene pipeline incorporado)

            # escalo variables numéricas igual que en el modelo NB (tomo rangos de variables de base train)
            X_NB.NUM_TIPO = X_NB.apply(lambda row: min(1, max(0, (row.NUM_TIPO - 1)/(5-1))), axis = 1) # rango 1 a 5
            X_NB.AREA = X_NB.apply(lambda row: min(1, max(0, (row.AREA - 0.04)/(3051-0.04))), axis = 1) # rango 0.04 a 3051
            X_NB.NUM_INT_TOT = X_NB.apply(lambda row: min(1, max(0, (row.NUM_INT_TOT - 0)/(619-0))), axis = 1) # rango 0 a 619
            X_NB.NUM_INT_TIPO = X_NB.apply(lambda row: min(1, max(0, (row.NUM_INT_TIPO - 0)/(458-0))), axis = 1) # rango 0 a 458
            X_NB.HOG_TOT_TIPO = X_NB.apply(lambda row: min(1, max(0, (row.HOG_TOT_TIPO - 18)/(3386-18))), axis = 1) # rango 18 a 3386
            X_NB.rename(columns = {'NUM_TIPO': 'num__NUM_TIPO', 'AREA': 'num__AREA', 'NUM_INT_TOT': 'num__NUM_INT_TOT', 'NUM_INT_TIPO': 'num__NUM_INT_TIPO', 'HOG_TOT_TIPO': 'num__HOG_TOT_TIPO'}, inplace = True)

            # recategorizo en 1 y 0 variables cualitativas
            X_NB['cat__AÑOS_ULT_INT_TIPO_AGRUP_mas_de_3'] = X_NB.apply(lambda row: 1 if row.AÑOS_ULT_INT_TIPO_AGRUP == 'mas_de_3' else 0, axis = 1)
            X_NB['cat__TIPO_Vivienda Nucleada'] = X_NB.apply(lambda row: 1 if row.TIPO == 'Vivienda Nucleada' else 0, axis = 1)
            X_NB['cat__TIPO_Área Rural'] = X_NB.apply(lambda row: 1 if row.TIPO == 'Área Rural' else 0, axis = 1)
            X_NB.drop(columns = ['TIPO', 'NUM', 'AÑOS_ULT_INT_TIPO_AGRUP'], inplace = True)

            # agrego constante
            X_NB['const'] = np.ones(len(X_NB))

            # convierto clase de variables
            col = X_NB.columns[X_NB.columns.to_series().str.contains('^cat__')]
            for i in col:
                X_NB[i] = pd.Categorical(X_NB[i])
            col = X_NB.columns[X_NB.columns.to_series().str.contains('^num__')]
            for i in col:
                X_NB[i] = X_NB[i].astype(float)

            # MODELOS DE MACHINE LEARNING A UTILIZAR

            # KNN
            KNN_model = pickle.load(open(ubic + str('Modelos/best_model_KNN.obj'), 'rb'))
            # Poisson
            POIS_model = pickle.load(open(ubic + str('Modelos/best_model_Poisson.obj'), 'rb'))
            # Negative Binomial
            NB_model = pickle.load(open(ubic + str('Modelos/best_model_NegativeBinomial.obj'), 'rb'))

            # PREDICCIONES

            y_KNN = np.around(KNN_model.predict(X_KNN))
            y_POIS = np.around(POIS_model.predict(X_POIS))
            y_NB = np.around(NB_model.predict(X_NB))

            # agrego predicciones
            base['KNN_pred'] = y_KNN
            base['POIS_pred'] = y_POIS
            base['NB_pred'] = y_NB
            base['PROM_pred'] = base.apply(lambda row: np.around((row.KNN_pred + row.POIS_pred + row.NB_pred)/3), axis = 1)

            # edito variables a mostrar
            base.HOG_TOT_PAR = pd.to_numeric(base.HOG_TOT_PAR, downcast = 'integer')
            base.HOG_TOT_TIPO = pd.to_numeric(base.HOG_TOT_TIPO, downcast = 'integer')
            base.NUM_INT_TOT = pd.to_numeric(base.NUM_INT_TOT, downcast = 'integer')
            base.NUM_INT_TIPO = pd.to_numeric(base.NUM_INT_TIPO, downcast = 'integer')
            base.KNN_pred = pd.to_numeric(base.KNN_pred, downcast = 'integer')
            base.POIS_pred = pd.to_numeric(base.POIS_pred, downcast = 'integer')
            base.NB_pred = pd.to_numeric(base.NB_pred, downcast = 'integer')
            base.PROM_pred = pd.to_numeric(base.PROM_pred, downcast = 'integer')

            # RESULTADOS A MOSTRAR EN APP
            base.rename(columns = {'AÑOS_ULT_INT_TIPO_AGRUP': 'AÑOS_ULT_INT_TIPO', 'HOG_TOT_PAR': 'HOG_PAR', 'HOG_TOT_TIPO': 'HOG_TIPO'}, inplace = True)
            st.dataframe(base.drop(columns = ['PIPL', 'NUM', 'NUM_TIPO', 'FECHA_INSCRIPCION']).style.applymap(lambda _:"background-color: palegreen;", subset = ['KNN_pred', 'POIS_pred', 'NB_pred']).applymap(lambda _:"background-color: darkseagreen;", subset = ['PROM_pred']).format(), hide_index = True)

            @st.cache_data
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode("utf-8")
            base = convert_df(base)

            st.download_button(
                label = "Descargar datos",
                data = base,
                file_name = "base.csv",
                mime = "text/csv",
            )

    if (pipl is not None) & (fecha is not None) & (len(tipo) > 0) & (kml_file is None): 
        st.write('## 4. Glosario:')
        st.markdown('''
        * **PIPL:** Nombre del Plan Integral.
        * **NUM:** Número de llamado dentro del Plan Integral.
        * **TIPO:** Tipo de llamado. Tipos posibles son Vivienda Nucleada, Planta Urbana y Área Rural.
        * **NUM_TIPO:** Número de llamado dentro del Plan Integral y para ese tipo de llamado.
        * **FECHA_INSCRIPCION:** Fecha de inscripción para la cual se ejecuta la predicción de familias inscriptas.
        ''')

    if (pipl is not None) & (fecha is not None) & (len(tipo) > 0) & (kml_file is not None): 
        st.write('## 4. Glosario:')
        st.markdown('''
        * **PIPL:** Nombre del Plan Integral.
        * **NUM:** Número de llamado dentro del Plan Integral.
        * **TIPO:** Tipo de llamado. Tipos posibles son Vivienda Nucleada, Planta Urbana y Área Rural.
        * **NUM_TIPO:** Número de llamado dentro del Plan Integral y para ese tipo de llamado.
        * **FECHA_INSCRIPCION:** Fecha de inscripción para la cual se ejecuta la predicción de familias inscriptas.
        * **AREA:** Área del llamado en kilómetros cuadrados.
        * **HOG_PAR:** Cantidad de hogares particulares (rurales y urbanos) dentro de los límites del llamado (Censo 2011).
        * **HOG_TIPO:** Cantidad de hogares totales dentro de los límites del llamado según tipo: hogares totales (rurales y urbanos) para Vivienda Nucleada, hogares urbanos para Planta Urbana y hogares rurales para Área Rural.
        * **NUM_INT_TOT:** Cantidad de intervenciones históricas de MEVIR dentro de los límites del llamado.
        * **NUM_INT_TIPO:** Cantidad de intervenciones históricas de MEVIR (del mismo tipo del llamado) dentro de los límites del llamado.
        * **AÑOS_ULT_TIPO_AGRUP:** Años desde la última intervención realizada dentro de los límites del llamado, se agrupa en "hasta 3" y "más de 3".
        * **KNN_pred:** Predicción con el modelo KNN ("K" vecinos más cercanos).
        * **POIS_pred:** Predicción con el modelo Poisson.
        * **NB_pred:** Predicción con el modelo Binomial Negativo.
        * **PROM_pred:** Promedio de las tres predicciones anteriores.
        ''')

if __name__ == "__main__":
    main()

# desde consola: streamlit run app.py
