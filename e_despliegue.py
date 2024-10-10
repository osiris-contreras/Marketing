import numpy as np
import pandas as pd
import sqlite3 as sql
import a_funciones as fn ## para procesamiento
import openpyxl
import sys
import os

ruta =os.getcwd()

####Paquete para sistema basado en contenido ####
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors


def preprocesar():

    #### conectar_base_de_Datos#################
    conn=sql.connect(f'{ruta}\\data\\db_movies2')
    cur=conn.cursor()

    ######## convertir datos crudos a bases filtradas por usuarios que tengan cierto número de calificaciones
    fn.ejecutar_sql(f'{ruta}\\preprocesamientos.sql', cur)
   
    ##### llevar datos que cambian constantemente a python ######
    movies=pd.read_sql('select * from movies_final', conn )
    
    #### transformación de datos crudos - Preprocesamiento ################
    # Extraer el año de la columna 'title'
    movies['year'] = movies['title'].str.extract('\(([^)]*)\)$', expand=False)
    nulos_year = movies.loc[movies['year'].isna()]
    movies.loc[(movies['title'] == "Ready Player One") & (movies['year'].isna()), 'year'] = 2018
    movies['year']=movies.year.astype('int')

    # Normalizar la columna 'year'
    scaler = MinMaxScaler()
    movies[['year_normalized']] = scaler.fit_transform(movies[['year']])

    # Separar los géneros en columnas individuales
    genres = movies['genres'].str.split('|')
    te = TransactionEncoder()
    genres_matrix = te.fit_transform(genres)
    genres_df = pd.DataFrame(genres_matrix, columns=te.columns_)
    
    # Unir las nuevas columnas de géneros con el DataFrame original
    movies = pd.concat([movies, genres_df], axis=1)

    # Remover el año del título de la columna 'title'
    movies['title'] = movies['title'].str.replace(r'\s\(\d{4}\)$', '', regex=True)
    
    # Eliminar columnas innecesarias para los modelos
    final2 = movies.drop(columns=['movieId','genres','title','year'])

    return final2,movies, conn, cur

##########################################################################
###############Función para entrenar modelo por cada usuario ##########
###############Basado en contenido todo lo visto por el usuario Knn#############################
#user_id=604 ### para ejemplo manual
def recomendar(user_id):
    
    final2, movies, conn, cur= preprocesar()
    
    ratings=pd.read_sql('select *from ratings_final where user_id=:user',conn, params={'user':user_id})
    l_movies_r=ratings['movieId'].to_numpy()
    final2[['movieId','title']]=movies[['movieId','title']]
    movies_r=final2[final2['movieId'].isin(l_movies_r)]
    movies_r=movies_r.drop(columns=['movieId','title'])
    movies_r["indice"]=1 ### para usar group by y que quede en formato pandas tabla de centroide
    movies_r.info()
    centroide=movies_r.groupby("indice").mean()
    
    
    movies_nr=final2[~final2['movieId'].isin(l_movies_r)]
    movies_nr=movies_nr.drop(columns=['movieId','title'])
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(movies_nr)
    dist, idlist = model.kneighbors(centroide)
    
    ids=idlist[0]
    recomend_b=movies.loc[ids][['title','movieId']]
    
    
    return recomend_b


##### Generar recomendaciones para usuario lista de usuarios ####
##### No se hace para todos porque es muy pesado #############
def main(list_user):
    
    recomendaciones_todos=pd.DataFrame()
    for user_id in list_user:
            
        recomendaciones=recomendar(user_id)
        recomendaciones["user_id"]=user_id
        recomendaciones.reset_index(inplace=True,drop=True)
        
        recomendaciones_todos=pd.concat([recomendaciones_todos, recomendaciones])

    recomendaciones_todos.to_excel(f'{ruta}\\salidas\\recomendaciones.xlsx')
    recomendaciones_todos.to_csv(f'{ruta}\\salidas\\recomendaciones.csv')


if __name__=="__main__":
    list_user=[604,373,39,323 ]
    main(list_user)
    

import sys
sys.executable