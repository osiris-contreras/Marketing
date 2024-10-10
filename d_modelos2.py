####################################################################
##            APLICACIONES DE LA ANALÍTICA EN MARKETING           ##
##                              GRUPO 4                           ##
##                     OSIRIS CONTRERAS TRILLOS                   ##
##                         JUAN FELIPE OSORIO                     ##
##                          JUAN JOSE MOLINA                      ##
####################################################################


import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact ## para análisis interactivo
from sklearn import neighbors ### basado en contenido un solo producto consumido
import joblib
####Paquete para sistemas de recomendación surprise
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise.model_selection import train_test_split
from surprise import SVD, SVDpp, NMF, CoClustering

#############################################################
####                Conectar base de Datos               ####
#############################################################

conn=sql.connect('data\\db_movies2')
cur=conn.cursor()

#############################################################
#### 3 Sistema de recomendación basado en contenido KNN  ####
####     Con base en todo lo visto por el usuario        ####
#############################################################



##### cargar data frame escalado y con dummies ###
final2 = joblib.load('data\\fianl2.joblib')

### carga data frame normal que tiene nombres de las peliculas
movies =pd.read_sql('select * from full_ratings', conn )

#### seleccionar usuario para recomendaciones ####
usuarios=pd.read_sql('select distinct (user_id) as user_id from ratings_final',conn)
user_id=609 ### para ejemplo manual

def recomendar(user_id=list(usuarios['user_id'].value_counts().index)):
    
    ###seleccionar solo los ratings del usuario seleccionado
    ratings=pd.read_sql('select *from ratings_final where user_id=:user',conn, params={'user':user_id,})
    
    ###convertir ratings del usuario a array
    l_movies_v=ratings['rating'].to_numpy()
    
    ###agregar la columna de rating y titulo de la pelicula a dummie para filtrar y mostrar nombre
    final2[['rating','title']]=movies[['rating','title']]
    
    ### filtrar libros calificados por el usuario
    movies_v=final2[final2['rating'].isin(l_movies_v)]
    
    ## eliminar columna nombre e rating
    movies_v=movies_v.drop(columns=['rating','title'])
    movies_v["indice"]=1 ### para usar group by y que quede en formato pandas tabla de centroide
    ##centroide o perfil del usuario
    centroide=movies_v.groupby("indice").mean()
    
    
    ### filtrar peliculas no vistas
    movies_nv=final2[~final2['rating'].isin(l_movies_v)]
    ## eliminbar nombre e rating
    movies_nv=movies_nv.drop(columns=['rating','title'])
    
    ### entrenar modelo 
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(movies_nv)
    dist, idlist = model.kneighbors(centroide)
    
    ids=idlist[0] ### queda en un array anidado, para sacarlo
    recomend_b=movies.loc[ids][['title','rating']]
    vistas=movies[movies['rating'].isin(l_movies_v)][['title','rating']]
    
    return recomend_b

recomendar(609)

print(interact(recomendar))


############################################################
##### 4.Sistema de recomendación filtro colaborativo   #####
############################################################

### datos originales en pandas
## knn solo sirve para calificaciones explicitas
ratings = pd.read_sql('select * from ratings_final where rating>0', conn)

print(ratings['rating'].unique())

####los datos deben ser leidos en un formato espacial para surprise
reader = Reader(rating_scale=(1, 5)) ### la escala de la calificación
###las columnas deben estar en orden estándar: user item rating
data   = Dataset.load_from_df(ratings[['user_id','movieId','rating']], reader)


#####Existen varios modelos 
models=[KNNBasic(),KNNWithMeans(),KNNWithZScore(),KNNBaseline()] 
results = {}

###knnBasiscs: calcula el rating ponderando por distancia con usuario/Items
###KnnWith means: en la ponderación se resta la media del rating, y al final se suma la media general
####KnnwithZscores: estandariza el rating restando media y dividiendo por desviación 
####Knnbaseline: calculan el desvío de cada calificación con respecto al promedio y con base en esos calculan la ponderación


#### for para probar varios modelos ##########
model=models[1]
for model in models:
 
    CV_scores = cross_validate(model, data, measures=["MAE","RMSE"], cv=5, n_jobs=-1)  
    
    result = pd.DataFrame.from_dict(CV_scores).mean(axis=0).\
             rename({'test_mae':'MAE', 'test_rmse': 'RMSE'})
    results[str(model).split("algorithms.")[1].split("object ")[0]] = result


performance_df = pd.DataFrame.from_dict(results).T
performance_df.sort_values(by='RMSE')

###################se escoge el mejor knn withmeans#########################

## min support es la cantidad de items o usuarios que necesita para calcular recomendación
## name medidas de distancia

### se afina si es basado en usuario o basado en ítem

###################
################### Ampliando los hiperparámetros del modelo KNNWithMeans #########################

param_grid = { 
    'sim_options' : {
        'name': ['msd', 'cosine', 'pearson'],  # Agregamos 'pearson' como otra opción
        'min_support': [1, 2, 5, 10],  # Exploramos un rango más amplio de min_support
        'user_based': [False, True]  # Continuamos explorando basado en usuarios o ítems
    },
    'k': [10, 20, 40]  # Agregamos el parámetro 'k' para cambiar la cantidad de vecinos
}

## Ejecutamos GridSearchCV para buscar los mejores parámetros
gridsearchKNNWithMeans = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse'], cv=3, n_jobs=-1)

# Entrenamos el GridSearch
gridsearchKNNWithMeans.fit(data)

# Mostramos los mejores parámetros encontrados y el mejor score de RMSE
print("Mejores parámetros:", gridsearchKNNWithMeans.best_params["rmse"])
print("Mejor score de RMSE:", gridsearchKNNWithMeans.best_score["rmse"])

# Guardamos el mejor modelo encontrado
gs_model = gridsearchKNNWithMeans.best_estimator['rmse']  # Mejor estimador de GridSearch


################# Entrenar con todos los datos y Realizar predicciones con el modelo afinado
###################
trainset = data.build_full_trainset() ### esta función convierte todos los datos en entrnamiento, las funciones anteriores dividen  en entrenamiento y evaluación
model=gs_model.fit(trainset) ## se reentrena sobre todos los datos posibles (sin dividir)


predset = trainset.build_anti_testset() ### crea una tabla con todos los usuarios y los libros que no han leido
#### en la columna de rating pone el promedio de todos los rating, en caso de que no pueda calcularlo para un item-usuario
len(predset)


predictions = gs_model.test(predset) ### función muy pesada, hace las predicciones de rating para todos los libros que no hay leido un usuario
### la funcion test recibe un test set constriuido con build_test method, o el que genera crosvalidate
predictions[0:10] 
####### la predicción se puede hacer para una pelicula puntual
model.predict(uid=5, iid='47',r_ui='') ### uid debía estar en número e isb en comillas

predictions_df = pd.DataFrame(predictions) ### esta tabla se puede llevar a una base donde estarán todas las predicciones
predictions_df.shape
predictions_df.head()
predictions_df['r_ui'].unique() ### promedio de ratings
predictions_df.sort_values(by='est',ascending=False)


##### funcion para recomendar las 10 peliculas con mejores predicciones y llevar base de datos para consultar resto de información
def recomendaciones(user_id,n_recomend=10):
    
    predictions_userID = predictions_df[predictions_df['uid'] == user_id].\
                    sort_values(by="est", ascending = False).head(n_recomend)

    recomendados = predictions_userID[['iid','est']]
    recomendados.to_sql('reco',conn,if_exists="replace")
    
    recomendados=pd.read_sql('''select a.*, b.title 
                             from reco a left join full_ratings b
                             on a.iid=b.movieId ''', conn)
    # Eliminar duplicados si existen y limitar el número de resultados a n_recomend
    recomendados = recomendados.drop_duplicates(subset=['iid']).head(n_recomend)
    # Restablecer el índice
    recomendados = recomendados.reset_index(drop=True)

    return(recomendados)
 
recomendaciones(user_id=5,n_recomend=10)
########################################################################################
# Consultar 5 user_id distintos
user_ids = pd.read_sql('SELECT DISTINCT user_id FROM full_ratings LIMIT 5', conn)
print("User IDs disponibles:")
print(user_ids)

# Consultar 5 movieId distintos
movie_ids = pd.read_sql('SELECT DISTINCT movieId FROM full_ratings LIMIT 5', conn)
print("\nMovie IDs disponibles:")
print(movie_ids)
##########################################################################################