import numpy as np
import pandas as pd
import sqlite3 as sql
import plotly.graph_objs as go ### para gráficos
import plotly.express as px
import a_funciones as fn
## pip install mlxtend
from mlxtend.preprocessing import TransactionEncoder

#pip install --upgrade nbformat
###pip install  pysqlite3



###### para ejecutar sql y conectarse a bd ###
conn=sql.connect('data\\db_movies2') ### crear cuando no existe el nombre de cd  y para conectarse cuando sí existe.
cur=conn.cursor() ###para funciones que ejecutan sql en base de datos

##conn.close() ### cerrar conexion base de datos

##Verificar tablas que hay en la base de datos
cur.execute("SELECT name FROM sqlite_master where type='table' ")
cur.fetchall()

##### consultar trayendo para pandas ###
ratings=pd.read_sql("select * from ratings", conn)
movies=pd.read_sql("select * from movies", conn)

##### EXPLORACIÓN INICIAL
movies.info()
movies.head()
movies.duplicated().sum()

# ver duplicados por titulo 
duplicados = movies[movies.duplicated(subset=['title'], keep=False)]
duplicados.sort_values(by='title')
duplicados.info()
##### CANTIDAD DE CALIFICACIONES POR PELICULA
ratings.info()
ratings.head()
ratings.duplicated().sum()


##### DESCRIPCIÓN DE BASE RATINGS

### calcular distribución de calificaciones
cr=pd.read_sql(""" select 
                          "rating", 
                          count(*) as conteo 
                          from ratings
                          group by "rating"
                          order by conteo desc""", conn)

data  = go.Bar( x=cr.rating,y=cr.conteo, text=cr.conteo, textposition="outside")
Layout=go.Layout(title="Count of ratings",xaxis={'title':'Rating'},yaxis={'title':'Count'})
go.Figure(data,Layout)



### CANTIDAD DE PELICULAS CALIFICADAS POR USUARIO
rating_users=pd.read_sql(''' select "userId",
                         count(*) as cnt_rat
                         from ratings
                         group by "userId"
                         order by cnt_rat asc
                         ''',conn )

fig  = px.histogram(rating_users, x= 'cnt_rat', title= 'Histrograma de # calificaciones por usario')
fig.show() 

rating_users.describe()
### la mayoria de usarios tiene pocas Peliculas calificadas, pero los que más tienen muchos



#### EXCLUIR USUARIOS CON MENOS DE 50 Y MAS DE 450 PELICULAS CALIFICADAS
rating_users2=pd.read_sql(''' select "userId",
                         count(*) as cnt_rat
                         from ratings
                         group by "userId"
                         having cnt_rat >=10 and cnt_rat <=450
                         order by cnt_rat asc
                         ''',conn )

rating_users2.describe()

fig  = px.histogram(rating_users2, x= 'cnt_rat', title= 'Hist frecuencia de numero de calificaciones por usario')
fig.show() 



#### CANTIDAD DE CALIFICACIONES POR PELICULA
rating_movies=pd.read_sql(''' select movieId ,
                         count(*) as cnt_rat
                         from ratings
                         group by "movieId"
                         order by cnt_rat desc
                         ''',conn )

fig  = px.histogram(rating_movies, x= 'cnt_rat', title= 'Hist frecuencia de numero de calificaciones para cada pelicula')
fig.show() 

rating_movies.describe()


#### EXCLUIR PELICULAS CON MENOS DE 10 CALIFICACIONES
rating_movies2=pd.read_sql(''' select movieId ,
                         count(*) as cnt_rat
                         from ratings
                         group by "movieId"
                         having cnt_rat>=10
                         order by cnt_rat desc
                         ''',conn )

rating_movies2.describe()

fig  = px.histogram(rating_movies2, x= 'cnt_rat', title= 'Hist frecuencia de numero de calificaciones para cada pelicula')
fig.show() 



#### EJECUCIÓN DEL PREPROCESAMIENTO - SQL
fn.ejecutar_sql('preprocesamientos.sql', cur)
## Verificar tablas que hay en la base de datos
cur.execute("select name from sqlite_master where type='table' ")
cur.fetchall()

usuarios_sel=pd.read_sql("select * from usuarios_sel", conn)
movies_sel=pd.read_sql("select * from movies_sel", conn)
ratings_final=pd.read_sql("select * from ratings", conn)
movies_final=pd.read_sql("select * from movies", conn)
full_ratings=pd.read_sql('select * from full_ratings',conn)

### verficar tamaño de tablas con filtros ####

####movies
pd.read_sql('select count(*) from movies', conn)
pd.read_sql('select count(*) from movies_final', conn)

##ratings
pd.read_sql('select count(*) from ratings', conn)
pd.read_sql('select count(*) from ratings_final', conn)

## 3 tablas cruzadas ###
pd.read_sql('select count(*) from full_ratings', conn)

ratings=pd.read_sql('select * from full_ratings',conn)
ratings.duplicated().sum() ## al cruzar tablas a veces se duplican registros
ratings.info()
ratings.head(10)


### Este tratamiento se debe revisar para que tablas se hace el tratamiento
### ______________________________________________________________________

### tratamiento para separar generos en columnas
genres=movies['genres'].str.split('|')
te = TransactionEncoder()
genres = te.fit_transform(genres)
genres = pd.DataFrame(genres, columns = te.columns_)

### unir tablas
movies_tratada = pd.concat([movies, genres], axis=1)
movies_tratada = movies_tratada.drop('genres', axis=1)



### para llevar un data frame de pandas a SQL
completo = movies_tratada
completo.to_sql('movies',conn,index=False, if_exists='replace')