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
import plotly.express as px
import plotly.graph_objs as go
##%pip install mlxtend
from mlxtend.preprocessing import TransactionEncoder
import joblib
from sklearn import neighbors ### basado en contenido un solo producto consumido
from sklearn.neighbors import NearestNeighbors
from ipywidgets import interact

#pip install dash plotly pandas
import dash
from dash import dcc, html

# Conectar a la base de datos
conn = sql.connect('data/db_movies2')

# Cargar la tabla 'full_ratings'
movies = pd.read_sql('SELECT * FROM movies_final', conn)

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
genres_df
# Unir las nuevas columnas de géneros con el DataFrame original
movies = pd.concat([movies, genres_df], axis=1)

# Remover el año del título de la columna 'title'
movies['title'] = movies['title'].str.replace(r'\s\(\d{4}\)$', '', regex=True)
# Verificar los cambios
print(movies[['title', 'year']].head())
movies

# Eliminar columnas innecesarias para los modelos
final2 = movies.drop(columns=['movieId','genres','title','year'])

joblib.dump(final2,"data\\fianl2.joblib") ### para utilizar en segundos modelos

###############################################################################################
# ***** 1. Recomendación basada en popularidad *****
# Opción 1.  10 películas más calificadas
# Seleccionar las películas más calificadas
top_rated_movies = pd.read_sql('''
    SELECT movieId, title, COUNT(*) AS num_ratings, AVG(rating) AS avg_rating
    FROM full_ratings
    GROUP BY title
    HAVING num_ratings >= 10
    ORDER BY avg_rating DESC, num_ratings DESC
    LIMIT 10
''', conn)

# Mostrar las 10 películas más calificadas
fig_most_rated = px.bar(top_rated_movies, 
                        x='title', 
                        y='avg_rating', 
                        title='Top 10 Películas Más Calificadas', 
                        labels={'avg_rating': 'Calificaciones', 'title': 'Película'}, 
                        text='num_ratings')

fig_most_rated.show()

###################################################
# Opción 2. Película mejor calificada por año
# Calcular la calificación promedio por película y año de lanzamiento
full_ratings = pd.read_sql('SELECT * FROM full_ratings', conn)

# Extraer el año de la columna 'title'
full_ratings['year'] = full_ratings['title'].str.extract('\(([^)]*)\)$', expand=False)
nulos_year = full_ratings.loc[full_ratings['year'].isna()]
full_ratings.loc[(full_ratings['title'] == "Ready Player One") & (full_ratings['year'].isna()), 'year'] = 2018
full_ratings['year']=full_ratings.year.astype('int')

top_movies_by_year = full_ratings.groupby(['year', 'title'])['rating'].mean().reset_index()

# Ordenar las películas por año y calificación promedio
top_movies_by_year = top_movies_by_year.sort_values(by=['year', 'rating'], ascending=[True, False])

# Obtener las películas mejor calificadas por año (una por cada año)
df = top_movies_by_year.groupby('year').head(1)

#----------Gráfico con las películas mejor calificadas por año--------------
# Iniciar la aplicación Dash
app = dash.Dash(__name__)

# Layout de la aplicación
app.layout = html.Div([
    html.H1("Mejores Películas por Año"),
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': str(year), 'value': year} for year in sorted(df['year'].unique())],
        value=sorted(df['year'].unique())[0],  # Valor por defecto
        clearable=False
    ),
    dcc.Graph(id='rating-graph')
])

# Callback para actualizar el gráfico
@app.callback(
    dash.dependencies.Output('rating-graph', 'figure'),
    [dash.dependencies.Input('year-dropdown', 'value')]
)
def update_graph(selected_year):
    filtered_df = df[df['year'] == selected_year]
    fig = px.bar(filtered_df, x='title', y='rating', 
                  title=f'Mejores Películas del Año {selected_year}',
                  labels={'rating': 'Calificación', 'title': 'Título'},
                  color='rating', 
                  text='rating')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis=dict(title='Calificación'), xaxis_title='Título')
    return fig

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)

#----------------------------------

# Gráfico de barras con las películas mejor calificadas por año
###fig = px.bar(best_movies_per_year, x='year', y='rating', color='title',
###             title='Películas Mejor Calificadas por Año de Lanzamiento',
###             labels={'rating': 'Calificación Promedio', 'year': 'Año de Lanzamiento', 'title': 'Título de la Película'},
###             hover_data=['title'])

# Mostrar el gráfico
###fig.show()
#####################################################
# Opción 3.  Película mejor calificada por género
# Crear una lista de géneros
genres_list = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 
               'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
               'Thriller', 'War', 'Western']

# Separar los géneros en columnas individuales
genres = full_ratings['genres'].str.split('|')
te = TransactionEncoder()
genres_matrix = te.fit_transform(genres)
genres_df = pd.DataFrame(genres_matrix, columns=te.columns_)
genres_df
# Unir las nuevas columnas de géneros con el DataFrame original
full_ratings = pd.concat([full_ratings, genres_df], axis=1)

popular_movies_by_genre = []

# Encontrar la película más popular por cada género usando el DataFrame
for genre in genres_list:
    # Filtrar las películas que pertenecen al género actual
    genre_movies = full_ratings[full_ratings[genre] == True]
    # Calcular la calificación promedio de cada película en ese género
    top_movie = genre_movies.groupby('title')['rating'].mean().reset_index()
    # Ordenar por la calificación promedio y seleccionar la película con la mejor calificación
    top_movie = top_movie.sort_values(by='rating', ascending=False).head(1)
    top_movie['genre'] = genre
    popular_movies_by_genre.append(top_movie)

# Combinar los resultados en un solo DataFrame
df2 = pd.concat(popular_movies_by_genre, ignore_index=True)

#----------Gráfico con las películas más populares por género--------------
# Iniciar la aplicación Dash
app = dash.Dash(__name__)

# Layout de la aplicación
app.layout = html.Div([
    html.H1("Mejores Películas por Género"),
    dcc.Dropdown(
        id='genre-dropdown',
        options=[{'label': genre, 'value': genre} for genre in sorted(df2['genre'].unique())],
        value=sorted(df2['genre'].unique())[0],  # Valor por defecto
        clearable=False
    ),
    dcc.Graph(id='rating-graph')
])

# Callback para actualizar el gráfico
@app.callback(
    dash.dependencies.Output('rating-graph', 'figure'),
    [dash.dependencies.Input('genre-dropdown', 'value')]
)
def update_graph(selected_genre):
    filtered_df = df2[df2['genre'] == selected_genre]
    fig = px.bar(filtered_df, x='title', y='rating', 
                  title=f'Mejores Películas del Género: {selected_genre}',
                  labels={'rating': 'Calificación', 'title': 'Título'},
                  color='rating', 
                  text='rating')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis=dict(title='Calificación'), xaxis_title='Título')
    return fig

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)

#----------------------------------
### Gráfico con las películas más populares por género
#fig_genre_popularity = px.bar(df2, x='genre', y='rating', color='title',
#                              title='Película Más Popular por Género',
#                              labels={'rating': 'Calificación Promedio', 'genre': 'Género', 'title': 'Título de la Película'},
#                              hover_data=['title'])

# Mostrar el gráfico
#fig_genre_popularity.show()
#######################################################################################

# ***** 2. Sistema de recomendación basado en contenido KNN un solo producto visto *****
# Intento 1
final2

# Entrenar el modelo KNN con 15 vecinos para asegurarse de encontrar suficientes recomendaciones
model = NearestNeighbors(n_neighbors=15, metric='euclidean')
#model = NearestNeighbors(n_neighbors=15, metric='cosine')
model.fit(final2)

# Obtener distancias y listas de vecinos
dist, idlist = model.kneighbors(final2)
distancias=pd.DataFrame(dist)
id_list=pd.DataFrame(idlist) ## para saber esas distancias a que item corresponde
# Modelo de recomendación aplicado a cualquier película
def MovieRecommender(movie_name):
    # Obtener el índice de la película seleccionada
    try:
        movie_id = movies[movies['title'] == movie_name].index[0]
    except IndexError:
        return ["Película no encontrada."]
    
    # Lista para almacenar las películas recomendadas
    list_name = []

    # Obtener las películas más similares, excluyendo la película seleccionada
    for newid in idlist[movie_id]:
        if movies.loc[newid].title != movie_name:  # Filtrar la película seleccionada
            list_name.append(movies.loc[newid].title)
        if len(list_name) >= 3:  # Limitar a 3 recomendaciones
            break
    
        # Devolver las películas recomendadas
    return list_name

# Mostrar recomendaciones utilizando un menú interactivo
interact(MovieRecommender, movie_name=list(movies['title'].value_counts().index));

# En muchos casos no se devuelve una recomendación, se intentó ampliando el número de vecinos
# y aún así hay muchas películas sin recomendación. Se cambió la metrica con cosine y continúa igual

################################################################################################

# Intento 2, para solucionar el problema del intento 1, si no se obtiene recomendación, 
# se seleccionarán 3 películas al azar pertenecientes al mismo género

# Entrenar el modelo KNN con más vecinos para asegurar suficientes recomendaciones

#model = NearestNeighbors(n_neighbors=15, metric='cosine')
#model.fit(final2)

# Obtener distancias y listas de vecinos
#dist, idlist = model.kneighbors(final2)

# Modelo de recomendación aplicado a cualquier película
# def MovieRecommender(movie_name):
    # Obtener el índice de la película seleccionada
#    try:
#        movie_id = movies[movies['title'] == movie_name].index[0]
#    except IndexError:
#        return ["Película no encontrada."]
    
    # Lista para almacenar las películas recomendadas
#    list_name = []

    # Obtener las películas más similares, excluyendo la película seleccionada
#    for newid in idlist[movie_id]:
#        if movies.loc[newid].title != movie_name:  # Filtrar la película seleccionada
#            list_name.append(movies.loc[newid].title)
#        if len(list_name) >= 5:  # Limitar a 5 recomendaciones
#            break

    # Si no se encontraron suficientes películas, añadir películas del mismo género
#    if len(list_name) < 5:
        # Identificar los géneros de la película seleccionada
#        movie_genres = movies.loc[movie_id, movies.columns[3:]].astype(bool)
#        genre_columns = movie_genres.index[movie_genres].tolist()
        
        # Filtrar las películas que pertenecen al mismo género
#        same_genre_movies = movies.copy()
#        same_genre_movies = same_genre_movies[same_genre_movies[genre_columns].any(axis=1)]
#        same_genre_movies = same_genre_movies[~same_genre_movies['title'].isin(list_name + [movie_name])]
        
        # Seleccionar películas aleatorias del mismo género
#        additional_recommendations = same_genre_movies['title'].sample(n=min(5-len(list_name), len(same_genre_movies))).tolist()
#        list_name.extend(additional_recommendations)

    # Devolver las películas recomendadas
#    return list_name

# Mostrar recomendaciones utilizando un menú interactivo
#interact(MovieRecommender, movie_name=list(movies['title'].value_counts().index));
