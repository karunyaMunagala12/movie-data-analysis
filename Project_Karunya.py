#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import re
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def scrape_dataset():
    start_time = time.time()
    print('Loading imdb dataset...') 
    print("Estimated time 5 seconds\n")
    url = 'https://www.imdb.com/search/title/?count=100&groups=top_1000&sort=user_rating'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    movie_name = []
    year = []
    rating = []
    metascore = []
    gross = []
    Director = []
    movie_data = soup.findAll('div', attrs= {'class': 'lister-item mode-advanced'})
    for store in movie_data:
        name = store.h3.a.text
        movie_name.append(name)
        year_of_release = store.h3.find('span', class_ = 'lister-item-year text-muted unbold').text.replace('(', '').replace(')', '')
        year.append(year_of_release)
        rate = store.find('div', class_ = 'inline-block ratings-imdb-rating').text.replace('\n', '')
        rating.append(rate)
        meta  = store.find('span', class_ = 'metascore').text.replace(' ', '') if store.find('span', class_ = 'metascore') else '^^^^^^'
        metascore.append(meta)
        value = store.find_all('span', attrs = {'name': 'nv'})
        grosses = value[1].text if len(value) >1 else '*****'
        gross.append(grosses)
        cast = store.find("p", class_ = '')
        cast = cast.text.replace('\n', '').split('|')
        cast = [x.strip() for x in cast]
        cast = [cast[i].replace(j, "") for i,j in enumerate(["Director:", "Stars:"])]
        Director.append(cast[0])
        
    movie_DF = pd.DataFrame({'original_title': movie_name, 'Year of release': year, 'Movie Rating': rating, "Director": Director,  'Gross collection': gross})
    movie_DF.to_csv("IMDB_Movies.csv")
    pd.set_option('display.max_columns', None)  
    pd.set_option('display.width', 120)  
    pd.set_option('display.max_colwidth', 80)
    print(movie_DF)
    print("Data has been saved to IMDB_Movies.csv file.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Scraping dataset completed in {elapsed_time:.2f} seconds.\n')

import sys
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

def api_dataset():
    start_time = time.time()
    print('Loading data from API...')
    print("Estimated time  5 seconds\n")
    api_key = 'b8c7204130f70ad93ed1bb6daec91852'
    base_url = f'https://api.themoviedb.org/3/discover/movie?api_key={api_key}&language=en-US&sort_by=popularity.desc&include_adult=false&include_video=false&page={{page_number}}&release_date.gte=1980-01-01&release_date.lte=2022-12-31'
    movie_name = []
    release_year = []
    gross_collection = []
    movie_rating = []
    director = []
    popularity = []

    for page_number in range(1, 11):  
        url = base_url.format(page_number=page_number)
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            movies = data['results']
            for movie in movies:
                movie_name.append(movie['title'])

                try:
                    release_year.append(movie['release_date'][:4])
                except KeyError:
                    release_year.append('Unknown')

                try:
                    gross_collection.append(movie['revenue'])
                except KeyError:
                    gross_collection.append('Unknown')

                try:
                    movie_rating.append(movie['vote_average'])
                except KeyError:
                    movie_rating.append('Unknown')

                try:
                    director.append(movie['credits']['crew'][0]['name'])
                except KeyError:
                    director.append('Unknown')

                try:
                    popularity.append(movie['popularity'])
                except KeyError:
                    popularity.append('Unknown')

    movie_DF = pd.DataFrame({'Name of movie': movie_name, 'Year of release': release_year, 'Gross collection': gross_collection, 'Movie Rating': movie_rating, 'Director': director, 'Popularity': popularity})
    movie_DF.to_csv("Apidata_movies.csv")
    print(movie_DF)

    print("Movie data has been saved to Apidata_movies.csv file.")
   
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Fetching data from API completed in {elapsed_time:.2f} seconds.\n')
def database():
    start_time = time.time()
    print('Reading data from database...')
    print("Estimated time 5 seconds\n")
    movie_df = pd.read_csv("Dataset_movies.csv")
    print(movie_df)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Reading data from database completed in {elapsed_time:.2f} seconds.\n')


# In[ ]:


scrape_dataset()
api_dataset()
database()


# In[2]:


# read the first CSV file into a dataframe
df1 = pd.read_csv('IMDB_Movies.csv')
# read the second CSV file into a dataframe
df2 = pd.read_csv('Dataset_movies.csv')

df3=pd.read_csv('Apidata_movies.csv')
dfk=df2.merge(df1,how='right')
dfk.to_csv('merged_data1.csv')
con=pd.concat([dfk,df3])
con.to_csv('Final_combined_dataset.csv')


# In[3]:


# read the CSV file into a pandas dataframe
df = pd.read_csv('Final_combined_dataset.csv')

# drop the columns you don't need
df = df.drop(['Unnamed: 0', 'homepage', 'release_date', 'release_year', 'revenue_adj', 'Director','budget_adj','production_companies','vote_average'], axis=1)

# Load the CSV file into a pandas dataframe
df = pd.read_csv('Final_movie_dataset.csv')

# Fill NaN values in categorical columns with '0'
categorical_cols = ['original_title', 'cast', 'director', 'tagline', 'keywords', 'overview', 'genres']
df[categorical_cols] = df[categorical_cols].fillna('0')

# Replace NaN values in numerical columns with mean values
numerical_cols = ['budget', 'revenue', 'runtime', 'vote_count', 'Movie Rating','popularity']
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Save the modified dataframe back to a CSV file
df.to_csv('Movies_file.csv', index=False)


# In[4]:


# convert 'Gross collection' column to string
df['Gross collection'] = df['Gross collection'].astype(str)

# clean the 'Gross collection' column and convert it to numeric
df['Gross collection'] = df['Gross collection'].str.replace('$', '').str.replace('M', '').str.replace('*', '').str.strip()
df['Gross collection'] = pd.to_numeric(df['Gross collection'], errors='coerce')

# calculate the mean of the 'Gross collection' column
mean_gross_collection = df['Gross collection'].mean()

# replace the missing values with the mean
df['Gross collection'] = df['Gross collection'].fillna(mean_gross_collection)


# In[5]:


# Top 10 movies based on the rating given by public
df_1=df[['original_title','Movie Rating','Year of release']]
df_1["Movie Rating"].nlargest(10)
df_1= df_1.sort_values("Movie Rating", ascending=False)
print(df_1.head(10))
print("TheTop 10 movies based on the rating given by public")


# In[6]:


start_time=time.time()
# set the figure size of the plot
plt.figure(figsize=(15,10))

# create a countplot of the "Year of release" column from the dataframe df_1, using the seaborn library
sns.countplot(y=df_1['Year of release'],palette='coolwarm')

# set the title of the plot
plt.title("Count of movies vs year")

# set the label for the y-axis of the plot
plt.ylabel("Year")

# set the label for the x-axis of the plot
plt.xlabel("Number of movies")
print("Loading countplot..")
print("Estimated time:5sec\n")

# display the plot
plt.show()
end_time = time.time()
elapsed_time = end_time - start_time


# In[7]:


# Create a new column called "Rating Range" and plotting the graph between no of ratings and their respective movies
start_time=time.time()
def rating_range(rating):
    if rating >= 9:
        return "9-10"
    elif rating >= 8:
        return "8-9"
    elif rating >= 7:
        return "7-8"
    elif rating >= 6:
        return "6-7"
    elif rating >= 5:
        return "5-6"
    else:
        return "0-5"

df_1["Range_rating"] = df_1["Movie Rating"].apply(rating_range)

# Group the DataFrame by the "Rating Range" column and count the number of movies in each group
df_count = df_1.groupby("Range_rating").count()

# Plot the number of movies against the rating range using matplotlib
plt.figure(figsize=(10,7))
plt.bar(df_count.index, df_count["original_title"])
plt.xlabel("Rating Range")
plt.ylabel("Number of movies")
print("loading plot the graph between no of ratings and their respective movies")
print("Estimated time:5sec\n")

plt.show()
end_time = time.time()
elapsed_time = end_time - start_time


# In[11]:


start_time=time.time()
print("Loading sns pairplot...")
sns.pairplot(df[['popularity', 'budget', 'revenue', 'runtime', 'vote_count', 'Year of release', 'Movie Rating', 'Gross collection']])
print("Estimated time:5sec\n")
end_time = time.time()
elapsed_time = end_time - start_time


# In[12]:


# Clean the "Year of release" column
df["Year of release"] = df["Year of release"].apply(lambda x: re.sub("[^0-9]+", "", str(x)))

# Convert the "Year of release" column to an integer type
df["Year of release"] = df["Year of release"].astype(int)

# Create a new column called "Year Range"
def year_range(year):
    if year >= 1970 and year <= 1980:
        return "1970-1980"
    elif year >= 1980 and year <= 1990:
        return "1980-1990"
    elif year >= 1990 and year <= 2000:
        return "1990-2000"
    elif year>= 2000 and year <= 2010:
        return "2000-2010"
    elif year>= 2010 and year <= 2020:
        return "2010-2020"
    else:
        return "..."

# Apply the "year_range" function to create a new column called "Year Range"
df["Year Range"] = df["Year of release"].apply(year_range)

# Calculate the movie rating average per year range
year_ranges = ["1970-1980", "1980-1990", "1990-2000", "2000-2010", "2010-2020"]
averages = []
for year_range in year_ranges:
    filtered = df[df["Year Range"] == year_range]
    rating_average = filtered["Movie Rating"].mean()
    averages.append(rating_average)

# Plot the graph

start_time=time.time()

plt.plot(year_ranges, averages)
plt.title("Movie Rating Average per Decade")
plt.xlabel("Decade")
plt.ylabel("Rating Average")
print("Plot loading for Movie Rating Average per Decade")
print("Estimated time:5sec\n")
plt.show()
end_time = time.time()
elapsed_time = end_time - start_time


# In[13]:


start_time=time.time()
# group data by popularity and calculate mean runtime for each group
popularity_groups = df.groupby('Movie Rating')['runtime'].mean()

# plot bar graph for top 30 popularity values

top_30_popularity = popularity_groups.nlargest(30)
plt.figure(figsize=(12,6))
plt.bar(top_30_popularity.index, top_30_popularity.values)
plt.xlabel('Mean Movie Rating')
plt.ylabel('Mean Runtime')
plt.title('Top 30 Popular Movies by Mean Runtime')
print("Plot loading for Top 30 Popular Movies by Mean Runtime ")
print("Estimated time:5sec\n")
plt.show()
end_time = time.time()
elapsed_time = end_time - start_time


# In[15]:


print("the data set of orginal title and genres")
print(df[['original_title','genres']].head())


# In[19]:



# Create a dictionary to store total rating and count for each genre
genre_rating = {}

# Loop over each row in the dataframe
for index, row in df.iterrows():
    # Split the genre column by '|'
    genres = row['genres'].split('|')
    # Loop over each genre and update the genre_rating dictionary
    for genre in genres:
        if genre not in genre_rating:
            genre_rating[genre] = {'total_rating': 0, 'count': 0}
        genre_rating[genre]['total_rating'] += row['Movie Rating']
        genre_rating[genre]['count'] += 1

# Create a list of tuples containing genre and average rating
genre_avg_rating = [(k, v['total_rating'] / v['count']) for k, v in genre_rating.items()]

# Sort the list in descending order by average rating and get top 10 genres
top_genres = sorted(genre_avg_rating, key=lambda x: x[1], reverse=True)[:10]
print("table containing the values oftop 10 genres and their average rating")
# Print the top 10 genres and their average rating
for genre, avg_rating in top_genres:
    print(f"{genre}: {avg_rating:.2f}")
    


    


# In[20]:


start_time=time.time()

# Plot the top 10 genres by average rating
top_genres_names = [genre[0] for genre in top_genres]
top_genres_ratings = [genre[1] for genre in top_genres]

plt.bar(top_genres_names, top_genres_ratings)
plt.xlabel('Genres')
plt.xticks(rotation=45)
plt.ylabel('Average Rating')
plt.title('Top 10 Genres by Average Rating')
print("Plot for Top 10 Genres by Average Rating is loading... ")
print("Estimated time:5sec\n")
plt.show()
end_time = time.time()
elapsed_time = end_time - start_time


# In[21]:


start_time=time.time()

# Create a new variable to indicate whether a movie is high-grossing or low-grossing
df["gross_category"] = pd.qcut(df["revenue"], q=2, labels=["low", "high"])

# Create a subset of the data for low-grossing movies
low_grossing = df[df["gross_category"] == "low"]

# Create a subset of the data for high-grossing movies
high_grossing = df[df["gross_category"] == "high"]

# Perform a t-test to compare the budgets of high-grossing and low-grossing movies
t, p = stats.ttest_ind(high_grossing["budget"], low_grossing["budget"], equal_var=False)
print("t-test to compare the budgets of high-grossing and low-grossing movies")
print("Estimated time:5sec\n")
# Print the results of the t-test
print("T-value: ", t)
print("P-value: ", p)
end_time = time.time()
elapsed_time = end_time - start_time


# In[25]:


start_time=time.time()

# Create a subset of the data with just budget, revenue, and popularity
subset = df[["budget", "revenue", "popularity"]]

# Standardize the subset using the StandardScaler function
scaler = StandardScaler().fit(subset)
scaled_subset = scaler.transform(subset)

# Perform k-means clustering on the scaled subset with k=3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(scaled_subset)

# Add the cluster labels back to the original dataset
df["cluster"] = kmeans.labels_

# Visualize the clusters
plt.scatter(df["budget"], df["revenue"], c=df["cluster"])
plt.ticklabel_format(style='plain', axis='both')
plt.title("Budget vs revenue")
plt.xlabel("Budget")
plt.ylabel("Revenue")
print(" k-means clustering plot loading ....")
print("Estimated time:5sec\n")

plt.show()
end_time = time.time()
elapsed_time = end_time - start_time


# In[26]:



start_time=time.time()

# define X and y
X = df['Movie Rating'].values.reshape(-1, 1)
y = df['Gross collection'].values.reshape(-1, 1)

# Create scatter plot
plt.scatter(X, y)
plt.title("Movies vs Gross collection")
plt.xlabel('Movie Ratings')
plt.ylabel('Gross Collection')

# Create linear regression object and fit the model
model = LinearRegression()
model.fit(X, y)
print("linear regression graph loading....")
print("Estimated time:5sec\n")

# Plot regression line
plt.plot(X, model.predict(X), color='red', linewidth=1)

plt.show()
end_time = time.time()
elapsed_time = end_time - start_time


# In[28]:


start_time=time.time()


# select columns of interest
cols = ['budget', 'revenue', 'runtime', 'vote_count', 'Year of release']
data = df[cols]

# calculate Pearson correlation coefficient
pearsoncorr = data.corr(method='pearson')

# plot the heatmap using seaborn
plt.figure(figsize=(10,10))
sns.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,

            linewidth=0.5)
print("Loading the correlation....")
print("Estimated time:5sec\n")
plt.show()

end_time = time.time()
elapsed_time = end_time - start_time


# In[29]:


start_time=time.time()

# list of columns to calculate correlation with
cols = ['budget', 'revenue', 'runtime', 'vote_count', 'Year of release']

# create a DataFrame to store the correlation coefficients
corr_df = pd.DataFrame(columns=['Column', 'Pearson', 'Kendall', 'Spearman'])
print("correlation coefficients table.. ")

print("Estimated time:5sec\n")


# iterate over columns and calculate correlation with 'Gross collection'
for col in cols:
    # convert column to numeric values
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # calculate Pearson correlation coefficient
    pearson_corr = df['Gross collection'].corr(df[col], method='pearson')
    
    # calculate Kendall correlation coefficient
    kendall_corr = df['Gross collection'].corr(df[col], method='kendall')
    
    # calculate Spearman correlation coefficient
    spearman_corr = df['Gross collection'].corr(df[col], method='spearman')
    
    # append the results to the DataFrame
    corr_df = corr_df.append({'Column': col, 'Pearson': pearson_corr, 
                              'Kendall': kendall_corr, 'Spearman': spearman_corr}, 
                             ignore_index=True)

    
# print the correlation coefficients table
print(corr_df)
end_time = time.time()
elapsed_time = end_time - start_time


# In[ ]:




