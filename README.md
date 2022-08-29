# Cryptocurrencies


1.	Overview of the Analysis

              Purpose of the Analysis
              
The purpose of this project was to analyze a dataset from many alternative cryptocurrencies to spot trends that make a firm or person want to invest in them. The problem with cryptos is that the most common ones, like bitcoin or ethereum, are becoming unaffordable for the common public. That being said, I will be using unsupervised machine learning to see if we can spot any trends that result in opportunities of these altcoins.

Resources

	Python

	Jupyter notebook

	Sklearn, pandas, and hvplot libraries

	Unsupervised Machine Learning

2.	Results

Follow the code closely in the crypto_clustering.ipynb

The data were preprocessed and transformed in order that unsupervised machine learning could work. This included dropping null values, using only tradaeble and mined cryptocurrencies, numerically encoding categorical columns using the pandas.get_dummies method, and scaling the data using the StandardScaler() method as well.

I.	Three Principal Components

To create a DataFrame with three principal components, the following code was used:

#Create a DataFrame with the three principal components.

pcs_df = pd.DataFrame(crypto_pca_df, columns=["PC 1", "PC 2", "PC 3"], index=X.index)

pcs_df.head(10)



![image](https://user-images.githubusercontent.com/104377031/187316830-78e74827-233d-450e-b8eb-ff9377b7c83c.png)



Crypto_df DataFrame for deliverable 1
 
DataFrame for deliverable 2

II. Elbow Curve

To create the elboe curve, the code below was used:

#create an elbow curve to find the best value for k.
inertia = []
k = list(range(1, 11))
 
#calculate the inertia for the rangr of k values

for i in k:

    km = KMeans(n_clusters=i, random_state=0)
    
    km.fit(pcs_df)
    
    inertia.append(km.inertia_)
    
#Define a DataFrame to plot the Elbow Curve using hvPlot

elbow_data = {"k":k, "inertia":inertia}

elbow_df = pd.DataFrame(elbow_data)

elbow_df.hvplot.line(x="k", y="inertia", title="Elbow Curve", Xticks=k) 




![image](https://user-images.githubusercontent.com/104377031/187316964-c4a8f4d3-785d-4811-8fde-5144d2927837.png)




Curve from deliverable 3





III    K-Means model

#Initialize the K-Means model.

model = KMeans(n_clusters=4, random_state=0)

#Fit the model
model.fit(pcs_df)

#Predict clusters

predictions = model.predict(pcs_df)

print(predictions)

the KMeans analysis to fit the pca dataframe and predict the clustering. The product was this clustered_df with a 'Class' column that showed the predictions to which group it belonged to.

Then, Iused the following codes to create a new DataFrame including predicted clusters and cryptocurrencies features:

#Create a new DataFrame including predicted clusters and cryptocurrencies features.
#Concatentate the crypto_df and pcs_df DataFrames on the same columns.
clustered_df = pd.concat([crypto_df, pcs_df], axis=1)

#Add a new column, "CoinName" to the clustered_df DataFrame that holds the names of the cryptocurrencies. 
clustered_df["CoinName"] = crypto_name_df

#Add a new column, "Class" to the clustered_df DataFrame that holds the predictions.
clustered_df["Class"] = model.labels_

#Print the shape of the clustered_df
print(clustered_df.shape)
clustered_df.head(10)





![image](https://user-images.githubusercontent.com/104377031/187317130-c0545da9-cebe-460f-aa42-6a59d060953a.png)





 
IV.  3D-Scatter with the PCA and the Clusters

The following codes were used to create the 3D-Scatter:

#Creating a 3D-Scatter with the PCA data and the clusters

fig = px.scatter_3d(
    clustered_df, 
    x="PC 1", 
    y="PC 2", 
    z="PC 3", 
    color="Class", 
    symbol="Class", 
    hover_name="CoinName", 
    hover_data=["Algorithm"])
fig.update_layout(legend=dict(x=0, y=1))
fig.show()




![image](https://user-images.githubusercontent.com/104377031/187317217-6959103c-981a-4adb-a1b3-b60a127cd1ec.png)


3D-Scatter
 
 
 
This first one was a 3D scatter plot which located each clustered crypto in relation to the 3 principal components created on the PCA. As it is seen, there are 3 major groups and one outlier.


![image](https://user-images.githubusercontent.com/104377031/187317349-164769b7-53a3-4c95-b044-5033730ee596.png)



 
Similarly, when trying to graph the clustered cryptos by total supply and mined coins, we can observe two outliers. The first one with a lot of supply and a lot of mined coins (BitTorrent Crypto) and another one with a lot of supply but not too many coins mined (TurtleCoin).


3.	Summary


It is the job of unsupervised machine learning to discover patterns or groups of data when there is unknown output. Having said that, this analysis was successful at grouping cryptocurrencies into 4 groups. If we were to create a crypto investment portfolio, we would need to further analyse the clusters. We have a good start point where we can see that the most profitable and known cryptos are somewhat in the 2 groups that have less supply and mined coins in comparison to others. 

