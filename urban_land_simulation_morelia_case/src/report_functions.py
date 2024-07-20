#Implementation for computing the
#TOC= Total Operating Charcateristic Curve
#Author: Rodrigo Lopez-Farias
#Centro de Investigación en Ciencias de Información Geoespacial AC
#Querétaro, México.


import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import geopandas as gpd
crs_m = 'epsg:4326'
import itertools as it
import numpy as np
import seaborn as sns

def simulate(xtrain, ytrain, xvalid, yvalid, Models):
    dict ={}
    for i, M in enumerate(Models):
        dict["m_{i}_haty_train".format(i=i)] = M.predict(xtrain)
        dict["m_{i}_haty_valid".format(i=i)] = M.predict(xvalid)
    
    return dict

def graficarVariablesEspaciales(data):
    
    geometry = [Point(xy) for xy in zip(data["lat"], data["lon"])]
    geodata = gpd.GeoDataFrame(data, crs = crs_m, geometry = geometry)
    columns = data.columns

    fig, axs = plt.subplots(3, 3, figsize = (10, 10))
    for ix, c in enumerate(it.product(np.arange(3), repeat = 2)):

        i = c[0]
        j = c[1]

        column_name = columns[3+ix]
        geodata.plot(ax = axs[i, j], figsize = (20/4, 15/4), column = geodata[column_name], cmap = "viridis_r", legend = True, markersize = 0.01)
        
        
    plt.tight_layout()
    plt.show()
    
    
def plotParesdeScatterPlots(data):
    
    sns.set_theme(style = "ticks")
    data = data.sample(frac=0.001)
    sns.pairplot(data[data.columns[3:]], hue = "incremento_urbano")

def model_report_comparison(xtrain, ytrain, xvalid, yvalid, Models):


    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for i, M in enumerate(Models):
        print("i:", i)
        axs[i].plot(M.history["loss"], label = "Entrenamiento")
        axs[i].plot(M.history["val_loss"], label = "Validación")
        axs[i].set_title('Convergencia Modelo A')
        axs[i].set_xlabel('Época')
        axs[i].set_ylabel('Entropía Cruzada')
        axs[i].legend()


    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()



def cm_comparison(xtrain, ytrain, xvalid, yvalid, Models):


    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i, M in enumerate(Models):
        
        y_train_hat = M.predict(xtrain)
        y_valid_hat = M.predict(xvalid)

        q = np.sum(yvalid)
        tr = np.sort(y_valid_hat.flatten())[::-1][q]

        axs[1,i].plot(np.sort(y_valid_hat.flatten())[::-1][:q*10], label = "Sorted ranks")
        axs[1,i].vlines(x = q, ymin = 0, ymax = 1, color = "red",linestyles='dotted', label = "Threshold {t} producing {q} urban cells".format(t=np.round(tr,4), q=q))



    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    
    plt.show()
