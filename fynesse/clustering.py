import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from .util import *

# Assign nodes to the nearest centres
def assign_centres(df, centres, drop_cols=[]):
  num = len(df.index)
  assignment = {}
  for i,c in enumerate(centres):
    assignment[i] = []

  for i in range(num):
      vec1 = np.array(df.iloc[i].drop(drop_cols)).astype(np.float64)

      min_centre = 0
      min_diff = np.iinfo(np.int32).max

      for j,c in enumerate(centres):
        diff = np.sqrt(((vec1 - c)**2).sum())

        if min_diff > diff:
          min_diff = diff
          min_centre = j
      #print(min_diff, min_centre)

      assignment[min_centre].append(i)
  return assignment

def new_centres(df, cluster, drop_cols=[]):
  ret_centres = []
  for k,v in cluster.items():
    vecs = np.array(df.iloc[v].drop(drop_cols, axis=1)).astype(np.float64)
    new_centre = vecs.sum(axis=0) / len(vecs)
    ret_centres.append(new_centre)
  return ret_centres

def lloyd_clustering(df, k, initial_centres=None, drop_cols=[]):
  num = len(df.index)
  assert(k < num)

  if initial_centres is None:
    centres = np.array(df.iloc[np.random.choice(num, k, replace=False)].drop(drop_cols,axis=1).astype(np.float64))
  else:
    centres = np.array(df.iloc[initial_centres].drop(drop_cols,axis=1).astype(np.float64))

  cluster = assign_centres(df, centres, drop_cols)
  while True:
    centres = new_centres(df, cluster, drop_cols)
    new_cluster = assign_centres(df, centres, drop_cols)
    if new_cluster == cluster:
      return new_cluster
    else:
      cluster = new_cluster


# Run Lloyds algorithm with optimal start indices
# This is computed by finding the furthest away indices to start with, rather than the first few
def lloyds_altered(df, k, drop_cols=[]):
  num  = len(df.index)
  i = np.random.choice(num, 1)[0]
  d = np.array(df.drop(drop_cols, axis=1).astype(np.float64))
  count = 1
  initial_indices = [i]

  s = (((d - d[i])**2).sum(axis=1))
  while count < k:
    v = (((d - d[i])**2).sum(axis=1))
    s = np.minimum.reduce([s, (((d - d[i])**2).sum(axis=1))])

    i = s.argmax(axis=0)
    initial_indices.append(i)
    count+=1

  print(initial_indices)
  return lloyd_clustering(df, k, initial_centres= initial_indices, drop_cols=drop_cols)



colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']


#--------

# Plot clustering of a geo dataframe, using lloyds_altered
def plot_kmeans_clustering_locations(gdf, df, number, colors, key='LAD23CD', drop_cols= ['LAD23CD', 'LAD23NM'], ax=None, england_only=True):
  merged = gdf.merge(df, left_on=key, right_on=key)
  if england_only:
    merged = merged[merged['LAD23CD'].str.contains('E')]
  clusters = lloyds_altered(merged[['avg_rag_estate', 'avg_rag_flats', 'avg_rag_shopping']], number, drop_cols = [])
  
  if ax is None:
    fig, ax = plt.subplots(figsize=(10,10))
  
  merged['colours'] = 'black'
  
  for k,v in clusters.items():
    c = colors[k]
    merged.loc[v, 'colours'] = c
  
  lon = inset_axes(ax, width="30%", height="30%", loc="upper left") 
  lon.set_xticks([])
  lon.set_yticks([])

  london_lads = in_london(merged, name_col='LAD23NM_x')
  london_lads.plot(edgecolor="black", color=london_lads['colours'], ax = lon)

  merged.plot(color = merged['colours'], alpha=0.7, edgecolor='black', ax=ax)
