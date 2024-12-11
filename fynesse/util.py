import numpy as np
import pandas as pd

def flatten_tags(tags):
  tags_list = []
  for t in tags:
    if isinstance(tags[t], list):
      tags_list += tags[t]
    else:
      tags_list.append(t)
  return tags_list

# Retrieve the rows that are in London
def in_london(df, name_col='lad_name'):
  london_names = ["Barking and Dagenham", "Barnet", "Bexley", "Brent", 
                "Bromley", "Camden", "Croydon", "Ealing", "Enfield", 
                "Greenwich", "Hackney", "Hammersmith and Fulham", 
                "Haringey", "Harrow", "Havering", "Hillingdon", 
                "Hounslow", "Islington", "Kensington and Chelsea", 
                "Kingston upon Thames", "Lambeth", "Lewisham", 
                "Merton", "Newham", "Redbridge", "Richmond upon Thames", 
                "Southwark", "Sutton", "Tower Hamlets", 
                "Waltham Forest", "Wandsworth", "Westminster"]

  london_lads = df[df[name_col].isin(london_names)]
  return london_lads


def get_colour_percentiles(df, number, start, end, num_col, col_name='colours'):
  start_col = np.array([*start])
  end_col = np.array([*end])
  colors = [ tuple(x) for x in np.linspace(start_col, end_col, number)]
  df['colours'] = pd.qcut(df[num_col].to_numpy(), number, labels=colors)

def get_colour_outliers(df, number, cutoff, num_col, high='red', low='blue', col_name='colours', alpha=None):
  low_col = np.array([*low])
  high_col = np.array([*high])
  
  labels = np.arange(number)
  df['pos'] = pd.qcut(df[num_col].to_numpy(), number, labels=labels)
  df['neg'] = pd.qcut(-df[num_col].to_numpy(), number, labels=labels)
  df[col_name] = 'grey'
  df.loc[comparison['pos'] >= number-cutoff[0], col_name] = 'red'
  df.loc[comparison['neg'] >= number-cutoff[1], col_name] = 'blue'

  if alpha is not None:
    df['alpha'] = alpha[0]
    df.loc[df['neg'] >= number-2, 'alpha'] = alpha[1]
    df.loc[df['pos'] >= number-2, 'alpha'] = alpha[1]