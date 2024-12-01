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