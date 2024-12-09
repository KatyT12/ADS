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




  def split_data(k, n):
  ind = np.random.choice(range(n), size=(n,), replace=False)
  k_sets = np.array_split(ind, k)

  return k_sets

# Retrieve the training data from the splitted indices
def get_train(splitted, d, i):

  left = [d.iloc[splitted[j]] for j in range(i)]
  right = [d.iloc[splitted[j]] for j in range(i+1,len(splitted))]
  if i == 0:
      return pd.concat(right)
  elif i == len(splitted)-1:
      return pd.concat(left)
  else:
    return pd.concat([pd.concat(left), pd.concat(right)])

  total = np.concatenate(left, right)
  return total

def rmse_calc(predicted, actual):
  return np.sqrt(((predicted - actual)**2).mean())


# Training data - augmented, with resposne
# Subset of training data
def run_on_lad_subset(connection, training, subset_training, subset_test, response_col, design_func, regularized = False, alpha = 0.1, weight = 1.0):
  # Train
  
  if regularized:
    model = fynesse.address.fit_model_OLS(connection, subset_training, subset_training, response_col, design_func, augmented=subset_training, ridge=alpha, reg_weight=weight)
  else:
    model = fynesse.address.fit_model_OLS(connection, subset_training, subset_training, response_col, design_func, augmented=subset_training)
  
  X = design_func(subset_test, training)
  prediction = model.predict(X)
  actual = subset_test[response_col]

  # Extract data
  c = pd.DataFrame({
      "actual": actual,
      "predicted": prediction
      })

  corr = c.corr()['actual'].iloc[1]
  r2 = r2_score(actual, prediction)
  rmse = rmse_calc(prediction, actual)

  #print(corr, rmse, r2)
  return [corr, rmse, r2]


def get_k_folded_results(k, training, response, design_func, regularized = False, alpha = 0.01, weight = 1.0):
  n = len(training.index)
  splitted = split_data(k, n)

  ret = []
  for i in range(len(splitted)):
    training_data = get_train(splitted, training, i)
    test_data = training.iloc[splitted[i]]

    #print(age_df.shape, a.shape, b.shape)
    ret.append(run_on_lad_subset(connection, training, training_data, test_data, response, design_func, regularized=regularized, alpha=alpha, weight=weight))

  dat = pd.DataFrame(ret)
  dat.columns = ['corr', 'rmse', 'r2']
  return dat