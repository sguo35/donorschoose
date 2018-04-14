def total_price_data():
  import pandas as pd
  import numpy as np

  data = pd.read_csv('./resources.csv')
  d1 = data.quantity * data.price 
  d2 = pd.DataFrame(d1)
  d2.columns = ['total price']
  d3 = data.join(d2)
  d4 = d3.groupby('id').sum()
  del d4['quantity']
  del d4['price']

  trainset = pd.read_csv('./train.csv')
  sortedtrain = trainset.sort_values(by='id')
  sortedtrain.index = sortedtrain['id']
  i = sortedtrain.index
  sortedtrain = sortedtrain.reindex(d4.index)
  sortedtrain = sortedtrain.join(d4)
  sortedtrain.reindex(i)
  sortedtrain = sortedtrain.sample(n=3000, replace=False)
  return sortedtrain.values
