# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:08:55 2020

A script that pulls from EDDB to calculate the best TritiumTrade
for the day

@author: Schneiderpi
"""

import requests
import os
import datetime
import json
import pandas
import numpy as np
from tqdm import tqdm

import dask.dataframe as dd

eddb_api_url = "https://eddb.io/archive/v6/"
min_landing_pad = "L"

pandas.set_option('mode.chained_assignment',None)

def pull_from_eddb(data_path,filename):
  print("Pulling {} from eddb".format(filename))
  
  r = requests.get(eddb_api_url+filename)
  
  r.raise_for_status()
  
  with open(data_path+filename, 'wb') as f:
    for chunk in r.iter_content(chunk_size=128):
      f.write(chunk)

def need_to_pull(data_path,filename):
  if not os.path.exists(data_path+filename):
    return True
  else:
    last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(data_path+filename))
    
    today = datetime.date.today()
    midnight = datetime.datetime.combine(today, datetime.time.min)
    
    if last_modified < midnight:
      return True
  
  return False

def read_data(data_path,filename):
  print("Reading {}".format(data_path+filename))
  with open(data_path+filename, 'r') as f:
    if filename.endswith(".json"):
      return json.loads(f.read())
    elif filename.endswith(".csv"):
      return pandas.read_csv(f,header=0)

def get_distance(x1,y1,z1,x2,y2,z2):
  return np.sqrt(np.power(np.abs(x1-x2),2)+np.power(np.abs(y1-y2),2)+np.power(np.abs(z1-z2),2))

def cartesian_product_simplified(left, right):
  #Stolen from https://stackoverflow.com/questions/53699012/performant-cartesian-product-cross-join-with-pandas
  la, lb = len(left), len(right)
  ia2, ib2 = np.broadcast_arrays(*np.ogrid[:la,:lb])

  return pandas.DataFrame(
      np.column_stack([left.values[ia2.ravel()], right.values[ib2.ravel()]]))

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)
  
def cartesian_product_multi(*dfs):
    idx = cartesian_product(*[np.ogrid[:len(df)] for df in dfs])
    return pandas.DataFrame(
        np.column_stack([df.values[idx[:,i]] for i,df in enumerate(dfs)]))  
def main():
  tqdm.pandas()
  
  jump_range = 21
  data_path = "data/"
  
  if not data_path[-1] == "/" and not data_path[-1] == "\\":
    data_path = data_path + "/"

  if not os.path.exists(data_path):
    os.makedirs(data_path)
  
  filenames = ["systems_populated.json","listings.csv","commodities.json","stations.json"]
  
  for name in filenames:
    if need_to_pull(data_path,name):
      pull_from_eddb(data_path,name)
  
  commodities = read_data(data_path,"commodities.json")
  prices = read_data(data_path,"listings.csv")
  systems = read_data(data_path,"systems_populated.json")
  stations = read_data(data_path,"stations.json")
  
  #print("Finding the Tritium Commodity ID")
  #for commodity in commodities:
  #  if commodity["name"] == "Tritium":
  #    tritium_id = commodity["id"]
  #    break
  
  filtered_trades_stations = {}
  
  result = None
  print("Trimming stations to only those that we can land at and have markets")
  for station in stations:
    if station["has_market"] and station["max_landing_pad_size"]=="L" and not station["type"] == "Fleet Carrier":      
          station_id = station["id"]
          system_id = station["system_id"]
          
          if station_id == 3:
            print(station)
          filtered_trades_stations[station_id] = {"station_id": station_id,"system_id": system_id,"station_name":station["name"]}
  
  systems_df = pandas.from_dict(systems)
  systems_df = systems_df[["id","name","x","y","z"]]
  
  filtered_stations_df = pandas.from_dict(filtered_trades_stations,orient="index")
  
  print("Joining systems and stations")
  filtered_stations_systems = filtered_stations_df.merge(systems_df,how='left',left_on='system_id',right_on='id')
  filtered_stations_systems.drop("id",inplace=True,axis=1)

  print("Joining stations and prices")
  prices = prices[["station_id","commodity_id","supply","buy_price","sell_price","demand"]]
  prices = prices.merge(filtered_stations_systems,how='right',left_on='station_id',right_on='station_id',suffixes=("_prices","_systems")).apply(lambda x: x)
  prices.dropna(how='any',inplace=True)
  
  prices = prices.astype({"station_id":"uint32","commodity_id":"uint16","supply":"uint32","buy_price":"uint64","sell_price":"uint64","demand":"uint32","system_id":"uint64","x":"float32","y":"float32","z":"float32"})
  
  for commodity in tqdm(commodities,desc='Commodities'):
    commodity_id = commodity['id']

    print("Filtering prices")
    stations_prices = prices[(prices['station_id'].isin(filtered_trades_stations.keys()))&(prices['commodity_id']==commodity_id)]
    
    station_prices_buy = stations_prices[(stations_prices['buy_price']>0)&(stations_prices['supply']>5000)]
    station_prices_sell = stations_prices[(stations_prices['sell_price']>0)&(stations_prices['demand']>0)]
      
    if not station_prices_buy.empty and not station_prices_sell.empty:
      print("Joining buy and sell stations")
      print(station_prices_buy.info())
      print(station_prices_sell.info())
      
      #station_prices_diff = station_prices_buy.merge(station_prices_sell,how='left',on='join_key',suffixes=('_buy','_sell'))
      station_prices_diff =  cartesian_product_multi(*[station_prices_buy,station_prices_sell])
      
      #station_prices_diff.columns = stations_prices.columns
      print(station_prices_diff.info())
      print(station_prices_diff)
      print("Getting profit for each route")
      station_prices_diff['profit'] = station_prices_diff['sell_price_sell'] - station_prices_diff['buy_price_buy']
      
      print("Getting distance of each route")
      station_prices_diff['route_distance'] = station_prices_diff.progress_apply(
          lambda row: get_distance(row["x_buy"],row["y_buy"],row["z_buy"],row["x_sell"],row["y_sell"],row["z_sell"]),axis=1)
      
      print("Getting jumps per route")
      station_prices_diff['route_jumps'] = np.ceil(station_prices_diff['route_distance'] / jump_range)
      
      print("Getting profit/jump")
      
      station_prices_diff[station_prices_diff['route_jumps']==0]=1
      station_prices_diff['profit_per_jump'] = station_prices_diff['profit'] / station_prices_diff['route_jumps']
      station_prices_diff.sort_values(by='profit_per_jump',ascending=False,inplace=True)
      
      station_prices_diff['commodity_id'] = commodity_id
      
      if result is None:
        result = station_prices_diff
      else:
        result = result.append(station_prices_diff)
  
  result.sort_values(by='profit_per_jump',ascending=False,inplace=True)
  
  commodities_df = pandas.DataFrame.from_dict(commodities)
  result = result.merge(commodities_df,how='left',left_on='commodity_id',right_on='id',suffixes=('_result','_commodity'))
  
  pandas.set_option('display.max_columns',None)
  print(result[['commodity_name','station_name_buy','name_buy','station_name_sell','name_sell','profit_per_jump','buy_price_buy','sell_price_sell','supply_buy','demand_sell']].head(10))
  
if __name__ == '__main__':
  main()