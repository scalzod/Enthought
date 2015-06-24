# -*- coding: utf-8 -*-
read_table("/Users/zod/Desktop/Enthought_June2015_course/materials/exercises/pandas/pandas_io/HistoricalS_and_P500.csv",  sep=",", index_col=0, parse_dates=True,)

df = read_table("/Users/zod/Desktop/Enthought_June2015_course/materials/exercises/pandas/pandas_io/HistoricalS_and_P500.csv", sep=",", index_col=0, parse_dates=True, converters={0:convert_date})
  
year = int(date[0:2])
  if 59 <= year <= 99:
      date = '19' + date
  else
      date = '20' + date
      
      
def convert_date(bytes_input):
    str_input = bytes_input.decode("utf-8")
    month, day, year = str_input.split("/")
    if int(year) < 50:
        new_year = "20" + year
    else:
        new_year = "19" + year
    return "/".join([month, day, new_year])
    
    

close = read_table("/Users/zod/Desktop/Enthought_June2015_course/materials/exercises/pandas/pandas_io/adj_close_stock_data_yahoo_2005_2010.txt", sep="\s+", parse_dates=[[0,1,2]], na_values=["-"], parse_dates[[0,1,2]], index_col=0)

from pandas import read_table, scatter_matrix
from pandas.io.data import get_data_yahoo, get_data_google, RemoteDataError


import numpy as np
from matplotlib.pyplot import colorbar, figure, imshow, show, title, xticks, \
    yticks
    
local_dataset = read_table("/Users/zod/Desktop/Enthought_June2015_course/materials/exercises/pandas/stock_returns/adj_close_stock_data_2005_2010.txt",
                           sep="\s+", parse_dates=[[0, 1, 2]], na_values="-")
local_dataset = local_dataset.set_index("year_month_day")
local_dataset.index.name = "Date"


import pandas as pd
import numpy as np
data = np.arange(12).reshape(4, 3)
df = pd.DataFrame(data,
     index=['one', 'two', 'three', 'four'],
     columns=['X', 'Y', 'Z'])
     
df_three = df.loc['three']
df.boxplot()
df_Y=df['Y']
df_Y_Z = df.


index = ['a', 'c','b', 'c', 'd']
df_2_4 =  df.iloc[1::2]
df_2_4 =  df.loc[['two','four']]
df_2_4 =  df.loc[['two','three'], ['Y','Z']]

index = ['a', 'c','b', 'c', 'd']
s = Series(range(5), index=index)
s2=Series(range(6), index=list('abccdc')
s.reindex(list(cbda), inplace=True)


import pandas.io.data as web
aapl =
web.get_data_yahoo('AAPL','1/1/2010')



1. Print the data from the last 4 weeks (see the .last method)
2. Extract the adjusted close column (“Adj Close”), resample the full data to a monthly period and plot. Do this 3 times, using the min, max, and mean of the resampling window.

aapl.resample('W', how='mean', ... closed='left', label='left')

df = DataFrame(randn(1000,5), columns=list('ABCDE'))

diffs = prices[1:] - prices[:-1]
returns = diffs / prices[:-1]

returns_shift = normalized_dataset / normalized_dataset.shift(1) - 1


#Person class

class Person(object):

def __initi__(self, first, last):
        self.first = first
        self.last = last
        
    def full_name(self):
        return '%s %s' % (self.first, self.last)    
        
           #or
           
#              def fullname(self):##               return self.first + ' ' + self.last   
    
    def __repr__(self):
        return '%s(%r,%r)' % (self.__class__.__name__, self.first, self.last)


#or

    def __repr__(self):
        return 'Person('{}', '{}')'.format(self.first, self.last)
                
    def __eq__(self, other):
        if not isinstance(other, Person):
                return Notimplemented
            return self.first == other.first and self.last == other.last




p = Person('eric', 'jones')
print p.full_name()
print p















