x = linspace(0,2*pi,50)
y1 = sin(x)
y2 = sin(2*x)
figure() # Create figure >>> hold(False)
plot(y1)
plot(x, y1)
# red dot-dash circle 
plot(x, y1, 'r-o')
# red marker only circle 
plot(x, y1, 'ro')
plot(x, y1, 'g-o', x, y2, 'b-+') 
clf() # Clear plot



         p(t) - p(t-1)
ret(t) = -------------
             p(t-1)


from numpy import arange, loadtxt, zeros
from matplotlib.pyplot import plot, title, show, xlim

prices = loadtxt("/Users/zod/Desktop/Enthought_June2015_course/materials/exercises/numpy/calc_return/aapl_2008_close_values.csv", usecols=[1], delimiter=",")

print("Prices for AAPL stock in 2008:")
print(prices)

diffs = prices[1:] - prices[:-1]
returns = diffs / prices[:-1]

return2 = (prices[1:] -prices[:-1]) / prices[:-1]

days = arange(len(returns))
zero_line = zeros(len(returns))

plot(days, zero_line, 'k-', days, returns * 100, 'b-')
title("Daily return of the AAPL stock in 2008 (%)")


xlim(xmax=len(returns))
show()


loadtxt

"/Users/zod/Desktop/Enthought_June2015_course/materials/exercises/numpy/
ary1 = loadtxt('/Users/zod/Desktop/Enthought_June2015_course/materials/exercises/numpy/load_text/float_data.txt')

print('example 1:')
print(ary1)

ary2 = loadtxt('/Users/zod/Desktop/Enthought_June2015_course/materials/exercises/numpy/load_text/float_data_with_header.txt', skiprows=1)

print('example 2:')
print(ary2)

ary3 = loadtxt('/Users/zod/Desktop/Enthought_June2015_course/materials/exercises/numpy/load_text/complex_data_file.txt', delimiter=",", comments="%",
               usecols=(0, 1, 2, 4), dtype=int, skiprows=1)

print('example 3:')
print(ary3)



wind data

wind_data = loadtxt('/Users/zod/Desktop/Enthought_June2015_course/materials/exercises/numpy/wind_statistics/wind.data')


#min, max mean SD all
data = wind_data[:,3:]
min_data = data.min()
max_data = data.max()
mean_data = data.mean()
std_data = data.std()

#min, max and mean windspeeds and standard deviations by location

loc_min = data.min(axis=0)
loc_max= data.max(axis=0)
loc_mean = data.mean(axis=0)
loc_std = data.std(axis=0)

#max and mean windspeed and standard deviations by day
day_min = data.min(axis=1)
day_max = data.max(axis=1)
day_mean = data.mean(axis=1)
day_std = data.std(axis=1)

#location of daily max
data.argmax(axis=1)
daily_max = data.max(axis=1)
max_row = daily_max.argmax()
[sum(daily_max ==) for station in range(12)


#day of max reading
year = int(wind_data[max_row, 0]))
month = int(wind_data[max_row, 1]))
day = int(wind_data[max_row, 2]))

import unravel_index
unravel_index(data.argmax(), data.shape)


#MPL exercise
import numpy as np
import matplotlib.pyplot as plt

# Create some sample data
# generate 100 numbers between 0 and 10
x1 = np.linspace(0 ,10, 100)   
# element-wise sin of each element in x1
y1 = np.sin(x1)                
# generate random floats in [0.0, 5.0)
e1 = np.random.ranf(100) * 5.0 
# make a noisy sin wave
y1_noisy = y1 + e1             
 # generate 100 numbers between 0 and 10
x2 = np.linspace(0, 10, 100)  
 # element-wise sin of each element in x2
y2 = np.sin(x2)               
# generate noise with amplitude 3.0
e2 = np.random.ranf(100)*3.0   
 # constant offset
offset = 3.0                  
 # make a noisy sin wav offset from the first one
y2_noisy = y2 + offset + e2   

# 1. Create a new figure.
plt.figure()

# 2. Create three columns and two rows of subplots and select the first one.
plt.subplot(2,3,1)

# 3. In the top, left subplot, plot y1_noisy using a line plot with black
#    triangle markers.
plt.plot(y1_noisy, 'k-^')

# 4. Overwrite the top left subplot with a plot of y1_noisy using a line plot
#    with a red dashed line.
plt.hold(False)
plt.plot(y1_noisy, 'r--')

# 5. In the top, middle subplot, plot y2_noisy using a line plot with
#    green circle markers.
plt.subplot(2,3,2)
plt.plot(y2_noisy, 'g-o')

# 6. In the top, right subplot, plot y1_noisy vs. y2_noisy using a scatter plot
#    with blue circles.
plt.subplot(2,3,3)
plt.scatter(y1_noisy, y2_noisy)

# 6. In the bottom, left subplot, plot the correlation matrix of
#        x1, y1, e1, y1_noisy, x2, y2, e2, y2_noisy
#    as an image.
#    Hint: syntax for correlation matrix is
#        result = np.corrcoef([arr1, arr2, arr3, ...])
corrmatrix = np.corrcoef([x1, y1, e1, y1_noisy, x2, y2, e2, y2_noisy])
plt.subplot(2,3,4)
plt.imshow(corrmatrix, interpolation='nearest')

# 7. Add a colorbar to the subplot.
plt.colorbar()

# 8. In the bottom, middle subplot, plot the histograms side by side for
#    y1_noisy and y2_noisy (set hold to True again first).
plt.hold(True)
plt.subplot(2,3,5)
plt.hist([y1_noisy, y2_noisy])

# 9. In the bottom, right subplot, plot the histogram of (y2_noisy - y1_noisy)
plt.subplot(2,3,6)
plt.hist(y2_noisy - y1_noisy)

# 10. Clear the figure.
plt.clf()

plt.show()

#statistics for January
january_indices = wind_data[:, 1] == 1
january_data = data[january_indices]
january_data.mean(axis=0))

wind_data[month ==, :].mean(axis=0)


#Dow selection
from numpy import loadtxt, sum, where
from matplotlib.pyplot import figure, hold, plot, show

# Constants that indicate what data is held in each column of
# the 'dow' array.
OPEN = 0
HIGH = 1
LOW = 2
CLOSE = 3
VOLUME = 4
ADJ_CLOSE = 5

# 0. The data has been loaded from a csv file for you.

# 'dow' is our NumPy array that we will manipulate.
dow = loadtxt('/Users/zod/Desktop/Enthought_June2015_course/materials/exercises/numpy/dow_selection/dow.csv', delimiter=',')


# 1. Create a "mask" array that indicates which rows have a volume
#    greater than 5.5 billion.
high_volume_mask = dow[:, VOLUME] > 5.5e9

# 2. How many are there?  (hint: use sum).
high_volume_days = sum(high_volume_mask)
print "The dow volume has been above 5.5 billion on" \
      " %d days this year." % high_volume_days

# 3. Find the index of every row (or day) where the volume is greater
#    than 5.5 billion. hint: look at the where() command.
high_vol_index = where(high_volume_mask)[0]

# BONUS:
# 1. Plot the adjusted close for EVERY day in 2008.
# 2. Now over-plot this plot with a 'red dot' marker for every
#    day where the dow was greater than 5.5 billion.

# Create a new plot.
figure()

# Plot the adjusted close for every day of the year as a blue line.
# In the format string 'b-', 'b' means blue and '-' indicates a line.
plot(dow[:, ADJ_CLOSE], 'b-')

# Plot the days where the volume was high with red dots...
plot(high_vol_index, dow[high_vol_index, ADJ_CLOSE], 'ro')

# Scripts must call the plot "show" command to display the plot
# to the screen.
show()
