import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy.stats as stats
print( 'The scikit learn version is {}.'.format(sklearn.__version__))
path = "C:/Users/Vedant/Desktop/UIUC/SEM 1/Machine Learning/Week 3/HY_Universe_corporate bond.csv"
treasury = pd.read_csv(path)
treasury = treasury.values
print(treasury)
print(("Number of Rows of Data = " + str(len(treasury)) + '\n'))
n_rows=len(treasury)
n_col=0
for column in treasury[0,:]:
    n_col=n_col+1
print("Number of columns of Data = " , n_col , '\n')



type = [0]*3
colCounts = []
for col in range(n_col):
 for row in treasury:
     try:
         a = float(row[col])
         if isinstance(a, float):
             type[0] += 1
     except ValueError:
         if len(row[col]) > 0:
             type[1] += 1
         else:
             type[2] += 1
 colCounts.append(type)
 type = [0]*3
print("Col#" + '\t' + "Number" + '\t' +"Strings" + '\t ' + "Other\n")
iCol = 0
for types in colCounts:
 print(str(iCol) + '\t\t' + str(types[0]) + '\t\t' +
 str(types[1]) + '\t\t' + str(types[2]) + "\n")
 iCol += 1
 
 
 
 type = [0]*3
colCounts = []
#generate summary statistics for column 9 (e.g.)
col = 9
colData = []
for row in treasury:
 colData.append(float(row[col]))
colArray = np.array(colData)

colMean = np.mean(colArray)
colsd = np.std(colArray)
print("Mean = " + '\t' + str(colMean) + '\t\t' + "Standard Deviation = " + '\t ' + str(colsd) + "\n")

#calculate quantile boundaries
ntiles = 4
percentBdry = []
for i in range(ntiles+1):
 percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
print("\nBoundaries for 4 Equal Percentiles \n")
print(percentBdry)
print(" \n")

#run again with 10 equal intervals
ntiles = 10
percentBdry = []
for i in range(ntiles+1):
 percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
print("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
print(" \n")

#The last column contains categorical variables
col = 14
colData = []
for row in treasury:
 colData.append(row[col])
unique = set(colData)
print("Unique Label Values \n")
print(unique)

#count up the number of elements having each value
catDict = dict(zip(list(unique),range(len(unique))))
catCount = [0]*len(unique)
for elt in colData:
  catCount[catDict[elt]] += 1
print("\nCounts for Each Value of Categorical Label \n")
print(list(unique))
print(catCount)



#generate summary statistics for column 10 (e.g.)
col_10 = 10
colData_10 = []
for row in treasury[:,10]:
 colData_10.append(float(row))
 
stats.probplot(colData_10, dist="norm", plot=pylab)
plt.title("Probability plot of Amount issued")
pylab.show()
 


#read  data into pandas data frame
findata = pd.read_csv(path)
#print head and tail of data frame
print(findata.head())
print(findata.tail())
#print summary of data frame
summary = findata.describe()
print("Summary of the data \n" ,summary)
findata=findata.values
#calculate correlations between real-valued attributes
dataCol27 =findata[1:,27]
dataCol28 = findata[1:,28]
plt.scatter(dataCol27, dataCol28)
plt.xlabel("28th Attribute")
plt.ylabel(("29th Attribute"))
plt.show()
dataCol36 = findata[1:,36]
plt.scatter(dataCol27, dataCol36)
plt.xlabel("28th Attribute(Liq Scores)")
plt.ylabel(("36th Attribute (volume_trades)"))
plt.show()


#change the targets to numeric values
target = []
for i in range(2721):
 #assign 0 or 1 target value based on "Yes" or "No" labels
 if findata[i,19] == "Yes":
     target.append(1.0)
 else:
     target.append(0.0)
#plot 35th attribute
dataRow = findata[:,19]
plt.scatter(target,dataRow)
plt.xlabel("Attribute Value")
plt.ylabel("Target Value")
plt.show()

#calculate correlations between real-valued attributes
dataCol10 = findata[:,10]
dataCol15 = findata[:,15]
dataCol20 = findata[:,20]
mean10 = 0.0; mean15 = 0.0; mean20 = 0.0
numElt = len(dataCol10)
for i in range(numElt):
 mean10 += dataCol10[i]/numElt
 mean15 += dataCol15/numElt
 mean20 += dataCol20[i]/numElt
var10 = 0.0; var15 = 0.0; var20 = 0.0
for i in range(numElt):
 var10 += (dataCol10[i] - mean10) * (dataCol10[i] - mean10)/numElt
 var15 += (dataCol15[i] - mean15) * (dataCol15[i] - mean15)/numElt
 var20 += (dataCol20[i] - mean20) * (dataCol20[i] - mean20)/numElt
corr1015 = 0.0; corr1020 = 0.0
for i in range(numElt):
 corr1015 += (dataCol10[i] - mean10) * \
 (dataCol15[i] - mean15) / (((var10*var15)**0.5) * numElt)
 corr1020 += (dataCol10[i] - mean10) * \
 (dataCol15[i] - mean15) / (((var10*var15)**0.5) * numElt)
print("Correlation between attribute 10 and 15 \n")
print(corr1015.mean())
print(" \n")
print("Correlation between attribute 2 and 21 \n")
print(corr1020.mean())
print(" \n")

from pandas import DataFrame
#read data into pandas data frame
findata = pd.read_csv(path)
#calculate correlations between real-valued attributes
corMat = DataFrame(findata.corr())
#visualize correlations using heatmap
plt.pcolor(corMat)
plt.show()


import seaborn as sns
sns.stripplot(x=findata['IN_ETF'],y=findata['Issued Amount'])
plt.ylabel('Amount')
plt.xlabel('IN ETF')
plt.legend()
plt.show()


findata=findata.values
from scipy.stats import skew
plt.style.use('ggplot')
plt.hist(findata[:,10])
plt.xlabel("Amount Issued")
plt.ylabel("Frequency")
plt.show()
print("The skewness in the amount issued is ", skew(findata[:,10]))





