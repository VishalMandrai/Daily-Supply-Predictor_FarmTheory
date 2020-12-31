
import pandas as pd
import numpy as np
import collections as Counter
from datetime import date


pd.options.display.max_columns = None



df = pd.read_csv("product_supply.csv")

## Editing our column names for ease in use.
columns = []
for i in df.columns.values:
    m = str(i)
    columns.append(m.replace(' ' , '_'))
    
df.columns = columns


## Our required Features....
df = df[["Invoice_Date" , 'Customer_Name', 'Customer_ID','Item_Name', 'Product_ID','Quantity', 'Item_Price']]


## Dropping all NAN Entries....
df.dropna(axis = 0 , inplace = True)

df.shape[0]



import datetime

Weekdays = []

for temp in range(df.shape[0]):
    dt = df['Invoice_Date'].iloc[temp]
    year, month, day = (int(x) for x in dt.split('-'))    
    ans = datetime.date(year, month, day)
    Weekdays.append(ans.strftime("%A"))



df['Weekday'] = Weekdays


## Moving all "Items" to columns and making different DataFrames for each "Customer"....

items_dummy = pd.get_dummies(df['Item_Name'] , prefix_sep = ' ' , drop_first = False)


## Editing our column names for ease in use.....
columns_temp = []
for i in items_dummy.columns.values:
    n = str(i)
    columns_temp.append(n.replace(' ' , '_'))
    
items_dummy.columns = columns_temp



items_dummy.mask(items_dummy < 1 , inplace = True)


## Joining the DataFrame together...
df = pd.concat([df[["Invoice_Date" , "Weekday" , 'Customer_Name', 'Customer_ID','Item_Name', 'Product_ID']] , items_dummy , 
                df[['Quantity', 'Item_Price']]] , axis = 1)

df.head(3)


## Creating the list of all Unique Customers....
Customers = list(df['Customer_Name'].unique())
print("No. of Unique customers are:" , len(Customers))


## Creating the list of all Unique Items sold by FarmTheory....
Items = list(df['Item_Name'].unique())
Items = sorted(Items)
print("No. of Unique Items are:" , len(Items))


## Passing all values under "Quantity" feature to all relevant "Item Name" 
## column..... 
for i in items_dummy.columns.values:
    df[i] = df[i]*df["Quantity"]


# ### ------------------------------------------------------------------------

df['Weekday'] = df['Weekday'].map({"Sunday":0 , "Monday":1 , "Tuesday":2 , 
  "Wednesday":3 ,"Thursday":4 , "Friday":5 , "Saturday":6})



from sklearn.tree import DecisionTreeRegressor
import pickle

for j in items_dummy.columns.values:
    temp = df[["Customer_ID","Weekday" , j]]
    temp.dropna(axis = 0 , inplace = True)

    if temp.empty == True:
        pass
    else:
        y = temp[j]
        y = y.values.reshape(-1,1)
        DT = DecisionTreeRegressor()
        DT.fit(temp[["Customer_ID","Weekday"]] , y)
        
        # Saving MODEL.....
        pickle.dump(DT, open("DT_"+j+'.pkl' , 'wb'))
        DT = pickle.load(open("DT_"+j+'.pkl' , 'rb'))       



pickle.dump(items_dummy.columns.values , open("Item_list.pkl" , "wb"))

Items = pickle.load(open("Item_list.pkl" , 'rb'))   