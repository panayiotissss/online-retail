import pandas as pd
import numpy as np


df = pd.read_csv("data/online_retail_II.csv")
#print(df.head())
# print(df.shape)
# print(df.dtypes)
# print(df.isnull().sum()) # Have 4382 descriptions missing and 243007 customers ids , since i am groupping with customer ids this can go descriptions useless anyway
# print(df['Country'].unique())

df = df.rename(columns={'Customer ID': 'CustomerID'})
df = df.dropna(subset=['CustomerID'])
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

#print(df.describe())

df = df[df['Quantity'] > 0]
df = df[df['Price'] > 0]


# print(df[df['Quantity'] > 1000].shape)
# print(df[df['Quantity'] > 10000][['CustomerID', 'Quantity', 'Price', 'Description']].head(20))

cap = df['Quantity'].quantile(0.99)
#print(f"99th percentile cap: {cap}")
df['Quantity'] = df['Quantity'].clip(upper=cap)

#print(df.head())

df['TotalPrice'] = df['Quantity']*df['Price']
#print(df['TotalPrice'])

reference_date = df['InvoiceDate'].max()
#print(reference_date)

rfm = df.groupby('CustomerID').agg(
    LastPurchaseDate =('InvoiceDate','max'),
    Frequency        = ('Invoice', 'nunique'),
    FirstPurchaseDate   = ('InvoiceDate', 'min'),
    Monetary         = ('TotalPrice', 'sum'),
    TotalItems       = ('Quantity', 'sum'),
    UniqueProducts = ('StockCode', 'nunique')                        
)

rfm['Recency']              = (reference_date - rfm['LastPurchaseDate']).dt.days
rfm['Tenure']               = (rfm['LastPurchaseDate'] - rfm['FirstPurchaseDate']).dt.days
rfm['AvgOrderValue']        = rfm['Monetary'] / rfm['Frequency']
rfm['AvgQuantityPerOrder']  = rfm['TotalItems'] / rfm['Frequency']
rfm['AvgDaysBetweenOrders'] = rfm['Tenure'] / rfm['Frequency']

rfm = rfm.drop(columns=['LastPurchaseDate', 'FirstPurchaseDate'])

print(rfm.head())
#print(rfm.describe())



skewed_cols = ['Monetary', 'Frequency', 'TotalItems', 'UniqueProducts',
               'AvgOrderValue', 'AvgQuantityPerOrder']

for col in skewed_cols:
    rfm[col] = np.log1p(rfm[col])



print(rfm.describe().loc['max'])
