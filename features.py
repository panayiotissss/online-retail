import pandas as pd
import numpy as np

def build_features(path:str):
#Load data from the path above
    df = pd.read_csv(path)

#Clean and transform columns
    df = df.rename(columns={'Customer ID': 'CustomerID'})
    df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    cap = df['Quantity'].quantile(0.99)
    df['Quantity'] = df['Quantity'].clip(upper=cap)


    #Engineer features
    df['TotalPrice'] = df['Quantity']*df['Price']
    reference_date = df['InvoiceDate'].max()

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


    skewed_cols = ['Monetary', 'Frequency', 'TotalItems', 'UniqueProducts',
                'AvgOrderValue', 'AvgQuantityPerOrder']

    for col in skewed_cols:
        rfm[col] = np.log1p(rfm[col])



#Return one row per customer
    return rfm


if __name__ == "__main__":
    df = build_features("data/online_retail_II.csv")
    print(df.shape)
    print(df.head())
