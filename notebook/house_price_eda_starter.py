import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/raw/Bengaluru_House_Data.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe(include="all"))

if "price" in df.columns:
    sns.histplot(df["price"].dropna(), kde=True)
    plt.title("House Price Distribution")
    plt.xlabel("Price (lakh rupees)")
    plt.show()

if {"total_sqft", "price"}.issubset(df.columns):
    plot_df = df[["total_sqft", "price"]].copy()
    plot_df["total_sqft"] = pd.to_numeric(plot_df["total_sqft"], errors="coerce")
    plot_df = plot_df.dropna()
    sns.scatterplot(data=plot_df, x="total_sqft", y="price", alpha=0.5)
    plt.title("Area vs House Price")
    plt.show()
