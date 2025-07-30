import sys
import os

# 设置项目根目录为当前 notebook 所在路径的上一级目录
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from src.data_load import load_ocrd, load_ordr, load_rdr1, load_odln, load_oitm



def data_preview(df):

    df["BrandName"] = df["CardCode"]
    df["DocDate"] = pd.to_datetime(df["DocDate"], format="%m/%d/%Y")
    df["Year"] = df["DocDate"].dt.year
    df["Month"] = df["DocDate"].dt.month

    df = df[
        (df["Year"]==2024) & 
        (df["Month"]==5)   &
        (df["DocTotal"] !=0.00)
    ]
    
    return df



def clean_order_date(df):
    df = df.copy()
    df["DocDate"] = pd.to_datetime(df["DocDate"],format="%m/%d/%Y")
    df["Year"] = df["DocDate"].dt.year
    df = df[df["Year"] != 2020]
    return df

def grouped_rev_year(df):
    df = df.copy()
    df["DocDate"] = pd.to_datetime(df["DocDate"],format="%m/%d/%Y")
    df["Year"] = df["DocDate"].dt.year
    df = df[df["Year"] != 2020]

    df_groupby = (
        df.groupby(["Year"])["DocTotal"]
        .sum()
        .reset_index(name="rev")
    )
    return df_groupby

def grouped_itemcate_rev(df):
    df = df.copy()
    df["DocDate"] = pd.to_datetime(df["DocDate"],format="%m/%d/%Y")
    df["Year"] = df["DocDate"].dt.year
    df["LineTotal"] = pd.to_numeric(df["LineTotal"], errors="coerce")
    df = df[df["LineTotal"].notnull()]
    df = df[df["Year"] != 2020]
    df = (
        df.groupby(["Category","Year"])["LineTotal"]
        .sum()
        .reset_index(name="rev")
    )
    return df

def grouped_revenue_cust(df):
    df = df.copy()
    df["DocDate"] = pd.to_datetime(df["DocDate"],format="%m/%d/%Y")
    df["Year"] = df["DocDate"].dt.year
    df = df[df["Year"] != 2020]
    df["BrandName"] = df["CardCode"]
    df = (
        df.groupby(["BrandName","Year"])["DocTotal"]
        .sum()
        .reset_index(name="rev")
        .sort_values(by=["Year","rev"],ascending=[True,False])
    )
    return df

def grouped_orderval_year(df):
    df = df.copy()
    df["DocDate"] = pd.to_datetime(df["DocDate"],format="%m/%d/%Y")
    df["Year"] = df["DocDate"].dt.year
    df = df[df["Year"] != 2020]

    df = (
        df.groupby(["Year"])
        .size()
        .reset_index(name="order_valume")
    )
    return df

def grouped_orderval_month(df):
    df = df.copy()
    df["DocDate"] = pd.to_datetime(df["DocDate"],format="%m/%d/%Y")
    df["Year"] = df["DocDate"].dt.year
    df["Month"] = df["DocDate"].dt.month
    df = df[df["Year"] != 2020]

    df = (
        df.groupby(["Year","Month"])
        .size()
        .reset_index(name="order_valume")
    )
    return df

def grouped_features_month(df):
    df = df.copy()
    df["DocDate"] = pd.to_datetime(df["DocDate"],format="%m/%d/%Y").dt.to_period("M")
    df = df[df["DocDate"]!='2020-8']
    df["Year"] = df["DocDate"].dt.year
    df["Month"] = df["DocDate"].dt.month
    df["BrandName"] = df["CardCode"]

    #print(df["LineTotal"].apply(type).value_counts())

    df = df.groupby("DocDate").agg(
            order_count = ("DocNum", "nunique"),
            sku_count = ("ItemCode", "nunique"),
            cust_count = ("BrandName", "nunique"),    
            rev = ("LineTotal", "sum")
        ).reset_index()
    df["Month"] = df["DocDate"].dt.month
    df["Year"] = df["DocDate"].dt.year
    return df


'''
df_load_ordr = load_ordr()
df_load_rdr1 = load_rdr1()
df_merged = pd.merge(df_load_ordr,df_load_rdr1, on="DocEntry", how="inner")

print(grouped_features_month(df_merged))'''