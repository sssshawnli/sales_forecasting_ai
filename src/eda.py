import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
#from data_load import load_ocrd, load_ordr, load_rdr1, load_odln, load_oitm

# Set display options for better readability in console output
pd.set_option("display.width", None)
pd.set_option("display.float_format", '{:,.2f}'.format)  # Format floats with 2 decimal places and comma separators

# ---------------------------------------------
# Plot customer count distribution by country
# ---------------------------------------------
def plot_customer_count_by_country(df):
    country_counts = df["Country"].value_counts()
    plt.clf()  # Clear previous figure
    plt.figure(figsize=(10,5))
    country_counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Customer Distribution by Country")
    plt.yscale("log")  # Use log scale for skewed data
    plt.ylabel("Number of Customers")
    plt.xlabel("Country")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------
# Revenue and customer share by country 
# -----------------------------------------------
def plot_cusomter_revenue_persent(df):
    # Group by customer to calculate total revenue
    df_cust_group_by_rev =(
        df.groupby(["CardCode","Country"])["DocTotal"]
        .sum()
        .reset_index(name="cust_rev")
        .sort_values(by="CardCode",ascending=False)
    )

    # Clean country field: classify US vs others
    df_cust_group_by_rev["Country_clean"] = np.where(
        df_cust_group_by_rev["Country"].str.contains("US",case=False,na=False),
        'US',
        'Others'
    )

    # Compute revenue per country
    df_country_rev = (
        df_cust_group_by_rev
        .groupby("Country_clean")["cust_rev"]
        .sum()
        .reset_index(name="country_rev")
        .sort_values(by="country_rev", ascending=False)
    )

    # Compute customer count per country
    df_cust_count = (
        df_cust_group_by_rev.groupby("Country_clean")
        .size()
        .reset_index(name="country_size")
    )

    # Merge revenue and count data
    df_combine = pd.merge(
        df_country_rev,
        df_cust_count,
        on="Country_clean",
        how="inner"
    )

    # Calculate share in percent
    df_combine["total_rev"] = df_combine["country_rev"].sum()
    df_combine["country_rev_per"] = df_combine["country_rev"] / df_combine ["total_rev"]
    df_combine["total_country_count"] = df_combine["country_size"].sum()
    df_combine["country_count_per"] = df_combine["country_size"] / df_combine["total_country_count"]

    # Plot two pie charts side by side
    fig,axes = plt.subplots(1,2, figsize=(12,6))
    labels = df_combine["Country_clean"]
    #sizes = df_combine[""]

    # Revenue share pie chart
    axes[0].pie(df_combine["country_rev_per"], labels=labels, autopct='%1.1f%%', startangle=90)
    axes[0].set_title("Revenue %")
    
    # Customer count share pie chart
    axes[1].pie(df_combine["country_count_per"], labels=labels, autopct='%1.1f%%', startangle=90)
    axes[1].set_title("Customer Count %")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------
# Plot top customers by revenue, grouped by year
# ---------------------------------------------
def plot_customer_sort_by_revenue(df):

    # 
    df["BrandName"] = df["CardCode"]
    df["DocDate"] = pd.to_datetime(df["DocDate"], format="%m/%d/%Y")
    df["Year"] = df["DocDate"].dt.year 

    # Get top 15 customers by total revenue
    top_brands = (
        df.groupby("BrandName")["DocTotal"].sum()
        .sort_values(ascending=False)
        .head(15)
        .index
    )

    # Aggregate revenue by brand and year
    df_grouped = ( 
        df[df["BrandName"].isin(top_brands)]
        .groupby(["BrandName","Year"])["DocTotal"]
        .sum()
        .reset_index(name="TotalAmount")
        .sort_values(by="TotalAmount",ascending=False)
    )

    plt.figure(figsize=(20,6))

  
    sns.barplot(
        data = df_grouped,
        x='BrandName',
        y='TotalAmount',
        hue='Year',
        palette="tab10"
    )

    # Format y-axis with commas
    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{int(x):,}')
    )
    plt.title("Top Customers by Total Revenue (Grouped by Year)")
    plt.xlabel("Customer")
    plt.ylabel("Total Amount")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    #return df


# ---------------------------------------------
# Compute and plot revenue percentage for top brands by year (Pareto chart logic coming next def)
# ---------------------------------------------
def calculate_brand_revenue_percent(df, cutoff=90): 
    """
    Calculates brand-wise revenue percentage and cumulative share per year.
    Returns a dictionary of yearly DataFrames, each with top brands + 'Others'.
    """
    df["BrandName"] = df["CardCode"]
    df["DocDate"] = pd.to_datetime(df["DocDate"], format="%m/%d/%Y")
    df["Year"] = df["DocDate"].dt.year 

    # Total revenue per brand-year
    df_brand_year= (
        df.groupby(["BrandName","Year"])["DocTotal"]
        .sum()
        .sort_values(ascending=False)
        .reset_index(name="TotalAmount")
    )
    top_brands = (
        df.groupby("BrandName")["DocTotal"]
        .sum()
        .sort_values(ascending=False)
        #.head(5)
        .index
    )
    df_brand_year= df_brand_year[df_brand_year["BrandName"].isin(top_brands)] 

    # Total revenue per year
    df_year_total = (
        df.groupby("Year")["DocTotal"]
        .sum()
        .reset_index(name="YearTotal")
    )

    # Merge brand revenue with total per year
    df_percent = pd.merge(df_brand_year,df_year_total, on="Year")
    df_percent["Percent"] = df_percent["TotalAmount"] / df_percent["YearTotal"] * 100
    df_percent = df_percent.sort_values(by=["Year", "TotalAmount"], ascending=[True,False])
    df_percent["Cum"] = df_percent.groupby("Year")["Percent"].cumsum()
      
    
    # Split each year's data into top + others
    year_combs = {}
    for year in sorted(df_percent["Year"].unique()):
        df_year = df_percent[df_percent["Year"] == year]
        df_top = df_year[df_year["Cum"] <= cutoff].copy()
        df_rest = df_year[df_year["Cum"] > cutoff].copy()

        if not df_rest.empty:
            others = pd.DataFrame({
                "BrandName": ["Others"],
                "TotalAmount": [df_rest["TotalAmount"].sum()],
                "YearTotal": [df_rest["YearTotal"].iloc[0]],
                "Cum": [100.00]
            })
            df_comb = pd.concat([df_top, others], ignore_index=True)
        else:
            df_comb = df_top

        year_combs[year] = df_comb[["BrandName", "TotalAmount", "YearTotal", "Cum"]]

    return year_combs

def plot_brand_pareto_chart(df_comb, year):
    """
    Plots a Pareto chart for a single year's top brands.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(df_comb["BrandName"], df_comb["TotalAmount"], color="skyblue", label="Revenue")
    ax1.set_ylabel("Revenue")
    ax1.tick_params(axis='x', rotation=90)
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    ax2 = ax1.twinx()
    ax2.plot(df_comb["BrandName"], df_comb["Cum"], color="red", marker='o', label="Cumulative %")
    ax2.axhline(90, color='gray', linestyle='--', label="90% Cutoff")
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 105)

    fig.suptitle(f"Top Brands by {year} Revenue with 90% Cumulative Cutoff")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
    plt.tight_layout()
    plt.grid(True)
    plt.show()


# ---------------------------------------------
# Calculate and plot top item categories by revenue (Pareto logic)
# ---------------------------------------------
def plot_item_sort_revenue(df,cutoff=95.00):

    df["DocDate"] = pd.to_datetime(df["DocDate"], format="%m/%d/%Y")
    df["Year"] = df["DocDate"].dt.year 
    df["LineTotal"] = pd.to_numeric(df["LineTotal"], errors="coerce")
    df = df[df["LineTotal"].notnull()]

    df_items_year = (
        df.groupby(["Category","Year"])["LineTotal"]
        .sum()
        .sort_values(ascending=False)
        .reset_index(name="Year_Cate_Total")
    )
    df_year_total = (
        df.groupby("Year")["LineTotal"]
        .sum()
        .reset_index(name="Year_Total")
    )

    df_comb = pd.merge(df_items_year,df_year_total,on="Year",how="left")
    df_comb["Percent"] = df_comb["Year_Cate_Total"] / df_comb["Year_Total"] * 100
    df_comb = df_comb.sort_values(by=["Year","Year_Cate_Total"],ascending=[True,False])
    df_comb["Cum"] = df_comb.groupby(["Year"])["Percent"].cumsum()

    year_combs = {}
    for years in sorted(df_comb["Year"].unique()):
        df_top = df_comb[(df_comb["Year"]==years)&(df_comb["Cum"]<=cutoff).copy()]
        df_rest = df_comb[(df_comb["Year"]==years)&(df_comb["Cum"]>cutoff).copy()]
        if not df_rest.empty:
            df_others = pd.DataFrame({
            "Category": ["Others"],
            "Year": [df_rest["Year"].iloc[0]],
            "Year_Cate_Total": [df_rest["Year_Cate_Total"].sum()],
            "Year_Total": [df_rest["Year_Total"].iloc[0]],
            "Percent": [df_rest["Percent"].sum()],
            "Cum": ["100.00"]
            })
            df_year_comb = pd.concat([df_top, df_others], ignore_index=True)
        else:
            df_year_comb = df_top
        year_combs[years] = df_year_comb


    # Plotting Pareto chart per year
    years = sorted(year_combs.keys())
    fig, axes = plt.subplots(1, len(years), figsize=(6 * len(years), 5), sharey=True)

    if len(years) == 1:
        axes = [axes]  # ensure iterable if only one year
    for i, year in enumerate(years):
            df_year = year_combs[year]
            ax1 = axes[i]

            # Bar chart
            ax1.bar(df_year["Category"], df_year["Year_Cate_Total"], color="skyblue", label="Amount")
            ax1.tick_params(axis='x', rotation=90)
            ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
            ax1.set_title(str(year))
            if i == 0:
                ax1.set_ylabel("Total Amount")

            # Cumulative percentage line
            df_year["Cum"] = pd.to_numeric(df_year["Cum"], errors="coerce")
            df_trim = df_year[df_year["Cum"] < 100]

            ax2 = ax1.twinx()
            ax2.plot(df_trim["Category"], df_trim["Cum"], color="red", marker="o", label="Cumulative %")
            for x, y in zip(df_trim["Category"], df_trim["Cum"]):
                ax2.text(x, y + 2, f"{y:.1f}%", ha="center", va="bottom", fontsize=8)

            ax2.axhline(cutoff, color="gray", linestyle="--", label=f"Cut-off {cutoff}%")
            ax2.set_ylim(0, 105)
            if i == len(years) - 1:
                ax2.set_ylabel("Cumulative %")

    # Legend
    fig.suptitle("Category-wise Revenue Pareto Chart by Year", fontsize=16)
    fig.legend(
            handles=[
                plt.Line2D([], [], color="skyblue", marker="s", linestyle="", label="Amount"),
                plt.Line2D([], [], color="red", marker="o", label="Cumulative %"),
                plt.Line2D([], [], color="gray", linestyle="--", label=f"Cut-off {cutoff}%")
            ],
            loc="upper right",
            bbox_to_anchor=(1.05, 1.0)
        )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------
# Calculate and visualize monthly order volume and value trends
# ---------------------------------------------
def calculate_item_revenue_percent(df):
    df["DocDate"] = pd.to_datetime(df["DocDate"], format="%m/%d/%Y")
    df["Year"] = df["DocDate"].dt.year 
    df["LineTotal"] = pd.to_numeric(df["LineTotal"], errors="coerce")
    df = df[df["LineTotal"].notnull()]

    df_item_year = (
        df.groupby(["ItemCode","Year"])["LineTotal"]
        .sum()
        .sort_values(ascending=False)
        .reset_index(name="TotalAmount")
    )

    df_top_item = (
        df.groupby(df["ItemCode"])["LineTotal"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .index
    )
    
    df_year_amount = (
        df.groupby("Year")["LineTotal"]
        .sum()
        .reset_index(name="YearTotal")

    )

    df_item_count_year = (
        df.groupby(["ItemCode","Year"])
        .size()
        .reset_index(name="OrderCount")
    )

    df_item_year=df_item_year[df_item_year["ItemCode"].isin(df_top_item)]

    df_item_merge = pd.merge(df_item_year,df_year_amount, on="Year")
    df_item_merge = pd.merge(df_item_merge,df_item_count_year, on=["ItemCode","Year"], how="inner")
    df_item_merge["Percent"] = df_item_merge["TotalAmount"] / df_item_merge["YearTotal"] * 100
    df_item_merge.sort_values(by=["Year","Percent"],ascending=[True,False])



# ---------------------------------------------
# Calculate and visualize monthly order volume and value trends
# ---------------------------------------------
def plot_sort_order_by_month(df):

    df["DocDate"] = pd.to_datetime(df["DocDate"], format="%m/%d/%Y")
    df["Year"] = df["DocDate"].dt.year
    df["Month"] = df["DocDate"].dt.month
    df = df[
        (df["CANCELED"]=='N') & 
        (df["DocTotal"] !=0.00) & 
        (df["Year"] != 2020)
    ]

    df_month = (
        df.groupby(["Year","Month"])
        .size()
        .reset_index(name="OrderCount")
    )

    df_month_revenue = (
        df.groupby(["Year","Month"])["DocTotal"]
        .sum()
        .reset_index(name="OrderAmount")
    )

    df_month_comb = pd.merge(df_month,df_month_revenue,on=["Year","Month"],how="inner")

    fig,ax1 = plt.subplots(figsize=(16,8))

    #Left ax------------------------------------------------------------------------------------------
    for year in sorted(df_month["Year"].unique()):
        df_year = df_month[df_month["Year"]==year]
        ax1.plot(df_year["Month"],df_year["OrderCount"], marker="o", label=f"{year} Count")

    ax1.set_xticks(range(1,13))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Order Count")
    ax1.grid(True)

    #Right ax-----------------------------------------------------------------------------------------
    ax2 = ax1.twinx()
    for year in sorted(df_month["Year"].unique()):
        df_year = df_month_revenue[df_month_revenue["Year"]==year]
        ax2.plot(df_year["Month"],df_year["OrderAmount"], linestyle="--",  marker="x", label=f"{year} Amount")
    ax2.set_ylabel("Order Amount")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))


    # combine legend
    line1, labels1 = ax1.get_legend_handles_labels()
    line2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        line1 + line2, 
        labels1 + labels2, 
        title="Year metric", 
        loc="center left",
        bbox_to_anchor=(1.08, 0.5)
    )


    plt.title("Monthly Order Volume and Amount by Year (Dual Y-Axis)")
    plt.tight_layout()
    plt.show()
    
    return df_month_comb
"""
df_ordr = load_ordr()
print(plot_sort_order_by_month(df_ordr))
"""


# ---------------------------------------------
# Visualize fulfillment duration (first vs last delivery date)
# ---------------------------------------------
    # Logic: For each order, find earliest and latest delivery
    # Then calculate fulfillment duration
    # Then create a Gantt-like scatter plot showing those durations
    # Optionally grouped by year/month/brand
def plot_fulfill_duration(df):

    df = df[["DocNum_y","DocNum_x","order_date","delivery_date","NumAtCard_y","CardCode_x","NumAtCard_x"]]
    df = df.drop_duplicates()
    df["BrandName"] = df["CardCode_x"]
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], format="%m/%d/%Y")
    df["order_date"] = pd.to_datetime(df["order_date"], format="%m/%d/%Y")



    # count delivery_date  
    df["delivery_rank"] = df.groupby(["DocNum_y","NumAtCard_y"]).cumcount() +1

    # line to col
    df_pivot = df.pivot_table(
        index=["DocNum_y","NumAtCard_y","BrandName","order_date",],
        columns="delivery_rank",
        values="delivery_date",
        aggfunc="first"
    ).reset_index()

    # rename new col
    delivery_cols = [col for col in df_pivot.columns if isinstance(col, float)]

    df_pivot[1.0] = df_pivot[delivery_cols].min(axis=1)
    df_pivot[2.0] = df_pivot[delivery_cols].max(axis=1)

    df_pivot["fastest_delivery_date"] = df_pivot[1.0] 
    df_pivot["latest_delivery_date"] = df_pivot[2.0] 
    df_pivot = df_pivot[["DocNum_y","NumAtCard_y","BrandName","order_date","fastest_delivery_date","latest_delivery_date"]]

    #df_pivot = df_pivot.drop(columns=["delivery_fast", "delivery_last"])

    df_pivot["latest_delivery_date"] = pd.to_datetime(df_pivot["latest_delivery_date"])
    df_pivot["fastest_delivery_date"] = pd.to_datetime(df_pivot["fastest_delivery_date"])
    df_pivot["order_date"] = pd.to_datetime(df_pivot["order_date"])
                
    df_pivot["last_fulfill_date"] = (df_pivot["latest_delivery_date"] - df_pivot["order_date"]).dt.days
    df_pivot["fast_fulfill_date"] = (df_pivot["fastest_delivery_date"] - df_pivot["order_date"]).dt.days
    df_pivot = df_pivot.sort_values(["order_date"],ascending=True)
    df_pivot["Month"] = df_pivot["order_date"].dt.month
    df_pivot["Year"] = df_pivot["order_date"].dt.year
    

    # Time Period（unit：day）
    bins = [-0.1,15,30,60,np.inf]
    labels = ["0–15d", "16–30d","31-60d", "60+d"]
    
    # last_fulfill_date
    df_pivot["fulfill_bin"] = pd.cut(df_pivot["last_fulfill_date"],bins=bins, labels=labels,right=True)
    
    # fast_fulfill_date


    fulfill_count = df_pivot.groupby("fulfill_bin").size().reset_index(name="OrderCount")

    missing_bin_rows = df_pivot[df_pivot["fulfill_bin"].isna()]

     # filter date & filter company

    df_pivot_2024 = df_pivot[df_pivot["Year"]==2024]

    plt.figure(figsize=(14,8))

    df_order_sorted = df_pivot_2024.sort_values("order_date").reset_index(drop=True).head(300)


    for i, row in df_order_sorted.iterrows():
        if pd.isna(row["fastest_delivery_date"]) or pd.isna(row["latest_delivery_date"]):
            continue
        plt.plot(
            [row["fastest_delivery_date"],row["latest_delivery_date"]],
            [i,i],
            color='skyblue',linewidth=0.5
        )
        plt.scatter(row["fastest_delivery_date"],i,color='green', label='Fastest' if i==0 else "")
        plt.scatter(row["latest_delivery_date"],i, color='red', label='Latest' if i == 0 else"")
    plt.xlabel("Delivery Date")
    plt.ylabel("Order Index")
    plt.title("Earliest vs latest Fulfillment per Order")
    plt.legend()
    plt.tight_layout()
    plt.show()
    """

    df_grouped = (
        df_pivot.groupby(["Year","Month"])[["fast_fulfill_date","last_fulfill_date"]]
        .mean()
        .reset_index()
        .rename(columns={
            "fast_fulfill_date": "AVG_fast_fulfill_date",
            "last_fulfill_date": "AVG_last_fulfill_date"
        })
    )

    df_grouped_brand = (
        df_pivot.groupby(["BrandName"])[["fast_fulfill_date","last_fulfill_date"]]
        .mean()
        .reset_index()
        .rename(columns={
            "fast_fulfill_date": "AVG_fast_fulfill_date",
            "last_fulfill_date": "AVG_last_fulfill_date"
        })
    )

    plt.figure(figsize=(12,6))
    for year in sorted(df_pivot["Year"].unique()):
        subset = df_pivot[df_pivot["Year"] == year]
        plt.plot(subset["Month"], subset["order_date"], marker="o", label=str(year))
        plt.plot(subset["Month"], subset["fastest_delivery_date"], marker="o", label=str(year))
        plt.plot(subset["Month"], subset["latest_delivery_date"], marker="o", label=str(year))
        


    plt.xticks(range(1,13))
    plt.xlabel("Month")
    plt.ylabel("Avg Fulfillment Time (days)")
    plt.title("Monthly Fufillment Time Trend")
    plt.legend(title="Year")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df_pivot


df_load_ocrd = load_ocrd()
df_load_ordr = load_ordr()
df_load_rdr1 = load_rdr1()
df_load_odln = load_odln()   
df_load_rdr1 = df_load_rdr1.rename(columns={"DocEntry": "RDR1_DocEntry"})
df_load_odln = df_load_odln.rename(columns={"DocEntry": "ODLN_DocEntry"})

df_merged_odln  = pd.merge(
    df_load_rdr1, 
    df_load_odln, 
    left_on="TrgetEntry",
    right_on="ODLN_DocEntry",
    how="left"
    )

df_merged_oldn_orod = pd.merge(
    df_merged_odln,
    df_load_ordr,
    left_on="RDR1_DocEntry",
    right_on="DocEntry",
    how="inner")

df_merged_oldn_orod = df_merged_oldn_orod.rename(columns={"DocEntry": "RDR1_DocEntry"})
df_merged_oldn_orod = df_merged_oldn_orod.rename(columns={"DocEntry": "RDR1_DocEntry"})
df_merged_oldn_orod = df_merged_oldn_orod.rename(columns={"DocDate_y": "order_date"})
df_merged_oldn_orod = df_merged_oldn_orod.rename(columns={"DocDate_x": "delivery_date"})



df_preview = plot_fulfill_duration(df_merged_oldn_orod)

#df_preview = df_preview[df_preview["latest_delivery_date"] >= df_preview["order_date"] ]

#df_preview = df_preview.sort_values("last_fulfill_date",ascending=False)



print(df_preview)
    """