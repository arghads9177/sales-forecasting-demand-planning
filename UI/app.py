import streamlit as st
import pandas as pd
import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
import os
import sys
import subprocess

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns
    
# Libraries for Time Series Analysis

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
    from statsmodels.tsa.seasonal import seasonal_decompose

try:
    from mlxtend.frequent_patterns import apriori, association_rules
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mlxtend"])
    from mlxtend.frequent_patterns import apriori, association_rules



def link_style_menu():
    st.sidebar.markdown(
        """
        <style>
        .sidebar-link {
            font-size: 18px;
            color: #007BFF;
            text-decoration: none;
            margin-bottom: 10px;
            display: block;
        }
        .sidebar-link:hover {
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    menu = {
        "Sales Forecasting": "Sales Forecasting",
        "Time Series Analysis": "Time Series Analysis",
        "Demographic Analysis": "Demographic Analysis",
        "Sub-Category Performance Analysis": "Sub-Category Performance Analysis",
        "Product Performance Analysis": "Product Performance Analysis",
        "Market Basket Analysis": "Market Basket Analysis"
    }
    
    # Display link-style menu
    selected_option = st.sidebar.radio(
        "",
        options=list(menu.values()),
        label_visibility="collapsed"
    )
    
    return selected_option

# Streamlit app
def main():
    st.title("Sales Forecasting and Demand Planing")
    choice = link_style_menu()

    # Sales Forecasting
    if choice == "Sales Forecasting":
        st.subheader("Future Sales Prediction")
        try:
            # Load the optimized Prophet model
            with open("../models/sf_prophet.pkl", "rb") as f:
                prophet_model = pickle.load(f)
            # Sidebar for user input
            st.header("Forecast Parameters")
            months_to_forecast = st.slider("Select the number of months for forecasting:", min_value=1, max_value=36, value=12)

            # Generate future dataframe
            st.header(f"Forecast for the Next {months_to_forecast} Months")
            last_date = prophet_model.history["ds"].max()
            future_dates = prophet_model.make_future_dataframe(periods=months_to_forecast, freq="ME")
            forecast = prophet_model.predict(future_dates)

            # Filter the forecast for the selected range
            forecast_filtered = forecast[forecast["ds"] > last_date][["ds", "yhat", "yhat_lower", "yhat_upper"]]
            forecast_filtered.columns = ["Date", "Sales", "Lower", "Upper"]

            # Plot forecast
            st.subheader("Forecast Graph")
            fig, ax = plt.subplots(figsize=(10, 6))
            prophet_model.plot(forecast, ax=ax)
            st.pyplot(fig)

            # Display forecast values
            st.subheader("Forecasted Values")
            st.write(forecast_filtered.tail(months_to_forecast))

            # Additional option to download forecast
            st.subheader("Download Forecast")
            csv = forecast_filtered[["Date", "Sales", "Lower", "Upper"]].to_csv(index=False)
            st.download_button("Download Forecast as CSV", data=csv, file_name="forecast.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Time Series Analysis
    elif choice == "Time Series Analysis":
        st.subheader("Time Series Analysis")
        data_path = "../data"
        csv_path = os.path.join(data_path, "ssf_cleaned.csv")
        data = pd.read_csv(csv_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index("Date", inplace= True)
        # Daily Sales
        fig, ax = plt.subplots(figsize=(15, 6))
        plt.plot(data["Sales"], label= "Daily Sales")
        plt.xlabel("Day")
        plt.ylabel('Sales')
        plt.title("Daily Sales")
        st.pyplot(fig)
        # Title
        st.markdown("### **Key Findings from Daily Sales**")
        
        st.markdown("""
        #### High Volatility:
        - The daily sales chart shows significant fluctuations, indicating high variability in customer purchasing patterns on a daily basis.
        - This could suggest irregular promotional campaigns, demand spikes for specific products, or seasonality factors.
        """)
        st.markdown("""
        #### Outliers:

        - There are notable peaks representing days with exceptionally high sales. These could align with events like flash sales, holidays, or major promotions.
        """)
        
        st.markdown("""
        #### Sales Consistency:

        - While there are fluctuations, there seems to be an underlying range within which most sales occur on regular days.
        """)
         # Weekly Sales
        fig, ax = plt.subplots(figsize=(15, 6))
        plt.plot(data["Sales"].resample('W').sum(), label= "Weekly Sales")
        plt.xlabel("Day")
        plt.ylabel('Sales')
        plt.title("Weekly Sales")
        st.pyplot(fig)

        # Findings
        st.markdown("### **Key Findings from Weekly Sales**")
        st.markdown("""
        #### Cyclic Patterns:

        - Weekly data smoothens the fluctuations observed in daily sales and highlights cyclic trends. Peaks and troughs suggest recurring high-sales and low-sales weeks.
        - This might point toward consumer behavior patterns (e.g., end-of-month shopping or seasonal purchasing).
        """)
        
        # Strengths
        st.markdown("""
        #### Seasonality:

        - Regular intervals of higher and lower sales weeks could correspond to holiday periods or promotional weeks.
        - Further decomposition would confirm seasonality impact.
        """)
        # Monthly Sales
        fig, ax = plt.subplots(figsize=(15, 6))
        plt.plot(data["Sales"].resample('ME').sum(), label= "Monthly Sales")
        plt.xlabel("Day")
        plt.ylabel('Sales')
        plt.title("Monthly Sales")
        st.pyplot(fig)

        # Findings
        st.markdown("## **Key Findings from Monthly Sales**")
        
        st.markdown("""
        #### Growth Trend:

        - Over the months, there appears to be a general upward trend in sales, which could indicate business growth, improved customer acquisition, or successful marketing efforts.
        """)
        st.markdown("""
        #### Seasonal Peaks:

        - Certain months consistently show spikes, which could be linked to festive seasons, end-of-year sales, or other significant periods for the business.
        """)
        st.markdown("""
        #### Sales Decline:

        - Some months depict a dip in sales, possibly due to off-seasons, low promotional activities, or external market factors.
        """)
        
        # Decompose the Time Series
        st.markdown("## **Decompose the Time Series**")
        st.markdown("Apply seasonal decomposition using moving averages to extract trend, seasonality, and residual components.")
        # Resample(Upsampling) the data to monthly
        df_monthly = data.resample("ME").sum()
        # Seasonal Decomposition of monthly sales
        decomposed = seasonal_decompose(df_monthly["Sales"], model= "multiplicative", period= 12)
        fig, ax = plt.subplots(4, 1, figsize=(12, 10))
        # Plot Observed
        plt.subplot(4,1,1)
        plt.plot(decomposed.observed, label="Observed", color="blue")
        plt.title("Observed")
        plt.legend()

        # Plot Trend
        plt.subplot(4,1,2)
        plt.plot(decomposed.trend, label="Trend", color= "green")
        plt.title("Trend")
        plt.legend()

        # Plot Seasonality
        plt.subplot(4,1,3)
        plt.plot(decomposed.seasonal, label= "Seasonality", color= "orange")
        plt.title("Seasonality")
        plt.legend()

        # Plot Residual
        plt.subplot(4, 1, 4)
        plt.plot(decomposed.resid, label="Residuals", color="red")
        plt.title("Residuals")
        plt.legend()

        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("""
        ### Key Findings

        From the provided seasonal decomposition plots (Observed, Trend, Seasonality, and Residuals), here are the key insights:

        #### Observed:

        - The observed time series plot captures the original monthly sales data.
        - There is visible seasonality (repeated patterns) and some variations over time, possibly influenced by trends or random factors.

        #### Trend:
        - The trend component shows a general upward movement in sales during the early part of the time period, stabilizing in the middle months.
        - Toward the end, the trend declines slightly, which may indicate a drop in overall performance or a seasonal slowdown.

        **Inference:** The trend reflects the long-term behavior of sales, potentially influenced by business growth or external factors like market conditions.

        #### Seasonality:
        - The seasonality plot highlights recurring patterns within the data.
        - The cyclical pattern suggests periodic fluctuations in sales, likely corresponding to annual cycles.

        **Inference:** Seasonal variations may be driven by factors like holidays, festive seasons, or promotions. These patterns are crucial for forecasting as they indicate predictable changes in sales.

        #### Residuals:
        - The residual plot captures the random fluctuations not explained by trend or seasonality.
        - The residuals seem to have high variance at some points, indicating potential noise or irregular events that might need investigation.

        **Inference:** The residuals appear relatively evenly distributed around 1.0, suggesting the decomposition model fits well. However, any visible clusters or spikes in residuals could point to events or anomalies (e.g., unanticipated promotions or stock issues).

        #### Overall Analysis:

        1. **Predictable Patterns:**
        The combination of a clear trend and seasonality indicates that the sales data is suitable for time series forecasting using models like ARIMA, SARIMA, or exponential smoothing.

        2. **Noise Analysis:**
        The residuals should be analyzed further to identify specific outliers or events that are not explained by the model.
        """)

    # Demographic Analysis
    elif choice == "Demographic Analysis":
        st.subheader("Demographic Analysis")
        data_path = "../data"
        csv_path = os.path.join(data_path, "stores_sales_forecasting.csv")
        data = pd.read_csv(csv_path, encoding="latin1")
        # Find State of customers
        col = "State"
        state_values = data[col].value_counts().reset_index()
        st.subheader(f"TOP 10 {col.upper()}(CUSTOMER COUNT) ")
        st.write(state_values.head(10))
        st.write(f"Total number of states: {state_values.shape[0]}")
        st.write("\n\n")

        # Visualize
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(data= state_values.head(10), x= col, y= "count", hue= col)
        plt.title(f"Top 10 {col} Counts")
        plt.xticks(rotation= 45)
        st.pyplot(fig)

        # Find City of customers
        col = "City"
        city_values = data[col].value_counts().reset_index()
        st.subheader(f"TOP 10 {col.upper()}(CUSTOMER COUNT) ")
        st.write(city_values.head(10))
        st.write(f"Total number of states: {city_values.shape[0]}")
        st.write("\n\n")

        # Visualize
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(data= city_values.head(10), x= col, y= "count", hue= col)
        plt.title(f"Top 10 {col} Counts")
        plt.xticks(rotation= 45)
        st.pyplot(fig)

        # Find Region of customers
        col = "Region"
        region_values = data[col].value_counts().reset_index()
        st.subheader(f"TOP 10 {col.upper()}(CUSTOMER COUNT) ")
        st.write(region_values.head(10))
        st.write(f"Total number of states: {region_values.shape[0]}")
        st.write("\n\n")

        # Visualize
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(data= region_values.head(10), x= col, y= "count", hue= col)
        plt.title(f"Top 10 {col} Counts")
        plt.xticks(rotation= 45)
        st.pyplot(fig)
        
    # Sub-Category Performance Analysis
    elif choice == "Sub-Category Performance Analysis":
        st.subheader("Sub-Category Performance Analysis")
        data_path = "../data"
        csv_path = os.path.join(data_path, "stores_sales_forecasting.csv")
        data = pd.read_csv(csv_path, encoding="latin1")
        # Find Sub-Category of customers
        col = "Sub-Category"
        subcat_values = data[col].value_counts().reset_index()
        st.markdown(f"### TOP 10 {col.upper()}(CUSTOMER COUNT) ")
        st.write(subcat_values.head(10))
        st.write(f"Total number of states: {subcat_values.shape[0]}")
        st.write("\n\n")

        # Visualize
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(data= subcat_values.head(10), x= col, y= "count", hue= col)
        plt.title(f"Top 10 {col} Counts")
        plt.xticks(rotation= 45)
        st.pyplot(fig)

        # Sub-Category Performance Analysis
        subcategory_performance = data.groupby("Sub-Category").agg(
            Total_Sales= ("Sales", "sum"),
            Total_Profit= ("Profit", "sum"),
            Avg_Sales= ("Sales", "mean"),
            Avg_Profit= ("Profit", "mean")
        ).reset_index()
        # Calculate Profit Percentage
        subcategory_performance["Profit_Percentage"] = subcategory_performance["Total_Profit"] / subcategory_performance["Total_Sales"] * 100

        st.markdown("### SALES AND PROFIT ANALYSIS FOR SUB-CATEGORY")
        st.write(subcategory_performance)
        st.write("\n\n")

        fig, ax = plt.subplots(figsize=(16, 6))
        # Visualize the Total Sales
        plt.subplot(1,2,1)
        sns.barplot(data = subcategory_performance, x= "Sub-Category", y= "Total_Sales", hue="Sub-Category")
        plt.title("Total Sales for Each Sub-Category")
        plt.ylabel("Total Sales")
        plt.xticks(rotation=45)

        # Visualize the Profit Percentage
        plt.subplot(1,2,2)
        sns.barplot(data = subcategory_performance, x= "Sub-Category", y= "Profit_Percentage", hue="Sub-Category", palette="Set2")
        plt.title("Profit Percentage for Each Sub-Category")
        plt.ylabel("Profit(%)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        st.markdown("""
        #### Top Performing Subcategory:

        - Furnishings have the highest profit margin (**14%**) and total profit (**$13,059**).

        #### Underperforming Subcategory:

        - Tables have a negative profit margin (**-8.56%**), indicating potential pricing or cost issues.

        #### Improvement Opportunities:

        - Focus on boosting sales of **Bookcases** and increasing margins for **Chairs**.
        - Investigate why **Tables** result in a loss and consider adjusting pricing or production costs.

        """)

        # Region wise Sub-Category Analysis
        region_subcategory_performance = data.groupby(["Region", "Sub-Category"]).agg(
            Total_Sales= ("Sales", "sum"),
            Total_Profit= ("Profit", "sum"),
            Avg_Sales= ("Sales", "mean"),
            Avg_Profit= ("Profit", "mean")
        ).reset_index()
        # Calculate Profit Percentage
        region_subcategory_performance["Profit_Percentage"] = region_subcategory_performance["Total_Profit"] / region_subcategory_performance["Total_Sales"] * 100

        st.markdown("### SALES AND PROFIT ANALYSIS FOR SUB-CATEGORY IN EACH REGION")
        st.write(region_subcategory_performance)
        st.write("\n\n")

        fig, ax = plt.subplots(figsize=(16, 6))
        # Visualize the Total Sales
        plt.subplot(1,2,1)
        sns.barplot(data = region_subcategory_performance, x= "Region", y= "Total_Sales", hue="Sub-Category")
        plt.title("Total Sales for Each Sub-Category Region Wise")
        plt.ylabel("Total Sales")
        plt.xticks(rotation=45)

        # Visualize the Profit Percentage
        plt.subplot(1,2,2)
        sns.barplot(data = region_subcategory_performance, x= "Region", y= "Profit_Percentage", hue="Sub-Category", palette="Set2")
        plt.title("Profit Percentage for Each Sub-Category Region Wise")
        plt.ylabel("Profit(%)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        # Key Findings
        st.markdown("""
        #### Top-performing regions and subcategories:

        - **Chairs** have highest sales in **West** region and a very decent sale in **East and Central** regions as well.
        - **Furnishings** have the highest profit margin(**25%**) in **West** region and maintain a decent profit margin(**20%**) in **East and South** regions.

        #### Underperforming products in specific regions:

        - **Tables** in the **East** region show a negative profit margin (**-28%**), requiring investigation into pricing or cost-related issues.

        #### Improvement Opportunities:

        - Focus on boosting sales of **Bookcases** in **South** region and increasing margins for **Chairs** in **West** Region.
        - Investigate why **Furnishings** in **Central** region result in a loss and consider adjusting pricing or production costs.
        - Investigate why **Tables** in all regions except **West** region result in a loss and consider adjusting pricing or production costs.
        """)
    # Product Performance Analysis
    elif choice == "Product Performance Analysis":
        st.subheader("Product Performance Analysis")
        data_path = "../data"
        csv_path = os.path.join(data_path, "stores_sales_forecasting.csv")
        data = pd.read_csv(csv_path, encoding="latin1")

        # Product Performance Analysis
        # Calculate Total, Sales, Proft, Quantity Sold and Rank the accrodingly
        product_performance = data.groupby("Product Name").agg(
            Total_Sales= ("Sales", "sum"),
            Total_Profit= ("Profit", "sum"),
            Avg_Sales= ("Sales", "mean"),
            Avg_Profit= ("Profit", "mean"),
            Total_Quantity= ("Quantity", "sum")
        ).reset_index()
        # Calculate Profit Percentage
        product_performance["Profit_Percentage"] = product_performance["Total_Profit"] / product_performance["Total_Sales"] * 100

        # Set Rank of Products by Total Sales, Quantity and Profut
        product_performance["Sales_Rank"] = product_performance["Total_Sales"].rank(ascending= False)
        product_performance["Profit_Rank"] = product_performance["Profit_Percentage"].rank(ascending= False)
        product_performance["Quantity_Rank"] = product_performance["Total_Quantity"].rank(ascending= False)

        st.markdown("### TOP 10 HIGH SELLING PRODUCTS")
        st.write(product_performance[["Product Name", "Total_Sales", "Avg_Sales", "Sales_Rank"]].sort_values("Sales_Rank").head(10))
        st.write("\n\n")

        fig, ax = plt.subplots(figsize=(16, 6))
        # Visualize the Total Sales
        sns.barplot(data = product_performance.sort_values("Sales_Rank").head(10), x= "Product Name", y= "Total_Sales", hue="Product Name")
        plt.title("Top 10 Selling Products")
        plt.ylabel("Total Sales")
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.markdown("### TOP 10 HIGH PROFIT PRODUCTS")
        st.write(product_performance[["Product Name", "Total_Profit", "Profit_Percentage", "Profit_Rank"]].sort_values("Profit_Rank").head(10))
        st.write("\n\n")

        # Visualize the Profit Percentage
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.barplot(data = product_performance.sort_values("Profit_Rank").head(10), x= "Product Name", y= "Profit_Percentage", hue="Product Name")
        plt.title("Top 10 Profitable Products")
        plt.ylabel("Profit (%)")
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.markdown("### TOP 10 HIGH DEMAND PRODUCTS")
        st.write(product_performance[["Product Name", "Total_Quantity", "Quantity_Rank"]].sort_values("Quantity_Rank").head(10))
        st.write("\n\n")

        # Visualize the Low Demand
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.barplot(data = product_performance.sort_values("Quantity_Rank").head(10), x= "Product Name", y= "Total_Quantity", hue="Product Name")
        plt.title("Top 10 Demanding Products")
        plt.ylabel("Total Quantity")
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.markdown("### 10 LOW DEMAND PRODUCTS")
        st.write(product_performance[["Product Name", "Total_Quantity", "Quantity_Rank"]].sort_values("Quantity_Rank").tail(10))
        st.write("\n\n")

        # Visualize the Low Demand
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.barplot(data = product_performance.sort_values("Quantity_Rank").tail(10), x= "Product Name", y= "Total_Quantity", hue="Product Name")
        plt.title("10 Low Demanding Products")
        plt.ylabel("Total Quantity")
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # Categorize the Products in terms of Sales and Profit Margin
        # Define thresholds (using median as an example)
        sales_median = product_performance['Total_Sales'].median()
        profit_median = product_performance['Profit_Percentage'].median()

        # Categorize products
        def categorize(row):
            if row['Total_Sales'] > sales_median and row['Profit_Percentage'] > profit_median:
                return 'High Sales, High Profit'
            elif row['Total_Sales'] > sales_median and row['Profit_Percentage'] <= profit_median:
                return 'High Sales, Low Profit'
            elif row['Total_Sales'] <= sales_median and row['Profit_Percentage'] > profit_median:
                return 'Low Sales, High Profit'
            else:
                return 'Low Sales, Low Profit'

        product_performance['Segment'] = product_performance.apply(categorize, axis=1)

        # Count of products in each segment
        st.markdown("### SEGMENTATION SUMMARY")
        segment_counts = product_performance['Segment'].value_counts()
        st.write(segment_counts)

        # Visualization: Scatter plot of Sales vs. Profit
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.scatterplot(
            data=product_performance,
            x='Total_Sales', y='Profit_Percentage',
            hue='Segment', palette='Set2'
        )
        plt.title('Product Segmentation: Sales vs. Profit')
        plt.xlabel('Total Sales')
        plt.ylabel('Profit (%)')
        plt.legend(title='Segment')
        st.pyplot(fig)

        st.markdown("""
        #### High Performers:

        - Products with high sales and profit margin should be prioritized for promotions or further analysis to replicate their success.

        #### Low Performers:

        - Products with low sales and profit margins should be reviewed for potential issues (e.g., high costs, low demand).
        #### Opportunities:

        - Products with high sales but low profit margins may require price optimization or cost reduction strategies.

        #### Segmentation

        - **High Sales, High Profit:**

            - These are your star products. Focus on expanding their reach or replicating their success.

        - **High Sales, Low Profit:**

            - These are potential opportunities for cost optimization or price adjustment to improve profitability.

        - **Low Sales, High Profit:**

            - These products are niche but profitable. Consider targeted marketing to grow their sales.

        - **Low Sales, Low Profit:**

            - These are underperforming products. Investigate for possible discontinuation, rebranding, or cost reduction.
        """)
        # Market Basket Analysis
    elif choice == "Market Basket Analysis":
        st.subheader("Market Basket Analysis")
        data_path = "../data"
        csv_path = os.path.join(data_path, "stores_sales_forecasting.csv")
        data = pd.read_csv(csv_path, encoding="latin1")
        # Prepare Data: Convert the data to basket format
        basket= data.groupby(["Order ID", "Sub-Category"])["Quantity"].sum().unstack().fillna(0)
        basket= basket.applymap(lambda x: 1 if x > 0 else 0)
        # Find frequent itemsets with a minimum support threshold
        frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
        # Generate Association rules
        rules = association_rules(frequent_itemsets, data.shape[0], metric= "lift", min_threshold= 0.3)

        # Visualize top 10 rules by Lift
        top_rules = rules.head(10)
        fig, ax = plt.subplots(figsize=(16, 6))
        plt.barh(range(len(top_rules)), top_rules['lift'], color='skyblue')
        plt.yticks(range(len(top_rules)), [f"{list(a)} -> {list(c)}" for a, c in zip(top_rules['antecedents'], top_rules['consequents'])])
        plt.xlabel('Lift')
        plt.ylabel('Association Rules')
        plt.title('Top 10 Association Rules by Lift')
        st.pyplot(fig)

        # key Findings
        st.markdown("""
        From the given chart showing the **Top 10 Association Rules by Lift**, we can derive the following insights:

        #### Frequent Co-Purchases:

        - Customers frequently buy Chairs and Tables together. This indicates a strong complementary relationship between these products.
        - Similarly, Furnishings and Chairs are often bought in combination, suggesting they are likely used in the same setting or context (e.g., home/office decor).

        #### Bidirectional Relationships:

        The rules highlight bidirectional associations. For instance:
        - ["Chairs"] → ["Tables"]
        - ["Tables"] → ["Chairs"] These reciprocal rules indicate strong dependencies, meaning customers often consider both items when making purchasing decisions.

        #### Lift Values:

        - Lift values for these associations are relatively **low (below 1)**, which suggests these rules, while valid, may not represent highly dominant patterns across the entire dataset.
        - A lower lift could mean that while the rules are valid, **the combinations are not highly exceptional compared to random co-purchases**.

        #### Potential Cross-Selling Opportunities:

        - **Furnishings and Chairs** have an association. Stores can bundle these items or place them in close proximity to encourage cross-sales.
        - The relationship between **Chairs and Tables** suggests a similar bundling or discount strategy.

        #### Segmentation-Based Promotions:

        - The rules point to specific product categories that tend to be purchased together. Marketing campaigns targeting customers purchasing Chairs could include discounts or recommendations for Tables or Furnishings.

        #### Customer Behavior Insight:

        - These associations reveal customers' preference to purchase items that complement each other in functionality or aesthetics, particularly for home or office use.
        """)

#Calling the main function
if __name__ == "__main__":
    main()