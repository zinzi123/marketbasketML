import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import streamlit as st
import numpy as np
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("C:/Users/PhilBiju(G10XIND)/Desktop/Work/POCs/market_basket_pr/dataset/data/Foodmart_dataset.csv")

# Create a transaction ID
df['transaction_id'] = df['customer_id'].astype(str) + df['time_id'].astype(str)

# Select relevant columns for market basket analysis
cols = [75, 1, 24, 7]  # Assuming these are 'transaction_id', 'product_id', 'product_name', and 'unit_sales'
order_products = df[df.columns[cols]]


# Aggregate data for each transaction and product
basket = order_products.groupby(['transaction_id', 'product_name'])['unit_sales'].sum().unstack().reset_index().fillna(0).set_index('transaction_id')


# Encode the data for the apriori algorithm
def encode_units(x):
    return 0 if x <= 0 else 1

basket_sets = basket.applymap(encode_units)
for col in basket_sets.columns:
    if col != 0:
        basket_sets[col] = basket_sets[col].astype('int32')

product_names = df['product_name'].unique()

# Apply the apriori algorithm
frequent_itemsets = apriori(basket_sets, min_support=0.001, use_colnames=True)


# Generate association rules
apriori_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Filter association rules
filtered_rules = apriori_rules[(apriori_rules['lift'] >= 1) & (apriori_rules['confidence'] >= 0.01)]
filtered_rules['antecedents'] = filtered_rules['antecedents'].apply(lambda x: list(x))
filtered_rules['consequents'] = filtered_rules['consequents'].apply(lambda x: list(x))
exploded_1_df = filtered_rules.explode('antecedents')
exploded_df = exploded_1_df['antecedents'].unique()
n = st.sidebar.number_input("Max number of recommendation",value=1,min_value=1,max_value=10,step=1)

# Define recommendation function
def recommendations_using_Apriori(item):
    recommend = []
    for i in range(len(filtered_rules)):
        if item in filtered_rules.iloc[i]['antecedents']:
            recommend.append((filtered_rules.iloc[i]['consequents'], filtered_rules.iloc[i]['lift']))
    return recommend

# Define Streamlit app
import streamlit as st

def main():
    st.title("Product Bundling Solution")
    
    st.header("Basket Recommendation For Product",divider='orange')
    product_name_input = st.selectbox("Select the product for which you want to know basket recommendation",exploded_df)
    #product_name_input = st.text_input("Enter a product name:", "Better Chicken Noodle Soup")
    if st.button("Get Recommendations"):
        recommendations = recommendations_using_Apriori(product_name_input)
        if recommendations:
            st.write(f"Top Basket Recommendation for {product_name_input}:")
            for i, (recommendation, lift) in enumerate(recommendations[:n], start=1):
                st.write(f"{i}. {recommendation[0]} (Lift: {lift:.2f})")
        else:
            st.write("No recommendations found.")

if __name__ == "__main__":
    main()
