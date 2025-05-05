import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import streamlit as st
import numpy as np
warnings.filterwarnings('ignore')

st.title("Product Bundling Solution")
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

# Apply the apriori algorithm
frequent_itemsets = apriori(basket_sets, min_support=0.001, use_colnames=True)


# Generate association rules
apriori_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
#apriori_rules['combined'] = apriori_rules['antecedents'] + apriori_rules['consequents']
n = st.selectbox("Specifiy the minimum number of items you want in a basket ?",options=[2,3,4,5,6],placeholder='Select the number of baskets required...')
# Filter association rules
if st.button("Run"):
    st.write("By default filtered for lift > 50 and confidence >0.01")
    filtered_rules = apriori_rules[(apriori_rules['confidence'] >= 0.01)]
    filtered_rules['antecedents'] = filtered_rules['antecedents'].apply(lambda x: list(x))
    filtered_rules['consequents'] = filtered_rules['consequents'].apply(lambda x: list(x))
    filtered_rules['combined'] = filtered_rules['antecedents'] + filtered_rules['consequents']
    tru_fil = filtered_rules[filtered_rules['combined'].str.len() >= n]
    st.write(tru_fil[['antecedents','consequents','lift']].sort_values(by='lift',ascending=False))
