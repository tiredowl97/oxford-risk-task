import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt

##access the datasets
personality_url = "https://raw.githubusercontent.com/karwester/behavioural-finance-task/refs/heads/main/personality.csv"
personality_df = pd.read_csv(personality_url)

supabase_url = "https://pvgaaikztozwlfhyrqlo.supabase.co/rest/v1/assets?select=*"
headers = {
    "apikey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2Z2FhaWt6dG96d2xmaHlycWxvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc4NDE2MjUsImV4cCI6MjA2MzQxNzYyNX0.iAqMXnJ_sJuBMtA6FPNCRcYnKw95YkJvY3OhCIZ77vI",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2Z2FhaWt6dG96d2xmaHlycWxvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc4NDE2MjUsImV4cCI6MjA2MzQxNzYyNX0.iAqMXnJ_sJuBMtA6FPNCRcYnKw95YkJvY3OhCIZ77vI"
}
response = requests.get(supabase_url, headers=headers)
assets_df = pd.DataFrame(response.json())
print("Assets DF Columns:", assets_df.columns)
print("Personality DF Columns:", personality_df.columns ###printfordebugging. The user id were named as _id

combined_df = pd.merge(assets_df, personality_df, on="_id", how="left") #replacing the id and merging the datasets

combined_df.rename(columns={'_id': 'user_id'}, inplace=True)
gbp_df = combined_df[combined_df['asset_currency'] == 'GBP']  #Filter only GBP assets
gbp_sum = gbp_df.groupby('user_id')['asset_value'].sum().reset_index() #Calculate total GBP assets per user

top_user = gbp_sum.sort_values(by='asset_value', ascending=False).iloc[0] #user with highest GBP asset value
top_user_id = top_user['user_id']

top_user_row = combined_df[combined_df['user_id'] == top_user_id] #risk tolerance score
risk_score = top_user_row['risk_tolerance'].iloc[0]

print(f"Highest asset value (in GBP) individual risk tolerance: {risk_score}")

import numpy as np
# Histogram of risk tolerance
sns.histplot(combined_df['risk_tolerance'].dropna(), kde=True)
plt.title("Risk Tolerance Distribution")
plt.xlabel("Risk Tolerance")
plt.ylabel("Count")
plt.show()

# Count of different asset currencies
sns.countplot(x=combined_df['asset_currency'])
plt.title("Distribution of Asset Currencies")
plt.xlabel("Currency")
plt.ylabel("Frequency")
plt.show()

# Boxplot of asset values by currency (log scale)
combined_df['log_asset_value'] = np.log1p(combined_df['asset_value'])
sns.boxplot(data=combined_df, x='asset_currency', y='log_asset_value')
plt.title("Asset Value Distribution by Currency (Log Scale)")
plt.xlabel("Currency")
plt.ylabel("Log(Asset Value + 1)")
plt.show()


###Correlation between asset_value and risk_tolerance (GBP only)
#gbp_corr_df = gbp_df[['asset_value', 'risk_tolerance']].dropna()
#print("\nCorrelation matrix (GBP only):")
#print(gbp_corr_df.corr())

###additional behavioural data analysis
# --- CORRELATION BETWEEN PERSONALITY TRAITS
personality_traits = ['confidence', 'composure', 'risk_tolerance', 'impulsivity', 'impact_desire']
corr_matrix = combined_df[personality_traits].corr()

# Heatmap of personality trait correlations
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Personality Traits")
plt.show()

#  CORRELATION BETWEEN PERSONALITY & ASSET VALUE

combined_df['log_asset_value'] = np.log1p(combined_df['asset_value'])

# Correlation between traits and log_asset_value
traits_and_assets = combined_df[personality_traits + ['log_asset_value']].dropna()
corr_assets = traits_and_assets.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_assets[['log_asset_value']].sort_values(by='log_asset_value', ascending=False), annot=True, cmap='viridis')
plt.title("Correlation Between Traits and Asset Value")
plt.show()

##EVEN MORE ADDITONAL BEHAVIOURAL DATA ANALYSIS
import numpy as np
from sklearn.cluster import KMeans
print("KMeans is working!")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
print("KMeans is working!")
import seaborn as sns
import matplotlib.pyplot as plt

#  CLUSTER ANALYSIS
traits = ['confidence', 'composure', 'risk_tolerance', 'impulsivity', 'impact_desire']
df_traits = combined_df[traits].dropna()
scaler = StandardScaler()
scaled_traits = scaler.fit_transform(df_traits)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_traits)
df_traits['cluster'] = clusters

#
combined_df['cluster'] = np.nan
combined_df.loc[df_traits.index, 'cluster'] = clusters

#  VISUALIZE CURRENCY PREFERENCE BY CLUSTER
currency_pref = combined_df.groupby(['cluster', 'asset_currency']).size().unstack(fill_value=0)
currency_pref.plot(kind='bar', stacked=True)
plt.title("Currency Preferences by Behavioral Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.legend(title="Currency")
plt.tight_layout()
plt.savefig("currency_preference_by_cluster.png")
plt.show()

#  RISK TOLERANCE QUANTILES AND ASSET VALUE
combined_df['risk_quantile'] = pd.qcut(combined_df['risk_tolerance'], 4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

# Boxplot of asset value by risk tolerance quartile
sns.boxplot(data=combined_df, x='risk_quantile', y='asset_value')
plt.title("Asset Value by Risk Tolerance Quartile")
plt.xlabel("Risk Tolerance Quartile")
plt.ylabel("Asset Value")
plt.tight_layout()
plt.savefig("asset_value_by_risk_quartile.png")
plt.show()



######Insights

### Personality & Risk
#- Risk tolerance is normally distributed and peaks around 0.5
#- Confidence, composure, and risk tolerance are **highly positively correlated** (r > 0.5)
#- Impulsivity and impact desire are relatively independent

### Assets & Currency
#- JPY asset values are **significantly higher** than other currencies
#- Currency distribution is balanced across the dataset

### Personality vs Wealth
#- No strong correlation between personality traits and total asset value
#- Impact desire has the strongest (though weak) negative relationship (r ≈ -0.07)

### Extended Analysis
#- **Behavioral clusters** (via KMeans) show different currency preferences
#- **Low risk tolerance** individuals hold **higher asset values** on average