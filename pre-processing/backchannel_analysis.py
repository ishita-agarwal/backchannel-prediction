import pickle
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths
conversation_ids_path = 'data/input/analysis/conversation_ids.pkl'
backchannelers_path = 'data/input/analysis/backchannelers.pkl'
backchannel_rate_final_path = 'data/input/analysis/backchannel_rate_final.pkl'

# Load data from pickle files
with open(conversation_ids_path, 'rb') as file:
    conversation_ids = pickle.load(file)

with open(backchannelers_path, 'rb') as file:
    backchannelers = pickle.load(file)

with open(backchannel_rate_final_path, 'rb') as file:
    backchannel_rate_final = pickle.load(file)

# Assuming each loaded variable is a list and corresponds to a column in a DataFrame
data = {
    # 'conversation_id': conversation_ids,
    'user_id': backchannelers,
    'backchannel_rate': backchannel_rate_final
}

# Create a DataFrame
backchannel_df = pd.DataFrame(data)
backchannel_df['backchannel_rate'] = backchannel_df['backchannel_rate'].apply(lambda x: x[0] if x else None)

# print(backchannel_df.head(5))

metadata_df = pd.read_csv("/Users/ishitaagarwal/Documents/Embeddings/final/src/data/analysis/combined_outcome.csv")



metadata_age_df = metadata_df[['age', 'user_id']]

# print(metadata_age_df.head(5))

backchannel_age_df = pd.merge(backchannel_df, metadata_age_df, on='user_id', how='inner')

# print(backchannel_age_df.head(5))

# print(backchannel_age_df.dtypes)

grouped_df = backchannel_age_df.groupby('user_id').mean()

# print(grouped_df.head(5))

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='backchannel_rate', data=grouped_df)
plt.title('Scatter Plot of Backchannel Rate vs Age')
plt.ylabel('Backchannel Rate')
plt.xlabel('Age')
plt.show()

# Calculate and print Pearson correlation coefficient
correlation = grouped_df['backchannel_rate'].corr(grouped_df['age'])
print("Pearson correlation coefficient:", correlation)

# Calculate the correlation matrix
corr_matrix = grouped_df.corr()

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Heatmap of Correlation Matrix')
plt.show()

