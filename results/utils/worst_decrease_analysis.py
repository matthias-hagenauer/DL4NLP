import pandas as pd


"""
Here I use the dataframe that was produced during the creation of the plots. 
I save it as a pkl and load it here. 
"""


# Load your dataframe
df = pd.read_pickle("results.pkl")

# Filter for English → German or Dutch translations only 
# ALSO I am filtering out the 2bit and 4bit models here! 
df_en_de = df[(df['src_lang'] == 'en') & (df['tgt_lang'] == 'nl') & (df['model'] != 'TM_2bit') & (df['model'] != 'TM_4bit')]

# Pivot so we can compare models easily
df_pivot = df_en_de.pivot(index='id', columns='model', values='metrics_comet_seg')

# Compute absolute difference between TM and TM_8bit
df_pivot['abs_diff'] = (df_pivot['TM'] - df_pivot['TM_8bit']).abs()

# Get top k differences
k = 15
top_diff = df_pivot.sort_values('abs_diff', ascending=False).head(k)

# Merge back with original dataframe to get sentence text and other info
top_sentences = df_en_de[df_en_de['id'].isin(top_diff.index)]

# Add the difference column for clarity
top_sentences = top_sentences.merge(top_diff['abs_diff'], left_on='id', right_index=True)

# Select relevant columns to display
result = top_sentences[['id', 'src', 'ref', 'pred', 'model', 'metrics_comet_seg', 'abs_diff']]

# Sort by absolute difference descending
result = result.sort_values('abs_diff', ascending=False)

print(result)
result.to_excel(f"top_{k}_differences.xlsx", index=False)
