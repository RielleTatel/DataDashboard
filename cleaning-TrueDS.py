import pandas as pd

# Read the CSV with proper handling of quoted text and mixed types
df = pd.read_csv('True.csv', encoding='latin-1', quoting=1, low_memory=False)  # QUOTE_ALL

# Check the structure
print("Original columns:")
print(df.columns.tolist())
print(f"\nOriginal shape: {df.shape}")

# Remove unnamed columns (these are likely parsing errors)
df_clean = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print(f"Shape after removing unnamed columns: {df_clean.shape}")
print(f"Cleaned columns: {df_clean.columns.tolist()}")

# Remove rows with missing values in CONTENT column
df_clean = df_clean.dropna(subset=['CONTENT'])

print(f"Shape after removing missing CONTENT: {df_clean.shape}")
print(f"Removed {len(df) - len(df_clean)} rows with missing CONTENT")

# Show the value counts for SUBJECT
print("\nSUBJECT value counts:")
print(df_clean['SUBJECT'].value_counts())

# Calculate sample size (30% of total)
sample_size = int(0.3 * len(df_clean))
n_categories = df_clean['SUBJECT'].nunique()

# Calculate samples per category, but ensure it's at least 1
samples_per_category = max(1, sample_size // n_categories)

print(f"\nTotal rows (after cleaning): {len(df_clean)}")
print(f"Number of categories: {n_categories}")
print(f"Samples per category: {samples_per_category}")

# Stratified sample with error handling
sampled_df = (
    df_clean.groupby('SUBJECT', group_keys=False)
      .apply(lambda x: x.sample(n=min(samples_per_category, len(x)), random_state=42))
)

print(f"Sampled rows: {len(sampled_df)}")

# Save with proper quoting to handle commas in text
sampled_df.to_csv('True_sampled.csv', index=False, quoting=1)

# Optional: Check for any remaining missing values
print(f"\nMissing values in final sample:")
print(sampled_df.isnull().sum())