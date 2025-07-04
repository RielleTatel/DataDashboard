import pandas as pd

# Read the CSV with proper handling of quoted text and mixed types
df = pd.read_csv('Fake.csv', encoding='latin-1', quoting=1, low_memory=False)  # QUOTE_ALL

# Check the structure
print("Original columns:")
print(df.columns.tolist())
print(f"\nOriginal shape: {df.shape}")

# Remove unnamed columns (these are likely parsing errors)
df_clean = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print(f"Shape after removing unnamed columns: {df_clean.shape}")
print(f"Cleaned columns: {df_clean.columns.tolist()}")

# Remove rows with missing values in any column
df_clean = df_clean.dropna()

print(f"Shape after removing rows with any missing values: {df_clean.shape}")
print(f"Removed {len(df) - len(df_clean)} rows with missing values in any column")

# Show the value counts for SUBJECT
print("\nSUBJECT value counts:")
print(df_clean['SUBJECT'].value_counts())

print(f"\nTotal rows (after cleaning): {len(df_clean)}")
print(f"Number of categories: {df_clean['SUBJECT'].nunique()}")

# Save the cleaned dataset with proper quoting to handle commas in text
df_clean.to_csv('Fake_UnSampledData.csv', index=False, quoting=1)

# Optional: Check for any remaining missing values
print(f"\nMissing values in final cleaned dataset:")
print(df_clean.isnull().sum())