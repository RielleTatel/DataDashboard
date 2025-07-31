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

# Enhanced cleaning: Remove rows with missing values, empty strings, and whitespace-only strings
print(f"\nEnhanced cleaning analysis:")
print(f"Original rows: {len(df_clean):,}")

# Check for different types of missing data
missing_nan = df_clean.isnull().sum().sum()
print(f"Missing values (NaN): {missing_nan:,}")

# Check for empty strings
empty_strings = 0
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        empty_strings += (df_clean[col] == '').sum()
print(f"Empty strings: {empty_strings:,}")

# Check for whitespace-only strings
whitespace_only = 0
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        whitespace_only += (df_clean[col].str.strip() == '').sum()
print(f"Whitespace-only strings: {whitespace_only:,}")

# Remove rows with NaN values
df_clean = df_clean.dropna()
print(f"After removing NaN: {len(df_clean):,} rows")

# Remove rows with empty strings
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean = df_clean[df_clean[col] != '']
print(f"After removing empty strings: {len(df_clean):,} rows")

# Remove rows with whitespace-only strings
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean = df_clean[df_clean[col].str.strip() != '']
print(f"After removing whitespace-only: {len(df_clean):,} rows")

print(f"Total rows removed: {len(df) - len(df_clean):,}")

# Show the value counts for SUBJECT
print("\nSUBJECT value counts:")
print(df_clean['SUBJECT'].value_counts())

print(f"\nTotal rows (after cleaning): {len(df_clean)}")
print(f"Number of categories: {df_clean['SUBJECT'].nunique()}")

# Save the cleaned dataset as Excel
df_clean.to_excel('True_UnsampledData1.xlsx', index=False)

# Optional: Check for any remaining missing values
print(f"\nMissing values in final cleaned dataset:")
print(df_clean.isnull().sum())