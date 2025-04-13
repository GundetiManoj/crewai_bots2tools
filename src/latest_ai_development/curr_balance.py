import pandas as pd
df = pd.read_csv('src\latest_ai_development\input_output.csv')
current_balance = []
balance = 10000000  # Starting balance

# Iterate through each row
for index, row in df.iterrows():
    if row['Dr/Cr'] == 'Debit':
        balance -= row['Amount']
    elif row['Dr/Cr'] == 'Credit':
        balance += row['Amount']
    current_balance.append(balance)

# Add the new column
df['Current Balance'] = current_balance
# Save to new CSV
df.to_csv('src\latest_ai_development\input_output_with_balance.csv', index=False)
print("✅ output_with_balance.csv generated successfully!")
