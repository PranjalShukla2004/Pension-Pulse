import sqlite3
import random

# Connect to SQLite
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create BANKTABLE if not exists
c.execute('''
CREATE TABLE IF NOT EXISTS BANKTABLE (
    BankName TEXT PRIMARY KEY,
    Alpha REAL,
    Beta REAL
)
''')

# Expanded list of dummy banks
banks = [
    "Bank1", "Bank2", "Bank3", "Bank4", "Bank5",
    "Lloyds", "HSBC", "Barclays", "Santander", "Nationwide",
    "Goldman Sachs", "Morgan Stanley", "JP Morgan", "Wells Fargo", "Citibank",
    "Deutsche Bank", "UBS", "BNP Paribas", "Societe Generale", "Credit Suisse"
]

# Generate random Alpha and Beta values for each bank
bank_records = [(bank, round(random.uniform(1.5, 5.5), 5), round(random.uniform(2.0, 6.5), 5)) for bank in banks]

# Insert dummy data, avoiding duplicates
c.executemany("INSERT OR IGNORE INTO BANKTABLE (BankName, Alpha, Beta) VALUES (?, ?, ?)", bank_records)

# Commit changes
conn.commit()
conn.close()

print("More dummy bank records added successfully!")