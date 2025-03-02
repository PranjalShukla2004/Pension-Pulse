import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('users.db')

# Create a cursor
c = conn.cursor()

# Create the BANKTABLE
c.execute('''
CREATE TABLE IF NOT EXISTS BANKTABLE (
    BankName TEXT PRIMARY KEY,
    Alpha REAL,
    Beta REAL
)
''')

# Create the USERTABLE with corrected constraints
c.execute('''
CREATE TABLE IF NOT EXISTS USERTABLE (
    UserID INTEGER PRIMARY KEY AUTOINCREMENT,
    Age INTEGER CHECK(Age BETWEEN 20 AND 65),
    Income REAL CHECK(Income BETWEEN 20000 AND 120000),
    Capital REAL CHECK(Capital BETWEEN 0 AND 10000000),
    Market_Knowledge INTEGER CHECK(Market_Knowledge > 0)
)
''')

# Commit and close the connection
conn.commit()
conn.close()

print("Tables created successfully if they did not exist.")