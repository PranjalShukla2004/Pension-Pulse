import sqlite3
import random

# Define ranges
age_range = list(range(20, 66))
income_range = list(range(20000, 120001, 10000))
capital_tuple = (10000, 10000000)
knowledge_range = list(range(1, 11))

# Connect to SQLite
conn = sqlite3.connect("users.db")
c = conn.cursor()

# Generate all possible pairs of (age, income)
age_income_pairs = [(age, income) for age in age_range for income in income_range]

# Shuffle to ensure randomness
random.shuffle(age_income_pairs)
random.shuffle(knowledge_range)

# Insert data into USERTABLE
for (age, income) in age_income_pairs:
    capital = random.randint(capital_tuple[0], capital_tuple[1])
    
    # Ensure we don't run out of knowledge values
    if not knowledge_range:
        knowledge_range = list(range(1, 11))  # Reset knowledge values if exhausted
        random.shuffle(knowledge_range)
    
    knowledge = knowledge_range.pop()
    
    c.execute("INSERT INTO USERTABLE (Age, Income, Capital, Market_Knowledge) VALUES (?, ?, ?, ?)", 
              (age, income, capital, knowledge))

# Commit and close
conn.commit()
conn.close()

print(f"Inserted {len(age_income_pairs)} records into USERTABLE successfully!")