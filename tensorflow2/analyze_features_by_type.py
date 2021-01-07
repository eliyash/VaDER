import pandas as pd
import csv

KEYS_FILE = r"../data/keys.xlsx"
DATA_FILE = r"../data/stard_all.csv"
dfs = pd.read_excel(KEYS_FILE, sheet_name=None)
dect_names_to_values = {}
for sheet_name in dfs:
    sheet = dfs[sheet_name]
    for question in sheet.question:
        question = question.lower()
        if question not in dect_names_to_values:
            dect_names_to_values[question] = []
            # print('{} is already in'.format(question))
        dect_names_to_values[question].append(sheet_name)

with open(DATA_FILE) as file:
    file_iterator = csv.reader(file)
    names = list(next(file_iterator))

[name for name in names if name in dect_names_to_values]