
import pandas as pd
import re
from tkinter import filedialog

# load file
filepath_input = input("Input filepath: ")
df = pd.read_csv(filepath_input)
print("Before: ", df.shape)

# cleaning function
def clean_text(text):
    text = re.sub(r"<[^>]>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# using for whole column
df["review"] = df["review"].apply(clean_text)
print("After: ", df.shape)  # should be same as before

# save as new file
path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save CSV-file")

if path:
    df.to_csv(path, index=False)
    print("Successfully saved at: ", path)
else:
    print("Save failed")
