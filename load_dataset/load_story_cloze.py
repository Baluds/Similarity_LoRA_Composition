import random
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("lecslab/story_cloze", cache_dir="/project/pi_wenlongzhao_umass_edu/6/sudharshan/data")
train_dataset = dataset['train']
train_df = train_dataset.to_pandas()
test_dataset = dataset['test']
test_df = test_dataset.to_pandas()

output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/story_cloze/"

def format_row(row):
    options = [("chosen", row["chosen"]), ("rejected", row["rejected"])]
    random.shuffle(options)

    option_a_label, option_a_text = options[0]
    option_b_label, option_b_text = options[1]

    correct_option = "A" if option_a_label == "chosen" else "B"
    options_text = f"Option A: {option_a_text}\nOption B: {option_b_text}"
    
    return pd.Series([options_text, correct_option])

# Apply to train and test DataFrames
train_df[['options', 'correct_option']] = train_df.apply(format_row, axis=1)
test_df[['options', 'correct_option']] = test_df.apply(format_row, axis=1)

# Save to CSV
train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)

print(train_df.head())
