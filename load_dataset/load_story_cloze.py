import random
from datasets import load_dataset


dataset = load_dataset("lecslab/story_cloze", cache_dir="/project/pi_wenlongzhao_umass_edu/6/sudharshan/data")
train_dataset = dataset['train']
train_df = train_dataset.to_pandas()
test_dataset = dataset['test']
test_df = test_dataset.to_pandas()

output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/story_cloze/"

def format_options(row):
    options = [("chosen", row["chosen"]), ("rejected", row["rejected"])]
    random.shuffle(options)
    
    option_a_label, option_a_text = options[0]
    option_b_label, option_b_text = options[1]
    
    # You can also store which one was correct for training labels
    row["correct_option"] = "A" if option_a_label == "chosen" else "B"

    return f"Option A: {option_a_text}\nOption B: {option_b_text}"

# Add the new column
train_df['options'] = train_df.apply(format_options, axis=1)
test_df['options'] = test_df.apply(format_options, axis=1)
train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)
print(train_df.head())