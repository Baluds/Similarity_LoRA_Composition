from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb", cache_dir="/project/pi_wenlongzhao_umass_edu/6/sudharshan/data")
train_dataset = dataset['train']
train_df = train_dataset.to_pandas()
test_dataset = dataset['test']
test_df = test_dataset.to_pandas()
output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/imdb/"
def format_options(row):
    return "Negative" if row["label"] == 0 else "Positive"

# Add the new column
train_df['name_label'] = train_df.apply(format_options, axis=1)
test_df['name_label'] = test_df.apply(format_options, axis=1)
train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)
print(train_df.head())