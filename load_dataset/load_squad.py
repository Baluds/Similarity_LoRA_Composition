from datasets import load_dataset

dataset = load_dataset("rajpurkar/squad", cache_dir="/project/pi_wenlongzhao_umass_edu/6/sudharshan/data")
train_dataset = dataset['train']
train_df = train_dataset.to_pandas()
test_dataset = dataset['validation']
test_df = test_dataset.to_pandas()
output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/squad/"
def format_options(row):
    return "\n".join([f"{text}" for text in row["answers"]["text"]])

# Add the new column
train_df['answer'] = train_df.apply(format_options, axis=1)
test_df['answer'] = test_df.apply(format_options, axis=1)
train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)
print(train_df.head())