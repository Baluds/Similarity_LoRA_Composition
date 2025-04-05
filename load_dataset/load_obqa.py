from datasets import load_dataset

dataset = load_dataset("allenai/openbookqa", cache_dir="/project/pi_wenlongzhao_umass_edu/6/sudharshan/data")
train_dataset = dataset['train']
train_df = train_dataset.to_pandas()
test_dataset = dataset['test']
test_df = test_dataset.to_pandas()
output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/obqa/"
def format_options(row):
    return "\n".join([f"Option {label}: {text}" for label, text in zip(row['choices']['label'], row['choices']['text'])])

# Add the new column
train_df['CombinedOptions'] = train_df.apply(format_options, axis=1)
test_df['CombinedOptions'] = test_df.apply(format_options, axis=1)
train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)
print(train_df.head())