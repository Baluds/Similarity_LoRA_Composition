import pandas as pd


train_df = pd.read_csv("/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/paws/train.csv")

test_df = pd.read_csv("/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/paws/test.csv")

output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/paws/"

# train_df["label"] = train_df["label"].map({0: "Not a Paraphrase", 1: "Paraphrase"})
# test_df["label"] = test_df["label"].map({0: "Not a Paraphrase", 1: "Paraphrase"})
def format_options_s1(row):
    return f"Sentence 1:\n{row['sentence1']}"
def format_options_s2(row):
    return f"Sentence 2:\n{row['sentence2']}"

# Add the new column
train_df['sentence1'] = train_df.apply(format_options_s1, axis=1)
train_df['sentence2'] = train_df.apply(format_options_s2, axis=1)
test_df['sentence1'] = test_df.apply(format_options_s1, axis=1)
test_df['sentence2'] = test_df.apply(format_options_s2, axis=1)

train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)
print(train_df.head())