from datasets import load_dataset

dataset = load_dataset("facebook/anli", cache_dir="/project/pi_wenlongzhao_umass_edu/6/sudharshan/data")
train_dataset = dataset['train_r1']
train_df = train_dataset.to_pandas()
test_dataset = dataset['test_r1']
test_df = test_dataset.to_pandas()
output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/anli_r1/"

train_df["label"] = train_df["label"].map({0: "entailment", 1: "neutral", 2: "contradiction"})
test_df["label"] = test_df["label"].map({0: "entailment", 1: "neutral", 2: "contradiction"})

train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)
print(train_df.head())


train_dataset = dataset['train_r2']
train_df = train_dataset.to_pandas()
test_dataset = dataset['test_r2']
test_df = test_dataset.to_pandas()
output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/anli_r2/"

train_df["label"] = train_df["label"].map({0: "entailment", 1: "neutral", 2: "contradiction"})
test_df["label"] = test_df["label"].map({0: "entailment", 1: "neutral", 2: "contradiction"})

train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)
print(train_df.head())


train_dataset = dataset['train_r3']
train_df = train_dataset.to_pandas()
test_dataset = dataset['test_r3']
test_df = test_dataset.to_pandas()
output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/anli_r3/"

train_df["label"] = train_df["label"].map({0: "entailment", 1: "neutral", 2: "contradiction"})
test_df["label"] = test_df["label"].map({0: "entailment", 1: "neutral", 2: "contradiction"})

train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)
print(train_df.head())