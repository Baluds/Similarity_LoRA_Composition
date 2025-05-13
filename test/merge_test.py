import os
import argparse
import yaml
import pandas as pd
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from  vectorDB.retriever import weigh_datasets

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model_path = config['base_model_path']
    adapter_dir = config['adapter_dir']
    test_data_path = config['test_data_path']
    adapter_names = config['adapter_names']
    allowed_tokens = config['allowed_tokens']
    output_csv = config['output_csv']
    ground_truth_column = config['ground_truth_column']

    print("\n✅ Loaded configuration:")
    print(f"base_model_path      : {base_model_path}")
    print(f"adapter_dir          : {adapter_dir}")
    print(f"test_data_path       : {test_data_path}")
    print(f"adapter_names        : {adapter_names}")
    print(f"allowed_tokens       : {allowed_tokens}")
    print(f"output_csv           : {output_csv}")
    print(f"ground_truth_column  : {ground_truth_column}\n")


    # Load tokenizer + base model
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    print("Loading Base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", device_map="auto")

    print("Loading First adapter model...")
    first_adapter_path = os.path.join(adapter_dir, adapter_names[0])
    model = PeftModel.from_pretrained(base_model, first_adapter_path, adapter_name=adapter_names[0])

    # Load remaining adapters
    for adapter_name in adapter_names[1:]:
        adapter_path = os.path.join(adapter_dir, adapter_name)
        if os.path.exists(adapter_path):
            print(f"Loading adapter: {adapter_name}")
            model.load_adapter(adapter_path, adapter_name=adapter_name)
        else:
            print(f"Skipping missing adapter: {adapter_name}")

    print([tok for tok in allowed_tokens])
    # Prepare allowed tokens
    allowed_token_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str(tok)))[0] for tok in allowed_tokens]

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return allowed_token_ids

    # Load test dataset
    df = pd.read_csv(test_data_path)
    results = []
    correct_count = 0

    print("STARTING TEST")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['Text']
        ground_truth = str(row[ground_truth_column]).strip()
        print(prompt)
        # Hardcoded weights — update later if needed per row
        weights = weigh_datasets(prompt,"chroma_store")
        # weights = {name: 1.0 for name in adapter_names}

        sorted_adapters = sorted(weights.items(), key=lambda x: -x[1])
        print(sorted_adapters)
        model.add_weighted_adapter(
            adapters=[name for name, _ in sorted_adapters],
            weights=[w for _, w in sorted_adapters],
            combination_type="cat",
            adapter_name="merged_cat"
        )
        model.set_adapter("merged_cat")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
        )
        prompt_length = inputs['input_ids'].shape[-1]

        # Slice only the generated part (skip input prompt length)
        generated_tokens = outputs[0][prompt_length:]

        # Decode only the generated tokens
        decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        results.append(decoded_output)

        # Compare with ground truth
        if decoded_output.lower() == ground_truth.lower():
            correct_count += 1

    # Calculate accuracy
    accuracy = correct_count / len(df)
    print(f"✅ Accuracy: {accuracy * 100:.2f}% ({correct_count}/{len(df)})")

    # Save results
    df['Result'] = results
    df.to_csv(output_csv, index=False)
    print(f"✅ All results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapter merging and generation script")
    parser.add_argument("config_path", type=str,
                            help="Path to YAML config file")
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)
