from datasets import DatasetDict,Dataset

def parse_iob2(file_path):
    sentences, labels = [], []
    words, tags = [], []
    unique_labels = set()

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # Skip metadata lines
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words, tags = [], []
                continue
            parts = line.split()  # Split by whitespace
            if len(parts) >= 2:
                words.append(parts[0])  # First column: token
                tags.append(parts[1])  # Second column: label
                unique_labels.add(parts[1])
        if words:
            sentences.append(words)
            labels.append(tags)

    return {"tokens": sentences, "ner_tags": labels}, sorted(unique_labels)

def generate_data_files(language_code: str) -> dict:
    return {
        "train": f"data/train/{language_code}.txt",
        "validation": f"data/val/{language_code}.txt",
        "test": f"data/test/{language_code}.txt"
    }

# Convert labels to integers
def convert_labels(dataset,label2id):
    dataset["ner_tags"] = [[label2id[tag] for tag in tags] for tags in dataset["ner_tags"]]
    return dataset

def load_data(language_code: str):
    data_files = generate_data_files(language_code)
    dataset_dict = {}
    all_labels = set()
    for split, file in data_files.items():
        parsed_data, labels = parse_iob2(file)
        if language_code in ['bg', 'ru']:
            dataset_dict[split] = parsed_data
        else:
            if "test" not in dataset_dict:
                dataset_dict["test"] = parsed_data  # Initialize 'test' split if it doesn't exist
            else:
                # Append parsed_data to the 'test' split
                dataset_dict["test"]["tokens"].extend(parsed_data["tokens"])
                dataset_dict["test"]["ner_tags"].extend(parsed_data["ner_tags"])
        all_labels.update(labels)

    # Convert label list to mapping
    label_list = sorted(all_labels)  # Ensure consistent order
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    dataset_dict = {split: convert_labels(dataset , label2id) for split, dataset in dataset_dict.items()}
    raw_datasets = DatasetDict({
        split: Dataset.from_dict(dataset_dict[split]) for split in dataset_dict
    })

    return raw_datasets, label_list, label2id, id2label