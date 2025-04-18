from datasets import DatasetDict,Dataset
#import stanza
import re
import numpy as np
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

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
        "train": f"data/train/lemmatized_{language_code}.txt",
        "validation": f"data/val/lemmatized_{language_code}.txt",
        "test": f"data/test/lemmatized_{language_code}.txt"
    }

# Convert labels to integers
def convert_labels(dataset,label2id):
    dataset["ner_tags"] = [[label2id[tag] for tag in tags] for tags in dataset["ner_tags"]]
    return dataset

def load_data(language_code: str, train: bool) :
    
    data_files = generate_data_files(language_code)
    dataset_dict = {}
    all_labels = set()
    #for _,file in data_files.items():
    if train:
      splits=["train","validation"]
    else:
      splits=["test"]
    for split in splits:
      parsed_data, labels = parse_iob2(data_files[split])
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

# Lemmatization

#def ensure_stanza_model(lang: str):
#    try:
#        return stanza.Pipeline(lang, processors="tokenize,lemma", verbose=False)
#    except:
#        stanza.download(lang)
#        return stanza.Pipeline(lang, processors="tokenize,lemma", verbose=False)

def lemmatize_token(token, nlp):
    doc = nlp(token)
    lemma = doc.sentences[0].words[0].lemma
    if token.istitle():
        return lemma.capitalize()
    elif token.isupper():
        return lemma.upper()
    return lemma.lower()

#def lemmatize_file(input_file, output_file, lang):
#    nlp = ensure_stanza_model(lang)

#    with open(input_file, "r", encoding="utf-8") as infile, \
#         open(output_file, "w", encoding="utf-8") as outfile:

#        lines = infile.readlines()
#        for line in lines:
#            stripped = line.strip()
#            if stripped.startswith("#"):
#                outfile.write(f"{stripped}\n")
#            elif stripped:
#                parts = stripped.split()
#                if len(parts) >= 2:
#                    token, tag = parts[0], parts[1]
#                    lemma = lemmatize_token(token, nlp)
#                    outfile.write(f"{lemma}\t{tag}\n")
#            else:
#                outfile.write("\n")

# Transliteration

def transliterate_slovenian_to_cyrillic(text):
    digraphs = {
        "lj": "ль", "Lj": "Ль", "LJ": "ЛЬ",
        "nj": "нь", "Nj": "Нь", "NJ": "НЬ"
    }
    for latin, cyril in digraphs.items():
        text = text.replace(latin, cyril)

    def initial_jvowel(m):
        mapping = {'ja': 'я', 'je': 'йе', 'jo': 'йо', 'ju': 'ю', 'ji': 'йи'}
        combo = m.group(0)
        lower = combo.lower()
        result = mapping.get(lower, combo)
        return result.capitalize() if combo[0].isupper() else result

    text = re.sub(r'\b[Jj][aeioui]', initial_jvowel, text)

    def j_iotation(m):
        before, after = m.group(1), m.group(2)
        lower_map = {'a': 'я', 'e': 'йе', 'o': 'йо', 'u': 'ю', 'i': 'йи'}
        upper_map = {'a': 'Я', 'e': 'ЙЕ', 'o': 'Йо', 'u': 'Ю', 'i': 'ЙИ'}
        mapping = upper_map if after.isupper() else lower_map
        return before + mapping.get(after.lower(), 'й' + after)

    text = re.sub(r'([aeiouAEIOU])j([aeiouiAEIOUI])', j_iotation, text)
    text = re.sub(r'j', 'й', text)
    text = re.sub(r'J', 'Й', text)

    mapping = {
        'a': 'а', 'b': 'б', 'c': 'ц', 'č': 'ч', 'd': 'д', 'e': 'е', 'f': 'ф',
        'g': 'г', 'h': 'х', 'i': 'и', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
        'o': 'о', 'p': 'п', 'r': 'р', 's': 'с', 'š': 'ш', 't': 'т',
        'u': 'у', 'v': 'в', 'z': 'з', 'ž': 'ж',
        'A': 'А', 'B': 'Б', 'C': 'Ц', 'Č': 'Ч', 'D': 'Д', 'E': 'Е', 'F': 'Ф',
        'G': 'Г', 'H': 'Х', 'I': 'И', 'K': 'К', 'L': 'Л', 'M': 'М', 'N': 'Н',
        'O': 'О', 'P': 'П', 'R': 'Р', 'S': 'С', 'Š': 'Ш', 'T': 'Т',
        'U': 'У', 'V': 'В', 'Z': 'З', 'Ž': 'Ж'
    }

    return ''.join(mapping.get(char, char) for char in text)

def transliterate_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            stripped = line.strip()
            if stripped.startswith("# text ="):
                prefix, text = stripped.split("=", 1)
                translit_text = transliterate_slovenian_to_cyrillic(text.strip())
                outfile.write(f"{prefix.strip()} = {translit_text}\n")
            elif stripped.startswith("#") or not stripped:
                outfile.write(line)
            else:
                parts = stripped.split()
                if len(parts) == 2:
                    word, tag = parts
                    word_translit = transliterate_slovenian_to_cyrillic(word)
                    outfile.write(f"{word_translit}\t{tag}\n")
                else:
                    outfile.write(line)

# Tokenization and Label Alignment 

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(label_ids[-1])
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def preprocess(language_code: str, model_name: str, train:bool, cyrillic: bool = False):

    # Lemmatize
    #for split in ["train", "validation", "test"]:
        #original = f"data/{split}/{language_code}.txt"
        #lemmatized = f"data/{split}/lemmatized_{language_code}.txt"
        #lemmatize_file(original, lemmatized, language_code)

    # Optional transliteration
    #if language_code == "sl":
    #    for split in ["train", "validation", "test"]:
    #        lemma_file = f"data/{split}/lemmatized_{language_code}.txt"
    #        translit_file = f"data/{split}/{language_code}.txt"
    #        transliterate_file(lemma_file, translit_file)
    #else:
    #    for split in ["train", "validation", "test"]:
    #        lemma_file = f"data/{split}/lemmatized_{language_code}.txt"
    #        final_file = f"data/{split}/{language_code}.txt"
    #        import shutil
    #        shutil.copyfile(lemma_file, final_file)

    # Load dataset
    if language_code=="sl" and cyrillic:
       language_code+="_cyrilic"
    raw_datasets, label_list, label2id, id2label = load_data(language_code,train)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_datasets = raw_datasets.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    return tokenized_datasets, label_list, label2id, id2label, tokenizer


def save_preds(trainer,language_code:  str, target_language: str, tokenized_dataset, id2label):
  predictions=trainer.predict(tokenized_dataset["test"])
  predictions = predictions.predictions.argmax(2)

  converted_predictions_labels = [
      [id2label[p] for p in sentence] for sentence in predictions
  ]
  iob_predictions=[]
  for sent_idx, final_pred in enumerate(converted_predictions_labels):

      tokens = tokenized_dataset["test"]["tokens"][sent_idx]  # Tokens from your dataset
      sentence_iob = []
      pred_label_cleaned = final_pred[1:len(final_pred) - 1] # Adjusting slicing to align with tokens

      # Ensure both lists have the same length for proper iteration
      min_len = min(len(tokens), len(pred_label_cleaned))
      for token_idx in range(min_len):
          pred_label = pred_label_cleaned[token_idx]  # Get the label from final_predictions
          sentence_iob.append(f"{token_idx+1}\t{tokens[token_idx]}\t{pred_label}")

      iob_predictions.append(sentence_iob)


  with open(f"preds/{language_code}_{target_language}_predictions.iob", "w") as f:
    for sentence in iob_predictions:
        for line in sentence:
            f.write(f"{line}\n")
        f.write("\n")  # Separate sentences with a blank line


def compute_metrics(eval_preds):
    """
    Function to compute the evaluation metrics for Named Entity Recognition (NER) tasks.
    The function computes precision, recall, F1 score and accuracy.

    Parameters:
    eval_preds (tuple): A tuple containing the predicted logits and the true labels.

    Returns:
    A dictionary containing the precision, recall, F1 score and accuracy.
    """
    metric = evaluate.load("seqeval")
    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax

    # We remove all the values where the label is -100
    predictions = [
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
      [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
       for prediction, label in zip(pred_logits, labels)
   ]
    results = metric.compute(predictions=predictions, references=true_labels)
    return {
   "precision": results["overall_precision"],
   "recall": results["overall_recall"],
   "f1": results["overall_f1"],
  "accuracy": results["overall_accuracy"],
  }

def build_model(model_parameters, data):
    model_name=model_parameters["model_name"]
    tokenized_dataset, label_list, label2id, id2label, tokenizer= data
    config = AutoConfig.from_pretrained(model_name, num_labels=len(label_list) , id2label=id2label, label2id=label2id)
    model = AutoModelForTokenClassification.from_config(config)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    args = TrainingArguments(
        output_dir=model_parameters["out_dir"],
        learning_rate=model_parameters["learning_rate"],
        per_device_train_batch_size=model_parameters["per_device_train_batch_size"],
        per_device_eval_batch_size=model_parameters["per_device_eval_batch_size"],
        num_train_epochs=model_parameters["num_train_epochs"],
        weight_decay=model_parameters["weight_decay"]
    )

    trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
    )
    
    return model, trainer