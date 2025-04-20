from datasets import DatasetDict,Dataset
#import stanza
import re, os
import numpy as np
import evaluate
import torch.nn as nn
from transformers import XLMRobertaConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from seqeval.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import EvalPrediction
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    """
    This class extends the RobertaPreTrainedModel class for token classification tasks.
    The model architecture is based on the XLMRoberta model.
    """
    config_class = XLMRobertaConfig

    def __init__(self, config):
        """
        Constructor for the XLMRobertaForTokenClassification class.
        
        Parameters:
        config (XLMRobertaConfig): Configuration object containing information about how to build the model.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        # Load model body
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # Set up token classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Load and initialize weights
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                labels=None, **kwargs):
        """
        Forward method for the XLMRobertaForTokenClassification class.
        
        Parameters:
        input_ids (torch.Tensor): Tensor of input ids of shape (batch_size, sequence_length).
        attention_mask (torch.Tensor): Tensor of attention masks of shape (batch_size, sequence_length).
        token_type_ids (torch.Tensor): Tensor of token type ids of shape (batch_size, sequence_length).
        labels (torch.Tensor): Tensor of labels of shape (batch_size, sequence_length).
        
        Returns:
        TokenClassifierOutput: An object that contains the loss and the logits.
        """
        kwargs.pop('num_items_in_batch', None)
        # Use model body to get encoder representations
        outputs = self.roberta(input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids, **kwargs)
        # Apply classifier to encoder representation
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        # Calculate losses
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # Return model output object
        return TokenClassifierOutput(loss=loss, logits=logits, 
                                     hidden_states=outputs.hidden_states, 
                                     attentions=outputs.attentions)

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
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
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
    tokenized_datasets = encode_dataset(raw_datasets, tokenizer)

    return tokenized_datasets, label_list, label2id, id2label, tokenizer

def encode_dataset(corpus, tokenizer):
    return corpus.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True, 
                      remove_columns=[ 'ner_tags', 'tokens'])

def generate_list_for_compute_metrics(predictions, label_ids, index2tag):
    """
    Function to generate prediction and true labels lists for computing metrics.

    Parameters:
    predictions (np.ndarray): A 2D numpy array containing the predicted label IDs for each token in each example.
    label_ids (np.ndarray): A 2D numpy array containing the true label IDs for each token in each example.

    Returns:
    preds_labels_list (list): A list of lists, where each sublist contains the predicted labels for each token in an example.
    true_labels_list (list): A list of lists, where each sublist contains the true labels for each token in an example.
    """
    # Get the predicted labels by taking the argmax over the second dimension of the predictions array
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    preds_labels_list, true_labels_list = [], []

    # Iterate over each example in the batch
    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        # Iterate over each token in the example
        for seq_idx in range(seq_len):
            # Ignore tokens with label ID = -100 (these are special tokens or subwords that we masked during training)
            if label_ids[batch_idx, seq_idx] != -100:
                # Append the predicted and true labels for the token to the lists for the current example
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])
                example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
        # Append the lists for the current example to the main lists
        preds_labels_list.append(example_preds)
        true_labels_list.append(example_labels)

    # Return the lists of predicted and true labels
    return preds_labels_list, true_labels_list

def get_compute_metrics(id2label):
    def compute_metrics(eval_preds: EvalPrediction):
        """
        Function to compute the evaluation metrics for Named Entity Recognition (NER) tasks.
        The function computes precision, recall, F1 score and accuracy.

        Parameters:
        eval_preds (tuple): A tuple containing the predicted logits and the true labels.

        Returns:
        A dictionary containing the precision, recall, F1 score and accuracy.
        """
        # the logits and the probabilities are in the same order,
        # so we don’t need to apply the softmax

        # We remove all the values where the label is -100
        y_pred, y_true = generate_list_for_compute_metrics(eval_preds.predictions, 
                                        eval_preds.label_ids, id2label)
        return {"f1": f1_score(y_true, y_pred)}
    return compute_metrics

def get_config(model_name,label_list, id2label, label2id):
    def my_config():
        model_config = AutoConfig.from_pretrained(model_name, 
                                         num_labels=len(label_list),
                                         id2label=id2label, label2id=label2id)
        return model_config
    return my_config

def create_model_init(model_name, data, device):
    def model_init():
        tokenized_dataset, label_list, label2id, id2label, tokenizer= data
        model_config=get_config(model_name,label_list, id2label, label2id)()
        return (XLMRobertaForTokenClassification
                .from_pretrained(model_name, config=model_config).to(device))
    return model_init

def get_f1_score(model, dataset):
    return model.predict(dataset).metrics["test_f1"]

def evaluate_lang_performance(lang, trainer):
    panx_ds = encode_dataset(panx_ds_combined[lang])
    return get_f1_score(trainer, panx_ds["test"])

def fine_tuning_training_on_single_corpus(dataset, num_samples):
    """
    Function to train the model on a single corpus of data.

    Parameters:
    dataset (DatasetDict): The dataset to train on. It should be a HuggingFace DatasetDict containing 'train', 'validation' and 'test' splits.
    num_samples (int): The number of samples from the training set to use for training.

    Returns:
    results (pd.DataFrame): A pandas DataFrame containing the number of training samples used and the F1 score on the test set.
    """
    # Shuffle the training data and select the first 'num_samples' examples.
    train_ds = dataset["train"].shuffle(seed=42).select(range(num_samples))
    # The validation and test sets are not shuffled or truncated.
    valid_ds = dataset["validation"]
    test_ds = dataset["test"]

    # Update the logging steps in the training arguments to log progress after each batch.
    training_args.logging_steps = len(train_ds) // batch_size

    # Initialize a Trainer instance. This is a HuggingFace class that handles training.
    trainer = Trainer(model_init=model_init, args=training_args,
        data_collator=data_collator, compute_metrics=compute_metrics,
        train_dataset=train_ds, eval_dataset=valid_ds, tokenizer=xlmr_tokenizer)
    
    # Train the model.
    trainer.train()

    # If the training arguments specify to push the model to the HuggingFace model hub, do so with a commit message.
    if training_args.push_to_hub:
        trainer.push_to_hub(commit_message="Training completed!")
    
    # After training, compute the F1 score on the test set.
    f1_score = get_f1_score(trainer, test_ds)

    # Return the results as a pandas DataFrame.
    return pd.DataFrame.from_dict(
        {"num_samples": [len(train_ds)], "f1_score": [f1_score]})

def save_preds(model, language_code, target_language, tokenized_dataset, tokenizer):
    
    decoded_sentences = [
    tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)).split()
    for input_ids in tokenized_dataset["input_ids"]
    ]
    # Assuming your tokenizer is already defined
    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)
    
    # Make sure your dataset is tokenized and has correct features
    test_loader = DataLoader(
        tokenized_dataset,  # should be Hugging Face Dataset
        batch_size=16,
        collate_fn=data_collator
    )

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    
    # Move model to device
    model.to(device)
    model.eval()

    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            # Move input tensors to the same device as the model
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.append(preds.cpu().numpy())  # convert to CPU numpy for saving
    
    label_ids = np.array(tokenized_dataset["labels"])
    # Convert logits to labels
    #predictions = torch.argmax(outputs.logits, dim=-1)
    id2label = model.config.id2label
    all_predictions=np.concatenate(all_predictions, axis=0)
    batch_size, seq_len = all_predictions.shape
    preds_labels_list = []

    # Iterate over each example in the batch
    for batch_idx in range(batch_size):
        sentence_iob = []
        token_idx=0
        # Iterate over each token in the example

        for seq_idx in range(seq_len):
            if token_idx >= len(decoded_sentences[batch_idx]):
                break  # all real tokens processed, exit early!

            if label_ids[batch_idx][seq_idx] != -100:
                token = decoded_sentences[batch_idx][token_idx]
                pred_label = id2label[all_predictions[batch_idx][seq_idx]]
                token_idx += 1
                sentence_iob.append(f"{token_idx}\t{token}\t{pred_label}")
        # Append the lists for the current example to the main lists
        preds_labels_list.append(sentence_iob)

    os.makedirs(f"data/preds/{language_code}", exist_ok=True)
    with open(f"data/preds/{language_code}/{target_language}_predictions.iob", "w") as f:
        for sentence in preds_labels_list:
            for line in sentence:
                f.write(f"{line}\n")
            f.write("\n")  # Separate sentences with a blank line

def readNlu(path):
    # reads labels from last column, assumes conll-like file
    # with 1 word per line, tab separation, and empty lines
    # for sentence splits. The BIO annotation is expected in the
    # third column (index 2), following universalNER.
    annotations = []
    cur_annotation = []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if line == '':
            annotations.append(cur_annotation)
            cur_annotation = []
        elif line[0] == '#' and len(line.split('\t')) == 1:
            continue
        else:
            cur_annotation.append(line.split('\t')[2])
    return annotations

def toSpans(tags):
    # Converts a list of tags to a list of spans
    # in: ['B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O']
    # out: {'7-9:ORG', '0-2:PER'}
    spans = set()
    for beg in range(len(tags)):
        if tags[beg][0] == 'B':
            end = beg
            for end in range(beg+1, len(tags)):
                if tags[end][0] != 'I':
                    break
            spans.add(str(beg) + '-' + str(end) + ':' + tags[beg][2:])
    return spans

def getBegEnd(span):
    return [int(x) for x in span.split(':')[0].split('-')]

def getLooseOverlap(spans1, spans2):
    # returns the overlap of spans without taking the exact boundaries
    # into account. If entities overlap they also count as found.
    found = 0
    for spanIdx, span in enumerate(spans1):
        spanBeg, spanEnd = getBegEnd(span)
        label = span.split(':')[1]
        match = False
        for span2idx, span2 in enumerate(spans2):
            span2Beg, span2End = getBegEnd(span2)
            label2 = span2.split(':')[1]
            if label == label2:
                if span2Beg >= spanBeg and span2Beg <= spanEnd:
                    match = True
                if span2End <= spanEnd and span2End >= spanBeg:
                    match = True
        if match:
            found += 1
    return found

def getUnlabeled(spans1, spans2):
    # Counts the overlap in spans after removing the labels
    return len(set([x.split(':')[0] for x in spans1]).intersection([x.split(':')[0] for x in spans2]))

def evaluate_f1(gold, pred):
    # Evaluates the output of a NER system against gold data.
    gold_ners = readNlu(gold)
    pred_ners = readNlu(pred)
    
    tp = 0
    fp = 0
    fn = 0
    
    recall_loose_tp = 0
    recall_loose_fn = 0
    precision_loose_tp = 0
    precision_loose_fp = 0
    
    tp_ul = 0
    fp_ul = 0
    fn_ul = 0 
    
    for gold_ner, pred_ner in zip(gold_ners, pred_ners):
        gold_spans = toSpans(gold_ner)
        pred_spans = toSpans(pred_ner)
        overlap = len(gold_spans.intersection(pred_spans))
        tp += overlap
        fp += len(pred_spans) - overlap
        fn += len(gold_spans) - overlap
        
        overlap_ul = getUnlabeled(gold_spans, pred_spans)
        tp_ul += overlap_ul
        fp_ul += len(pred_spans) - overlap_ul
        fn_ul += len(gold_spans) - overlap_ul
    
        overlap_loose = getLooseOverlap(gold_spans, pred_spans)
        recall_loose_tp += overlap_loose
        recall_loose_fn += len(gold_spans) - overlap_loose
    
        overlap_loose = getLooseOverlap(pred_spans, gold_spans)
        precision_loose_tp += overlap_loose
        precision_loose_fp += len(pred_spans) - overlap_loose
    
    prec = 0.0 if tp+fp == 0 else tp/(tp+fp)
    rec = 0.0 if tp+fn == 0 else tp/(tp+fn)
    print('recall:   ', rec)
    print('precision:', prec)
    f1 = 0.0 if prec+rec == 0.0 else 2 * (prec * rec) / (prec + rec)
    print('slot-f1:  ', f1)
    
    tp = tp_ul
    fp = fp_ul
    fn = fn_ul
    print()
    print('unlabeled')
    prec = 0.0 if tp+fp == 0 else tp/(tp+fp)
    rec = 0.0 if tp+fn == 0 else tp/(tp+fn)
    print('ul_recall:   ', rec)
    print('ul_precision:', prec)
    f1 = 0.0 if prec+rec == 0.0 else 2 * (prec * rec) / (prec + rec)
    print('ul_slot-f1:  ', f1)
    
    print()
    print('loose (partial overlap with same label)')
    prec = 0.0 if precision_loose_tp + precision_loose_fp == 0 else precision_loose_tp/(precision_loose_tp+precision_loose_fp)
    rec = 0.0 if recall_loose_tp+recall_loose_fn == 0 else recall_loose_tp/(recall_loose_tp+recall_loose_fn)
    print('l_recall:   ', rec)
    print('l_precision:', prec)
    f1 = 0.0 if prec+rec == 0.0 else 2 * (prec * rec) / (prec + rec)
    print('l_slot-f1:  ', f1)

