from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
from datasets import Dataset

@dataclass
class Pipeline:
    path:str = "FacebookAI/roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")


class Transformer:
    def __init__(self):
        pipeline = Pipeline()
        self.model = pipeline.model
        self.tokenizer = pipeline.tokenizer

    def prepare_data(self, dataset):
        assert isinstance(dataset, Dataset), "Incorrect dataset type. This method only accepts datasets.Dataset"
        tokenized_dataset = dataset.map(self.tokenizer(
            dataset['text'], 
            truncation=True,
            padding=True,
            return_tensors="pt"
        ))

        return tokenized_dataset