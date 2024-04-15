from transformers import TrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    is_apex_available,
    set_seed,
)
from transformers import Trainer
import torch
from torch import nn



model_id="ariamtos/wav2vec2-large-xlsr-fine-tuning-brazilian-portuguese"
model_name = model_id.split("/")[-1]


class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, class_weights, return_outputs=False):
        #labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss_func = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_func(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def training():

    training_args = TrainingArguments(
        f"{model_name}-finetuned-dataset",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=5,
        fp16=False,
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        learning_rate=3e-5,
        save_total_limit=5,
    )
    return training_args

torch.cuda.empty_cache()



trainer.train()