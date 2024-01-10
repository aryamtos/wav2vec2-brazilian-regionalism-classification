from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from datasets import Dataset
import torchaudio


def get_label_list(dataset, output_column):
    label_list = dataset.unique(output_column)
    label_list.sort()
    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")
    return label_list, num_labels   


def create_config(model_name_or_path, label_list, pooling_mode):
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(label_list),
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
    )
    setattr(config, 'pooling_mode', pooling_mode)
    return config


def create_feature_extractor(model_name_or_path):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
    target_sampling_rate = feature_extractor.sampling_rate
    print(f"The target sampling rate: {target_sampling_rate}")
    return feature_extractor


def speech_file_to_array_fn(batch, seconds_stop=10, s_rate=16_000):
    start = 0
    stop = seconds_stop
    srate = s_rate
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array[0, :stop * sampling_rate]
    batch["speech"] = speech_array
    batch["sampling_rate"] = srate
    batch["parent"] = batch["label"]
    return batch


def prepare_dataset(batch, feature_extractor):
    assert len(set(batch["sampling_rate"])) == 1, f"Sampling rate {feature_extractor.sampling_rate}."
    batch["input_values"] = feature_extractor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    batch["labels"] = batch["parent"]
    return batch
