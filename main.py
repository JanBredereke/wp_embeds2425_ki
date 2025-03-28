import json
import os

import state
from load_data.load_audio_files_from_folder import load_audio_files_from_folder
from models.Model2RNN import Model2RNN
from models.PreviousModel import PreviousModel
from modify_data.save_model_to_onnx import save_model_to_onnx
from preprocess_data.preprocess_data import preprocess_data
from training.train_models import train_models


def load_config():
    """
    loads data from config.json into state

    @author <Friederike Korte>
    Returns: None

    """
    with open('config.json') as json_file:
        state.config = json.load(json_file)

    data_folder = state.config["data_folder"]

    state.commands_folder_path = os.path.join(data_folder, 'commands')
    state.background_folder_path = os.path.join(data_folder, 'background')
    state.negatives_folder_path = os.path.join(data_folder, 'negatives')
    state.blank_audios_folder_path = os.path.join(data_folder, 'blank')


def main():
    """
    @author <Serkay-Günay Celik, Sebastian Schramm, Friederike Korte>

    reads audio files from data-folder, preprocesses them, trains different neural networks with it,
    quantizices the models and saves them as 0nnx-files in the trained_models-folder

    :return: None
    """
    load_config()

    """load audio data, set length and add background noise"""
    samples_dict: dict = load_audio_files_from_folder()

    """preprocess data"""
    train_data, test_data, labels, in_features = preprocess_data(samples_dict=samples_dict, with_background_noise=True)

    """define different models"""
    models_list = []
    print('\nCreating models...', end='')
    models_list.append(
        Model2RNN(in_features=in_features, number_of_categories=len(labels))
    )
    print(' first one done...', end='')
    models_list.append(
        PreviousModel(in_features=in_features, number_of_categories=len(labels))
    )
    print('done!\n')

    """train models with test"""
    trained_models = train_models(
        models_list=models_list,
        train_data=train_data,
        test_data=test_data,
        number_of_categories=len(labels),
        categories=labels
    )

    """transform models to onnx-format and save in folder trained_models"""
    for model in trained_models:
        save_model_to_onnx(model, model.__class__.__name__, (in_features,))


if __name__ == "__main__":
    main()
