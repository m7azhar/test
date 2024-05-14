import os
import zipfile
import io
import sys
import keras
from PIL import Image
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from keras.utils import pad_sequences
from hate.components.data_transforamation import DataTransformation
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts


class PredictionPipeline:
    def __init__(self, model_path, dataset_zip_path):
        self.model_path = model_path
        self.dataset_zip_path = dataset_zip_path
        self.data_transformation = DataTransformation(data_transformation_config=DataTransformationConfig,
                                                       data_ingestion_artifacts=DataIngestionArtifacts)

    def load_model(self):
        """
        Loads the model from the specified path.
        """
        logging.info("Loading model from path: {}".format(self.model_path))
        try:
            self.model = keras.models.load_model(self.model_path)
            logging.info("Model loaded successfully.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def load_data_from_zip(self):
        """
        Loads preprocessed text data from the dataset zip file.

        Returns:
            A list of preprocessed text data.
        """
        logging.info("Loading data from dataset zip: {}".format(self.dataset_zip_path))
        try:
            processed_text = []
            with zipfile.ZipFile(self.dataset_zip_path, 'r') as zip_ref:
                for filename in zip_ref.namelist():
                    if filename.endswith('.txt'):  # Assuming text data is in txt files
                        with zip_ref.open(filename) as file:
                            text = file.read().decode('utf-8')
                            processed_text.append(self.data_transformation.concat_data_cleaning(text))
            return processed_text
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, text):
        """
        Predicts the class of the input text.

        Args:
            text: The text to be classified (can be ignored for batch prediction).

        Returns:
            A list of predicted class labels (one for each text in the data).
        """
        logging.info("Running the predict function")
        try:
            # Use load_data_from_zip for batch prediction from the dataset
            processed_text = self.load_data_from_zip()

            # Assuming processed text is a list

            pred = self.model.predict(processed_text)

            # Assuming multiple predictions, return a list
            return [p[0] for p in pred]  # Assuming single prediction per element

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self):
        """
        Runs the entire prediction pipeline.

        Returns:
            A list of predicted class labels for all data in the dataset zip.
        """
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            self.load_model()  # Load the model before prediction
            predicted_text = self.predict(None)  # Predict using data from zip
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_text
        except Exception as e:
            raise CustomException(e, sys) from e
