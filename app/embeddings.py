from sentence_transformers import SentenceTransformer
import json
import numpy as np

class Embeddings:
    """
    A class to generate and store sentence embeddings for a given dataset.

    Attributes:
        model (SentenceTransformer): Pre-trained transformer model for generating embeddings.
        data (list): List of dictionaries representing the dataset.
        embeddings (np.ndarray): NumPy array storing computed embeddings in float32 format.
    """

    def __init__(self, data_path, model="sentence-transformers/all-MiniLM-L6-v2", max_length=512, import_emb=False):
        """
        Initializes the Embeddings class by loading data and computing embeddings.

        Args:
            data_path (str): Path to the JSON file containing text data.
            model (str, optional): Name of the SentenceTransformer model to use. Defaults to "sentence-transformers/all-MiniLM-L6-v2".
            max_length (int, optional): Maximum token length for embeddings (not currently used). Defaults to 512.
        """
        self.model = SentenceTransformer(model)  # Load the sentence transformer model

        if import_emb:
            # get pre-embedded documents and embeddings
            self.data = self.load_data(data_path)
            self.embeddings = np.array(self.data['embeddings'])
            self.documents = self.data['documents']
            self.access_counts = np.array(self.data['access_count'])
            self.urls = self.data['urls']
        else:
            self.data = self.load_data(data_path)  # Load dataset from JSON file
            # Compute and store embeddings as float32 for FAISS compatibility
            self.embeddings = np.array(
                # [self.get_embedding(" ".join(map(str, item.values()))) for item in self.data], # should not just be mashing all together
                [self.get_embedding(item['document']) for item in self.data],
                dtype=np.float32
            )
            self.documents = [item['document'] for item in self.data]
            self.urls = [item['url'] for item in self.data]
            self.access_counts = None  # loop through the available docs, exact match str gets old count else 0

    def load_data(self, data_path):
        """
        Loads JSON data into a list of dictionaries.

        Args:
            data_path (str): Path to the JSON file.

        Returns:
            list: A list of dictionaries, where each dictionary represents an entry in the dataset.
        """
        with open(data_path, "r", encoding="utf-8") as file:
            return json.load(file)  # swapping out json for list of docs

    def get_embedding(self, text):
        """
        Generates an embedding for a given text using the SentenceTransformer model.

        Args:
            text (str): Input text to encode.

        Returns:
            np.ndarray: The generated embedding as a NumPy array in float32 format.
        """
        return self.model.encode(text, convert_to_numpy=True).astype(np.float32)