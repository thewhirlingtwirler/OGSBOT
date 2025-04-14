import faiss

class Retrieval:
    """
    A class to perform similarity-based text retrieval using FAISS.

    Attributes:
        dataset (Embeddings): An instance of the Embeddings class containing text data and embeddings.
        index (faiss.IndexFlatL2): FAISS index built using L2 distance for fast retrieval.
    """

    def __init__(self, dataset):
        """
        Initializes the Retrieval class by creating a FAISS index from the dataset's embeddings.

        Args:
            dataset (Embeddings): An instance of the Embeddings class containing text data and computed embeddings.
        """
        self.dataset = dataset  # Store dataset instance
        # Extract the embedding dimension from the dataset
        embedding_dim = self.dataset.embeddings.shape[1]
        # Create a FAISS index with L2 (Euclidean) distance for similarity search
        self.index = faiss.IndexFlatL2(embedding_dim)
        # Add dataset embeddings to the FAISS index
        self.index.add(self.dataset.embeddings)

    def search(self, query, k=1):
        """
        Finds the top-k most relevant documents using FAISS similarity search.

        Args:
            query (str): The query text to search for similar entries.
            k (int, optional): The number of top results to return. Defaults to 1.

        Returns:
            list or dict: If k=1, returns a single dictionary with 'title' and 'url'.
                          If k>1, returns a list of such dictionaries.
        """
        # Convert query text into an embedding
        query_vector = self.dataset.get_embedding(query).reshape(1, -1)
        # Search FAISS index for the k nearest neighbors
        distances, indices = self.index.search(query_vector, k)
        # Extract only 'url' and 'title' from the retrieved results
        """
        {
            "title": self.dataset.data[i].get("title", "No Title"),
            "url": self.dataset.data[i].get("url", "No URL"),
            "content": self.dataset.data[i].get("content", "No Content")
        }
        """
        results = [
            self.dataset.documents[i] for i in indices[0] if i < len(self.dataset.documents)  # Ensure index is valid
        ]
        urls = [
            self.dataset.urls[i] for i in indices[0] if i < len(self.dataset.urls)  # Ensure index is valid
        ]
        return {'documents': results, 'urls': urls}