import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
import pickle


class BookRecommender:
    def __init__(self, sentence_transformer_path=None):
        if sentence_transformer_path:
            try:
                with open(sentence_transformer_path, "rb") as f:
                    self.trans_model = pickle.load(f)
                print("Sentence Transformer model loaded from pickle file.")
            except Exception as e:
                print(f"Error loading Sentence Transformer from pickle: {e}")
                print("Loading Sentence Transformer from scratch...")
                self.trans_model = SentenceTransformer("all-mpnet-base-v2")
        else:
            self.trans_model = SentenceTransformer("all-mpnet-base-v2")

    def check_datashape(self, dataset: pd.DataFrame):
        no_records, no_attributes = dataset.shape
        print(f"number of records: {no_records}")
        print(f"number of attributes: {no_attributes}")

    def check_heads(self, dataset: pd.DataFrame):
        print(dataset.head())

    def check_datatypes(self, dataset: pd.DataFrame):
        print(dataset.dtypes)

    def check_value_sets(self, dataset: pd.DataFrame):
        for col in dataset.columns:
            print(f"{col} with {len(set(dataset[col]))} values:\t{set(dataset[col])}")

    def demographic_preprocess(self, dataset: dict, quantile_rate: float = 0.9) -> dict:
        count_threshold = dataset["rating_count"].quantile(quantile_rate)
        n_samples = len(dataset["id"])

        preprocessed_dataset = {}
        for key in dataset.keys():
            preprocessed_dataset[key] = pd.Series(
                [
                    dataset[key][i]
                    for i in range(n_samples)
                    if dataset["rating_count"][i] >= count_threshold
                ]
            )
        return preprocessed_dataset

    def get_popular_books(self, dataset: pd.DataFrame, n: int = 10):
        """
        Returns the n most popular books based on aggregate user ratings and interaction counts on the whole dataset.
        Uses weighted rating formula to balance rating score and popularity.

        Args:
            dataset: DataFrame containing book data
            n: Number of books to return
        Returns:
            List of book IDs ordered by popularity score
        """
        count_threshold = dataset["rating_count"].quantile(0.9)
        avg_mean = dataset["avg_rating"].mean()

        def weighted_rating(sample):
            v = sample["rating_count"]
            R = sample["avg_rating"]
            return (v / (v + count_threshold) * R) + (
                count_threshold / (v + count_threshold) * avg_mean
            )

        dataset["score"] = dataset.apply(weighted_rating, axis=1)
        popular_books = dataset.sort_values("score", ascending=False).head(n)
        return popular_books["id"].tolist()

    # HELPER FUNCTIONS
    def convert_dataset(
        self, dataset: dict, attribute_keys: list[str], label_key: str
    ) -> list[dict]:
        """
        Convert dataset with attributes and label
        [{'id': id, 'attributes': {'attribute1': value1, 'attribute2': value2, ...}, 'label': label}...]
        """
        num_points = len(dataset["id"])
        converted_dataset = []
        for index in range(num_points):
            attributes_dict = {}
            for key in attribute_keys:
                attributes_dict[key] = dataset[key][index]

            converted_dataset.append(
                {
                    "id": dataset["id"][index],
                    "attributes": attributes_dict,
                    "label": dataset[label_key][index],
                }
            )
        return converted_dataset

    def create_label_dict(self, dataset: list[dict]) -> dict:
        """
        Create label dictionary:
            id: {'label': label, 'embedding': embedding}
        """
        labels = {}
        for record in dataset:
            labels[record["id"]] = {
                "label": record["label"],
                "embedding": self.encode_sentence(record["label"]),
            }

        return labels

    def encode_sentences(self, data_input: pd.DataFrame) -> np.ndarray:
        """
        Encode sentences using SentenceTransformer
        """
        sentences = data_input.tolist()  # Extract sentences into a list
        embeddings = self.trans_model.encode(sentences)  # Encode the sentences

        return embeddings

    def encode_sentence(self, sentence: str) -> np.ndarray:
        """
        Encode sentence using SentenceTransformer
        """
        embedding = self.trans_model.encode(sentence)

        return embedding

    def z_score_standardize_matrix(self, data_matrix: np.ndarray) -> np.ndarray:
        """
        Standardize each column of a matrix using Z-score normalization
        """
        means = np.mean(
            data_matrix, axis=0, keepdims=True
        )  # Calculate mean of each column
        stds = np.std(
            data_matrix, axis=0, keepdims=True
        )  # Calculate standard deviation of each column

        # Handle cases where standard deviation is zero (to avoid division by zero)
        stds[stds == 0] = (
            1  # or np.finfo(float).eps to use smallest possible float value
        )

        standardized_matrix = (data_matrix - means) / stds
        return standardized_matrix

    def hamming_distance(self, point_1: str, point_2: str) -> int:
        """
        Calculate Hamming distance
        """
        # Create character arrays
        point_1_chars = np.array(list(map(list, point_1)))
        point_2_chars = np.array(list(map(list, point_2)))

        distance = np.sum(point_1_chars != point_2_chars)
        return distance

    def mahattan_distance(self, point_1: np.ndarray, point_2: np.ndarray) -> float:
        """
        Caculate Mahattan distance
        """
        return np.sum(np.abs(point_1 - point_2))

    def calculate_distance_matrix(
        self, input_list: list[dict], attribute_key: str, method: str = "cosine"
    ) -> np.ndarray:
        """
        Calculate distance matrix with different methods
        """
        attribute_points = np.array(
            [point["attributes"][attribute_key] for point in input_list]
        )

        if method == "cosine":
            # Check if the elements are already NumPy arrays
            if isinstance(attribute_points[0], np.ndarray):
                # Reshape to 2D for cosine_distances if necessary
                attribute_points = attribute_points.reshape(
                    attribute_points.shape[0], -1
                )
            else:
                # Convert the list of sparse matrices to a 3D NumPy array
                attribute_points = np.array(
                    [item.toarray() for item in attribute_points]
                )
                # Reshape to 2D for cosine_distances
                attribute_points = attribute_points.reshape(
                    attribute_points.shape[0], -1
                )

            return cosine_distances(attribute_points, attribute_points)

        num_points = len(input_list)
        distances = np.zeros((num_points, num_points))

        for i in range(num_points):
            for j in range(i + 1, num_points):
                if method == "digits":
                    distance = self.hamming_distance(
                        attribute_points[i], attribute_points[j]
                    )
                elif method == "number":
                    distance = self.mahattan_distance(
                        attribute_points[i], attribute_points[j]
                    )
                else:
                    # else return one distance
                    distance = int(attribute_points[i] != attribute_points[j])
                distances[i, j] = distances[j, i] = distance
        return distances

    def calculate_std_distance_matrix(
        self, input_list: list[dict], attribute_key: str, method: str = "cosine"
    ) -> np.ndarray:
        """
        Calculate distance matrix and standardize
        """
        return self.z_score_standardize_matrix(
            self.calculate_distance_matrix(input_list, attribute_key, method)
        )

    def add_ones(self, matrix: np.ndarray) -> np.ndarray:
        """
        Add ones into matrix (N, d) to [1 matrix] (N, d+1)
        """
        return np.hstack((np.ones((len(matrix), 1)), matrix))

    def exp_points(self, points: np.ndarray, lamdas: np.ndarray):
        """
        Transform each point X to exp(lamda * X)
        """
        return np.exp(lamdas * points)

    def preprocess_input(
        self, attribute_matrixes: list[np.ndarray], lambdas: np.ndarray = None
    ) -> np.ndarray:
        """
        Calculate value matrix: total_N x (total_N-1) x (d+1)
            from list of attribute matrix: total_N x total_N
        """
        total_N = attribute_matrixes[0].shape[0]
        d = len(attribute_matrixes)

        if lambdas is None:
            lambdas = np.ones((d,))

        value_matrix = []
        for target in range(total_N):
            target_attribute_relation_list = [
                np.delete(attribute_matrixes[attribute][target], target)
                for attribute in range(d)
            ]

            # Add a new dimension to each array
            target_attribute_relation_array = [
                np.expand_dims(relation, axis=1)
                for relation in target_attribute_relation_list
            ]

            # Concatenate the arrays along the column dimension
            target_attribute_vector = np.concatenate(
                target_attribute_relation_array, axis=1
            )

            # Calculate value by preprocessing functions to [1 e^(-lambda * X)]
            target_value_vector = self.add_ones(
                self.exp_points(target_attribute_vector, (-1) * lambdas)
            )
            value_matrix.append(target_value_vector)

        return np.array(value_matrix)

    def choose_top_N(
        self, X: np.ndarray, W: np.ndarray, N: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate weighted sum of X and choose top N bests
        """
        # Calculate overall score
        weighted_sum = X @ W
        weighted_sum = weighted_sum.flatten()

        # Sort and get top indices
        sorted_indices = np.argsort(weighted_sum)[::-1]

        # Get top indices
        top_indices = sorted_indices[:N]

        # Ensure top_indices is a 1D array
        top_indices = top_indices.flatten()

        return top_indices, weighted_sum[top_indices]

    def calculate_delta(
        self, predicts: list[str], target: str, ids: list[str], labels: dict
    ) -> np.ndarray:
        """
        Calculate Delta matrix by embeddings cosine distance
        """
        N = len(predicts)

        embedded_y_predicts = np.array(
            [labels[ids[predict]]["embedding"] for predict in predicts]
        )  # N x embedding_dim
        embedded_y_target = np.array(
            [labels[ids[target]]["embedding"]]
        )  # 1 x embedding_dim

        # Calculate cosine distances
        delta = cosine_distances(embedded_y_predicts, embedded_y_target)

        return delta

    def train_model(
        self,
        Xs: np.ndarray,
        ids: list[str],
        labels: dict,
        N: int = 10,
        initial_W: np.ndarray = None,
        lr: float = 0.3,
        max_epoch: int = 10,
    ) -> np.ndarray:
        """
        Train Linear model by Gradient Descent for each X in Xs: total_N x (total_N-1) x (d+1)
            1. predict by choose top N best values of X
            2. calculate delta matrix and gradient
            3. update weight matrix W
        """
        total_N = Xs.shape[0]
        d = Xs.shape[2]

        # Initialize W
        if initial_W is None:
            initial_W = np.random.rand(d).reshape(-1, 1)  # (d+1) x 1
        elif initial_W.ndim == 1:  # Check if initial W is (d+1,)
            initial_W = initial_W.reshape(-1, 1)  # Convert to (d+1, 1)

        W = initial_W  # (d+1) x 1

        for _ in range(max_epoch):
            for target in range(total_N):
                X = Xs[target]  # (total_N-1) x (d+1)

                # Predict N bests
                predicts, best_values = self.choose_top_N(X, W, N)
                # Convert predicts to integer indices
                predicts = predicts.astype(int).tolist()

                # Calculate delta
                delta = self.calculate_delta(predicts, target, ids, labels)  # N x 1

                # Calculate gradient = X[N].T @ delta/N (Mean One Error)
                gradient = X[predicts].T @ delta / N  # (d+1) x 1

                # Update W
                W = W - lr * gradient  # (d+1) x 1
        return W

    def apply_model(
        self,
        target: int,
        Xs: np.ndarray,
        W: np.ndarray,
        ids: list[str],
        labels: dict,
        N: int = 10,
    ) -> tuple[list[str], np.ndarray]:
        """
        Apply model to predict top N related with target (having best value)
        """
        print(
            f"*Top {N} related of ID: {ids[target]}, label: {labels[ids[target]]['label']}"
        )

        predicts, best_values = self.choose_top_N(Xs[target], W, N)
        predict_ids = [ids[predict] for predict in predicts]

        decoded_labels = [labels[ids[predict]]["label"] for predict in predicts]

        for index in range(N):
            print(
                f"Top {index+1}\tID: {predict_ids[index]}\tLabel: {decoded_labels[index]}\n\tBest value: {best_values[index]}"
            )
        return predict_ids, best_values

    # --- Preprocessing and data storage methods ---
    def generate_content_attribute_matrixes(
        self, content_input: list[dict]
    ) -> list[np.ndarray]:
        """
        Generate content attribute matrixes by processing content input in attributes:
            isbn13, title, authors, published_year, page_count, language, description
        """
        content_attribute_matrixes = []
        content_attribute_matrixes.append(
            self.calculate_std_distance_matrix(content_input, "isbn13", "digits")
        )

        content_attribute_matrixes.append(
            self.calculate_std_distance_matrix(content_input, "title", "cosine")
        )
        content_attribute_matrixes.append(
            self.calculate_std_distance_matrix(content_input, "authors", "cosine")
        )

        content_attribute_matrixes.append(
            self.calculate_std_distance_matrix(
                content_input, "published_year", "number"
            )
        )
        content_attribute_matrixes.append(
            self.calculate_std_distance_matrix(content_input, "page_count", "number")
        )

        content_attribute_matrixes.append(
            self.calculate_std_distance_matrix(content_input, "language", "one")
        )

        content_attribute_matrixes.append(
            self.calculate_std_distance_matrix(content_input, "description", "cosine")
        )
        return content_attribute_matrixes

    def preprocess_and_store_embeddings(self, dataset: pd.DataFrame):
        """
        Preprocesses the dataset, precomputes Sentence Transformer embeddings,
        and stores them along with other necessary data.
        """
        content_dataset = {}
        content_dataset["id"] = list(dataset["id"].astype(str))
        content_dataset["isbn13"] = list(dataset["isbn13"].astype(str))

        # Precompute embeddings and store them directly in content_dataset
        content_dataset["title"] = list(
            self.encode_sentences(dataset["title"].astype(str))
        )
        content_dataset["authors"] = list(
            self.encode_sentences(dataset["authors"].astype(str))
        )

        content_dataset["published_year"] = list(dataset["published_year"].astype(int))
        content_dataset["page_count"] = list(dataset["page_count"].astype(int))

        content_dataset["language"] = list(dataset["language"].astype(str))

        content_dataset["description"] = list(
            self.encode_sentences(dataset["description"].astype(str))
        )

        content_dataset["category"] = list(dataset["category"].astype(str))

        # Store the preprocessed dataset
        self.content_dataset = content_dataset

        # Other preprocessing steps
        content_attribute_keys = [
            "isbn13",
            "title",
            "authors",
            "published_year",
            "page_count",
            "language",
            "description",
        ]
        content_label_key = "category"

        self.content_converted_dataset = self.convert_dataset(
            content_dataset, content_attribute_keys, content_label_key
        )
        self.content_labels = self.create_label_dict(self.content_converted_dataset)

        # Precompute attribute matrices
        self.content_full_attribute_matrixes = self.generate_content_attribute_matrixes(
            self.content_converted_dataset
        )

        # Precompute value matrix for faster apply_model
        CONTENT_LAMBDAS = np.array([1, 1, 1, 1, 1, 1, 1])
        self.content_full_preprocessed_input = self.preprocess_input(
            self.content_full_attribute_matrixes, CONTENT_LAMBDAS
        )

        # Train the model and store content_W for faster prediction
        content_initial_W = None
        content_learning_rate = 0.03
        content_max_epoch = 100
        self.content_W = self.train_model(
            self.content_full_preprocessed_input,
            self.content_dataset["id"],
            self.content_labels,
            10,
            content_initial_W,
            content_learning_rate,
            content_max_epoch,
        )

    def save_precomputed_data(
        self,
        embeddings_path="embeddings.pkl",
        labels_path="labels.pkl",
        weights_path="content_w.pkl",
        preprocessed_input_path="preprocessed_input.pkl",
    ):
        """Saves the precomputed data to pickle files."""
        with open(embeddings_path, "wb") as f:
            pickle.dump(self.content_dataset, f)
        with open(labels_path, "wb") as f:
            pickle.dump(self.content_labels, f)
        with open(weights_path, "wb") as f:
            pickle.dump(self.content_W, f)
        with open(preprocessed_input_path, "wb") as f:
            pickle.dump(self.content_full_preprocessed_input, f)

    def load_precomputed_data(
        self,
        embeddings_path="embeddings.pkl",
        labels_path="labels.pkl",
        weights_path="content_w.pkl",
        preprocessed_input_path="preprocessed_input.pkl",
    ):
        """Loads precomputed data from pickle files."""
        try:
            with open(embeddings_path, "rb") as f:
                self.content_dataset = pickle.load(f)
            with open(labels_path, "rb") as f:
                self.content_labels = pickle.load(f)
            with open(weights_path, "rb") as f:
                self.content_W = pickle.load(f)
            with open(preprocessed_input_path, "rb") as f:
                self.content_full_preprocessed_input = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Precomputed data files not found. Run 'preprocess_and_store_embeddings' and 'save_precomputed_data' first."
            )

    def find_similar_books(self, book_id: int, n: int = 10):
        """
        Finds books similar to the given book_id using precomputed data.
        """
        # Use precomputed embeddings, labels, and weights
        if (
            not hasattr(self, "content_dataset")
            or not hasattr(self, "content_labels")
            or not hasattr(self, "content_W")
        ):
            raise Exception(
                "Precomputed data not loaded. Call 'load_precomputed_data' first."
            )

        # Find the index of the target book_id
        try:
            target_index = self.content_dataset["id"].index(str(book_id))
        except ValueError:
            raise ValueError(f"Book ID {book_id} not found in the dataset.")

        # Use pre-trained W and preprocessed input for faster prediction
        content_predict_ids, content_best_values = self.apply_model(
            target_index,
            self.content_full_preprocessed_input,
            self.content_W,
            self.content_dataset["id"],
            self.content_labels,
            n,
        )

        return content_predict_ids


def main():
    # export sentence transformer model
    model = SentenceTransformer("all-mpnet-base-v2")
    with open("sentence_transformer.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
