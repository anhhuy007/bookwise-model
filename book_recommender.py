import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
import pickle

class RatingDataset:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

        self.dataset={}
        self.dataset['id'] = list(self.df['id'].astype(int))

        self.dataset['avg_rating'] = self.df['avg_rating'].astype(float)
        self.dataset['rating_count'] = self.df['rating_count'].astype(int)
    
    def get_dataset(self) -> dict:
        return self.dataset
    
class ContentDataset:
    def __init__(self, dataframe: pd.DataFrame, encoding_model_path: str, scaler: str='min-max'):
        self.df = dataframe
        self.attribute_keys = ['isbn13', 'title', 'authors', 'published_date', 'page_count', 'language', 'description']
        self.label_key = 'category'

        # Load Sentence Transformer encoding model from pickle file
        if encoding_model_path:
            try:
                with open(encoding_model_path, "rb") as f:
                    self.encoding_model = pickle.load(f)
                print("Sentence Transformer model is loaded from pickle file.")
            except Exception as e:
                print(f"Error loading Sentence Transformer from pickle: {e}")
                print("Loading Sentence Transformer from scratch...")
                self.encoding_model = SentenceTransformer("all-mpnet-base-v2")
                with open("preprocessed_data/sentence_transformer.pkl", "wb") as f:
                    pickle.dump(self.encoding_model, f)
        else:
            self.encoding_model = SentenceTransformer("all-mpnet-base-v2")

        # Ensure data type consistency
        self.dataset = {}
        self.ids = self.dataset['id'] = list(self.df['id'].astype(int))

        self.dataset['isbn13'] = list(self.df['isbn13'].astype(str))
        self.dataset['language'] = list(self.df['language'].astype(str))

        self.dataset['title'] = list(Utils.encode_sentences(self.df['title'].astype(str), self.encoding_model))
        self.dataset['authors'] = list(Utils.encode_sentences(self.df['authors'].astype(str), self.encoding_model))
        self.dataset['description'] = list(Utils.encode_sentences(self.df['description'].astype(str), self.encoding_model))

        self.dataset['published_date'] = pd.to_datetime(self.df['published_date'], errors="coerce").dt.year.fillna(0).astype(int)  

        self.dataset['page_count'] = list(self.df['page_count'].astype(int))

        self.dataset['category'] = list(self.df['category'].astype(str))

        # Converting
        self.dataset = Utils.convert_dataset(self.dataset, self.attribute_keys, self.label_key)

        # Generating attribute matrixes
        self.attribute_matrixes = []

        self.attribute_matrixes.append(Utils.calculate_std_distance_matrix(self.dataset, 'title', 'cosine', scaler))
        self.attribute_matrixes.append(Utils.calculate_std_distance_matrix(self.dataset, 'authors', 'cosine', scaler))
        self.attribute_matrixes.append(Utils.calculate_std_distance_matrix(self.dataset, 'description', 'cosine', scaler))

        self.attribute_matrixes.append(Utils.calculate_std_distance_matrix(self.dataset, 'isbn13', 'digits', scaler))

        self.attribute_matrixes.append(Utils.calculate_std_distance_matrix(self.dataset, 'published_date', 'number', scaler))
        self.attribute_matrixes.append(Utils.calculate_std_distance_matrix(self.dataset, 'page_count', 'number', scaler))

        self.attribute_matrixes.append(Utils.calculate_std_distance_matrix(self.dataset, 'language', 'one', scaler))
       
    def get_inputs(self) -> list:
        return self.attribute_matrixes
    
    def get_ids(self) -> list[int]:
        return self.ids
    
    def get_labels(self) -> dict:
        return Utils.create_label_dict(self.dataset, self.encoding_model)

    def check_id(self, id: int) -> bool:
        return id in self.ids

class UserDataset:
    def __init__(self, dataframe: pd.DataFrame, encoding_model_path: str, scaler: str='min-max'):
        self.df = dataframe
        self.attribute_keys = ['gender', 'university', 'faculty', 'age', 'language', 'factor']
        self.label_key = 'goal'

        # Load Sentence Transformer encoding model pickle file
        if encoding_model_path:
            try:
                with open(encoding_model_path, "rb") as f:
                    self.encoding_model = pickle.load(f)
                print("Sentence Transformer model is loaded from pickle file.")
            except Exception as e:
                print(f"Error loading Sentence Transformer from pickle: {e}")
                print("Loading Sentence Transformer from scratch...")
                self.encoding_model = SentenceTransformer("all-mpnet-base-v2")
                with open("preprocessed_data/sentence_transformer.pkl", "wb") as f:
                    pickle.dump(self.encoding_model, f)
        else:
            self.encoding_model = SentenceTransformer("all-mpnet-base-v2")

        # Ensure data type consistency
        self.dataset = {}
        self.ids = self.dataset['id'] = list(self.df['id'].astype(int))

        self.dataset['gender'] = list(self.df['gender'].astype(str))
        self.dataset['faculty'] = list(self.df['faculty'].astype(str))
        self.dataset['language'] = list(self.df['language'].astype(str))

        self.dataset['university'] = list(Utils.encode_sentences(self.df['university'].astype(str), self.encoding_model))

        self.dataset['factor'] = list(Utils.fill_head_zeros_list(self.df['factor'].astype(str), fixed_length=8))

        self.dataset['age'] = list(self.df['age'].astype(int))

        self.dataset['goal'] = list(self.df['goal'].astype(str))

        # Converting
        self.dataset = Utils.convert_dataset(self.dataset, self.attribute_keys, self.label_key)

        # Generating attribute matrixes
        self.attribute_matrixes = []
        self.attribute_matrixes.append(Utils.calculate_std_distance_matrix(self.dataset, 'university', 'cosine', scaler))

        self.attribute_matrixes.append(Utils.calculate_std_distance_matrix(self.dataset, 'language', 'digits', scaler))
        self.attribute_matrixes.append(Utils.calculate_std_distance_matrix(self.dataset, 'factor', 'digits', scaler))

        self.attribute_matrixes.append(Utils.calculate_std_distance_matrix(self.dataset, 'age', 'number', scaler))

        self.attribute_matrixes.append(Utils.calculate_std_distance_matrix(self.dataset, 'gender', 'one', scaler))
        self.attribute_matrixes.append(Utils.calculate_std_distance_matrix(self.dataset, 'faculty', 'one', scaler))

    def get_inputs(self) -> list:
        return self.attribute_matrixes
    
    def get_ids(self) -> list[str]:
        return self.ids
    
    def get_labels(self) -> dict:
        return Utils.create_label_dict(self.dataset, self.encoding_model)
    
    def check_id(self, id: int) -> bool:
        return id in self.ids

class FavouriteDataset:
    def __init__(self, dataframe: pd.DataFrame=None):
        self.df = dataframe
        self.dataset={}

        if dataframe is not None:
            self.dataset['user_id'] = list(self.df['user_id'].astype(int))
            self.dataset['book_id'] = list(self.df['book_id'].astype(int))

            self.dict = {}
            users_set = set(self.dataset['user_id']) # remove duplicates

            n_books = min(len(self.dataset['user_id']), len(self.dataset['book_id']))
            for user_id in list(users_set):
                books_lst = [self.dataset['book_id'][i] for i in range(n_books) if self.dataset['user_id'][i] == int(user_id)]
                self.dict[user_id] = books_lst

    def get_favourite_books(self, user_id: int) -> list[int]:
        return self.dict[user_id]
    
    def save_dataset(self, dict_path: str):
        """
        Save model to a pickle file
        """
        with open(dict_path, "wb") as f:
            pickle.dump(self.dict, f)

    def load_dataset(self, dict_path: str):
        """
        Load model from a pickle file
        """
        with open(dict_path, "rb") as f:
            self.dict = pickle.load(f)
        self.df = self.dataset = None


class WeightedRatingModel:
    def __init__(self, quantile_rate: float=0.0, N: int=10):
        self.quantile_rate = quantile_rate
        self.N = N

        self.dataset = None
        self.count_threshold = self.avg_mean = None

    def fit(self, rating_dataset: RatingDataset):
        dataset = rating_dataset.get_dataset()

        self.count_threshold = dataset['rating_count'].quantile(self.quantile_rate)
        self.avg_mean = dataset['avg_rating'].mean()
        
        # Preprocessing
        n_samples = len(dataset['id'])
        self.dataset = {}
        for key in dataset.keys():
            self.dataset[key] = pd.Series([dataset[key][i] for i in range(n_samples) if dataset['rating_count'][i] >= self.count_threshold])

    def compute(self) -> tuple[list[int], np.ndarray]:
        def weighted_rating(sample:dict):
            v = sample['rating_count']
            R = sample['avg_rating']
            return (v / (v + self.count_threshold) * R) + (self.count_threshold / (v + self.count_threshold) * self.avg_mean)
        
        dataframe = pd.DataFrame(self.dataset)
        dataframe['score'] = dataframe.apply(weighted_rating, axis=1)

        popular_samples = dataframe.sort_values('score', ascending=False).head(self.N)

        popular_lst = popular_samples['id'].tolist()
        score_arr = np.array(popular_samples['score'])

        print(f"*Top {self.N} highest weighted rating:")
        for index in range(self.N):
            print(f"Top {index+1}\tID: {popular_lst[index]}\tWeighted rating: {score_arr[index]}")
        
        return popular_lst, score_arr
    
    def save_model(self, N_path: str, dataset_path: str, count_threshold_path: str, avg_mean_path: str):
        """
        Save model to a pickle file
        """
        with open(N_path, "wb") as f:
            pickle.dump(self.N, f)
        with open(dataset_path, "wb") as f:
            pickle.dump(self.dataset, f)
        with open(count_threshold_path, "wb") as f:
            pickle.dump(self.count_threshold, f)
        with open(avg_mean_path, "wb") as f:
            pickle.dump(self.avg_mean, f)

    def load_model(self, N_path: str, dataset_path: str, count_threshold_path: str, avg_mean_path: str):
        """
        Load model from a pickle file
        """
        with open(N_path, "rb") as f:
            self.N = pickle.load(f)
        with open(dataset_path, "rb") as f:
            self.dataset = pickle.load(f)
        with open(count_threshold_path, "rb") as f:
            self.count_threshold = pickle.load(f)
        with open(avg_mean_path, "rb") as f:
            self.avg_mean = pickle.load(f)
        self.quantile_rate = 0.0 

class GradientDescentExpModel:
    def __init__(self, lambdas: np.ndarray=np.array([]), learing_rate: float=0.0, max_epoch: int=0, N: int=10):
        self.lambdas = lambdas
        self.learning_rate = learing_rate
        self.max_epoch = max_epoch
        self.N = N

        self.inputs = self.ids = self.labels = None
        self.W = None

    def fit(self, inputs: np.ndarray, ids: list[str], labels: dict):
        """
        Fit input attributes after preprocessing and label with IDs to model
        """
        self.inputs =  Utils.preprocess_input(inputs, self.lambdas)
        self.ids = ids
        self.labels = labels

    def train(self, initial_W: np.ndarray=None) -> np.ndarray:
        """
        Train Exponent model by Gradient Descent for each X in Xs: D x (D-1) x (d+1)
        Step 1. Initialize Weight Vector W

        Step 2. For each epoch, take preprocessed Input Matrix X with shape (D-1) x (d+1) for each target in Dataset

        Step 3. Calculate Overall Score (OS) by taking Weighted Sum of Exponent of X with Lambda Vector:
            OS(X)=W.explambda(X)=W.e-lambda*X=i=0dwie-lambdaixi

        Step 4. Choose top N highest Overall Score samples as N predicts

        Step 5. Calculate Delta Matrix by calculating Cosine Distance (subtracting 1 to in range [-1, 1]) between embedded predict labels and target label:
            Deltai=CosineDistance(predicti.label, target.label)-1

        Step 6. Calculate Gradient Vector by taking the product of Input Vectors of predicts and Delta in average:
            Gradient=X[predicts]T*Delta / N

        Step 7. Update Weight Vector by Gradient Vector and Learning Rate:
            W'=W-lr*Gradient
        """
        if self.inputs is None or self.ids is None or self.labels is None:
            return None
        
        # Initialize W
        d = self.inputs.shape[2]
        if initial_W is None:
            initial_W = np.random.rand(d).reshape(-1, 1) # (d+1) x 1
        elif initial_W.ndim == 1:  # Check if initial W is (d+1,)
            initial_W = initial_W.reshape(-1, 1) # Convert to (d+1, 1)

        self.W = initial_W # (d+1) x 1
        self.D = self.inputs.shape[0]
        for _ in range(self.max_epoch):
            for target in range(self.D):
                X = self.inputs[target] # (total_N-1) x (d+1)

                # Predict N bests
                predicts, best_values = Utils.choose_top_N(X, self.W, self.N)

                # Convert predicts to integer indices
                predicts = predicts.astype(int).tolist()

                # Calculate delta
                delta = Utils.calculate_delta(predicts, target, self.ids, self.labels) # N x 1

                # Calculate gradient = X[N].T @ delta/N (Mean One Error)
                gradient = X[predicts].T @ delta/self.N # (d+1) x 1

                # Update W
                self.W = self.W - self.learning_rate * gradient # (d+1) x 1
        return self.W
    
    def predict(self, target: int) -> tuple[list[int], np.ndarray]:
        """
        Apply model to predict top N related with target (having best value)
        """
        try:
            if (target not in self.ids):
                print("Dataset has no target ID")
                return [], None
            
            target_pos = self.ids.index(target)
            print(f"*Top {self.N} related of ID: {target}, label: {self.labels[target_pos]['label']}")
            predicts, best_values = Utils.choose_top_N(self.inputs[target_pos], self.W, self.N)
            
            predict_ids = [self.ids[predict] for predict in predicts]
            decoded_labels = [self.labels[self.ids[predict]]['label'] for predict in predicts]

            for index in range(self.N):
                print(f"Top {index+1}\tID: {predict_ids[index]}\tLabel: {decoded_labels[index]}\n\tBest value: {best_values[index]}")
            return predict_ids, best_values
        except Exception as e:
            print(f"Error loading Sentence Transformer from pickle: {e}")
            return [], None          
        
    def get_W(self) -> np.ndarray:
        return self.W
    
    def save_model(self, N_path: str, inputs_path: str, ids_path: str, labels_path: str, W_path: str):
        """
        Save model to a pickle file
        """
        with open(N_path, "wb") as f:
            pickle.dump(self.N, f)
        with open(inputs_path, "wb") as f:
            pickle.dump(self.inputs, f)
        with open(ids_path, "wb") as f:
            pickle.dump(self.ids, f)
        with open(labels_path, "wb") as f:
            pickle.dump(self.labels, f)
        with open(W_path, "wb") as f:
            pickle.dump(self.W, f)

    def load_model(self, N_path: str, inputs_path: str, ids_path: str, labels_path: str, W_path: str):
        """
        Load model from a pickle file
        """
        with open(N_path, "rb") as f:
            self.N = pickle.load(f)
        with open(inputs_path, "rb") as f:
            self.inputs = pickle.load(f)
        with open(ids_path, "rb") as f:
            self.ids = pickle.load(f)
        with open(labels_path, "rb") as f:
            self.labels = pickle.load(f)
        with open(W_path, "rb") as f:
            self.W = pickle.load(f)
        self.lambdas = np.array([])
        self.learning_rate = 0.0
        self.max_epoch = 0


class BookRecommender:
    def __init__(self, rating_model: WeightedRatingModel, content_model: GradientDescentExpModel, user_model: GradientDescentExpModel, favourite_dataset: FavouriteDataset):
        self.rating_model = rating_model
        self.content_model = content_model
        self.user_model = user_model
        self.favorite_dataset = favourite_dataset
    
    def get_popular_books(self) -> tuple[list[int], np.ndarray]:
        """
        Returns the n most popular books based on aggregate user ratings and interaction counts on the whole dataset.
        Uses weighted rating formula to balance rating score and popularity.

        Args:
            dataset: DataFrame containing book data
            n: Number of books to return
        Returns:
            List of book IDs ordered by popularity score
            Best values
        """
        return self.rating_model.compute()

    def find_similar_books(self, book_id: int) -> tuple[list[int], np.ndarray]:
        """
        Finds N books similar to a given book based on content features ('isbn13', 'title', 'authors', 'published_year', 'page_count', 'language', 'description' and 'category').

        Args:
            book_id: Reference book identifier
        Returns:
            List of N similar book IDs
            Best values
        """
        content_predict_ids, content_best_values = self.content_model.predict(book_id)
        return content_predict_ids, content_best_values
    
    def get_collaborative_recommendations(self, user_id: int) -> tuple[list[int], np.ndarray]:
        """
        Finds N users similar to a given user based on content features ('gender', 'school', 'faculty', 'age', 'language', 'factor' and 'goal').

        Args:
            user_id: Reference user identifier
        Returns:
            List of recommended book IDs
            No return best value
        """
        user_predict_ids, user_best_values = self.user_model.predict(user_id)
        print("User predict ids:", user_predict_ids)
        print("User best values:", user_best_values)

        # Return all favorite books of similar users
        favourite_books = []
        for user_id in user_predict_ids:
            favourite_books += self.favorite_dataset.get_favourite_books(user_id)

        # Remove duplicates
        unique_favourite_books = list(set(favourite_books))

        # Randomly pick N books
        return Utils.random_pick_unique(unique_favourite_books, self.content_model.N), None

    def get_personalized_recommendations(self, user_id: int) -> tuple[list[int], np.ndarray]:
        """
        Generates personalized book recommendations based on user's reading preferences.
        Uses content-based filtering comparing book features with user's highly rated books.

        Args:
            user_id: Target user identifier
        Returns:
            List of recommended book IDs
            No return best value
        """
        favourite_books = self.favorite_dataset.get_favourite_books(user_id)
        print(f"Favorite book IDs for user {user_id}:", favourite_books)

        # Take all related books from user preference
        similar_books = []
        for book_id in favourite_books:
            try:
                content_predict_ids, content_best_values = self.find_similar_books(book_id)
                similar_books.extend(content_predict_ids)
            except ValueError:
                print(f"Skipping invalid book_id: {book_id}")

        # Remove duplicates
        unique_similar_books = list(set(similar_books))

        # Randomly pick N books
        return Utils.random_pick_unique(unique_similar_books, self.content_model.N), None

class Utils:
    @staticmethod
    def encode_sentences(data_input: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
        '''
        Encode sentences using SentenceTransformer
        '''
        sentences = data_input.tolist()  # Extract sentences into a list
        embeddings = model.encode(sentences) # Encode the sentences
        return embeddings
    
    @staticmethod
    def encode_sentence(sentence: str, model: SentenceTransformer) -> np.ndarray:
        '''
        Encode sentence using SentenceTransformer
        '''
        embedding = model.encode(sentence)
        return embedding

    @staticmethod    
    def fill_head_zeros(binary_str: str, fixed_length: int) -> str:
        """
        Fill head zeros to get binary string with fixed length
        """
        if len(binary_str) >= fixed_length:
            return binary_str
        return '0' * (fixed_length - len(binary_str)) + binary_str

    @staticmethod
    def fill_head_zeros_list(binary_str_list: list[str], fixed_length: int) -> list[str]:
        """
        Fill head zeros with fixed length to a list of binary string
        """
        return [Utils.fill_head_zeros(binary_str, fixed_length) for binary_str in binary_str_list]

    
    @staticmethod
    def convert_dataset(dataset: dict, attribute_keys: list[str], label_key: str) -> list[dict]:
        """
        Convert dataset with attributes and label
        [{'id': id, 'attributes': {'attribute1': value1, 'attribute2': value2, ...}, 'label': label}...]
        """
        num_points = len(dataset['id'])
        converted_dataset = []
        for index in range(num_points):
            attributes_dict = {}
            for key in attribute_keys:
                attributes_dict[key] = dataset[key][index]

            converted_dataset.append({
                'id': dataset['id'][index],
                'attributes': attributes_dict,
                'label': dataset[label_key][index]
            })
        return converted_dataset

    @staticmethod
    def hamming_distance(point_1: str, point_2: str) -> int:
        """
        Calculate Hamming distance
        """
        # Create character arrays
        point_1_chars = np.array(list(map(list, point_1)))
        point_2_chars = np.array(list(map(list, point_2)))

        distance = np.sum(point_1_chars != point_2_chars)
        return distance

    @staticmethod
    def mahattan_distance(point_1: np.ndarray, point_2: np.ndarray) -> float:
        '''
        Caculate Mahattan distance
        '''
        return np.sum(np.abs(point_1 - point_2))
    
    @staticmethod
    def calculate_distance_matrix(input_list: list[dict], attribute_key: str, method: str='cosine') -> np.ndarray:
        """
        Calculate distance matrix with different methods
        """
        attribute_points = np.array([point['attributes'][attribute_key] for point in input_list])

        if method=='cosine':
            # Check if the elements are already NumPy arrays
            if isinstance(attribute_points[0], np.ndarray):
                # Reshape to 2D for cosine_distances if necessary
                attribute_points = attribute_points.reshape(attribute_points.shape[0], -1)
            else:
                # Convert the list of sparse matrices to a 3D NumPy array
                attribute_points = np.array([item.toarray() for item in attribute_points])
                # Reshape to 2D for cosine_distances
                attribute_points = attribute_points.reshape(attribute_points.shape[0], -1)
            
            distances = cosine_distances(attribute_points, attribute_points) - 1
        else:
            num_points = len(input_list)
            distances = np.zeros((num_points, num_points))

            for i in range(num_points):
                for j in range(i + 1, num_points):
                    if method=='digits':
                        distance = Utils.hamming_distance(attribute_points[i], attribute_points[j])
                    elif method=='number':
                        distance = Utils.mahattan_distance(attribute_points[i], attribute_points[j])
                    else:
                        # else return one distance
                        distance = int(attribute_points[i] != attribute_points[j])
                    distances[i, j] = distances[j, i] = np.abs(distance)
        return np.abs(distances)

    @staticmethod
    def min_max_scaler_matrix(data_matrix: np.ndarray) -> np.ndarray:
        """
        Scaler each column of a matrix using Min-Max
        """
        maxs = np.max(data_matrix, axis=0, keepdims=True)
        mins = np.min(data_matrix, axis=0, keepdims=True)
        ranges = maxs - mins

        # Handle cases where standard deviation is zero (to avoid division by zero)
        ranges[ranges == 0] = 1 # or np.finfo(float).eps to use smallest possible float value

        scalered_matrix = (data_matrix - mins) / ranges
        return scalered_matrix

    @staticmethod
    def z_score_scaler_matrix(data_matrix: np.ndarray) -> np.ndarray:
        """
        Scaler each column of a matrix using Z-score
        """
        means = np.mean(data_matrix, axis=0, keepdims=True)  # Calculate mean of each column
        stds = np.std(data_matrix, axis=0, keepdims=True)   # Calculate standard deviation of each column

        # Handle cases where standard deviation is zero (to avoid division by zero)
        stds[stds == 0] = 1 # or np.finfo(float).eps to use smallest possible float value

        scalered_matrix = (data_matrix - means) / stds
        return scalered_matrix

    @staticmethod
    def calculate_std_distance_matrix(input_list: list[dict], attribute_key: str, method: str='cosine', scaler: str='min-max') -> np.ndarray:
        """
        Calculate distance matrix and standardize
        """
        if scaler == 'z-score':
            return Utils.z_score_scaler_matrix(Utils.calculate_distance_matrix(input_list, attribute_key, method))
        return Utils.min_max_scaler_matrix(Utils.calculate_distance_matrix(input_list, attribute_key, method))

    @staticmethod
    def create_label_dict(dataset: list[dict], encoding_model: SentenceTransformer) -> dict:
        """
        Create label dictionary:
            id: {'label': label, 'embedding': embedding}
        """
        labels = {}
        for record in dataset:
            labels[record['id']] = {'label': record['label'], 'embedding': Utils.encode_sentence(record['label'], encoding_model)}

        return labels

    @staticmethod
    def add_ones(matrix: np.ndarray) -> np.ndarray:
        """
        Add ones into matrix (N, d) to [1 matrix] (N, d+1)
        """
        return np.hstack((np.ones((len(matrix), 1)), matrix))

    @staticmethod
    def exp_points(points: np.ndarray, lamdas: np.ndarray):
        """
        Transform each point X to exp(lamda * X)
        """
        return np.exp(lamdas * points)
    
    @staticmethod
    def preprocess_input(attribute_matrixes: list[np.ndarray], lambdas: np.ndarray=None) -> np.ndarray:
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
            target_attribute_relation_list = [np.delete(attribute_matrixes[attribute][target], target) for attribute in range(d)]

            # Add a new dimension to each array
            target_attribute_relation_array = [np.expand_dims(relation, axis=1) for relation in target_attribute_relation_list]

            # Concatenate the arrays along the column dimension
            target_attribute_vector = np.concatenate(target_attribute_relation_array, axis=1)

            # Calculate value by preprocessing functions to [1 e^(-lambda * X)]
            target_value_vector = Utils.add_ones(Utils.exp_points(target_attribute_vector, (-1) * lambdas))
            value_matrix.append(target_value_vector)
        return np.array(value_matrix)
    
    @staticmethod
    def choose_top_N(X: np.ndarray, W: np.ndarray, N: int=10) -> tuple[np.ndarray, np.ndarray]:
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
    
    @staticmethod
    def calculate_delta(predicts: list[str], target: str, ids: list[str], labels: dict) -> np.ndarray:
        """
        Calculate Delta matrix by embeddings cosine distance
        """
        embedded_y_predicts = np.array([labels[ids[predict]]['embedding'] for predict in predicts])  # N x embedding_dim
        embedded_y_target = np.array([labels[ids[target]]['embedding']])  # 1 x embedding_dim

        # Calculate cosine distances
        delta = cosine_distances(embedded_y_predicts, embedded_y_target) - 1

        return delta
    
    @staticmethod
    def random_pick_unique(lst: list, N: int):
        """
        Picks N unique random elements from a list.
        """
        if N > len(lst):
            raise ValueError("Sample size is larger than population")
        return random.sample(lst, N)