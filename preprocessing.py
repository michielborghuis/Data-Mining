from collections import Counter
from typing import List
import string

import numpy as np
import nltk
from nltk.corpus import stopwords

#nltk.download('stopwords')

class ReviewProcessor:
    def __init__(self) -> None:
        self.translator = str.maketrans('', '', string.punctuation)
        self.stop_words = set(stopwords.words('english'))
    
    def _lowercase_review(self, review: str) -> str:
        return review.lower()
    
    def _strip_review(self, review: str) -> str:
        return review.strip()
    
    def _remove_punctuation(self, review: str) -> str:
        return review.translate(self.translator)
    
    def _get_unigrams(self, review: str) -> List[str]:
        return review.split()
    
    def _get_bigrams(self, tokens: List[str]) -> List[str]:
        return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    
    def _process_review(self, review: str) -> str:
        review = self._lowercase_review(review)
        review = self._remove_punctuation(review)
        review = self._strip_review(review)
        return review
        
    def _get_token_indices(self, review: str, train: bool=True, include_bigrams: bool=False) -> List[str]:
        # Get 'tokens' (i.e. unigrams and/or bigrams in current review)
        tokens = self._get_unigrams(review)
        if include_bigrams:
            tokens = tokens + self._get_bigrams(tokens)
        
        # Convert list of tokens to list of token indices
        token_indices = []
        for token in tokens:
            # Filter out stopwords
            if token in self.stop_words:
                continue

            # Add token to vocabulary (NOTE: ignores tokens in test that are not in train)
            if token not in self.token_index_dict:
                if train:
                    idx = len(self.token_index_dict)
                    self.token_index_dict[token] = idx
                    self.index_token_dict[idx] = token

                    token_indices.append(self.token_index_dict[token])
            else:
                token_indices.append(self.token_index_dict[token])
        
        return token_indices

    def process_train_reviews(self, reviews: np.ndarray, include_bigrams: bool=False) -> np.ndarray:
        """
        Converts a list of reviews into a bag-of-words counts matrix, 
        with the number of times review i contains word j at position (i, j).
        """
        
        # Reset token/index mappings
        self.token_index_dict = {}
        self.index_token_dict = {}

        # Convert raw text reviews into lists of token indices (builds mappings)
        token_indices = []
        for review in reviews:
            review = self._process_review(review) # Perform generic preprocessing steps
            token_indices.append(self._get_token_indices(review, train=True, include_bigrams=include_bigrams))

        # Use final token to index mapping to create counts matrix
        count_matrix = np.zeros((len(reviews), len(self.token_index_dict)))
        for review_idx, cur_token_indices in enumerate(token_indices):
            counts = Counter(cur_token_indices)
            count_matrix[review_idx, list(counts.keys())] = list(counts.values())

        return count_matrix
    
    def process_test_reviews(self, reviews: np.ndarray, include_bigrams: bool=False) -> np.ndarray:
        """
        Converts a list of reviews into a bag-of-words counts matrix, 
        with the number of times review i contains word j at position (i, j).
        """

        # Convert raw text reviews into lists of token indices
        token_indices = []
        for review in reviews:
            review = self._process_review(review) # Perform generic preprocessing steps
            token_indices.append(self._get_token_indices(review, train=False, include_bigrams=include_bigrams))

        # Use token to index mapping to create counts matrix
        count_matrix = np.zeros((len(reviews), len(self.token_index_dict)))
        for review_idx, cur_token_indices in enumerate(token_indices):
            counts = Counter(cur_token_indices)
            count_matrix[review_idx, list(counts.keys())] = list(counts.values())

        return count_matrix
    
    def filter_rare_terms(self, count_matrix: np.ndarray, min_review_freq: float) -> np.ndarray:
        # Filter out terms (unigrams/bigrams) that occur only in relatively few reviews

        assert count_matrix.shape[1] == len(self.token_index_dict), "Count matrix dimension does not match vocabulary"

        occurence_matrix = (count_matrix > 0)*1.0
        review_frequencies = np.sum(occurence_matrix/occurence_matrix.shape[0], axis=0)
        drop_token_indices = np.where(review_frequencies < min_review_freq)[0]
        
        new_token_index_dict = {}
        new_index_token_dict = {}
        new_matrix = np.zeros((count_matrix.shape[0], count_matrix.shape[1]-len(drop_token_indices)))
        for i in range(len(self.token_index_dict)):
            if i not in drop_token_indices:
                new_index = len(new_token_index_dict)
                new_matrix[:,new_index] = count_matrix[:,i]
                new_token_index_dict[self.index_token_dict[i]] = new_index
                new_index_token_dict[new_index] = self.index_token_dict[i]

        self.token_index_dict = new_token_index_dict
        self.index_token_dict = new_index_token_dict

        return new_matrix