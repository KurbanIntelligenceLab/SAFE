# Importing the required libraries
import os
import math
from collections import Counter
from dotenv import load_dotenv
from explanation import Explainer, LocalExplainer
from generation import Generator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import torch

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selfcheck_nli = SelfCheckNLI(device=device) # set device to 'cuda' if GPU is available



def semantic_entropy(cluster_assignments):
    """Calculate semantic 
      based on cluster assignments."""
    frequency = Counter(cluster_assignments)
    total_count = len(cluster_assignments)
    entropy = 0
    for count in frequency.values():
        prob = count / total_count
        entropy += prob * math.log(prob)
    return -entropy

def selfcheck(ground, samples):
  sent_scores_nli = selfcheck_nli.predict(
      sentences = [ground],                          # lowtemp answer
      sampled_passages = samples, # list of hightemp
  )
  return sent_scores_nli[0]

class EnrichmentFramework():
    def __init__(self, gen_repository: str, model_gen:str, sae_name: str, layer: str, emb_repository: str, openrouter_key: str,
                 neuronpedia_key: str, n_answers: int = 10, high_temperature: float = 1.0,
                 low_temperature: float = 0.1, local_explainer=True, device='cuda'):
        """
        Initializes the EnrichmentFramework with the given parameters.

        Args:
            gen_repository (str): Path to the generator repository.
            emb_repository (str): Path to the embedding repository.
            openrouter_key (str): API key for OpenRouter.
            neuronpedia_key (str): API key for Neuronpedia.
            n_answers (int, optional): Number of answers to generate for each question. Defaults to 10.
            high_temperature (float, optional): Temperature for generating high-temperature answers. Defaults to 1.0.
            low_temperature (float, optional): Temperature for generating low-temperature answers. Defaults to 0.1.
            local_explainer (bool, optional): Whether to use the local explainer. Defaults to True.
            device (str, optional): Device to use for the local explainer. Defaults to 'cuda'.

        Attributes:
            gen_repository (str): Path to the generator model repository.
            model (str): Model name extracted from the generator repository path.
            explainer (Explainer): Instance of the Explainer class.
            generator_high (Generator): Instance of the Generator class initialized with the high-temperature generator.
            generator_low (Generator): Instance of the Generator class initialized with the low-temperature generator.
            embedder (SentenceTransformer): Instance of the SentenceTransformer class initialized with the embedding repository.
            n_answers (int): Number of answers to generate for each question.
        """
        self.gen_repository = gen_repository
        self.model = gen_repository.split('/')[1]
        self.model_gen = model_gen
        if local_explainer:
            logging.info(
                f'Using local explainer: {gen_repository} - {sae_name} - {layer}')
            self.explainer = LocalExplainer(
                gen_repository, sae_name, layer, device=device)
        else:
            logging.info(f'Using remote explainer: {gen_repository}')
            self.explainer = Explainer(model, neuronpedia_key)
        self.generator_high = Generator(
            gen_repository, model_gen, openrouter_key, temperature=high_temperature)
        self.generator_low = Generator(
            gen_repository, model_gen, openrouter_key, temperature=low_temperature)
        self.embedder = SentenceTransformer(emb_repository, device=device)
        self.n_answers = n_answers
        logging.info(
            f'EnrichmentFramework initialized with model: {self.model}')

    def get_qa_embedding(self, question, answer=None, is_question=False):
        """
        This function takes a question and answer as input and returns the embedding from Sentence-BERT.
        It uses the Sentence-BERT model to generate sentence embeddings directly.

        Parameters:
        question (str): The input question.
        answer (str): The input answer. If None, only the question is considered.

        Returns:
        cls_embedding_np (numpy array): The sentence embedding representing the question and answer.
        """

        # Prepare input text
        if is_question or answer is None:
            input_text = question
        else:
            input_text = f"{question} [SEP] {answer}"

        # Generate the embedding using Sentence-BERT
        cls_embedding_np = self.embedder.encode(input_text, show_progress_bar=False)

        return cls_embedding_np

    def clustering_answers(self, question, answers):
        """
        Clusters the provided answers based on their similarity to the given question.

        This method computes the embeddings for each answer in relation to the question,
        calculates the cosine similarity between these embeddings, and then performs
        agglomerative clustering on the resulting distance matrix.

        Parameters:
        question (str): The question to which the answers are related.
        answers (list): A list of answers to be clustered.

        Returns:
        list: A list of cluster labels corresponding to each answer.
        """
        response_embeddings = [self.get_qa_embedding(
            question, str(a)) for a in answers]
        cosine_sim_matrix = cosine_similarity(response_embeddings)
        cosine_dist_matrix = 1 - cosine_sim_matrix

        try:
            clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=0.1, affinity='precomputed', linkage='average')
            cluster_labels = clustering.fit_predict(cosine_dist_matrix)
        except TypeError:
            clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=0.1, metric='precomputed', linkage='average')
            cluster_labels = clustering.fit_predict(cosine_dist_matrix)

        return list(cluster_labels)

    def get_diffs(self, explanation_q, answer, density_threshold: float = 0.1):
        """
        Computes the difference between the features of the explanation for a given question and answer.

        Args:
            explanation_q (list): The explanation for the question, where each element is a dictionary containing feature details.
            answer (str): The answer for which the explanation is to be generated.
            density_threshold (float, optional): The threshold value to determine the density of the explanation. Defaults to 0.1.

        Returns:
            list: A list of features that are present in the explanation for the answer but not in the explanation for the question.
        """
        # Get the explanation for the answer
        a_indexes = self.generator_high.get_question_indexes(answer)
        explanation_a = self.explainer.get_explanation(
            answer, a_indexes, density_threshold, 5)
        logging.debug(f'Explanation for answer: {explanation_a}')

        # Get the features of the explanation for the question and the answer
        # This commented code is for the API explainer. I don't think we will use it again!
        # features_q = set([feature['neuron']['explanations'][0]
        #                  ['description'] for feature in explanation_q])
        # features_a = set([feature['neuron']['explanations'][0]
        #                  ['description'] for feature in explanation_a])
        features_q = set([feature['description'] for feature in explanation_q])
        features_a = set([feature['description'] for feature in explanation_a])
        # Get the difference between the features
        features_diff = list(features_a - features_q)
        logging.debug(f'Feature differences computed: {features_diff}')
        return features_diff

    def embed_answer_features(self, diffs: list):
        """
        Embeds the features extracted from the given list of differences.

        Args:
            diffs (list): A list of lists, where each inner list contains features.

        Returns:
            numpy.ndarray: An array of embeddings for the union of all features.
        """
        # Union of all features
        all_features = list(set().union(*diffs))
        logging.info(f'Embedding {len(all_features)} features.')
        # Embed all features
        embeddings = self.embedder.encode(all_features, show_progress_bar=False)
        logging.debug(f'Embeddings generated: {embeddings}')
        return embeddings

    # def compute_sparsity(self, embeddings) -> float:
    #     """
    #     Compute the sparsity of the given embeddings.
    #     Sparsity is calculated as 1 minus the average similarity of the embeddings.

    #     Args:
    #         embeddings (numpy.ndarray): An array where each item represents an embedding vector.

    #     Returns:
    #         float: The sparsity value, which is 1 minus the average similarity of the embeddings.
    #     """
    #     # Calculate the similarity matrix
    #     similarity_matrix = self.embedder.similarity(embeddings, embeddings)

    #     # Extract upper triangle (excluding diagonal) of the similarity matrix
    #     n = len(similarity_matrix)
    #     upper_triangle = [similarity_matrix[i, j]
    #                       for i in range(n) for j in range(i + 1, n)]

    #     # Calculate the average similarity
    #     avg_similarity = np.mean(upper_triangle) if upper_triangle else 0
    #     sparsity = 1 - avg_similarity
    #     logging.info(f'Sparsity computed: {sparsity:.3f}')
    #     return sparsity
    
    def compute_sparsity(self, embeddings) -> bool:
        """
        Compute the sparsity of the given embeddings and check if there is an outlier.
        Sparsity is calculated as 1 minus the average similarity of the embeddings.
        An outlier is defined as a similarity score below the lower bound (Q1 - 1.5 * IQR).

        Args:
            embeddings (numpy.ndarray): An array where each item represents an embedding vector.

        Returns:
            bool: True if there is an outlier, False otherwise.
        """
        # Calculate the similarity matrix
        similarity_matrix = self.embedder.similarity(embeddings, embeddings)

        # Extract upper triangle (excluding diagonal) of the similarity matrix
        n = len(similarity_matrix)
        upper_triangle = [similarity_matrix[i, j]
                        for i in range(n) for j in range(i + 1, n)]

        # Calculate IQR for outlier detection
        if upper_triangle:  # Check to avoid empty lists
            q1 = np.percentile(upper_triangle, 25)
            q3 = np.percentile(upper_triangle, 50)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr

            # Check if any value in the upper triangle is below the lower bound
            outlier_present = any(score < lower_bound for score in upper_triangle)
        else:
            lower_bound = 0
            outlier_present = False

        logging.info(f'Lower bound computed: {lower_bound:.3f}')
        logging.info(f'Outlier present: {outlier_present}')

        return outlier_present

    def enrich_question(self, question: str, diff_sets: list, embed_answer_features: list, sparsity: float, threshold: float, features_added: list) -> str:
        """
        Enriches a given question based on the similarity of its embedding with a set of answer features.

        Parameters:
        question (str): The question to be enriched.
        diff_sets (list of sets): A list of sets containing features present in the answer and not in the question.
        embed_answer_features (list): A list of embedded answer features.
        sparsity (float): A measure of how sparse the features are.
        threshold (float): A threshold value to determine whether features are considered sparse or not.
        features_added (list): A list of features that have been already added to the question.

        Returns:
        str: The enriched question with additional notes based on feature similarity.
        """
        # Concatenate all features
        all_features = set().union(*diff_sets)
        # Remove features that have already been added
        all_features = list(all_features - set(features_added))

        # Calculate similarity with question
        question_embedding = self.embedder.encode([question])
        similarities = self.embedder.similarity(
            question_embedding, embed_answer_features)

        # Find the most similar and most distant features
        most_similar_idx = np.argmax(similarities)
        most_distant_idx = np.argmin(similarities)
        most_similar_feature = all_features[most_similar_idx]
        most_distant_feature = all_features[most_distant_idx]
        logging.info(f'Most similar feature: {most_similar_feature}')
        logging.info(f'Most distant feature: {most_distant_feature}')

        new_question = question
        # If the sparsity is below the threshold, then all features are close and we remove the most distant feature
        if sparsity == True:
            if 'NOTE:' not in new_question:
                new_question = f'{new_question} - NOTE: do not consider {most_distant_feature}'
            else:
                new_question = f'{new_question} and do not consider {most_distant_feature}'
            features_added.append(most_distant_feature)
        else:
            # If the sparsity is above the threshold, then all features are sparse and we add the most similar feature
            if 'NOTE:' not in new_question:
                new_question = f'{new_question} - NOTE: you must consider {most_similar_feature}'
                
            else:
                new_question = f'{new_question} and you must consider {most_similar_feature}'
            features_added.append(most_similar_feature)
        logging.info(f'Enriched question: {new_question}')
        return new_question, features_added

    def __call__(self, question: str, entropy_threshold: float = 1.0, density_threshold: float = 0.1, sparsity_threshold: float = 0.7) -> str:
        """
        Processes a given question to generate an enriched version based on generated answers and their feature differences.
        Args:
            question (str): The question to be processed.
            entropy_threshold (float, optional): The threshold value to determine the entropy of the explanation. Defaults to 1.0.
            density_threshold (float, optional): The threshold value to determine the density of the explanation. Defaults to 0.1.
            sparsity_threshold (float, optional): The threshold value to determine the strategy for enriching the question. Defaults to 0.7.
        Returns:
            str: The enriched version of the input question.
        """
        logging.info(f'Processing question: {question}')
        new_question = question
        diff_sets = []
        sparsity = 0
        features_added = []
        # Find the entropy and answers
        logging.info(f'Generating {self.n_answers} answers for the question.')
        low_temp_answer = self.generator_low.generate(question)
        answers = [self.generator_high.generate(
            question) for _ in range(self.n_answers)]
        logging.info(f'Generated answers: {answers}')

        # cluster_assignments = self.clustering_answers(question, answers)
        #first_entropy = semantic_entropy(cluster_assignments)
        first_entropy = selfcheck(low_temp_answer,answers)
        final_entropy = first_entropy
        logging.info(f'Initial selfcheck score: {first_entropy:.3f}')

        loop_count = 0
        while loop_count < 3:
            if final_entropy <= entropy_threshold:
                logging.info("selfcheck score threshold met. Stopping enrichment.")
                logging.info('----------------------------------------')
                break

            loop_count += 1
            logging.info(f'Starting loop number {loop_count}')

            # Generate the explanation for the question
            q_indexes = self.generator_high.get_question_indexes(new_question)
            explanation_q = self.explainer.get_explanation(
                new_question, q_indexes, density_threshold, 5)

            # Process each question-answer pair and check alignment
            logging.info('Computing feature differences for each answer.')
            diff_sets = [self.get_diffs(explanation_q, answer, density_threshold)
                         for answer in answers]

            # Calculate sparsity
            logging.info('Embedding answer features and computing sparsity.')
            answer_feature_embeddings = self.embed_answer_features(diff_sets)
            sparsity = self.compute_sparsity(answer_feature_embeddings)

            # Based on the sparsity and the threshold, find the strategy to enrich the question
            new_question, features_added = self.enrich_question(
                new_question, diff_sets, answer_feature_embeddings, sparsity, sparsity_threshold, features_added)

            # Generate the answer to the enriched question
            answers = [self.generator_high.generate(
                new_question) for _ in range(self.n_answers)]
            logging.info(f'Answers to enriched question: {answers}')

            # cluster_assignments = self.clustering_answers(question, answers)
            final_entropy = selfcheck(low_temp_answer,answers)
            # final_entropy = semantic_entropy(cluster_assignments)
            logging.info(f'New score: {final_entropy:.3f}')
            logging.info('----------------------------------------')
        logging.info('----------------------------------------')
        return (question, new_question, low_temp_answer, answers, diff_sets, sparsity, first_entropy, final_entropy, loop_count)

if __name__ == '__main__':
    df = pd.read_parquet(
        'hf://datasets/truthfulqa/truthful_qa/generation/validation-00000-of-00001.parquet')

    # df = pd.read_csv('bioasq_questions.csv')
    
    # df = pd.read_json("hf://datasets/medalpaca/medical_meadow_medical_flashcards/medical_meadow_wikidoc_medical_flashcards.json")

    # df = pd.read_parquet("hf://datasets/pminervini/HaluEval/qa/data-00000-of-00001.parquet")
    load_dotenv()

    # Setting the API key and the model repository
    NEURONPEDIA_KEY = os.environ.get('NEURONPEDIA_KEY')
    OPENROUTER_KEY = os.getenv('OPENROUTER_KEY')
    
    embed_repository = 'sentence-transformers/all-MiniLM-L6-v2'
    
    ##Gemma-2-9b
    repository_gen = 'google/gemma-2-9b-it'
    repository = repository_gen
    # model = repository.split('/')[1]
    # sae_name = 'gemma-scope-9b-it-res-canonical'
    # layer = 'layer_20/width_131k/canonical'
    
    ##Llama - 3
    repository_gen = 'meta-llama/llama-3-8B-Instruct'
    repository = 'meta-llama/Meta-Llama-3-8B-Instruct'
    model = repository.split('/')[1]
    sae_name = 'llama-3-8b-it-res-jh' 
    layer = 'blocks.25.hook_resid_post'

        # Configure logging for real run
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('enrichment_framework.log'),  # Log to a file
            logging.StreamHandler()  # Log to console
        ]
    )

    # Process all questions (truthfulqa has 817 questions)
    enricher = EnrichmentFramework(
        repository, repository_gen, sae_name, layer, embed_repository, OPENROUTER_KEY, NEURONPEDIA_KEY, n_answers=10)
    logging.info('Processing all questions...')
    logging.info(f'Dataset size: {len(df)}')
    results = []
    for _, row in tqdm(df.head(400).iterrows()):
        question = row['question']
        try:
            results.append(enricher(question, entropy_threshold=0.6,
                    density_threshold=0.05, sparsity_threshold=0.6))
        except Exception as e:
            logging.warning(f"Skipping question due to error: {e}")
        # results.append(enricher(question, entropy_threshold=0.6,
        #             density_threshold=0.05, sparsity_threshold=0.6))

    # # Save the enriched questions
    enriched_df = pd.DataFrame(results, columns=[
                            'question', 'enriched_question', 'low_temp_answer', 'answers',
                            'diff_sets', 'sparsity', 'first_score', 'final_score', 'loop_count'])
    enriched_df.to_csv('results/truthfulqa_llama_enriched_SELFCHECK.csv', index=False)