import requests
from IPython.display import IFrame, display
import torch
import random
from tqdm import tqdm
import pandas as pd
from sae_lens import (
    SAE,
    HookedSAETransformer,
)
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
# Defining the functions to fetch the feature names and explanations


class Explainer():

    def __init__(self, model='gemma-2-9b-it', neuropedia_key=None):
        """
        Initializes the Explainer class with the specified model and Neuropedia API key.

        Args:
            model (str): The name of the model to be used. Default is 'gemma-2-9b-it'.
            neuropedia_key (str, optional): The API key for accessing Neuropedia services. Default is None.
        """
        self.model = model
        self.neuropedia_key = neuropedia_key

    def get_feature_name(self, index: int) -> str:
        """
        Fetches the feature name description from the Neuronpedia API for a given index.
        Args:
            index (int): The index of the feature to retrieve.
        Returns:
            str: The description of the feature.
        """
        url = f'https://www.neuronpedia.org/api/feature/{self.model}/20-gemmascope-res-131k/{index}'

        headers = {
            'X-Api-Key': self.neuropedia_key
        }

        response = requests.get(url, headers=headers)
        response = response.json()
        return response['explanations'][0]['description']

    def get_similar_features(self, feature, top_n=3):
        """
        Fetches similar features from the Neuronpedia API for a given index.
        Args:
            index (int): The index of the feature to retrieve.
            top_n (int, optional): The number of similar features to return. Defaults to 3.
        Returns:
            list: A list of similar features.
        """
        top_similar_indexes = feature['neuron']['topkCosSimIndices'][1:top_n+1]
        top_similar_values = feature['neuron']['topkCosSimValues'][1:top_n+1]
        top_similar_features = [self.get_feature_name(
            x) for x in top_similar_indexes]

        for feature, value in zip(top_similar_features, top_similar_values):
            print(f'- {feature} - Cosine Similarity: {value}')

        return [(feature, value) for feature, value in zip(top_similar_features, top_similar_values)]

    def get_explanation(self, text, indexes=[], density_threshold=0.5, n_results=10):
        """
        Fetches explanations from the GemmaScope API for a given text using a specified model.
        Args:
            text (str): The input text for which explanations are to be generated.
            indexes (list): The indexes of the features to be explained.
            n_results (int, optional): The number of results to return. Defaults to 10.
        Returns:
            list: A list of explanation results from the GemmaScope API.
        """
        url = 'https://www.neuronpedia.org/api/search-all'
        payload = {
            'modelId': self.model,
            'sourceSet': 'gemmascope-res-131k',
            'text': text,
            'selectedLayers': ['20-gemmascope-res-131k'],
            'sortIndexes': indexes,
            'ignoreBos': True,
            'densityThreshold': density_threshold,
            'numResults': n_results,
        }

        headers = {
            'Content-Type': 'application/json',
            'X-Api-Key': self.neuropedia_key
        }

        response = requests.post(url, json=payload, headers=headers)
        while response.status_code != 200:
            print('Retrying GemmaScope explanation...')
            print(response)
            response = requests.post(url, json=payload, headers=headers)
        response = response.json()
        results = response['result']
        return results

    def print_result(self, res):
        print(f"Description of feature {res['neuron']['index']}: {res['neuron']['explanations'][0]['description']}")
        # print(f'Values: {res["values"]}')
        print(f"Highest activation: {res['maxValue']}, sum of activations: {round(sum(res['values']), 2)}")
        print(f"Activation density: {res['neuron']['frac_nonzero'] * 100}%")
        print('Similar features:\n')
        self.get_similar_features(res, 3)
        print('-------------------')


# from IPython.display import IFrame

# html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"


# def get_dashboard_html(sae_release=model, sae_id="20-gemmascope-res-131k", feature_idx=0):
#     return html_template.format(sae_release, sae_id, feature_idx)


# html = get_dashboard_html(
#     sae_release=model, sae_id="20-gemmascope-res-131k", feature_idx=8213
# )
# IFrame(html, width=1200, height=600)

class LocalExplainer():

    def __init__(self, mname='gpt2-small', sae_name='gpt2-small-res-jb', layer='blocks.7.hook_resid_pre', device='cuda'):
        """
        Initializes the Explainer class with the specified model and layer.

        Args:
            mname (str): The name of the model to be used. Default is 'gpt2-small'.
            sae_name (str): The name of the SAE model to be used. Default is 'gpt2-small-res-jb'.
            layer (str): The specific layer of the model to be used. Default is 'blocks.7.hook_resid_pre'.
            device (str): The device to run the model on. Default is 'cuda'.

        Attributes:
            mname (str): The name of the model.
            sae_name (str): The name of the SAE model.
            layer (str): The specific layer of the model.
            model (HookedSAETransformer): The loaded model.
            sae (SAE): The loaded SAE model.
            cfg_dict (dict): Configuration dictionary for the SAE model.
            autointerp_df (pd.DataFrame): DataFrame containing auto-interpretation data filtered by layer.
            idx_to_exp (dict): Dictionary mapping indices to explanations.
        """
        self.mname = mname
        self.sae_name = sae_name
        self.layer = layer
        self.model = HookedSAETransformer.from_pretrained(mname, device=device)
        self.sae, self.cfg_dict, _ = SAE.from_pretrained(
            release=sae_name,
            sae_id=layer,
            device=device,
        )
        self.autointerp_df = self.get_autointerp_df()
        self.autointerp_df = self.autointerp_df[self.autointerp_df['layer'].str.startswith(
            str(self.sae.cfg.hook_layer))]

        self.idx_to_exp = {idx: exp for idx, exp in zip(
            self.autointerp_df['index'], self.autointerp_df['description'])}

    def display_dashboard(
        self,
        latent_idx=0,
        width=800,
        height=600,
    ):
        release = get_pretrained_saes_directory()[self.sae_name]
        neuronpedia_id = release.neuronpedia_id[self.layer]

        url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

        print(url)
        display(IFrame(url, width=width, height=height))

    def get_autointerp_df(self) -> pd.DataFrame:
        """
        Fetches and returns a DataFrame containing auto-interpretation data from Neuronpedia.

        This method retrieves the auto-interpretation data for a specific model and layer
        from the Neuronpedia API. The model and layer information are determined by the
        `sae_name` and `layer` attributes of the instance.
s
        Returns:
            pd.DataFrame: A DataFrame containing the auto-interpretation data.
        """
        release = get_pretrained_saes_directory()[self.sae_name]
        neuronpedia_id = release.neuronpedia_id[self.layer]

        url = "https://www.neuronpedia.org/api/explanation/export?modelId={}&saeId={}".format(
            *neuronpedia_id.split("/"))
        headers = {"Content-Type": "application/json"}
        response = requests.get(url, headers=headers)

        data = response.json()
        return pd.DataFrame(data)

    def get_explanation(self, prompt, indexes=[], density_threshold=0.5, n_results=10):
        """
        Fetches explanations from the SAE for a given text using a specified model.
        Args:
            prompt (str): The input text for which explanations are to be generated.
            density_threshold (float, optional): The threshold for the density of the latent space. Defaults to 0.1.
            n_results (int, optional): The number of results to return. Defaults to 10.
        Returns:
            list: A list of explanation results from the SAE.
        """
        _, cache = self.model.run_with_cache_with_saes(prompt, saes=[self.sae])

        sae_acts_post = cache[f"{self.sae.cfg.hook_name}.hook_sae_acts_post"][0, 1:, :].sum(dim=0)

        results = []
        for act, ind in zip(*sae_acts_post.topk(100)):
            url = f'https://www.neuronpedia.org/api/feature/{get_pretrained_saes_directory()[self.sae_name].neuronpedia_id[self.layer]}/{ind}'
            frac_nonzero = requests.get(url).json()['frac_nonzero'] * 100
            if frac_nonzero >= density_threshold:
                # print(f"SKIPPED - Latent {ind} - {self.idx_to_exp.get(str(int(ind)))} had activation {act:.2f} with density {frac_nonzero:.2f}%")
                continue
            results.append(
                {
                    'index': ind.item(),
                    'description': self.idx_to_exp.get(str(int(ind.item()))),
                    'density': frac_nonzero,
                    'values': act.item(),
                }
            )
            # print(f"ADDED - Latent {ind} - {self.idx_to_exp.get(str(int(ind)))} had activation {act:.2f} with density {frac_nonzero:.2f}%")
            if len(results) >= n_results:
                break
        return results
