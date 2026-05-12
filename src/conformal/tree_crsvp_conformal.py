import random
import copy
from typing import Optional
import numpy as np

from tqdm import tqdm
from datetime import datetime

from src.conformal.utils import compute_threshold
from src.conformal.base_conformal import BaseConformal
from src.aggregator.base_aggregator import BaseAggregator
from src.aggregator.sum_aggregator import SumAggregator
from src.logger import get_logger

class TreeHierarchicalRestrictConformal(BaseConformal):
    def __init__(self, random_seed: int = 42, aggregator: Optional[BaseAggregator] = None):
        super().__init__()
        log_name = f"logs/{__name__}_{datetime.now().timestamp()}.log"
        self.logger = get_logger(name=f"{__name__}_{datetime.now().timestamp()}", log_file=log_name)
        self.random_seed = random_seed
        self.aggregator = aggregator if aggregator is not None else SumAggregator()
        self.rng = np.random.default_rng(random_seed)


    def initialize(self, data, seed=None):
        # Take data from node_processor, all nodes are considered as leaf nodes
        shuffled_results = copy.deepcopy(data)
        if seed is not None:
            random.seed(seed)
        random.shuffle(shuffled_results)

        mid = len(shuffled_results) // 2
        calib_data = shuffled_results[:mid]
        test_data = shuffled_results[mid:]
        return calib_data, test_data
    
    def compute_tau_star(self, alpha, calib_data):
        leaf_nodes_scores = []
        for d in calib_data:
            r_score = self.compute_restricted_set_rscore(d)
            leaf_nodes_scores.append(r_score)
        # form Intermediate nodes until root node by binary merging from leaf nodes

        self._record_calibration_scores(leaf_nodes_scores)
        tau_star = compute_threshold(alpha=alpha, r_scores=leaf_nodes_scores)
        return tau_star
    
    def compute_restricted_set_rscore(self, data):
        true_fail, probabilities = data['true_fail'], data['probability']
        noise = data['noise'] if 'noise' in data else 0.0
        probabilities = np.clip(np.array(probabilities) + noise, 0, 1)
        
        # Step 1: Reconstruct binary tree structure
        tree = self._build_binary_tree(probabilities)
        
        # Step 2: Implement the hierarchical conformal algorithm
        r_score = self._compute_hierarchical_rscore(tree, true_fail)
        
        return r_score
    
    def _build_binary_tree(self, leaf_probabilities):
        """
        Build a binary tree from leaf probabilities.
        Each parent node's probability is the sum of its children's probabilities.
        
        Args:
            leaf_probabilities: List of probabilities for leaf nodes
            
        Returns:
            Dictionary representing the tree structure with node probabilities
        """
        n_leaves = len(leaf_probabilities)
        if n_leaves == 0:
            return {}
        
        if n_leaves == 1:
            tree = {
                'leaf_0': {
                    'probability': leaf_probabilities[0],
                    'is_leaf': True,
                    'leaf_index': 0,
                    'children': [],
                    'parent': None
                },
                'root': 'leaf_0'
            }
            return tree
        
        tree = {}
        node_counter = 0
        
        # Initialize leaf nodes
        current_level = []
        for i, prob in enumerate(leaf_probabilities):
            node_key = f"leaf_{i}"
            tree[node_key] = {
                'probability': prob,
                'is_leaf': True,
                'leaf_index': i,
                'children': [],
                'parent': None
            }
            current_level.append(node_key)
        
        # Build tree bottom-up
        level = 0
        while len(current_level) > 1:
            next_level = []
            
            # Pair adjacent nodes to create parents
            i = 0
            while i < len(current_level):
                left_child_key = current_level[i]
                right_child_key = current_level[i + 1] if i + 1 < len(current_level) else None
                
                # Create parent node
                parent_key = f"internal_{level}_{node_counter}"
                node_counter += 1
                
                parent_prob = tree[left_child_key]['probability']
                children = [left_child_key]
                
                if right_child_key:
                    parent_prob = self.aggregator.combine(parent_prob, tree[right_child_key]['probability'])
                    children.append(right_child_key)
                    i += 2
                else:
                    i += 1
                
                tree[parent_key] = {
                    'probability': parent_prob,
                    'is_leaf': False,
                    'leaf_index': None,
                    'children': children,
                    'parent': None
                }
                
                # Update parent pointers in children
                tree[left_child_key]['parent'] = parent_key
                if right_child_key:
                    tree[right_child_key]['parent'] = parent_key
                
                next_level.append(parent_key)
            
            current_level = next_level
            level += 1
        
        # Set root
        tree['root'] = current_level[0] if current_level else None
        
        return tree
    
    def _compute_hierarchical_rscore(self, tree, true_fail_index):
        """
        Implement the hierarchical conformal algorithm from the paper.
        
        Algorithm:
        1. Y_hat ← arg max_c∈Y_hat P(c|x_i)  # Find leaf with max probability
        2. p_hat_Y ← P(Y_hat|x_i), p_hat_Y' ← 0
        3. while y_i ∉ Y_hat do
        4:     p_hat_Y' ← p_hat_Y
        5:     Y_hat ← pa(Y_hat), p_hat_Y ← P(Y_hat|x_i)
        6: end while
        7: τ_i ← p_hat_Y - u_i·(p_hat_Y - p_hat_Y')
        
        Args:
            tree: Binary tree structure
            true_fail_index: Index of the true failure node
            
        Returns:
            r_score (τ_i): Conformal score for this example
        """
        if 'root' not in tree or tree['root'] is None:
            return 0.0
            
        # Step 1: Find leaf with maximum probability (Y_hat)
        max_prob = -1
        max_leaf_key = None
        
        for key, node in tree.items():
            if key != 'root' and node['is_leaf'] and node['probability'] > max_prob:
                max_prob = node['probability']
                max_leaf_key = key
        
        if max_leaf_key is None:
            return 1.0
        
        # Step 2: Initialize
        current_node_key = max_leaf_key
        p_hat_Y = tree[current_node_key]['probability']
        p_hat_Y_prime = 0.0
        
        # Step 3: Check if true_fail is in current set (initially just the max leaf)
        current_set = self._get_leaf_indices_in_subtree(tree, current_node_key)
        
        # Step 4-6: While true failure is not in current set, move up the tree
        
        while true_fail_index not in current_set:
            p_hat_Y_prime = p_hat_Y
            
            # Move to parent
            parent_key = tree[current_node_key]['parent']
            if parent_key is None:
                # Reached root, include all leaves
                current_set = self._get_leaf_indices_in_subtree(tree, tree['root'])
                p_hat_Y = tree[tree['root']]['probability']
                break
                
            current_node_key = parent_key
            p_hat_Y = tree[current_node_key]['probability']
            current_set = self._get_leaf_indices_in_subtree(tree, current_node_key)
        
        # Step 7: Compute τ_i
        # u_i is typically a uniform random variable [0,1]
        u_i = self.rng.uniform(0, 1)
        
        tau_i = p_hat_Y - u_i * (p_hat_Y - p_hat_Y_prime)
        
        # Ensure tau_i is in valid range [0, 1]
        tau_i = np.clip(tau_i, 0.0, 1.0)
        
        return tau_i
    
    def _get_leaf_indices_in_subtree(self, tree, node_key):
        """
        Get all leaf indices that are descendants of the given node.
        
        Args:
            tree: Binary tree structure
            node_key: Key of the current node
            
        Returns:
            Set of leaf indices in the subtree rooted at node_key
        """
        if tree[node_key]['is_leaf']:
            return {tree[node_key]['leaf_index']}
        
        leaf_indices = set()
        
        # Recursively collect leaf indices from children
        for child_key in tree[node_key]['children']:
            leaf_indices.update(self._get_leaf_indices_in_subtree(tree, child_key))
        
        return leaf_indices
    
    def evaluate_all(self, tau_star, test_data):
        correct_predictions = 0
        total_predictions = len(test_data)
        removal_rates = []

        for data in test_data:
            prediction_set = self._crsvp_inference(data, tau_star)

            if data['true_fail'] in prediction_set:
                correct_predictions += 1

            total_leaves = len(data['probability'])
            removed_leaves = total_leaves - len(prediction_set)
            removal_rates.append(removed_leaves / total_leaves if total_leaves > 0 else 0.0)

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_removal_rate = np.mean(removal_rates) if removal_rates else 0.0
        return accuracy, total_predictions, correct_predictions, avg_removal_rate
    
    def _crsvp_inference(self, data, tau_star):
        """
        Algorithm 2: CRSVP inference
        
        Input: data (containing probabilities), τ*, u
        Output: Set-valued prediction Y'_hat
        
        Args:
            data: Data containing probabilities and other info
            tau_star: Threshold from calibration
            
        Returns:
            Set of leaf indices in the prediction set
        """
        probabilities = data['probability']
        noise = data['noise'] if 'noise' in data else 0.0
        probabilities = np.clip(np.array(probabilities) + noise, 0, 1)
        
        # Build binary tree structure
        tree = self._build_binary_tree(probabilities)
        
        if 'root' not in tree or tree['root'] is None:
            return set()
        
        # Step 1: Y_hat ← arg max_c∈Y_hat P(c|x), Y' ← ∅
        max_prob = -1
        max_leaf_key = None
        
        for key, node in tree.items():
            if key != 'root' and node['is_leaf'] and node['probability'] > max_prob:
                max_prob = node['probability']
                max_leaf_key = key
        
        if max_leaf_key is None:
            return set()
        
        current_Y_hat = max_leaf_key
        Y_prime_hat = set()  # Empty set initially
        
        # Step 2: p_hat_Y ← P(Y_hat|x), p_hat_Y' ← 0
        p_hat_Y = tree[current_Y_hat]['probability']
        #p_hat_Y_prime = 0.0
        
        # Get total number of leaves K
        K = len([key for key, node in tree.items() if key != 'root' and node['is_leaf']])
        
        # Generate uniform random variable u for this iteration
        u = self.rng.uniform(0, 1)
        #if max with u greater than tau_star, we stop immediately
        if u*p_hat_Y > tau_star:
            Y_prime_hat = self._get_leaf_indices_in_subtree(tree, current_Y_hat)
            return Y_prime_hat

        # Step 3: while |Y'| ≠ K do
        
        while len(Y_prime_hat) != K:
            u = self.rng.uniform(0, 1)
            # Get parent of current Y_hat
            parent_key = tree[current_Y_hat]['parent']
            
            if parent_key is None:
                # Already at root, include all leaves
                Y_prime_hat = self._get_leaf_indices_in_subtree(tree, tree['root'])
                break
            
            # Calculate P(pa(Y_hat) \ Y_hat | x) - probability of parent minus current
            parent_prob = tree[parent_key]['probability']
            current_prob = tree[current_Y_hat]['probability']
            prob_diff = parent_prob - current_prob
            
            # Step 4: if p_hat_Y + u * P(pa(Y_hat) \ Y_hat | x) > τ* then
            if p_hat_Y + u * prob_diff > tau_star:
                # Step 5: break
                break
            
            # Step 7: Y' ← Y_hat, Y_hat ← pa(Y_hat)
            current_Y_hat = parent_key
            Y_prime_hat = self._get_leaf_indices_in_subtree(tree, current_Y_hat)
            
            # Step 8: p_hat_Y' ← p_hat_Y, p_hat_Y ← P(Y_hat | x)
            #p_hat_Y_prime = p_hat_Y
            p_hat_Y = tree[current_Y_hat]['probability']
        
        # Step 10: return Y'_hat
        if len(Y_prime_hat) == 0:
            # If we never entered the loop or broke immediately, return the initial max leaf
            Y_prime_hat = self._get_leaf_indices_in_subtree(tree, max_leaf_key)
        
        return Y_prime_hat