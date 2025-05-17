import pickle
import random
import numpy as np
from tqdm import tqdm

import networkx as nx
import numpy as np
from collections import defaultdict

import numpy as np
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import pandas as pd



import pickle
import random
import numpy as np
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
import pandas as pd

def create_edge_mapping(G, features_df, dataset_type='xa'):
    """Create mapping between feature idx and edges with dataset-specific handling"""
    edge_mapping = {}
    edge_props = {}
    
    print("\nDebug: Edge Properties")
    print(f"Number of edges in graph: {len(G.edges())}")
    print(f"Number of features: {len(features_df)}")
    print(f"Dataset type: {dataset_type}")
    print(f"Available columns: {features_df.columns.tolist()}")
    
    # Fix issue with dataset_type parameter if not provided correctly
    if dataset_type not in ['xa', 'pt', 'cd']:
        print(f"Warning: Unknown dataset_type {dataset_type}, assuming 'cd'")
        dataset_type = 'cd'
    
    # Determine ID column based on dataset type and available columns
    if 'idx' in features_df.columns:
        id_column = 'idx'
    elif 'road_id' in features_df.columns:
        id_column = 'road_id'
    else:
        # If neither column exists, create a new index
        print("Warning: Neither 'idx' nor 'road_id' found. Creating new index.")
        features_df['temp_idx'] = range(len(features_df))
        id_column = 'temp_idx'
    
    print(f"Using column '{id_column}' as ID column")
    
    # Create edge properties dictionary
    for u, v, data in G.edges(data=True):
        length = data.get('length', 0)
        highway = data.get('highway', '')
        edge_props[(u, v)] = (length, highway)
        edge_props[(v, u)] = (length, highway)  # Add reverse direction
    
    # Create mapping with validation
    matched = 0
    unmatched = []
    
    for _, row in features_df.iterrows():
        idx = row[id_column]
        edge_mapping[idx] = None
        feat_length = row['length']
        feat_highway = row['highway']
        
        best_match = None
        min_length_diff = float('inf')
        
        for (u, v), (length, highway) in edge_props.items():
            length_diff = abs(length - feat_length)
            # More lenient matching for non-xa datasets
            if dataset_type != 'xa':
                if length_diff < min_length_diff and highway == feat_highway:
                    min_length_diff = length_diff
                    best_match = (u, v)
            else:
                # Original strict matching for XA
                if length_diff < min_length_diff and highway == feat_highway:
                    min_length_diff = length_diff
                    best_match = (u, v)
        
        if best_match is not None:
            edge_mapping[idx] = best_match
            matched += 1
        else:
            unmatched.append(idx)
    
    print(f"\nMatched edges: {matched}/{len(features_df)}")
    if unmatched:
        print(f"First 5 unmatched indices: {unmatched[:5]}")
    
    if matched == 0:
        raise ValueError("No edges were matched! Please check the data format and highway types.")
        
    return edge_mapping

def process_trajectories(base_path, dataset_type='cd'):
    """Process trajectories with dataset-specific handling"""
    print(f"Processing trajectories for dataset type: {dataset_type}")
    print("Loading data...")
    with open(f'{base_path}/graph.pkl', 'rb') as f:
        G = pickle.load(f)
    
    features_df = pd.read_csv(f'{base_path}/edge_features.csv')
    trajectories_df = pd.read_csv(f'{base_path}/filtered_lines.csv', sep=';')
    trajectories_df['path'] = trajectories_df['path'].apply(eval)
    
    # Print some debug information
    print("\nFeatures DataFrame info:")
    print(features_df.info())
    print("\nSample of features:")
    print(features_df.head())
    
    # Filter out extremely long trajectories for non-xa datasets
    if dataset_type != 'xa':
        max_length = 200  # Reasonable maximum length threshold
        trajectories_df['path_length'] = trajectories_df['path'].apply(len)
        print(f"\nBefore length filtering: {len(trajectories_df)} trajectories")
        trajectories_df = trajectories_df[trajectories_df['path_length'] <= max_length]
        print(f"After length filtering: {len(trajectories_df)} trajectories")
    
    normal_trajectories = trajectories_df['path'].tolist()
    
    print("\nCreating edge mapping...")
    edge_mapping = create_edge_mapping(G, features_df, dataset_type)
    
    # Rest of the function remains the same...
    # Initialize pattern generator
    generator = PatternGenerator(G, edge_mapping)
    print("\nBuilding path connections...")
    generator.build_connections(normal_trajectories)
    
    # Calculate target sizes (50% of normal trajectories)
    target_size = len(normal_trajectories) // 2
    
    print("\nGenerating patterns...")
    switch_patterns = []
    detour_patterns = []
    
    with tqdm(total=target_size) as pbar:
        for traj in normal_trajectories:
            if len(switch_patterns) >= target_size:
                break
            switch = generator.generate_switch_pattern(traj)
            if switch:
                switch_patterns.append(switch)
                pbar.update(1)
    
    with tqdm(total=target_size) as pbar:
        for traj in normal_trajectories:
            if len(detour_patterns) >= target_size:
                break
            detour = generator.generate_detour_pattern(traj)
            if detour:
                detour_patterns.append(detour)
                pbar.update(1)
    
    # Save results
    print("\nSaving datasets...")
    
    normal_df = pd.DataFrame({'trajectory': normal_trajectories})
    normal_df.to_csv(f'{base_path}/normal_trajectories.csv', index=False)
    
    switch_df = pd.DataFrame({'trajectory': switch_patterns})
    switch_df.to_csv(f'{base_path}/switch_patterns.csv', index=False)
    
    detour_df = pd.DataFrame({'trajectory': detour_patterns})
    detour_df.to_csv(f'{base_path}/detour_patterns.csv', index=False)
    
    print(f"\nFinal dataset sizes:")
    print(f"  Normal trajectories: {len(normal_trajectories)}")
    print(f"  Switch patterns: {len(switch_patterns)}")
    print(f"  Detour patterns: {len(detour_patterns)}")

class PatternGenerator:
    def __init__(self, G: nx.Graph, edge_mapping: dict):
        self.G = G
        self.edge_mapping = edge_mapping
        self.path_frequency = defaultdict(int)
        self.next_edges = defaultdict(set)
        self.prev_edges = defaultdict(set)
    
    def build_connections(self, trajectories):
        """Build edge connections and calculate path frequencies"""
        print("\nBuilding connections...")
        
        for trajectory in tqdm(trajectories):
            if len(trajectory) < 3:
                continue
            
            # Convert edge sequence to node sequence
            node_sequence = []
            edge_conversion_failed = False
            
            for edge_idx in trajectory:
                if edge_idx not in self.edge_mapping:
                    edge_conversion_failed = True
                    break
                    
                u, v = self.edge_mapping[edge_idx]
                if not node_sequence or node_sequence[-1] != u:
                    node_sequence.append(u)
                node_sequence.append(v)
            
            if edge_conversion_failed:
                continue
                
            # Build connections and frequencies
            for i in range(len(node_sequence) - 2):
                source = node_sequence[i]
                middle = node_sequence[i + 1]
                dest = node_sequence[i + 2]
                self.path_frequency[(source, middle, dest)] += 1
                
                # Find edge indices
                edge1 = None
                edge2 = None
                
                for idx, (u, v) in self.edge_mapping.items():
                    if (u == source and v == middle) or (u == middle and v == source):
                        edge1 = idx
                        break
                
                for idx, (u, v) in self.edge_mapping.items():
                    if (u == middle and v == dest) or (u == dest and v == middle):
                        edge2 = idx
                        break
                
                if edge1 is not None and edge2 is not None:
                    self.next_edges[edge1].add(edge2)
                    self.prev_edges[edge2].add(edge1)

    def generate_switch_pattern(self, trajectory: list, max_attempts: int = 3) -> list:
        """Generate route switching pattern"""
        if len(trajectory) < 5:
            return None
        
        for _ in range(max_attempts):
            switch_start = random.randint(1, len(trajectory) - 4)
            
            # Get node sequence
            node_sequence = []
            for edge_idx in trajectory[switch_start:switch_start+4]:
                if edge_idx not in self.edge_mapping:
                    continue
                u, v = self.edge_mapping[edge_idx]
                if not node_sequence or node_sequence[-1] != u:
                    node_sequence.append(u)
                node_sequence.append(v)
            
            if len(node_sequence) < 4:
                continue
            
            start_node = node_sequence[0]
            end_node = node_sequence[-1]
            current_nodes = node_sequence[1:-1]
            
            current_freq = sum(self.path_frequency.get((current_nodes[i], current_nodes[i+1], 
                             current_nodes[i+2]), 0) for i in range(len(current_nodes)-2))
            
            alternative_paths = []
            for neighbor1 in self.G.neighbors(start_node):
                if neighbor1 in current_nodes:
                    continue
                for neighbor2 in self.G.neighbors(neighbor1):
                    if neighbor2 in current_nodes or neighbor2 == start_node:
                        continue
                    if end_node in self.G.neighbors(neighbor2):
                        alt_freq = (self.path_frequency.get((start_node, neighbor1, neighbor2), 0) +
                                  self.path_frequency.get((neighbor1, neighbor2, end_node), 0))
                        
                        if alt_freq < current_freq:
                            path = [neighbor1, neighbor2]
                            alternative_paths.append((alt_freq, path))
            
            if alternative_paths:
                chosen_path = min(alternative_paths, key=lambda x: x[0])[1]
                new_edges = []
                node_pairs = list(zip([start_node] + chosen_path, chosen_path + [end_node]))
                
                for u, v in node_pairs:
                    for idx, (edge_u, edge_v) in self.edge_mapping.items():
                        if (u == edge_u and v == edge_v) or (u == edge_v and v == edge_u):
                            new_edges.append(idx)
                            break
                
                if len(new_edges) == len(node_pairs):
                    new_trajectory = trajectory.copy()
                    new_trajectory[switch_start+1:switch_start+3] = new_edges
                    return new_trajectory
        
        return None

    def generate_detour_pattern(self, trajectory: list) -> list:
        """Generate detour pattern"""
        if len(trajectory) < 3:
            return None
            
        max_attempts = 3
        for _ in range(max_attempts):
            start_idx = random.randint(0, len(trajectory) - 3)
            
            node_sequence = []
            for edge_idx in trajectory[start_idx:start_idx+3]:
                if edge_idx not in self.edge_mapping:
                    continue
                u, v = self.edge_mapping[edge_idx]
                if not node_sequence or node_sequence[-1] != u:
                    node_sequence.append(u)
                node_sequence.append(v)
            
            if len(node_sequence) < 3:
                continue
            
            source = node_sequence[0]
            middle = node_sequence[1]
            dest = node_sequence[-1]
            
            current_freq = self.path_frequency[(source, middle, dest)]
            potential_middles = set()
            
            for neighbor in self.G.neighbors(source):
                if neighbor != middle and dest in self.G.neighbors(neighbor):
                    potential_middles.add(neighbor)
            
            if not potential_middles:
                continue
            
            alt_paths = []
            for alt_middle in potential_middles:
                freq = self.path_frequency.get((source, alt_middle, dest), 0)
                if freq < current_freq:
                    alt_paths.append((freq, alt_middle))
            
            if alt_paths:
                chosen_middle = min(alt_paths, key=lambda x: x[0])[1]
                new_edges = []
                node_pairs = [(source, chosen_middle), (chosen_middle, dest)]
                
                for u, v in node_pairs:
                    for idx, (edge_u, edge_v) in self.edge_mapping.items():
                        if (u == edge_u and v == edge_v) or (u == edge_v and v == edge_u):
                            new_edges.append(idx)
                            break
                
                if len(new_edges) == 2:
                    new_trajectory = trajectory.copy()
                    new_trajectory[start_idx+1:start_idx+2] = new_edges
                    return new_trajectory
        
        return None



if __name__ == "__main__":
    base_path = '../ATFL/data/data/pt'
    process_trajectories(base_path)




#Final dataset sizes:
#   Normal trajectories: 199999
#   Switch patterns: 4214
#   Detour patterns: 755
