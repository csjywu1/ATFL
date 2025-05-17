import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx
import os
from math import ceil
from collections import defaultdict
import random

class PatternGenerator:
    # [Previous PatternGenerator class implementation remains the same]
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
                
            for i in range(len(node_sequence) - 2):
                source = node_sequence[i]
                middle = node_sequence[i + 1]
                dest = node_sequence[i + 2]
                self.path_frequency[(source, middle, dest)] += 1
                
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

def create_edge_mapping(G, features_df, dataset_type='xa'):
    """Create mapping between feature idx and edges with dataset-specific handling"""
    # [Previous implementation remains the same]
    edge_mapping = {}
    edge_props = {}

    print(f"\nDebug: Edge Properties for {dataset_type}")
    print(f"Number of edges in graph: {len(G.edges())}")
    print(f"Number of features: {len(features_df)}")
    print(f"Available columns: {features_df.columns.tolist()}")

    if 'idx' in features_df.columns:
        id_column = 'idx'
    elif 'road_id' in features_df.columns:
        id_column = 'road_id'
    else:
        print("Warning: Neither 'idx' nor 'road_id' found. Creating new index.")
        features_df['temp_idx'] = range(len(features_df))
        id_column = 'temp_idx'

    print(f"Using column '{id_column}' as ID column")

    for u, v, data in G.edges(data=True):
        length = data.get('length', 0)
        highway = data.get('highway', '')
        edge_props[(u, v)] = (length, highway)
        edge_props[(v, u)] = (length, highway)

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
            if dataset_type != 'xa':
                if length_diff < min_length_diff and highway == feat_highway:
                    min_length_diff = length_diff
                    best_match = (u, v)
            else:
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
        raise ValueError(f"No edges were matched for {dataset_type}! Please check the data format and highway types.")

    return edge_mapping

def create_batches(trajectories, num_batches=100):
    """Divide trajectories into specified number of batches"""
    total_trajectories = len(trajectories)
    batch_size = ceil(total_trajectories / num_batches)
    batches = []
    
    for i in range(0, total_trajectories, batch_size):
        batch = trajectories[i:i + batch_size]
        batches.append(batch)
    
    return batches

def process_region_data(region_dir, batch, batch_idx, pattern_generator, is_test=False):
    """Process batch data and generate patterns"""
    # Generate patterns for 10% of the trajectories
    num_patterns = max(1, int(len(batch) * 2))
    
    switch_patterns = []
    detour_patterns = []
    
    # Generate patterns
    for traj in random.sample(batch, min(len(batch), num_patterns)):
        switch = pattern_generator.generate_switch_pattern(traj)
        if switch:
            switch_patterns.append(switch)
        
        detour = pattern_generator.generate_detour_pattern(traj)
        if detour:
            detour_patterns.append(detour)
    
    if is_test:
        return batch, switch_patterns, detour_patterns
    else:
        # Combine all trajectories for non-test batches
        all_trajectories = batch + switch_patterns + detour_patterns
        batch_df = pd.DataFrame({'trajectory': all_trajectories})
        filename = os.path.join(region_dir, f'batch_{batch_idx}.csv')
        batch_df.to_csv(filename, index=False)
        
        print(f"Batch {batch_idx}: Normal={len(batch)}, Switch={len(switch_patterns)}, Detour={len(detour_patterns)}")
        return None

def divide_trajectories_by_region(base_path, dataset_type='cd', output_dir='data1'):
    """Divide trajectories into three regions and create batches with pattern generation"""
    print(f"\nProcessing {dataset_type} dataset...")

    # Load data
    with open(f'{base_path}/{dataset_type}/graph.pkl', 'rb') as f:
        G = pickle.load(f)

    features_df = pd.read_csv(f'{base_path}/{dataset_type}/edge_features.csv')
    trajectories_df = pd.read_csv(f'{base_path}/{dataset_type}/filtered_lines.csv', sep=';')
    trajectories_df['path'] = trajectories_df['path'].apply(eval)

    if dataset_type != 'xa':
        max_length = 200
        trajectories_df['path_length'] = trajectories_df['path'].apply(len)
        print(f"Before length filtering: {len(trajectories_df)} trajectories")
        trajectories_df = trajectories_df[trajectories_df['path_length'] <= max_length]
        print(f"After length filtering: {len(trajectories_df)} trajectories")

    edge_mapping = create_edge_mapping(G, features_df, dataset_type)
    pattern_generator = PatternGenerator(G, edge_mapping)

    # Define region boundaries
    x_coords = [G.nodes[n]['x'] for n in G.nodes()]
    lon_min, lon_max = min(x_coords), max(x_coords)
    lon_range = lon_max - lon_min
    region_width = lon_range / 3

    region_bounds = [
        (lon_min, lon_min + region_width),
        (lon_min + region_width, lon_min + 2 * region_width),
        (lon_min + 2 * region_width, lon_max)
    ]

    def get_trajectory_region(traj):
        traj_lons = []
        for edge_id in traj:
            if edge_id in edge_mapping:
                u, v = edge_mapping[edge_id]
                traj_lons.extend([G.nodes[u]['x'], G.nodes[v]['x']])
        
        if not traj_lons:
            return None

        avg_lon = np.mean(traj_lons)
        for i, (start, end) in enumerate(region_bounds, 1):
            if start <= avg_lon <= end:
                return i
        return None

    # Create output directory
    city_output_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(city_output_dir, exist_ok=True)

    # Group trajectories by region
    region_groups = {1: [], 2: [], 3: []}
    
    print("Grouping trajectories by region...")
    for _, row in tqdm(trajectories_df.iterrows(), total=len(trajectories_df)):
        region = get_trajectory_region(row['path'])
        if region:
            region_groups[region].append(row['path'])

    # Process each region
    print("\nCreating and saving batches with patterns...")
    for region, trajectories in region_groups.items():
        print(f"\nProcessing region {region}")
        region_dir = os.path.join(city_output_dir, f'region{region}')
        os.makedirs(region_dir, exist_ok=True)
        
        # Create test directory for this region
        test_dir = os.path.join(region_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)
        
        # Create batches
        batches = create_batches(trajectories, num_batches=100)
        pattern_generator.build_connections(trajectories)
        
        # Initialize test data containers
        test_normal = []
        test_switch = []
        test_detour = []
        
        # Process batches
        for batch_idx, batch in enumerate(batches, 1):
            if batch_idx > len(batches) - 3:  # Last three batches are for testing
                normal, switch, detour = process_region_data(region_dir, batch, batch_idx, pattern_generator, is_test=True)
                test_normal.extend(normal)
                test_switch.extend(switch)
                test_detour.extend(detour)
            else:
                process_region_data(region_dir, batch, batch_idx, pattern_generator, is_test=False)
        
        # Save test data
        if test_normal:
            print(f"\nSaving test data for region {region}...")
            
            # Save normal test trajectories
            test_normal_df = pd.DataFrame({'trajectory': test_normal})
            test_normal_df.to_csv(os.path.join(test_dir, 'normal.csv'), index=False)
            
            # Save switch test trajectories
            test_switch_df = pd.DataFrame({'trajectory': test_switch})
            test_switch_df.to_csv(os.path.join(test_dir, 'switch.csv'), index=False)
            
            # Save detour test trajectories
            test_detour_df = pd.DataFrame({'trajectory': test_detour})
            test_detour_df.to_csv(os.path.join(test_dir, 'detour.csv'), index=False)
            
            print(f"Test data sizes for region {region}:")
            print(f"  Normal: {len(test_normal)}")
            print(f"  Switch: {len(test_switch)}")
            print(f"  Detour: {len(test_detour)}")

def process_all_cities(base_path, cities=['xa', 'pt', 'cd']):
    """Process all cities and create batched regional data with pattern generation"""
    output_dir = 'data1'
    os.makedirs(output_dir, exist_ok=True)
    
    for city in cities:
        try:
            print(f"\nProcessing city: {city}")
            divide_trajectories_by_region(base_path, dataset_type=city, output_dir=output_dir)
        except Exception as e:
            print(f"Error processing {city}: {str(e)}")
            continue

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Replace with your actual base path
    base_path = '../ATFL/data/data/'
    cities = ['xa', 'pt', 'cd']
    
    # Process all cities
    process_all_cities(base_path, cities)
