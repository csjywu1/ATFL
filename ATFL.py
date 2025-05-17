import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import copy
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
import math
import pickle
import torch.nn.functional as F
import random
from torch.distributions import Categorical

import warnings

# Filter out the specific RNN warning
warnings.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk of memory")

class TileEncoder:
    def __init__(self, graph_path, features_path, tile_size_x=50, tile_size_y=2, dataset_type='xa'):
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.road_to_idx = {}
        self.dataset_type = dataset_type
        
        # Load road network graph and features
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)
        self.features_df = pd.read_csv(features_path)

        # Initialize coordinate bounds
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.update_coordinate_bounds()
        
        # Create edge mapping similar to pattern generator
        self.edge_mapping = self.create_edge_mapping()
        
    def update_coordinate_bounds(self):
        """Update coordinate bounds based on all nodes in the graph"""
        if self.G.nodes():
            self.min_x = min(self.G.nodes[n]['x'] for n in self.G.nodes())
            self.min_y = min(self.G.nodes[n]['y'] for n in self.G.nodes())
            self.max_x = max(self.G.nodes[n]['x'] for n in self.G.nodes())
            self.max_y = max(self.G.nodes[n]['y'] for n in self.G.nodes())
            self.avg_lat = sum(self.G.nodes[n]['y'] for n in self.G.nodes()) / len(self.G.nodes())
            # Conversion factors for geographic distance
            self.lon_scale = 111 * abs(np.cos(np.radians(self.avg_lat)))  # km per degree of longitude
            self.lat_scale = 111  # km per degree of latitude
        
    def create_edge_mapping(self):
        """Create mapping between feature idx and edges with dataset-specific handling"""
        edge_mapping = {}
        edge_props = {}
        
        # Get properties from graph
        for u, v, data in self.G.edges(data=True):
            length = data.get('length', 0)
            highway = data.get('highway', '')
            edge_props[(u, v)] = (length, highway)
            edge_props[(v, u)] = (length, highway)  # Add reverse direction
            
        # Determine ID column based on dataset type and available columns
        if 'idx' in self.features_df.columns:
            id_column = 'idx'
        elif 'road_id' in self.features_df.columns:
            id_column = 'road_id'
        else:
            self.features_df['temp_idx'] = range(len(self.features_df))
            id_column = 'temp_idx'
            
        for _, row in self.features_df.iterrows():
            idx = row[id_column]
            feat_length = row['length']
            feat_highway = row['highway']
            
            best_match = None
            min_length_diff = float('inf')
            
            for (u, v), (length, highway) in edge_props.items():
                length_diff = abs(length - feat_length)
                if self.dataset_type != 'xa':
                    if length_diff < min_length_diff and highway == feat_highway:
                        min_length_diff = length_diff
                        best_match = (u, v)
                else:
                    if length_diff < min_length_diff and highway == feat_highway:
                        min_length_diff = length_diff
                        best_match = (u, v)
                        
            if best_match is not None:
                edge_mapping[idx] = best_match
                
        if not edge_mapping:
            raise ValueError("No edges were matched! Please check the data format and highway types.")
            
        return edge_mapping
        
    def get_tile_id(self, x_coord, y_coord):
        """
        Get tile ID based on coordinates using Definition 3 formula
        
        Args:
            x_coord: x-coordinate (longitude)
            y_coord: y-coordinate (latitude)
            
        Returns:
            tuple: (tile_x, tile_y) indices
        """
        # Map point to its corresponding tile using the formula from Definition 3
        tile_x = int((x_coord - self.min_x) / self.tile_size_x)
        tile_y = int((y_coord - self.min_y) / self.tile_size_y)
        
        return (tile_x, tile_y)
        
    def set_vocabulary(self, road_to_idx):
        """Set the vocabulary mapping from external source"""
        self.road_to_idx = road_to_idx
        
    def build_vocabulary(self, trajectories):
        """Build mapping of road IDs to indices"""
        unique_roads = set()
        for trajectory in trajectories:
            for road_id in trajectory:
                unique_roads.add(road_id)
        self.road_to_idx = {road_id: idx for idx, road_id in enumerate(sorted(unique_roads))}
        return len(self.road_to_idx)
        
    def get_trajectory_bounds(self, trajectory):
        """
        Calculate geographical bounding box for a trajectory
        
        Args:
            trajectory: List of road IDs
            
        Returns:
            tuple: (min_x, min_y, max_x, max_y) of the trajectory
        """
        coords_x = []
        coords_y = []
        
        for road_id in trajectory:
            if road_id in self.edge_mapping:
                u, v = self.edge_mapping[road_id]
                if u in self.G.nodes and v in self.G.nodes:
                    coords_x.extend([self.G.nodes[u]['x'], self.G.nodes[v]['x']])
                    coords_y.extend([self.G.nodes[u]['y'], self.G.nodes[v]['y']])
        
        if not coords_x or not coords_y:
            return self.min_x, self.min_y, self.max_x, self.max_y
            
        return min(coords_x), min(coords_y), max(coords_x), max(coords_y)
        
    def encode_trajectory(self, trajectory):
        """
        Encode trajectory as a tile sequence following Definition 3
        
        Args:
            trajectory: List of road IDs
            
        Returns:
            list: Sequence of tile contents (each tile contains list of road indices)
        """
        # 1. Determine trajectory's geographical bounding box
        traj_min_x, traj_min_y, traj_max_x, traj_max_y = self.get_trajectory_bounds(trajectory)
        
        # Store road IDs for each tile
        tile_contents = {}
        
        # 2. Map each point (road) to its corresponding tile
        for road_id in trajectory:
            if road_id in self.edge_mapping:
                try:
                    u, v = self.edge_mapping[road_id]
                    if u in self.G.nodes and v in self.G.nodes:
                        # Get coordinates for both endpoints
                        u_x, u_y = self.G.nodes[u]['x'], self.G.nodes[u]['y']
                        v_x, v_y = self.G.nodes[v]['x'], self.G.nodes[v]['y']
                        
                        # Get tile IDs for both endpoints
                        u_tile = self.get_tile_id(u_x, u_y)
                        v_tile = self.get_tile_id(v_x, v_y)
                        
                        # Add road to both tiles
                        for tile_id in [u_tile, v_tile]:
                            if tile_id not in tile_contents:
                                tile_contents[tile_id] = []
                            
                            if road_id in self.road_to_idx:
                                mapped_idx = self.road_to_idx[road_id]
                                if mapped_idx not in tile_contents[tile_id]:
                                    tile_contents[tile_id].append(mapped_idx)
                except KeyError:
                    continue  # Skip if node not found
        
        # 3. Create raw tile sequence by ordering tiles
        sorted_tiles = sorted(tile_contents.items(), key=lambda x: (x[0][0], x[0][1]))
        raw_tile_sequence = [tile_id for tile_id, _ in sorted_tiles]
        roads_in_tiles = [roads for _, roads in sorted_tiles]
        
        # 4. Remove consecutive duplicate tiles
        if not raw_tile_sequence:
            return []
            
        unique_tiles = [raw_tile_sequence[0]]
        unique_roads = [roads_in_tiles[0]]
        
        for i in range(1, len(raw_tile_sequence)):
            if raw_tile_sequence[i] != raw_tile_sequence[i-1]:
                unique_tiles.append(raw_tile_sequence[i])
                unique_roads.append(roads_in_tiles[i])
        
        # Return the sequence of road lists for each unique tile
        return unique_roads


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions_x = []
        self.actions_y = []
        self.probs_x = []
        self.probs_y = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = 32  # Added fixed batch size

    def clear_memory(self):
        self.states = []
        self.actions_x = []
        self.actions_y = []
        self.probs_x = []
        self.probs_y = []
        self.vals = []
        self.rewards = []
        self.dones = []

    # def generate_batches(self):
    #     n_states = len(self.states)
    #     batch_start = np.arange(0, n_states, self.batch_size)
    #     indices = np.arange(n_states, dtype=np.int64)
    #     np.random.shuffle(indices)
    #     batches = [indices[i:i+self.batch_size] for i in batch_start]
    #     return batches

    def get_memory_size(self):
        return len(self.states)

class TileSizeRL:
    def __init__(self, hidden_dim=32, gamma=0.99, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=32, n_epochs=10, entropy_coef=0.01, device='cuda'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.device = device
        
        # State dimension is fixed to 3 for our three state components
        self.actor_critic = TileSizeActorCritic(state_dim=3, hidden_dim=hidden_dim).to(device)
        
        self.memory = PPOMemory()
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=0.0003)
    
    def learn(self, client_rewards):
        print("learn")
        
        # Get number of clients from the provided rewards list
        num_clients = len(client_rewards)
        
        # Initialize loss
        total_loss = 0
        
        # Process each client
        for i in range(num_clients):
            # Get reward for this client
            client_reward = client_rewards[i]
            
            # Create a single-element tensor for this client's reward
            client_rewards_tensor = torch.tensor([client_reward], dtype=torch.float32).to(self.device)
            
            # Use the current state for policy update
            current_state = torch.zeros(3, device=self.device)  # Create a dummy state vector
            x_probs, y_probs, values = self.actor_critic(current_state)
            
            # Create advantages based on current reward
            advantages = client_rewards_tensor - values.squeeze()
            
            # Calculate policy loss directly
            policy_loss = -(x_probs.mean() * advantages + y_probs.mean() * advantages).mean()
            value_loss = F.mse_loss(values.view(-1), client_rewards_tensor.view(-1))
            
            # Add to total loss
            total_loss += policy_loss + 0.5 * value_loss
        
        # Average loss across clients
        total_loss /= num_clients
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


    def calculate_state(self, tile_contrast_loss, point_distribution, tile_usage):
        """
        Calculate state vector from the three components
        Args:
            tile_contrast_loss (float): L3 loss value
            point_distribution (torch.Tensor): variance in point density across tiles
            tile_usage (torch.Tensor): variance in tile utilization
        Returns:
            torch.Tensor: State vector [s1, s2, s3]
        """
        state = torch.tensor([
            tile_contrast_loss,
            point_distribution,
            tile_usage
        ], dtype=torch.float32, device=self.device)
        
        # Normalize state components
        state = (state - state.mean()) / (state.std() + 1e-8)
        return state

    def select_action(self, tile_contrast_loss, point_distribution, tile_usage):
        state = self.calculate_state(tile_contrast_loss, point_distribution, tile_usage)
        
        with torch.no_grad():
            x_probs, y_probs, _ = self.actor_critic(state)
        
        x_choices = self.actor_critic.x_choices.to(self.device)
        y_choices = self.actor_critic.y_choices.to(self.device)
        
        x_dist = Categorical(x_probs)
        y_dist = Categorical(y_probs)
        
        x_actions = x_dist.sample()
        y_actions = y_dist.sample()
        
        self.memory.states.append(state.cpu())
        self.memory.actions_x.append(x_actions.cpu())
        self.memory.actions_y.append(y_actions.cpu())
        self.memory.probs_x.append(x_dist.log_prob(x_actions).cpu())
        self.memory.probs_y.append(y_dist.log_prob(y_actions).cpu())
        
        x_sizes = x_choices[x_actions].unsqueeze(0)
        y_sizes = y_choices[y_actions].unsqueeze(0)
        
        return torch.cat([x_sizes, y_sizes], dim=0).t()


class SpatialEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Point-wise feature extraction
        self.point_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Global tile pooling
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Local tile pooling
        self.local_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for concatenation with attention
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final state pooling with output dimension matching embedding_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)  # Output matches embedding_dim
        )
        
        # Self-attention for global context
        self.attention = MultiHeadAttention(hidden_dim, num_heads=4)
        
    def forward(self, tile_embeddings):
        batch_size, n_tiles, n_points, _ = tile_embeddings.size()
        
        # 1. Point-wise feature extraction
        point_features = self.point_mlp(tile_embeddings)
        # Average pooling over points in each tile
        tile_features = point_features.mean(dim=2)  # [batch_size, n_tiles, hidden_dim]
        
        # 2. Global tile pooling
        global_context = tile_features.mean(dim=1)  # [batch_size, hidden_dim]
        global_features = self.global_mlp(global_context)
        
        # 3. Self-attention for global dependencies
        attention_features = self.attention(
            global_features.unsqueeze(1),
            global_features.unsqueeze(1),
            global_features.unsqueeze(1)
        ).squeeze(1)
        
        # 4. Local tile pooling with global context
        # Expand attention features for concatenation
        expanded_attention = attention_features.unsqueeze(1).expand(-1, n_tiles, -1)
        local_input = torch.cat([tile_features, expanded_attention], dim=-1)
        local_features = self.local_mlp(local_input)
        
        # 5. Final state pooling
        final_features = local_features.mean(dim=1)  # [batch_size, hidden_dim]
        output = self.final_mlp(final_features)
        
        return output, attention_features
    
class TwoStreamModel(nn.Module):
    def __init__(self, vocab_size=None, embedding_dim=64, hidden_dim=128, num_heads=4, device='cuda', tile_rl=None):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tile_encoder = None  # Initialize tile_encoder as None
        
        # Initialize model components
        self.road_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.road_encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
        # Replace tile encoder with spatial encoder
        self.spatial_encoder = SpatialEncoder(embedding_dim, hidden_dim)
        
        # Initialize RL component with only relevant parameters
        self.tile_rl = tile_rl
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.decoder_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.fusion_to_embed = nn.Linear(hidden_dim, embedding_dim)
        
        if vocab_size is not None:
            self.initialize_vocab_components(vocab_size)
        else:
            self.embedding = None
            self.output_layer = None
    
    def to(self, device):
        super().to(device)
        self.device = device
        if self.tile_rl is not None:
            self.tile_rl.device = device
            self.tile_rl.actor_critic = self.tile_rl.actor_critic.to(device)
        return self
    

    def initialize_geographic_info(self, G, edge_mapping, min_x, min_y, avg_lat, lon_scale, lat_scale, tile_encoder=None):
        """Initialize geographic information from TileEncoder"""
        self.G = G
        self.edge_mapping = edge_mapping
        self.min_x = min_x
        self.min_y = min_y
        self.avg_lat = avg_lat
        self.lon_scale = lon_scale
        self.lat_scale = lat_scale
        self.tile_encoder = tile_encoder  # Store the tile_encoder instance
    
    def initialize_vocab_components(self, vocab_size):
        """Initialize or update vocabulary-specific components"""
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
            vocab_size, 
            self.embedding_dim, 
            padding_idx=vocab_size-1
        )
        self.road_output_layer = nn.Linear(self.hidden_dim, vocab_size)
        self.output_layer = nn.Linear(self.hidden_dim, vocab_size)
        
        # Move components to the same device as the model if it's on GPU
        if next(self.parameters()).is_cuda:
            device = next(self.parameters()).device
            self.embedding = self.embedding.to(device)
            self.road_output_layer = self.road_output_layer.to(device)
            self.output_layer = self.output_layer.to(device)

    def generate_augmented_sequences(self, roads, tiles, attention_mask):
        """
        Generate augmented sequences for contrastive learning with geographic coherence
        """
        batch_size = roads.size(0)
        device = roads.device
        
        # Get all valid roads (excluding padding tokens)
        batch_roads = roads[roads != self.embedding.padding_idx].unique().tolist()
        
        # Generate enhanced sequences (original + geographically close road)
        enhanced_roads = roads.clone()
        enhanced_tiles = tiles.clone() if tiles is not None else None
        
        # For each sequence, add a geographically close road at the end
        random_roads = []  # Collect random roads for all batches
        for b in range(batch_size):
            # Get the last valid road in sequence
            valid_roads = roads[b][attention_mask[b] == 1]
            if len(valid_roads) > 0:
                last_road = valid_roads[-1]
                nearby_roads = self.get_nearby_roads(roads[b], [last_road])
                # Select random nearby road
                if nearby_roads:
                    random_road = random.choice(nearby_roads)
                    random_roads.append(random_road)
                else:
                    # If no nearby roads, use the last road again
                    random_roads.append(last_road.item() if torch.is_tensor(last_road) else last_road)
        
        # Convert list of random roads to tensor with correct batch dimension
        random_roads_tensor = torch.tensor(random_roads, device=device).unsqueeze(1)  # [batch_size, 1]
        enhanced_roads = torch.cat([enhanced_roads, random_roads_tensor], dim=1)
        
        if enhanced_tiles is not None:
            # Update tiles for enhanced sequence
            enhanced_tiles = torch.cat([enhanced_tiles, enhanced_tiles[:, -1:]], dim=1)

        # Generate abnormal sequences (violate geographic coherence)
        abnormal_roads = roads.clone()
        abnormal_tiles = tiles.clone() if tiles is not None else None
        
        # Replace some roads with geographically distant ones
        for i in range(3):  # Replace 3 positions
            valid_positions = attention_mask.nonzero(as_tuple=True)[1]
            if len(valid_positions) > 0:
                pos_indices = torch.randint(0, len(valid_positions), (batch_size,))
                positions = valid_positions[pos_indices]
                
                for b in range(batch_size):
                    pos = positions[b]
                    current_road = roads[b, pos]
                    # Get geographically distant roads
                    distant_roads = self.get_distant_roads(roads[b], current_road)
                    if distant_roads:
                        abnormal_roads[b, pos] = random.choice(distant_roads)

        # Update tile sequences for abnormal roads
        abnormal_tiles = self.create_tile_sequence(abnormal_roads, 
                                                self.get_tile_sizes(abnormal_roads)) if tiles is not None else None
        
        # Generate enhanced abnormal sequences
        abnormal_enhanced_roads = torch.cat([abnormal_roads, 
                                        abnormal_roads[:, -1:]], dim=1)
        abnormal_enhanced_tiles = torch.cat([abnormal_tiles, 
                                        abnormal_tiles[:, -1:]], dim=1) if abnormal_tiles is not None else None
        
        # Create mask for extended sequences
        extended_mask = torch.cat([attention_mask, 
                                torch.ones(batch_size, 1, device=device)], dim=1)
        
        return (enhanced_roads, enhanced_tiles,
                abnormal_roads, abnormal_tiles,
                abnormal_enhanced_roads, abnormal_enhanced_tiles,
                extended_mask)

    def get_nearby_roads(self, roads, reference_roads, threshold_km=1.0):
        """
        Get roads that are geographically close to the reference roads
        
        Args:
            roads: tensor of road IDs
            reference_roads: list of reference road IDs
            threshold_km: distance threshold in kilometers
        
        Returns:
            List of road IDs that are geographically close
        """
        nearby_roads = []
        
        # Convert threshold to degrees (approximately)
        threshold_deg = threshold_km / (111.0)  # rough conversion
        
        for ref_road in reference_roads:
            if ref_road == self.embedding.padding_idx:
                continue
                
            # Convert tensor to integer for dictionary lookup
            ref_road_idx = ref_road.item() if torch.is_tensor(ref_road) else ref_road
            
            try:
                ref_node = self.edge_mapping[ref_road_idx][0]
                ref_x = self.G.nodes[ref_node]['x']
                ref_y = self.G.nodes[ref_node]['y']
                
                for road in roads:
                    if road == self.embedding.padding_idx or road == ref_road:
                        continue
                    
                    # Convert tensor to integer for dictionary lookup
                    road_idx = road.item() if torch.is_tensor(road) else road
                    
                    try:
                        node = self.edge_mapping[road_idx][0]
                        x = self.G.nodes[node]['x']
                        y = self.G.nodes[node]['y']
                        
                        # Calculate geographic distance
                        dist = np.sqrt((x - ref_x)**2 + (y - ref_y)**2)
                        if dist < threshold_deg:
                            nearby_roads.append(road.item() if torch.is_tensor(road) else road)
                    except KeyError:
                        continue  # Skip if road not in edge_mapping
                        
            except KeyError:
                continue  # Skip if reference road not in edge_mapping
                    
        return list(set(nearby_roads)) if nearby_roads else roads.tolist()

    def get_distant_roads(self, roads, reference_road, min_distance_km=5.0):
        """
        Get roads that are geographically distant from the reference road.
        
        Args:
            roads: tensor of road IDs
            reference_road: reference road ID (tensor or int)
            min_distance_km: minimum distance threshold in kilometers
        
        Returns:
            List of road IDs that are geographically distant.
            If reference road is invalid, returns a filtered list of valid roads.
        """
        min_distance_deg = min_distance_km / 111.0
        valid_roads = []
        
        # Convert reference_road tensor to integer if needed
        ref_road_idx = reference_road.item() if torch.is_tensor(reference_road) else reference_road
        
        # If reference road is padding token, return valid roads from input
        if ref_road_idx == self.embedding.padding_idx:
            return [r.item() if torch.is_tensor(r) else r for r in roads 
                    if r != self.embedding.padding_idx and r != reference_road]
        
        # Try to get reference road location
        try:
            ref_node = self.edge_mapping[ref_road_idx][0]
            ref_x = self.G.nodes[ref_node]['x']
            ref_y = self.G.nodes[ref_node]['y']
            
            # Process each road
            for road in roads:
                # Skip padding tokens and reference road
                if road == self.embedding.padding_idx or road == reference_road:
                    continue
                    
                road_idx = road.item() if torch.is_tensor(road) else road
                
                try:
                    node = self.edge_mapping[road_idx][0]
                    x = self.G.nodes[node]['x']
                    y = self.G.nodes[node]['y']
                    
                    # Calculate geographic distance
                    dist = np.sqrt((x - ref_x)**2 + (y - ref_y)**2)
                    if dist > min_distance_deg:
                        valid_roads.append(road_idx)
                except (KeyError, IndexError):
                    continue
                    
        except (KeyError, IndexError):
            # If reference road is invalid, return all other valid roads
            return [r.item() if torch.is_tensor(r) else r for r in roads 
                    if r != self.embedding.padding_idx and r != reference_road]
        
        # If no distant roads found, return valid roads except reference
        if not valid_roads:
            return [r.item() if torch.is_tensor(r) else r for r in roads 
                    if r != self.embedding.padding_idx and r != reference_road]
        
        return valid_roads

    def compute_contrastive_loss(self, h1, h2, h3, h4, g1, g2, g3, g4, margin=1.0):
        # Calculate similarities using cosine similarity
        sim_h12 = F.cosine_similarity(h1, h2)
        sim_h34 = F.cosine_similarity(h3, h4)
        loss_road = torch.mean(torch.relu(sim_h34 - sim_h12 + margin))
        
        sim_g12 = F.cosine_similarity(g1, g2)
        sim_g34 = F.cosine_similarity(g3, g4)
        loss_tile = torch.mean(torch.relu(sim_g34 - sim_g12 + margin))
        
        return loss_road + loss_tile

    def contrastive_forward(self, roads, tiles, attention_mask):
        """
        Forward pass for contrastive learning
        Returns sequential and spatial embeddings for original and augmented sequences
        """
        # Generate augmented sequences
        (enhanced_roads, enhanced_tiles,
        abnormal_roads, abnormal_tiles,
        abnormal_enhanced_roads, abnormal_enhanced_tiles,
        extended_mask) = self.generate_augmented_sequences(roads, tiles, attention_mask)
        
        # Process original roads
        road_outputs_1, hidden_1 = self.encode_roads(roads, attention_mask)
        h1 = hidden_1.squeeze(0)  # [batch_size, hidden_dim]
        
        # Process enhanced roads
        road_outputs_2, hidden_2 = self.encode_roads(enhanced_roads, extended_mask)
        h2 = hidden_2.squeeze(0)  # [batch_size, hidden_dim]
        
        # Process abnormal roads
        road_outputs_3, hidden_3 = self.encode_roads(abnormal_roads, attention_mask)
        h3 = hidden_3.squeeze(0)  # [batch_size, hidden_dim]
        
        # Process abnormal enhanced roads
        road_outputs_4, hidden_4 = self.encode_roads(abnormal_enhanced_roads, extended_mask)
        h4 = hidden_4.squeeze(0)  # [batch_size, hidden_dim]
        
        # Process original tiles
        tile_features_1 = self.compute_tile_embeddings(tiles)
        g1 = tile_features_1  # [batch_size, embedding_dim]
        
        # Process enhanced tiles
        tile_features_2 = self.compute_tile_embeddings(enhanced_tiles)
        g2 = tile_features_2  # [batch_size, embedding_dim]
        
        # Process abnormal tiles
        tile_features_3 = self.compute_tile_embeddings(abnormal_tiles)
        g3 = tile_features_3  # [batch_size, embedding_dim]
        
        # Process abnormal enhanced tiles
        tile_features_4 = self.compute_tile_embeddings(abnormal_enhanced_tiles)
        g4 = tile_features_4  # [batch_size, embedding_dim]
        
        return h1, h2, h3, h4, g1, g2, g3, g4
        
        # Process each road sequence through road encoder
    def encode_roads(self, roads, mask):
        """
        Encode roads with both sequential embeddings and final state
        Returns:
            seq_embeddings: Sequential embeddings for each timestep
            final_state: Final state embedding from GRU
        """
        # Get initial embeddings
        road_embeddings = self.embedding(roads)  # [batch_size, seq_len, embedding_dim]
        
        # Apply attention
        attended = self.road_attention(
            road_embeddings,
            road_embeddings,
            road_embeddings,
            mask=mask.unsqueeze(1).unsqueeze(2)
        )  # [batch_size, seq_len, embedding_dim]
        
        # Get sequential outputs and final state from GRU
        seq_outputs, final_state = self.road_encoder(attended)
        # seq_outputs: [batch_size, seq_len, hidden_dim]
        # final_state: [1, batch_size, hidden_dim]
        
        return seq_outputs, final_state
        

    
    def create_tile_sequence(self, roads, tile_sizes):
        """
        Create tile sequences using geographic coordinates of road endpoints.
        Each road can be assigned to multiple tiles based on its endpoints.
        Tiles are maintained as sets without ordering.
        """
        batch_size = roads.size(0)
        device = roads.device
        tiles = []
        

        for b in range(batch_size):
            batch_tiles = {}
            # Extract single value if tile_sizes is a tensor
            if torch.is_tensor(tile_sizes):
                tile_size_x = tile_sizes[0].item() if tile_sizes.dim() == 1 else tile_sizes[b, 0].item()
                tile_size_y = tile_sizes[1].item() if tile_sizes.dim() == 1 else tile_sizes[b, 1].item()
            else:
                tile_size_x, tile_size_y = tile_sizes
       
            for road_id in roads[b]:
                # Skip padding tokens
                road_idx = road_id.item() if torch.is_tensor(road_id) else road_id
                if road_idx == self.vocab_size - 1:
                    continue
                    
                try:
                    # Get both endpoints of the road
                    source_node, target_node = self.edge_mapping[road_idx]
                    
                    # Get coordinates for both endpoints
                    source_x = self.G.nodes[source_node]['x']
                    source_y = self.G.nodes[source_node]['y']
                    target_x = self.G.nodes[target_node]['x']
                    target_y = self.G.nodes[target_node]['y']
                    
                    # Convert both endpoints to meters
                    source_x_meters = (source_x - self.min_x) * self.lon_scale * 1000
                    source_y_meters = (source_y - self.min_y) * self.lat_scale * 1000
                    target_x_meters = (target_x - self.min_x) * self.lon_scale * 1000
                    target_y_meters = (target_y - self.min_y) * self.lat_scale * 1000
                    
                    # Get tile coordinates for both endpoints
                    source_tile_x = int(source_x_meters // tile_size_x)
                    source_tile_y = int(source_y_meters // tile_size_y)
                    target_tile_x = int(target_x_meters // tile_size_x)
                    target_tile_y = int(target_y_meters // tile_size_y)
                    
                    # Add road to source tile
                    if (source_tile_x, source_tile_y) not in batch_tiles:
                        batch_tiles[(source_tile_x, source_tile_y)] = set()
                    batch_tiles[(source_tile_x, source_tile_y)].add(road_idx)
                    
                    # Add road to target tile if different from source tile
                    if (target_tile_x, target_tile_y) != (source_tile_x, source_tile_y):
                        if (target_tile_x, target_tile_y) not in batch_tiles:
                            batch_tiles[(target_tile_x, target_tile_y)] = set()
                        batch_tiles[(target_tile_x, target_tile_y)].add(road_idx)
                        
                except (KeyError, IndexError):
                    continue
            
            # Convert sets of roads in each tile to tensors
            tile_tensors = []
            for tile_roads in batch_tiles.values():
                roads_tensor = torch.tensor(list(tile_roads), device=device)
                tile_tensors.append(roads_tensor)
                
            # Handle empty case
            if not tile_tensors:
                tile_tensors.append(torch.tensor([self.vocab_size - 1], device=device))
                
            tiles.append(tile_tensors)
            
        return self.pad_tile_sequences(tiles)


    def pad_tile_sequences(self, tile_sequences):
        """
        Pad tile sequences to same dimensions without enforcing any particular order
        """
        device = tile_sequences[0][0].device
        
        # Find maximum dimensions
        max_tiles = max(len(seq) for seq in tile_sequences)
        max_roads = max(len(roads) for seq in tile_sequences for roads in seq)
        
        # Pad sequences
        padded_sequences = []
        for seq in tile_sequences:
            padded_tiles = []
            
            # Pad roads in each tile
            for roads in seq:
                padding_size = max_roads - len(roads)
                padded_roads = F.pad(roads, (0, padding_size), value=self.vocab_size-1)
                padded_tiles.append(padded_roads)
            
            # Pad number of tiles
            while len(padded_tiles) < max_tiles:
                padding_tile = torch.full((max_roads,), self.vocab_size-1, device=device)
                padded_tiles.append(padding_tile)
            
            padded_sequences.append(torch.stack(padded_tiles))
        
        return torch.stack(padded_sequences)


    def compute_tile_embeddings(self, tiles):
        """
        Compute tile embeddings using point-wise embeddings and spatial encoder
        """
        batch_size, n_tiles, max_roads = tiles.size()
        
        # Get embeddings for all roads in tiles
        flat_roads = tiles.view(-1)
        all_embeddings = self.embedding(flat_roads)
        embeddings = all_embeddings.view(batch_size, n_tiles, max_roads, -1)
        
        # Create mask for padding tokens
        mask = (tiles != self.embedding.padding_idx).unsqueeze(-1).float()
        masked_embeddings = embeddings * mask
        
        # Process through spatial encoder
        tile_features, attention_features = self.spatial_encoder(masked_embeddings)
        
        return tile_features

    def decode(self, hidden, seq_embeddings, target_length):
        """
        Decode sequence using standard GRU
        Args:
            hidden: Initial hidden state [1, batch_size, hidden_dim]
            seq_embeddings: Sequential embeddings [batch_size, seq_len, hidden_dim]
            target_length: Length of sequence to generate
        Returns:
            torch.Tensor: Decoded sequence logits [batch_size, seq_len, vocab_size]
        """
        batch_size = hidden.size(1)
        device = hidden.device
        
        # Initialize decoder input from spatial context
        decoder_input = self.fusion_to_embed(hidden[0])  # [batch_size, embedding_dim]
        decoder_input = decoder_input.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        outputs = []
        current_hidden = hidden
        
        for t in range(target_length):
            # Process through standard GRU
            output, current_hidden = self.decoder_gru(decoder_input, current_hidden)
            
            # Generate next token logits
            output = self.output_layer(output)
            outputs.append(output)
            
            # Prepare next timestep input
            top_roads = output.argmax(dim=-1)
            decoder_input = self.embedding(top_roads)
        
        return torch.cat(outputs, dim=1)  # [batch_size, seq_len, vocab_size]
    
        
    def calculate_state_components(self, roads, tiles, attention_mask, margin=1.0):
        """Calculate state components using contrastive tile loss"""
        # Generate augmented sequences for contrastive loss
        (enhanced_roads, enhanced_tiles,
         abnormal_roads, abnormal_tiles,
         abnormal_enhanced_roads, abnormal_enhanced_tiles,
         extended_mask) = self.generate_augmented_sequences(roads, tiles, attention_mask)
        
        # Get tile embeddings for all sequences
        g1 = self.compute_tile_embeddings(tiles)
        g2 = self.compute_tile_embeddings(enhanced_tiles)
        g3 = self.compute_tile_embeddings(abnormal_tiles)
        g4 = self.compute_tile_embeddings(abnormal_enhanced_tiles)
        
        # Calculate contrastive loss for tiles
        sim_g12 = F.cosine_similarity(g1, g2)
        sim_g34 = F.cosine_similarity(g3, g4)
        loss_tile = torch.mean(torch.relu(sim_g34 - sim_g12 + margin)).item()
        
        # Calculate distribution metrics
        point_distribution = torch.var((tiles != self.embedding.padding_idx).float().sum(dim=2), dim=1).mean()
        tile_usage = torch.var((tiles != self.embedding.padding_idx).float().sum(dim=(1,2)))
        
        return loss_tile, point_distribution, tile_usage


    def get_state_and_tiles(self, roads, attention_mask, tile_sizes=None):
        """Centralized method to compute state components and tiles"""
        if tile_sizes is None:
            tile_sizes = torch.tensor([[500, 500]], device=self.device).expand(roads.size(0), -1)
            
        tiles = self.create_tile_sequence(roads, tile_sizes)
        tile_contrast_loss, point_distribution, tile_usage = self.calculate_state_components(
            roads, tiles, attention_mask
        )
        return {
            'tiles': tiles,
            'tile_contrast_loss': tile_contrast_loss,
            'point_distribution': point_distribution,
            'tile_usage': tile_usage
        }

    def get_tile_sizes(self, batch_size=1):
        """
        Get current tile sizes from the model
        Args:
            batch_size: Number of tile size pairs to return (can be tensor or int)
        Returns:
            torch.Tensor: Tensor of shape [batch_size, 2] containing x,y tile sizes
        """
        if not hasattr(self, 'current_tile_sizes'):
            # Default tile sizes if none set
            self.current_tile_sizes = torch.tensor([[500, 500]], device=self.device)
            
        # Convert batch_size to integer if it's a tensor
        if torch.is_tensor(batch_size):
            batch_size = batch_size.item() if batch_size.dim() == 0 else batch_size.size(0)
            
        return self.current_tile_sizes.expand(int(batch_size), -1)

    def update_tile_sizes(self, new_sizes):
        """
        Update the current tile sizes used by the model
        Args:
            new_sizes: torch.Tensor of shape [batch_size, 2] containing new x,y tile sizes
        """
        if new_sizes.dim() == 1:
            new_sizes = new_sizes.unsqueeze(0)
        self.current_tile_sizes = new_sizes.to(self.device)

    def forward(self, roads, attention_mask, tile_sizes=None):
        """
        Forward pass through the entire model
        Args:
            roads: Input road sequence [batch_size, seq_len]
            attention_mask: Attention mask for padding [batch_size, seq_len]
            tile_sizes: Optional tile sizes for spatial encoding [batch_size, 2]
        Returns:
            dict: Dictionary containing model outputs and states
        """
        # 1. Get road sequence embeddings
        seq_outputs, final_state = self.encode_roads(roads, attention_mask)
        
        # 2. Get state components and tiles
        state_info = self.get_state_and_tiles(roads, attention_mask, tile_sizes)
        tiles = state_info['tiles']
        
        # 3. Process tiles with spatial encoder
        tile_features = self.compute_tile_embeddings(tiles)  # [batch_size, embedding_dim]
        tile_hidden = tile_features.unsqueeze(0)  # [1, batch_size, embedding_dim]
        
        # 4. Fuse spatial and sequential information
        combined = torch.cat([final_state[0], tile_hidden[0]], dim=-1)  # [batch_size, hidden_dim * 2]
        fused = self.fusion_layer(combined).unsqueeze(0)  # [1, batch_size, hidden_dim]
        
        # 5. Decode sequence
        decoded = self.decode(fused, seq_outputs, attention_mask.size(1))
        
        return {
            'decoded': decoded,  # [batch_size, seq_len, vocab_size]
            'road_hidden': final_state,  # [1, batch_size, hidden_dim]
            'tile_hidden': tile_hidden,  # [1, batch_size, embedding_dim]
            'fused_hidden': fused,  # [1, batch_size, hidden_dim]
            'state_info': state_info,  # Dict containing state components
            'seq_outputs': seq_outputs  # [batch_size, seq_len, hidden_dim]
        }

class TileSizeActorCritic(nn.Module):
    def __init__(self, state_dim=64, hidden_dim=32):
        super().__init__()
        # Define fixed tile size choices (in meters)
        self.x_choices = torch.tensor([100, 150, 200, 250, 300, 350, 400, 450, 500,
                                     550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
        self.y_choices = torch.tensor([100, 150, 200, 250, 300, 350, 400, 450, 500,
                                     550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
        
        # Actor network for X dimension
        self.actor_x = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.x_choices))
        )
        
        # Actor network for Y dimension
        self.actor_y = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.y_choices))
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        # Get action probabilities for both dimensions
        x_probs = F.softmax(self.actor_x(state), dim=-1)
        y_probs = F.softmax(self.actor_y(state), dim=-1)
        value = self.critic(state)
        
        return x_probs, y_probs, value
        
    def evaluate_actions(self, state, actions_x, actions_y):
        x_probs = F.softmax(self.actor_x(state), dim=-1)
        y_probs = F.softmax(self.actor_y(state), dim=-1)
        value = self.critic(state)
        
        x_dist = Categorical(x_probs)
        y_dist = Categorical(y_probs)
        
        x_action_log_probs = x_dist.log_prob(actions_x)
        y_action_log_probs = y_dist.log_prob(actions_y)
        
        x_dist_entropy = x_dist.entropy()
        y_dist_entropy = y_dist.entropy()
        
        return x_action_log_probs, y_action_log_probs, value, x_dist_entropy, y_dist_entropy


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None, return_attention=False):
        batch_size = q.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(context)
        
        if return_attention:
            return output, attn
        return output



class TileTrajectoryDataset(Dataset):
    def __init__(self, trajectories, tile_encoder, pad_token=None):
        self.tile_encoder = tile_encoder
        self.pad_token = pad_token
        
        # Filter out roads that aren't in the vocabulary
        filtered_trajectories = []
        for traj in trajectories:
            filtered_traj = [road_id for road_id in traj 
                           if road_id in self.tile_encoder.road_to_idx 
                           and road_id in self.tile_encoder.edge_mapping]
            if filtered_traj:  # Only keep trajectories that have valid roads
                filtered_trajectories.append(filtered_traj)
        
        # Calculate max values from filtered data
        self.max_tiles = 0
        self.max_roads_per_tile = 0
        
        # First pass to calculate maximums
        for traj in filtered_trajectories:
            tiled_traj = self.tile_encoder.encode_trajectory(traj)
            if tiled_traj:  # Check if the encoded trajectory is not empty
                self.max_tiles = max(self.max_tiles, len(tiled_traj))
                for roads in tiled_traj:
                    if roads:
                        self.max_roads_per_tile = max(self.max_roads_per_tile, len(roads))
        
        # Now process trajectories with calculated maximums
        self.tile_trajectories = []
        self.road_trajectories = []
        self.road_lengths = []
        
        for traj in filtered_trajectories:
            tiled_traj = self.tile_encoder.encode_trajectory(traj)
            if not tiled_traj:  # Skip if encoded trajectory is empty
                continue
                
            tile_roads = []
            road_sequence = []
            
            for roads in tiled_traj:
                if roads:
                    padded_roads = roads.copy()
                    while len(padded_roads) < self.max_roads_per_tile:
                        padded_roads.append(pad_token)
                    tile_roads.append(padded_roads)
                    road_sequence.extend(roads)
                else:
                    tile_roads.append([pad_token] * self.max_roads_per_tile)
            
            while len(tile_roads) < self.max_tiles:
                tile_roads.append([pad_token] * self.max_roads_per_tile)
            
            self.tile_trajectories.append(tile_roads)
            self.road_trajectories.append(road_sequence)
            self.road_lengths.append(len(road_sequence))

    def __len__(self):
        return len(self.tile_trajectories)

    def __getitem__(self, idx):
        return {
            'tiles': torch.tensor(self.tile_trajectories[idx]),
            'roads': torch.tensor(self.road_trajectories[idx]),
            'length': self.road_lengths[idx],
            'pad_token': self.pad_token
        }

class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model
        self.device = global_model.device  # Add this line to get the device from the global model
        self.tile_rl = TileSizeRL(hidden_dim=32, device=self.device)
        self.metrics_log = {}  # Store metrics for each client
        self.client_rewards = []  # Initialize client_rewards list for the RL component
        
    def train_federated(self, clients, num_epochs, start_batch=0):
        os.makedirs("tile_size_logs", exist_ok=True)
        
        # Initialize with default tile sizes for first state calculation
        default_tile_sizes = torch.tensor([[500, 500]], device=self.device)
        for client in clients:
            client.model.update_tile_sizes(default_tile_sizes)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Get total number of batches
            num_batches = len(clients[0].batch_files)
            
            # Process batches starting from specified batch
            for batch_idx in range(start_batch, num_batches):
                print(f"\nProcessing batch {batch_idx}")
                
                self.client_rewards = []  # Reset client rewards for this batch
                
                for client in clients:
                    # 1. Calculate initial state using previous tile sizes
                    initial_state = client.get_batch_state(batch_idx)
                    
                    # 2. Select new tile sizes based on current state
                    new_tile_sizes = self.tile_rl.select_action(
                        initial_state['tile_contrast_loss'],
                        initial_state['point_distribution'],
                        initial_state['tile_usage']
                    )
                    
                    # Log selected tile sizes
                    self._log_tile_sizes(client.city_name, epoch, batch_idx, new_tile_sizes)
                    
                    # 3. Update client's tile sizes
                    client.model.update_tile_sizes(new_tile_sizes)
                    
                    # 4. Train for 1 rounds with fixed tile size
                    print(f"\nTraining {client.city_name} for 1 rounds with tile sizes:",
                          f"X: {new_tile_sizes[0].item():.2f}, Y: {new_tile_sizes[1].item():.2f}")
                    final_state = client.train_batch(batch_idx, num_rounds=1)
                    
                    # 5. Calculate reward using final state
                    reward = -final_state['loss']  # Negative loss as reward
                    self.tile_rl.memory.rewards.append(reward)
                    self.client_rewards.append(reward)  # Store for reference in learn method
                    
                    # Log final metrics
                    self._log_metrics(client.city_name, epoch, batch_idx, final_state)
                
                # 6. Update RL policy using collected experiences
                self.tile_rl.learn(self.client_rewards)
                
                # 7. Aggregate spatial encoders across clients
                self.aggregate_models(clients)
                self.calibrate_models(clients)
                
                print(f"\nCompleted batch {batch_idx} processing")
    
    def _log_tile_sizes(self, city_name, epoch, batch_idx, tile_sizes):
        """Log tile sizes to file"""
        log_path = f"tile_size_logs/{city_name}_tile_sizes.txt"
        with open(log_path, 'a') as f:
            f.write(f"\nEpoch {epoch + 1}, Batch {batch_idx}:\n")
            f.write(f"Selected sizes - X: {tile_sizes[0].item():.2f}, Y: {tile_sizes[1].item():.2f}\n")
            f.write("-" * 50 + "\n")
    
    def _log_metrics(self, city_name, epoch, batch_idx, metrics):
        """Store metrics for analysis"""
        if city_name not in self.metrics_log:
            self.metrics_log[city_name] = []
        
        self.metrics_log[city_name].append({
            'epoch': epoch,
            'batch': batch_idx,
            'metrics': metrics
        })

    def compute_wasserstein_distance(self, state1, state2):
        """Compute Wasserstein distance between two states"""
        return torch.abs(state1 - state2)

    def compute_alignment_weights(self, clients, lambda_param=1.0):
        """Compute alignment weights based on Wasserstein distances"""
        num_clients = len(clients)
        num_states = 3  # tile_contrast_loss, point_distribution, tile_usage
        
        # Get states for all clients
        client_states = torch.zeros((num_clients, num_states), device=self.global_model.device)
        for i, client in enumerate(clients):
            state = client.get_current_state()
            client_states[i] = torch.tensor([
                state['tile_contrast_loss'],
                state['point_distribution'],
                state['tile_usage']
            ], device=self.global_model.device)
        
        # Compute pairwise Wasserstein distances for each state
        W = torch.zeros((num_clients, num_clients, num_states), device=self.global_model.device)
        for i in range(num_clients):
            for j in range(num_clients):
                if i != j:
                    W[i,j] = self.compute_wasserstein_distance(client_states[i], client_states[j])
        
        # Compute beta (mean Wasserstein distance) for each client and state
        beta = W.mean(dim=1)  # Average across other clients
        
        # Compute global importance weights (gamma) for each state
        beta_mean = beta.mean(dim=0)  # Average across clients
        gamma = torch.softmax(-beta_mean, dim=0)
        
        # Compute pairwise alignment weights for each state
        alpha = torch.zeros_like(W)
        for k in range(num_states):
            for i in range(num_clients):
                mask = torch.ones(num_clients, device=self.global_model.device)
                mask[i] = 0  # Exclude self
                alpha[i,:,k] = torch.softmax(-lambda_param * W[i,:,k] * mask, dim=0)
        
        # Compute final alignment weights
        final_weights = (alpha * gamma.view(1, 1, -1)).sum(dim=2)
        
        return final_weights
        
    def aggregate_models(self, clients, lambda_param=1.0):
        """Aggregate models using Wasserstein distance-based alignment"""
        # Compute alignment weights
        alignment_weights = self.compute_alignment_weights(clients, lambda_param)
        
        # Initialize shared state dictionary
        shared_state_dict = OrderedDict()
        
        # Define components to exclude/include in aggregation
        exclude_components = ['embedding', 'road_output_layer', 'output_layer', 
                            'road_attention', 'road_encoder', 'fusion_layer', 'decoder']
        include_components = ['spatial_encoder']
        
        # Personalized aggregation for each client
        for i, client in enumerate(clients):
            client_weights = alignment_weights[i]
            client_state = client.model.state_dict()
            
            # Aggregate parameters with computed weights
            for name, param in self.global_model.state_dict().items():
                if any(x in name for x in exclude_components):
                    continue
                    
                if any(x in name for x in include_components):
                    aggregated_param = torch.zeros_like(param)
                    for j, other_client in enumerate(clients):
                        if i != j:  # Skip self
                            other_param = other_client.model.state_dict()[name]
                            aggregated_param += client_weights[j] * other_param
                            
                    # Update client's spatial encoder parameters
                    client_state[name] = aggregated_param
                    
            # Update client model
            client.model.load_state_dict(client_state)
            
        # Update global model (optional, can be average of all clients)
        global_state = self.global_model.state_dict()
        for name, param in global_state.items():
            if any(x in name for x in include_components):
                global_state[name] = torch.mean(torch.stack([
                    client.model.state_dict()[name] for client in clients
                ]), dim=0)
        self.global_model.load_state_dict(global_state)

    def calibrate_models(self, clients):
        """
        Calibrate client models after aggregation
        """
        for client in clients:
            client.model.eval()
            with torch.no_grad():
                # Normalize weights for spatial encoder components
                for name, param in client.model.named_parameters():
                    if 'spatial_encoder' in name:
                        if param.dim() > 1:  # Only normalize matrices, not bias vectors
                            param.data = F.normalize(param.data, dim=-1)

    def evaluate_test_batches(self, clients, start_batch=120, end_batch=167):
        """Run evaluation on test batches (120-167) without RL policy updates"""
        print(f"\nStarting test evaluation on batches {start_batch}-{end_batch}")
        results = {client.city_name: [] for client in clients}
        
        # Log the starting tile sizes (from batch 119) for traceability
        print("\nStarting test evaluation with tile sizes from last training batch:")
        for client in clients:
            current_sizes = client.model.get_tile_sizes(batch_size=1)
            print(f"{client.city_name}: X: {current_sizes[0][0].item():.2f}, Y: {current_sizes[0][1].item():.2f}")
        
        # Process each test batch sequentially
        for batch_num in range(start_batch, end_batch + 1):
            print(f"\nEvaluating test batch {batch_num}")
            
            for client in clients:
                # Code for loading data
                batch_dir = f'{client.batch_path.replace("train_batches", "test_batches")}/batch{batch_num}'
                if not os.path.exists(batch_dir):
                    print(f"Skipping batch {batch_num} for {client.city_name} - directory not found")
                    continue
                    
                # Load test data
                normal_path = f'{batch_dir}/normal.csv'
                switch_path = f'{batch_dir}/switch.csv'
                detour_path = f'{batch_dir}/detour.csv'
                
                if not all(os.path.exists(p) for p in [normal_path, switch_path, detour_path]):
                    print(f"Skipping batch {batch_num} for {client.city_name} - missing data files")
                    continue
                
                # Log the current tile sizes being used for this batch
                current_sizes = client.model.get_tile_sizes(batch_size=1)
                print(f"Using tile sizes for {client.city_name}, batch {batch_num}: X: {current_sizes[0][0].item():.2f}, Y: {current_sizes[0][1].item():.2f}")
                
                # Calculate state using CURRENT tile sizes (inherited from previous batch)
                state = client.get_test_batch_state(normal_path)
                
                # Select new tile sizes using RL agent (without updating)
                with torch.no_grad():
                    new_tile_sizes = self.tile_rl.select_action(
                        state['tile_contrast_loss'],
                        state['point_distribution'],
                        state['tile_usage']
                    )
                
                # Debug the tensor shape
                print(f"new_tile_sizes shape: {new_tile_sizes.shape}, dim: {new_tile_sizes.dim()}")
                
                # Extract values based on tensor shape
                if new_tile_sizes.dim() == 2:  # Expected case [batch_size, 2]
                    next_x = new_tile_sizes[0, 0].item()
                    next_y = new_tile_sizes[0, 1].item()
                elif new_tile_sizes.dim() == 1:  # Case where it's a 1D tensor
                    if new_tile_sizes.size(0) >= 2:
                        next_x = new_tile_sizes[0].item()
                        next_y = new_tile_sizes[1].item()
                    else:
                        next_x = next_y = new_tile_sizes[0].item()
                else:  # Case where it's a scalar
                    next_x = next_y = new_tile_sizes.item()
                
                # Create data loaders and evaluate using CURRENT tile sizes
                normal_loader = self.create_test_loader(normal_path, client)
                switch_loader = self.create_test_loader(switch_path, client)
                detour_loader = self.create_test_loader(detour_path, client)
                
                # Calculate AUC scores
                switch_auc = evaluate_model(client.model, normal_loader, switch_loader)
                detour_auc = evaluate_model(client.model, normal_loader, detour_loader)
                
                # Store and log results
                print(f"{client.city_name} - Batch {batch_num}: Switch AUC: {switch_auc:.4f}, Detour AUC: {detour_auc:.4f}")
                
                results[client.city_name].append({
                    'batch': batch_num,
                    'switch_auc': switch_auc,
                    'detour_auc': detour_auc,
                    'current_tile_size_x': current_sizes[0][0].item(),
                    'current_tile_size_y': current_sizes[0][1].item(),
                    'next_tile_size_x': next_x,
                    'next_tile_size_y': next_y
                })
                
                # Save batch results with both current and next tile sizes
                self.save_batch_result(batch_num, client.city_name, switch_auc, detour_auc, 
                                    current_sizes, torch.tensor([[next_x, next_y]], device=self.device))
                
                # Update client's tile sizes for NEXT batch - use the tensor with correct shape
                client.model.update_tile_sizes(torch.tensor([[next_x, next_y]], device=self.device))
            
        # Calculate and save overall results
        self.save_overall_results(results)
        return results

    def save_batch_result(self, batch_num, city_name, switch_auc, detour_auc, current_sizes, next_sizes):
        """Save individual batch evaluation results"""
        os.makedirs("test_results", exist_ok=True)
        
        # Safely extract values regardless of tensor shape
        if next_sizes.dim() == 2:
            next_x = next_sizes[0, 0].item()
            next_y = next_sizes[0, 1].item()
        elif next_sizes.dim() == 1:
            next_x = next_sizes[0].item()
            next_y = next_sizes[1].item() if next_sizes.size(0) > 1 else next_x
        else:
            next_x = next_y = next_sizes.item()
            
        with open(f"test_results/batch{batch_num}_{city_name}.txt", 'w') as f:
            f.write(f"Test Results for Batch {batch_num} - {city_name}\n")
            f.write(f"Current Tile Size X: {current_sizes[0][0].item():.2f}\n")
            f.write(f"Current Tile Size Y: {current_sizes[0][1].item():.2f}\n")
            f.write(f"Next Tile Size X: {next_x:.2f}\n")
            f.write(f"Next Tile Size Y: {next_y:.2f}\n")
            f.write(f"Switch AUC: {switch_auc:.4f}\n")
            f.write(f"Detour AUC: {detour_auc:.4f}\n")

    def create_test_loader(self, csv_path, client):
        """Create data loader from test CSV file"""
        try:
            df = pd.read_csv(csv_path)
            trajectories = [eval(traj) for traj in df['trajectory']]
            
            dataset = TileTrajectoryDataset(
                trajectories,
                client.model.tile_encoder,
                pad_token=client.model.vocab_size-1
            )
            
            return DataLoader(
                dataset,
                batch_size=32,
                shuffle=False,
                collate_fn=collate_trajectories
            )
        except Exception as e:
            print(f"Error creating data loader from {csv_path}: {str(e)}")
            return None


    def save_overall_results(self, results):
        """Save aggregated evaluation results"""
        os.makedirs("test_results", exist_ok=True)
        
        with open("test_results/overall_results.txt", 'w') as f:
            f.write("Overall Test Evaluation Results\n")
            f.write("============================\n\n")
            
            for city, city_results in results.items():
                switch_aucs = [r['switch_auc'] for r in city_results]
                detour_aucs = [r['detour_auc'] for r in city_results]
                
                f.write(f"Results for {city}:\n")
                f.write(f"Number of test batches: {len(city_results)}\n")
                f.write(f"Switch AUC: Mean={np.mean(switch_aucs):.4f}, Std={np.std(switch_aucs):.4f}\n")
                f.write(f"Detour AUC: Mean={np.mean(detour_aucs):.4f}, Std={np.std(detour_aucs):.4f}\n\n")
            
            # Calculate overall statistics across all cities
            all_switch = [r['switch_auc'] for city_results in results.values() for r in city_results]
            all_detour = [r['detour_auc'] for city_results in results.values() for r in city_results]
            
            f.write("Overall Statistics (All Cities):\n")
            f.write(f"Switch AUC: Mean={np.mean(all_switch):.4f}, Std={np.std(all_switch):.4f}\n")
            f.write(f"Detour AUC: Mean={np.mean(all_detour):.4f}, Std={np.std(all_detour):.4f}\n")

class FederatedClient:
    def __init__(self, city_name, model, batch_path, device):
        self.city_name = city_name
        self.model = copy.deepcopy(model)
        self.batch_path = batch_path
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.current_state = None  # Initialize current state
        
        # Get list of batch files
        self.batch_files = sorted([f for f in os.listdir(batch_path) 
                                 if f.startswith('batch') and f.endswith('.csv')])
    
    def get_current_state(self):
        """
        Return the current state information of the client
        If no current state exists, calculate it from the first batch
        """
        if self.current_state is None:
            # Calculate initial state if none exists
            self.current_state = self.get_batch_state(0)
        return self.current_state
    
    def get_batch_state(self, batch_idx):
        """Calculate state for the current batch"""
        batch_data = self._load_batch_data(batch_idx)
        batch = {k: v.to(self.device) for k, v in batch_data.items()}
        
        with torch.no_grad():
            outputs = self.model(batch['roads'], batch['attention_mask'])
            # Store the current state for later reference
            self.current_state = outputs['state_info']
            return outputs['state_info']
    
    def _load_batch_data(self, batch_idx):
        """Load data for specific batch"""
        batch_file = self.batch_files[batch_idx]
        batch_df = pd.read_csv(os.path.join(self.batch_path, batch_file))
        
        # Convert data to model input format
        trajectories = [eval(traj) for traj in batch_df['trajectory']]
        dataset = TileTrajectoryDataset(trajectories, self.model.tile_encoder, 
                                      pad_token=self.model.vocab_size-1)
        
        # Return first batch from dataloader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, 
                              collate_fn=collate_trajectories)
        return next(iter(dataloader))
    
    def train_batch(self, batch_idx, lambda_contrast=0.5, num_rounds=30):
        """Train on current batch for multiple rounds with fixed tile size"""
        self.model.train()
        batch_data = self._load_batch_data(batch_idx)
        batch = {k: v.to(self.device) for k, v in batch_data.items()}
        
        roads = batch['roads']
        attention_mask = batch['attention_mask']
        
        # Train for specified number of rounds
        final_metrics = None
        for round_idx in range(num_rounds):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(roads, attention_mask)
            
            # Calculate losses
            main_loss = self.criterion(
                outputs['decoded'].contiguous().view(-1, self.model.vocab_size),
                roads.view(-1)
            )
            main_loss = (main_loss.view(roads.size()) * attention_mask).sum() / attention_mask.sum()
            
            # Compute contrastive loss
            h1, h2, h3, h4, g1, g2, g3, g4 = self.model.contrastive_forward(
                roads, batch.get('tiles', None), attention_mask)
            contrastive_loss = self.model.compute_contrastive_loss(h1, h2, h3, h4, g1, g2, g3, g4)
            
            # Combined loss
            total_loss = main_loss + lambda_contrast * contrastive_loss
            total_loss.backward()
            
            self.optimizer.step()
            
            # Store final round metrics
            if round_idx == num_rounds - 1:
                final_metrics = {
                    'loss': main_loss.item(),
                    'contrast_loss': contrastive_loss.item(),
                    'tile_contrast_loss': outputs['state_info']['tile_contrast_loss'],
                    'point_distribution': outputs['state_info']['point_distribution'],
                    'tile_usage': outputs['state_info']['tile_usage']
                }
                # Update current state with final metrics
                self.current_state = outputs['state_info']
            
            if round_idx % 5 == 0:  # Log every 5 rounds
                print(f"\tRound {round_idx + 1}/{num_rounds}, Loss: {main_loss.item():.4f}")
        
        return final_metrics
    
    def get_test_batch_state(self, normal_path):
        """Calculate state for test batch"""
        # Load normal data
        df = pd.read_csv(normal_path)
        trajectories = [eval(traj) for traj in df['trajectory']]
        
        # Create dataset and dataloader
        dataset = TileTrajectoryDataset(
            trajectories,
            self.model.tile_encoder,
            pad_token=self.model.vocab_size-1
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=collate_trajectories
        )
        
        batch = next(iter(dataloader))
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = self.model(batch['roads'], batch['attention_mask'])
            return outputs['state_info']


def evaluate_model(model, normal_loader, anomaly_loader, device='cuda'):
    model.eval()
    normal_scores = []
    anomaly_scores = []
    criterion = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        # Process normal data
        for batch in normal_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            roads = batch['roads']
            attention_mask = batch['attention_mask']
            
            outputs = model(roads, attention_mask)
            
            loss = criterion(
                outputs['decoded'].contiguous().view(-1, model.vocab_size),
                roads.view(-1)
            ).view(roads.size())
            
            masked_loss = loss * attention_mask
            scores = masked_loss.sum(dim=1) / attention_mask.sum(dim=1)
            normal_scores.extend(scores.cpu().numpy())
        
        # Process anomaly data
        for batch in anomaly_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            roads = batch['roads']
            attention_mask = batch['attention_mask']
            
            outputs = model(roads, attention_mask)
            
            loss = criterion(
                outputs['decoded'].contiguous().view(-1, model.vocab_size),
                roads.view(-1)
            ).view(roads.size())
            
            masked_loss = loss * attention_mask
            scores = masked_loss.sum(dim=1) / attention_mask.sum(dim=1)
            anomaly_scores.extend(scores.cpu().numpy())

    return roc_auc_score([0] * len(normal_scores) + [1] * len(anomaly_scores),
                        normal_scores + anomaly_scores)

def collate_trajectories(batch):
    # Get pad_token from the first item in batch
    pad_token = batch[0]['pad_token']
    
    # Find max length in this batch
    max_length = max(item['length'] for item in batch)
    
    # Pad sequences to max length in batch
    roads_batch = []
    for item in batch:
        roads = item['roads'].tolist()
        padded_roads = roads + [pad_token] * (max_length - len(roads))
        roads_batch.append(padded_roads)
    
    return {
        'tiles': torch.stack([item['tiles'] for item in batch]),
        'roads': torch.tensor(roads_batch),
        'lengths': torch.tensor([item['length'] for item in batch]),
        'attention_mask': torch.tensor([[1 if i < item['length'] else 0 
                                       for i in range(max_length)] 
                                      for item in batch])
    }

class DualInputGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # GRU gates for main input
        self.input_r = nn.Linear(input_size, hidden_size)
        self.input_z = nn.Linear(input_size, hidden_size)
        self.input_n = nn.Linear(input_size, hidden_size)
        
        # GRU gates for sequential input
        self.seq_r = nn.Linear(input_size, hidden_size)
        self.seq_z = nn.Linear(input_size, hidden_size)
        self.seq_n = nn.Linear(input_size, hidden_size)
        
        # Hidden state transformations
        self.hidden_r = nn.Linear(hidden_size, hidden_size)
        self.hidden_z = nn.Linear(hidden_size, hidden_size)
        self.hidden_n = nn.Linear(hidden_size, hidden_size)

    def forward(self, input, seq_input, hidden):
        # Calculate reset gates for both inputs
        r_input = torch.sigmoid(self.input_r(input) + self.hidden_r(hidden))
        r_seq = torch.sigmoid(self.seq_r(seq_input) + self.hidden_r(hidden))
        r = (r_input + r_seq) / 2  # Combine reset gates
        
        # Calculate update gates for both inputs
        z_input = torch.sigmoid(self.input_z(input) + self.hidden_z(hidden))
        z_seq = torch.sigmoid(self.seq_z(seq_input) + self.hidden_z(hidden))
        z = (z_input + z_seq) / 2  # Combine update gates
        
        # Calculate new candidate state using both inputs
        n = torch.tanh(self.input_n(input) + self.seq_n(seq_input) + self.hidden_n(r * hidden))
        
        # Update hidden state
        hidden = (1 - z) * n + z * hidden
        
        return hidden

class DualInputGRU(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gru_cell = DualInputGRUCell(input_size, hidden_size)
        
    def forward(self, input, seq_input, hidden=None):
        if self.batch_first:
            batch_size, seq_len, _ = input.size()
        else:
            seq_len, batch_size, _ = input.size()
            input = input.transpose(0, 1)
            seq_input = seq_input.transpose(0, 1)
            
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=input.device)
            
        hidden = hidden.squeeze(0)  # Remove time dimension if present
        output = []
        
        for t in range(seq_len):
            hidden = self.gru_cell(
                input[:, t],
                seq_input[:, t],
                hidden
            )
            output.append(hidden)
            
        output = torch.stack(output, dim=1)  # [batch_size, seq_len, hidden_size]
        hidden = hidden.unsqueeze(0)  # Add time dimension back
        
        if not self.batch_first:
            output = output.transpose(0, 1)
            
        return output, hidden
    
import os
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    cities = ['xa', 'pt', 'cd']
    base_path = '../ATFL/data/data'
    
    # Initialize server with model template
    global_model_template = TwoStreamModel(
        vocab_size=None,  
        embedding_dim=64,
        hidden_dim=64,
        num_heads=4,
        device=device
    ).to(device)
    
    server = FederatedServer(global_model_template)
    clients = []
    
    # Initialize clients
    for city in cities:
        city_path = f'{base_path}/{city}'
        print(f"\nInitializing client for city {city}...")
        
        # Initialize tile encoder
        tile_encoder = TileEncoder(
            graph_path=f'{city_path}/graph.pkl',
            features_path=f'{city_path}/edge_features.csv',
            dataset_type=city
        )
        
        # Get vocabulary size from training data
        train_batch_path = f'{city_path}/{city}/train_batches'
        first_batch_df = pd.read_csv(os.path.join(train_batch_path, 'batch0.csv'))
        train_trajectories = [eval(traj) for traj in first_batch_df['trajectory']]
        vocab_size = tile_encoder.build_vocabulary(train_trajectories)
        
        # Initialize city model
        city_model = copy.deepcopy(global_model_template)
        city_model.to(device).initialize_vocab_components(vocab_size + 1)  # +1 for padding
        city_model.initialize_geographic_info(
            tile_encoder.G, tile_encoder.edge_mapping,
            tile_encoder.min_x, tile_encoder.min_y, 
            tile_encoder.avg_lat, tile_encoder.lon_scale, 
            tile_encoder.lat_scale,
            tile_encoder=tile_encoder  # Pass the tile_encoder instance
        )
        
        # Create client
        client = FederatedClient(city, city_model, train_batch_path, device)
        clients.append(client)
    
    # Train federated model
    # server.train_federated(clients, num_epochs=1)


    #     # Two training scenarios
    # print("\n--- Training Starting from Batch 0 ---")
    # server.train_federated(clients, num_epochs=1, start_batch=0)
    
    print("\n--- Training Starting from Batch 119 ---")
    server.train_federated(clients, num_epochs=1, start_batch=119)
    
    # After training, run test evaluation
    print("\nRunning test evaluation...")
    server.evaluate_test_batches(clients, start_batch=120, end_batch=121)

if __name__ == '__main__':
    main()