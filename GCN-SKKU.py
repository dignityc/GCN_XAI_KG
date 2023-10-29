import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

import wandb, os

###### PARSER custom args:

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--sampling_size', type=int, default=260000, help='the training sample size')
parser.add_argument('--batch_size', type=int, default=65536, help='the batch size')
parser.add_argument('--alpha', type=float, default=0.0, help='the alpha')
parser.add_argument('--lr', type=float, default=0.001, help='the learning rate')
args = parser.parse_args()

sampling_size = args.sampling_size
batch_size = args.batch_size

os.environ['WANDB_WATCH'] = 'all'
wandb.login(key = "d737480c1d812ca4c1c791b14b0888051f72b45a")
wandb.init(name=f"{str(args.alpha)}-alpha,lr:{args.lr}",
           project="GCN-recommendaiton",
           tags=['for_testing_in_A100'],
           group=f"FinalModel:{sampling_size}"
           )


class InteractionDataset(Dataset):
    def __init__(self, positive_users, positive_items, negative_items):
        assert len(positive_users) == len(positive_items) == len(negative_items)
        self.positive_users = positive_users
        self.positive_items = positive_items
        self.negative_items = negative_items

    def __len__(self):
        return len(self.positive_users)

    def __getitem__(self, idx):
        return self.positive_users[idx], self.positive_items[idx], self.negative_items[idx]

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, adj_matrix, emb_size=64, n_layers=3):
        super(LightGCN, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.adj_matrix = adj_matrix
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.emb_size)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.emb_size)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        
    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.adj_matrix, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def forward(self, user_indices, item_indices):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        all_emb = torch.cat([all_users, all_items])
        return all_emb

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        return output
        
class Combined(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_users, num_items, adj_matrix):
        super(Combined, self).__init__()
        self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
        self.gc2 = GraphConvolutionLayer(hidden_dim, hidden_dim)
        self.gc3 = GraphConvolutionLayer(hidden_dim, embedding_dim)
        self.lg = LightGCN(num_users, num_items, adj_matrix[:(num_users + num_items), :(num_users + num_items)], emb_size=64, n_layers=3)

    def forward(self, x, adj, user_indices, item_indices):
        adj = adj[num_users:, num_users:]
        NGCF = F.relu(self.gc1(x, adj))
        NGCF = F.relu(self.gc2(x, adj))
        NGCF = self.gc3(NGCF, adj)
        
        Light = self.lg(user_indices, item_indices)
        
        return NGCF, Light
        
class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, positive_scores, negative_scores):
        loss = torch.mean(torch.nn.functional.softplus(negative_scores - positive_scores))
        return loss
    
#Data prep.
ui = pd.read_csv('final/final_ui_interaction.csv')
ie = pd.read_csv('final/final_ie_interaction.csv')

#data ordering and slicing for getting dense set.
item_counts = ui.groupby('userID').size().reset_index(name='item_count')
sorted_users = item_counts.sort_values('item_count', ascending=True)
ui = ui.merge(sorted_users[['userID']], on='userID')
ui = ui.iloc[:sampling_size]
merged_table = pd.merge(ie, ui, on='itemID')
ie = merged_table.drop(columns=['userID'])
ui = ui[['userID', 'itemID']]
ie = ie[['itemID', 'entityID']]
#Data sampling.
#ui = ui[['userID', 'itemID']].sample(sampling_size, random_state=42)
#ie = ie[['itemID', 'entityID']].sample(sampling_size, random_state=42)
# 1. 유니크한 사용자, 아이템, 엔터티 리스트 생성
unique_users = ui['userID'].unique()
unique_items = pd.concat([ui['itemID'], ie['itemID']]).unique()  # ui와 ie에서 아이템 가져오기
unique_entities = ie['entityID'].unique()

####임베딩 추가####
user_embed = pd.read_csv('embeddings/user_embedding.csv')
item_embed = pd.read_csv('embeddings/item_embedding.csv')
entity_embed = pd.read_csv('embeddings/entity_embedding.csv')
def str_to_array(data_str):
    # 쉼표가 있으면 쉼표를 구분자로 사용, 없으면 공백을 구분자로 사용
    if ',' in data_str:
        data_list = [float(x.strip()) for x in data_str[1:-1].split(',')]
    else:
        data_list = [float(x) for x in data_str[1:-1].split()]
    return np.array(data_list)

user_embed['embedding'] = user_embed['embedding'].apply(str_to_array)
item_embed['embedding'] = item_embed['embedding'].apply(str_to_array)
entity_embed['embedding'] = entity_embed['embedding'].apply(str_to_array)

user_embed = user_embed[user_embed['remap_id'].isin(ui['userID'])]
item_embed = item_embed[item_embed['remap_id'].isin(ui['itemID']) | item_embed['remap_id'].isin(ie['itemID'])]
entity_embed = entity_embed[entity_embed['entity'].isin(ie['entityID'])]
##################

# 2. 새로운 ID 부여
user_to_id = {user: idx for idx, user in enumerate(unique_users)}
item_to_id = {item: idx + len(unique_users) for idx, item in enumerate(unique_items)}  # 사용자 수만큼 오프셋
entity_to_id = {entity: idx + len(unique_users) + len(unique_items) for idx, entity in enumerate(unique_entities)}

# 3. 새로운 ID로 데이터 업데이트
ui['userID'] = ui['userID'].map(user_to_id)
ui['itemID'] = ui['itemID'].map(item_to_id)
ie['itemID'] = ie['itemID'].map(item_to_id)
ie['entityID'] = ie['entityID'].map(entity_to_id)

###############
user_embed['remap_id'] = user_embed['remap_id'].map(user_to_id)
item_embed['remap_id'] = item_embed['remap_id'].map(item_to_id)
entity_embed['entity'] = entity_embed['entity'].map(entity_to_id)
entity_embed = entity_embed.rename(columns={'entity':'remap_id'})

merged_df = pd.concat([user_embed, item_embed, entity_embed])

sorted_df = merged_df.sort_values(by='remap_id').reset_index(drop=True)

user_features = sorted_df.iloc[:len(unique_users)]['embedding']
node_features = sorted_df.iloc[len(unique_users):]['embedding']

################

# 4. 인접 행렬 생성
num_total_nodes = len(unique_users) + len(unique_items) + len(unique_entities)
adj_matrix = np.zeros((num_total_nodes, num_total_nodes))

# ui 상호작용 추가
for _, row in ui.iterrows():
    adj_matrix[row['userID'], row['itemID']] = 1
    adj_matrix[row['itemID'], row['userID']] = 1

# ie 상호작용 추가
for _, row in ie.iterrows():
    adj_matrix[row['itemID'], row['entityID']] = 1
    adj_matrix[row['entityID'], row['itemID']] = 1

# 결과 리턴: 새로운 ID 부여된 데이터와 인접행렬
new_ui = ui
new_ie = ie
total_nodes_length = len(unique_users) + len(unique_items) + len(unique_entities)
num_users = len(unique_users)
num_items = len(unique_items)

import random

# 긍정적 유저-아이템 데이터 추출
positive_samples = new_ui.reset_index(drop=True)
positive_users = positive_samples['userID'].tolist()
positive_items = positive_samples['itemID'].tolist()

# 부정적 유저-아이템 데이터 생성
all_items = set(new_ui['itemID'].unique())
negative_users = []
negative_items = []

while len(negative_users) < sampling_size:
    i = 0
    random_user = positive_users[i]
    random_item = random.choice(list(all_items))

    # 해당 유저가 해당 아이템과 상호작용하지 않았을 경우만 부정적 데이터로 추가
    if not new_ui[(new_ui['userID'] == random_user) & (new_ui['itemID'] == random_item)].empty:
        continue

    negative_users.append(random_user)
    negative_items.append(random_item)
    i += 1

print(len(positive_users), len(positive_items), len(negative_items))

num_nodes = total_nodes_length
epochs = 10000
dataset = InteractionDataset(positive_users, positive_items, negative_items)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
user_features = torch.FloatTensor(np.array(user_features.tolist()))
user_features = user_features.to(device)
node_features = torch.FloatTensor(node_features.tolist())
node_features = node_features.to(device)
adj = torch.FloatTensor(adj_matrix)
adj = adj.to(device)
model = Combined(input_dim=node_features.size(1), hidden_dim=384, embedding_dim=384, num_users = num_users, num_items = num_items, adj_matrix = adj)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

bpr_loss = BPRLoss()


# Group by userID and aggregate itemIDs into sets
grouped = ui.groupby('userID')['itemID'].apply(set)

# Convert the grouped Series into a dictionary
ground_truth = grouped.to_dict()

def generate_recommendations(model, node_features, adj, user_ids, num_recommendations):
    model.eval()
    with torch.no_grad():
        NGCF, Light = model(node_features, adj, batch_users, batch_pos_items)
    recommendations = {}
    # Correcting the range of item_ids
    item_ids = torch.arange(len(unique_users), len(unique_users) + len(unique_items)).to(device)
    for user_id in user_ids:
        # Ensure user_id is a valid user ID
        if user_id not in unique_users:
            print(f'Skipping user_id {user_id} as it is out of bounds')
            continue
        user_index = np.where(unique_users == user_id)[0][0]  # Find the index of user_id in unique_users
        scores = alpha * torch.sum(user_features[user_index] * NGCF[item_ids - num_users], dim=1) +  (1 - alpha) * torch.sum(Light[user_index] * Light[item_ids], dim=1)
        _, top_indices = torch.topk(scores, num_recommendations)
        recommended_items = [item_ids[i].item() for i in top_indices]  # Converting tensor indices to Python integers
        recommendations[user_id] = set(recommended_items)
    return recommendations



def calculate_recall(recommendations, ground_truth):
    total_relevant_items = 0
    total_recommended_relevant_items = 0
    for user_id, recommended_items in recommendations.items():
        relevant_items = ground_truth.get(user_id, set())
        total_relevant_items += len(relevant_items)
        total_recommended_relevant_items += len(relevant_items.intersection(recommended_items))
    recall = total_recommended_relevant_items / total_relevant_items if total_relevant_items > 0 else 0
    return recall

def calculate_ndcg(recommendations, ground_truth, p):
    ndcg_scores = []
    for user_id, recommended_items in recommendations.items():
        relevant_items = ground_truth.get(user_id, set())
        dcg = 0
        idcg = 0
        for i, item in enumerate(recommended_items):
            if i >= p:
                break
            rel = 1 if item in relevant_items else 0
            dcg += (2**rel - 1) / np.log2(i + 2)  # log2(i+2) because i is 0-indexed but ranks start from 1.
        for i in range(min(p, len(relevant_items))):
            idcg += 1 / np.log2(i + 2)  # Ideal DCG is when all relevant items are ranked first.
        ndcg = dcg / (idcg + 1e-10)  # Adding a small value to avoid division by zero.
        ndcg_scores.append(ndcg)
    return np.mean(ndcg_scores)  # Return the mean NDCG score across all users.

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    alpha = args.alpha
    for batch_users, batch_pos_items, batch_neg_items in dataloader:
        batch_users = batch_users.to(device)
        batch_pos_items = batch_pos_items.to(device)
        batch_neg_items = batch_neg_items.to(device)        
        
        optimizer.zero_grad()

        NGCF, Light = model(node_features, adj, batch_users, batch_pos_items)
        positive_scores = alpha * torch.sum(user_features[batch_users] * NGCF[batch_pos_items - num_users], dim=1) + (1 - alpha) * torch.sum(Light[batch_users] * Light[batch_pos_items], dim=1)
        negative_scores = alpha * torch.sum(user_features[batch_users] * NGCF[batch_neg_items - num_users], dim=1) + (1 - alpha) * torch.sum(Light[batch_users] * Light[batch_neg_items], dim=1)

        loss = bpr_loss(positive_scores, negative_scores)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if (epoch + 1) % 200 == 0:
        user_ids = list(unique_users)
        # @10
        num_recommendations = 10
        recommendations = generate_recommendations(model, node_features, adj, user_ids, num_recommendations)
        Recall10 = calculate_recall(recommendations, ground_truth)
        Ndcg10 = calculate_ndcg(recommendations, ground_truth, num_recommendations)

        # @50
        num_recommendations = 50
        recommendations = generate_recommendations(model, node_features, adj, user_ids, num_recommendations)
        Recall50 = calculate_recall(recommendations, ground_truth)
        Ndcg50 = calculate_ndcg(recommendations, ground_truth, num_recommendations)

        # @100
        num_recommendations = 100
        recommendations = generate_recommendations(model, node_features, adj, user_ids, num_recommendations)
        Recall100 = calculate_recall(recommendations, ground_truth)
        Ndcg100 = calculate_ndcg(recommendations, ground_truth, num_recommendations)

        Loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {Loss}, Recall10: {Recall10}, NDCG10: {Ndcg10}, Recall50: {Recall50}, NDCG50: {Ndcg50}, Recall100: {Recall100}, NDCG100: {Ndcg100}")
        wandb.log({"Loss": Loss, "Recall10": Recall10, "NDCG10": Ndcg10, "Recall50": Recall50, "NDCG50": Ndcg50, "Recall100": Recall100, "NDCG100": Ndcg100})
    else: 
        Loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {Loss}")
        wandb.log({"Loss": Loss})