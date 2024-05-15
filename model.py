import torch
import torch.nn as nn
from egnn_pytorch import EGNN


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)  # 每个词和位置先随机初始化
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SequenceConv(nn.Module):
    def __init__(self, MAX_LENGTH, node_dim=26):
        super().__init__()
        self.dim = 128
        self.conv = 16
        self.protein_kernel = [4, 8, 12]
        self.MAX_LENGH = MAX_LENGTH

        self.protein_embed = Embeddings(node_dim, self.dim, self.MAX_LENGH, 0.1)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 4, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 4, out_channels=self.conv * 8, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        self.Protein_max_pool = nn.MaxPool1d(
            self.MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)

    def forward(self, protein):
        protein_embed = self.protein_embed(protein)
        protein_embed = protein_embed.permute(0, 2, 1)
        proteinConv = self.Protein_CNNs(protein_embed).permute(0, 2, 1)

        return proteinConv


class EquivariantGraph(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim=128)
        self.egnn1 = EGNN(dim=128, m_dim=256, edge_dim=1)
        self.elu = nn.ELU(inplace=True)
        self.egnn2 = EGNN(dim=128, m_dim=256, edge_dim=1)

    def forward(self, data):
        x, pos, edge, batch = data.x, data.pos, data.edge, data.batch
        node_num = len(x)
        x = self.embed(x)
        x = torch.unsqueeze(x, 0)
        pos = torch.unsqueeze(pos, 0)
        edge = edge.reshape((node_num, node_num, -1))
        edge = torch.unsqueeze(edge, 0)

        x, pos = self.egnn1(x, pos, edge)
        x = self.elu(x)
        pos = self.elu(pos)
        x, pos = self.egnn2(x, pos, edge)
        return x


class MyModel(nn.Module):
    def __init__(self,
                 ligase_ligand_model,
                 ligase_model,
                 target_ligand_model,
                 target_model,
                 linker_model,
                 dim=128,
                 drop_out=0.2
                 ):
        super().__init__()
        self.ligase_ligand_model = ligase_ligand_model
        self.ligase_model = ligase_model
        self.target_ligand_model = target_ligand_model
        self.target_model = target_model
        self.linker_model = linker_model

        self.attention_layer = nn.Linear(dim, dim)
        self.target_attention_layer = nn.Linear(dim, dim)
        self.ligase_attention_layer = nn.Linear(dim, dim)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        self.dropout3 = nn.Dropout(drop_out)
        self.fc1 = nn.Linear(dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 2)

    def forward(self,
                ligase_ligand,
                ligase,
                target_ligand,
                target,
                linker, ):
        v_0 = self.ligase_ligand_model(ligase_ligand)
        v_1 = self.target_ligand_model(target_ligand)
        v_2 = self.linker_model(linker)
        v_t = self.target_model(target)
        v_l = self.ligase_model(ligase)
        v_d = torch.cat([v_1, v_2, v_0], dim=1)

        v_t_d = torch.cat([v_t, v_d], dim=1)
        v_l_d = torch.cat([v_l, v_d], dim=1)

        target_att = self.target_attention_layer(v_t_d)
        ligase_att = self.ligase_attention_layer(v_l_d)

        t_att_layers = target_att.unsqueeze(2).repeat(1, 1, v_l_d.shape[-2], 1)
        l_att_layers = ligase_att.unsqueeze(1).repeat(1, v_t_d.shape[-2], 1, 1)

        Atten_matrix = self.attention_layer(self.relu(t_att_layers + l_att_layers))
        target_atte = self.sigmoid(Atten_matrix.mean(2))
        ligase_atte = self.sigmoid(Atten_matrix.mean(1))

        v_t_d = v_t_d * 0.5 + v_t_d * target_atte
        v_l_d = v_l_d * 0.5 + v_l_d * ligase_atte

        fully1 = self.relu(self.fc1(torch.cat([torch.sum(v_t_d, 1), torch.sum(v_l_d, 1)], dim=1)))
        fully1 = self.dropout2(fully1)
        fully2 = self.relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict, target_atte, ligase_atte
