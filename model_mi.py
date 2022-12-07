# coding=utf-8

#TODO
# 1. Sampling method for computing graph mutual information
# 2. GNN for computing ELBO, reparameterize
# 3. MinMax training style
# 4. Dynamic neigbhourhood sampling

import torch
import torch_geometric as tg
from torch_geometric.nn import GCNConv,VGAE, GATConv, SAGEConv, GINConv
import torch.nn.functional as F
import scipy


EPS = 1e-15
MAX_LOGSTD = 10


class Stochastic_encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Stochastic_encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)
        self.conv_mu = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.conv_logstd = GCNConv(hidden_channels, hidden_channels, cached=False)
        # self.conv_mu = torch.nn.Linear(hidden_channels, hidden_channels)
        # self.conv_logstd = torch.nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        # return self.conv_mu(x), self.conv_logstd(x)

class Encoder(torch.nn.Module):

    def __init__(self, gcn_layers,in_channels,hidden_channels,out_channels):
        super(Encoder, self).__init__()
        # self.gcn_encoder1 = Stochastic_encoder(in_channels,hidden_channels)
        self.encoders = torch.nn.ModuleList()
        self.encoders.append(Stochastic_encoder(in_channels,hidden_channels))
        for i in range(gcn_layers-2):
            self.encoders.append(Stochastic_encoder(hidden_channels,hidden_channels))
        # self.gcn_encoder3 = Stochastic_encoder(hidden_channels,out_channels)
        self.encoders.append(Stochastic_encoder(hidden_channels,out_channels))

        self.sampled_edge_index = None

    def forward(self,x,edge_index):

        self.mu_list = []
        self.logstd_list = []
        self.z = x
        self.sampled_edge_index = edge_index
        for i, l in enumerate(self.encoders):
            mu,logstd = self.encoders[i](self.z,self.sampled_edge_index)
            logstd= logstd.clamp(max=MAX_LOGSTD)
            self.z = self.reparametrize(mu,logstd)
            self.mu_list.append(mu)
            self.logstd_list.append(logstd)
            self.sampled_edge_index = self.neigbhour_sampling(self.z,edge_index)

        # edge_index should come from neigbhour_sampling
        # if self.sampled_edge_index ==None:
        # self.sampled_edge_index = edge_index
        # self.sampled_edge_index = self.neigbhour_sampling(z,edge_index)
        # self.sampled_edge_index = self.neigbhour_sampling(z,edge_index)

        # mu, logstd = self.gcn_encoder2(z,self.sampled_edge_index)
        # logstd = logstd.clamp(max=MAX_LOGSTD)
        # z = self.reparametrize(mu,logstd)
        # self.mu_list.append(mu)
        # self.logstd_list.append(logstd)

        return self.z, self.mu_list, self.logstd_list


    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
        # return mu + torch.randn_like(logstd) * torch.exp(logstd)

    def neigbhour_sampling(self, z, edge_index,num_hops=2):

        r"""Sample neibhour for each node after GCN layer

        Args:
            z: the latent representation for nodes
            num_hops: (int): The number of hops :math:`k`.
            edge_index: the edge_index for last layer
        Return:
            edge_index: new edge_index based on Bernoulli distribution based sampling
        """
        # node_size = z.shape[0]

        prob = torch.sigmoid(z.matmul(z.t()))
        # prob = prob.clamp(0.0,0.5)

        bern = torch.distributions.Bernoulli(prob)

        sample = bern.sample()

        adj = tg.utils.to_dense_adj(edge_index)
        # print(f'adj1:{adj.sum()}')

        adj = adj.mul(sample)
        # print(f'adj2:{adj.sum()}')
        sampled_edge_index = tg.utils.dense_to_sparse(adj)[0]

        # sparse = scipy.sparse.coo.coo_matrix(sample.cpu())

        # s = tg.utils.k_hop_subgraph([i for i in range(node_size)],num_hops,edge_index)

        # edge_selection=[]
        # for index in range(s[1].shape[1]):
            # node1 = s[1][0][index]
            # node2 = s[1][1][index]
            # edge_selection.append(bool(sample[node1][node2]))


        # edge_selection = torch.tensor(edge_selection)

        # row = s[1][0][edge_selection]
        # col = s[1][1][edge_selection]
        # assert row.shape == col.shape

        # sampled_edge_index = torch.stack([row,col],dim=0)
        assert sampled_edge_index.shape[0]==2

        return sampled_edge_index


class GNN_MI(tg.nn.VGAE):

    def __init__(self, encoder, decoder=None):
        super(GNN_MI, self).__init__(encoder,decoder)


    def encode(self, *args, **kwargs):
        z, self.mu_list, self.logstd_list = self.encoder(*args,**kwargs)
        return z


    def mutual_information(self):

        r"""Exploits the kl_loss to compute the mutual information upper bound for I(Z_{X}^{l}; Z_{X}^{l-1},Z_{A}^{l})
        Args:
            mu_list: the latent space mu from each gcn layer
            logstd_list: the latent space for logstd from each gcn layer
        Return:
            mi_total: total mi for each layer
            mi_list: a list of mi from each layer
        """
        mi_total = 0
        mi_list = []

        for i in range(len(self.mu_list)):
            mi = self.kl_loss(self.mu_list[i],self.logstd_list[i])
            mi_list.append(mi)
            mi_total += mi
        return mi_total, mi_list

class Stochastic_encoder1(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels,out_channels):
		super(Stochastic_encoder1, self).__init__()
		self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)

		self.conv_mu = GCNConv(hidden_channels, out_channels, cached=False)
		self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=False)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		# x = F.dropout(x,0.5)
		return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
		# return self.conv_mu(x), self.conv_logstd(x)


class Encoder1(torch.nn.Module):

	def __init__(self, gcn_layers,in_channels,hidden_channels,out_channels):
		super(Encoder1, self).__init__()
		# self.gcn_encoder1 = Stochastic_encoder(in_channels,hidden_channels)
		self.encoders = torch.nn.ModuleList()
		# self.encoders.append(Stochastic_encoder(in_channels,hidden_channels))
		self.encoders.append(GCNConv(in_channels,hidden_channels,cached = False))
		for i in range(gcn_layers-2):
			# self.encoders.append(Stochastic_encoder(hidden_channels,hidden_channels))
			self.encoders.append(GCNConv(hidden_channels,hidden_channels, cached = False))
		# self.gcn_encoder3 = Stochastic_encoder(hidden_channels,out_channels)
		self.encoders.append(GCNConv(hidden_channels,out_channels,cached = False))

		self.sampled_edge_index = None

	def forward(self,x,edge_index):

		self.mu_list = []
		self.logstd_list = []
		self.z = x
		self.sampled_edge_index = edge_index

		for i in range(len(self.encoders)):
			# print('i:{}'.format(i),self.z.shape)
			self.z = self.encoders[i](self.z, self.sampled_edge_index)
			self.z = F.relu(self.z)
			self.z = torch.nn.Dropout(0.1)(self.z)
				# print("after gcn z shape",self.z.shape)
			# else:
				# mu,logstd = self.encoders[i](self.z,self.sampled_edge_index)
				# logstd= logstd.clamp(max=MAX_LOGSTD)
				# self.mu_list.append(mu)
				# self.logstd_list.append(logstd)
				# self.z = self.reparametrize(mu,logstd)
			self.sampled_edge_index = self.neigbhour_sampling(self.z,edge_index)

		# for i, l in enumerate(self.encoders):
			# mu,logstd = self.encoders[i](self.z,self.sampled_edge_index)
			# logstd= logstd.clamp(max=MAX_LOGSTD)
			# self.z = self.reparametrize(mu,logstd)
			# self.mu_list.append(mu)
			# self.logstd_list.append(logstd)
			# self.sampled_edge_index = self.neigbhour_sampling(self.z,edge_index)

		# edge_index should come from neigbhour_sampling
		# if self.sampled_edge_index ==None:
		# self.sampled_edge_index = edge_index
		# self.sampled_edge_index = self.neigbhour_sampling(z,edge_index)
		# self.sampled_edge_index = self.neigbhour_sampling(z,edge_index)

		# mu, logstd = self.gcn_encoder2(z,self.sampled_edge_index)
		# logstd = logstd.clamp(max=MAX_LOGSTD)
		# z = self.reparametrize(mu,logstd)
		# self.mu_list.append(mu)
		# self.logstd_list.append(logstd)

		return self.z,[],[]

	def reparametrize(self, mu, logstd):
		if self.training:
			return mu + torch.randn_like(logstd) * torch.exp(logstd)
		else:
			return mu
		# return mu + torch.randn_like(logstd) * torch.exp(logstd)

	def neigbhour_sampling(self, z, edge_index,num_hops=2):

		r"""Sample neibhour for each node after GCN layer

		Args:
			z: the latent representation for nodes
			num_hops: (int): The number of hops :math:`k`.
			edge_index: the edge_index for last layer
		Return:
			edge_index: new edge_index based on Bernoulli distribution based sampling
		"""
		# node_size = z.shape[0]

		prob = torch.sigmoid(z.matmul(z.t()))
		# prob = prob.clamp(0.0,0.5)

		bern = torch.distributions.Bernoulli(prob)

		sample = bern.sample()

		adj = tg.utils.to_dense_adj(edge_index)
		# print(f'adj1:{adj.sum()}')

		adj = adj.mul(sample)
		# print(f'adj2:{adj.sum()}')
		sampled_edge_index = tg.utils.dense_to_sparse(adj)[0]

		# sparse = scipy.sparse.coo.coo_matrix(sample.cpu())

		# s = tg.utils.k_hop_subgraph([i for i in range(node_size)],num_hops,edge_index)

		# edge_selection=[]
		# for index in range(s[1].shape[1]):
			# node1 = s[1][0][index]
			# node2 = s[1][1][index]
			# edge_selection.append(bool(sample[node1][node2]))


		# edge_selection = torch.tensor(edge_selection)

		# row = s[1][0][edge_selection]
		# col = s[1][1][edge_selection]
		# assert row.shape == col.shape

		# sampled_edge_index = torch.stack([row,col],dim=0)
		assert sampled_edge_index.shape[0]==2

		return sampled_edge_index

class GCN(torch.nn.Module):
	def __init__(self,in_channels,hidden_channels,out_channels):
		super(GCN,self).__init__()
		self.conv1 = GCNConv(in_channels,hidden_channels,cached=False)
		self.conv2 = GCNConv(hidden_channels,out_channels,cached=False)

		self.linear = torch.nn.Linear(out_channels,5)
		# self.conv3 = GCNConv(hidden_channels,hidden_channels)
		# self.conv4 = GCNConv(hidden_channels,out_channels)
		# self.conv1 = ChebConv(data.num_features, 16, K=2)
		# self.conv2 = ChebConv(16, data.num_features, K=2)

	def forward(self,x,edge_index):
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x,training=self.training)
		x = self.conv2(x, edge_index)
		x = self.linear(x)
		return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
	def __init__(self,in_channels,hidden_channels,out_channels):
		super(GAT,self).__init__()
		self.conv1 = GATConv(in_channels,hidden_channels)
		self.conv2 = GATConv(hidden_channels,out_channels)
		# self.conv1 = ChebConv(data.num_features, 16, K=2)
		# self.conv2 = ChebConv(16, data.num_features, K=2)

	def forward(self,x,edge_index):
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x,0.1)
		x = self.conv2(x, edge_index)
		# x = F.dropout(x,0.1)
		# x = F.sigmoid(x)
		return x

class GraphSAGE(torch.nn.Module):
	def __init__(self,in_channels,hidden_channels,out_channels):
		super(GraphSAGE,self).__init__()
		self.conv1 = SAGEConv(in_channels,hidden_channels)
		self.conv2 = SAGEConv(hidden_channels,out_channels)
		# self.conv1 = ChebConv(data.num_features, 16, K=2)
		# self.conv2 = ChebConv(16, data.num_features, K=2)

	def forward(self,x,edge_index):
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x,0.1)
		x = self.conv2(x, edge_index)
		# x = F.dropout(x,0.1)
		# x = F.sigmoid(x)
		return x

class GIN(torch.nn.Module):
	def __init__(self,in_channels,hidden_channels,out_channels):
		super(GIN,self).__init__()
		self.conv1 = GINConv(in_channels,hidden_channels)
		self.conv2 = GINConv(hidden_channels,out_channels)
		# self.conv1 = ChebConv(data.num_features, 16, K=2)
		# self.conv2 = ChebConv(16, data.num_features, K=2)

	def forward(self,x,edge_index):
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x,0.1)
		x = self.conv2(x, edge_index)
		# x = F.dropout(x,0.1)
		# x = F.sigmoid(x)
		return x
