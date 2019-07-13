# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel


class BertKnrm(BertPreTrainedModel):
    """Implementation of K-NRM on top of BERT for Ad-hoc
    ranking.
    
    See [1] that creates such a model and [2] and [3]
    for K-NRM and Convolutional K-NRM.
    
    References
    ----------
    [1] - MacAvaney, S., Yates, A., Cohan, A., & Goharian, N. 
          (2019). CEDR: Contextualized Embeddings for Document 
          Ranking. CoRR.
          (https://arxiv.org/pdf/1904.07094.pdf)
    
    [2] - Xiong, C., Dai, Z., Callan, J., Liu, Z., & Power, R. 
          (2017, August). End-to-end neural ad-hoc ranking with 
          kernel pooling. In Proceedings of the 40th International 
          ACM SIGIR conference on research and development in 
          information retrieval (pp. 55-64). ACM.
          (http://www.cs.cmu.edu/~zhuyund/papers/end-end-neural.pdf)
    
    [3] - Dai, Z., Xiong, C., Callan, J., & Liu, Z. (2018, February). 
          Convolutional neural networks for soft-matching n-grams 
          in ad-hoc search. In Proceedings of the eleventh ACM 
          international conference on web search and data mining 
          (pp. 126-134). ACM.
          (http://www.cs.cmu.edu/~./callan/Papers/wsdm18-zhuyun-dai.pdf)
    
    """
    def __init__(self, config, use_knrm=False, K=11, lamb=0.5, 
                 use_exact=True, last_layer_only=True, N=None, 
                 method="mean", weights=None, mu_sigma_learnable=False):
        super(BertKnrm, self).__init__(config)
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # bert encoding from different layers options
        if not last_layer_only:
            # if N is None, will consider last 5 layers
            if N is None:
                N = 5
            self.N = N
            if method not in ("avg", "wavg", "sum", "wsum", "max", "selfattn"):
                method = "avg"
            self.method = method
            if method.startswith("w"):
                if weights is None or len(weights) != self.N:
                    # better weights setting? maybe make them learnable if used?
                    # fix me: hard coded 12 layers
                    self.weights = torch.linspace(0.01, 1.0, self.N if self.N else 12)
                else:
                    self.weights = torch.tensor(weights, dtype=torch.float)
        self.last_layer_only = last_layer_only
        
        if use_knrm:
            # kernels options
            self.K = K
            # make mu and sigma learnable, otherwise use values from paper
            if not mu_sigma_learnable:
                self.mus = torch.tensor(
                    self.kernal_mus(K, use_exact), 
                    dtype=torch.float
                )
                self.sigmas = torch.tensor(
                    self.kernel_sigmas(K, lamb, use_exact), 
                    dtype=torch.float
                )
            else:
                self.mus = nn.Parameter(torch.randn(K).float())
                self.sigmas = nn.Parameter(torch.randn(K).float())
            self.mu_sigma_learnable = mu_sigma_learnable
            # output layers for final score
            self.linear = nn.Linear(K, 1)
        else:
            self.linear = nn.Linear(768, 1)
        self.use_knrm = use_knrm
        
        self.activation = nn.Tanh()
        self.apply(self.init_bert_weights)
    
    def to_device(self, device):
        if not self.mu_sigma_learnable:
            self.mus = self.mus.to(device)
            self.sigmas = self.sigmas.to(device)
        if not self.last_layer_only and self.method.startswith("w"):
            self.weights = self.weights.to(device)
    
    def kernal_mus(self, n_kernels, use_exact):
        """Get mu value for each Gaussian kernel. Mu is
        the middle of each bin.
        
        Parameters
        ----------
            n_kernels : int
                Number of kernel (including exact match),
                first one is exact match.
            use_exact : bool
                Whether to use exact match kernel.
        
        Returns
        -------
            l_mu : list of float
                List of mu values.
        
        References
        ----------
            Taken from K-NRM source:
            https://github.com/AdeDZY/K-NRM/blob/master/knrm/model/model_base.py
        
        """
        if use_exact:
            l_mu = [1]
        else:
            l_mu = [2]
        if n_kernels == 1:
            return l_mu
        
        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu
    
    def kernel_sigmas(self, n_kernels, lamb, use_exact):
        """Get sigma value for each Gaussian kernel.
        
        Parameters
        ----------
            n_kernels : int
                Number of kernels (including exact match).
            lamb : float
                Defines the gaussian kernels' sigma value.
            use_exact : bool
                Whether to use exact match kernel.
        
        Returns
        -------
            l_sigma : list of float
                List of sigma values.
        
        References
        ----------
            Taken from K-NRM source:
            https://github.com/AdeDZY/K-NRM/blob/master/knrm/model/model_base.py
        
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.00001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma
        
        l_sigma += [bin_size * lamb] * (n_kernels - 1)
        return l_sigma
    
    def encoded_layers_transform(self, encoded_layers):
        """Utility function to play with BERT layers."""
        if self.last_layer_only:
            return encoded_layers[-1]
        
        # list of N layers of shape B x L x H
        if self.N:
            encoded_layers = encoded_layers[-self.N:]
        else:
            self.N = len(encoded_layers)
        
        N = self.N
        B, L, H = encoded_layers[0].shape
        
        # > [(B x L x H), ..., (B x L x H)] -> B x L x NH
        encoded_layers = torch.cat(encoded_layers, dim=2)
        # > B x L x N x H
        encoded_layers = encoded_layers.reshape(B, L, N, H)
        # > B x N x L x H
        encoded_layers = encoded_layers.permute(0, 2, 1, 3)
        
        if self.method == "selfattn":
            # > B x N x LH
            encoded_layers = encoded_layers.contiguous().view(B, N, L*H)
            # > B x N x LH * B x LH x N --bmm--> B x N x N
            attention_scores = encoded_layers.bmm(encoded_layers.transpose(-1, -2))
            attention_scores = torch.softmax(attention_scores, dim=-1)
            
            # soft (attended) layers
            # > B x N x LH
            soft_encoded_layers = attention_scores.bmm(encoded_layers)
            # add with residual
            encoded_layers = encoded_layers + soft_encoded_layers
            # average over all layers
            # > B x LH
            encoded_layers = encoded_layers.mean(dim=1)
            # > B x L x H
            output = encoded_layers.view(B, L, H)
        else:
            if self.method.startswith("w"):
                # > N --unsqueeze--> N x 1 --expand--> N x LH 
                #     --unsqueeze--> 1 x N x LH --expand--> B x N x LH
                weights = self.weights.unsqueeze(-1).expand(-1, L*H).unsqueeze(0).expand(2, -1, -1)
                # > B x N x L x H
                weights = weights.reshape(B, N, L, H)
                output = weights * encoded_layers
                # > B x L x H
                output = output.sum(1) if "sum" in self.method else output.mean(1)
            else:
                # > B x L x H
                if self.method == "sum":
                    output = encoded_layers.sum(1)
                elif self.method == "avg":
                    output = encoded_layers.mean(1)
                else:
                    output, _ = encoded_layers.max(1)
        
        return output
    
    def knrm(self, embedded, segment_ids, input_mask):
        #
        # input_mask : B x L
        # segment_ids : B x L
        #
        # * note: BERT default LRs [2|3|5]e-5 did not
        # worked for KNRM. Loss does not change because
        # LR is too small, changing to 1e-4 works.
        #
        # Original K-NRM implementation:
        # https://github.com/AdeDZY/K-NRM/blob/master/knrm/model/model_knrm.py#L74
        #
        document_ids_mask = segment_ids * input_mask
        query_ids_mask = (1 - segment_ids) * input_mask
        # batch wise outer product to get query-doc masks
        query_doc_mask = torch.bmm(
            query_ids_mask.unsqueeze(2).float(), 
            document_ids_mask.unsqueeze(1).float()
        )
        
        embedded_normalized = F.normalize(embedded, p=2, dim=2)
        # B x L x H * B x H x L --> B x L x L
        M = embedded_normalized.bmm(embedded_normalized.transpose(1, 2))
        # B x L x L x 1
        phi_M = M.unsqueeze(-1)
        
        # eq. 4 numerator and denominator
        # > (B x L x L x 1 - K) --broadcasted--> B x L x L x K
        numerator = phi_M - self.mus
        numerator = -(numerator ** 2)
        # denominator is K sized vector
        denominator = 2 * (self.sigmas ** 2)
        # eq. 4 without summation
        # > B x L x L x K
        phi_M = torch.exp(numerator/denominator)
        
        # apply masks
        query_doc_mask = query_doc_mask.unsqueeze(-1).float()
        phi_M = phi_M * query_doc_mask
        
        # sum along document dimension (eq. 4 with summation)
        # > B x L x K
        phi_M = phi_M.sum(2)
        
        # clip small values
        phi_M[phi_M < 1e-10] = 1e-10
        phi_M = torch.log(phi_M) * 0.01
        
        # sum over query features to get TF-soft features
        # > B x K
        phi_M = phi_M.sum(1)
        
        return phi_M
    
    def forward(self, input_ids, segment_ids, input_mask):
        #
        # embedded : B x L x H
        # cls_embed : B x H
        # where, L is joint length i.e. "query len + doc len"
        #
        embedded, cls_embed = self.bert(
            input_ids, segment_ids, input_mask, 
            output_all_encoded_layers=not self.last_layer_only
        )
        if not self.last_layer_only:
            embedded = embedded[-self.N:]
            embedded = self.encoded_layers_transform(embedded)
            cls_embed = self.bert.pooler(embedded)
        
        if self.use_knrm:
            phi_M = self.knrm(embedded, segment_ids, input_mask)
            output = self.linear(phi_M).squeeze(-1)
        else:
            output = self.linear(cls_embed).squeeze(-1)
        
        output = self.activation(output)
        
        return output


if __name__=="__main__":
    # dummy testing to see if forward pass is OK
    input_ids_q = torch.tensor([
        [123, 121, 4311, 0, 0],
        [1, 102, 54, 15, 0]
    ], dtype=torch.long)
    segment_ids_q = torch.tensor([
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 1]
    ], dtype=torch.long)
    input_mask_q = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ], dtype=torch.long)
    
    bert_model_dir = "path/to/bert_model"
    model = BertKnrm.from_pretrained(bert_model_dir, use_knrm=True)
    out = model(input_ids_q, segment_ids_q, input_mask_q)
    print(out)
