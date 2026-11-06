import torch
import torch.nn as nn

from typing import List, Tuple, Type, Optional
from configs.config_DNN import config_DNN


from typing import List, Tuple, Type, Optional


class TabularBlock(nn.Module):
    """
    Baseline: Linear -> [BatchNorm1d] -> Activation -> [Dropout]
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_cls: Type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = True,
        dropout_p: float = 0.0
    ):
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_features))
        
        layers.append(activation_cls())

        if dropout_p > 0.0:
            layers.append(nn.Dropout(dropout_p))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        return self.block(x)

class TitanicModel(nn.Module):
    def __init__(
        self,
        num_continuous_features : int,
        emb_dims: Optional[List[Tuple[int, int]]] = None,
        hidden_dim: int = config_DNN.model.params.hidden_dims,
        num_blocks: int = config_DNN.model.params.num_blocks,
        activation_cls: Type[nn.Module] = getattr(nn,config_DNN.model.params.activation_cls),
        use_batchnorm: bool = config_DNN.model.params.use_batchnorm,
        dropout_p: float = config_DNN.model.params.dropout
    ):
        """
            Parameters:
            ----------
            num_continuous_features : number of continuous (and one-hot) features.
            emb_dims : list of tuples (num_classes, emb_dim) for category features.
                    If None or empty — embedding is disabled (just One-Hot).
            hidden_dim : dim of hidden layers.
            num_blocks : number of TabularBlock.
            activation_cls : activation function class (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU etc..).
            use_batchnorm : is BatchNorm1d avaible inside the block.
            dropout_p : probability dropout (0.0 — disabled).
            """
        super().__init__()
        self.use_embeddings = emb_dims is not None and (len(emb_dims) > 0)

        if self.use_embeddings:
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_classes, emb_dim) for num_classes, emb_dim in emb_dims
            ])
            total_emb_dim = sum(emb_dim for _,emb_dim in emb_dims)
        else:
            self.embeddings = None

            total_emb_dim = 0
        #1st block dim
        in_dim = num_continuous_features + total_emb_dim

        # Stacking linear block
        blocks = []
        for i in range(num_blocks):
            current_in = in_dim if i==0 else hidden_dim
            blocks.append(
                TabularBlock(
                    in_features = current_in,
                    out_features= hidden_dim,
                    activation_cls = activation_cls,
                    use_batchnorm = use_batchnorm,
                    dropout_p = dropout_p
                )
            )
        self.feature_extractor = nn.Sequential(*blocks)

        #Classification layer
        hidden_in = hidden_dim if num_blocks > 0 else in_dim
        self.classifier = nn.Linear(hidden_in, config_DNN.model.params.output_dim)
    
    def forward(self, x: torch.Tensor, x_cat: Optional[torch.tensor] = None) -> torch.Tensor:
        """
        x_num:  continious one-hot feautures tensor [batch_size, num_continuous_features]
        x_cat: category features tensor [batch_size, len(emb_dims)], if use_embeddings=True
        """
        if self.use_embeddings:
            if x_cat is None:
                num_cats = len(self.embeddings)
                x_num = x[:, :-num_cats]
                x_cat = x[:, -num_cats:].long()
            else:
                x_num = x
            emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x_emb = torch.cat(emb_outs, dim=1)
            x = torch.cat([x_num, x_emb], dim=1)
        
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits