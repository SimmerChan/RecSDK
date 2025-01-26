from dataset import Batch
from torchrec import EmbeddingBagCollection, EmbeddingBagConfig, PoolingType
import torch


class TestModel(torch.nn.Module):
    def __init__(self, table_names, feat_names, embed_dims, num_embeds):
        super().__init__()
        table_configs = []
        for table_name, feat_name, dim, num_embed in zip(
            table_names, feat_names, embed_dims, num_embeds
        ):
            config = EmbeddingBagConfig(
                name=table_name,
                embedding_dim=dim,
                num_embeddings=num_embed,
                feature_names=feat_name,
                pooling=PoolingType.SUM,
            )
            table_configs.append(config)

        self.ebc = EmbeddingBagCollection(device="npu", tables=table_configs)
        self.input_dim = sum([len(f) * d for f, d in zip(feat_names, embed_dims)])

    def forward(self, batch: Batch):
        result = self.ebc(batch.sparse_features)
        result: torch.Tensor = result.values()
        loss = result.mean() + result.sum() + result.max() + result.min()
        return loss, result
