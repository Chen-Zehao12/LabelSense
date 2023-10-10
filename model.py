from typing import Optional, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class SemanticModel(BertPreTrainedModel):
    def __init__(self,
                 config,
                 pretrained_model,
                 alpha=1,
                 beta=1,
                 gamma=1,
                 pooling='cls',
                 index_table=None,
                 centroids=None,
                 momentum_rate=0.9):
        super().__init__(config)
        
        self.num_labels = config.num_labels
        self.config = config
        self.pretrained_model = pretrained_model

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pooling = pooling

        self.index_table = index_table
        self.centroids = centroids
        self.momentum_rate = momentum_rate
    
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            global_idxs: Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        output = self.pretrained_model(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            pooled_output = output.last_hidden_state[:, 0]  # [batch, 768]
        
        elif self.pooling == 'pooler':
            pooled_output = output.pooler_output            # [batch, 768]
        
        elif self.pooling == 'last-avg':
            last = output.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            pooled_output = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]
        
        elif self.pooling == 'first-last-avg':
            first = output.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            last = output.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]                   
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            pooled_output = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]


        # 1. Semantic In Batch Negative
        if labels is not None:
            y_true = torch.arange(pooled_output.shape[0], device=pooled_output.device)
            use_row = torch.arange(pooled_output.shape[0], device=pooled_output.device)
            y_true = (use_row - use_row % 2 * 2) + 1
            # Calculate pairwise similarities within a batch to obtain a similarity matrix (diagonal matrix)
            sim = F.cosine_similarity(pooled_output.unsqueeze(1), pooled_output.unsqueeze(0), dim=-1)
            # Set the diagonal of the similarity matrix to a very small value to eliminate self-influence
            sim = sim - torch.eye(pooled_output.shape[0], device=pooled_output.device) * 1e12
            # Select relevant rows
            sim = torch.index_select(sim, 0, use_row)
            # Divide the similarity matrix by a temperature coefficient
            sim = sim / 0.05
            # Calculate the cross-entropy loss between the similarity matrix and y_true
            sim_loss = F.cross_entropy(sim, y_true)
        
        # 2. Text Classification
        cls_output = self.dropout(output.last_hidden_state[:, 0])
        logits = self.classifier(cls_output)
        if labels is not None:
            cls_loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        
        # 3. Centroid
        if labels is not None and self.centroids is not None and self.index_table is not None:
            batch_centorids = torch.zeros(pooled_output.shape, device=pooled_output.device)
            for i in range(len(global_idxs)):
                class_id, in_class_number = self.index_table[global_idxs[i].item()]
                batch_centorids[i] = self.centroids[class_id].mean(dim=0).to(pooled_output.device)
                # 3.1 Momentum update centroids
                self.centroids[class_id][in_class_number] = \
                    self.momentum_rate * self.centroids[class_id][in_class_number] + \
                    (1 - self.momentum_rate) * pooled_output[i].detach().clone().cpu()
            target = torch.ones(pooled_output.shape[0], device=pooled_output.device)
            centroid_loss = F.cosine_embedding_loss(pooled_output, batch_centorids, target)

        # 4. Loss
        loss = self.alpha * sim_loss + self.beta * cls_loss if labels is not None else None
        loss = loss + self.gamma * centroid_loss if loss is not None and self.centroids is not None else loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=pooled_output,
            attentions=None,
        )