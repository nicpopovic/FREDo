import torch
import torch.nn as nn
from src.models.base_model import BaseEncoder

class Encoder(BaseEncoder):
    def __init__(self, config, model, cls_token_id=0, sep_token_id=0, markers=True):
        super().__init__(config=config, model=model, exemplar_method=self.sie_sbn, cls_token_id=cls_token_id, sep_token_id=sep_token_id, markers=markers)
        self.dev = True

    def sie_sbn(self,
                input_ids=None,
                attention_mask=None,
                entity_positions=None,
                labels=None, 
                type_labels=None):

        num_exemplars = input_ids.size(-2)
        batch_size = input_ids.size(0)

        sequence_output, attention = self.encode(input_ids.view(-1, input_ids.size(-1)), attention_mask.view(-1, attention_mask.size(-1)))
        sequence_output = sequence_output.view(-1, num_exemplars, sequence_output.size(-2), sequence_output.size(-1))

        batch_exemplars = []
        batch_label_ids = []
        batch_label_types = []
        for batch_i in range(batch_size):
            episode_label_ids = []
            episode_label_types = []
            entity_embeddings = [[] for _ in entity_positions[batch_i]]
            
            relation_embeddings = []
            label_ids, label_types = [], []
            for batch_item in labels[batch_i]:
                li_in_batch = []
                lt_in_batch = []
                for l_h, l_t, l_r in batch_item:
                    li_in_batch.append((l_h, l_t))
                    lt_in_batch.append(l_r)
                label_ids.append(li_in_batch)
                label_types.append(lt_in_batch)
            
            rts = []
            
            for i, batch_item in enumerate(entity_positions[batch_i]):
                for entity in batch_item:
                    mention_embeddings = []
                    for mention in entity:
                        if self.markers:
                            m_e = sequence_output[batch_i,i,mention[0],:]
                        else:
                            m_e = torch.mean(sequence_output[batch_i,i,mention[0]:mention[1],:], 0)
                        mention_embeddings.append(m_e)

                    e_e = torch.mean(torch.stack(mention_embeddings, 0), 0)

                    entity_embeddings[i].append(e_e)
            
                for i_h, h in enumerate(entity_embeddings[i]):
                    for i_t, t in enumerate(entity_embeddings[i]):
                        if i_h == i_t:
                            continue

                        if (i_h, i_t) in label_ids[i]:
                            episode_label_ids.append(len(relation_embeddings))
                            types_for_label = []
                            for li, lt in zip(label_ids[i], label_types[i]):
                                if li == (i_h, i_t):
                                    types_for_label.append(lt)
                                    rts.append(lt)
                            episode_label_types.append(types_for_label)
                        else:
                            episode_label_ids.append(len(relation_embeddings))
                            episode_label_types.append(["NOTA"])
                        relation_embeddings.append(torch.cat([h, t]))
            
            batch_exemplars.append(torch.stack(relation_embeddings, 0))
            batch_label_ids.append(episode_label_ids)
            batch_label_types.append(episode_label_types)
        
        # create prototype embeddings
        batch_prototypes = []
        k = 5
        if not self.training:
            k = 20
        for exemplars, label_ids, label_types, type_index in zip(batch_exemplars, batch_label_ids, batch_label_types, type_labels):
            episodes_prototypes = [None for _ in type_index]
            # print(label_types)
            for relation_type in type_index:
                embeddings = []
                for i, t in zip(label_ids, label_types):
                    if relation_type in t:
                        embeddings.append(exemplars[i])
                embeddings = torch.stack(embeddings, 0)
                if relation_type != "NOTA" and self.training:
                    embeddings = torch.mean(embeddings, 0, keepdim=True)
                elif relation_type == "NOTA" and self.first_run and self.training:
                    self.nota_embeddings.data = torch.mean(embeddings, 0, keepdim=True)
                    indexes = torch.randperm(embeddings.shape[0])
                    self.nota_embeddings.data = embeddings[indexes[:20], :]
                    self.first_run = False

                episodes_prototypes[type_index.index(relation_type)] = embeddings
            
            selected_notas = []
            selected_indices = []
            for relation_type in type_index:
                if relation_type == "NOTA":
                    continue
                if self.training:
                    class_scores = episodes_prototypes[type_index.index("NOTA")] * episodes_prototypes[type_index.index(relation_type)]
                    class_scores = torch.sum(class_scores, dim=-1)
                    indices = torch.topk(class_scores, k)[1]
                else:
                    class_scores = episodes_prototypes[type_index.index("NOTA")].unsqueeze(0) * episodes_prototypes[type_index.index(relation_type)].unsqueeze(1)
                    class_scores = torch.sum(class_scores, dim=-1)
                    indices = torch.topk(class_scores, min(k, class_scores.size(1)), dim=-1)[1].view(-1)
                selected_indices.extend(indices.tolist())


            selected_indices = list(set(selected_indices))
            if self.training or self.dev:
                episodes_prototypes[type_index.index("NOTA")] = torch.cat((episodes_prototypes[type_index.index("NOTA")][selected_indices,:], self.nota_embeddings))
            else:
                episodes_prototypes[type_index.index("NOTA")] = episodes_prototypes[type_index.index("NOTA")][selected_indices,:]

            batch_prototypes.append(episodes_prototypes)
        
        return batch_prototypes