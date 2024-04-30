from torch.utils.data import DataLoader

import torch.nn as nn
import torch


def parse_data(data, device: str):
    return [torch.LongTensor(data[:, i].T).to(device=device) for i in range(data.shape[1])]


class FeatExtractor():
    def __init__(self, params, model, data):
        self.params = params
        self.model = model
        self.data = data
        self.bce_loss_func = nn.BCELoss(reduction='none')

    def eval(self, return_metrics=True):
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=True)
        # ensemble_model: when `_is_frozen=True`, `to_score` in sub-models is False
        self.model.freeze_sub_models()

        result = dict()
        self.model.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                head_ids, tails_ids, target_labels = parse_data(batch, self.params.device)
                score_pos, _other_loss = self.model(head_ids, tails_ids)

                head_ids = head_ids.cpu().numpy().flatten().tolist()
                tail_ids = tails_ids.cpu().numpy().flatten().tolist()

                print("len(head_ids, tail_ids)", len(head_ids), len(tail_ids))
                print("score_pos.shape", score_pos.shape)
                score_pos = score_pos.cpu().numpy()

                for i, [_head_id, _tail_id] in enumerate(zip(head_ids, tail_ids)):
                    result[(_head_id, _tail_id)] = score_pos[i]
                print("len(result)", len(result))

        return result
