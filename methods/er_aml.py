import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from methods.er import ER
from model import normalize

class ER_AML(ER):
    def __init__(self, model, train_tf, args):
        super(ER_AML, self).__init__(model, train_tf, args)

        # can still be task-based, but we want to sample
        # across all classes in the sampling process
        self.sample_kwargs['exclude_task'] = None


    def sup_con_loss(self, anchor_feature, features, anch_labels=None, labels=None,
                    mask=None, temperature=0.1, base_temperature=0.07):

        device = features.device

        if features.ndim < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if features.ndim > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            anch_labels = anch_labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                print(f"len of labels: {len(labels)}")
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(anch_labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (temperature / base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


    def process_inc(self, inc_data):
        """ get loss from incoming data """

        if inc_data['t'] > 0 or (self.args.task_free and len(self.buffer) > 0):
            # do fancy pos neg
            pos_x, neg_x_same_t, neg_x_diff_t, invalid_idx, pos_y, neg_y_same_t, _ = \
                    self.buffer.sample_pos_neg(
                        inc_data,
                        task_free=self.args.task_free
                    )

            # normalized hidden incoming
            hidden  = self.model.return_hidden(inc_data['x'])
            hidden_norm = normalize(hidden[~invalid_idx])

            all_xs  = torch.cat((pos_x, neg_x_same_t))
            all_hid = normalize(self.model.return_hidden(all_xs))
            all_hid = all_hid.reshape(2, pos_x.size(0), -1)
            pos_hid, neg_hid_same_t = all_hid[:, ~invalid_idx]

            if (~invalid_idx).any():

                inc_data['y'] = inc_data['y'][~invalid_idx]
                pos_y = pos_y[~invalid_idx]
                neg_y_same_t = neg_y_same_t[~invalid_idx]
                hid_all = torch.cat((pos_hid, neg_hid_same_t), dim=0)
                y_all = torch.cat((pos_y, neg_y_same_t), dim=0)

                loss = self.sup_con_loss(
                        labels=y_all,
                        features=hid_all.unsqueeze(1),
                        anch_labels=inc_data['y'].repeat(2),
                        anchor_feature=hidden_norm.repeat(2, 1),
                        temperature=self.args.supcon_temperature,
                )

            else:
                loss = 0.

        else:
            # do regular training at the start
            loss = self.loss(self.model(inc_data['x']), inc_data['y'])

        return loss


class ER_AML_Triplet(ER_AML):

    def process_inc(self, inc_data):
        """ get loss from incoming data """

        if inc_data['t'] > 0 or (self.args.task_free and len(self.buffer) > 0):
            # do fancy pos neg
            pos_x, neg_x_same_t, neg_x_diff_t, invalid_idx, _, _, _ = \
                    self.buffer.sample_pos_neg(
                        inc_data,
                        task_free=self.args.task_free
                    )

            if self.args.buffer_neg > 0:
                all_xs  = torch.cat((pos_x, neg_x_same_t, neg_x_diff_t))
                all_hid = normalize(self.model.return_hidden(all_xs))
                all_hid = all_hid.reshape(3, pos_x.size(0), -1)
                pos_hid, neg_hid_same_t, neg_hid_diff_t = all_hid[:, ~invalid_idx]
            else:
                all_xs  = torch.cat((pos_x, neg_x_same_t))
                all_hid = normalize(self.model.return_hidden(all_xs))
                all_hid = all_hid.reshape(2, pos_x.size(0), -1)
                pos_hid, neg_hid_same_t = all_hid[:, ~invalid_idx]

            hidden = self.model.return_hidden(inc_data['x'])
            hidden_norm = normalize(hidden[~invalid_idx])

            if (~invalid_idx).any():
                loss = self.args.incoming_neg * \
                        F.triplet_margin_loss(
                                hidden_norm,
                                pos_hid,
                                neg_hid_same_t,
                                self.args.margin
                        )
                if self.args.buffer_neg > 0:
                    loss += self.args.buffer_neg * \
                            F.triplet_margin_loss(
                                    hidden_norm,
                                    pos_hid,
                                    neg_hid_diff_t,
                                    self.args.margin
                            )
            else:
                loss = 0.

        else:
            # do regular training at the start
            loss = self.loss(self.model(inc_data['x']), inc_data['y'])

        return loss
