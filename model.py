import config
from transformers import BertModel, BertTokenizer
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable


class Omni(nn.Module):
    def __init__(self, geo_embed_model, multi_class=False, device='cuda', att_aff=False, finetuning=True,
                 c_emb=config.c_em, g_emb=config.g_emb, dropout=0.1):

        super().__init__()

        self.multi_class = multi_class
        hidden_size = config.lm_hidden
        self.run_att_aff = att_aff
        self.language_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.device = device
        self.finetuning = finetuning

        self.geo_embed_model = geo_embed_model

        self.drop = nn.Dropout(dropout)
        self.coord_linear = nn.Linear(1, 2 * c_emb)
        self.min_dist_linear = nn.Linear(1, 2 * c_emb)

        if att_aff:
            self.linear1 = nn.Linear(7 * hidden_size + 2 * g_emb + 2 * c_emb,
                                     (7 * hidden_size + 2 * g_emb + 2 * c_emb) // 2)
            if multi_class:
                self.linear2 = nn.Linear((7 * hidden_size + 2 * g_emb + 2 * c_emb) // 2, 4)
            else:
                self.linear2 = nn.Linear((7 * hidden_size + 2 * g_emb + 2 * c_emb) // 2, 2)

        else:
            self.linear1 = nn.Linear(hidden_size + 2 * g_emb + 2 * c_emb,
                                     (hidden_size + 2 * g_emb + 2 * c_emb) // 2)
            if multi_class:
                self.linear2 = nn.Linear((7 * hidden_size + 2 * g_emb + 2 * c_emb) // 2, 4)
            else:
                self.linear2 = nn.Linear((hidden_size + 2 * g_emb + 2 * c_emb) // 2, 2)

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x, x_coord, x_min_dists, att_mask, x_val_pos, x_cos_pos, geo_left, geo_right, training=True):

        x_coord = x_coord.to(self.device)

        while len(x_coord.shape) < 2:
            x_coord = x_coord.unsqueeze(0)

        while len(x_min_dists.shape) < 2:
            x_min_dists = x_min_dists.unsqueeze(0)

        x = x.to(self.device)
        att_mask = att_mask.to(self.device)
        if len(x.shape) < 2:
            x = x.unsqueeze(0)

        if len(att_mask.shape) < 2:
            att_mask = att_mask.unsqueeze(0)

        if training and self.finetuning:
            self.language_model.train()
            self.train()

            self.geo_embed_model.train()

            output = self.language_model(x, attention_mask=att_mask)
            pooled_output = output[0][:, 0, :]

            name_outputs = []
            type_outputs = []

            if self.run_att_aff:
                for i, e in enumerate(x_val_pos):
                    name_output = [output[0][i, token_idx, :] for token_idx in [e[0], e[2]]]
                    type_output = [output[0][i, token_idx, :] for token_idx in [e[1], e[3]]]

                    name_outputs.append(name_output)
                    type_outputs.append(type_output)

        else:

            self.language_model.eval()
            self.geo_embed_model.eval()
            with torch.no_grad():
                output = self.language_model(x, attention_mask=att_mask)
                pooled_output = output[0][:, 0, :]

                name_outputs = []
                type_outputs = []

                if self.run_att_aff:
                    for i, e in enumerate(x_val_pos):
                        name_output = [output[0][i, token_idx, :] for token_idx in [e[0], e[2]]]
                        type_output = [output[0][i, token_idx, :] for token_idx in [e[1], e[3]]]

                        name_outputs.append(name_output)
                        type_outputs.append(type_output)

        x_coord = x_coord.transpose(0, 1)
        x_coord = self.coord_linear(x_coord)

        x_min_dists = x_min_dists.transpose(0, 1)
        x_min_dists = self.min_dist_linear(x_min_dists)

        x_geo_left = self.geo_embed_model(geo_left)
        x_geo_right = self.geo_embed_model(geo_right)

        if self.run_att_aff:

            name_conc = torch.stack([torch.cat(y, dim=0) for y in name_outputs])
            name_multp = torch.stack([torch.add(y[0], y[1]) for y in name_outputs])

            type_conc = torch.stack([torch.cat(y, dim=0) for y in type_outputs])
            type_multp = torch.stack([torch.add(y[0], y[1]) for y in type_outputs])

            name_concat = torch.cat([name_conc, name_multp], 1)
            type_concat = torch.cat([type_conc, type_multp], 1)
            affinity = torch.cat([name_concat, type_concat], 1)

            output = torch.cat([pooled_output, affinity, x_geo_left, x_geo_right, x_coord], 1)
        else:
            output = torch.cat([pooled_output, x_geo_left, x_geo_right, x_coord], 1)

        output = self.linear2(self.drop(self.gelu(self.linear1(output))))

        return F.log_softmax(output, dim=1)


class OmniSmall(nn.Module):
    def __init__(self, geo_embed_model, multi_class=False, device='cuda', att_aff=False, finetuning=True,
                 c_emb=config.c_em, g_emb=config.g_emb, dropout=0.1):

        super().__init__()

        self.multi_class = multi_class
        hidden_size = config.lm_hidden
        self.run_att_aff = att_aff
        self.language_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.device = device
        self.finetuning = finetuning

        self.geo_embed_model = geo_embed_model

        self.drop = nn.Dropout(dropout)
        self.coord_linear = nn.Linear(1, c_emb)
        self.min_dist_linear = nn.Linear(1, c_emb)

        self.linear1 = nn.Linear(2 + hidden_size + 2 * g_emb + 2 * c_emb,
                                 (2 + hidden_size + 2 * g_emb + 2 * c_emb) // 2)
        if multi_class:
            self.linear2 = nn.Linear((2 + hidden_size + 2 * g_emb + 2 * c_emb) // 2, 4)
        else:
            self.linear2 = nn.Linear((2 + hidden_size + 2 * g_emb + 2 * c_emb) // 2, 2)

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x, x_coord, x_min_dists, att_mask, x_val_pos, x_cos_pos, geo_left, geo_right, training=True):

        x_coord = x_coord.to(self.device)

        while len(x_min_dists.shape) < 2:
            x_min_dists = x_min_dists.unsqueeze(0)

        while len(x_coord.shape) < 2:
            x_coord = x_coord.unsqueeze(0)

        x = x.to(self.device)
        att_mask = att_mask.to(self.device)
        if len(x.shape) < 2:
            x = x.unsqueeze(0)

        if len(att_mask.shape) < 2:
            att_mask = att_mask.unsqueeze(0)

        if training and self.finetuning:
            self.language_model.train()
            self.train()

            self.geo_embed_model.train()

            output = self.language_model(x, attention_mask=att_mask)
            pooled_output = output[0][:, 0, :]

            name_outputs = []
            # type_outputs = []
            address_outputs = []

            # for i in range(len(x_cos_pos['names'])):
            #
            #     name_output = [torch.mean(output[0][i, x_cos_pos['names'][i][0]:x_cos_pos['names'][i][1], :], dim=0),
            #                    torch.mean(output[0][i, x_cos_pos['names'][i][2]:x_cos_pos['names'][i][3], :], dim=0)]
            #     # type_output = [torch.mean(output[0][i, x_cos_pos['types'][i][0]:x_cos_pos['types'][i][1], :], dim=0),
            #     #                torch.mean(output[0][i, x_cos_pos['types'][i][2]:x_cos_pos['types'][i][3], :], dim=0)]
            #     address_output = [torch.mean(output[0][i, x_cos_pos['addresses'][i][0]:x_cos_pos['addresses'][i][1], :], dim=0),
            #                       torch.mean(output[0][i, x_cos_pos['addresses'][i][2]:x_cos_pos['addresses'][i][3], :], dim=0)]
            for i in range(len(x_cos_pos['attribute1'])):
                name_output = [
                    torch.mean(output[0][i, x_cos_pos['attribute1'][i][0]:x_cos_pos['attribute1'][i][1], :], dim=0),
                    torch.mean(output[0][i, x_cos_pos['attribute1'][i][2]:x_cos_pos['attribute1'][i][3], :], dim=0)]
                # type_output = [torch.mean(output[0][i, x_cos_pos['types'][i][0]:x_cos_pos['types'][i][1], :], dim=0),
                #                torch.mean(output[0][i, x_cos_pos['types'][i][2]:x_cos_pos['types'][i][3], :], dim=0)]
                address_output = [
                    torch.mean(output[0][i, x_cos_pos['attribute2'][i][0]:x_cos_pos['attribute2'][i][1], :], dim=0),
                    torch.mean(output[0][i, x_cos_pos['attribute2'][i][2]:x_cos_pos['attribute2'][i][3], :], dim=0)]

                name_outputs.append(name_output)
                # type_outputs.append(type_output)
                address_outputs.append(address_output)

        else:

            self.language_model.eval()
            self.geo_embed_model.eval()
            with torch.no_grad():
                output = self.language_model(x, attention_mask=att_mask)
                pooled_output = output[0][:, 0, :]

                name_outputs = []
                # type_outputs = []
                address_outputs = []

                # for i in range(len(x_cos_pos['names'])):
                #     name_output = [
                #         torch.mean(output[0][i, x_cos_pos['names'][i][0]:x_cos_pos['names'][i][1], :], dim=0),
                #         torch.mean(output[0][i, x_cos_pos['names'][i][2]:x_cos_pos['names'][i][3], :], dim=0)]
                #     # type_output = [torch.mean(output[0][i, x_cos_pos['types'][i][0]:x_cos_pos['types'][i][1], :], dim=0),
                #     #                torch.mean(output[0][i, x_cos_pos['types'][i][2]:x_cos_pos['types'][i][3], :], dim=0)]
                #     address_output = [
                #         torch.mean(output[0][i, x_cos_pos['addresses'][i][0]:x_cos_pos['addresses'][i][1], :], dim=0),
                #         torch.mean(output[0][i, x_cos_pos['addresses'][i][2]:x_cos_pos['addresses'][i][3], :], dim=0)]

                for i in range(len(x_cos_pos['attribute1'])):
                    name_output = [
                        torch.mean(output[0][i, x_cos_pos['attribute1'][i][0]:x_cos_pos['attribute1'][i][1], :],
                                   dim=0),
                        torch.mean(output[0][i, x_cos_pos['attribute1'][i][2]:x_cos_pos['attribute1'][i][3], :],
                                   dim=0)]
                    # type_output = [torch.mean(output[0][i, x_cos_pos['types'][i][0]:x_cos_pos['types'][i][1], :], dim=0),
                    #                torch.mean(output[0][i, x_cos_pos['types'][i][2]:x_cos_pos['types'][i][3], :], dim=0)]
                    address_output = [
                        torch.mean(output[0][i, x_cos_pos['attribute2'][i][0]:x_cos_pos['attribute2'][i][1], :],
                                   dim=0),
                        torch.mean(output[0][i, x_cos_pos['attribute2'][i][2]:x_cos_pos['attribute2'][i][3], :],
                                   dim=0)]

                    name_outputs.append(name_output)
                    # type_outputs.append(type_output)
                    address_outputs.append(address_output)

        x_coord = x_coord.transpose(0, 1)
        x_coord = self.coord_linear(x_coord)

        x_min_dists = x_min_dists.transpose(0, 1)
        x_min_dists = self.min_dist_linear(x_min_dists)

        x_geo_left = self.geo_embed_model(geo_left)
        x_geo_right = self.geo_embed_model(geo_right)

        name_stacked_tensor = torch.stack([torch.stack(tensors) for tensors in name_outputs])
        name_cos_sim = F.cosine_similarity(name_stacked_tensor[:, 0, :], name_stacked_tensor[:, 1, :], dim=1).unsqueeze(
            1)

        address_stacked_tensor = torch.stack([torch.stack(tensors) for tensors in address_outputs])
        address_cos_sim = F.cosine_similarity(address_stacked_tensor[:, 0, :], address_stacked_tensor[:, 1, :],
                                              dim=1).unsqueeze(1)

        affinity = torch.cat([name_cos_sim, address_cos_sim], 1)

        output = torch.cat([pooled_output, affinity, x_geo_left, x_geo_right, x_coord, x_min_dists], 1)

        output = self.linear2(self.drop(self.gelu(self.linear1(output))))

        return F.log_softmax(output, dim=1)
