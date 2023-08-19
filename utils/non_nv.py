# Copyright (C) 2022 NVIDIA Corporation.  All rights reserved.
#
# This file contain modified functions snippets from 
# https://github.com/yuvalatzmon/COSMO
# and
# https://github.com/openai/CLIP
#
# The license for the original version of these functions can be
# found in root directory: LICENSE_COSMO, LICENSE_CLIP, respetively. The modifications
# to this file are subject to the License located at the root directory.
# ---------------------------------------------------------------

import torch

num_tokens = 77

def encode_text_with_learnt_tokens(self, text, asterix_token, learnable_codes, base_token = None, is_emb = False):
    """
        CLIP text encoder replacing the "asterix_token" with learnable_codes

        This function has been modified from a file in the following repository:
        https://github.com/openai/CLIP
        The license for the original version of this file can be
        found in the root directory (LICENSE_CLIP). The modifications
        to this function are subject to the License
        located at the root directory.
    """
    inds_for_insert = torch.where(text == asterix_token)
    x = self.token_embedding(text).type(self.dtype)
    if base_token is None:
        for i in range(len(text)):
            if is_emb:
                learnable_codes.weight[0].retain_grad()
                x_i_longer = torch.cat((x[i][:inds_for_insert[1][i]], learnable_codes.weight, x[i][inds_for_insert[1][i]+1:]), 0)
            else:
                x_i_longer = torch.cat((x[i][:inds_for_insert[1][i]], learnable_codes.squeeze(0), x[i][inds_for_insert[1][i]+1:]), 0)
            x[i] = x_i_longer[:num_tokens]

    x = x + self.positional_embedding.type(self.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x).type(self.dtype)

    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

    return x