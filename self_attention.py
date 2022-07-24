import torch
import torch.nn as nn


def ScaledDotProductAttention(query: torch.Tensor,
                              keys: torch.Tensor,
                              values: torch.Tensor):
    """
    This function apply Scaled Dot-Product Attention Matrix Operation.

    Input: Query, Keys and Values matrixes.

    Returns: torch.Tensor, with the embedding size of the inputs.
    """
    print("query shape: ", query.shape)
    print("keys shape: ", keys.shape)
    print("values shape: ", values.shape)
    outs = torch.dot(query, keys) / query.shape[-1]
    print("outs shape: ", outs.shape)
    outs = torch.softmax(outs)
    return torch.matmul(outs, values)
                              
if __name__ == "__main__":
    Q = torch.ones((20,))
    K = torch.ones((20,))
    V = torch.ones((20,))

    outs = ScaledDotProductAttention(Q, K, V)
    print(outs)
    print(outs.shape)
