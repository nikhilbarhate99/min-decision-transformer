# Decision Transformer

Clean and readable code for [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345).

Notable difference from official implementation are:

- Simple GPT implementation (causal transformer)
- Uses PyTorch's Dataset and Dataloader class and removes redundant computations for calculating rewards to go and state normalization for efficient training


## Results

![](https://github.com/nikhilbarhate99/min-decision-transformer/blob/master/media/halfcheetah-medium-v2.png)

## References

- official [code](https://github.com/kzl/decision-transformer) and [paper](https://arxiv.org/abs/2106.01345)
- minimal GPT (causal transformer) [tweet](https://twitter.com/MishaLaskin/status/1481767788775628801?cxt=HHwWgoCzmYD9pZApAAAA) and [colab notebook](https://colab.research.google.com/drive/1NUBqyboDcGte5qAJKOl8gaJC28V_73Iv?usp=sharing)
