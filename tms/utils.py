import einops
import torch as tc
import numpy as np
from transformer_lens import HookedTransformerConfig, HookedTransformer

import jax
jax.config.update('jax_default_matmul_precision', 'float32')
import jax.numpy as jnp
from tracr.compiler import compiling
from tracr.compiler import lib
from tracr.rasp import rasp

def load_model(input_size=10, vocab_size=10):
    vocab = {*range(vocab_size)}
    program = lib.make_sort(rasp.tokens, rasp.tokens, max_seq_len=input_size, min_key=0)

    tracr_model = compiling.compile_rasp_to_model(
        program=program,
        vocab=vocab,
        max_seq_len=input_size,
        compiler_bos="bos",
        mlp_exactness=100)

    cfg = cfg_from_tracr(tracr_model)
    model = HookedTransformer(cfg)
    model = load_tracr_weights(model, tracr_model, cfg)

    return model, tracr_model, cfg

def cfg_from_tracr(assembled_model):
    """generate an empty model from the tracr model code taken from:
    https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Tracr_to_Transformer_Lens_Demo.ipynb#scrollTo=bgM5a_Ct5k1V


    Args:
        assembled_model: Tracr model

    Returns:
        HookedTransformer: empty model
    """
    n_heads = assembled_model.model_config.num_heads
    n_layers = assembled_model.model_config.num_layers
    d_head = assembled_model.model_config.key_size
    d_mlp = assembled_model.model_config.mlp_hidden_size
    # Activation function is a GeLu, this the standard activation for tracr as far as I can tell
    act_fn = "relu"
    normalization_type = "LN" if assembled_model.model_config.layer_norm else None
    attention_type = "causal" if assembled_model.model_config.causal else "bidirectional"

    n_ctx = assembled_model.params["pos_embed"]['embeddings'].shape[0]
    # Equivalent to length of vocab, with BOS and PAD at the end
    d_vocab = assembled_model.params["token_embed"]['embeddings'].shape[0]
    # Residual stream width, I don't know of an easy way to infer it from the above config.
    d_model = assembled_model.params["token_embed"]['embeddings'].shape[1]

    # Equivalent to length of vocab, WITHOUT BOS and PAD at the end because we never care about
    # these outputs. In practice, we always feed the logits into an argmax
    d_vocab_out = assembled_model.params["token_embed"]['embeddings'].shape[0] - 2

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_head,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        d_vocab_out=d_vocab_out,
        d_mlp=d_mlp,
        n_heads=n_heads,
        act_fn=act_fn,
        attention_dir=attention_type,
        normalization_type=normalization_type,
    )

    return cfg


def load_tracr_weights(tr_model, model, cfg):
    """
    Load the weights from the tracr model, code taken from:
    https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Tracr_to_Transformer_Lens_Demo.ipynb#scrollTo=bgM5a_Ct5k1V

    Args:
        tr_model: HookedTransformer, the empty model to which the weights will be loaded
        model: tracr model, the model from which the weights will be loaded
        cfg: configuration of the model

    Returns:
        HookedTransformer: model with weights from tracr model
    """

    n_heads = cfg.n_heads
    n_layers = cfg.n_layers
    d_head = cfg.d_head
    d_model = cfg.d_model
    d_vocab_out = cfg.d_vocab_out

    sd = {}
    sd["pos_embed.W_pos"] = model.params["pos_embed"]['embeddings']
    sd["embed.W_E"] = model.params["token_embed"]['embeddings']
    # Equivalent to max_seq_len plus one, for the BOS

    # The unembed is just a projection onto the first few elements of the residual stream, these store output tokens
    sd["unembed.W_U"] = jnp.eye(d_model, d_vocab_out)
    sd["unembed.b_U"] = jnp.ones(d_vocab_out)

    for l in range(n_layers):
        sd[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/key"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_K"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/key"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/query"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/query"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/value"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_V"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/value"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/linear"]["w"],
            "(n_heads d_head) d_model -> n_heads d_head d_model",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_O"] = model.params[f"transformer/layer_{l}/attn/linear"]["b"]

        sd[f"blocks.{l}.mlp.W_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["w"]
        sd[f"blocks.{l}.mlp.b_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["b"]
        sd[f"blocks.{l}.mlp.W_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["w"]
        sd[f"blocks.{l}.mlp.b_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["b"]

    for k, v in sd.items():
        # I cannot figure out a neater way to go from a Jax array to a numpy array
        sd[k] = tc.tensor(np.array(v))

    tr_model.load_state_dict(sd, strict=False)

    for k, v in tr_model.state_dict().items():
        if k in sd.keys():
            assert v.shape == sd[k].shape, f'expected {sd[k].shape} but got {v.shape} for {k}'
            v = v.to(sd[k].dtype)
            assert tc.allclose(v, sd[k].to(v.device)), f'difference between {k} and {v} in the model and the tracr model: {tc.norm(v - sd[k].to(v.device))}'

    return tr_model



if "__name__" == "__main__":
    import jax
    jax.config.update('jax_default_matmul_precision', 'float32')
    from tracr.compiler import compiling
    from tracr.compiler import lib
    from tracr.rasp import rasp
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    input_size = 10
    vocab = {*range(input_size)}
    program = lib.make_sort(rasp.tokens, rasp.tokens, max_seq_len=input_size, min_key=0)

    assembled_model = compiling.compile_rasp_to_model(
        program=program,
        vocab=vocab,
        max_seq_len=input_size,
        compiler_bos="bos",
        mlp_exactness=100)
    cfg = cfg_from_tracr(assembled_model)

    model = HookedTransformer(cfg)
    model = load_tracr_weights(model, assembled_model, cfg)
    print('run')

