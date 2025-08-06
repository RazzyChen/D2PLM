from transformers import PretrainedConfig


class DITConfig(PretrainedConfig):
    """DIT模型的配置类"""
    
    model_type = "dit"
    
    def __init__(
        self,
        vocab_size: int = 25,
        max_position_embeddings: int = 512,
        hidden_size: int = 1024,
        num_hidden_layers: int = 10,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        time_embedding_dim: int = 128,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        mask_token_id: int = 1,
        cls_token_id: int = 2,
        eos_token_id: int = 3,
        initializer_range: float = 0.02,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.time_embedding_dim = time_embedding_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.eos_token_id = eos_token_id
        self.initializer_range = initializer_range 