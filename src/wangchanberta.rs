use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, Activation, Dropout, Embedding, LayerNorm, VarBuilder};
use serde::Deserialize;

// https://github.com/huggingface/transformers/blob/b05df6611e6e3e6834acca2b50baeb7cdd5fbe3c/src/transformers/models/camembert/configuration_camembert.py#L29
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: Activation,
    pub hidden_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub position_embedding_type: String,
    pub use_cache: bool,
    pub classifier_dropout: Option<f64>
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 25005,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: Activation::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 1,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            position_embedding_type: "absolute".to_string(),
            use_cache: true,
            classifier_dropout: None
        }
    }
}

// https://github.com/huggingface/transformers/blob/b05df6611e6e3e6834acca2b50baeb7cdd5fbe3c/src/transformers/models/camembert/modeling_camembert.py#L80
struct WangchanbertaEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    padding_idx: u32,
    span: tracing::Span,
}

impl WangchanbertaEmbeddings {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings")
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings")
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings")
        )?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm")
        )?;
        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob),
            padding_idx: config.pad_token_id,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    fn create_position_ids_from_input_ids(&self, input_ids: &Tensor, past_key_values_length: u32) -> Result<Tensor> {
        let mask = input_ids
            .ne(self.padding_idx)?
            .to_dtype(DType::U32)?;
        let cumsum = mask.cumsum(1)?;
        let incremental_indices = ((cumsum) + past_key_values_length * mask)?;
        incremental_indices + self.padding_idx
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_bsize, _) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        //let mut embeddings = 
    }
}
