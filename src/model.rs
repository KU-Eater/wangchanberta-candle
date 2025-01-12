use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Activation, Dropout, Embedding, LayerNorm, Linear, VarBuilder};
use serde::Deserialize;

// https://github.com/huggingface/transformers/blob/b05df6611e6e3e6834acca2b50baeb7cdd5fbe3c/src/transformers/models/camembert/configuration_camembert.py#L29
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
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

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_bsize, _) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let mut embeddings = (&input_embeddings + token_type_embeddings)?;
        if let Some(position_embeddings) = &self.position_embeddings {
            let mask = input_ids
                .ne(self.padding_idx)?
                .to_dtype(input_embeddings.dtype())?;
            let cumsum = mask.cumsum(1);
            let position_ids = (cumsum * mask)?
                .broadcast_add(
                    &Tensor::try_from(self.padding_idx)?
                        .to_dtype(input_embeddings.dtype())?
                        .to_device(input_embeddings.device())?,
                )?
                .to_dtype(candle_core::DType::U32)?;
            embeddings = embeddings.broadcast_add(&position_embeddings.forward(&position_ids)?)?;
        }
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings, true)?;
        Ok(embeddings)
    }
}

struct WangchanbertaSelfAttention {
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
    query: Linear,
    key: Linear,
    value: Linear
}

impl WangchanbertaSelfAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attention_head_size = cfg.hidden_size / cfg.num_attention_heads;
        let all_head_size = cfg.num_attention_heads * attention_head_size;
        Ok(Self {
            num_attention_heads: cfg.num_attention_heads,
            attention_head_size,
            all_head_size,
            query: linear(cfg.hidden_size, all_head_size, vb.pp("query"))?,
            key: linear(cfg.hidden_size, all_head_size, vb.pp("key"))?,
            value: linear(cfg.hidden_size, all_head_size, vb.pp("value"))?
        })
    }

    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = x.dims().to_vec();
        new_x_shape[2] = self.num_attention_heads;
        new_x_shape.push(self.attention_head_size);
        let x = x.reshape(new_x_shape)?;
        x.permute((0, 2, 1, 3))?.contiguous()
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: &Tensor,
        past_key_value: Option<(&Tensor, &Tensor)>,
        encoder_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mixed_query_layer = self.query.forward(hidden_states)?;
        let is_cross_attention = encoder_hidden_states.is_some();
        let (key_layer, value_layer, attention_mask) = if is_cross_attention
            && past_key_value.is_some()
        {
            let key_layer = past_key_value.unwrap().0.clone();
            let value_layer = past_key_value.unwrap().1.clone();
            let attention_mask = encoder_attention_mask.unwrap().clone();
            (key_layer, value_layer, Some(attention_mask))
        } else if is_cross_attention {
            let key_layer =
                self.transpose_for_scores(&self.key.forward(encoder_hidden_states.unwrap())?)?;
            let value_layer =
                self.transpose_for_scores(&self.value.forward(encoder_hidden_states.unwrap())?)?;
            let attention_mask = encoder_attention_mask.unwrap();
            (key_layer, value_layer, Some(attention_mask.clone()))
        } else if past_key_value.is_some() {
            let mut key_layer = self.transpose_for_scores(&self.key.forward(hidden_states)?)?;
            let mut value_layer = self.transpose_for_scores(&self.value.forward(hidden_states)?)?;
            key_layer = Tensor::cat(
                &[
                    past_key_value.clone().as_ref().unwrap().0.clone(),
                    key_layer,
                ],
                2,
            )?;
            value_layer = Tensor::cat(
                &[past_key_value.as_ref().unwrap().1.clone(), value_layer],
                2,
            )?;
            (key_layer, value_layer, Some(attention_mask.clone()))
        } else {
            let key_layer = self.transpose_for_scores(&self.key.forward(hidden_states)?)?;
            let value_layer = self.transpose_for_scores(&self.value.forward(hidden_states)?)?;
            (key_layer, value_layer, Some(attention_mask.clone()))
        };

        let query_layer = self.transpose_for_scores(&mixed_query_layer)?;
        let mut attention_scores = query_layer.matmul(&key_layer.transpose(2, 3)?)?;
        let scale = 1f64 / f64::sqrt(self.attention_head_size as f64);

        attention_scores = (attention_scores * scale)?;
        attention_scores = match attention_mask {
            None => attention_scores,
            Some(mask) => {
                attention_scores.broadcast_add(&mask.to_dtype(attention_scores.dtype())?)?
            }
        };
        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;

        let context_layer = attention_probs
            .matmul(&value_layer)?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let mut new_context_layer_shape =
            context_layer.dims()[..context_layer.dims().len() - 2].to_vec();
        new_context_layer_shape.push(self.all_head_size);
        let context_layer = context_layer.reshape(new_context_layer_shape)?;

        Ok(context_layer)
    }
}

struct WangchanbertaSelfOutput {
    dense: Linear,
    layernorm: LayerNorm
}

impl WangchanbertaSelfOutput {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
        let layernorm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { dense, layernorm })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.layernorm.forward(&(hidden_states + input_tensor)?)?;
        Ok(hidden_states)
    }
}

struct WangchanbertaAttention {
    output: WangchanbertaSelfOutput,
    self_attention: WangchanbertaSelfAttention
}

impl WangchanbertaAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let output = WangchanbertaSelfOutput::new(cfg, vb.pp("output"))?;
        let self_attention = WangchanbertaSelfAttention::new(cfg, vb.pp("self"))?;
        Ok(Self {
            output, self_attention
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        past_key_value: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        let self_outputs = self.self_attention.forward(
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            past_key_value,
            encoder_attention_mask,
        )?;
        let attention_output = self.output.forward(&self_outputs, hidden_states)?;
        Ok((attention_output, self_outputs))
    }
}

struct WangchanbertaIntermediate {
    dense: Linear,
    intermediate_act_fn: Activation
}

impl WangchanbertaIntermediate {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("dense"))?;
        let intermediate_act_fn = cfg.hidden_act;
        Ok(Self {
            dense,
            intermediate_act_fn,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.intermediate_act_fn.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

struct WangchanbertaOutput {
    dense: Linear,
    layernorm: LayerNorm
}

impl WangchanbertaOutput {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("dense"))?;
        let layernorm =
            candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { dense, layernorm })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.layernorm.forward(&(hidden_states + input_tensor)?)?;
        Ok(hidden_states)
    }
}

struct WangchanbertaLayer {
    attention: WangchanbertaAttention,
    intermediate: WangchanbertaIntermediate,
    output: WangchanbertaOutput
}

impl WangchanbertaLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attention = WangchanbertaAttention::new(cfg, vb.pp("attention"))?;
        let intermediate = WangchanbertaIntermediate::new(cfg, vb.pp("intermediate"))?;
        let output = WangchanbertaOutput::new(cfg, vb.pp("output"))?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        past_key_value: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        let self_attention_outputs = self.attention.forward(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
        )?;
        let attention_output = self_attention_outputs.0;
        let outputs = self_attention_outputs.1;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok((layer_output, outputs))
    }
}

struct WangchanbertaEncoder {
    layers: Vec<WangchanbertaLayer>,
}

impl WangchanbertaEncoder {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| WangchanbertaLayer::new(cfg, vb.pp(format!("layer.{}", i))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        past_key_value: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer_module in self.layers.iter() {
            let layer_outputs = layer_module.forward(
                &hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
            )?;
            hidden_states = layer_outputs.0;
        }
        Ok(hidden_states)
    }
}

pub struct WangchanbertaModel {
    encoder: WangchanbertaEncoder,
    embeddings: WangchanbertaEmbeddings
}

impl WangchanbertaModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let encoder = WangchanbertaEncoder::new(cfg, vb.pp("encoder"))?;
        let embeddings = WangchanbertaEmbeddings::load(vb.pp("embeddings"), cfg)?;
        Ok(Self { encoder, embeddings })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: &Tensor,
        past_key_value: Option<(&Tensor, &Tensor)>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let hidden_states = self.embeddings.forward(input_ids, token_type_ids)?;
        let attention_mask = prepare_4d_attention_mask(attention_mask, DType::F32, None)?
            .to_device(hidden_states.device())?;
        let hidden_states = self.encoder.forward(
            &hidden_states,
            &attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
        )?;
        Ok(hidden_states)
    }
}

fn prepare_4d_attention_mask(
    mask: &Tensor,
    dtype: DType,
    tgt_len: Option<usize>,
) -> Result<Tensor> {
    let bsz = mask.dim(0)?;
    let src_len = mask.dim(1)?;
    let tgt_len = tgt_len.unwrap_or(src_len);

    let expanded_mask = mask
        .unsqueeze(1)?
        .unsqueeze(2)?
        .expand((bsz, 1, tgt_len, src_len))?
        .to_dtype(dtype)?;

    let inverted_mask = (1.0 - expanded_mask)?;

    (inverted_mask * get_dtype_min_val(dtype))?.to_dtype(dtype)
}

fn get_dtype_min_val(dtype: DType) -> f64 {
    match dtype {
        DType::F32 => f32::MIN as f64,
        DType::F64 => f64::MIN,
        _ => panic!("Unsupported data type"),
    }
}