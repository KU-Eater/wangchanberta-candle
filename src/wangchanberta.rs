use crate::model::{WangchanbertaModel, Config};
use crate::tokenizer::Tokenizer;

use std::path::PathBuf;
use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

#[allow(dead_code)]
pub struct Wangchanberta {
    model: WangchanbertaModel,
    config: Config,
    tokenizer: Tokenizer,
    device: Device,
}

impl Wangchanberta {
    pub fn new(
        weights_path: Option<String>,
        config_path: Option<String>,
        tokenizer_path: Option<String>
        //cuda: bool
    ) -> Result<Self> {

        // Set device
        // TODO: Set to GPU when cuda is true
        let device = Device::Cpu;

        // Load config

        let config_path = match config_path {
            Some(path) => PathBuf::from(path.to_string()),
            None => PathBuf::from("./config.json")
        };
        
        let read_config = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&read_config)?;

        // Load tokenizer

        let tokenizer_path = match tokenizer_path {
            Some(path) => PathBuf::from(path.to_string()),
            None => PathBuf::from("./sentencepiece.bpe.model")
        };

        let tokenizer: Tokenizer = Tokenizer::new(tokenizer_path, device.clone())?;

        // Load model
        let weights_path = match weights_path {
            Some(path) => PathBuf::from(path.to_string()),
            None => PathBuf::from("./pytorch_model.bin")
        };

        let vb = if weights_path.ends_with("model.safetensors") {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F16, &device).unwrap()
            }
        } else {
            VarBuilder::from_pth(&weights_path, candle_core::DType::F16, &device).unwrap()
        };
        let model: WangchanbertaModel = WangchanbertaModel::new(&config, vb)?;

        Ok(Self {
            model: model,
            config: config,
            tokenizer: tokenizer,
            device: device
        })
    }

    pub fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(text).map_err(E::msg)?;

        let input_ids: Vec<i64> = encoding.get_ids().iter()
            .map(|&id| id as i64)
            .collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter()
            .map(|&mask| mask as i64)
            .collect();

        let input_ids = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;
        let attention_mask = Tensor::new(&attention_mask[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::zeros(input_ids.dims(), input_ids.dtype(), &self.device)?;

        let output = self.model.forward(
            &input_ids, &attention_mask, &token_type_ids, None, None, None
        )?.to_dtype(candle_core::DType::F32)?;

        let cls_embedding = output.get(0)?.get(0)?.to_vec1()?;

        Ok(cls_embedding)
    }

    pub fn batch_generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let embedding = self.generate_embedding(text)?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }
}