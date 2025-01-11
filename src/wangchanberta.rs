use crate::model::{WangchanbertaModel, Config};

use std::path::PathBuf;
use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::{PaddingParams, Tokenizer};

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
        tokenizer_path: Option<String>,
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
            None => PathBuf::from("./tokenizer.json")
        };

        let mut tokenizer: Tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            pad_id: config.pad_token_id,
            ..Default::default()
        })).with_truncation(None).map_err(E::msg)?;

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

    // TODO: Create a function to translate input to vectors
}