use anyhow::Result;
use candle_core::Device;
use sentencepiece::SentencePieceProcessor as SPP;
use tokenizers::Encoding;
use std::{collections::HashMap, path::Path};

#[allow(dead_code)]
pub struct Tokenizer {
    sp: SPP,
    device: Device
}

impl Tokenizer {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        device: Device
    ) -> Result<Self> {
        let sp = SPP::open(model_path.as_ref()).unwrap();
        Ok(Self {sp, device})
    }

    pub fn encode(&self, text: &str) -> Result<Encoding> {
        let pieces = self.sp.encode(text)?;

        let tokens: Vec<String> = pieces.iter()
            .map(|p| p.piece.clone())
            .collect();
        let ids: Vec<u32> = pieces.iter()
            .map(|p| p.id as u32)
            .collect();

        let attention_mask = vec![1u32; ids.len()];
        let type_ids = vec![0u32; ids.len()];
        let special_tokens_mask = vec![0u32; ids.len()];

        let offsets: Vec<(usize, usize)> = tokens.iter()
            .scan(0, |state, token| {
                let start = *state;
                *state += token.len();
                Some((start, *state))
            }).collect();
        
        Ok(Encoding::new(
            ids,
            type_ids,
            tokens,
            vec![],
            offsets,
            special_tokens_mask,
            attention_mask,
            vec![],
            HashMap::new()
        ))
    }
}