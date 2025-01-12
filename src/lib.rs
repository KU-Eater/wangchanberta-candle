mod model;
mod tokenizer;
pub mod wangchanberta;

// TODO: Figure how to keep the embeddings generation consistent on CPU

#[cfg(test)]
mod tests {
    use crate::wangchanberta::Wangchanberta;

    #[test]
    fn vectors_generation() {
        let text = "สวัสดีโลก!";
        let bert = Wangchanberta::new(
            Some("./share/updated_pytorch_model.bin".to_string()),
            Some("./share/config.json".to_string()),
            Some("./share/sentencepiece.bpe.model".to_string())
        ).unwrap();
        let vectors = bert.generate_embedding(text);
        match vectors {
            Ok(v) => {
                println!("{:?} {:?}", v, v.len());
                assert_eq!(v.len(), v.len());
            },
            Err(e) => panic!("{:?}", e)
        }
    }
}