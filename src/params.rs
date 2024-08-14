use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers  #[128, ]
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers  #[128, 128]
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers  #[64, 128]
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers  #[64, 128]
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers  #[128, 128]
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers  #[128, ]
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers  #[384, 128]
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers  #[384, 128]
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers  #[128, 384]
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )  #[128, ]
    pub lm_head: Tensor<T>,   // (vocab_size, dim)  #[2048, 128]
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| {
            match safetensor.tensor(name) {
                Ok(tv) => Tensor::<f32>::new(
                    (0..(tv.data().len()/4))
                        .into_iter()
                        .map(|f| f32::from_le_bytes(tv.data()[(f * 4)..(f * 4 + 4)].try_into().unwrap()))
                        .collect(), 
                    &Vec::from(tv.shape())
                ),
                Err(e) => panic!("from safetensors error: {}", e),
            }
        };
        
        let layers_range = 0..config.num_hidden_layers;

        Self {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: layers_range.clone().map(|f| get_tensor(format!("model.layers.{f}.input_layernorm.weight").as_str())).collect(),
            wq: layers_range.clone().map(|f| get_tensor(format!("model.layers.{f}.self_attn.q_proj.weight").as_str())).collect(),
            wk: layers_range.clone().map(|f| get_tensor(format!("model.layers.{f}.self_attn.k_proj.weight").as_str())).collect(),
            wv: layers_range.clone().map(|f| get_tensor(format!("model.layers.{f}.self_attn.v_proj.weight").as_str())).collect(),
            wo: layers_range.clone().map(|f| get_tensor(format!("model.layers.{f}.self_attn.o_proj.weight").as_str())).collect(),
            rms_ffn_w: layers_range.clone().map(|f| get_tensor(format!("model.layers.{f}.post_attention_layernorm.weight").as_str())).collect(),
            w_up: layers_range.clone().map(|f| get_tensor(format!("model.layers.{f}.mlp.up_proj.weight").as_str())).collect(),
            w_gate: layers_range.clone().map(|f| get_tensor(format!("model.layers.{f}.mlp.gate_proj.weight").as_str())).collect(),
            w_down: layers_range.clone().map(|f| get_tensor(format!("model.layers.{f}.mlp.down_proj.weight").as_str())).collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
