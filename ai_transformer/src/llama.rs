use llm_chain::executor;
use llm_chain::{parameters, prompt};
use llm_chain::options::*;
use llm_chain::options;
use anyhow::Result;

pub async fn ask_llama(prompt: &str) -> Result<String> {
    let opts = options!(
        Model: ModelRef::from_path("models/"),
        ModelType: "llama",
        MaxContextSize: 512_usize,
        NThreads: 4_usize,
        MaxTokens: 0_usize,
        TopK: 40_i32,
        TopP: 0.95,
        TfsZ: 1.0,
        TypicalP: 1.0,
        Temperature: 0.8,
        RepeatPenalty: 1.1,
        RepeatPenaltyLastN: 64_usize,
        FrequencyPenalty: 0.0,
        PresencePenalty: 0.0,
        Mirostat: 0_i32,
        MirostatTau: 5.0,
        MirostatEta: 0.1,
        PenalizeNl: true,
        StopSequence: vec!["\n".to_string()]
    );
    let exec = executor!(llama, opts)?;
    let res = prompt!(prompt)
        .run(
            &parameters!(),
            &exec,
        )
        .await?;
    let output = res.to_immediate()
        .await?
        .primary_textual_output()
        .expect("empty response from model");
    Ok(output)
}