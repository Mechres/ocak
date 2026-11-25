use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateFamily {
    Zephyr,
    ChatMl,
    Llama2,
    Llama3,
    Raw,
}

pub fn detect_family(chat_template: Option<&str>) -> TemplateFamily {
    let Some(t) = chat_template else {
        return TemplateFamily::Raw;
    };
    if t.contains("<|im_start|>") {
        TemplateFamily::ChatMl
    } else if t.contains("<|start_header_id|>") {
        TemplateFamily::Llama3
    } else if t.contains("<|user|>") {
        TemplateFamily::Zephyr
    } else if t.contains("[INST]") {
        TemplateFamily::Llama2
    } else {
        TemplateFamily::Raw
    }
}

pub fn render(
    family: TemplateFamily,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    let mut out = String::new();
    match family {
        TemplateFamily::Zephyr => {
            for m in messages {
                out.push_str("<|");
                out.push_str(&m.role);
                out.push_str("|>\n");
                out.push_str(&m.content);
                out.push_str("</s>\n");
            }
            if add_generation_prompt {
                out.push_str("<|assistant|>\n");
            }
        }
        TemplateFamily::ChatMl => {
            for m in messages {
                out.push_str("<|im_start|>");
                out.push_str(&m.role);
                out.push('\n');
                out.push_str(&m.content);
                out.push_str("<|im_end|>\n");
            }
            if add_generation_prompt {
                out.push_str("<|im_start|>assistant\n");
                if !enable_thinking {
                    out.push_str("<think>\n\n</think>\n\n");
                }
            }
        }
        TemplateFamily::Llama3 => {
            for m in messages {
                out.push_str("<|start_header_id|>");
                out.push_str(&m.role);
                out.push_str("<|end_header_id|>\n\n");
                out.push_str(&m.content);
                out.push_str("<|eot_id|>");
            }
            if add_generation_prompt {
                out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
        }
        TemplateFamily::Llama2 => {
            let mut sys = String::new();
            let mut idx = 0;
            if let Some(first) = messages.first() {
                if first.role == "system" {
                    sys = first.content.clone();
                    idx = 1;
                }
            }
            let mut first_turn = true;
            for m in &messages[idx..] {
                if m.role == "user" {
                    out.push_str("[INST] ");
                    if first_turn && !sys.is_empty() {
                        out.push_str("<<SYS>>\n");
                        out.push_str(&sys);
                        out.push_str("\n<</SYS>>\n\n");
                    }
                    first_turn = false;
                    out.push_str(&m.content);
                    out.push_str(" [/INST]");
                } else if m.role == "assistant" {
                    out.push(' ');
                    out.push_str(&m.content);
                    out.push_str(" </s>");
                }
            }
        }
        TemplateFamily::Raw => {
            for m in messages {
                out.push_str(&m.content);
                out.push('\n');
            }
        }
    }
    out
}
