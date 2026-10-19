//! トークン数の計数 (reasoning_tokens、count_tokens、文脈長の事前検査)。
//! ds41rt と同じ `tokenizer.json` を読む。読めない場合は文字数からの概算にする。

use std::sync::Arc;

use deepseek_recipe_core::conversation::Conversation;
use deepseek_recipe_core::multimodal::IMAGE_SPECIAL_TOKEN;
use deepseek_recipe_encoding::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::PromptEncoding;
use tokenizers::Tokenizer;

pub struct Counter {
    tok: Option<Arc<Tokenizer>>,
    image_estimate: usize,
}

impl Counter {
    pub fn load(path: &std::path::Path, image_estimate: usize) -> Self {
        let tok = match Tokenizer::from_file(path) {
            Ok(t) => Some(Arc::new(t)),
            Err(e) => {
                eprintln!("ds41gw: tokenizer {} を読めない ({e})。文字数からの概算にする", path.display());
                None
            }
        };
        Self { tok, image_estimate }
    }

    pub fn none(image_estimate: usize) -> Self {
        Self { tok: None, image_estimate }
    }

    pub fn has_tokenizer(&self) -> bool {
        self.tok.is_some()
    }

    /// 文字列のトークン数 (特殊トークンを含めて数える)。
    pub fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        match &self.tok {
            Some(t) => heavy(text.len(), || t.encode(text, false).map(|e| e.get_ids().len()).unwrap_or_else(|_| estimate(text))),
            None => estimate(text),
        }
    }

    /// 会話を ds41rt と同じ形で描画して数える。画像は 1 枚 image_estimate で概算する。
    pub fn count_conversation(&self, conv: &Conversation) -> usize {
        let rendered = heavy(64 << 10, || DeepseekV41Encoding::new().render_conversation(conv));
        let images = rendered.image_sources.len();
        let n = self.count(&rendered.prompt);
        n + images * self.image_estimate.saturating_sub(1)
    }
}

/// 概算: ASCII は 4 文字 1 トークン、それ以外は 1 文字 1 トークン。
fn estimate(text: &str) -> usize {
    let mut ascii = 0usize;
    let mut other = 0usize;
    for c in text.chars() {
        if c.is_ascii() {
            ascii += 1;
        } else {
            other += 1;
        }
    }
    let n = ascii / 4 + other;
    // 画像プレースホルダは 1 トークン
    n.saturating_sub(text.matches(IMAGE_SPECIAL_TOKEN).count() * 4) + 1
}

/// 大きな文字列のトークン化は数十〜数百 ms かかる。tokio の worker を塞ぐと、同じ worker の
/// 他のストリームの差分が止まるので、multi-thread runtime では block_in_place で逃がす。
fn heavy<T>(size: usize, f: impl FnOnce() -> T) -> T {
    if size >= 64 << 10 {
        if let Ok(h) = tokio::runtime::Handle::try_current() {
            if h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread {
                return tokio::task::block_in_place(f);
            }
        }
    }
    f()
}
