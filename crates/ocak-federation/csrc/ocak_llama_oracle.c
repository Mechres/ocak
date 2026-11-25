#include "llama.h"
#include <string.h>

/* Runs `tokens[0..n_tokens]` on CPU reference via libllama.so and copies last-token logits. */
int32_t ocak_oracle_eval(
    const char * model_path,
    const int32_t * tokens,
    int32_t n_tokens,
    float * out_logits,
    int32_t max_vocab
) {
    if (!model_path || !tokens || n_tokens <= 0 || !out_logits || max_vocab <= 0) {
        return -1;
    }
    llama_backend_init();
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        return -2;
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t)(n_tokens + 64);
    cparams.n_batch = (uint32_t)n_tokens;
    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return -3;
    }
    struct llama_batch batch = llama_batch_get_one((llama_token *)tokens, n_tokens);
    int32_t rc = llama_decode(ctx, batch);
    if (rc != 0) {
        llama_free(ctx);
        llama_model_free(model);
        return -4;
    }
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) {
        llama_free(ctx);
        llama_model_free(model);
        return -5;
    }
    int32_t copy_n = n_vocab < max_vocab ? n_vocab : max_vocab;
    memcpy(out_logits, logits, (size_t)copy_n * sizeof(float));
    llama_free(ctx);
    llama_model_free(model);
    return n_vocab;
}
