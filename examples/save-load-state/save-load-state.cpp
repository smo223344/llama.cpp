#include "common.h"
#include "llama.h"
#include "build-info.h"

#include <vector>
#include <cstdio>
#include <chrono>

gpt_params params;
llama_context_params lparams;

struct state_stack_entry
{
    uint8_t * state_mem;
    size_t state_size;
    int n_past;
};

std::vector<state_stack_entry> llama_state_stack;

void llama_push_state(llama_context * ctx, int n_past)
{
    const size_t state_size = llama_get_state_size(ctx);
    uint8_t * state_mem = new uint8_t[state_size];
    llama_copy_state_data(ctx, state_mem);
    llama_state_stack.push_back({state_mem, state_size, n_past});
}

void llama_pop_state(llama_context * ctx, int& n_past)
{
    state_stack_entry stack_entry = llama_state_stack.back();
    
//    llama_free(ctx);
//    ctx = llama_init_from_file(params.model.c_str(), lparams);

    llama_set_state_data(ctx, stack_entry.state_mem);
    n_past = stack_entry.n_past;
    llama_state_stack.pop_back();
    delete[] stack_entry.state_mem;
}

struct beam_result
{
    std::string text;
    float prob_sum;
    int n_tokens;
};

std::vector<beam_result> beam_results;

void recurse_beam_search(llama_context * ctx, int n_past, int beam_width, int prob_sum, int current_depth, int max_depth, std::string text)
{
    auto logits = llama_get_logits(ctx);
    auto n_vocab = llama_n_vocab(ctx);
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }
    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    llama_sample_softmax(ctx, &candidates_p);

    int i = 0;
    for (; i < beam_width; i++)
    {
        auto next_token = candidates_p.data[i].id;
        auto next_token_str = llama_token_to_str(ctx, next_token);
        if (current_depth == max_depth || beam_width == 1)
        {
            beam_results.push_back({text + next_token_str, prob_sum + candidates_p.data[i].p, current_depth});
            printf(" (%0.2f) %s\n", prob_sum + candidates_p.data[i].p, (text + next_token_str).c_str());
        } else
        {
            llama_push_state(ctx, n_past);
            llama_eval(ctx, &next_token, 1, n_past, 1);
            recurse_beam_search(ctx, n_past + 1, beam_width - 1, prob_sum + candidates_p.data[i].p, current_depth + 1, max_depth, text + next_token_str);
            llama_pop_state(ctx, n_past);
        }
    }
}

int main(int argc, char ** argv) {
//    gpt_params params;
    params.model = "models/7B/ggml-model-q5_1.bin";
    params.seed = 42;
    params.n_threads = 8;
    params.repeat_last_n = 64;
    params.prompt = "Once upon a time";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    fprintf(stderr, "%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    lparams = llama_context_default_params();

    lparams.n_ctx     = params.n_ctx;
    lparams.n_parts   = params.n_parts;
    lparams.seed      = params.seed;
    lparams.f16_kv    = params.memory_f16;
    lparams.use_mmap  = params.use_mmap;
    lparams.use_mlock = params.use_mlock;

    auto n_past = 0;
    auto last_n_tokens_data = std::vector<llama_token>(params.repeat_last_n, 0);

    // init
    auto ctx = llama_init_from_file(params.model.c_str(), lparams);
    auto tokens = std::vector<llama_token>(params.n_ctx);
    auto n_prompt_tokens = llama_tokenize(ctx, params.prompt.c_str(), tokens.data(), tokens.size(), true);

    if (n_prompt_tokens < 1) {
        fprintf(stderr, "%s : failed to tokenize prompt\n", __func__);
        return 1;
    }

    // evaluate prompt
    llama_eval(ctx, tokens.data(), n_prompt_tokens, n_past, params.n_threads);

    last_n_tokens_data.insert(last_n_tokens_data.end(), tokens.data(), tokens.data() + n_prompt_tokens);
    n_past += n_prompt_tokens;

    const size_t state_size = llama_get_state_size(ctx);
    uint8_t * state_mem = new uint8_t[state_size];

    // Save state (rng, logits, embedding and kv_cache) to file
//    llama_copy_state_data(ctx, state_mem); // could also copy directly to memory mapped file

    // save state (last tokens)
    const auto last_n_tokens_data_saved = std::vector<llama_token>(last_n_tokens_data);
    const auto n_past_saved = n_past; 

    // run
    printf("\n%s", params.prompt.c_str());

    recurse_beam_search(ctx, n_past, 4, 0, 0, 6, params.prompt);
/*
    for (auto i = 0; i < params.n_predict; i++) {
        auto logits = llama_get_logits(ctx);
        auto n_vocab = llama_n_vocab(ctx);
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
        auto next_token = llama_sample_token(ctx, &candidates_p);
        auto next_token_str = llama_token_to_str(ctx, next_token);
        last_n_tokens_data.push_back(next_token);

        printf("%s", next_token_str);
        if (llama_eval(ctx, &next_token, 1, n_past, params.n_threads)) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            return 1;
        }
        n_past += 1;
    }

    printf("\n\n");
*/
    // free old model
//    llama_free(ctx);

    // load new model
//    auto ctx2 = llama_init_from_file(params.model.c_str(), lparams);
/*
    // Load state (rng, logits, embedding and kv_cache) from file
    {
        llama_set_state_data(ctx, state_mem);  // could also read directly from memory mapped file
    }

    // restore state (last tokens)
    last_n_tokens_data = last_n_tokens_data_saved;
    n_past = n_past_saved;

    // second run
    printf("\n%s", params.prompt.c_str());

    for (auto i = 0; i < params.n_predict; i++) {
        auto logits = llama_get_logits(ctx);
        auto n_vocab = llama_n_vocab(ctx);
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
        auto next_token = llama_sample_token(ctx, &candidates_p);
        auto next_token_str = llama_token_to_str(ctx, next_token);
        last_n_tokens_data.push_back(next_token);

        printf("%s", next_token_str);
        if (llama_eval(ctx, &next_token, 1, n_past, params.n_threads)) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            return 1;
        }
        n_past += 1;
    }

    printf("\n\n");
*/
    return 0;
}
