#include "../common.h"
#include "build-info.h"

#include <vector>
#include <cstdio>
#include <chrono>
#include <queue>
#include <string>
#include <fstream>
#include <algorithm>

gpt_params params;
llama_context_params lparams;

struct state_stack_entry
{
    uint8_t * state_mem;
    size_t state_size;
    int n_past;
    std::string text;
    int depth;
    float prob_sum;
};

std::vector<state_stack_entry> llama_state_stack;
std::queue<state_stack_entry> llama_state_queue;

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
    llama_set_state_data(ctx, stack_entry.state_mem);
    n_past = stack_entry.n_past;
    llama_state_stack.pop_back();
    delete[] stack_entry.state_mem;
}

void llama_queue_state(llama_context * ctx, int n_past, const std::string& text, int depth, float prob_sum)
{
    const size_t state_size = llama_get_state_size(ctx);
    uint8_t * state_mem = new uint8_t[state_size];
    llama_copy_state_data(ctx, state_mem);
    llama_state_queue.push({state_mem, state_size, n_past, text, depth, prob_sum});
}

void llama_dequeue_state(llama_context * ctx, int& n_past, std::string& text, int& depth, float& prob_sum)
{
    state_stack_entry stack_entry = llama_state_queue.front();
    llama_set_state_data(ctx, stack_entry.state_mem);
    n_past = stack_entry.n_past;
    text = stack_entry.text;
    depth = stack_entry.depth;
    prob_sum = stack_entry.prob_sum;
    llama_state_queue.pop();
    delete[] stack_entry.state_mem;
}

// trim the queue by removing the bottom entries based on prob_sum if there are more than max_width entries
void llama_queue_trim(int max_width)
{
    if (llama_state_queue.size() > max_width)
    {
        std::vector<state_stack_entry> queue_entries;
        queue_entries.reserve(llama_state_queue.size());
        while (!llama_state_queue.empty())
        {
            queue_entries.push_back(llama_state_queue.front());
            llama_state_queue.pop();
        }
        std::sort(queue_entries.begin(), queue_entries.end(), [](const state_stack_entry& a, const state_stack_entry& b) {
            return a.prob_sum > b.prob_sum;
        });
        for (int i = 0; i < max_width; i++)
        {
            llama_state_queue.push(queue_entries[i]);
        }
    }
}

struct beam_result
{
    std::string text;
    float prob_sum;
    int n_tokens;
};

std::vector<beam_result> beam_results;

void recurse_beam_search(llama_context * ctx, int n_past, int beam_width, float prob_sum, int current_depth, int max_depth, std::string text)
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
    int max_width = 2;
    for (; i < beam_width; i++)
    {
        if (candidates_p.data[i].p < 0.1)
        {
            max_width = i > max_width ? i : max_width;
            break;
        }
        auto next_token = candidates_p.data[i].id;
        auto next_token_str = llama_token_to_str(ctx, next_token);
        printf(" (%.2f) %s\n", prob_sum + candidates_p.data[i].p, (text + next_token_str).c_str());
    }
    for (i = 0; i < max_width; i++)
    {
        auto next_token = candidates_p.data[i].id;
        auto next_token_str = llama_token_to_str(ctx, next_token);
        if (current_depth == max_depth || beam_width == 1)
        {
            beam_results.push_back({text + next_token_str, prob_sum + candidates_p.data[i].p, current_depth});
//            printf(" (%0.2f) %s\n", prob_sum + candidates_p.data[i].p, (text + next_token_str).c_str());
        } else
        {
            llama_push_state(ctx, n_past);
            llama_eval(ctx, &next_token, 1, n_past, 1);
            recurse_beam_search(ctx, n_past + 1, beam_width, prob_sum + candidates_p.data[i].p, current_depth + 1, max_depth, text + next_token_str);
            llama_pop_state(ctx, n_past);
        }
    }
}

void iterate_beam_search(llama_context * ctx, int n_past, int beam_width, int max_depth, std::string text, float p_threshold)
{
    int i = 0;
    int depth = 1;
    float prob_sum = 0.0;
    llama_queue_state(ctx, n_past, text, depth, prob_sum);

    while (!llama_state_queue.empty())
    {
        if ((depth > 10) & (depth % 3 == 0))
        {
            llama_queue_trim(beam_width);
        }
        llama_dequeue_state(ctx, n_past, text, depth, prob_sum);

        auto logits = llama_get_logits(ctx);
        auto n_vocab = llama_n_vocab(ctx);
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
        llama_sample_softmax(ctx, &candidates_p);

        bool is_leaf = true;
        for (i = 0; i < beam_width; i++)
        {
            if (candidates_p.data[i].p < p_threshold)
            {
                break;
            }
            is_leaf = false;

            auto next_token = candidates_p.data[i].id;
            auto next_token_str = llama_token_to_str(ctx, next_token);
            printf(" (%.2f) %s\n", prob_sum + candidates_p.data[i].p, (text + next_token_str).c_str());
            if (depth == max_depth || is_leaf)
            {
                beam_results.push_back({text + next_token_str, prob_sum + candidates_p.data[i].p, depth});
            } else
            {
                llama_push_state(ctx, n_past);
                llama_eval(ctx, &next_token, 1, n_past, 12);
                llama_queue_state(ctx, n_past + 1, text + next_token_str, depth + 1, prob_sum + candidates_p.data[i].p);
                llama_pop_state(ctx, n_past);
            }
        }
        if (is_leaf)
        {
            beam_results.push_back({text, prob_sum, depth});
        }
    }
}

std::string get_best_beam_result()
{
    std::string best_text;
    float best_prob_sum = 0.0;
    int best_n_tokens = 0;
    for (auto& beam_result : beam_results)
    {
        if (beam_result.prob_sum > best_prob_sum)
        {
            best_text = beam_result.text;
            best_prob_sum = beam_result.prob_sum;
            best_n_tokens = beam_result.n_tokens;
        }
    }
    return best_text;
}

void sort_beam_results()
{
    std::sort(beam_results.begin(), beam_results.end(), [](const beam_result& a, const beam_result& b) {
        return a.prob_sum > b.prob_sum;
    });
}

void print_beam_results()
{
    for (auto& beam_result : beam_results)
    {
        printf("(%.2f) =====\n%s\n", beam_result.prob_sum, beam_result.text.c_str());
    }
}

std::string load_prompt_from_file(std::string prompt_filename)
{
    std::string prompt;
    std::ifstream prompt_file(prompt_filename);
    if (prompt_file.is_open())
    {
        std::string line;
        while (std::getline(prompt_file, line))
        {
            prompt += line + "\n";
        }

        prompt_file.close();
    }
    return prompt;
}

int main(int argc, char ** argv) {
    params.model = "models/7B/ggml-model-q5_1.bin";
    params.seed = 42;
    params.n_threads = 8;
    params.repeat_last_n = 64;
//    params.prompt = "Once upon a time, there was a doctor named House. House was addicted to";
    params.prompt = load_prompt_from_file("and_prompt.txt");

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

    const auto last_n_tokens_data_saved = std::vector<llama_token>(last_n_tokens_data);
    const auto n_past_saved = n_past; 

    printf("\n%s", params.prompt.c_str());
//    recurse_beam_search(ctx, n_past, 2, 0, 0, 60, params.prompt);
//    iterate_beam_search(ctx, n_past, 8, 60, params.prompt, 0.1);
    iterate_beam_search(ctx, n_past, 8, 200, "", 0.1);

    sort_beam_results();
    print_beam_results();
    return 0;
}
