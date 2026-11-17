#pragma once

#include <memory>

namespace asr::engine::detail {

// The caller's owning list frees published states before its context. Until
// both insertions succeed, this guard owns the state, including during init.
template<class State, class Deleter, class FreeList, class OwningList, class Initialize>
void initialize_owned_state(std::unique_ptr<State, Deleter> state,
                            FreeList& free_states, OwningList& all_states,
                            Initialize&& initialize) {
    initialize(state.get());
    free_states.push_back(state.get());
    try {
        all_states.push_back(state.get());
    } catch (...) {
        // Do not leave a dangling pointer if the second insertion allocates.
        free_states.pop_back();
        throw;
    }
    (void) state.release();
}

}  // namespace asr::engine::detail
