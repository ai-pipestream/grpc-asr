#include "engine/state_ownership.h"

#include <array>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Only the selected container's allocations fail; state allocation and the
// assertions remain functional. Rebinding also covers deque's map allocation.
template<class T> struct FailingAllocator {
    using value_type = T;
    bool* fail;
    explicit FailingAllocator(bool& flag) noexcept : fail(&flag) {}
    template<class U> FailingAllocator(const FailingAllocator<U>& other) noexcept : fail(other.fail) {}
    T* allocate(std::size_t count) {
        if (*fail) throw std::bad_alloc();
        return std::allocator<T>{}.allocate(count);
    }
    void deallocate(T* pointer, std::size_t count) noexcept {
        std::allocator<T>{}.deallocate(pointer, count);
    }
    template<class U> bool operator==(const FailingAllocator<U>& other) const noexcept {
        return fail == other.fail;
    }
};

struct Tracker {
    std::array<int, 4096> frees{};
    std::size_t created = 0;
};
struct State {
    Tracker& tracker;
    std::size_t id;
    ~State() { ++tracker.frees[id]; }
};
using Owner = std::unique_ptr<State>;
Owner make_state(Tracker& tracker) {
    require(tracker.created < tracker.frees.size(), "test capacity exceeded");
    return std::make_unique<State>(tracker, tracker.created++);
}
struct InitFailure {};

enum class Failure { initialization, free_list, owning_list };
void exercise(Failure failure) {
    Tracker tracker;
    bool fail_free = false, fail_owned = false;
    std::deque<State*, FailingAllocator<State*>> free{FailingAllocator<State*>(fail_free)};
    std::vector<State*, FailingAllocator<State*>> owned{FailingAllocator<State*>(fail_owned)};
    const auto publish = [&](Owner state, bool fail_init) {
        asr::engine::detail::initialize_owned_state(std::move(state), free, owned,
            [&](State* value) {
                require(value != nullptr, "state must exist during initialization");
                require(tracker.frees[value->id] == 0, "state freed before initialization");
                if (fail_init) throw InitFailure{};
            });
    };
    // Leave an existing state in the pool so rollback must preserve it.
    publish(make_state(tracker), false);
    fail_free = failure == Failure::free_list;
    fail_owned = failure == Failure::owning_list;
    bool injected = false;
    while (!injected) {
        const auto before = owned.size();
        const auto id = tracker.created;
        try {
            publish(make_state(tracker), failure == Failure::initialization);
            // Deque/vector may have spare capacity. Continue until the real
            // allocator is reached rather than assuming an implementation size.
            require(free.size() == owned.size(), "successful publication inconsistent");
        } catch (const InitFailure&) {
            require(failure == Failure::initialization, "unexpected init failure");
            injected = true;
        } catch (const std::bad_alloc&) {
            require(failure != Failure::initialization, "unexpected allocation failure");
            injected = true;
        }
        if (injected) {
            require(free.size() == before && owned.size() == before, "failed publication not rolled back");
            require(tracker.frees[id] == 1, "failed state must be freed exactly once");
        }
    }
    fail_free = fail_owned = false;
    const auto before_retry = owned.size();
    publish(make_state(tracker), false);
    require(owned.size() == before_retry + 1 && free.size() == owned.size(), "retry failed");
    for (std::size_t i = 0; i < owned.size(); ++i) {
        require(owned[i] == free[i], "free list contains stale or reordered state");
        require(tracker.frees[owned[i]->id] == 0, "published state freed early");
        delete owned[i];
    }
    free.clear();
    owned.clear();
    for (std::size_t i = 0; i < tracker.created; ++i)
        require(tracker.frees[i] == 1, "state leaked or freed more than once");
}
}  // namespace

int main() {
    try {
        exercise(Failure::initialization);
        exercise(Failure::free_list);
        exercise(Failure::owning_list);
        std::cout << "PASS: init, deque insertion, vector insertion failures; rollback/retry/free exactly once\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
