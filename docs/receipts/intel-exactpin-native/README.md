# Exact-pin native smoke, RPC qualification pending

Host krick-1, Intel Arc Pro B70, Level Zero 1.15.39122, oneAPI compiler
2025.3.3. Native Whisper d1be6fde11ac6e0407606b4e42fe72d34add8037;
archive SHA256 645445da884df6dbd52b30f82d7be7c01d1767c938714e477e786b88d529e9f4.
Generated source SHA256 5bb518beb0defd5098badee6c7c6e576f15e98a1ce7a8e233408f52ac51fccc6.
Executable SHA256 e55daf531508d3f91398cc2ee70f339668dff72179b9590dd85162562b3d624d.
Model tiny.en SHA256 921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f.

Both modes use the same native executable, JFK fixture, model, two threads,
greedy decoding, and seven real transcriptions. Text and ordered timestamps
are asserted. The cancellation flag is set before inference: this is only
pre-cancel smoke, followed by successful inference; not in-flight RPC proof.
All four graph stage traces pass scripts/check-placement.sh: SYCL has no CPU
or other execution; explicit CPU has no GPU or other execution.

Cold run 0: SYCL 6713.926 ms, CPU 483.976 ms. Warm runs 1..5 median:
SYCL 176.470 ms, CPU 437.285 ms (2.48x). Cold time includes initialization/JIT
and is excluded. Shared host compilation/training was active, and these
sequential native measurements are not a clean-host throughput benchmark.
The full RPC test remains mandatory. Historical wrong-cache receipts are
preserved separately and do not support these exact-pin results.
