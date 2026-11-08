# Unpadded PCM timestamp receipts

Fixed source 7245acbf2a6df8c105a1fc58e82e8285c4ddf9ec; old source e6fd4c68dbf797a9f09e60ecce7c4693d0bb0009. Native tiny.en CPU on krick, use_gpu=false, threads=2. No GPU or timing claim.

RED: original JFK end12400 > available11000. Delayed fixture end12000 > available10000. Fixed suite PASS: window10/30 x delay0/500ms x words off/on, plus existing engine tests.

Comparison PASS: 232 non-time records match, including text, token counts, indices/finality, hex-float confidence/probability and speaker metadata. Both versions emit 'ask what you You can do for your country' for delay500/window10. The first phrase assertion failure is retained; the test accepts this observed variant only in that setting. No trailing fixture padding added.

Separate diagnostics record old/new timestamp output; the regression requires partial/final bounds, word containment/order, and consumed PCM duration. Existing raw confidence is preserved; engine structs have no separate raw timestamp field. Final CUDA RPC awaits both runtime reviews.

Source snapshots, generated build commands, raw logs and binary/model/fixture hashes included. Dependency archives reused from qualified pinned build.
