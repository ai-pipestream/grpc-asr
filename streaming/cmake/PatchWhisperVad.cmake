# Pinned upstream 7246b731 returns true after a failed VAD graph compute.
# Propagate that error instead of treating the untouched probability as silence.
set(vad_source "${whisper_SOURCE_DIR}/src/whisper.cpp")
file(READ "${vad_source}" vad_text)
set(vad_old [=[WHISPER_LOG_ERROR("%s: failed to compute VAD graph\n", __func__);
            break;]=])
set(vad_new [=[WHISPER_LOG_ERROR("%s: failed to compute VAD graph\n", __func__);
            ggml_backend_sched_reset(sched);
            return false; // pipestream: propagate VAD compute failure]=])
string(FIND "${vad_text}" "${vad_old}" vad_position)
if(NOT vad_position EQUAL -1)
  string(REPLACE "${vad_old}" "${vad_new}" vad_text "${vad_text}")
  file(WRITE "${vad_source}" "${vad_text}")
else()
  string(FIND "${vad_text}" "${vad_new}" vad_position)
  if(vad_position EQUAL -1)
    message(FATAL_ERROR "Pinned Whisper VAD error-handling patch no longer matches")
  endif()
endif()
