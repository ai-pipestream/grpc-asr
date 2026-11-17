#!/bin/sh
# Usage: check-placement.sh cpu|sycl TRACE_LOG
# This checks scheduler execution receipts, never requested configuration.
set -eu
case "${1:-}" in cpu|sycl) ;; *) echo 'expected cpu or sycl and trace log' >&2; exit 64;; esac
[ "$#" -eq 2 ] || exit 64
awk -v requested="$1" '
/grpc-asr placement stage=/ {
  stage=""; cpu=-1; gpu=-1; other=-1
  for(i=1;i<=NF;i++) {
    split($i,a,"=")
    if(a[1]=="stage") stage=a[2]
    if(a[1]=="cpu_ops") cpu=a[2]+0
    if(a[1]=="gpu_ops") gpu=a[2]+0
    if(a[1]=="other_ops") other=a[2]+0
  }
  if(stage!="conv" && stage!="encoder" && stage!="cross" && stage!="decoder") next
  seen[stage]++
  if(other!=0 || (requested=="sycl" && (cpu!=0 || gpu<=0)) ||
     (requested=="cpu" && (gpu!=0 || cpu<=0))) {
    print "FAIL mixed or missing execution: " $0 > "/dev/stderr"; failed=1
  }
}
END {
  split("conv encoder cross decoder", required, " ")
  for(i=1;i<=4;i++) if(!seen[required[i]]) {
    print "FAIL missing stage: " required[i] > "/dev/stderr"; failed=1
  }
  if(failed) exit 1
  print "PASS actual scheduler placement for all Whisper stages: " requested
}' "$2"
