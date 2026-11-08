import collections,hashlib,json,os,pathlib,platform,re,signal,subprocess,sys,time
root=pathlib.Path('/Users/krickert/work/astra-metal-20261108');repo=root/'grpc-asr';out=root/'asr-timeline-metal'/'focused-metal'
out.mkdir(exist_ok=False)
expected=sys.argv[1]
def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
assert git('rev-parse','HEAD')==expected
assert not git('status','--porcelain','--untracked-files=no')
assert platform.machine()=='arm64'
files=git('ls-files').splitlines()
manifest={'source_commit':expected,'source_tree':git('rev-parse','HEAD^{tree}'),'architecture':platform.machine(),'macos':platform.mac_ver()[0],'source_sha256':{f:digest(repo/f) for f in files if f=='CMakeLists.txt' or f.startswith(('src/','cmake/','tests/','proto/','tools/','streaming/src/','streaming/CMakeLists.txt'))},'build_sha256':{},'model_sha256':{}}
for f in ['build-metal/grpc-asr-server','build-metal/asr-service-test','build-metal/video-demux-test','build-metal/observed-whisper.cpp','build-metal/_deps/whisper-src/src/whisper.cpp','build-metal/CMakeCache.txt']:
 manifest['build_sha256'][f]=digest(repo/f)
for f in ['models/asr/ggml-tiny.en.bin','models/asr/jfk.wav']:manifest['model_sha256'][f]=digest(root/f)
assert manifest['build_sha256']['build-metal/_deps/whisper-src/src/whisper.cpp']=='891168a838301bd2baa5e8b87422a854dafeeb7da914d40a2e4c6bf33ff448ed'
assert manifest['build_sha256']['build-metal/observed-whisper.cpp']=='5bb518beb0defd5098badee6c7c6e576f15e98a1ce7a8e233408f52ac51fccc6'
(out/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
env=os.environ.copy();env.update(GRPC_ASR_TEST_FIXTURE_FFMPEG=str(repo/'tests/macos_fixture_ffmpeg.py'),GRPC_ASR_TEST_MODEL=str(root/'models/asr/ggml-tiny.en.bin'),GRPC_ASR_TEST_SAMPLE=str(root/'models/asr/jfk.wav'),GRPC_ASR_SERVER_BINARY=str(repo/'build-metal/grpc-asr-server'),GRPC_ASR_TEST_REQUIRE_MODELS='1',GRPC_ASR_TEST_BACKEND='metal',GRPC_ASR_TEST_MEDIA_TIMELINE_ONLY='1',GRPC_ASR_TRACE_PLACEMENT='1',VECLIB_MAXIMUM_THREADS='2',OMP_NUM_THREADS='2')
start=time.monotonic()
with (out/'metal-rpc.log').open('w') as log:
 p=subprocess.Popen(['/usr/bin/time','-l','nice','-n','10',str(repo/'build-metal/asr-service-test')],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=p.wait(timeout=300)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  raise
elapsed=time.monotonic()-start
(out/'rpc.exitcode').write_text(str(code)+'\n')
text=(out/'metal-rpc.log').read_text()
print(text[-2500:]);assert code==0,'full Metal suite failed'
assert 'asr-service-test native timeline passed (loopback TCP RPC)' in text and 'video cases not run' not in text
stages=collections.Counter()
for stage,cpu,gpu,other in re.findall(r'placement stage=(\w+) cpu_ops=(\d+) gpu_ops=(\d+) other_ops=(\d+)',text):
 assert int(cpu)==0 and int(gpu)>0 and int(other)==0,(stage,cpu,gpu,other)
 stages[stage]+=1
assert set(stages)=={'conv','encoder','cross','decoder'}
timeline=[line for line in text.splitlines() if 'native_timeline_rpc PASS' in line]
assert len(timeline)==3,'all focused timeline RPC assertions must execute'
previous=json.loads((out.parent/'source-manifest.json').read_text())
assert previous['build_sha256']['build-metal/grpc-asr-server']==manifest['build_sha256']['build-metal/grpc-asr-server']
for f,h in previous['source_sha256'].items():
 if f.startswith(('src/','cmake/')):assert manifest['source_sha256'][f]==h,f
(out/'assertions.json').write_text(json.dumps({'source_commit':expected,'exit_code':code,'elapsed_seconds':elapsed,'stage_observations':dict(stages),'all_observed_graph_ops_gpu':True,'timeline_assertions':timeline,'cpu_comparison':'not repeated; previous source17b85d25071c6d395ae9cceb8fb062d6c5d42270'},indent=2)+'\n')
(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes())
print('PASS focused Metal timeline RPC; production binary unchanged',dict(stages),'elapsed',elapsed,flush=True)
