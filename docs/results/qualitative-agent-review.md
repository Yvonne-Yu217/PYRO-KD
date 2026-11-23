# Development qualitative review by an agent

This is a targeted inspection of existing **development** artifacts, not human annotation, a blinded study, or an error-rate estimate. No heldout correctness was read and no model/threshold was changed. Cases were selected to expose traceable mechanisms rather than to represent the dataset. Machine VAD/Whisper intervals are not human speech labels. Only case4 carries a visual-grounding judgment, based on opening its actual four model-input frames. Raw predictions, speech, and frames remain unchanged.

All cases use the ten-clip development experiment at `/scratch/yy5919/pyro-resume-results/experiments/reframed_local_dev_pilot_v1`. P1 is the original128-token development condition. Reference excerpts below come from the US AD track in validation data; they are context, not newly invented annotations.

## 1. Mid-speech cue moved into a gap

The stored speech-cue candidate is at40.0s, inside machine-VAD interval[39.0,41.3]. It is snapped to41.55s: a1.55s lag, leaving the0.25s boundary margin. The model emits “Man in suit listens.” (4words, limit5). This demonstrates the event/placement distinction, not an independent confirmation that the VAD boundary is perceptually correct. The selected four-frame window[40.5,42.5] still straddles the VAD boundary: narration placement and observed visual time are separate quantities. The nearby reference starts41.29s,0.26s before placement; temporal agreement alone does not prove the generated action is correct.

Run ID: `ad-A_cue-P1-qwen-tl1-80779200`.

### `v2gCnXXTSp6W_A_cue_001`

Placement: 41.55s; generated text:

> Man in suit listens.

Word count / limit: 4/5; generation error: `None`.

Reference context [41.29,44.42]s:

> The funny man wears a mustache and stares at the goateed man.

Record hashes (SHA256 of UTF-8 compact, sorted-key JSON): moment `9ba444ecd9aa1bd04cbf4c607ed1501dce8f937ac5733f7490befd2f53a2fcb2`; generation `65becc879e0556d38a074a7eca7aeea87db31fe8d6b726efe279016984838b69`; reference `0fc46830a1fccb03b748f72cb7bdd2ed0824d57ef74ab51e2df1a67792abd38a`.

| Frame time (s) | Original frame path | File SHA256 |
|---:|---|---|
|40.5|`/scratch/yy5919/pyro-kd-derived/cache/reframed/v2gCnXXTSp6W/frames/000082.jpg`|`d6c08d4f6954d13f65936facd66eca41c6577b8bae84df22c7123346dec18533`|
|41.0|`/scratch/yy5919/pyro-kd-derived/cache/reframed/v2gCnXXTSp6W/frames/000083.jpg`|`5115a908df2c72be05d6ca88bd5cb88d5b555bd8c32d976ad1b822e7a434a656`|
|42.0|`/scratch/yy5919/pyro-kd-derived/cache/reframed/v2gCnXXTSp6W/frames/000085.jpg`|`879bc129c5b435e7be88a743c61f7130fcde2a3a647e02f4b704a81c5f673164`|
|42.5|`/scratch/yy5919/pyro-kd-derived/cache/reframed/v2gCnXXTSp6W/frames/000086.jpg`|`bec5fed071a7aabd92d847556a7a74fc0a0687bfa59c1af45b941aedb75560ff`|

Actual rendered overlap row:

```json
{
  "duration": 1.41,
  "end": 42.959999999999994,
  "fits_speech_free_video": true,
  "id": "v2gCnXXTSp6W_A_cue_001",
  "out_of_video_seconds": 0.0,
  "speech_free_in_video_seconds": 1.41,
  "speech_overlap_seconds": {
    "union": 0.0,
    "vad": 0.0,
    "whisper": 0.0
  },
  "start": 41.55
}
```

Rendered audio: `/scratch/yy5919/pyro-kd-derived/tts/local-dev-pilot-v1/ad-A_cue-P1-qwen-tl1-80779200/overlap/51c334101a8bdf8325d018ac260a53bddfd26d1828a6d746cc8d150df5cb04ae.wav`; SHA256 `e53b473bd373e96f1d8b8aef06b90a1028218f781907473c412a8eb70f4df653`.

## 2. Word-cap compliance does not guarantee rendered speech fit

The P1 caption “Car drives through tunnel, explosion occurs.” uses exactly6words under the6-word cap. At1.2× playback, its actual rendered duration is2.525125s, occupying[18.5,21.025125]. Machine VAD speech begins20.7s, so measured speech collision is0.325125s; the Whisper and VAD union reports the same amount. The word cap approximates duration and is not a guarantee. This is an artifact-backed collision against machine speech bounds, not a human audibility rating. Nearby professional AD is supplied only as context; its timing does not temporally match this selected moment within±2s.

Run ID: `ad-V-P1-qwen-tl1-f33721e3`.

### `l5lLQICPOcnb_V_002`

Placement: 18.5s; generated text:

> Car drives through tunnel, explosion occurs.

Word count / limit: 6/6; generation error: `None`.

Reference context [22.4,30.03]s:

> The remains of the aircraft pass them by. In the enclosed cockpit, the pilot wears a stupefied look. The plane emerges from the tunnel and explodes.

Record hashes (SHA256 of UTF-8 compact, sorted-key JSON): moment `f8201abb114f65821e376303d968d3c93aa13f51801c5b0337028381b2ff3822`; generation `5c2d4b67a4de334a51116244dd7e84f82ac218014b578134ba5b0b24d498d1f5`; reference `9ca0a98ecc4da9e534d5bc772fc764c398bedc588c7979076ff3ed59488671df`.

| Frame time (s) | Original frame path | File SHA256 |
|---:|---|---|
|17.5|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000036.jpg`|`3ce197508562cfe99c4a8cf55c324126bd39362d349263305a628fddddc3a1ae`|
|18.0|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000037.jpg`|`47ec5e7699d0ccdafbe4888f4d9366d28975fb7370296af6a9465fccec83a43f`|
|19.0|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000039.jpg`|`163121afe3a78118d6bc3ae727d11d694d8692e87744b34776b07ac345a7efb0`|
|19.5|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000040.jpg`|`755769945c9289d27dd73f82e54fe01150dad15b74ceca4fa46a45467b729cd6`|

Actual rendered overlap row:

```json
{
  "duration": 2.525125,
  "end": 21.025125,
  "fits_speech_free_video": false,
  "id": "l5lLQICPOcnb_V_002",
  "out_of_video_seconds": 0.0,
  "speech_free_in_video_seconds": 2.2,
  "speech_overlap_seconds": {
    "union": 0.3251249999999999,
    "vad": 0.3251249999999999,
    "whisper": 0.3251249999999999
  },
  "start": 18.5
}
```

Rendered audio: `/scratch/yy5919/pyro-kd-derived/tts/local-dev-pilot-v1/ad-V-P1-qwen-tl1-f33721e3/overlap/a2d7586504a322616a5d7e80412fb1e80b341797366730daaac7ec21cd31fbb0.wav`; SHA256 `af847d4cc5c69c7d495fe6cd12f78fe0a841986edd74d34fbe1b9ca23d78cc6d`.

## 3. Two individually fitting descriptions overlap one another

The first rendered caption occupies[27.5,34.4288125]; the second occupies[34.0,36.8296875]. They overlap for0.4288125s despite a6.5s separation between selected timestamps. Both have fits_speech_free_video=true and zero VAD/Whisper overlap. Thus per-caption original-speech fit does not establish collision-free combined narration. The first output has27words under33; the second has9under13. This pair was chosen to illustrate a verified scheduling failure, not sampled to estimate its frequency.

Run ID: `ad-A_cue-P1-qwen-tl1-80779200`.

### `l5lLQICPOcnb_A_cue_001`

Placement: 27.5s; generated text:

> A man in goggles sits inside a vehicle. He looks out. Flames engulf another car behind him. Two men in hats sit in front of the fire.

Word count / limit: 27/33; generation error: `None`.

Reference context [22.4,30.03]s:

> The remains of the aircraft pass them by. In the enclosed cockpit, the pilot wears a stupefied look. The plane emerges from the tunnel and explodes.

Record hashes (SHA256 of UTF-8 compact, sorted-key JSON): moment `3375536dfe191b9dd52ee5969175d9e58be749c47b1904edf4cee8f38e69b6a0`; generation `83ab76f1a223cc81dc5c0e5cb318afcddcf4ce3c9ea5d3cb5b9943ddb0667150`; reference `9ca0a98ecc4da9e534d5bc772fc764c398bedc588c7979076ff3ed59488671df`.

| Frame time (s) | Original frame path | File SHA256 |
|---:|---|---|
|26.5|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000054.jpg`|`7f0e9935ef5df3ec890c3b2a4187c9252c3808ce314ca2214f6f08ce77ad215f`|
|27.0|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000055.jpg`|`5d5f3f8669ae8779fdf589462a63e3d5e5f4271a743cf009a8b085abd40d1858`|
|28.0|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000057.jpg`|`bf9c689853c265b9dfca8996cbc002d57365e8868a44ae509d866110a9546028`|
|28.5|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000058.jpg`|`a5560444149a84f3fe015ee98eb006cbeca80b8777de655c84c06b6807989301`|

Actual rendered overlap row:

```json
{
  "duration": 6.9288125,
  "end": 34.4288125,
  "fits_speech_free_video": true,
  "id": "l5lLQICPOcnb_A_cue_001",
  "out_of_video_seconds": 0.0,
  "speech_free_in_video_seconds": 6.9288125,
  "speech_overlap_seconds": {
    "union": 0.0,
    "vad": 0.0,
    "whisper": 0.0
  },
  "start": 27.5
}
```

Rendered audio: `/scratch/yy5919/pyro-kd-derived/tts/local-dev-pilot-v1/ad-A_cue-P1-qwen-tl1-80779200/overlap/5fcab3ed3ce64adb87980f7f3f43aa751c17ba8870c2e859eef8aea3a25be12e.wav`; SHA256 `925a9e89596dd69cccc5a7e60924f01c5d90b574e3063355ca63dfe9ec1761d5`.

### `l5lLQICPOcnb_A_cue_002`

Placement: 34.0s; generated text:

> A car crashes into a cave, causing an explosion.

Word count / limit: 9/13; generation error: `None`.

Reference context [31.32,34.54]s:

> Driving toward the fire, Indy crashes the car through scraps of metal.

Record hashes (SHA256 of UTF-8 compact, sorted-key JSON): moment `e662262a5f5cdeb3d2125f3ebcd4d4a3b91327886e69de8ed682d5dbe1bbd78f`; generation `cf4f30b1fdee6559186723c87d8f84940abf8ca0153191926c8972fcd9bf0a25`; reference `4fc1de0309cc0748c083039e58097bba014a2141aeb23a32fb2252000f547dbd`.

| Frame time (s) | Original frame path | File SHA256 |
|---:|---|---|
|33.0|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000067.jpg`|`c320281b14160ddfb26da34d4cf26479ed8d7a3e1cd4b14a9b1e4b71650b7e9f`|
|33.5|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000068.jpg`|`dcc6b6653b725b9002cfdca9d62b59183175b7f55a13c2ecb68db073fe6f8a38`|
|34.5|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000070.jpg`|`27b7a913055fd2cc95df461354e63df91fd1a23a63d7ed75795befb2db7323a9`|
|35.0|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000071.jpg`|`86fed4c5ebea328e92d65a3500a813c4fa305b202ac53dafbef7d6788d701cf2`|

Actual rendered overlap row:

```json
{
  "duration": 2.8296875,
  "end": 36.8296875,
  "fits_speech_free_video": true,
  "id": "l5lLQICPOcnb_A_cue_002",
  "out_of_video_seconds": 0.0,
  "speech_free_in_video_seconds": 2.8296875,
  "speech_overlap_seconds": {
    "union": 0.0,
    "vad": 0.0,
    "whisper": 0.0
  },
  "start": 34.0
}
```

Rendered audio: `/scratch/yy5919/pyro-kd-derived/tts/local-dev-pilot-v1/ad-A_cue-P1-qwen-tl1-80779200/overlap/62c325a41d86f50df3b79a9f57b3582803e7cd0fce75b160a7193b0bcfc08af5.wav`; SHA256 `3072385353fd64540f2aaa32111f438ffdfbfafac59156bff372fdebc105c37d`.

## 4. Unsupported action/object attribution in the actual input frames

Agent inspection opened all four actual model-input JPEGs at39.5,40.0,40.5,41.5s. The first shows two men through a car windshield; subsequent frames show an object dropping below a plane, the plane over the road/car, and an explosion on the ground. P0 instead says the men stand and concludes “The plane explodes.” Neither standing men nor a plane exploding is supported by these four inputs. The professional AD describes a bomb dropped in front of the car and the plane flying off, consistent with the image sequence. This is a bounded visual-grounding discrepancy in this example, not an event-detector false-positive label or a measured hallucination rate. P1 at the same moment also says “stand outdoors”; it describes a ground explosion, illustrating that one corrected phrase does not validate the entire caption. No human rating or full-clip review was performed.

Run ID: `ad-A_cue-P0-qwen-tl1-b56da784`.

### `l5lLQICPOcnb_A_cue_003`

Placement: 40.35s; generated text:

> A man in a brown suit and hat stands with another man in a military uniform. They appear to be in a desert setting. A plane flies overhead. The plane explodes.

Word count / limit: 31/197; generation error: `None`.

Reference context [40.25,45.25]s:

> The other plane drops a bomb in front of them. The car falls in the resulting crater and the plane flies off.

Record hashes (SHA256 of UTF-8 compact, sorted-key JSON): moment `cdd5129e9f74af32fbe69957cf6cd763de253410cd86f5a33fae2bd9ec6c4fa3`; generation `a61e85b6f55e19a84c5474008206509fcb77706f648a0eb5d40e2e0236ad2c82`; reference `89809c03a02291fd57dc521e7c1dcca1eb280adaeee36db93cd2500a0dc4e387`.

| Frame time (s) | Original frame path | File SHA256 |
|---:|---|---|
|39.5|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000080.jpg`|`8a72a10842c2d89c564a3a37be58adfe643d5d6494d577b8722e6f5cd4da8879`|
|40.0|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000081.jpg`|`b59ba19a985605e70f289f6c535dfcac9f8003774cc81a6bd43cbe12c47354a2`|
|40.5|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000082.jpg`|`c0d8047412a8ac0451a085eea02b529abed507b0087e19f039e315405aa38213`|
|41.5|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/frames/000084.jpg`|`d77b6118da7d6962c42b0a087ab1266569f2d3aa39a2f61d899a6faccce46f74`|

Actual rendered overlap row:

```json
{
  "duration": 8.7505,
  "end": 49.100500000000004,
  "fits_speech_free_video": true,
  "id": "l5lLQICPOcnb_A_cue_003",
  "out_of_video_seconds": 0.0,
  "speech_free_in_video_seconds": 8.7505,
  "speech_overlap_seconds": {
    "union": 0.0,
    "vad": 0.0,
    "whisper": 0.0
  },
  "start": 40.35
}
```

Rendered audio: `/scratch/yy5919/pyro-kd-derived/tts/local-dev-pilot-v1/ad-A_cue-P0-qwen-tl1-b56da784/overlap/57be2351c350f928957e7816d9dc3fb09b056a2d3b4b6ca6e29810bb9d4651d5.wav`; SHA256 `94f40e6543a21a379a55b7230eaebde12e6417be53433f9b7ff478a69e41340b`.

## Limits and provenance

No conclusion about missed silent visual events or AST/CLAP false positives is established by these four cases. Such categories need their own grounded review and, for detector accuracy, the deferred human labels. Case4 supports a limited caption-grounding observation only. The same examples were not used to tune the formal settings.

Reproduction: locate the named IDs in the recorded generation/moment JSONL files, recompute compact sorted-key record hashes, open the original frames for case4, and inspect the recorded overlap rows and rendered WAV hashes. The small arithmetic checks are overlap=max(0,min(end1,end2)−max(start1,start2)) and VAD-membership at the triggering cue.

Source file hashes:

| Source artifact | SHA256 |
|---|---|
|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/audio_features.json`|`48c0c6d659410fde5855a322c97b72d0b85c6db047ffe14cae18895dc9f51b8d`|
|`/scratch/yy5919/pyro-kd-derived/cache/reframed/l5lLQICPOcnb/speech_features.json`|`3ab735ba6a0e4ff9eeae808258c2c49937cbf32c07d2c8d988d05cc536251627`|
|`/scratch/yy5919/pyro-kd-derived/cache/reframed/v2gCnXXTSp6W/audio_features.json`|`86aed4f8823bb78354280a97c26160366b73a8d0536f58438c8b6ce9b522d198`|
|`/scratch/yy5919/pyro-kd-derived/cache/reframed/v2gCnXXTSp6W/speech_features.json`|`3d949c4788fb50dd4614c67fb7d2cea24fea0a21934c66859eee593bcfd8837c`|
|`/scratch/yy5919/pyro-kd-derived/datasets/reframed/validation.jsonl`|`30e8e2fce70d36d9caed78d31ecbc38c85042c04d70d00064d3a8ef056b80a46`|
|`/scratch/yy5919/pyro-kd-derived/tts/local-dev-pilot-v1/ad-A_cue-P0-qwen-tl1-b56da784/overlap/report.json`|`c04e88a59d48aa12c1c265e1dfdbe73a56a0963a906c509f09c6c445d034450e`|
|`/scratch/yy5919/pyro-kd-derived/tts/local-dev-pilot-v1/ad-A_cue-P1-qwen-tl1-80779200/overlap/report.json`|`9a67ff7d0f0238cda6b6a6b4d76bf1c95957ef8e8d485c54acf30a143a8ef7c1`|
|`/scratch/yy5919/pyro-kd-derived/tts/local-dev-pilot-v1/ad-V-P1-qwen-tl1-f33721e3/overlap/report.json`|`469a07ca9f86d4a9dc51dcaae3c81cffe1f1046ae9afa092aa8d4b12ae114c14`|
|`/scratch/yy5919/pyro-resume-results/experiments/reframed_local_dev_pilot_v1/generations/ad-A_cue-P0-qwen-tl1-b56da784.jsonl`|`3f78edcf7968db597a531b0414aaaad238dc4fa8aae34e7a594e9ba8d734be4a`|
|`/scratch/yy5919/pyro-resume-results/experiments/reframed_local_dev_pilot_v1/generations/ad-A_cue-P1-qwen-tl1-80779200.jsonl`|`129ef15e455d52c3348fbfeab3d39979885c52cab01f994aed889dacdeed6446`|
|`/scratch/yy5919/pyro-resume-results/experiments/reframed_local_dev_pilot_v1/generations/ad-V-P1-qwen-tl1-f33721e3.jsonl`|`ab7bc82f7a8a9f085bc5ddcef3446985ee87422e14115712c8b32e4841609064`|
|`/scratch/yy5919/pyro-resume-results/experiments/reframed_local_dev_pilot_v1/moments/A_cue-b8ed44b2.jsonl`|`123485199936dbc25d21488999a4a7f509be385347e6e09196419eac43c96cb7`|
|`/scratch/yy5919/pyro-resume-results/experiments/reframed_local_dev_pilot_v1/moments/V-1cd0fb55.jsonl`|`5a7d6903eeda95551072ca5e1d0813248e525e3935cf6dc6525613ca64b7c147`|
|`/scratch/yy5919/pyro-resume-results/experiments/reframed_local_dev_pilot_v1/selections/A_cue-b8ed44b2.jsonl`|`6ca2fa4bc1c4d5cd8d897f8dc8fed4e05a4598e7dfbfb87f520d2e329b640870`|
|`/scratch/yy5919/pyro-revision15-records/configs/reframed_splits.json`|`96bc45bab00287e11423f5eef68083a389349253ca49c0d67f64a2a63f5e22c6`|
