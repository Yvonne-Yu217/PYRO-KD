# Development failure analysis

These are automated checks and agent analysis, not human detector labels or
blinded quality ratings. No positive-results requirement was imposed.

Across the ten local core cells,1110 requested moments became1008 selections,
804 safely speakable prepared inputs,705 valid generated captions and688
individually speech-fitting rendered captions. Missing stages stay visible;
conditional text/TTS scores do not imply all requested moments succeeded.

84of789 complete decodes exceeded their word budgets (10.65%);15additional
outputs were truncated and have unknown complete-decode budget compliance.
The512-token dev diagnostic rescued5of15 at128–139tokens. The other10repeated
phrases and still reached512tokens; repeated bigrams occurred39–111times in
individual outputs. The formal192-token ceiling covers completed dev responses
while bounding these loops. Full raw outputs and failed IDs remain in scratch.

Actual1.2x Piper renders also overlap each other, even when individually free
of original speech. Excess narration seconds are sum(rendered durations) minus
the union of their placement intervals; this measures overlapping layers, not
speech collision. The proposal's fixed2-second separation does not prevent this.
A combined narration scheduling improvement would require a separately declared
protocol; current experiments retain the specified rule and expose this limitation.

| Selector | Prompt | Excess overlapping narration seconds (10dev clips) |
|---|---|---:|
|U|P0|65.851|
|U|P1|171.360|
|V|P0|153.519|
|V|P1|201.086|
|A_cue|P0|191.371|
|A_cue|P1|334.828|
|AV|P0|170.073|
|AV|P1|341.069|
|O|P0|105.185|
|O|P1|184.284|

The VAD-gated Whisper baseline emits zero bracketed event tags on these10clips.
That is a limitation of the event-source baseline, not evidence that the clips
lack sounds. AST/CLAP precision and peer preference remain unmeasured until real
human annotations arrive. Long A/AV capacity limitations are recorded separately
in long-selection-budget-audit.md.
