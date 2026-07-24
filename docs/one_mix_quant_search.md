# Please analyze the source code of ONE compliler /home/cmj/w…

> Please analyze the source code of ONE compliler /home/cmj/works/ONE, espacially for the optimization and quantization. I am consider how to reduce the separation quanlity decreazing for quantizating the audio separation model, while keep the separation latency not increasing. I tried the unint8 quantization, the latency is ok, but the quanlity loss is large. And with int16, the latency is too big but the quanlity is goog. So I need you to investigate the source code of the ONE compiler and find a way for make the quality not downgrade too much while keep the latency low. You could make a md doc for the analyzation and propasals.

<details><summary>25 previous messages</summary>

> I’ll ground this in the live `ONE` checkout and preserve the result as a repo doc under `docs/`. I’ll first map the quantization/optimization code paths, then turn that into concrete proposals that fit the audio separation/NPU constraints.
>
> I found the prior project memory that says the authoritative ASS worktree is `/home/cmj/works/ASS`, while this task really does target the separate `/home/cmj/works/ONE` compiler tree. I also noticed unrelated ASS edits already present, so I’ll leave them alone and only add a new analysis doc.
>
> The first pass shows the normal `one-quantize` path is a wrapper around `record-minmax` plus `circle-quantizer`, while `circle2circle` has optimization passes that can alter Q/DQ boundaries. I’m narrowing now into the source files that decide calibration ranges, dtype selection, and whether mixed precision is actually usable.
>
> A useful thread appeared: `circle-mpqsolver` exists and is specifically about mixed-precision quantization, not just basic uint8 calibration. I’m checking whether it produces deployable mixed models or only analysis artifacts, because that determines whether “some int16, mostly uint8” is a realistic route.
>
> The calibration story is already actionable: `record-minmax` records raw activation extrema, and `minmax-embedder` can replace them with percentiles before quantization. For audio separation, that matters because a few transient frames can blow up uint8 scale and smear quieter speech/music/effects detail.
>
> Another important detail: the public docs still say `one-quantize` is layer-wise uint8 for convolution and that int16/channel-wise were future work, but the lower-level `circle-quantizer` code now exposes channel granularity and weight-only int8/int16. I’ll separate “documented one-command path” from “lower-level tool capability” in the proposal.
>
> The source confirms mixed-precision support is implemented as Q ops around selected layers: default dtype around the graph, chosen op dtype inside the layer, and direct re-quantization of const weights so they are not degraded through uint8 first. The tradeoff is extra Quantize boundaries, so candidate layers must be few and high-impact.
>
> I’m checking weight and bias paths next. The key question is whether we can get most of the benefit through channel-wise weights and percentile activation ranges, which should not add runtime ops, before paying the heavier cost of int16 activations.
>
> There is also a more surgical route through `q-implant`: it can implant externally computed qparams and quantized weights from JSON/NumPy. That looks like a candidate for custom audio-aware scales or QAT-derived qparams without changing runtime latency, but it needs a complete qparam set for the graph.
>
> `circle-mpqsolver` is now clearly the compiler-native mixed-precision path. It can search a uint8/int16 split using a calibration dataset, and it also has fixed patterns for layernorm/softmax. I’m reading the scoring code so the doc can say what its “quality” proxy really measures and where it may fail for separation metrics.
>
> The pattern solver is narrow but still relevant for separation models if softmax-like attention or normalization survived export as primitive subgraphs. It marks only the numerically fragile subpart int16, which is exactly the latency/quality compromise we want, but it probably needs extension for SFC or mask-generation patterns.
>
> For optimization, I’m focusing on passes that affect quantized graph boundaries and NPU latency: fusing conv/bias/activation, removing redundant reshapes/transposes/slices, folding dequantize, and removing redundant quantize. These are the pieces most likely to change latency without changing the model architecture.
>
> One compiler-side caution: the pass named `remove_qdq_for_mpo` only recognizes `FULLY_CONNECTED` and `BATCH_MATMUL`. If our separator is mostly Conv2D/TransposeConv, mixed precision will likely keep more Q/DQ boundaries unless we add a Conv2D-aware pass or avoid many mixed islands.
>
> The ONE manual path in the prompt is under ASS, not the ONE root, so I’ll read `/home/cmj/works/ASS/OPERATION_MANUAL_PYTORCH_TO_ONE_NPU.md` for the project’s exact import/opt/quant flow and keep the doc aligned with that.
>
> The repo manual reinforces a practical point: success must be artifact-checked at each import/opt/quant stage, not inferred from return code alone. I’ll include a proposed experiment matrix with concrete artifact and metric checks so we can compare quality and latency cleanly.
>
> The high-level `one-quantize` wrapper already exposes the knobs we need: channel granularity, percentile vs moving-average calibration, save-intermediate, AMPQ, and per-tensor force/copy qparams. That means the first experiments can be done without patching ONE.
>
> Before drafting, I’m doing a small repo-grounding pass on the ASS side so the recommendations name the likely separator family and the current compile flow, not just generic compiler ideas.
>
> There are several old ASS logs around `one-quantize` and NPU verification; I’m using only the high-signal bits so the new document stays focused on the compiler mechanisms and next experiments, not a full archaeology dig.
>
> I found `BandSFCNetNPU` and a prior “quant positive” operation note, which matches the current goal pretty closely. I’ll fold those local observations into the proposal, but the main recommendations still come from the ONE compiler source.
>
> The ASS verifier currently emits uint8/channel configs by default, so the “lowest-friction” next step is to add controlled variants of the quantization stage rather than changing model code first. I’m now drafting the markdown doc with source references and a staged experiment plan.
>
> One source-level improvement looks especially promising: `record-minmax` calibrates from per-sample tensor min/max values, then chooses percentiles over those extrema. For audio, an element-level histogram or MSE-based clipping observer could improve uint8 scale choices with zero runtime latency change.
>
> I have enough evidence now. I’ll create the doc under `docs/` and keep it practical: findings from source, what to try immediately, and what source changes are worth making if experiments confirm the bottleneck.
>
> I’m adding the doc now. I’ll include an “immediate no-compiler-patch path” and a “compiler patch path” so we can choose based on how much quality we recover from calibration and AMPQ first.
>
> <details><summary>Created a file</summary>
>
> - Wrote `./docs/ONE_QUANTIZATION_QUALITY_LATENCY_ANALYSIS.md` (+363 -0)
>
> </details>
>
> The doc is written. I’m doing a quick sanity pass for typos, ASCII cleanliness, and whether the commands/source paths line up with the inspected code.
</details>

Created [docs/ONE_QUANTIZATION_QUALITY_LATENCY_ANALYSIS.md](./docs/ONE_QUANTIZATION_QUALITY_LATENCY_ANALYSIS.md:1).

Main conclusion: the best low-latency path is not full int16. Try better uint8 calibration first, then small contiguous int16 “quality islands” through `quant_config` or `one-quantize --ampq`. The strongest compiler-side proposal is adding histogram/MSE calibration to `record-minmax`, because it can improve uint8 quality without changing runtime graph latency.

I did not modify ONE source. I left existing ASS/ONE worktree edits untouched and only added the new doc. Verification done: doc is ASCII-only and source references/commands were sanity-checked.

> How to add histogram/MSE calibration to record-minmax? Give me the neccessary tools/scripts for this

<details><summary>59 previous messages</summary>

> I’ll turn this into something usable rather than just a recipe. I’ll inspect the `record-minmax` internals and build files, then add a focused implementation plus a runner script so you can generate histogram/MSE-calibrated Circle artifacts and compare them against the current percentile path.
>
> I’ll keep the implementation narrow: add a new calibration computer/collector path while preserving the existing `percentile` and `moving_average` modes. The goal is a drop-in extra mode, not a rewrite of quantization.
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> I’ve got the core direction now: `record-minmax` only records per-sample tensor extrema today, so true calibration needs new activation statistics, not just another formula on the existing min/max list. I’m going to add a low-overhead sampled MSE clipping mode that feeds the existing `circle-quantizer --quantize_with_minmax` path, plus a sweep script so you can compare it against percentile calibration.
>
> A useful detail from the code: the min/max recorder is already the calibration boundary used by `one-quantize`, so we can improve uint8 quality without adding runtime graph ops. The patch will only change how ranges are chosen during calibration; the deployed model still consumes ordinary embedded min/max metadata.
>
> I’m going to make this a calibration-only extension with a deliberately small surface: one new `mse_clip` mode, a few tuning flags, and a helper script that runs the standard ONE quantization pipeline. That keeps it easy to test against your existing `uint8` flow.
>
> I’m about to edit the ONE compiler sources and add one ASS-side helper script. The key code change is in `record-minmax`: when `--mode mse_clip` is selected, it will collect a bounded reservoir sample of activation values per tensor and search clipping ranges that minimize simulated uint8 dequantization MSE.
>
> <details><summary>Edited 4 files</summary>
>
> - Wrote `~/works/ONE/compiler/record-minmax/include/MinMaxVectors.h` (+3 -0)
> - Wrote `~/works/ONE/compiler/record-minmax/include/MinMaxObserver.h` (+33 -4)
> - Wrote `~/works/ONE/compiler/record-minmax/src/MinMaxObserver.cpp` (+5 -3)
> - Wrote `~/works/ONE/compiler/record-minmax/include/MinMaxComputer.h` (+29 -0)
>
> </details>
>
> One subtle choice here: I’m using reservoir sampling instead of storing full tensors or doing a two-pass exact histogram. That gives us the same practical goal, clipping by MSE over activation distributions, but keeps calibration memory bounded even for large audio separation graphs.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `~/works/ONE/compiler/record-minmax/src/MinMaxComputer.cpp` (+146 -14)
>
> </details>
>
> I noticed one place where a helper could have changed the old percentile/moving-average behavior by normalizing ranges. I’m tightening that now so the existing modes remain byte-for-byte semantic equivalents, and only the new `mse_clip` mode applies zero-inclusive clipping logic.
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `~/works/ONE/compiler/record-minmax/src/MinMaxComputer.cpp` (+1 -2)
> - Wrote `~/works/ONE/compiler/record-minmax/src/RecordMinMax.cpp` (+1 -1)
>
> </details>
>
> Next I’m exposing the mode through both command layers: direct `record-minmax` usage and the higher-level `one-quantize` wrapper. That matters because your normal pipeline should not need manual intermediate file juggling.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `~/works/ONE/compiler/record-minmax/driver/Driver.cpp` (+20 -2)
>
> </details>
>
> I’m also guarding the new numeric flags so a typo like `--mse_max_samples -1` cannot silently become a giant unsigned value. Small thing, but those are exactly the little cliffs calibration tooling likes to hide.
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `~/works/ONE/compiler/record-minmax/driver/Driver.cpp` (+7 -3)
> - Wrote `~/works/ONE/compiler/one-cmds/one-quantize` (+14 -3)
>
> </details>
>
> The wrapper changes are intentionally generic: `one-quantize --mode mse_clip ...` will pass the extra knobs down to `record-minmax`, then continue with the same `circle-quantizer --quantize_with_minmax` step as before.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `~/works/ONE/compiler/one-cmds/one-quantize` (+4 -0)
>
> </details>
>
> I’m wiring validation now. The MSE mode reuses `--min_percentile` and `--max_percentile` as the outer candidate range, then searches inward, so the same calibration sweep dimensions still make sense.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `~/works/ONE/compiler/one-cmds/one-quantize` (+14 -1)
>
> </details>
>
> I found two record-minmax call sites in `one-quantize`: normal PTQ and AMPQ pre-recording. I’m passing the new flags through both so MSE calibration can also be combined with mixed precision experiments later.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `~/works/ONE/compiler/one-cmds/one-quantize` (+4 -0)
>
> </details>
>
> I checked the local NPU helper style too. The new script will live under `tools/online/` alongside the existing compile/verify tooling and will default to the same ONE build command directory when available.
>
> The helper will build the same `PATH`/`LD_LIBRARY_PATH` style the repo already uses for ONE tools, so it should work from a plain shell as long as the build artifacts exist.
>
> I’m adding a sweep runner rather than a one-off command generator. For this problem, the useful answer is not “MSE always wins”; it is a repeatable grid over percentile outer ranges plus MSE clipping, so you can check SDR/SI-SDR and latency artifacts side by side.
>
> <details><summary>Created a file</summary>
>
> - Wrote `./tools/online/run_one_mse_calibration_sweep.py` (+229 -0)
>
> </details>
>
> One practical note I found: the built `one-cmds/one-quantize` is a generated copy, while `record-minmax` is symlinked to the compiled binary. After rebuilding, use the built command under `/home/cmj/works/ONE/build/compiler/one-cmds` for the sweep.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./tools/online/run_one_mse_calibration_sweep.py` (+1 -1)
>
> </details>
>
> Now I’m adding a short operation note with the exact build and run commands, because this is one of those changes where the code path and the experiment path need to stay glued together.
>
> <details><summary>Created a file</summary>
>
> - Wrote `./docs/one_record_minmax_mse_calibration_operation.md` (+82 -0)
>
> </details>
>
> The doc now has the direct command and the sweep workflow. I’m going to do a build/parse pass next, because C++ calibration changes are only useful if the local ONE build accepts them cleanly.
>
> The diff showed permission-bit noise on the ONE source files, so I’m cleaning that up before building. The only executable bit I want to keep is on the new ASS sweep script.
>
> The permission cleanup hit an ownership wall on the ONE checkout, so I’ll avoid relying on chmod there and keep the content changes focused. I’ll still build from the working tree; if the mode bit remains noisy in `git diff`, I’ll call it out clearly.
>
> I’ve finished the code edits and the Python syntax checks passed. Now I’m rebuilding `record-minmax` plus the generated `one-quantize` command so the local build tree actually contains the new CLI options.
>
> The build did not start because this ONE build cache was generated in `/app/ONE/build`, while we are invoking it through `/home/cmj/works/ONE/build`. I’m checking the actual path mapping now instead of reconfiguring the compiler build.
>
> The cached build also points to `/usr/bin/cmake`, which is not present here; only the ONE command venv has `cmake`. I’ll see whether the original `/app/ONE` path can be restored as a symlink, then otherwise I’ll fall back to source-level validation and help output from the Python wrapper.
>
> Since `/app` cannot be recreated from this user account, I’m checking whether the generated Makefiles have a “fast” target that skips the stale CMake regeneration check. That can sometimes compile a single target without touching the old cache.
>
> The generated Makefiles also hard-code `cd /app/ONE/build`, so the old build tree is not directly reusable from this path. I’m checking the project’s configured build entry point; if a small fresh build is too expensive, I’ll leave the patch and scripts with clear rebuild instructions.
>
> A fresh configure may be more disruptive than this task deserves, so I’m checking the size of the existing build tree before deciding whether to make a relocated copy for validation. If it is too large, I’ll stop at syntax/source validation and document the stale-cache rebuild blocker.
>
> The build tree is 8.1 GB, so copying it just to relocate paths would be a bit much. I’m going to try a narrower validation: reuse the existing generated compile and link recipes, substituting only `/app/ONE` to the current checkout path for the changed `record-minmax` objects.
>
> I’m doing a manual single-target rebuild from the generated compile flags now. It will overwrite only the existing `record-minmax` binary in the build tree and then I’ll copy the updated Python wrapper into `build/compiler/one-cmds`.
>
> The manual `record-minmax` rebuild succeeded. I’m updating the generated `one-quantize` copy next and then I’ll check both command-line help paths for the new options.
>
> `record-minmax --help` shows the new mode. The `one-quantize --help` probe did not print through the wrapper, so I’m checking whether its bundled venv entry is broken or whether this is just the wrapper’s preamble behavior.
>
> Both help paths now expose `mse_clip`. I’m doing one small negative validation next to make sure the new argument checks trigger before model loading, then I’ll inspect the final diffs for anything accidental.
>
> The validation behaved as expected: the wrapper rejects invalid MSE settings cleanly, and the rebuilt C++ binary throws before loading the model. I’ll make one noninteractive attempt to clear the root-owned executable-bit noise; if sudo is unavailable, I’ll leave it noted.
>
> One small refinement before I wrap up: the MSE computer was copying each tensor’s sampled activation vector while calculating ranges. I’m switching that path to a const reference so calibration RAM stays closer to the intended bound.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `~/works/ONE/compiler/record-minmax/src/MinMaxComputer.cpp` (+5 -3)
>
> </details>
>
> That refinement is in. I’ll relink `record-minmax` once more and rerun the quick help/syntax checks.
>
> The relink and Python compile checks passed. I’m doing the final command probes and then I’ll summarize exactly what changed and how to run the sweep.
>
> The final probes pass. I’m checking the exact file list now so the close-out is crisp and doesn’t bury you in the giant unrelated dirty state of these worktrees.
>
> One housekeeping scan wandered too broadly through the repo. I’m stopping that and then I’ll finish; the actual validation already completed.
>
> The syntax checks left two targeted `.pyc` files behind; I’m removing only those generated artifacts.
>
> Everything is now clean apart from the intended source/doc/script files. I’m doing one final content skim of the changed C++ path so I can give you the important implementation details rather than just a file list.
>
> One last doc adjustment: the normal CMake build command is correct for a healthy build tree, but this local cache was created under `/app/ONE`. I’m adding that caveat so future-you does not trip on the same stale-cache wrinkle.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./docs/one_record_minmax_mse_calibration_operation.md` (+2 -0)
>
> </details>
>
> All planned work is complete. I’m just collecting the exact file references for the final response now.
>
> I spotted two long lines in the new helper while collecting references; not harmful, but I’m wrapping them now so the script matches the surrounding repo style better.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./tools/online/run_one_mse_calibration_sweep.py` (+13 -2)
>
> </details>
>
> Formatting cleanup is done; I’m rerunning the helper syntax check and removing the generated bytecode again.
</details>

Implemented the calibration path and the helper tooling.

What changed:
- Added `record-minmax --mode mse_clip` with bounded activation reservoir sampling and uint8 dequantization-MSE range search in [MinMaxComputer.cpp](~/works/ONE/compiler/record-minmax/src/MinMaxComputer.cpp:56).
- Added sample collection in the observer in [MinMaxObserver.h](~/works/ONE/compiler/record-minmax/include/MinMaxObserver.h:43) and [MinMaxObserver.cpp](~/works/ONE/compiler/record-minmax/src/MinMaxObserver.cpp:90).
- Exposed `--mse_max_samples` and `--mse_search_steps` in [Driver.cpp](~/works/ONE/compiler/record-minmax/driver/Driver.cpp:86).
- Forwarded those options through `one-quantize`, including AMPQ pre-recording, in [one-quantize](~/works/ONE/compiler/one-cmds/one-quantize:135).
- Added sweep runner: [run_one_mse_calibration_sweep.py](./tools/online/run_one_mse_calibration_sweep.py:1).
- Added operation doc with commands: [one_record_minmax_mse_calibration_operation.md](./docs/one_record_minmax_mse_calibration_operation.md:1).

Main usage:

```bash
python /home/cmj/works/ASS/tools/online/run_one_mse_calibration_sweep.py \
  --input-circle /path/to/model.opt.circle \
  --calib-data /path/to/calibration.h5 \
  --min-percentiles 0.01,0.05,0.1,0.5,1.0 \
  --max-percentiles 99.0,99.5,99.9,99.95,99.99
```

Verification done:
- Rebuilt/relinked local `record-minmax` manually because the existing CMake cache points to `/app/ONE/build`.
- Copied the updated `one-quantize` wrapper into `/home/cmj/works/ONE/build/compiler/one-cmds/one-quantize`.
- Verified `record-minmax --help` and `one-quantize --help` show `mse_clip` / `mse_` options.
- Ran Python syntax checks for the new sweep script and `one-quantize`.
- Checked invalid `--mse_max_samples 0` is rejected.

One caveat: the ONE checkout has root-owned `777` files, so `git` reports executable-bit mode changes on the touched ONE files and I could not clear them without sudo password. Content changes are the intended ones above.

> We should not change the code of ONE compiler, We need tools that just reuse existing ONE tools

<details><summary>25 previous messages</summary>

> Got it. We’ll pivot: no ONE compiler source changes. I’m going to remove the ONE source edits I made and keep this as an external calibration/sweep workflow that calls stock `one-import`, `one-optimize`, `record-minmax`, `circle-quantizer`, `circle-inspect`, and `one-quantize` only.
>
> The important practical constraint is: stock `record-minmax` only emits min/max metadata, not activation histograms. So the external replacement has to do “MSE calibration” at the model-selection level: sweep existing calibration knobs, evaluate MSE/ASS metrics, and optionally combine with stock AMPQ for selective int16 islands.
>
> I saved the compiler patch to `/tmp/one_record_minmax_mse.patch` as a safety snapshot, then I’m applying the reverse to remove the ONE source edits. After this, the replacement work will live only under ASS tools/docs.
>
> The reverse patch applied cleanly. I’m checking the targeted ONE files now; there may still be repo-wide mode noise from the existing checkout, but the source-content changes should be gone.
>
> There is one more place to clean up: I had updated the built `record-minmax` binary and generated `one-quantize` copy during validation. I’m rebuilding/copying those from the restored stock source so the local ONE tools themselves are back to stock behavior too.
>
> The built stock tools are restored. I’m verifying that `mse_clip` is no longer present in the ONE command help, then I’ll replace the ASS-side script with a stock-tool sweep.
>
> I’ll replace the previous `mse_clip` helper with a stock-only sweep runner. It will use existing ONE calibration modes plus optional AMPQ, and it can pass `--evaluate_result --print_mse` so the “MSE” decision happens outside the compiler by selecting the best generated candidate.
>
> I’m deleting the misleading source-patch operation note and replacing it with a stock-only operation note. The old helper script also gets replaced so there are no `mse_clip` arguments left anywhere in the workflow.
>
> <details><summary>Deleted 2 files</summary>
>
> - Deleted `./tools/online/run_one_mse_calibration_sweep.py` (+0 -240)
> - Deleted `./docs/one_record_minmax_mse_calibration_operation.md` (+0 -84)
>
> </details>
>
> Now I’m adding the stock-only sweep runner. It will create percentile, moving-average, and optional AMPQ candidates using `one-quantize` exactly as shipped.
>
> <details><summary>Created a file</summary>
>
> - Wrote `./tools/online/run_one_stock_quant_sweep.py` (+352 -0)
>
> </details>
>
> The new runner uses only stock `percentile`, `moving_average`, and `--ampq` paths. I’m adding the operation doc with commands and the limitation about true histogram calibration requiring compiler-side activation statistics.
>
> <details><summary>Created a file</summary>
>
> - Wrote `./docs/one_stock_quant_calibration_operation.md` (+82 -0)
>
> </details>
>
> I’m also editing the earlier analysis doc so it no longer reads like source modification is the active recommendation. It will still preserve the reasoning, but the actionable path will point to the stock-only tool.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./docs/ONE_QUANTIZATION_QUALITY_LATENCY_ANALYSIS.md` (+5 -5)
>
> </details>
>
> The new script parses and its help is clean. The older analysis doc still had a later “patch ONE” section, so I’m rewriting that section as “not active under the stock-only constraint” rather than leaving conflicting instructions.
>
> I’m also aligning the AMPQ sweep numbers in the analysis doc with the stock sweep script, then I’ll clean the generated `.pyc`.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./docs/ONE_QUANTIZATION_QUALITY_LATENCY_ANALYSIS.md` (+13 -57)
>
> </details>
>
> The docs now describe the no-source-change path consistently. I’m running final checks: no `mse_clip` references in the active workflow, Python parses, and ONE source diff is empty.
>
> One grep check had a shell quoting hiccup because of backticks in the pattern; the actual Python and ONE-source checks passed. I’m rerunning the text scan with simpler patterns.
>
> The ONE source tree is back to no targeted diffs, and the active ASS artifacts are stock-only. I’m doing a quick dry-run with temporary placeholder inputs to make sure the runner generates stock `one-quantize` commands only.
>
> The dry-run produced only stock ONE commands: percentile, moving_average, and AMPQ. I’ll remove the temporary dry-run directory and then close out.
>
> Temporary files are gone. Final status check now, then I’ll summarize the corrected stock-only workflow.
>
> The script and doc are in place, but I want the runner to be directly executable like the repo’s other tool scripts, so I’m setting its executable bit and doing one final no-source-change check.
>
> The targeted ONE diff is empty, so we are no longer changing compiler source. Because the ASS files are new/untracked, `git diff --stat` does not show them; I’m checking status instead.
</details>

Corrected. I reverted the ONE compiler source changes and restored the local built `record-minmax` / `one-quantize` behavior back to stock. Targeted ONE source diff is empty now.

I replaced the previous compiler-patch approach with stock-only tooling:

- Added [run_one_stock_quant_sweep.py](./tools/online/run_one_stock_quant_sweep.py:1)
  - Calls stock `one-quantize`
  - Sweeps stock `percentile`
  - Sweeps stock `moving_average`
  - Sweeps stock `--ampq`
  - Optionally passes `--evaluate_result --print_mse`
  - Writes per-run logs and `summary.json`

- Added [one_stock_quant_calibration_operation.md](./docs/one_stock_quant_calibration_operation.md:1)
  - Explains the no-ONE-source-change workflow
  - Notes that true per-tensor histogram/MSE calibration is not available from stock `record-minmax`
  - Uses external MSE or ASS quality metrics to select among stock-generated candidates

- Updated [ONE_QUANTIZATION_QUALITY_LATENCY_ANALYSIS.md](./docs/ONE_QUANTIZATION_QUALITY_LATENCY_ANALYSIS.md:1)
  - Now points to the stock-only sweep instead of recommending a compiler patch.

Example:

```bash
python /home/cmj/works/ASS/tools/online/run_one_stock_quant_sweep.py \
  --input-circle /path/to/model.opt.circle \
  --calib-data /path/to/calibration.h5 \
  --modes percentile,moving_average,ampq \
  --min-percentiles 0.01,0.05,0.1,0.5,1.0 \
  --max-percentiles 99.0,99.5,99.9,99.95,99.99 \
  --ampq-qerror-ratios 0.01,0.03,0.05
```

Validation:
- Python compile passed.
- Help output works.
- Dry-run generated only stock ONE commands: `--mode percentile`, `--mode moving_average`, and `--ampq`.
- No `mse_clip` or ONE compiler modification remains in the active workflow.

> could you make it also support a list txt file for the calib-data(not only h5 file)

<details><summary>16 previous messages</summary>

> Yes, good catch. The stock ONE path can already accept `list/filelist`, but my helper defaulted to `h5`, so a `.txt` calibration list would silently be passed with the wrong format. I’m going to make the script infer `h5`, `list`, or `directory` from `--calib-data` unless you explicitly override it, and update the docs/examples.
>
> I found the exact spot: `--input-data-format` was defaulting to `h5`. I’m changing that to `auto` and adding extension/path inference, so `.txt`, `.list`, `.lst`, or `.filelist` calibration manifests become stock ONE `input_data_format=list`.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./tools/online/run_one_stock_quant_sweep.py` (+30 -1)
>
> </details>
>
> The inference is in. Next I’m updating the operation doc to show both H5 and text file-list calibration examples, and to mention the evaluation-data caveat.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./docs/one_stock_quant_calibration_operation.md` (+17 -0)
>
> </details>
>
> I’ll also adjust the analysis note where it still shows only H5 examples, so the docs do not imply H5 is required.
>
> That doc section had drifted slightly from the exact snippet I expected, so I’m opening the relevant lines and patching the actual text.
>
> I found the exact snippets. I’m updating them to say “h5 or list manifest” and to use the smaller AMPQ ratio range already used by the helper.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./docs/ONE_QUANTIZATION_QUALITY_LATENCY_ANALYSIS.md` (+7 -7)
>
> </details>
>
> The script now infers list format from the calibration file extension. I’m adding one small validation for `--test-data` existence too, since the same command can pass evaluation data through stock `one-quantize`.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./tools/online/run_one_stock_quant_sweep.py` (+2 -0)
>
> </details>
>
> I’ll run syntax checks and dry-runs for both `.txt` and `.h5` calibration inputs to verify the generated stock ONE command uses `list` for text manifests and `h5` for HDF5.
>
> Dry-runs look right: `.txt` now maps to `--input_data_format list`, while `.h5` stays `h5`. I’m removing the syntax-check bytecode and checking the final file status.
>
> The support is in and verified. I’m doing one tiny formatting cleanup in the argument block, then I’ll give you the exact usage.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./tools/online/run_one_stock_quant_sweep.py` (+9 -2)
>
> </details>
>
> I’ll rerun the quick syntax check after that cleanup and remove the generated bytecode.
</details>

Done. The sweep tool now supports text calibration file lists.

Updated [run_one_stock_quant_sweep.py](./tools/online/run_one_stock_quant_sweep.py:92) so `--input-data-format` defaults to `auto`:

- `.h5` / `.hdf5` -> `h5`
- `.txt` / `.lst` / `.list` / `.filelist` -> `list`
- directory path -> `directory`
- manual override still works: `--input-data-format list`

Example:

```bash
python /home/cmj/works/ASS/tools/online/run_one_stock_quant_sweep.py \
  --input-circle /path/to/model.opt.circle \
  --calib-data /path/to/calibration_files.txt \
  --modes percentile,ampq
```

I also updated [one_stock_quant_calibration_operation.md](./docs/one_stock_quant_calibration_operation.md:47) with the list-file example.

Validation:
- Python compile passed.
- Dry-run confirmed `.txt` generates stock ONE command with `--input_data_format list`.
- No ONE compiler source changes.

> for mixture quantization, could you create some useful tools for choose which layers/nodes I should put into int16 or uint8 according to the ONE comipler source code?

<details><summary>36 previous messages</summary>

> Yes. I’ll keep this stock-ONE-only: no compiler edits. I’m going to inspect the existing ONE mixed-precision config/AMPQ code paths, then add an ASS-side analyzer that reads Circle operator names/counts plus optional quantization logs and emits ranked node suggestions and starter `quant_config` files.
>
> I’m starting from the schema, because the useful output needs to be directly consumable by stock `one-quantize --quant_config`, not just a pretty ranking table.
>
> The first source search was noisy, but it confirmed the key entry point: `circle-quantizer` reads a JSON `layers` list with per-node `name`, `dtype`, and `granularity`. I’m narrowing into that parser and the AMPQ output format now so the tool emits exactly what ONE expects.
>
> The config examples are a little counterintuitive but useful: qconfig entries name tensor/layer outputs and assign dtype/granularity per entry, while the command-line default remains the fallback for everything else. I’ll inspect how ONE applies those names next, especially where mixed dtype boundaries insert Quantize ops.
>
> A key ONE-source constraint showed up: mixed precision boundaries are not free. ONE inserts or rewrites Quantize nodes for dtype mismatches, and special handling exists only for certain ops like Transpose, FC, Mul, Div, Add, and BatchMatMul. So the tool should recommend contiguous islands, not isolated single nodes.
>
> The config name matching is exact: ONE walks active graph nodes and requires every configured `name` to match a Circle node name. I’m checking the available stock introspection tools so the helper can extract those names without importing private compiler code.
>
> `circle-inspect` gives useful summaries, but not enough structure for qconfig generation. `circledump` is the stock tool that can expose tensor/operator names; I’m checking its output shape and whether we can parse it robustly from Python.
>
> There’s a generated Python Circle schema already in the ONE tree, so the tool can parse `.circle` directly rather than scraping `circledump` text. That should make the candidate list much more reliable.
>
> I’m going to implement this as a two-stage helper: first it reads the Circle model and scores candidate nodes using ONE mixed-precision behavior; then it writes grouped qconfig proposals like “top-k int16 islands” that you can pass back to stock `one-quantize --quant_config`.
>
> Good, `circledump` has enough structure too, but the schema parser is still better for automation. I’m checking the generated enum style now, then I’ll write the tool.
>
> For the ranking heuristic, I’m using two buckets from ONE source behavior: “quality-sensitive candidates” like Softmax/BatchMatMul/Conv/mask heads, and “boundary-risk ops” where isolated int16 can create extra Quantize conversions. The tool will prefer contiguous islands and output both individual and island configs.
>
> I’m going to give the tool a direct Circle reader and also an optional `--ampq-config`/`--visq-json` path. That way it can either make a heuristic proposal from graph structure alone, or refine the ranking using stock AMPQ/VISQ artifacts when you have them.
>
> I’m going to emit several proposal styles, because the “right” mixed precision split is empirical: top-ranked nodes, small local islands, and depth front/back splits matching ONE AMPQ’s bisection idea. You can then quantize/evaluate each proposal with the stock sweep runner.
>
> <details><summary>Created a file</summary>
>
> - Wrote `./tools/online/suggest_one_mixed_precision_qconfig.py` (+645 -0)
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> I’ll make this as a stock-ONE helper around the existing `--quant_config` path: inspect the Circle graph, rank sensitive/expensive nodes, and emit candidate qconfig JSONs you can try with the sweep tool. I’m checking whether the script patch landed cleanly, then I’ll validate it on real Circle files.
>
> The new helper file is present. I’m going through it in chunks now to catch any syntax or logic problems before I add the usage doc and run it through real Circle parsing.
>
> One wrinkle: `git diff --stat` in ONE is showing many zero-line changes under broad directories, which usually means metadata or normalization noise rather than source content. I’ll verify the exact diff later so we do not accidentally carry compiler changes.
>
> The helper compiles cleanly. I also found the ONE noise is only executable-bit flips, so I’ll restore those metadata bits after the ASS-side tooling is validated.
>
> The script shape looks good. I’m going to tighten one small compatibility detail before testing: qconfig readers should accept both ONE’s `name` and `names` forms, since the compiler parser supports both.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./tools/online/suggest_one_mixed_precision_qconfig.py` (+9 -5)
>
> </details>
>
> That compatibility tweak is in. Next I’m running the helper against actual Circle models from the ONE checkout so the parser, node names, and generated qconfig format are exercised end to end.
>
> The first compile pass is clean. The Circle file scan and sample run are still executing; likely just walking a large build tree/importing schema, so I’m polling those now.
>
> The sample execution is taking longer than expected for a tiny model, so I’m checking the process state instead of waiting blindly. It may be stuck during flatbuffer import or model parsing.
>
> The helper successfully parsed a stock ONE Circle test and produced qconfig JSON using the exact output tensor/node name (`ofm`). I’m checking the existing sweep wrapper now so the generated qconfigs can be fed directly into calibration/quantization runs without hand-editing commands.
>
> Good news: the existing sweep wrapper already has `--quant-config`, so this new selector plugs into it directly. I’m waiting on one larger Circle sample now, then I’ll add the workflow doc.
>
> The selector worked on both a one-node BatchMatMul and a small FC+GELU graph, generating ranked CSVs and multiple qconfig proposals. I’m adding the markdown operation note now with the ONE-source assumptions and the exact commands to use.
>
> I’m also going to validate the handoff into `run_one_stock_quant_sweep.py` with a dry-run. That confirms the qconfig file appears in the actual stock `one-quantize` command line, including list-file calibration input.
>
> <details><summary>Created a file</summary>
>
> - Wrote `./docs/one_mixed_precision_selection_operation.md` (+130 -0)
>
> </details>
>
> The operation doc is in place. I’m doing the last validation pass now: compile the Python tools, dry-run the sweep with a generated qconfig and list-style calibration, then clean generated Python cache files.
>
> The dry-run shows the full stock command includes `--input_data_format list` and `--quant_config`, exactly what we need for list-file calibration plus mixed precision. I’m restoring the accidental ONE file mode flips now and then I’ll give you the final map of what was created.
>
> Small self-inflicted path mistake: I tried restoring ONE modes from the ASS working directory, so `chmod` looked in the wrong place and changed nothing. I’ll rerun it with absolute ONE paths.
>
> The ONE mode restoration can’t be applied from this user because those ONE files are root-owned. I’ll verify it is mode-only/no content diff so we can be clear that the new work remains ASS-side only.
>
> One more robustness pass: the selector currently knows the local Python 3.12 schema path. I’m making that path discovery a little more tolerant, so it works if the ONE venv moves to another Python minor version.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./tools/online/suggest_one_mixed_precision_qconfig.py` (+13 -5)
>
> </details>
>
> The schema path discovery is now environment-aware (`ONE_ROOT`, `ONE_CMDS`, or `ONE_CIRCLE_SCHEMA_SITE`) and still defaults to the local ONE build. I’m rerunning the quick parser test after that tweak.
</details>

Created the mixed-precision selection tool and operation doc.

Main new tool: [suggest_one_mixed_precision_qconfig.py](./tools/online/suggest_one_mixed_precision_qconfig.py)

It parses a `.circle` model with ONE’s Circle schema, ranks nodes for `int16` promotion, and emits stock ONE `--quant_config` JSONs:

- `nodes.csv`: ranked nodes with op type, depth, rough compute, boundary risk, reasons
- `qconfig_best_island*_int16.json`: small contiguous int16 islands, recommended first
- `qconfig_depth_*_int16.json`: AMPQ-style front/back depth splits
- `qconfig_top*_int16.json`: diagnostic top-ranked nodes

Usage doc: [one_mixed_precision_selection_operation.md](./docs/one_mixed_precision_selection_operation.md)

Example:

```bash
python tools/online/suggest_one_mixed_precision_qconfig.py \
  --circle logs/one/model.opt.circle \
  --out-dir logs/one_mixed_precision_suggestions/model \
  --prefer-regex 'mask|head|decoder|sfc|attn|softmax|out'
```

Then test one generated qconfig with the stock sweep wrapper:

```bash
python tools/online/run_one_stock_quant_sweep.py \
  --input-circle logs/one/model.opt.circle \
  --calib-data data/calibration_files.txt \
  --input-data-format list \
  --modes percentile \
  --quant-config logs/one_mixed_precision_suggestions/model/qconfig_best_island3_int16.json
```

Validation done:

- Python compile passed for the selector and sweep wrapper.
- Selector successfully parsed real ONE Circle samples and generated qconfigs.
- Dry-run confirmed `run_one_stock_quant_sweep.py` passes both `--input_data_format list` and `--quant_config` into stock `one-quantize`.

I did not change ONE compiler source. I verified the relevant ONE diff is content-empty; there are root-owned mode-only diffs in the ONE checkout that this user cannot chmod back, but no insertions/deletions.

> Please make a review for these tools

<details><summary>6 previous messages</summary>

> I’ll review the two ASS-side helper scripts as code, not modify them yet: first for correctness against stock ONE behavior, then for CLI edge cases and failure modes. I’ll pull line-numbered snippets so any findings are concrete.
>
> I have the broad structure. Now I’m checking the command construction and qconfig generation details against ONE’s actual CLI, because small flag mismatches here would waste full quantization runs.
>
> I found one important selection bug already: the “depth_back_25” proposal currently means “everything after 25 percent depth,” not “last 25 percent.” That can silently create a much larger int16 region than intended, so I’m checking for a couple more of these before I report.
>
> There is another practical mismatch I’m checking: `--quant_config` is relevant to the normal quantization path, while ONE’s AMPQ path computes its own config. If the sweep mixes `--modes ampq` with an externally generated qconfig, the command may not mean what the user expects.
>
> Confirmed: the sweep can currently print an AMPQ command with `--quant_config`, but ONE’s `one-quantize` branches into `_ampq_solve` and does not feed that config to `circle-mpqsolver`. That is a review finding, not a code change yet.
>
> I have enough for the review, but I’m doing one clean compile/dry-run check so the residual risk section is anchored in something current.
</details>

**Findings**

1. **Medium: `depth_back_25` selects too much of the model.**
   [suggest_one_mixed_precision_qconfig.py](./tools/online/suggest_one_mixed_precision_qconfig.py:411) uses `depth >= max_depth * fraction` for back splits, so `0.25` means “all nodes after 25% depth,” not “last 25%.” That can promote most of the graph to int16 and hurt latency. Use `depth >= max_depth * (1.0 - fraction)` for back-region proposals.

2. **Medium: `--quant-config` is misleading when combined with AMPQ mode.**
   [run_one_stock_quant_sweep.py](./tools/online/run_one_stock_quant_sweep.py:133) adds `--quant_config` to every run, including AMPQ specs from [line 197](./tools/online/run_one_stock_quant_sweep.py:197). But stock `one-quantize` branches directly to `_ampq_solve` when `--ampq` is set, and its AMPQ path has `# TODO Add config` at [one-quantize](~/works/ONE/compiler/one-cmds/one-quantize:860). So an AMPQ dry-run may show `--quant_config`, but the actual AMPQ solver will not apply that config. The wrapper should reject this combo or omit `--quant_config` for AMPQ runs.

3. **Medium: island qconfig files can overwrite each other.**
   [suggest_one_mixed_precision_qconfig.py](./tools/online/suggest_one_mixed_precision_qconfig.py:595) names files by actual island length, not requested size. If `3,5,8` all collapse to the same reachable/scored node count, they all write the same `qconfig_best_islandN_int16.json`. Use requested size in the filename, or skip duplicate layer sets.

4. **Low: `--quant-config` path is not validated early.**
   The sweep parser accepts it at [run_one_stock_quant_sweep.py](./tools/online/run_one_stock_quant_sweep.py:296), but the validation block only checks ONE binary, input circle, calib data, and test data. A typo would fail after launching runs. Add an existence check near [line 317](./tools/online/run_one_stock_quant_sweep.py:317).

5. **Low: ONE library path discovery is still tied to the default build.**
   `ONE_CMDS` is configurable at [run_one_stock_quant_sweep.py](./tools/online/run_one_stock_quant_sweep.py:16), but `load_lib_dirs()` hardcodes `/home/cmj/works/ONE/build/compiler` at [line 22](./tools/online/run_one_stock_quant_sweep.py:22). If testing another ONE build, the executable and libraries may mismatch.

**Validation**

I ran:

```bash
python -m py_compile tools/online/suggest_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py
```

and a dry-run confirmed list calibration plus qconfig command generation works for the percentile path. I did not run a full quantization pass in this review.

> does these findings be fixed?

<details><summary>9 previous messages</summary>

> Not yet. The previous turn was review-only, so the findings are still present. I’ll fix them now and re-run the dry checks so the tools match the review recommendations.
>
> I’m making the fixes narrowly: correct the depth-back cutoff, make island filenames non-overwriting, reject the misleading AMPQ+qconfig combo, validate qconfig paths early, and derive ONE library paths from the selected ONE build.
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `./tools/online/suggest_one_mixed_precision_qconfig.py` (+7 -2)
> - Wrote `./tools/online/run_one_stock_quant_sweep.py` (+10 -1)
>
> </details>
>
> The code edits are in and the tools still compile. I’m refreshing the docs now where the behavior changed, especially to warn that fixed qconfig testing must use percentile/moving_average, not AMPQ mode.
>
> The selector now generates separate `qconfig_best_island3/5/8` files even when the actual reachable island is smaller, so there is no overwrite. I’m cleaning the compile cache and adding the qconfig/AMPQ caveat to the docs.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./docs/one_mixed_precision_selection_operation.md` (+2 -0)
>
> </details>
>
> The doc now matches the wrapper behavior: fixed qconfig runs are separated from AMPQ search runs. I’ll run the specific negative/positive dry-runs next so we know the guard catches the misleading case while valid qconfig sweeps still work.
>
> The fixed guard behaves as intended: valid percentile+qconfig dry-run passes, AMPQ+qconfig is rejected before any run starts, missing qconfig is rejected early, and invalid depth fractions now get a CLI error. I’m doing one final compile/status check.
>
> One note on `git diff`: these tool/doc files are still untracked, so Git does not show a normal diff stat for them yet. The final status check is clean aside from the expected untracked ASS-side files.
</details>

Yes, fixed now.

I updated:

- [suggest_one_mixed_precision_qconfig.py](./tools/online/suggest_one_mixed_precision_qconfig.py)
  - fixed `depth_back_*` so `0.25` means last 25%, not everything after 25%
  - made `qconfig_best_island3/5/8_int16.json` filenames stable, no overwrite
  - added validation for positive island sizes and depth fractions in `(0, 1]`

- [run_one_stock_quant_sweep.py](./tools/online/run_one_stock_quant_sweep.py)
  - validates `--quant-config` exists before running
  - rejects `--quant-config` with `--modes ampq`, because stock ONE AMPQ does not consume external qconfig
  - derives ONE library path from `ONE_CMDS.parent` / `ONE_BUILD_COMPILER`, not a hardcoded build root

- [one_mixed_precision_selection_operation.md](./docs/one_mixed_precision_selection_operation.md)
  - added the warning to use fixed qconfigs only with `percentile` or `moving_average`, not AMPQ mode

Validation passed:

```bash
python -m py_compile tools/online/suggest_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py
```

Also tested:

- valid `percentile + --quant-config` dry-run passes
- `ampq + --quant-config` is rejected early
- missing qconfig is rejected early
- invalid depth fractions are rejected
- AMPQ without qconfig still dry-runs correctly

No ONE compiler source changes were made.

> Current it is still difficult to make a suitable mixed quantization config. Could you check the ONE compiler source code, especially about the quantization. After you get a deep understand of the ONE quantization, you may get some useful insights for making a better tool/scripts for propasaling a mixed quantization config(reduce the loss of uint8). I already get an idea for this: make a script that calculate the zero point/scales of int16 and uint8 for each nodes of the pytorch model with some calibration data, then evaluate the uint8 error based on int16. After got these loss for each nodes, we could choose the nodes with less loss as uint8. It may need use forward hook during the pytorch model forward pass. And also need make a mapping of the nodes between pytorch model and circle model. Anyway this is just an idea, you could learn from it. But I think you may get better hints from the ONE compiler's source code. The mission is build a generic tool that could find out a best mixed quantization configuration for ONE compiler.

<details><summary>20 previous messages</summary>

> Context compaction
>
> Source: automatic
> Status: completed
>
> I’ll trace the stock ONE quantization path end to end, then compare it with the current ASS-side helpers and build the smallest useful next tool around measured activation error. The key question is whether the ranking should be based on raw tensor quantization error, downstream output sensitivity, or ONE’s actual inserted `Quantize/Dequantize` boundaries; I’ll verify that from source before editing.
>
> <details><summary>Explored a search</summary>
>
> - Searched for `one-quantize|AMPQ|quant_config|mixed precision|mixed quant|record-minmax|suggest_one_mixed|run_one_stock` in `MEMORY.md`
>
> </details>
>
> The prior helper review confirms one important constraint in this checkout: stock AMPQ searches its own qconfig and does not consume an external `--quant_config`, so the new scorer must emit fixed configs for the normal percentile/moving-average path and keep AMPQ as a separate baseline. I’m now reading the actual quantizer, min/max recorder, and solver code to identify the calibration semantics and where node-level error is meaningful.
>
> <details><summary>Explored 43 files, 9 searches, 4 lists</summary>
>
> - Read `./tools/online/suggest_one_mixed_precision_qconfig.py`
> - Read `./tools/online/run_one_stock_quant_sweep.py`
> - Searched for `quant_config|default_quantization_dtype|layers_info|QuantizeWithMinMax|InsertQuantizeOpOnDTypeMismatch|quantization_dtype|granularity|MinMax|record-minmax|moving_average|percentile|bisection_type|i16_back` in `circle-quantizer`
> - Listed files in `online`
> - Searched for `torch.onnx|register_forward_hook|forward hook|calib|activation|quant` in `tools`
> - Read `./compiler/record-minmax/README.md`
> - Read `./compiler/one-cmds/one-cmds/one_quantize.py`
> - Read `./compiler/circle-mpqsolver/src/core/Dumper.cpp`
> - Read `./compiler/circle-mpqsolver/src/bisection/BisectionSolver.cpp`
> - Read `./compiler/circle-quantizer/src/CircleQuantizer.cpp`
> - Read `./compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp`
> - Read `./compiler/luci/pass/src/InsertQuantizeOpOnDTypeMismatch.cpp`
> - Searched for `quant_config|layers_info|default_quantization_dtype|granularity|QuantizeWithMinMax|InsertQuantizeOpOnDTypeMismatch|Quantization` in `circle-quantizer`
> - Read `./compiler/circle-quantizer/src/CircleQuantizer.cpp`
> - Read `./compiler/circle-quantizer/src/CircleQuantizer.cpp`
> - Read `./compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp`
> - Read `./compiler/luci/pass/src/InsertQuantizeOpOnDTypeMismatch.cpp`
> - Listed files in `record-minmax`
> - Searched for `(README|\.cpp$|\.h$|\.py$)`
> - Searched for `class.*MinMax|record-minmax|moving_average|percentile|quant_config|input_data_format|ampq|_ampq_solve` in `one-cmds`
> - Searched for `quant_config|input_data_format|def main|ampq|moving_average|percentile` in `one-cmds`
> - Searched for `quantize_with_minmax|layers_info|LayerInfoMap|quantization_dtype|dtype\(\)|Quantize` in `pass`
> - Read `./compiler/circle-mpqsolver/src/core/Quantizer.cpp`
> - Read `./compiler/circle-mpqsolver/src/core/Evaluator.cpp`
> - Read `./compiler/circle-mpqsolver/src/bisection/BisectionSolver.cpp`
> - Read `./compiler/record-minmax/README.md`
> - Listed files in `record-minmax`
> - Read `./compiler/record-minmax/include/MinMaxComputer.h`
> - Read `./compiler/record-minmax/src/MinMaxComputer.cpp`
> - Read `./compiler/record-minmax/include/MinMaxObserver.h`
> - Read `./compiler/record-minmax/src/MinMaxObserver.cpp`
> - Read `./compiler/circle-mpqsolver/src/core/Quantizer.cpp`
> - Read `./compiler/circle-mpqsolver/src/core/Evaluator.cpp`
> - Searched for `MAEMetric|DatasetEvaluator|qerror|quantize\(` in `core`
> - Read `./compiler/record-minmax/src/RecordMinMax.cpp`
> - Read `./compiler/record-minmax/src/ListFileIterator.cpp`
> - Read `./compiler/record-minmax/src/HDF5Iterator.cpp`
> - Read `./compiler/record-minmax/include/RecordFunction.h`
> - Read `./compiler/record-minmax/src/RecordFunction.cpp`
> - Listed files in `one-cmds`
> - Searched for `quant_config|input_data_format|ampq|moving_average|percentile` in `one-cmds`
> - Read `./compiler/record-minmax/src/MinMaxComputer.cpp`
> - Read `./compiler/record-minmax/src/MinMaxObserver.cpp`
> - Read `./compiler/record-minmax/src/RecordMinMax.cpp`
> - Read `./compiler/record-minmax/src/ListFileIterator.cpp`
> - Read `./compiler/record-minmax/include/RecordFunction.h`
> - Read `./compiler/one-cmds/one-quantize`
> - Read `./compiler/circle-mpqsolver/src/core/Quantizer.cpp`
> - Read `./compiler/circle-mpqsolver/src/core/Evaluator.cpp`
> - Read `./compiler/circle-mpqsolver/src/core/ErrorMetric.cpp`
> - Read `./compiler/one-cmds/one-quantize`
> - Read `./compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp`
> - Read `./compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp`
> - Read `./compiler/luci/pass/src/InsertQuantizeOpOnDTypeMismatch.cpp`
> - Read `./compiler/record-minmax/src/MinMaxObserver.cpp`
> - Read `./compiler/record-minmax/src/MinMaxComputer.cpp`
>
> </details>
>
> The source gives a stronger direction than the current heuristic selector. ONE calibrates each activation tensor independently from the observed output range, then applies dtype-specific affine quantization; mixed precision is implemented by assigning `int16` to selected Circle node names while the default remains `uint8`, with explicit Q/DQ boundaries inserted where neighboring dtypes differ. That means a useful selector should score each node using the same calibration range and quantizer math, then account for boundary conversions and downstream sensitivity.
>
> <details><summary>Explored 30 files, 23 searches, 3 lists, ran 6 commands</summary>
>
> <details><summary>Explored 20 files, 14 searches, a list</summary>
>
> - Read `./compiler/record-minmax/src/MinMaxComputer.cpp`
> - Read `./compiler/record-minmax/include/MinMaxComputer.h`
> - Read `./compiler/record-minmax/include/RecordFunction.h`
> - Read `./docs/one_mixed_precision_selection_operation.md`
> - Read `./tools/online/run_one_stock_quant_sweep.py`
> - Searched for `nudge|quantparam|zero_point|zero_point|scale|quantize.*uint8|int16|QuantizeWithMinMax` in `pass`
> - Searched for `class Quantizer|CircleQuantizer|quantize_with_minmax|InsertQuantizeOpOnDTypeMismatch` in `luci`
> - Searched for `QuantizeOpOnDTypeMismatch|DTypeMismatch|int16.*uint8|uint8.*int16` in `pass`
> - Searched for `quantization_params|quantparam|scale|zerop` in `pass`
> - Searched for `QuantizeWithMinMaxPass|InsertQuantizeOpOnDTypeMismatch|LayerInfoMap|quantizeWithMinMax|quantize_with_minmax` in `luci`
> - Searched for `nudged|nudge|quantparam.*scale|quantparam.*zerop|quantization.*range|zero.*point|min.*max` in `compiler`
> - Read `./compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp`
> - Searched for `QuantizeWithMinMax` in `pass`
> - Searched for `*Quantize*` in `pass`
> - Read `./compiler/circle-quantizer/src/CircleQuantizer.cpp`
> - Read `./compiler/luci/pass/src/QuantizeWeightsOnly.cpp`
> - Searched for `compute_sym_scale|compute.*scale|nudged_min|nudged_max` in `luci`
> - Read `./compiler/luci/pass/src/QuantizationUtils.cpp`
> - Searched for `QuantizeWithMinMaxPass|quantize_with_minmax|LayerInfoMap` in `pass`
> - Read `./compiler/luci/pass/src/InsertQuantizeOpOnDTypeMismatch.cpp`
> - Read `./compiler/circle-mpqsolver/src/core/Quantizer.cpp`
> - Read `./compiler/circle-mpqsolver/README.md`
> - Read `./compiler/circle-mpqsolver/src/core/Dumper.cpp`
> - Read `./compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp`
> - Read `./compiler/circle-quantizer/src/CircleQuantizer.cpp`
> - Read `./compiler/luci/pass/src/helpers/LayerInfoMap.cpp`
> - Read `./compiler/luci/pass/include/luci/helpers/LayerInfoMap.h`
> - Read `./compiler/luci/pass/include/luci/Pass/QuantizeWithMinMaxPass.h`
> - Searched for `torch.onnx.export|dynamo_export|onnx.export|onnx` in `tools`
> - Listed files in `tools`
> - Searched for `class .*Dataset|DataLoader|calib|calibration|input.*list|h5` in `online`
> - Read `./pyproject.toml`
> - Read `./requirements.txt`
> - Searched for `python` in `.venv`
> - Searched for `*.onnx` in `logs`
>
> </details>
>
> - Ran `python - <<'PY'
> from pathlib import Path
> import onnx
> p=Path('logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.onnx')
> m=onnx.load(str(p))
> for i,n in enumerate(m.graph.node[:80]):
>  print(i, n.op_type, repr(n.name), '->', list(n.output))
> PY`
>
> <details><summary>Explored 10 files, 2 searches</summary>
>
> - Searched for `*.txt` in `logs`
> - Read `./compiler/record-minmax/src/ListFileIterator.cpp`
> - Read `./compiler/record-minmax/include/ListFileIterator.h`
> - Read `./compiler/record-minmax/include/DataSetIterator.h`
> - Read `./compiler/record-minmax/src/DirectoryIterator.cpp`
> - Read `./compiler/record-minmax/include/DirectoryIterator.h`
> - Read `./compiler/record-minmax/include/DataBuffer.h`
> - Read `./compiler/one-cmds/one-quantize`
> - Read `./tools/online/run_one_stock_quant_sweep.py`
> - Read `./tools/online/run_one_stock_quant_sweep.py`
> - Searched for `*.log` in `logs`
> - Read `./tools/online/run_one_stock_quant_sweep.py`
>
> </details>
>
> - Ran `python - <<'PY'
> import sys
> sys.path.insert(0,'/home/cmj/works/ONE/build/compiler/one-cmds/venv/lib/python3.12/site-packages')
> from circle_schema.v0_9.circle.Model import Model
> from circle_schema.v0_9.circle.BuiltinOperator import BuiltinOperator
> m=Model.GetRootAsModel('logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle'.encode() if False else open('logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle','rb').read(),0)
> rev={v:k for k,v in vars(BuiltinOperator).items() if isinstance(v,int)}
> g=m.Subgraphs(0)
> for i in range(min(50,g.OperatorsLength())):
>  op=g.Operators(i); code=rev.get(m.OperatorCodes(op.Opcode()).BuiltinCode(), '?')
>  ins=[g.Tensors(op.Inputs(j)).Name().decode(errors='replace') for j in range(op.InputsLength()) if op.Inputs(j)>=0]
>  outs=[g.Tensors(op.Outputs(j)).Name().decode(errors='replace') for j in range(op.OutputsLength())]
>  print(i,code,outs, 'in',ins)
> PY`
>
> <details><summary>Explored a file, a search</summary>
>
> - Searched for `operator_code|Opcode|Inputs\(` in `suggest_one_mixed_precision_qconfig.py`
> - Read `./tools/online/suggest_one_mixed_precision_qconfig.py`
>
> </details>
>
> - Ran `python - <<'PY'
> import sys
> sys.path.insert(0,'/home/cmj/works/ONE/build/compiler/one-cmds/venv/lib/python3.12/site-packages')
> from circle_schema.v0_9.circle.Model import Model
> from circle_schema.v0_9.circle.BuiltinOperator import BuiltinOperator
> raw=open('logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle','rb').read();m=Model.GetRootAsModel(raw,0)
> rev={v:k for k,v in vars(BuiltinOperator).items() if isinstance(v,int)};g=m.Subgraphs(0)
> for i in range(min(35,g.OperatorsLength())):
>  op=g.Operators(i); code=rev.get(m.OperatorCodes(op.OpcodeIndex()).BuiltinCode(), '?')
>  outs=[g.Tensors(op.Outputs(j)).Name().decode(errors='replace') for j in range(op.OutputsLength())]
>  print(i,code,outs)
> PY`
>
> - Ran `python - <<'PY'
> import sys
> sys.path.insert(0,'/home/cmj/works/ONE/build/compiler/one-cmds/venv/lib/python3.12/site-packages')
> from circle_schema.v0_9.circle.Model import Model
> from circle_schema.v0_9.circle.QuantizationParameters import QuantizationParameters
> print([x for x in dir(QuantizationParameters) if 'Min' in x or 'Max' in x or 'Scale' in x or 'Zero' in x or 'Quant' in x])
> from circle_schema.v0_9.circle.Tensor import Tensor
> print([x for x in dir(Tensor) if 'Quant' in x or 'Name' in x])
> PY`
>
> <details><summary>Explored 8 files, 4 searches, a list</summary>
>
> - Searched for `predefined.*qparam|PREDEFINED|predefined` in `pass`
> - Searched for `activation_qtype|make_predefined_qparam|PreDefinedSoftmax|PreDefinedLogistic|PreDefinedTanh` in `luci`
> - Read `./compiler/luci/pass/include/QuantizeActivation.h`
> - Read `./compiler/luci/pass/src/QuantizeActivation.cpp`
> - Read `./compiler/luci/pass/src/QuantizationUtils.cpp`
> - Read `./compiler/luci/pass/src/QuantizeActivation.cpp`
> - Read `./tools/online/suggest_one_mixed_precision_qconfig.py`
> - Read `./tools/online/suggest_one_mixed_precision_qconfig.py`
> - Searched for `record-minmax` in `ONE`
> - Searched for `record-minmax` in `ONE`
> - Listed files in `compiler`
> - Read `./compiler/record-minmax/driver/Driver.cpp`
> - Read `./compiler/record-minmax/src/RecordMinMax.cpp`
>
> </details>
>
> - Ran `python - <<'PY'
> for mod in ('onnx','onnxruntime','torch','numpy'):
>  try:
>   m=__import__(mod); print(mod, getattr(m,'__version__','ok'))
>  except Exception as e: print(mod, type(e).__name__, e)
> PY
> .venv/bin/python - <<'PY'
> for mod in ('onnx','onnxruntime','torch','numpy'):
>  try:
>   m=__import__(mod); print('venv',mod, getattr(m,'__version__','ok'))
>  except Exception as e: print('venv',mod, type(e).__name__, e)
> PY`
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./tools/online/run_one_stock_quant_sweep.py`
> - Read `./tools/online/suggest_one_mixed_precision_qconfig.py`
>
> </details>
>
> - Ran `python - <<'PY'
> import onnx
> p='logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.onnx'
> m=onnx.load(p)
> for i in m.graph.input: print(i.name, i.type.tensor_type.elem_type, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
> for o in m.graph.output: print('out',o.name,[d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
> PY
> python - <<'PY'
> import sys
> sys.path.insert(0,'/home/cmj/works/ONE/build/compiler/one-cmds/venv/lib/python3.12/site-packages')
> from circle_schema.v0_9.circle.Model import Model
> raw=open('logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle','rb').read();m=Model.GetRootAsModel(raw,0);g=m.Subgraphs(0)
> for i in range(g.InputsLength()):
>  t=g.Tensors(g.Inputs(i)); print('in',i,t.Name(),[t.Shape(j) for j in range(t.ShapeLength())],t.Type())
> PY`
>
> <details><summary>Explored 2 searches, a list</summary>
>
> - Searched for `*.bin` in `ASS`
> - Listed files in `logs`
> - Searched for `calib|input|data|list`
>
> </details>
>
> </details>
>
> I’m going to add two layers of tooling in ASS. The first computes ONE-compatible per-node `uint8` versus `int16` reconstruction error from the exported ONNX graph, while using stock `record-minmax` to obtain the exact Circle calibration ranges. The second performs a bounded greedy search by actually invoking stock `one-quantize` and measuring final output MSE, so the final qconfig is chosen by the compiler’s real behavior rather than by a heuristic alone.
>
> <details><summary>Created 2 files, explored a file, a search, ran a command</summary>
>
> - Wrote `./tools/online/analyze_one_mixed_precision_calibration.py` (+482 -0)
>
> <details><summary>Explored a file, a search</summary>
>
> - Searched for `def inspect_circle|def build_command|def main` in `run_one_stock_quant_sweep.py`
> - Read `./tools/online/run_one_stock_quant_sweep.py`
>
> </details>
>
> - Wrote `./tools/online/search_one_mixed_precision_qconfig.py` (+296 -0)
>
> - Ran `python -m py_compile tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py tools/online/suggest_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py`
>
> </details>
>
> The first implementation is in place and compiles. I found one integration detail while reviewing the generated command: stock `one-quantize` has a single `--input_data_format` value shared by calibration and test data, so I’m tightening the search tool to reject mismatched formats instead of silently applying the test format to both inputs.
>
> <details><summary>Edited a file, ran 4 commands</summary>
>
> - Wrote `./tools/online/search_one_mixed_precision_qconfig.py` (+5 -2)
> - Ran `python tools/online/search_one_mixed_precision_qconfig.py --help | head -100`
> - Ran `python tools/online/analyze_one_mixed_precision_calibration.py --help | head -80`
> - Ran `git status --short -- tools/online docs | sed -n '1,120p'`
> - Ran `mkdir -p /tmp/one_mp_calib_inputs && truncate -s 8200 /tmp/one_mp_calib_inputs/x.bin && truncate -s 163840 /tmp/one_mp_calib_inputs/state0.bin && truncate -s 163840 /tmp/one_mp_calib_inputs/state1.bin && truncate -s 163840 /tmp/one_mp_calib_inputs/state2.bin && truncate -s 163840 /tmp/one_mp_calib_inputs/state3.bin && truncate -s 163840 /tmp/one_mp_calib_inputs/state4.bin && truncate -s 163840 /tmp/one_mp_calib_inputs/state5.bin && truncate -s 163840 /tmp/one_mp_calib_inputs/state6.bin && truncate -s 163840 /tmp/one_mp_calib_inputs/state7.bin && printf '%s\n' '/tmp/one_mp_calib_inputs/x.bin /tmp/one_mp_calib_inputs/state0.bin /tmp/one_mp_calib_inputs/state1.bin /tmp/one_mp_calib_inputs/state2.bin /tmp/one_mp_calib_inputs/state3.bin /tmp/one_mp_calib_inputs/state4.bin /tmp/one_mp_calib_inputs/state5.bin /tmp/one_mp_calib_inputs/state6.bin /tmp/one_mp_calib_inputs/state7.bin' > /tmp/one_mp_calib_inputs/list.txt`
>
> </details>
>
> The dry command checks pass. I’m running the new analyzer once on an existing ASS ONNX/Circle pair with a one-record raw list to validate the real stock `record-minmax` invocation, Circle-to-ONNX name matching, qparameter math, and qconfig emission together. This is a plumbing check, not a quality result yet.
>
> <details><summary>Edited a file, explored a file, 3 searches, a list, ran 7 commands</summary>
>
> - Ran `python tools/online/analyze_one_mixed_precision_calibration.py --circle logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle --onnx logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.onnx --calib-data /tmp/one_mp_calib_inputs/list.txt --out-dir /tmp/one_mp_analyzer_check --max-samples 1 --max-values-per-node 1000 --top-k 8`
>
> <details><summary>Explored 2 searches, a list</summary>
>
> - Searched for `libluci_profile.so*` in `compiler`
> - Listed files in `compiler`
> - Searched for `profile|interpreter|dio-hdf5|luci`
>
> </details>
>
> - Ran `LD_LIBRARY_PATH=/home/cmj/works/ONE/build/compiler/luci/profile:/home/cmj/works/ONE/build/compiler/luci/import:/home/cmj/works/ONE/build/compiler/luci/export:/home/cmj/works/ONE/build/compiler/luci/pass:/home/cmj/works/ONE/build/compiler/luci/service:/home/cmj/works/ONE/build/compiler/luci-interpreter/src:/home/cmj/works/ONE/build/compiler/loco /home/cmj/works/ONE/build/compiler/record-minmax/record-minmax --help | head -20`
>
> - Wrote `./tools/online/analyze_one_mixed_precision_calibration.py` (+2 -8)
>
> - Ran `python tools/online/analyze_one_mixed_precision_calibration.py --circle logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle --onnx logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.onnx --calib-data /tmp/one_mp_calib_inputs/list.txt --out-dir /tmp/one_mp_analyzer_check --max-samples 1 --max-values-per-node 1000 --top-k 8`
>
> - Ran `truncate -s 40960 /tmp/one_mp_calib_inputs/state0.bin && truncate -s 40960 /tmp/one_mp_calib_inputs/state1.bin && truncate -s 40960 /tmp/one_mp_calib_inputs/state2.bin && truncate -s 40960 /tmp/one_mp_calib_inputs/state3.bin && truncate -s 40960 /tmp/one_mp_calib_inputs/state4.bin && truncate -s 40960 /tmp/one_mp_calib_inputs/state5.bin && truncate -s 40960 /tmp/one_mp_calib_inputs/state6.bin && truncate -s 40960 /tmp/one_mp_calib_inputs/state7.bin && python tools/online/analyze_one_mixed_precision_calibration.py --circle logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle --onnx logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.onnx --calib-data /tmp/one_mp_calib_inputs/list.txt --out-dir /tmp/one_mp_analyzer_check --max-samples 1 --max-values-per-node 1000 --top-k 8`
>
> - Ran `head -12 /tmp/one_mp_analyzer_check/nodes.csv; sed -n '1,100p' /tmp/one_mp_analyzer_check/qconfig_calibration_top8_int16.json; cat /tmp/one_mp_analyzer_check/summary.json`
>
> - Ran `python tools/online/search_one_mixed_precision_qconfig.py --circle logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle --calib-data /tmp/one_mp_calib_inputs/list.txt --test-data /tmp/one_mp_calib_inputs/list.txt --candidate-csv /tmp/one_mp_analyzer_check/nodes.csv --out-dir /tmp/one_mp_search_check --max-candidates 2 --max-int16 1 --dry-run`
>
> - Ran `python tools/online/search_one_mixed_precision_qconfig.py --circle logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle --calib-data /tmp/one_mp_calib_inputs/list.txt --test-data /tmp/one_mp_calib_inputs/list.txt --candidate-csv /tmp/one_mp_analyzer_check/nodes.csv --out-dir /tmp/one_mp_search_actual --max-candidates 1 --max-int16 1 --timeout 300`
>
> <details><summary>Explored a file, a search</summary>
>
> - Read `./compiler/one-cmds/one-create-quant-dataset`
> - Searched for `input_data_format|list|output_path|data_list` in `one-create-quant-dataset`
>
> </details>
>
> </details>
>
> The real run exposed a stock ONE limitation that the wrapper must handle: `circle-eval-diff` in this build does not evaluate a list file, even though `record-minmax` and `one-quantize` accept lists. I’m adapting the search tool to call stock `one-create-quant-dataset` and package list inputs into H5 before final MSE evaluation. This keeps list-file calibration support while preserving the compiler’s own evaluator.
>
> <details><summary>Edited 2 files, explored 2 files, a search, ran 6 commands</summary>
>
> - Wrote `./tools/online/search_one_mixed_precision_qconfig.py` (+34 -0)
>
> - Ran `python -m py_compile tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py && python tools/online/search_one_mixed_precision_qconfig.py --circle logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle --calib-data /tmp/one_mp_calib_inputs/list.txt --test-data /tmp/one_mp_calib_inputs/list.txt --candidate-csv /tmp/one_mp_analyzer_check/nodes.csv --out-dir /tmp/one_mp_search_actual2 --max-candidates 1 --max-int16 1 --timeout 300`
>
> <details><summary>Explored 2 files, a search</summary>
>
> - Read `/tmp/one_mp_search_actual2/search_summary.json`
> - Searched for `MSE for|Full|error|selected` in `*.log`
> - Read `/tmp/one_mp_search_actual2/qconfig_best_greedy_int16.json`
>
> </details>
>
> - Ran `ruff check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py`
>
> - Ran `.venv/bin/ruff check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py`
>
> - Ran `.venv/bin/ruff check --fix tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py`
>
> - Wrote `./tools/online/analyze_one_mixed_precision_calibration.py` (+27 -8)
> - Wrote `./tools/online/search_one_mixed_precision_qconfig.py` (+29 -7)
>
> - Ran `.venv/bin/ruff check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py && python -m py_compile tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py`
>
> - Wrote `./tools/online/analyze_one_mixed_precision_calibration.py` (+2 -1)
>
> - Ran `.venv/bin/ruff check --fix tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py; .venv/bin/ruff check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py; python -m py_compile tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py`
>
> </details>
>
> The closed-loop test now works end to end: a list file is packaged with stock `one-create-quant-dataset`, stock `one-quantize` produces the baseline and candidate models, and stock `circle-eval-diff` reports per-output MSE. In the smoke test, the analyzer’s top local candidate actually worsened final MSE, which is exactly why the greedy final-output search is necessary; I’m documenting that distinction so the local score is not mistaken for a deployment-quality result.
>
> <details><summary>Created a file, edited a file</summary>
>
> - Wrote `./docs/one_mixed_precision_quantization_source_and_search.md` (+192 -0)
> - Wrote `./tools/online/analyze_one_mixed_precision_calibration.py` (+2 -2)
>
> </details>
>
> The source-backed workflow is now documented, including the special fixed qparams for logistic/tanh/softmax and the fact that ONE’s AMPQ path is independent of external qconfigs. I’m doing the final verification pass now: lint/compile, a recorded-Circle analyzer rerun, qconfig JSON validation, and negative checks for invalid list/evaluator combinations.
>
> <details><summary>Explored a file, a search, ran 6 commands</summary>
>
> - Ran `.venv/bin/ruff check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py && python -m py_compile tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py tools/online/suggest_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py`
>
> - Ran `python tools/online/analyze_one_mixed_precision_calibration.py --circle logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle --onnx logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.onnx --calib-data /tmp/one_mp_calib_inputs/list.txt --recorded-circle /tmp/one_mp_analyzer_check/recorded.minmax.circle --out-dir /tmp/one_mp_analyzer_recheck --max-samples 1 --max-values-per-node 1000 --top-k 4 && python - <<'PY'
> import json
> from pathlib import Path
> p=Path('/tmp/one_mp_analyzer_recheck/qconfig_calibration_top4_int16.json')
> data=json.loads(p.read_text())
> assert data['default_quantization_dtype']=='uint8'
> assert len(data['layers'])==4
> assert all(x['dtype']=='int16' for x in data['layers'])
> print('qconfig validation: ok')
> PY`
>
> - Ran `python tools/online/search_one_mixed_precision_qconfig.py --circle logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle --calib-data /tmp/one_mp_calib_inputs/list.txt --test-data /tmp/one_mp_calib_inputs/list.txt --test-data-format h5 --candidate-csv /tmp/one_mp_analyzer_check/nodes.csv --out-dir /tmp/one_mp_search_bad --max-candidates 1 --max-int16 1 --dry-run; test $? -eq 1`
>
> - Ran `python tools/online/search_one_mixed_precision_qconfig.py --circle logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle --calib-data /tmp/one_mp_calib_inputs/list.txt --test-data /tmp/one_mp_calib_inputs/list.txt --candidate-csv /tmp/one_mp_analyzer_check/nodes.csv --out-dir /tmp/one_mp_search_recheck --max-candidates 1 --max-int16 1 --dry-run && rg -n -- '--input_data_format list|one-create-quant-dataset' /tmp/one_mp_search_recheck/*`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `def main|def qparams|def run_record_minmax|def package_list|def make_command|--calib-data|qconfig_best` in `analyze_one_mixed_precision_calibration.py`
>
> </details>
>
> - Ran `python - <<'PY'
> from pathlib import Path
> import json
> for p in [Path('/tmp/one_mp_analyzer_recheck/summary.json'), Path('/tmp/one_mp_search_actual2/search_summary.json')]:
>  data=json.loads(p.read_text())
>  print(p.name, 'ok', sorted(data)[:6])
> PY`
>
> - Ran `git status --short -- tools/online docs | sed -n '1,120p'; wc -l tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py docs/one_mixed_precision_quantization_source_and_search.md`
>
> <details><summary>Explored 2 files</summary>
>
> - Read `~/.codex/memories/MEMORY.md`
> - Read `~/.codex/memories/MEMORY.md`
>
> </details>
>
> </details>
>
> <details><summary>Plan</summary>
>
> Lint, compile, analyzer rerun, qconfig validation, list packaging, one-quantize/circle-eval-diff smoke search, and invalid-format checks all passed as expected.
>
> - [x] Inspect relevant memory, ASS tooling, and ONE quantization source/tests
> - [x] Design a stock-ONE-compatible per-node mixed-precision scoring workflow
> - [x] Implement the tool/scripts and documentation in ASS only
> - [x] Run focused validation and review the resulting workflow
>
> </details>
</details>

Implemented a source-aligned mixed-precision workflow without modifying ONE:

- [Calibration analyzer](<./tools/online/analyze_one_mixed_precision_calibration.py>)
  - Uses stock `record-minmax` for Circle ranges.
  - Collects ONNX intermediate activations with ONNX Runtime.
  - Reproduces ONE uint8/int16 scale, zero-point, clipping, and MSE behavior.
  - Maps ONNX outputs to Circle node names.
  - Emits `nodes.csv` and ranked qconfigs.

- [Closed-loop qconfig search](<./tools/online/search_one_mixed_precision_qconfig.py>)
  - Tests candidates using real stock `one-quantize`.
  - Measures final output MSE using `circle-eval-diff`.
  - Supports list calibration/test files.
  - Automatically packages lists to H5 using stock `one-create-quant-dataset`.
  - Produces `qconfig_best_greedy_int16.json`.

- [Source analysis and operation guide](<./docs/one_mixed_precision_quantization_source_and_search.md>)

Recommended workflow:

```bash
python tools/online/analyze_one_mixed_precision_calibration.py \
  --circle model.opt.circle \
  --onnx model.onnx \
  --calib-data calibration.list.txt \
  --out-dir logs/mp_analysis/model \
  --top-k 24
```

```bash
python tools/online/search_one_mixed_precision_qconfig.py \
  --circle model.opt.circle \
  --calib-data calibration.list.txt \
  --test-data test.list.txt \
  --candidate-csv logs/mp_analysis/model/nodes.csv \
  --out-dir logs/mp_search/model \
  --max-candidates 24 \
  --max-int16 8
```

Validation passed: lint, compilation, list-to-H5 packaging, stock quantization, Circle evaluation, and qconfig JSON validation. The analyzer is only a candidate prefilter; the final greedy search uses ONE’s actual output MSE.

> Please review the tools carefully

<details><summary>7 previous messages</summary>

> I’ll review the live ASS tools as a code review: first static correctness and CLI semantics, then the quantization math/mapping against the current ONE source, and finally real smoke paths including list calibration and stock evaluation. I’ll report findings first by severity, and I’ll leave files unchanged unless a defect is discovered and you explicitly want fixes.
>
> <details><summary>Explored a search, ran 3 commands</summary>
>
> - Ran `rg -n "TODO|FIXME|pass$|except Exception|subprocess|record|minmax|qparams|mapping|candidate|objective|selected|input_data_format|test_data_format|quant_config|ampq" tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py tools/online/suggest_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py`
>
> - Ran `git status --short -- tools/online docs; wc -l tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py tools/online/suggest_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py docs/one_mixed_precision_quantization_source_and_search.md`
>
> - Ran `.venv/bin/ruff check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py tools/online/suggest_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `one-quantize|AMPQ|quant_config|record-minmax|analyze_one_mixed|search_one_mixed` in `MEMORY.md`
>
> </details>
>
> </details>
>
> The initial review shows no lint failures, but static cleanliness is not enough here: the main risks are semantic, especially whether the analyzer’s local quantization model matches ONE’s actual node behavior and whether the greedy search can mis-rank candidates under its current objective. I’m tracing those paths line by line against the live source now.
>
> <details><summary>Explored 14 files, 7 searches, ran a command</summary>
>
> <details><summary>Explored 26 files, 4 searches</summary>
>
> - Read `./tools/online/analyze_one_mixed_precision_calibration.py`
> - Read `./tools/online/search_one_mixed_precision_qconfig.py`
> - Read `./tools/online/search_one_mixed_precision_qconfig.py`
> - Read `./tools/online/analyze_one_mixed_precision_calibration.py`
> - Read `./tools/online/run_one_stock_quant_sweep.py`
> - Read `./tools/online/suggest_one_mixed_precision_qconfig.py`
> - Read `./tools/online/search_one_mixed_precision_qconfig.py`
> - Read `./tools/online/analyze_one_mixed_precision_calibration.py`
> - Read `./tools/online/search_one_mixed_precision_qconfig.py`
> - Read `./tools/online/analyze_one_mixed_precision_calibration.py`
> - Read `./tools/online/analyze_one_mixed_precision_calibration.py`
> - Read `./tools/online/analyze_one_mixed_precision_calibration.py`
> - Read `./tools/online/search_one_mixed_precision_qconfig.py`
> - Read `./tools/online/search_one_mixed_precision_qconfig.py`
> - Read `./tools/online/search_one_mixed_precision_qconfig.py`
> - Read `./tools/online/search_one_mixed_precision_qconfig.py`
> - Searched for `Quantize.*S16|S16.*Quantize|quantize.*int16|int16.*quantize|U8.*S16` in `runtime`
> - Searched for `32767|32768|S16.*quant|quant.*S16|CircleQuantize|Quantize.*input` in `luci-interpreter`
> - Searched for `compute_sym_scale|compute_asym_scale_zp|QuantizationUtils` in `*test*`
> - Read `./compiler/luci-interpreter/src/kernels/Quantize.cpp`
> - Read `./compiler/luci-interpreter/pal/linux/PALQuantize.h`
> - Read `./compiler/luci-interpreter/src/kernels/Quantize.test.cpp`
> - Read `./compiler/luci/pass/src/QuantizeActivation.cpp`
> - Read `./compiler/luci/pass/src/QuantizationUtils.cpp`
> - Read `./compiler/luci-interpreter/src/kernels/TestUtils.cpp`
> - Read `./compiler/luci-interpreter/src/kernels/Quantize.h`
> - Searched for `AffineQuantize|kMin.*S16|INT16_MIN|numeric_limits<int16|S16` in `luci-interpreter`
> - Read `./compiler/luci-interpreter/pal/linux/PALQuantize.h`
> - Read `./compiler/luci-interpreter/src/kernels/Quantize.test.cpp`
> - Read `./compiler/luci-interpreter/src/kernels/Quantize.test.cpp`
>
> </details>
>
> - Ran `sed -n '80,150p' compiler/luci-interpreter/src/kernels/Quantize.test.cpp; rg -n "template.*AffineQuantize|AffineQuantize" /home/cmj/works/ONE/externals/TENSORFLOW-2.19.0 /home/cmj/works/ONE/externals/TENSORFLOW-2.19.0-EIGEN 2>/dev/null | head -50`
>
> <details><summary>Explored 6 files, 3 searches</summary>
>
> - Searched for `void InsertQuantizeOp|insert_out_quantize|create_in_quantize|if \(.*dtype|quantize\(` in `InsertQuantizeOpOnDTypeMismatch.cpp`
> - Read `./compiler/luci/pass/src/InsertQuantizeOpOnDTypeMismatch.cpp`
> - Read `./compiler/luci/pass/src/InsertQuantizeOpOnDTypeMismatch.cpp`
> - Searched for `InsertQuantizeOpOnDTypeMismatch|insert_quantize|QuantizeWithMinMaxPass` in `pass`
> - Read `./compiler/circle-quantizer/src/CircleQuantizer.cpp`
> - Read `./compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp`
> - Read `./compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp`
> - Read `./compiler/circle-quantizer/src/CircleQuantizer.cpp`
> - Searched for `InsertQuantizeOpOnDTypeMismatch|QuantizeOnnxFake|QuantizeWithMinMaxPass` in `circle-quantizer`
>
> </details>
>
> </details>
>
> A few concrete risks are emerging. The most important are not syntax errors: the search currently optimizes the mean MSE across all outputs, which can be wrong for multi-output streaming models, and the analyzer’s bounded activation cache takes the prefix of the first sample rather than a representative sample across the calibration set. I’m validating these against actual ONE outputs before assigning severity.
>
> <details><summary>Explored 6 files, 8 searches, ran 6 commands</summary>
>
> <details><summary>Explored a search</summary>
>
> - Searched for `max-values-per-node|first|bounded|local MSE|multi-output|mse_mean|latency-weight|boundary-weight` in `one_mixed_precision_quantization_source_and_search.md`
>
> </details>
>
> - Ran `python - <<'PY'
> import csv, json
> from pathlib import Path
> p=Path('/tmp/one_mp_analyzer_check/nodes.csv')
> rows=list(csv.DictReader(p.read_text().splitlines()))
> print('rows',len(rows),'unique mapped',len({r['mapped_output'] for r in rows}))
> print('top',rows[0]['name'],rows[0]['mapped_output'])
> PY`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `MSE for|mse_mean|mse_primary|mse_by_output` in `online`
>
> </details>
>
> - Ran `python - <<'PY'
> import sys
> sys.path.insert(0,'/home/cmj/works/ONE/build/compiler/one-cmds/venv/lib/python3.12/site-packages')
> from circle_schema.v0_9.circle.Model import Model
> for path in ['/tmp/one_mp_analyzer_check/recorded.minmax.circle','/tmp/one_mp_search_actual2/r01_n0127.q.circle']:
>  raw=open(path,'rb').read();m=Model.GetRootAs(raw,0);g=m.Subgraphs(0)
>  found=[]
>  for i in range(g.TensorsLength()):
>   t=g.Tensors(i); n=t.Name()
>   if n and b'/freq_mix/freq_mix.0/conv_4/Conv' in n:
>    q=t.Quantization(); found.append((n.decode(), None if q is None else (q.Min(0) if q.MinLength() else None,q.Max(0) if q.MaxLength() else None,q.Scale(0) if q.ScaleLength() else None,q.ZeroPoint(0) if q.ZeroPointLength() else None),t.Type()))
>  print(path,found)
> PY`
>
> <details><summary>Explored 2 files, 3 searches</summary>
>
> - Read `./compiler/luci/pass/src/helpers/LayerInfoMap.cpp`
> - Read `./compiler/luci/pass/src/helpers/LayerInfoMap.h`
> - Searched for `layer_info_map|info_by_name` in `QuantizeWithMinMaxPass.cpp`
> - Searched for `FusedActFunc|fusedActivationFunction|ActivationQType::IntScale|CircleOpcode::FLOOR|CircleOpcode::CEIL` in `QuantizationUtils.cpp`
> - Searched for `mse_mean|mse_primary|Mean Average|MSE` in `one_stock_quant_calibration_operation.md`
>
> </details>
>
> - Ran `python - <<'PY'
> import sys
> sys.path.insert(0,'tools/online')
> import analyze_one_mixed_precision_calibration as a
> import numpy as np
> print('numpy_round', np.round(np.array([-.5,.5,1.5])))
> print('one_expected_std_round', [round(-.5), round(.5), round(1.5)])
> values=np.arange(100000,dtype=np.float32)
> # Reproduce retention policy at one output over 4 records.
> chunks=[]; size=0; limit=20000
> for sample in range(4):
>  rem=limit-size
>  if rem>0:
>   chunks.append(values[sample*100000:(sample+1)*100000][:rem]); size += min(rem,100000)
> print('retained', chunks[0][0], chunks[-1][-1], 'count',size)
> PY`
>
> - Ran `rg -n "AffineQuantize" /home/cmj/works/ONE/externals /home/cmj/works/ONE/build/externals 2>/dev/null | head -80`
>
> - Ran `sed -n '45,105p' /home/cmj/works/ONE/externals/TENSORFLOW-2.19.0/tensorflow/lite/kernels/quantize.cc; rg -n "AffineQuantize" /home/cmj/works/ONE/externals/TENSORFLOW-2.19.0/tensorflow/lite/kernels/internal | head -30; sed -n '860,940p' /home/cmj/works/ONE/externals/TENSORFLOW-2.19.0/tensorflow/lite/kernels/internal/reference/reference_ops.h 2>/dev/null`
>
> <details><summary>Explored 5 files, 3 searches</summary>
>
> - Read `./externals/TENSORFLOW-2.19.0/tensorflow/lite/kernels/internal/reference/quantize.h`
> - Read `./externals/TENSORFLOW-2.19.0/tensorflow/lite/kernels/internal/optimized/optimized_ops.h`
> - Searched for `TfLiteRound` in `internal`
> - Searched for `RoundToNearest` in `internal`
> - Read `./externals/TENSORFLOW-2.19.0/tensorflow/lite/kernels/internal/cppmath.h`
> - Read `./externals/TENSORFLOW-2.19.0/tensorflow/lite/kernels/internal/optimized/optimized_ops.h`
> - Read `./tools/online/suggest_one_mixed_precision_qconfig.py`
> - Searched for `CircleLogSoftmax|CircleRmsNorm|CircleInstanceNorm|CircleFullyConnected|CircleSoftmax|CircleBatchMatMul|void visit\(luci::CircleNode` in `QuantizeWithMinMaxPass.cpp`
>
> </details>
>
> - Ran `if test -f /tmp/one_mpq_fix_check/nodes.csv; then head -3 /tmp/one_mpq_fix_check/nodes.csv; python tools/online/search_one_mixed_precision_qconfig.py --circle /home/cmj/works/ONE/build/compiler/circlechef/tests/CircleBatchMatMul_000.circle --calib-data /tmp/one_mp_calib_inputs/list.txt --test-data /tmp/one_mp_calib_inputs/list.txt --candidate-csv /tmp/one_mpq_fix_check/nodes.csv --out-dir /tmp/search_old_csv_check --dry-run; echo rc=$?; fi`
>
> </details>
>
> <details><summary>Plan</summary>
>
> The review is complete. Static checks passed, and the prioritized findings are grounded in the live tool code and current ONE quantization/evaluator source.
>
> - [x] Inspect current tool files, worktree state, and prior ONE-specific review context
> - [x] Audit implementation, quantization math, mapping, and stock CLI behavior
> - [x] Run focused static and runtime checks
> - [x] Report prioritized findings, assumptions, and residual risks
>
> </details>
</details>

## Findings

1. High: the greedy search optimizes the wrong metric for streaming models.

[search_one_mixed_precision_qconfig.py:298](./tools/online/search_one_mixed_precision_qconfig.py:298) uses `mse_mean`, averaging the separation output and all state outputs equally. For the ASS streaming model, this can select a config that improves state MSE while degrading the separated audio output.

The existing sweep tool already exposes `mse_primary` and `mse_by_output`. The search should support an explicit primary-output or weighted-output objective.

2. High: calibration activation samples are not representative.

[analyze_one_mixed_precision_calibration.py:317](./tools/online/analyze_one_mixed_precision_calibration.py:317) retains only the first `max-values-per-node` values. With the default 20,000 values, large tensors use only the first part of the first calibration sample; later samples are ignored.

This can strongly bias local MSE ranking by time/frequency position. A reservoir sampler or per-record quota is needed.

3. High: local qerror does not cover all ONE activation qtypes.

[analyze_one_mixed_precision_calibration.py:257](./tools/online/analyze_one_mixed_precision_calibration.py:257) detects only explicit `LOGISTIC`, `TANH`, and `SOFTMAX` opcodes.

ONE also assigns predefined quantization to fused `TANH`, propagates qtypes through reshape/transpose/split-like nodes, and uses integer-scale handling for `FLOOR`/`CEIL`. This is implemented in [QuantizationUtils.cpp](~/works/ONE/compiler/luci/pass/src/QuantizationUtils.cpp:300). Therefore the analyzer can rank some nodes using incorrect scales.

4. Medium: the analyzer materializes every ONNX intermediate output.

[analyze_one_mixed_precision_calibration.py:155](./tools/online/analyze_one_mixed_precision_calibration.py:155) adds all node outputs to the ONNX graph, and [analyze_one_mixed_precision_calibration.py:320](./tools/online/analyze_one_mixed_precision_calibration.py:320) requests all of them simultaneously. Large audio models may use substantial memory or fail with OOM despite the bounded retained sample size.

5. Medium: the search is incompatible with the sibling selector’s CSV format.

The analyzer writes `eligible=true/false`, but `suggest_one_mixed_precision_qconfig.py` writes `eligible=1/0`. [search_one_mixed_precision_qconfig.py:98](./tools/online/search_one_mixed_precision_qconfig.py:98) accepts only `"true"`, so passing the existing selector’s `nodes.csv` rejects every candidate.

6. Medium: the boundary penalty is only a rough graph heuristic.

[search_one_mixed_precision_qconfig.py:116](./tools/online/search_one_mixed_precision_qconfig.py:116) counts graph edges, but ONE’s actual inserted quantization boundaries depend on operator type, dtype, qparams, constants, and redundant-quantize removal. It should not be treated as a latency estimate.

7. Low: local quantization math is not completely identical to runtime behavior.

`np.round` may differ from the backend’s `TfLiteRound` behavior at half-way values, and the analyzer clamps int16 to `[-32767, 32767]` while runtime affine quantization uses the native int16 range. This is unlikely to dominate normal results, but it contradicts the claim of exact local reconstruction.

## Positive Checks

Lint and Python compilation pass. The list-to-H5 packaging path works with stock `one-create-quant-dataset`, and the stock `one-quantize` plus `circle-eval-diff` path was exercised successfully. No ONE compiler files were changed.

The most urgent fixes are the output metric selection and representative activation sampling. Until those are addressed, the generated qconfig should be treated as an experiment candidate, not an automatically trustworthy configuration.

> Please review again and then fix the found issues

<details><summary>17 previous messages</summary>

> I’ll recheck the live implementations and then patch the issues as one coherent workflow. The main changes will be: make separation-output MSE the default search objective, sample activations across the full calibration set, obtain node qparams from stock full-uint8/full-int16 Circle models instead of partially reimplementing ONE’s qtype rules, and bound ONNX intermediate-output memory.
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> I’m grounding the fixes in the current scripts and the exact qparam behavior exposed by stock `circle-quantizer`. I’ll also keep backward compatibility with the older selector CSV, since that is part of the practical workflow rather than merely a format detail.
>
> <details><summary>Explored 4 files, 2 searches, ran 4 commands</summary>
>
> - Ran `git status --short && rg -n "def (collect_stats|qparams|load_candidates|evaluate|boundary_count)|mse_mean|min-mse-improvement|latency-weight|eligible|augment_onnx" tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py && rg -n "quantize_with_minmax|activation_qtype|IntegerScale|Propagate|FLOOR|GATHER|RESHAPE" /home/cmj/works/ONE/compiler/luci/pass/src/QuantizationUtils.cpp /home/cmj/works/ONE/compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp /home/cmj/works/ONE/compiler/one-cmds -g '*quantize*'`
>
> <details><summary>Explored 17 files, 2 searches</summary>
>
> - Read `./tools/online/analyze_one_mixed_precision_calibration.py`
> - Read `./tools/online/analyze_one_mixed_precision_calibration.py`
> - Read `./tools/online/analyze_one_mixed_precision_calibration.py`
> - Read `./tools/online/analyze_one_mixed_precision_calibration.py`
> - Read `./tools/online/analyze_one_mixed_precision_calibration.py`
> - Read `./tools/online/analyze_one_mixed_precision_calibration.py`
> - Read `./tools/online/search_one_mixed_precision_qconfig.py`
> - Read `./tools/online/search_one_mixed_precision_qconfig.py`
> - Read `./tools/online/search_one_mixed_precision_qconfig.py`
> - Searched for `class OpInfo|def read_circle|return ops|rough_ops|circle_quantizer` in `suggest_one_mixed_precision_qconfig.py`
> - Read `./tools/online/search_one_mixed_precision_qconfig.py`
> - Read `./tools/online/search_one_mixed_precision_qconfig.py`
> - Read `./tools/online/suggest_one_mixed_precision_qconfig.py`
> - Read `./tools/online/suggest_one_mixed_precision_qconfig.py`
> - Read `./tools/online/search_one_mixed_precision_qconfig.py`
> - Read `~/works/ONE/compiler/one-cmds/one-quantize`
> - Searched for `^def read_circle|^class OpInfo|^@dataclass` in `suggest_one_mixed_precision_qconfig.py`
> - Read `./tools/online/suggest_one_mixed_precision_qconfig.py`
> - Read `./tools/online/suggest_one_mixed_precision_qconfig.py`
>
> </details>
>
> - Ran `ls -l /home/cmj/works/ONE/build/compiler/circle-quantizer/circle-quantizer /home/cmj/works/ONE/build/compiler/one-cmds/circle-quantizer 2>/dev/null; /home/cmj/works/ONE/build/compiler/circle-quantizer/circle-quantizer --help 2>&1 | head -80; rg -n "QUANTIZE|DEQUANTIZE" /tmp/one_mp_search_actual2/*.log /tmp/one_mp_search_actual2/search_summary.json 2>/dev/null | head -30`
>
> - Ran `for f in /tmp/one_mp_search_actual2/*.q.circle; do echo "$f"; /home/cmj/works/ONE/build/compiler/circle-inspect/circle-inspect --operators "$f" 2>/dev/null | rg 'QUANTIZE|DEQUANTIZE|Operator Count' || true; done`
>
> - Ran `rg -n '^def parse_circle_ops|^def summarize_mse' tools/online/run_one_stock_quant_sweep.py && sed -n '110,155p' tools/online/run_one_stock_quant_sweep.py && /home/cmj/works/ONE/build/compiler/circle-inspect/circle-inspect --operators /tmp/one_mp_search_actual2/r01_n0127.q.circle | tail -30`
>
> <details><summary>Explored a file</summary>
>
> - Read `./tools/online/analyze_one_mixed_precision_calibration.py`
>
> </details>
>
> </details>
>
> The source check confirms that ONE’s activation qtype is not a simple min/max formula: it includes fixed ranges and forward/backward qparam propagation. I’m therefore switching the analyzer to read scales and zero points from two stock-quantized Circle models, which removes the most fragile duplicated compiler logic.
>
> <details><summary>Explored a file, a search, a list</summary>
>
> - Listed files in `tests`
> - Searched for `importlib.util|tools/online|pytest` in `tests`
> - Read `./docs/one_mixed_precision_quantization_source_and_search.md`
>
> </details>
>
> I’m applying the analyzer changes now. Besides compiler-derived qparams, the collector will use deterministic reservoir sampling and evaluate mapped outputs in small batches, so later calibration examples influence the score and large audio graphs do not expose every intermediate at once.
>
> <details><summary>Created 2 files, edited a file, explored a file, ran 5 commands</summary>
>
> - Wrote `./tools/online/analyze_one_mixed_precision_calibration.py` (+719 -0)
>
> <details><summary>Explored a file</summary>
>
> - Read `./tools/online/search_one_mixed_precision_qconfig.py`
>
> </details>
>
> - Wrote `./tools/online/search_one_mixed_precision_qconfig.py` (+578 -0)
>
> - Ran `.venv/bin/python -m py_compile tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py && .venv/bin/ruff check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py`
>
> - Wrote `./tools/online/analyze_one_mixed_precision_calibration.py` (+2 -3)
>
> - Ran `.venv/bin/ruff check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py && .venv/bin/python tools/online/analyze_one_mixed_precision_calibration.py --help >/tmp/analyzer_help.txt && .venv/bin/python tools/online/search_one_mixed_precision_qconfig.py --help >/tmp/search_help.txt && tail -12 /tmp/analyzer_help.txt && tail -18 /tmp/search_help.txt`
>
> - Wrote `./tools/online/analyze_one_mixed_precision_calibration.py` (+2 -2)
>
> - Ran `.venv/bin/ruff check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py && .venv/bin/python tools/online/analyze_one_mixed_precision_calibration.py --help >/tmp/analyzer_help.txt && .venv/bin/python tools/online/search_one_mixed_precision_qconfig.py --help >/tmp/search_help.txt && tail -12 /tmp/analyzer_help.txt && tail -18 /tmp/search_help.txt`
>
> - Ran `.venv/bin/ruff check --fix tools/online/analyze_one_mixed_precision_calibration.py && sed -n '8,24p' tools/online/analyze_one_mixed_precision_calibration.py && .venv/bin/ruff check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py`
>
> - Ran `.venv/bin/python tools/online/analyze_one_mixed_precision_calibration.py --help >/tmp/analyzer_help.txt && .venv/bin/python tools/online/search_one_mixed_precision_qconfig.py --help >/tmp/search_help.txt && tail -14 /tmp/analyzer_help.txt && tail -20 /tmp/search_help.txt`
>
> </details>
>
> The static checks are clean. I’m now running the revised analyzer on the existing exported separation graph, including real uint8 and int16 stock quantization; this is the important check for tensor-name preservation, qparam extraction, and batched ONNX execution.
>
> <details><summary>Explored a search, a list, ran 4 commands</summary>
>
> - Ran `rm -rf /tmp/one_mp_analyzer_fixed && .venv/bin/python tools/online/analyze_one_mixed_precision_calibration.py --circle logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle --onnx logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.onnx --calib-data /tmp/one_mp_calib_inputs/list.txt --recorded-circle /tmp/one_mp_analyzer_check/recorded.minmax.circle --out-dir /tmp/one_mp_analyzer_fixed --max-samples 1 --max-values-per-node 1000 --onnx-output-batch-size 64 --top-k 4`
>
> - Ran `cat /tmp/one_mp_analyzer_fixed/summary.json && head -5 /tmp/one_mp_analyzer_fixed/nodes.csv && .venv/bin/python - <<'PY'
> import csv
> from pathlib import Path
> rows=list(csv.DictReader(Path('/tmp/one_mp_analyzer_fixed/nodes.csv').open()))
> print('rows',len(rows),'sampled',sorted({r['values_sampled'] for r in rows})[:8])
> print('seen_gt_sampled',sum(int(r['values_seen']) > int(r['values_sampled']) for r in rows))
> print('u8 zero set',sorted({r['u8_zero_point'] for r in rows})[:12])
> print('i16 zero set',sorted({r['i16_zero_point'] for r in rows})[:12])
> PY
> /home/cmj/works/ONE/build/compiler/circle-inspect/circle-inspect --operators /tmp/one_mp_analyzer_fixed/full_int16.q.circle | rg -c '^QUANTIZE$|^DEQUANTIZE$'`
>
> - Ran `python - <<'PY'
> import csv
> from pathlib import Path
> for p in ['/tmp/one_mp_analyzer_check/nodes.csv','/tmp/one_mp_analyzer_fixed/nodes.csv']:
>  print(p)
>  rows={r['name']:r for r in csv.DictReader(Path(p).open())}
>  n='/conv_1/Conv;/act_1/Relu'
>  print({k:rows[n].get(k) for k in ['range_min','range_max','u8_scale','u8_mse','i16_mse','clip_u8','selection_score']})
> PY
> rg -n '/conv_1/Conv;/act_1/Relu' /tmp/one_mp_analyzer_check/nodes.csv | head -1`
>
> - Ran `rm -rf /tmp/one_mp_search_fixed && .venv/bin/python tools/online/search_one_mixed_precision_qconfig.py --circle logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle --calib-data /tmp/one_mp_calib_inputs/list.txt --test-data /tmp/one_mp_calib_inputs/list.txt --candidate-csv /tmp/one_mp_analyzer_fixed/nodes.csv --out-dir /tmp/one_mp_search_fixed --max-candidates 1 --max-int16 1 --conversion-weight 0.01`
>
> <details><summary>Explored a search, a list</summary>
>
> - Searched for `DEFAULT_CIRCLE_INSPECT|DEFAULT_ONE_QUANTIZE|ONE_CMDS` in `run_one_stock_quant_sweep.py`
> - Listed files in `circle-inspect`
>
> </details>
>
> </details>
>
> The real run exposed one more integration issue: the shared helper assumed `circle-inspect` had been symlinked into `build/compiler/one-cmds`, but this checkout only has the stock binary in `build/compiler/circle-inspect`. I’m fixing that default centrally so both the sweep and mixed search inspect the generated models without extra command-line flags.
>
> <details><summary>Edited 2 files, explored a file, a list, ran 2 commands</summary>
>
> <details><summary>Explored a file, a list</summary>
>
> - Read `./tools/online/run_one_stock_quant_sweep.py`
> - Listed files in `one-create-quant-dataset`
>
> </details>
>
> - Wrote `./tools/online/run_one_stock_quant_sweep.py` (+6 -1)
>
> - Ran `rm -rf /tmp/one_mp_search_fixed && .venv/bin/python tools/online/search_one_mixed_precision_qconfig.py --circle logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle --calib-data /tmp/one_mp_calib_inputs/list.txt --test-data /tmp/one_mp_calib_inputs/list.txt --candidate-csv /tmp/one_mp_analyzer_fixed/nodes.csv --out-dir /tmp/one_mp_search_fixed --max-candidates 1 --max-int16 1 --conversion-weight 0.01`
>
> - Ran `python - <<'PY'
> import json
> p='/tmp/one_mp_search_fixed/search_summary.json'
> d=json.load(open(p))
> print({k:d[k] for k in ['objective','baseline_quality','final_quality','selected_names']})
> for t in d['trials']:
>  print(t['tag'],t.get('quality'),t.get('mse_primary'),t.get('mse_mean'),t.get('conversion_count'),t.get('extra_conversion_count'),t.get('boundary_proxy'),t.get('objective_value'), {k:v for k,v in t.get('operators',{}).items() if k in ('QUANTIZE','DEQUANTIZE')})
> PY`
>
> - Wrote `./tools/online/search_one_mixed_precision_qconfig.py` (+9 -1)
>
> - Wrote `./tools/online/search_one_mixed_precision_qconfig.py` (+1 -0)
>
> </details>
>
> The stock end-to-end search now uses the separation output by default and reports two inserted `QUANTIZE` operators for the tested int16 node. I’m tightening the stopping rule so optional compute/conversion penalties can actually veto a quality gain, then adding regression tests for the sampling, legacy CSV parsing, objective selection, and no-gain behavior.
>
> <details><summary>Created a file, edited 3 files, explored 2 files, a search, ran 6 commands</summary>
>
> - Wrote `./tests/test_one_mixed_precision_tools.py` (+109 -0)
>
> - Ran `.venv/bin/ruff check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py tests/test_one_mixed_precision_tools.py && .venv/bin/pytest -q tests/test_one_mixed_precision_tools.py`
>
> - Ran `.venv/bin/ruff check --fix tests/test_one_mixed_precision_tools.py tools/online/run_one_stock_quant_sweep.py || true; sed -n '1,22p' tools/online/run_one_stock_quant_sweep.py; sed -n '168,186p' tools/online/run_one_stock_quant_sweep.py; sed -n '418,430p' tools/online/run_one_stock_quant_sweep.py`
>
> - Wrote `./tools/online/run_one_stock_quant_sweep.py` (+7 -4)
>
> - Ran `.venv/bin/ruff check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py tests/test_one_mixed_precision_tools.py && .venv/bin/pytest -q tests/test_one_mixed_precision_tools.py`
>
> - Ran `.venv/bin/python -m pytest -q tests/test_one_mixed_precision_tools.py`
>
> - Ran `sed -n '1,240p' docs/one_mixed_precision_quantization_source_and_search.md && git diff -- tools/online/run_one_stock_quant_sweep.py && git status --short`
>
> <details><summary>Explored a file</summary>
>
> - Read `./docs/one_mixed_precision_quantization_source_and_search.md`
>
> </details>
>
> - Wrote `./docs/one_mixed_precision_quantization_source_and_search.md` (+54 -19)
>
> <details><summary>Explored 4 files, a search</summary>
>
> - Read `~/works/ONE/compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp`
> - Read `~/works/ONE/compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp`
> - Searched for `InsertQuantizeOpOnDTypeMismatch` in `pass`
> - Read `~/works/ONE/compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp`
> - Read `~/works/ONE/compiler/luci/pass/src/QuantizeWithMinMaxPass.cpp`
>
> </details>
>
> - Wrote `./docs/one_mixed_precision_quantization_source_and_search.md` (+3 -3)
>
> - Wrote `./tools/online/search_one_mixed_precision_qconfig.py` (+3 -0)
>
> - Ran `wc -l /tmp/one_mp_calib_inputs/list.txt; sed -n '1,12p' /tmp/one_mp_calib_inputs/list.txt; git diff --check; .venv/bin/python -m py_compile tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py tests/test_one_mixed_precision_tools.py`
>
> </details>
>
> The focused tests pass, and the live analyzer/search smoke tests both complete with stock ONE binaries. I’m doing the final review pass now: checking the generated metadata, failure handling, documentation against the exact `QuantizeWithMinMaxPass` sequence, and the complete diff for accidental scope creep.
>
> <details><summary>Edited a file, explored a file, ran 6 commands</summary>
>
> - Ran `.venv/bin/ruff format --check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py tests/test_one_mixed_precision_tools.py; .venv/bin/ruff check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py tests/test_one_mixed_precision_tools.py; .venv/bin/python -m pytest -q tests/test_one_mixed_precision_tools.py; git diff --check; git status --short`
>
> - Ran `.venv/bin/ruff format tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py && .venv/bin/ruff check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py tests/test_one_mixed_precision_tools.py && git diff --check`
>
> - Ran `wc -l tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py tests/test_one_mixed_precision_tools.py docs/one_mixed_precision_quantization_source_and_search.md && git diff --stat && git diff -- tools/online/run_one_stock_quant_sweep.py && rg -n "mse_mean|boundary_weight|boundary-weight|qparams\(|augment_onnx|np\.round|first" tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py docs/one_mixed_precision_quantization_source_and_search.md`
>
> - Wrote `./tools/online/run_one_stock_quant_sweep.py` (+5 -9)
>
> <details><summary>Explored 2 files</summary>
>
> - Read `~/.codex/memories/MEMORY.md`
> - Read `~/.codex/memories/MEMORY.md`
>
> </details>
>
> - Ran `find logs -name 'nodes.csv' -o -name '*candidate*.csv' | head -30; find /tmp -maxdepth 3 -name '*.csv' 2>/dev/null | rg 'one_mp|mixed|qconfig' | head -30`
>
> - Ran `.venv/bin/python - <<'PY'
> from pathlib import Path
> import sys
> sys.path.insert(0, 'tools/online')
> from suggest_one_mixed_precision_qconfig import read_circle
> from search_one_mixed_precision_qconfig import load_candidates
> circle=Path('logs/npu_efficiency_audit/sfc_small_conv2d_bn_npu_kvsplit_20260716/stream_rawmask.circle')
> _,ops,_,_=read_circle(circle)
> for p in Path('logs/one_mixed_precision_suggestions').glob('*/nodes.csv'):
>     candidates, skipped=load_candidates(p, 24, ops)
>     print(p, 'loaded=',len(candidates),'skipped=',len(skipped), 'first=', candidates[0]['index'] if candidates else None)
> PY
> head -2 logs/one_mixed_precision_suggestions/sfc_small_conv2d_bn_npu_20260713_v2/nodes.csv`
>
> - Ran ``rg -n '^class PriorityReservoir|^def quantize_recorded|^def collect_stats|^def main|^def load_candidates|^def quality_value|^def improvement_is_sufficient|conversion-weight|DEFAULT_CIRCLE_INSPECT' tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py tools/online/run_one_stock_quant_sweep.py && rg -n '^## Validation|^## Tools|default objective|stock `circle-quantizer`' docs/one_mixed_precision_quantization_source_and_search.md && rg -n '^def test_' tests/test_one_mixed_precision_tools.py && .venv/bin/ruff check tools/online/analyze_one_mixed_precision_calibration.py tools/online/search_one_mixed_precision_qconfig.py tests/test_one_mixed_precision_tools.py && .venv/bin/python -m pytest -q tests/test_one_mixed_precision_tools.py && git diff --check``
>
> </details>
>
> <details><summary>Plan</summary>
>
> Re-review and fixes are complete; stock ONE smoke tests and focused regressions passed.
>
> - [x] Revalidate findings against live ASS and ONE source
> - [x] Patch analyzer sampling, qparam fidelity, mapping coverage, memory behavior
> - [x] Patch search objective, CSV compatibility, candidate validation, stopping rules
> - [x] Update docs and add focused tests
> - [x] Run static and end-to-end validation
>
> </details>
</details>

**Review Complete**

All identified issues are fixed:

- Analyzer now reads real uint8/int16 qparams from stock `circle-quantizer`, uses deterministic reservoir sampling across all calibration records, batches ONNX outputs, and reports mapping coverage in [analyze_one_mixed_precision_calibration.py](./tools/online/analyze_one_mixed_precision_calibration.py:134).
- Search defaults to primary separation-output MSE, supports named/mean objectives, validates legacy and current CSVs against the live Circle graph, counts actual conversion operators, and rejects insignificant gains in [search_one_mixed_precision_qconfig.py](./tools/online/search_one_mixed_precision_qconfig.py:125).
- Fixed the stock `circle-inspect` default path in [run_one_stock_quant_sweep.py](./tools/online/run_one_stock_quant_sweep.py:19).
- Updated the source analysis and usage guide in [one_mixed_precision_quantization_source_and_search.md](./docs/one_mixed_precision_quantization_source_and_search.md:81).
- Added focused regression coverage in [test_one_mixed_precision_tools.py](./tests/test_one_mixed_precision_tools.py:18).

Validation passed:

- Ruff checks clean.
- `6` focused tests passed.
- Real analyzer smoke test matched all `250/250` Circle operators.
- Real stock ONE search used primary MSE and detected the two inserted `QUANTIZE` operators.
- No ONE compiler source was modified.

Pytest only reported an existing `.pytest_cache` permission warning; it did not affect the tests.
