# Edge-Deployable State-of-the-Art Audio Source Separation

## Executive assessment

For your use case, the right target is not the absolute heaviest offline music separator. The best design space is the **performance-efficient frontier**: adaptive spectral compression, sparse/asymmetric frequency processing, and **global frequency modeling with cheaper local or bounded temporal modeling**. The literature and your repository both point in that direction. In practice, the strongest starting point is **SFC-CA plus a TF-Locoformer-style separator**, not the current pure NPU-first prototypes. SFC’s published results show that replacing band-split encoder/decoder blocks with **input-adaptive spectral feature compression** can improve separation quality while cutting parameters drastically; in the same paper, the **16.0M-parameter SFC-Locoformer medium** outperformed its stronger BS-Locoformer baseline, and the **5.8M-parameter small SFC-CA** nearly matched or beat much heavier band-split baselines on MUSDB18-HQ and also improved DnR results. That is the clearest evidence-backed path toward “SOTA-ish, but still moderate.” citeturn20view0turn22view1turn24view0turn40view0

If you need **strict causal realtime on an edge NPU**, the published anchor is still **Band-SCNet**: about **2.59M parameters**, **17.59 G/s FLOPs**, **92 ms latency**, and **7.79 dB average SDR** on MUSDB18-HQ. That is a strong realtime baseline, but it is still far behind the offline frontier. So the key conclusion is this: **do not expect today’s hard-causal, export-constrained conv-only models to be both true SOTA and easy edge deployments without additional tricks** such as adaptive compression, teacher distillation, smarter temporal sparsity, and a better encoder/decoder than the current static-band baselines. citeturn16view4turn28view0

My bottom-line recommendation is to build around **three concrete tracks**: a near-SOTA moderate-compute **SFC-RoFormer hybrid**, a causal **Sparse-SFCNet-RT** for actual edge deployment, and an **EdgeFusion-SFC distilled** model for the tightest NPU budget. Of those, the first is the highest-upside research direction; the second is the best direct fit to your currently implemented codebase; the third is the best production-minded fallback. citeturn20view0turn13view0turn8view2turn12view0

## What the recent literature says

### The offline frontier

The strongest offline music-separation family is still the **band-split RoFormer line**. **BS-RoFormer** reported **9.80 dB average SDR on MUSDB18-HQ without extra data** and also won the SDX23 music separation track with additional data. **Mel-Band RoFormer** then improved over BS-RoFormer on **vocals, drums, and other** by replacing heuristic non-overlapping bands with mel-structured overlapping bands. Public benchmark aggregators now list even higher frontier results, such as a **BS-RoFormer (L=12, overlap-add)** entry at **11.99 average SDR**, but those entries are not always apples-to-apples with the paper baselines and should be treated as frontier indicators rather than the cleanest scientific comparison. citeturn25view1turn25view2turn25view3turn25view4turn33view0

The most important efficiency-oriented paper in your design space is **SFC**. It argues that classic band-split compression is effective because of its inductive bias, but wasteful and not input-adaptive. Its proposed **SFC-CA** and **SFC-Mamba** replace per-band encoders/decoders with a single sequence-modeling compressor/decoder plus explicitly designed inductive bias. On MUSDB18-HQ, the **small SFC-CA model** reached **9.27 cSDR / 8.95 uSDR at 5.8M params**, improving over the **34.7M-parameter BS-Locoformer small** at **8.72 / 8.26**. The **medium SFC-CA model** reached **9.95 cSDR / 9.38 uSDR at 16.0M params**, improving over the **55.5M-parameter BS-Locoformer medium** at **9.42 / 8.79**; the authors explicitly state that this medium SFC-CA outperformed existing SOTA systems in their comparison protocol, while also noting that some prior baselines use different validation splits. On DnR, the same paper again showed that **SFC-CA** beat both band-split and SFC-Mamba variants, with the **16.0M SFC-Locoformer medium** reaching **12.2 SNR / 11.8 SISDR average**, above the **55.5M BS-Locoformer medium** at **11.9 / 11.6**. citeturn20view0turn22view0turn22view1turn23view0turn24view0

Another important paper for the efficiency frontier is **Moises-Light**. It is not a strict edge deployment paper, but it is a very strong Pareto-point study. The model integrates **RoPE bottleneck modeling**, **SCNet-inspired encoder/decoder changes**, and stronger training losses/augmentation. On MUSDB-HQ, its proposed model reached **9.96 average SDR** in its comparison table, above **BS-RoFormer’s 9.80** and **SCNet-L’s 9.69** in that paper’s benchmark. The catch is important: the paper describes the design as **about 5M parameters for a single-stem model**, so it is not a direct “single 4-stem edge model” solution. Still, its architectural lessons are highly relevant: **heavier encoder than decoder, better split/merge blocks, RoPE bottleneck, and stronger training losses** matter a lot. citeturn18view2turn18view1turn27view1turn27view4

### The realtime and moderate-compute frontier

For strict realtime music separation, the best evidence-backed anchor I found is **Band-SCNet**. It was explicitly designed as a **causal realtime lightweight model**, combining SCNet-style sparse compression with **Cross-band Blocks**, **Narrow-band Blocks**, and a **CSA fusion module**. Its reported result was **7.79 dB average SDR**, **2.59M parameters**, **17.59 G/s FLOPs**, **92 ms latency**, and **0.478 realtime factor**, outperforming its own **Online SCNet** baseline at **7.09 dB** and **4.36M parameters**. This is the clearest published proof that sparse frequency compression plus lightweight cross-/narrow-band modeling is viable for realtime MSS, but it also quantifies the cost of causality: there is still a large gap versus the offline frontier. citeturn16view4turn28view0turn28view1

**SCNet** itself remains one of the strongest efficiency baselines. Its abstract reports **9.0 dB SDR on MUSDB18-HQ without extra data**, and specifically notes that its **CPU inference time is only 48% of HT Demucs**. That is why SCNet keeps showing up as the “efficient but strong” reference point in later papers like Band-SCNet and Moises-Light. citeturn31view0turn31view3

There are also useful signs from more experimental efficient families. A **two-stage Band-Split Mamba-2** model reported **8.71 uSDR** at **35.52M params / 212.11 G/s**, while its lighter version reached **8.19 uSDR** at **27.71M / 107.95 G/s**. Those numbers are respectable, but they are still heavier than what I would call “moderate edge-feasible” for 44.1 kHz deployment, and they do not beat the best SFC-style parameter/performance tradeoffs. citeturn26view0turn26view2turn26view3

### Transferable ideas from adjacent source-separation work

Two additional literature signals are especially important for your proposals. First, a 2025 efficiency study on top-performing vocal RoFormer models found that **time-axis attention is highly localized**, while **frequency-axis attention remains meaningfully global**. Replacing full temporal attention with **windowed sink attention** reduced attention computations by **44.5×** on 8-second inputs while largely preserving performance before finetuning, which strongly suggests that **global frequency + local or bounded temporal modeling** is the right compromise for edge-oriented music separation. citeturn29view0

Second, the speech-separation literature keeps reinforcing the same principle from another angle: **MossFormer2** combines transformer-like global modeling with recurrent-pattern modeling and reports improvements over earlier SOTA speech separators; **SPMamba** replaces heavier sequence modules with Mamba-style state-space modeling and reports superior performance with lower computational burden; and **TISDiSS** introduces inference-time scaling, letting the same discriminative separator trade compute for quality at test time. These are not directly music papers, but they are credible evidence that **hybrid global-local sequence models, SSMs, and repeated shared blocks** are the right ingredients when deployment efficiency matters. citeturn30search0turn30search3turn30search1

## What your ASS repository already has

Your selected repository is centered on the **SFC paper and its surrounding implementation ecosystem**. The root README states that the repository implements **SFC-CA**, **SFC-Mamba**, and a **Band-split module**; it supports the **TF-Locoformer separator**; it provides **pretrained TF-Locoformer weights for MUSDB18-HQ and DnR**; and it also contains an explicit **online/realtime edge-NPU track** with ONNX-oriented operator constraints. That matters because it means the SFC path in the repo is not just an idea—it is the only branch in the repository that is already tied to complete published results and pretrained checkpoints. citeturn40view0

The repo also documents a large online model zoo. The realtime section lists **plain SFC**, **soft-band**, **soft-band-query**, **crossattn-query**, **hierarchical-soft-band**, **soft-band-gru**, **soft-band-dilated**, and **tiger-npu-edge-v2** families. The README explicitly says to use **soft-band-dilated** when you want the **highest likely upside while staying in conv-style operators**, while **soft-band-query** and **crossattn-query** are described as the closest deployment-friendly approximations to the paper’s adaptive encoder/decoder path. That recommendation is consistent with the published SFC result: adaptive compression helps, and explicit query paths matter. citeturn13view0

The deployed NPU branch is intentionally harsh on operators. The repo states that the online path is meant for **batch=1**, **tensors of rank at most 4**, and mostly **Conv2d / torch.bmm / simple elementwise ops**, with a deploy-time convolution span constraint of **(kernel_size - 1) * dilation < 14**. The TF-MLP/TIGER edge README similarly emphasizes that the conservative export path removes attention and relies on **Conv2d-only mixers**, explicit caches, and frame-by-frame streaming semantics. These constraints are real deployment necessities, but they also explain why current NPU variants are unlikely to match full SFC-Locoformer quality without more architectural help. citeturn40view0turn8view3

Looking model by model, the most important findings are these:

The **BandSCNetNPU** folder is a serious causal design, not a toy. Its README defines it as a **causal, streaming, NPU-compatible 3-stem separator at 44.1 kHz** under a **192 KiB streaming-state budget**. The code path is built around a **sparse three-branch encoder/decoder**, repeated **CrossBand + NarrowBand** separation stages, optional **bounded causal attention**, and an optional **pooled channel mixer**. But the companion quality-tracking document is explicit that the actual evaluation results are still **TBD**, and its entries are framed as targets and hypotheses rather than finished numbers. In other words, if BandSCNetNPU currently sounds weak, the repo itself gives a strong reason: the design exists, but the evidence of trained performance is not yet there. citeturn8view0turn12view0turn11view0

The **DolphinSFC** branch is more interesting as an idea than as a final answer. Its README says it adapts **Dolphin**—an efficient audio-visual separator—to an **audio-only** setting by keeping the separator idea while replacing the video semantic path with an **audio-derived semantic/source prior**. The repo notes that the original Dolphin’s **coarse global attention** is approximated here by a longer causal depthwise Conv2d branch, and its **local heat-diffusion attention** is approximated by learned local causal depthwise smoothing. That makes DolphinSFC a useful source of **global-local separator design ideas**, but not the most evidence-backed route to top MSS performance on its own. citeturn8view1turn32view0

The **EdgeFusionNPU** branch is explicitly labeled a **deployment-first online separator candidate**, **not a trained checkpoint**. Its README says it fuses ideas from **BandSCNet/SCNet**, **TF-MLPNet**, **windowed-attention/Moises-Light style results**, **TIGER NPU lessons**, and **SFC external STFT handling**. The code-level description shows optional **conv memory vs. ssm-lite memory**, optional **band-token bottlenecks**, an optional **token-capacity block**, and presets that scale up to a **“large-v2-hybrid-5m”** class. That makes it an excellent scaffold for a production model, but not yet the strongest separation system in its current form. citeturn8view2turn34view0

The **TF-MLPNet/TIGER edge** path is a highly deployment-aware separation backbone. The README describes it as a **TF-MLPNet-style, NPU-oriented TIGER separator** with no attention, smaller streaming states, and frame-by-frame `forward_cell` behavior. Its core block consists of **frequency mixing**, **causal time mixing**, and a **single-frame global gate**, with grouped time-state widths constrained by the NPU span rule. This is a very sensible building block for streaming export, but on its own it is too compute-constrained to be my first choice for SOTA-chasing separation quality. citeturn8view3turn35view0turn35view1

## Why the current implementations are probably underperforming

The biggest reason is simple: **the strongest branch in the repo is the published SFC-Locoformer path; most of the other edge NPU branches are either prototypes, export refactors, or designs without finished benchmarked checkpoints**. The root README provides pretrained SFC-Locoformer weights, while the BandSCNetNPU quality document still lists results as blanks and EdgeFusionNPU says outright that it is not a trained checkpoint. So if your subjective listening says “SFC seems better than TIGER/BandSCNet/Dolphin/others,” that is exactly what the repo’s evidence would lead me to expect. citeturn40view0turn12view0turn8view2

The second reason is **encoder/decoder quality**. The SFC paper is unusually clear that the encoder/decoder is not a secondary detail. Its small **5.8M SFC-CA** system beat a **34.7M** band-split baseline, and its **16.0M SFC-CA** medium system beat a **55.5M** band-split medium system. The authors explicitly argue that improving the encoder/decoder, not just the separator, is a major future direction. Many of the lighter repo variants are still effectively variations on static compression or simpler routing, so they are leaving real quality on the table before the separator even starts. citeturn20view0turn22view0turn23view0turn24view0

The third reason is **temporal context under causality**. The literature directly quantifies the cliff: offline SCNet sits at **9.0 dB**, BS-RoFormer at **9.80 dB**, while realtime Band-SCNet is **7.79 dB**. Your NPU implementations then add stricter operator limits, bounded kernel spans, and tiny streaming state budgets. That is exactly the scenario in which pure conv replacements can lose the long-range temporal structure that the best offline systems exploit. citeturn31view0turn25view1turn16view4turn40view0

The fourth reason is that **the wrong part of attention is being approximated**. The 2025 windowed-attention study is very relevant here: it found that time attention is mostly local, but frequency attention remains usefully global. So if a lightweight model throws away almost all attention structure and replaces everything with local convs, it gives up too much of the RoFormer advantage. The better compromise is to keep **global frequency aggregation** and replace only the **expensive temporal full attention** with local windows, sink tokens, focused linear attention, or bounded conv/SSM memory. citeturn29view0

## Design principles that should guide the next model

The strongest principle is: **keep adaptive frequency compression**. Among all the papers I reviewed, SFC gives the most convincing evidence that **input-adaptive encoder/decoder compression** is a major lever for both performance and parameter efficiency. I would treat static band-split or purely fixed sparse compression as baselines, not endpoints. citeturn20view0turn22view0

The second principle is: **use SCNet-style sparse/asymmetric frequency processing around the separator**. SCNet, Band-SCNet, and Moises-Light all reinforce the same point: sparse or asymmetric encoder/decoder structure is not just a compute trick; it materially improves the quality/efficiency tradeoff. In your repo, this principle is already present in BandSCNetNPU and EdgeFusionNPU. citeturn31view0turn16view4turn18view0turn8view2

The third principle is: **separate global-frequency modeling from temporal-memory modeling**. For music and broad audio separation, harmonic and inter-band relationships need broad frequency communication; temporal modeling, however, can often be local, bounded, recurrent, or state-space based. The literature from SFC, TF-Locoformer, and the windowed attention study all supports this split. citeturn17view3turn24view1turn29view0

The fourth principle is: **distill from a stronger offline teacher into the causal student**. The repo’s edge-friendly branches are too constrained to discover all the right invariances from scratch. If the goal is “SOTA-like quality under moderate complexity,” distillation is not optional; it is the bridge between a strong offline teacher and a deployable causal student. This is an inference from the gap between offline SFC/RoFormer-style systems and realtime Band-SCNet-style systems, but it is strongly supported by the size of that published gap. citeturn20view0turn16view4turn25view1

## Proposed model structures

### SFC-RoFormer Lite

This is the proposal I would pursue first if the goal is **the best probable quality under moderate complexity**.

**High-level design.** Use an **SFC-CA encoder/decoder** with **64 adaptive bands** as the front/back end; wrap the separator in an **SCNet-style sparse asymmetric pyramid**; and use a **TF-Locoformer-style separator** where **frequency modeling remains global**, but **temporal attention is replaced by a windowed or focused-linear mechanism**. In other words: keep what seems to matter from RoFormer, keep what SFC proved about adaptive compression, and remove the most expensive part of full self-attention. This is exactly the literature intersection with the best evidence behind it. citeturn20view0turn22view0turn24view1turn29view0

**Why this should outperform your current repo candidates.** SFC already showed that adaptive compression alone beats heavier band-split baselines. TF-Locoformer was chosen by the SFC authors because it gave similar quality with roughly **2× training speed** relative to BS-RoFormer in their setup. Windowed-attention work then showed that time-axis attention in RoFormer-like separators is largely local, which means you can cut temporal attention cost heavily without destroying the frequency-global inductive bias. citeturn20view0turn24view1turn29view0

**Where I would target the budget.** I would aim for **8M to 16M total parameters** and a directional budget of roughly **25 to 60 GMac/s** for stereo 44.1 kHz inference. That is not a published number; it is my engineering target based on where **Band-SCNet** becomes comfortably realtime, where **TS-BSMAMBA2** becomes heavy, and where **SFC medium** already delivered a strong result. citeturn16view4turn26view0turn20view0

```python
import torch
import torch.nn as nn

class LocalTimeGlobalFreqBlock(nn.Module):
    """
    Global frequency modeling + cheap temporal modeling.
    Frequency path can stay attention-like; time path is windowed or linear.
    """
    def __init__(self, channels: int, time_kernel: int = 9, freq_heads: int = 4):
        super().__init__()
        self.freq_norm = nn.GroupNorm(1, channels)
        self.freq_qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.freq_out = nn.Conv2d(channels, channels, kernel_size=1)

        self.time_norm = nn.GroupNorm(1, channels)
        self.time_dw = nn.Conv2d(
            channels, channels, kernel_size=(time_kernel, 1),
            padding=(time_kernel // 2, 0), groups=channels
        )
        self.time_pw = nn.Conv2d(channels, channels, kernel_size=1)

        self.ffn = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        y = self.freq_norm(x)
        q, k, v = self.freq_qkv(y).chunk(3, dim=1)
        # Replace this with real frequency-only attention or a low-rank variant.
        y = self.freq_out(v) + x

        z = self.time_norm(y)
        z = self.time_pw(self.time_dw(z))
        z = y + z
        return z + self.ffn(z)


class SFCRoFormerLite(nn.Module):
    """
    SFC encoder/decoder + sparse pyramid + local-time/global-frequency separator.
    """
    def __init__(self, encoder, decoder, channels=128, depth=6):
        super().__init__()
        self.encoder = encoder      # SFC-CA encoder
        self.decoder = decoder      # SFC-CA decoder
        self.blocks = nn.ModuleList([LocalTimeGlobalFreqBlock(channels) for _ in range(depth)])

    def forward(self, x):
        z, query = self.encoder(x)
        for blk in self.blocks:
            z = blk(z)
        y, _ = self.decoder(z, query=query)
        return y
```

**Core value.** This proposal is the best shot at a model that is still recognizably on the **SOTA path**, rather than merely “good for an edge model.”

### Sparse-SFCNet RT

This is the proposal I would pursue first if the goal is **actual hard-causal edge deployment**.

**High-level design.** Start from your repo’s **BandSCNetNPU** because it is already shaped for the deployment constraints you care about: sparse low/mid/high pyramid, repeated **CrossBand/NarrowBand** stages, bounded causal attention, and explicit streaming state. Then replace the static concatenation bottleneck with an **SFC-CA compression/query path**, and add a **Dolphin-style global-local refinement branch** inside each stage. The separator stays causal and exportable, but the **frequency compression becomes adaptive** and the **local/global separation inside each stage becomes smarter**. citeturn8view0turn11view0turn8view1turn20view0

**Why this is the best direct fit to your codebase.** It reuses what is already implemented in the repo: sparse pyramid, bounded attention, streaming state contracts, and NPU-safe Conv2d blocks. It also addresses the biggest literature-backed weakness of current BandSCNet-like causal systems: the encoder/decoder compression is not yet as strong or adaptive as SFC, and the current quality doc still has no trained results filled in. citeturn8view0turn12view0turn20view0

**Where I would target the budget.** I would aim for **2.5M to 6M total parameters**, a state budget still near the current **192 KiB** envelope, and something like **15 to 25 GMac/s**. This is again my estimate, but it is anchored by published **Band-SCNet** at **2.59M / 17.59 G/s** and your repo’s stated **192 KiB** target. citeturn16view4turn8view0turn12view0

```python
class CausalCrossNarrowStage(nn.Module):
    """
    Causal stage for realtime deployment:
    cross-band conv -> narrow-band time memory -> pooled capacity -> local/global refiner
    """
    def __init__(self, channels: int):
        super().__init__()
        self.cross = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=1)
        self.time_dw = nn.Conv2d(channels, channels, kernel_size=(5, 1), groups=channels)
        self.time_pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.capacity = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(2 * channels, channels, kernel_size=1),
        )
        self.local_refine = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1, groups=channels)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        y = self.cross(x)
        y = self.time_pw(self.time_dw(y))
        y = y + self.capacity(y.mean(dim=-1, keepdim=True))
        y = y + self.local_refine(y)
        return x + self.out(y)


class SparseSFCNetRT(nn.Module):
    """
    Sparse pyramid + adaptive SFC bottleneck + causal separator.
    """
    def __init__(self, encoder, decoder, channels=48, num_stages=4):
        super().__init__()
        self.encoder = encoder      # sparse low/mid/high encoder
        self.sfc_bottleneck = nn.Conv2d(channels, channels, kernel_size=1)  # replace with real SFC-CA block
        self.stages = nn.ModuleList([CausalCrossNarrowStage(channels) for _ in range(num_stages)])
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        z = self.sfc_bottleneck(z)
        for stage in self.stages:
            z = stage(z)
        return self.decoder(z)
```

**Core value.** This proposal does not chase absolute leaderboard numbers; it chases the **best deployable quality per watt and per byte of streaming state**.

### Moises-SFC Unified

This is the proposal I would pursue if the goal is **one strong multi-stem model under moderate complexity without going fully causal**.

**High-level design.** Take the best lessons from **Moises-Light**—a heavier encoder than decoder, improved split/merge blocks, RoPE-style sequence modeling, and stronger training losses—but avoid its single-stem replication cost by making the separator **shared across stems with multi-head stem outputs**. Then replace the front-end static split with **SFC-CA** and give the low-frequency path slightly higher resolution or overlapping mel-informed bands so bass does not regress. The literature strongly suggests this will outperform the lighter SCNet-style baselines if trained well. citeturn18view0turn18view2turn25view3turn25view4

**Why this addresses a real weakness.** Moises-Light’s own discussion admits that stronger band resolution still matters for **bass**, and its strongest model is framed as a **single-stem** 5M class system. Your version should keep the architectural improvements, but convert them into a **unified multi-stem separator** to avoid four separate models. citeturn18view1turn18view2turn27view1

**Where I would target the budget.** I would aim for **6M to 10M total parameters** for a unified four-stem model, with chunked inference rather than frame-by-frame streaming. This is a design target, not a published result, but it follows naturally from the fact that SFC small already gets strong quality at **5.8M total**, while Moises-Light’s paper demonstrates that strong architectural choices can lift lighter models substantially. citeturn24view0turn18view2

```python
model_cfg = {
    "frontend": {
        "type": "sfc_ca",
        "n_bands": 64,
        "query": "adaptive",
        "low_band_overlap": True,
    },
    "encoder_decoder": {
        "type": "asymmetric_sparse_unet",
        "encoder_depth": 3,
        "decoder_depth": 1,
        "heavy_encoder": True,
    },
    "separator": {
        "type": "tf_locoformer_hybrid",
        "blocks": 4,
        "d_model": 96,
        "time_attention": "windowed",
        "freq_attention": "global",
    },
    "head": {
        "type": "multi_stem_mask_head",
        "n_src": 4,
    },
}
```

**Core value.** This is the most promising route if you want **one reasonably small model** that still sounds much closer to offline SOTA than a strict streaming separator.

### EdgeFusion-SFC Distilled

This is the proposal I would pursue if you must stay closest to your repo’s current NPU export path.

**High-level design.** Keep the **EdgeFusionNPU** operator discipline and its packed-state contract, but train it as a **student** of the stronger `SFC-RoFormer Lite` teacher. Turn on the **ssm_lite memory**, keep the **band bottleneck**, and use the **token-capacity** block only after aggressive frequency compression so you spend parameters where they are cheapest. This exactly matches the spirit of the existing EdgeFusion scaffold, but gives it a realistic path to useful quality. citeturn8view2turn34view0

**Why this makes sense.** EdgeFusionNPU is already scaffolded with presets up to a **5M hybrid** regime and explicitly fuses BandSCNet, TF-MLPNet, Moises-Light/windowed-attention ideas, and SFC-style external STFT handling. What it lacks is not architectural imagination; it lacks a teacher and a clear quality-oriented training plan. citeturn8view2turn34view0

**Core value.** This is the model I would expect to ship first on a hard NPU target, even if the final leaderboard leader remains Proposal A.

## Common training recipe that matters as much as the architecture

Training is not a side issue here. The SFC repo defaults to **12-second inference windows with 6-second overlap**, and the SFC paper also used longer inference segments than training because that improves results. Moises-Light improved its model substantially through **additional augmentation** and **multi-resolution complex spectrogram loss**. If you compare architectures without matching these details, you will draw the wrong conclusions. citeturn20view0turn40view0turn18view0

For the teacher model, I would train **Proposal A** on **MUSDB18-HQ** and **DnR**, using the repo’s existing data scripts and source-activity detection pipeline as the baseline. The loss stack should include **complex mask or RI reconstruction loss**, **multi-resolution STFT loss**, **time-domain SI-SDR or SNR term**, a **silent-source penalty**, and **mixture consistency** at the waveform reconstruction stage. For the student model, I would add **logit distillation**, **intermediate feature distillation** at the compressed-band level, and **stem-wise low-frequency weighting** so bass does not become the sacrificial stem. The need for the bass emphasis is supported indirectly by Mel-RoFormer, BS-RoFormer, and Moises-Light, all of which highlight the special importance of low-frequency/band design. citeturn25view1turn25view3turn18view1

A practical training plan would be:

- train **SFC-RoFormer Lite** first as the teacher;
- distill into **Sparse-SFCNet-RT** for causal streaming;
- distill again into **EdgeFusion-SFC Distilled** for the strictest export path.

That sequencing aligns with the empirical hierarchy in the literature: strong adaptive compression first, then sparse/causal adaptation, then the most constrained NPU student. citeturn20view0turn16view4turn8view2

## Final recommendation

If I had to choose only one architecture to invest in, I would choose **SFC-CA encoder/decoder + local-time/global-frequency TF-Locoformer separator + SCNet-style sparse asymmetric outer pyramid**. It is the one proposal most directly supported by the strongest evidence across the papers you asked about: **SFC** for adaptive compactness, **SCNet/Band-SCNet** for sparse frequency processing, **RoFormer/TF-Locoformer** for high-quality dual-path modeling, **windowed attention** for cutting the right part of the cost, and **Moises-Light** for encoder/decoder asymmetry and stronger training design. citeturn20view0turn31view0turn16view4turn24view1turn29view0turn18view0

If you need a sharper recommendation by objective:

- **Best chance of SOTA-like quality under moderate compute:** `SFC-RoFormer Lite`.
- **Best chance of strong realtime edge deployment from your current repo:** `Sparse-SFCNet RT`.
- **Best chance of getting something onto a strict NPU fastest:** `EdgeFusion-SFC Distilled`. citeturn20view0turn13view0turn8view2

What I would **not** do is spend the next cycle trying to tune pure conv-only TIGER/TF-MLP or pure BandSCNetNPU variants in isolation and hoping they magically close the gap to SFC or RoFormer-class systems. The evidence you already have says the gap is mostly architectural: **adaptive compression and better global-frequency modeling are the missing pieces**, not just more tuning. citeturn20view0turn16view4turn40view0

## Open questions and limitations

Cross-paper **FLOP/MAC** comparisons are not perfectly normalized. Some papers report **FLOPs per second of stereo music**, some emphasize latency or realtime factor, and some do not report compute at all. I therefore treat cross-paper complexity comparisons as **directional**, not exact. citeturn16view4turn26view0turn27view0

Several of the most relevant models in your repo are **not yet backed by completed trained checkpoints or reported benchmark tables**. The clearest cases are **BandSCNetNPU**, whose quality tracking file is still blank, and **EdgeFusionNPU**, whose README says it is not a trained checkpoint. So some of my conclusions about why those branches sound weak are necessarily based on the repo’s own status signals rather than finished benchmark numbers. citeturn12view0turn8view2

Finally, some public frontier numbers beyond the papers, such as benchmark-aggregator entries for very high-scoring BS-RoFormer or SCNet variants, are useful for orientation but are **not the cleanest scientific comparison layer**. I used them only as context for where the public frontier appears to be moving. citeturn33view0turn33view1