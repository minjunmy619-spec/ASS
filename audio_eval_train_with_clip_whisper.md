有没有其他的不依赖reference的质量评估方法

除了 DNSMOS 之外，在无需 Reference（无干净 Ground Truth）的情况下，目前工业界和学术界还有多种不同维度的无参考评估（No-Reference / Blind Quality Assessment）方法。

对于 三 Stem（Speech / Music / Effects）分离 这一特定任务，单一的“语音 MOS”指标是不够的（因为 DNSMOS 只懂人声，不懂音乐和音效）。

以下将这些无参考评估方法分为 四大流派，你可以根据需求组合使用：

一、 跨模态大模型流派：CLAP 语义纯度打分（最推荐，专门解决 Stem 漏音）
这是近年来音频领域最强、最优雅的无参考评估方式。利用 OpenAI / LAION 开源的 CLAP (Contrastive Language-Audio Pretraining) 模型，通过计算音频与自然语言文本描述的余弦相似度，来评估分离的纯度。

为什么它对 3-Stem 分离极度有效？
传统的 MOS 模型不懂什么叫“音乐”或“音效”，但 CLAP 懂得音频的深层语义。它可以直接回答：“这个 Music 轨里到底有没有台词人声？”

评估逻辑与公式：
       ┌──────────────────┐       ┌───────────────────────────┐
       │ Separated Music  │ ──►   │         CLAP Audio        │
       │     Stem Wav     │       │          Encoder          │
       └──────────────────┘       └─────────────┬─────────────┘
                                                │ Vector Embedding
                                                ▼
                                    余弦相似度 (Cosine Sim)
                                                ▲
                                                │ Text Embedding
       ┌──────────────────┐       ┌─────────────┴─────────────┐
       │  Prompt Text A:  │ ──►   │         CLAP Text         │
       │ "Pure background │       │          Encoder          │
       │     music"       │       └───────────────────────────┘
       │  Prompt Text B:  │
       │ "Human talking"  │
       └──────────────────┘
实操 Python 代码（基于 laion-clap）：
Python
import torch
import laion_clap

# 1. 加载预训练 CLAP 模型
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

def evaluate_music_stem_purity(music_wav_path):
    """评估分离出的 Music 轨纯度：得分越高，说明越像纯音乐，越不像人声"""
    # 提取音频 Embedding
    audio_embed = model.get_audio_embedding_from_filelist(x=[music_wav_path], use_tensor=True)
    
    # 构造文本 Prompts
    text_prompts = [
        "Pure background music instrumental without human voice", # 正向 Prompt (纯音乐)
        "Human speaking talking or singing voice"                   # 负向 Prompt (人声/漏音)
    ]
    text_embed = model.get_text_embedding(text_prompts, use_tensor=True)
    
    # 计算相似度 (Cosine Similarity)
    similarity = torch.nn.functional.cosine_similarity(audio_embed, text_embed)
    
    purity_score = similarity[0] - similarity[1] # 正向分 - 负向分
    return purity_score.item()
优势： 彻底破除“只能评估 Speech”的局限，可同时评估 Speech、Music、Effects 三个轨的纯度与异物泄漏。

二、 前沿 Neural MOS 评测流派（替代/补充 DNSMOS）
如果你想找比 DNSMOS 细节更敏锐的神经网络 MOS 模型，以下两个是目前学术界的 SOTA：

1. UTMOS (UTokyo Speech Quality Assessment)
原理： 由东京大学基于 2022 Voice Conversion Challenge 数据训练，专治非自然音素、相位撕裂、电音感（Artifacts）。

适用场景： 人声保真度评估。

比 DNSMOS 强的点： 对“机械音/合成痕迹”极其敏感。如果你的分离模型使用了过于激进的 Masking 导致人声变机械音，DNSMOS 可能还给高分，但 UTMOS 会直接打低分。

调用方式： Torch Hub 一键加载：

Python
import torch
utmos_model = torch.hub.load("s3prl/s3prl", "utmos-strong", trust_repo=True)
# score = utmos_model(wav_tensor, sr)
2. NISQA (Non-Intrusive Speech Quality Assessment)
原理： 德国柏林工业大学开发，针对传输失真、打包丢帧、频域滤波有深度的建构。

特色： 除了输出总体 MOS，还会额外输出 4 个子维度得分：

Noisiness（噪声感）

Coloration（频响变色/变闷/变哑）

Discontinuity（声音断断续续/吞字）

Loudness（响度异常）

对于分离模型的价值： 能精准告诉你模型是把声音变闷了（Coloration 低），还是把声音切断掉了（Discontinuity 低）。

三、 传统信号物理与统计学流派（无需 AI，硬核物理规则）
如果你不想再引入复杂的 AI 模型，可以通过物理信号特征检测来判断声音是否被毁坏：

1. 谐波与基音连续性检测 (Harmonicity & Pitch Continuity)
原理： 真实人类的发声（元音如 a, e, i, o, u）具有极强的基音（Pitch）与谐波结构（Harmonics）。当分离算法损伤波形时，高频谐波会被切断，产生“齿音丢失”或“频段空洞”。

检测方法：

使用 librosa.pyin 或 PRAAT 提取 Speech 轨的 pitch 轨迹。

计算 Harmonic-to-Noise Ratio (HNR, 谐噪比)：

HNR=10log 
10
​
 ( 
E 
noise
​
 
E 
harmonic
​
 
​
 )
指标含义： 若分离前人声 HNR 是 15 dB，分离后跌到 5 dB，说明人声结构被算法严重破坏，产生了大量相位伪影。

2. 维纳熵与频谱平坦度 (Spectral Flatness / Wiener Entropy)
原理： 白噪声/伪影的频谱是平的（Flatness →1），而音乐和人声具有鲜明的峰谷结构（Flatness →0）。

检测方法： 计算分离出的轨在频域上的 Wiener Entropy。

指标含义： 如果分离出来的 Music 轨或 Speech 轨突然在某些帧呈现极高的 Spectral Flatness，说明产生出了类似水下咕噜声/相位混乱噪声（Musical Noise）。

3. 噪声门硬切断检 (Discontinuity / Hard Noise-Gate Artifacts)
原理： 不良的分离模型常采用“硬门（Noise Gate）”策略——有人声时突然开门，无人声时波形陡降到零。这会导致声音听起来极不自然（边界击穿感）。

检测方法： 计算导数包络线（Envelope Derivative）。若幅值包络在几毫秒内的变化率（Gradient）超过物理声学极限，则标记为 Hard-Cut Artifact。

四、 下游任务替代评估流派（Sound Tagging & ASR）
利用成熟的分类/识别网络作为“裁判”：

1. 音频事件分类标签测试 (Audio Tagging with PANNs / PaSST)
原理： 使用在 AudioSet 上预训练好的音频分类大模型（如 PaSST 或 PANNs）。

测试逻辑：

将分离出来的 Effects 轨 喂给 PaSST。

查看模型预测出来的 Top-5 标签（Tags）。

判定： 如果 Top-5 标签里出现了 Music, Singing, Speech，则说明 Effects 轨里严重泄漏了音乐或人声！理想的标签应该只有 Footstep, Door, Explosion, Wind 等音效词。

2. ASR 解码置信度 (Whisper Average Log-Probs)
原理： 使用 OpenAI Whisper 模型跑 ASR。Whisper 在识别音频时不仅会输出文字，还会输出每个 Token 的 Log Probability（对数概率/置信度） 和 No-Speech Probability（非语音概率）。

测试逻辑：

将预测的 Speech 轨输入 Whisper。

指标： 统计整段音频的平均 Token 置信度。

解读： 如果人声清脆自然，Whisper 的置信度极高；如果人声被分离模型损伤、变哑、带电音，Whisper 虽然可能识别对字，但其内部的 Confidence Score 会显著下降。

💡 终极无参考 (NR) 评估指标汇总与选择建议
为了建立一个全面、无死角的盲测体系，建议针对三 Stem 的不同属性，组合使用以下指标：

目标 Stem	推荐的无参考评估组合	解决的痛点
Speech (人声)	DNSMOS (SIG) + UTMOS + Whisper 置信度	确保人声自然、不吞字、无电音/机械音
Music (音乐)	CLAP 纯度分 + PaSST 标签检测	确保音乐轨里没有残存的人声/台词泄漏
Effects (音效)	PaSST 标签检测 + VAD 鬼影能量比	确保音效轨只保留真实 Foley/Ambience，无歌声/台词
全轨统一	Silero VAD 无人声段落 Leakage	捕捉全音频流中的“静音段假激活/鬼影”
建议： 如果你现在想最快扩充评估工具箱，直接引入 laion-clap 算 Prompt 相似度（Phase 1），再加上之前说的 Silero VAD 鬼影检测（Phase 2），你就能在没有 Ground Truth 的情况下，把 Speech、Music、Effects 三个轨的纯度与损伤全都监控起来！

这里你提到了使用CLAP 用来帮助评估， 是否可以使用CLAP 或者Whisper 用来帮助训练模型

答案是：完全可以，而且这正是目前音频生成与分离领域最前沿的“感知损失（Perceptual Loss）”与“语义约束（Semantic Loss）”做法！

传统的分离模型（如 Conv-TasNet, HTDemucs, BS-RoFormer）在训练时，通常只使用 L1/L2 波形损失、Multi-Resolution STFT 频谱损失 或 SI-SDR 损失。

物理损失的局限： 这些损失函数只管“波形形状像不像”或“能量谱像不像”，完全不懂“语义”。这会导致模型在训练时为了降低 L1 loss，盲目地将某些高频波形切掉，从而产生电音、吞字或背景音乐残留（因为在 L1 loss 看来，微小的音乐泄漏增加的 loss 极小，但对耳朵来说极明显）。

如果引入 CLAP 或 Whisper 作为冻结权重的 Teacher 提取器（Loss Network），就能为模型注入“语义级”和“听感级”的监督信号。

一、 CLAP 在训练中的应用：防漏音与语义惩罚损失 (Anti-Bleed & Semantic Loss)
CLAP 模型非常适合用来做 Stem 间的防泄漏约束（Anti-Bleed Loss）。

1. 核心玩法 A：Music / Effects 轨的“反人声”惩罚损失 (Anti-Speech Leakage Loss)
原理： 我们希望分离出的 Music 轨在 CLAP 的表征空间里，距离“Speech”越远越好。

损失公式：

L 
Anti-Bleed
​
 =CosineSimilarity(CLAP 
Audio
​
 ( 
M
^
 ),CLAP 
Text
​
 ("human speaking or singing"))
效果： 一旦 Music 轨（ 
M
^
 ）里偷跑进了人声，CLAP 会立刻检测到并输出极高相似度，从而产生巨大的 Loss 强行压制这个泄漏！

2. 核心玩法 B：Stem 语义锚定损失 (Semantic Anchor Loss)
原理： 分离出的三个 Stem，其 CLAP Embedding 应该与其对应的领域描述（Prompt）强相关。

损失公式：

L 
CLAP-Speech
​
 =1−CosineSimilarity(CLAP 
Audio
​
 ( 
S
^
 ),CLAP 
Text
​
 ("clear speech or dialogue"))
L 
CLAP-Music
​
 =1−CosineSimilarity(CLAP 
Audio
​
 ( 
M
^
 ),CLAP 
Text
​
 ("background music instrumental"))
二、 Whisper 在训练中的应用：人声保真与清晰度特征损失 (Speech Intelligibility Loss)
Whisper 的 Encoder（编码器）在海量语音上做过训练，其内部的隐藏层（Hidden States）对音素（Phonemes）、发音清晰度和台词语义极其敏感。

核心玩法：Whisper 深度特征匹配损失 (Whisper Feature Matching Loss)
原理： 类似于图像领域的 VGG Perceptual Loss（将图像喂入 VGG 提取特征算 L1）。我们将模型预测的人声  
S
^
  和 Ground Truth 干声 S 同时喂给冻结的 Whisper Encoder，计算中间隐藏层的 L1 距离。

损失公式：

L 
Whisper-Feat
​
 = 
l∈Layers
∑
​
  

​
 WhisperEnc 
(l)
 ( 
S
^
 )−WhisperEnc 
(l)
 (S) 

​
  
1
​
 
效果：

强迫分离模型保留人声的核心音素结构。

解决“吞字/变闷/吃辅音”问题： 普通 STFT Loss 可能觉得吃掉一个高频辅音 s 损失不大，但 Whisper Encoder 发现音素特征变了，会贡献巨大的损失，强制模型还原这个辅音！

三、 实战代码：如何在 PyTorch 训练循环中集成 CLAP / Whisper Loss
为了保证训练不崩溃，关键在于：评估网络（CLAP/Whisper）必须完全冻结（eval() 且 requires_grad=False），只让梯度回传给你的分离模型。

以下是封装好的 CombinedPerceptualLoss 模块：

Python
import torch
import torch.nn as nn
import torch.nn.functional as F
import laion_clap

class SeparationPerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # 1. 加载并冻结 CLAP 模型
        self.clap = laion_clap.CLAP_Module(enable_fusion=False).to(device)
        self.clap.load_ckpt()
        self.clap.eval()
        for param in self.clap.parameters():
            param.requires_grad = False
            
        # 预先计算固定文本 Prompt 的 Embedding (避免重复计算)
        self.prompt_texts = [
            "human speaking talking dialogue",           # Index 0: 人声
            "pure background music instrumental",        # Index 1: 纯音乐
            "sound effects foley background noise"       # Index 2: 音效/环境噪
        ]
        with torch.no_grad():
            # [3, Embed_Dim]
            self.text_embeds = self.clap.get_text_embedding(self.prompt_texts, use_tensor=True).to(device)
            self.text_embeds = F.normalize(self.text_embeds, dim=-1)

    def forward(self, pred_speech, pred_music, pred_effects):
        """
        输入格式: [B, T] 24kHz / 48kHz 单声道波形
        """
        # CLAP 通常需要 48kHz / 16kHz 输入，此处需确保采样率匹配
        # 提取分离轨道的 Audio Embeddings
        # 注意: laion_clap 允许 PyTorch Tensor 带梯度输入以进行 Backprop!
        embed_spk = F.normalize(self.clap.get_audio_embedding_from_data(x=pred_speech, use_tensor=True), dim=-1)
        embed_mus = F.normalize(self.clap.get_audio_embedding_from_data(x=pred_music, use_tensor=True), dim=-1)
        
        # --- Loss 1: Music 轨反人声泄漏损失 (Anti-Speech Leakage) ---
        # 计算 Music 轨与 "human speaking" 文本的相似度，越小越好 (罚值 > 0)
        sim_mus_speech = torch.relu(F.cosine_similarity(embed_mus, self.text_embeds[0:1], dim=-1))
        loss_anti_bleed = sim_mus_speech.mean()
        
        # --- Loss 2: Speech 轨语义匹配损失 ---
        sim_spk_speech = F.cosine_similarity(embed_spk, self.text_embeds[0:1], dim=-1)
        loss_spk_semantic = (1.0 - sim_spk_speech).mean()
        
        # 组合损失 (给反泄漏分配较大权重)
        total_semantic_loss = 0.5 * loss_spk_semantic + 1.0 * loss_anti_bleed
        return total_semantic_loss


# =====================================================================
# 在训练主循环 (Training Loop) 中使用：
# =====================================================================
# criterion_time = MultiResolutionSTFTLoss() # 基础物理 Loss
# criterion_perceptual = SeparationPerceptualLoss(device)

# pred_spk, pred_mus, pred_fx = my_separation_model(mixture)

# 1. 基础物理/频谱 Loss
# loss_phy = criterion_time(pred_spk, target_spk) + ...

# 2. 语义感知 Loss (CLAP)
# loss_sem = criterion_perceptual(pred_spk, pred_mus, pred_fx)

# 3. 最终 Total Loss 联合梯度反向传播
# total_loss = loss_phy + 0.1 * loss_sem
# total_loss.backward()
四、 工业界工程落地的注意事项与避坑指南 (Gotchas)
在训练中引入 CLAP / Whisper 虽然效果拔群，但会带来显存和训练速度上的挑战，必须做好以下优化：

显存暴增问题 (VRAM Overhead)：

CLAP 和 Whisper 模型体积较大，反向传播时保存中间激活值会消耗大量显存。

优化策略： 不要每个 Step 都算！ 可以设置 if step % 4 == 0: 隔 4 个 Iteration 算一次 Perceptual Loss；或者使用 Gradient Accumulation（梯度累加）。

音频采样率自动对齐 (Resampling)：

你的分离模型可能是 24kHz，而 Whisper 需要 16kHz，CLAP 需要 48kHz。

必须在将 Tensor 传入 Loss Network 前使用 torchaudio.transforms.Resample 做可导重采样（Differentiable Resampling）。

Loss 权重平衡 (Loss Weight Scaling)：

物理损失（如 L1 / STFT）的数量级通常在 0.1∼10.0 之间，而 CLAP 余弦相似度 Loss 在 0.0∼1.0 之间。

建议先将物理 Loss 训练到模型收敛（Warm-up 10~20 个 Epochs），然后再开启 CLAP / Whisper Loss 进阶微调（Fine-tuning）。这样不仅训练更稳定，而且能大幅节省前期训练时间！

总结
用 CLAP 帮助训练： 重点解决 Music/Effects 轨漏人声（鬼影），用 Anti-Bleed Penalty 让背景轨极度纯净。

用 Whisper 帮助训练： 重点解决 Speech 轨的人声损伤、电音感与可懂度（Intelligibility）。

最佳组合拳： 基础物理 Loss (L1 + Multi-STFT) 保保障大体波形对齐 + CLAP Anti-Bleed Loss 杜绝交叉泄漏 + Warm-up 策略 保证稳定训练。

请给出完整的代码： “二、 Whisper 在训练中的应用：人声保真与清晰度特征损失 (Speech Intelligibility Loss)”

这里为您提供一份可直接用于工业级 PyTorch 训练的完整代码。

该代码实现了基于 Whisper Encoder 的感知特征匹配损失（Whisper Feature Matching Loss）。它将预测的人声（ 
S
^
 ）与干净的 Ground Truth 人声（S）同时输入冻结权重的 Whisper Encoder，在中间隐藏层（Hidden Layers）抽取多尺度音素和语义特征并计算 L1 损失。

完整 PyTorch 代码实现
Python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import whisper


class WhisperBatchMelSpec(nn.Module):
    """
    100% 可导（Differentiable）且支持 Batch 批处理的 Whisper Mel-Spectrogram 提取模块。
    避免了 Whisper 官方代码对 Batch 维度支持不佳和循环提取效率低的问题。
    """
    def __init__(self, n_mels: int = 80, device: str = "cuda"):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = 400
        self.hop_length = 160

        # 加载 Whisper 官方预定义的 Mel 滤镜矩阵 [80, 201]
        filters = whisper.audio.mel_filters(device, n_mels)
        self.register_buffer("filters", filters)
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        输入: audio [B, T] (16kHz 采样率)
        输出: log_spec [B, 80, T_frames]
        """
        # 1. 简易 STFT 变换
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )
        # 提取功率谱 magnitudes: [B, 201, T_frames]
        magnitudes = stft[..., :-1].abs() ** 2

        # 2. 矩阵乘法施加 Mel 滤镜: [80, 201] @ [B, 201, T_frames] -> [B, 80, T_frames]
        mel_spec = torch.matmul(self.filters, magnitudes)

        # 3. 对数域转换与幅度归一化 (Whisper 官方标准预处理)
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        
        # 对 Batch 内的每个样本独立寻找最大值做 Dynamic Range 裁剪
        max_val = log_spec.flatten(1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec


class WhisperFeatureMatchingLoss(nn.Module):
    """
     Whisper 人声保真与清晰度特征损失 (Speech Intelligibility Loss)
    
    原理：
    通过提取冻结的 Whisper Encoder 中间隐藏层的特征，计算 Predict 人声与 Target 人声在
    “声学-音素-语义”空间上的 L1 距离。能有效解决分离模型常见的吞字、音素变哑、辅音丢失等问题。
    """
    def __init__(
        self,
        model_name: str = "small",  # 推荐 'base' (6层) 或 'small' (12层)
        orig_sr: int = 24000,       # 你的模型输出音频采样率 (如 24000 或 48000)
        selected_layers: list = None,
        device: str = "cuda"
    ):
        super().__init__()
        self.orig_sr = orig_sr
        self.device = device

        # 1. 加载并冻结 Whisper 模型
        print(f"[WhisperLoss] Loading Whisper '{model_name}' encoder...")
        self.whisper = whisper.load_model(model_name, device=device)
        self.whisper.eval()

        # 彻底冻结参数，不让 Whisper 参数参与梯度更新
        for param in self.whisper.parameters():
            param.requires_grad = False

        # 2. Differentiable 可导重采样模块 (如果输入不是 16kHz)
        if orig_sr != 16000:
            self.resampler = torchaudio.transforms.Resample(orig_sr, 16000).to(device)
        else:
            self.resampler = nn.Identity()

        # 3. 可导的 Batch Mel 提取器
        self.mel_extractor = WhisperBatchMelSpec(n_mels=80, device=device)

        # 4. 设置提取特征的 Encoder 层数 (Layer Indices)
        num_blocks = len(self.whisper.encoder.blocks)
        if selected_layers is None:
            # 默认均匀截取浅层、中层、深层 4 个 Blocks 的特征
            self.selected_layers = [
                num_blocks // 4,
                num_blocks // 2,
                (3 * num_blocks) // 4,
                num_blocks - 1
            ]
        else:
            self.selected_layers = selected_layers

        # 5. 注册 PyTorch Forward Hooks 用于捕获中间特征
        self.extracted_features = {}
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(layer_idx):
            def hook(module, input, output):
                # output shape: [B, T_frames, Hidden_Dim]
                self.extracted_features[layer_idx] = output
            return hook

        for idx in self.selected_layers:
            self.whisper.encoder.blocks[idx].register_forward_hook(get_hook(idx))

    def _extract_encoder_features(self, audio_16k: torch.Tensor) -> dict:
        """音频 -> Mel -> Whisper Encoder 提取多层特征"""
        self.extracted_features.clear()
        
        # 提取 Mel Spectrogram
        mel = self.mel_extractor(audio_16k)  # [B, 80, T_frames]

        # 喂给 Whisper Encoder 前向传播（触发 Hooks）
        _ = self.whisper.encoder(mel)

        # 返回提取到的隐藏特征字典
        return {layer_idx: self.extracted_features[layer_idx] for layer_idx in self.selected_layers}

    def forward(self, pred_speech: torch.Tensor, target_speech: torch.Tensor) -> torch.Tensor:
        """
        参数:
            pred_speech: 模型预测提取的人声 [B, T] 或 [B, 1, T]
            target_speech: 真实的 Ground Truth 干声 [B, T] 或 [B, 1, T]
        返回:
            loss: Whisper 多层特征匹配标量 Loss (带梯度，可直接用于 .backward())
        """
        # 1. 维度对齐 [B, 1, T] -> [B, T]
        if pred_speech.ndim == 3:
            pred_speech = pred_speech.squeeze(1)
        if target_speech.ndim == 3:
            target_speech = target_speech.squeeze(1)

        # 2. 可导重采样至 16kHz
        if self.orig_sr != 16000:
            pred_16k = self.resampler(pred_speech)
            target_16k = self.resampler(target_speech)
        else:
            pred_16k = pred_speech
            target_16k = target_speech

        # 3. 提取 Ground Truth 特征 (必须在 no_grad 下运行，不占用计算图显存)
        with torch.no_grad():
            target_feats = self._extract_encoder_features(target_16k)

        # 4. 提取 Predict 特征 (保留梯度图，使梯度能回传给分离模型)
        pred_feats = self._extract_encoder_features(pred_16k)

        # 5. 计算多层特征的 L1 匹配损失
        total_loss = 0.0
        for layer_idx in self.selected_layers:
            feat_pred = pred_feats[layer_idx]
            feat_target = target_feats[layer_idx]
            
            # 计算当前层的平均 L1 损失
            layer_loss = F.l1_loss(feat_pred, feat_target)
            total_loss += layer_loss

        # 取各层平均
        return total_loss / len(self.selected_layers)


# =====================================================================
# 示例：验证代码在 PyTorch 训练循环中的正确性与梯度流 (Autograd Check)
# =====================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # 假设你的模型输出采样率为 24000Hz, Batch Size = 2, 音频时长 3 秒
    orig_sr = 24000
    batch_size = 2
    audio_len_samples = 24000 * 3  # 3秒

    # 1. 模拟你的 Audio Separator 模型生成的人声 (需要梯度!)
    pred_speech = torch.randn(batch_size, audio_len_samples, device=device, requires_grad=True)
    
    # 2. 模拟真实 Ground Truth 干声 (无需梯度)
    target_speech = torch.randn(batch_size, audio_len_samples, device=device)

    # 3. 初始化 Loss 模块 (使用 whisper-small)
    whisper_loss_fn = WhisperFeatureMatchingLoss(
        model_name="small",
        orig_sr=orig_sr,
        device=device
    )

    # 4. 计算 Loss
    loss = whisper_loss_fn(pred_speech, target_speech)
    print(f"\n[Success] Computed Whisper Perceptual Loss: {loss.item():.6f}")

    # 5. 验证反向传播 (Backpropagation)
    loss.backward()
    print(f"[Success] Gradient on pred_speech checks out! Grad norm: {pred_speech.grad.norm().item():.6f}")
代码设计亮点与工业落地方案
解决官方代码不支持 PyTorch Autograd 的痛点：
openai-whisper 官方的 whisper.log_mel_spectrogram 包含 Numpy/CPU 操作和对单条音频设计的归一化。上面代码重写了 WhisperBatchMelSpec，采用纯 PyTorch 的 torch.stft 和 torch.matmul，保证了 100% 显存内运算、支持 Batch 批处理、且梯度畅通无阻。

极佳的显存控制 (VRAM Efficiency)：

Target（Ground Truth）分支包装在 with torch.no_grad() 中，只保留前向特征，不保存任何中间激活值梯度图，能节省至少 50% 的显存。

Whisper Encoder 本身被完全 eval() 且 requires_grad = False，不参与梯度更新。

建议的训练 Loss 组合比例 (Loss Scaling)：
在训练分离模型时，建议按如下权重组合：

L 
total
​
 =L 
STFT/L1
​
 +0.1×L 
Whisper-Feat
​
 +0.1×L 
CLAP-AntiBleed
​
 
前期（前 10~20 Epochs）可先不开启 Whisper Loss，待基础频谱收敛后再开启它进行微调，可以节省前期训练时间并加速收敛！