import os
import zipfile

# 1. 音源分离量化工具包源码
toolkit_code = '''import copy
from typing import Dict, List, Optional, Tuple, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. 基础模块：Per-Channel / Per-Tensor 伪量化器 (Fake Quantizer)
# ==============================================================================

class FakeQuantizer(nn.Module):
    """支持 Per-Channel / Per-Tensor 的 INT8 伪量化器 (Symmetric / Asymmetric)"""
    def __init__(
        self,
        num_bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = True,
        channel_dim: int = 0
    ):
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_dim = channel_dim

        if symmetric:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = (2 ** (num_bits - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (2 ** num_bits) - 1

        self.register_buffer("scale", None)
        self.register_buffer("zero_point", None)

    def update_scale_zp(self, x: torch.Tensor):
        """计算并更新 Scale 与 Zero-Point"""
        with torch.no_grad():
            if self.per_channel and x.dim() > 1:
                reduce_dims = tuple(i for i in range(x.dim()) if i != self.channel_dim)
                max_val = torch.amax(x, dim=reduce_dims, keepdim=True)
                min_val = torch.amin(x, dim=reduce_dims, keepdim=True)
            else:
                min_val = x.min()
                max_val = x.max()

            if self.symmetric:
                max_abs = torch.maximum(min_val.abs(), max_val.abs())
                scale = max_abs / (self.qmax)
                scale = torch.clamp(scale, min=1e-8)
                zero_point = torch.zeros_like(scale)
            else:
                scale = (max_val - min_val) / (self.qmax - self.qmin)
                scale = torch.clamp(scale, min=1e-8)
                zero_point = torch.round(-min_val / scale) + self.qmin
                zero_point = torch.clamp(zero_point, self.qmin, self.qmax)

            self.scale = scale
            self.zero_point = zero_point

    def forward(self, x: torch.Tensor, v_param: Optional[torch.Tensor] = None):
        """前向传播伪量化：支持传入 AdaRound 的软舍入参数 v_param"""
        if self.scale is None:
            self.update_scale_zp(x)

        x_clamped = x
        if v_param is None:
            # 标准 Round-to-Nearest 舍入
            x_int = torch.round(x_clamped / self.scale) + self.zero_point
        else:
            # AdaRound 软/硬舍入机制
            h_v = torch.clamp(v_param * 1.2 - 0.1, 0.0, 1.0)
            x_floor = torch.floor(x_clamped / self.scale)
            x_int = x_floor + h_v + self.zero_point

        x_quant = torch.clamp(x_int, self.qmin, self.qmax)
        x_dequant = (x_quant - self.zero_point) * self.scale
        return x_dequant

# ==============================================================================
# 2. 核心模块：跨层权重等化 (Cross-Layer Equalization)
# ==============================================================================

class CrossLayerEqualizer:
    """自动消除相邻 Conv/Linear 层的权重量化阶梯差异 (W1 * S^-1) 与 (S * W2)"""
    @staticmethod
    @torch.no_grad()
    def equalize_pair(layer1: nn.Module, layer2: nn.Module, threshold: float = 0.2):
        w1 = layer1.weight.data
        w2 = layer2.weight.data

        if w1.shape[0] != w2.shape[1]:
            return

        w1_reduce_dims = tuple(range(1, w1.dim()))
        w2_reduce_dims = (0,) + tuple(range(2, w2.dim()))

        r1 = torch.amax(w1.abs(), dim=w1_reduce_dims)
        r2 = torch.amax(w2.abs(), dim=w2_reduce_dims)

        scale = torch.sqrt(r1 / (r2 + 1e-8))
        scale = torch.clamp(scale, min=0.1, max=10.0)

        s_w1 = scale.view(-1, *([1] * (w1.dim() - 1)))
        layer1.weight.data /= s_w1
        if layer1.bias is not None:
            layer1.bias.data /= scale

        s_w2 = scale.view(1, -1, *([1] * (w2.dim() - 2)))
        layer2.weight.data *= s_w2

# ==============================================================================
# 3. 核心模块：快速自适应舍入 (Fast AdaRound)
# ==============================================================================

class AdaRoundOptimizer:
    """微调权重舍入方向，极大地恢复音源分离相位与细节损失"""
    def __init__(self, layer: Union[nn.Conv1d, nn.Conv2d, nn.Linear], quantizer: FakeQuantizer):
        self.layer = layer
        self.quantizer = quantizer

    def optimize_layer(
        self,
        cached_inputs: List[torch.Tensor],
        cached_outputs: List[torch.Tensor],
        num_iters: int = 150,
        lr: float = 3e-2
    ):
        device = self.layer.weight.device
        w = self.layer.weight.data

        self.quantizer.update_scale_zp(w)
        scale = self.quantizer.scale

        x_floor = torch.floor(w / scale)
        rest = (w / scale) - x_floor
        v_init = -torch.log((1.0 / (rest + 1e-4) - 1.0) + 1e-4)
        v_param = nn.Parameter(v_init.clone().to(device))

        optimizer = torch.optim.Adam([v_param], lr=lr)
        self.layer.eval()

        for it in range(num_iters):
            optimizer.zero_grad()
            total_loss = 0.0

            w_quant = self.quantizer(w, v_param=torch.sigmoid(v_param))
            orig_w = self.layer.weight.data
            self.layer.weight.data = w_quant

            for inp, target in zip(cached_inputs, cached_outputs):
                inp, target = inp.to(device), target.to(device)
                pred = self.layer(inp)

                rec_loss = F.mse_loss(pred, target)
                beta = 1.0 - abs(it / num_iters - 0.5) * 2
                reg_loss = (1.0 - (2.0 * torch.abs(torch.sigmoid(v_param) - 0.5)).pow(2)).sum() * beta * 1e-4

                loss = rec_loss + reg_loss
                loss.backward()
                total_loss += loss.item()

            self.layer.weight.data = orig_w
            optimizer.step()

        with torch.no_grad():
            final_v = (torch.sigmoid(v_param) > 0.5).float()
            final_w_quant = self.quantizer(w, v_param=final_v)
            self.layer.weight.data = final_w_quant

# ==============================================================================
# 4. 核心模块：音频专用偏差校正 (Empirical Bias Correction)
# ==============================================================================

class AudioBiasCorrector:
    """消除量化引入的 DC Offset，彻底解决音源分离中的低频咔哒声与底噪"""
    @staticmethod
    @torch.no_grad()
    def correct_layer_bias(fp32_layer: nn.Module, quant_layer: nn.Module, calib_inputs: List[torch.Tensor]):
        device = next(fp32_layer.parameters()).device
        fp32_means, quant_means = [], []

        for x in calib_inputs:
            x = x.to(device)
            out_fp32 = fp32_layer(x)
            out_quant = quant_layer(x)

            reduce_dims = (0,) + tuple(range(2, out_fp32.dim()))
            fp32_means.append(out_fp32.mean(dim=reduce_dims))
            quant_means.append(out_quant.mean(dim=reduce_dims))

        mean_fp32 = torch.stack(fp32_means).mean(dim=0)
        mean_quant = torch.stack(quant_means).mean(dim=0)

        bias_delta = mean_fp32 - mean_quant

        if quant_layer.bias is not None:
            quant_layer.bias.data += bias_delta
        else:
            quant_layer.bias = nn.Parameter(bias_delta)

# ==============================================================================
# 5. 音频量化大师主控类
# ==============================================================================

class AudioQuantToolkit:
    """音源分离专属量化 Toolkit 工具主控类"""
    def __init__(self, model: nn.Module, calib_dataloader):
        self.fp32_model = model.eval()
        self.quant_model = copy.deepcopy(model).eval()
        self.calib_dataloader = calib_dataloader

        self.calib_inputs = []
        with torch.no_grad():
            for i, batch in enumerate(calib_dataloader):
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                self.calib_inputs.append(x)
                if i >= 16:
                    break

    def step1_apply_cle(self):
        print("🚀 [Step 1/3] 正在对全模型执行跨层权重等化 (CLE)...")
        modules = list(self.quant_model.named_children())

        for i in range(len(modules) - 1):
            m1_name, m1 = modules[i]
            m2_name, m2 = modules[i + 1]

            if isinstance(m1, (nn.Conv1d, nn.Conv2d, nn.Linear)) and isinstance(m2, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                CrossLayerEqualizer.equalize_pair(m1, m2)
                print(f"   └─ 已均衡权重对: {m1_name} <==> {m2_name}")

    def step2_apply_adaround(self, epochs_per_layer: int = 150):
        print("🚀 [Step 2/3] 正在执行逐层自适应舍入优化 (AdaRound)...")
        fp32_mods = dict(self.fp32_model.named_modules())
        quant_mods = dict(self.quant_model.named_modules())

        for name, q_mod in quant_mods.items():
            if isinstance(q_mod, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fp32_mod = fp32_mods[name]

                inputs_cache, outputs_cache = [], []
                def hook(module, input, output):
                    inputs_cache.append(input[0].detach())
                    outputs_cache.append(output.detach())

                h = fp32_mod.register_forward_hook(hook)
                with torch.no_grad():
                    for inp in self.calib_inputs:
                        _ = self.fp32_model(inp.to(next(self.fp32_model.parameters()).device))
                h.remove()

                quantizer = FakeQuantizer(num_bits=8, symmetric=True, per_channel=True)
                optimizer = AdaRoundOptimizer(q_mod, quantizer)
                optimizer.optimize_layer(inputs_cache, outputs_cache, num_iters=epochs_per_layer)
                print(f"   └─ 完成层 [{name}] 的 AdaRound 舍入优化")

    def step3_apply_bias_correction(self):
        print("🚀 [Step 3/3] 正在执行音源分离偏置校正 (Bias Correction)...")
        fp32_mods = dict(self.fp32_model.named_modules())
        quant_mods = dict(self.quant_model.named_modules())

        for name, q_mod in quant_mods.items():
            if isinstance(q_mod, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fp32_mod = fp32_mods[name]
                AudioBiasCorrector.correct_layer_bias(fp32_mod, q_mod, self.calib_inputs)
                print(f"   └─ 消除层 [{name}] 的直流偏置与均值漂移")

    def export_quantized_model(self) -> nn.Module:
        self.step1_apply_cle()
        self.step2_apply_adaround()
        self.step3_apply_bias_correction()
        print("\\n🎉 音源分离量化模型处理完成！")
        return self.quant_model
'''

# 2. SDR 评估脚本源码
eval_code = '''from typing import Dict, List, Tuple
import mir_eval
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from audio_quant_toolkit import AudioQuantToolkit

class AudioSeparationEvaluator:
    def __init__(self, stem_names: List[str] = ["Vocals", "Drums", "Other"]):
        self.stem_names = stem_names
        self.num_stems = len(stem_names)
        self.sisdr_metric = ScaleInvariantSignalDistortionRatio()

    def compute_mir_eval_metrics(self, references: np.ndarray, estimates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sdrs, sirs, sars, _ = mir_eval.separation.bss_eval_sources(
            references, estimates, compute_permutation=False
        )
        return sdrs, sirs, sars

    @torch.no_grad()
    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, Dict[str, float]]:
        model.eval()
        model.to(device)

        sdr_list = [[] for _ in range(self.num_stems)]
        sir_list = [[] for _ in range(self.num_stems)]
        sisdr_list = [[] for _ in range(self.num_stems)]

        for batch_idx, (mix_audio, ref_targets) in enumerate(dataloader):
            mix_audio = mix_audio.to(device)
            ref_targets = ref_targets.to(device)

            est_targets = model(mix_audio)
            batch_size = mix_audio.shape[0]

            for b in range(batch_size):
                ref_np = ref_targets[b].cpu().numpy()
                est_np = est_targets[b].cpu().numpy()

                sdrs, sirs, _ = self.compute_mir_eval_metrics(ref_np, est_np)

                for s in range(self.num_stems):
                    sdr_list[s].append(sdrs[s])
                    sir_list[s].append(sirs[s])
                    si_sdr = self.sisdr_metric(est_targets[b, s], ref_targets[b, s]).item()
                    sisdr_list[s].append(si_sdr)

        results = {}
        for s, stem in enumerate(self.stem_names):
            results[stem] = {
                "SDR": float(np.nanmean(sdr_list[s])),
                "SIR": float(np.nanmean(sir_list[s])),
                "SI-SDR": float(np.nanmean(sisdr_list[s])),
            }

        results["Overall Mean"] = {
            "SDR": float(np.nanmean([results[s]["SDR"] for s in self.stem_names])),
            "SIR": float(np.nanmean([results[s]["SIR"] for s in self.stem_names])),
            "SI-SDR": float(np.nanmean([results[s]["SI-SDR"] for s in self.stem_names])),
        }

        return results

def print_comparison_report(
    fp32_res: Dict[str, Dict[str, float]],
    quant_res: Dict[str, Dict[str, float]],
    stem_names: List[str]
):
    print("\\n" + "="*78)
    print("      📊 音源分离 FP32 vs. INT8 量化模型性能对比评估报告 (SDR / SIR)")
    print("="*78)
    print(f"{'Stem':<12} | {'Metric':<8} | {'FP32 Model':<12} | {'Quant Model':<12} | {'Delta (Loss)':<12}")
    print("-" * 78)

    all_stems = stem_names + ["Overall Mean"]

    for stem in all_stems:
        for metric in ["SDR", "SIR", "SI-SDR"]:
            fp32_val = fp32_res[stem][metric]
            quant_val = quant_res[stem][metric]
            delta = quant_val - fp32_val

            delta_str = f"{delta:+.3f} dB"
            if delta < -1.0:
                delta_str += " ⚠️"
            elif delta >= -0.3:
                delta_str += " ✅"

            print(f"{stem:<12} | {metric:<8} | {fp32_val:8.3f} dB   | {quant_val:8.3f} dB   | {delta_str:<12}")
        print("-" * 78)

if __name__ == "__main__":
    class DummyAudioSeparationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 16, kernel_size=15, padding=7)
            self.bn1 = nn.BatchNorm1d(16)
            self.conv2 = nn.Conv1d(16, 3, kernel_size=15, padding=7)

        def forward(self, x):
            feat = torch.relu(self.bn1(self.conv1(x)))
            return self.conv2(feat)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 8
    audio_len = 16000 * 2

    dummy_inputs, dummy_targets = [], []
    for _ in range(num_samples):
        stems = torch.randn(3, audio_len)
        mix = stems.sum(dim=0, keepdim=True)
        dummy_inputs.append(mix)
        dummy_targets.append(stems)

    dataset = torch.utils.data.TensorDataset(torch.stack(dummy_inputs), torch.stack(dummy_targets))
    test_loader = DataLoader(dataset, batch_size=2)

    fp32_model = DummyAudioSeparationModel().eval()

    print("🛠️ 正在对 FP32 模型执行 Toolkit 量化优化...")
    toolkit = AudioQuantToolkit(fp32_model, test_loader)
    quant_model = toolkit.export_quantized_model()

    stem_names = ["Vocals", "Drums", "Other"]
    evaluator = AudioSeparationEvaluator(stem_names=stem_names)

    print("\\n🔍 正在评估 FP32 模型性能...")
    fp32_results = evaluator.evaluate_model(fp32_model, test_loader, device=device)

    print("🔍 正在评估 INT8 量化模型性能...")
    quant_results = evaluator.evaluate_model(quant_model, test_loader, device=device)

    print_comparison_report(fp32_results, quant_results, stem_names)
'''

# 写入文件并压缩
def build_package():
    with open("audio_quant_toolkit.py", "w", encoding="utf-8") as f:
        f.write(toolkit_code)
    
    with open("eval_quant_sdr.py", "w", encoding="utf-8") as f:
        f.write(eval_code)

    zip_filename = "AudioQuantToolkit.zip"
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write("audio_quant_toolkit.py")
        zipf.write("eval_quant_sdr.py")

    print(f"✅ 成功生成文件：\n - audio_quant_toolkit.py\n - eval_quant_sdr.py")
    print(f"📦 已打包完成：{os.path.abspath(zip_filename)}")

if __name__ == "__main__":
    build_package()