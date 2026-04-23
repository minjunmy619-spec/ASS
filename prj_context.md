# Audio source separation model edge device adapte.
**The ultimate goal is to develop a audio three stems(Speech/Music/Effects) audio source separation model which could be depoyed in edge device(TV) with NPU for realtime usage.**

Following instructions may in English or Chinese.

**You should first connect to the docker in local host with name "/zealous_agnesi"**.

**All operations should be done in the docker environment.**

The directory: "/workdir" contains the tools for export onnx to mlir, which you should use in following steps.

The project's structure: /app/:
```
ASS         #The root dir for the project, your workspace
├── .venv           #the virtual enviroment you should use for running/testing
├── 2602.08671v1.pdf
├── ASS
├── ASS.code-workspace
├── Dolphin      #Dolphin baseline
├── DolphinSFC   #Dolphin integrate with SFC
├── DolphinSFCNPU   #Dolphin SFC for NPU
├── LICENSE
├── LICENSES
├── README.md
├── TF-MLPNet      #TF-MLPNet model variants
├── TIGER          #TIGER model variants
├── aiaccel        #acaccel package for integrating torch  lightning training with configuration 
├── data           #raw data folder
├── dcase2026baseline               #please ignore this folder
├── docs
├── hydra         #hydra package for parse configuration
├── logs
├── model_weights
├── prj_context.md
├── pyproject.toml
├── recipes      #recipes folder
├── requirements.txt
├── separate_sample.py
├── spectral_feature_compression  #SFC main folder
├── tests     #test cases
└── tools     #kinds of tools
```

## Your main tasks are:
1. First help user find/research latest model structures with sota performance and suitable structure for online edge device deployment(TV).
2. Help refactor the model structure and training/inference pipeline for online/causal usage.
3. Help refactor/replace the modules or operators that are not suppored in NPU of device.
4. Make sure the updated model structure keep the core ideas/design of the origin papers.
5. Make sure the updated model stucture only contains supported operators and control flows that NPU supported.
6. Make sure the updated model contains small parameters and GMacs.
7. Make the training/inferenceing(depoy) pipline， include the preprocessing and post-processing, are consistant and the performance should not sharp decline.
---
**NPU部署限制**：
- 因为NPU支持的算作有限，只能使用简单的conv2d，以及对应的反卷积， torch.bmm, 以及其它的一些基本的tensor操作
- tensor 维度不能超过4维， forward过程中，4维的tensor不能往第一个维度折叠（Batch维度），以保证导出onnx时第一维总是1， 果维度小于4， 那么折叠到batch维度也可以
- conv2d的 (kernel_size-1)*dilation <14，其他的一些带kernel_size的操作，也需要保证这一点（比如AVGPool）
- 模型中尽量不要出现tensor 常量，因为这些常量在NPU转换过程中能会带来问题，如果不能避免，可以通过输入参数传递
- 模型在设备端是realtime/online部署，所以需要保证模型在训练时跟部署时要兼容，比如时间维度要是Causal的， 而且部署时性能不能比训练时下降太多
- 为了维持Causal/Online， 需要把一些state或者cache（比如kv_cache）存储并传递到下一帧，并且这些状态的大小受到DSP存储的限制， 一般为192k的限额
- 这些存储的state/cache 需要通过模型输入输出进行传递，onnx导出时需要考虑把这些作为输入输出参数
- 在onnx导出时尽量不要包含动态的shape 计算，控制逻辑等(比如if/else), 尽量减少循环操作(for/while)
- 对于NPU部署，大的向量操作比多个小的算子操作要有效率得多，所以尽量减少模型的nodes数量

---
## Validation
Because we do not have the finally deply device, we could validate these tasks by following steps:
1. Write suitable test scripts to valide model structure
2. Write test scripts to validate the training/inference pipline
3. Write readme file with detailed explaination for changes/designs and examples for running any commands
4. Use multip version for different tries, not just overwrite the origin files
5. Export the model to onnx format
6. Export the onnx model to mlir format

Loop these steps to get a finally working model.

---
Current we are working on directory: "/app/ASS", which contains the models from "Spectral_feature_compression" and "TIGER" paper.
The venv(./.venv) has been created for this work, so please use it when needed.

**You should consider the above guide when you answer user questions or do any operations!**

