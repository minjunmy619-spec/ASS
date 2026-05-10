# Audio sound separation model for online/realtime/causal deployment on edge device with NPU

## Project Context
```
local WSL workspace： /home/cmj/works/ASS
docker mapping： 
    - docker name：/zealous_agnesi
    - workspace: /app/ASS
    - onnx2mlir tool: /workdir/onnx-mlir
```

**When run testing and verification, please make sure you are using the docker container and the .venv virtual environment!**

**Please ignore the "dcase2026baseline" subfolder for this work.**
Please first check the whole project structure，it may changes, so when needed, please re-check.
The project constructure:
```
ASS         #The root dir for the project, your workspace
├── .venv           #the virtual enviroment you should use for running/testing in the docker environment
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
├── dcase2026baseline               #Please ignore this folder for the NPU compilation work
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

The tools for verify the onnx to mlir in the docker, you could check the README.md in onnx-mlir for details
```
/workdir# tree -L 2
.
|-- llvm-project
|   |-- CODE_OF_CONDUCT.md
|   |-- CONTRIBUTING.md
|   |-- LICENSE.TXT
|   |-- README.md
|   |-- SECURITY.md
|   |-- bolt
|   |-- build
|   |-- clang
|   |-- clang-tools-extra
|   |-- cmake
|   |-- compiler-rt
|   |-- cross-project-tests
|   |-- flang
|   |-- flang-rt
|   |-- libc
|   |-- libclc
|   |-- libcxx
|   |-- libcxxabi
|   |-- libsycl
|   |-- libunwind
|   |-- lld
|   |-- lldb
|   |-- llvm
|   |-- llvm-libgcc
|   |-- mlir
|   |-- offload
|   |-- openmp
|   |-- orc-rt
|   |-- polly
|   |-- pyproject.toml
|   |-- runtimes
|   |-- third-party
|   `-- utils
`-- onnx-mlir
    |-- CHANGELOG.md
    |-- CMakeLists.txt
    |-- CODE_OF_CONDUCT.md
    |-- CODING_PRACTICE.md
    |-- CONTRIBUTING.md
    |-- Doxyfile
    |-- GOVERNANCE.md
    |-- LICENSE
    |-- MLIR.cmake
    |-- OMCRuntime.cmake
    |-- OMPyInfer.cmake
    |-- README.md
    |-- SECURITY.md
    |-- VERSION_NUMBER
    |-- build
    |-- docker
    |-- docs
    |-- include
    |-- requirements.txt
    |-- src
    |-- test
    |-- third_party
    `-- utils
```

## Target
The main mission is to convert a three stems(Speech/Music/Effects) audio sound separation model with SOTA performance into online/realtime/causal edge device deployment with NPU support.

### High Level Guide:
1. First help user find/research latest model structures with sota performance and check whether it is suitable for online edge device deployment(TV).
2. Help refactor the model structure and training/inference pipeline for online/causal usage.
3. Help refactor/replace the modules or operators that are not suppored in NPU of device.
4. Make sure the updated model structure keep the core ideas/design of the origin papers.
5. Make sure the updated model stucture only contains supported operators and control flows that NPU supported.
6. Make sure the updated model contains small parameters and GMacs.
7. Make the train/inference/deploy pipline， include the preprocessing and post-processing, are consistant and the performance should not sharp decline.


**Please do the correct things, not just for simpler!**

## Currrent Status
Current we had tried TIGER, spectral_feature_compression, TF-MLPNet, Dolphin, but not finished yet.
We need continue to verify the current models and also try to find any new candidates if needed.
You should first check the current progress and status always.

## Limitations and Rules
The main rules for convert/compile the pytorch model into NPU(the Intermediate format is onnx and then mlir):
```
rule1: Only limited basic operators are suppored by the NPU, so please make sure the refactored model does not contain complex operators and control flow
rule2: Basiclly only 2D Conv/Tranposed_2D Conv/bmm/softmax/sigmoid/resize/padding/reshape/transpose operators are supported(not list all them here), so please avoid 1D operators
rule3: The dims of tensors should not exceed 4, make sure for all tensors, dims<=4
rule4: If the dims of a tensor is 4, the first dim should be the batch_size dim. Then, when exported to onnx, it can be set as 1 for NPU compatible. So do not fold other dim to the batch_size dim
rule5: The kernel_size and dilation of operator should meets the limit: (kernel_size - 1)* dilation <=14
rule6: The stride for AvgPool2D should be one of [1, 2, 4], the stride for Tranposed_2D should only be 2
rule7: Do not use AdaptiveAvgPool2D, it is not supported for exporting to onnx. If needed, implement a customized AvgPool2D, and make sure the kernel_size and stride meet the rule5 and rule5
rule8: Do not use const. Especially the const tensors in the model forward pass. If needed, pre-calculation or extract that out of the model and passed as input parameters 
rule9: Avoid ScatterND, unflatten, and other similar operators.
rule10: Try to avoid memory operators, such as Slice, Transpose, Cat and so on, those would slow down the performance of the model, for the NPU is not effective at accessing memory
rule11: Try to control the number of nodes in the graph, the NPU is good at calculating with big matrics, while not for lots of small operators
rule12: The model should be temporal causul, it need be deployed for realtime/streaming/online inference
rule13: To support the oneline deployment, the size of the passed state or cache should be small. the DSP quotas(192k) for all the inputs/outputs should be considered carefully
rule14: To support onnx exporting, dynamic control flows should be avoid, and the number of inputs/outputs parameter should be small
```

## Validation
Because we do not have the finally deploy device with the NPU on hand, we could just validate the conversion/compliation by following steps:
1. Write suitable test scripts to valide model structure, make use it respect to the rules
2. Write test scripts to validate the training/inference pipline make the inputs/outputs are consistant
3. Export the model to onnx format
4. Export the onnx model to mlir format using the onnx-mlir
Loop these steps to get a finally working model.

---
## Other important points/notes:
1. Remember to write README file with detailed explaination for changes/designs and examples for running any commands
2. Use multip version for different tries/experiments, not just overwrite the origin versions, make sure the names are brief and representative 
3. You should also prepare the training/inference recipes

**Please consider the above context when you answer user questions or do any operations!**

