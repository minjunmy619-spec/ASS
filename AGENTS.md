# Audio sound separation model for online/realtime/causal deployment on edge device with NPU

## Project Context
```
local WSL workspace： /home/cmj/works/ASS
ONE compiler root: /home/cmj/works/ONE
```

**Please ignore the "dcase2026baseline" subfolder for this NPU compilation work.**
Please first check the whole project structure，it may changes, so when needed, please re-check.
The project constructure:
```
ASS         #The root dir for the project, your workspace
├── .venv           #the virtual enviroment you should use for running/testing in the docker environment
├── 2602.08671v1.pdf
├── ASS
├── ASS.code-workspace
├── AGENT.md                            #This file With the high-level guide for the NPU compilation tasks
├── OPERATION_MANUAL_PYTORCH_TO_ONE_NPU #The detailed guide for executing the specific NPU compilation with ONE compiler
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

The below is the ONE compiler toolchain, you should check the "/home/cmj/works/ASS/OPERATION_MANUAL_PYTORCH_TO_ONE_NPU.md" for how to use it.
```
ONE
├── CONTRIBUTORS
├── COPYRIGHT
├── LICENSE
├── Makefile.template
├── README.md
├── build
│   ├── CMakeCache.txt
│   ├── CMakeFiles
│   ├── CTestTestfile.cmake
│   ├── DartConfiguration.tcl
│   ├── Makefile
│   ├── Testing
│   ├── cmake_install.cmake
│   ├── compile_commands.json
│   ├── compiler
│   ├── externals
│   ├── install_manifest.txt
│   └── overlay
├── circle-mlir
│   ├── CMakeLists.txt
│   ├── Makefile.sample
│   ├── README.md
│   ├── circle-mlir
│   ├── externals
│   ├── infra
│   └── models
├── compiler
│   ├── CMakeLists.txt
│   ├── _deprecated
│   ├── adtidas
│   ├── angkor
│   ├── arser
│   ├── bcq-tools
│   ├── bino
│   ├── circle-eval-diff
│   ├── circle-execution-plan
│   ├── circle-input-names
│   ├── circle-inspect
│   ├── circle-interpreter
│   ├── circle-interpreter-cffi-test
│   ├── circle-interpreter-test
│   ├── circle-mpqsolver
│   ├── circle-operator
│   ├── circle-operator-test
│   ├── circle-opselector
│   ├── circle-part-driver
│   ├── circle-part-value-py-test
│   ├── circle-part-value-test
│   ├── circle-partitioner
│   ├── circle-partitioner-test
│   ├── circle-quantizer
│   ├── circle-quantizer-dredd-recipe-test
│   ├── circle-resizer
│   ├── circle-resizer-dredd-recipe-test
│   ├── circle-tensordump
│   ├── circle-verify
│   ├── circle2circle
│   ├── circle2circle-dredd-recipe-test
│   ├── circlechef
│   ├── circledump
│   ├── cli
│   ├── common-artifacts
│   ├── crew
│   ├── cwrap
│   ├── dalgona
│   ├── dalgona-test
│   ├── dio-hdf5
│   ├── dredd-rule-lib
│   ├── embedded-import-value-test
│   ├── exo
│   ├── fipe
│   ├── fm-equalize
│   ├── fm-equalize-value-py-test
│   ├── fme-apply
│   ├── fme-detect
│   ├── foder
│   ├── hermes
│   ├── hermes-std
│   ├── i5diff
│   ├── kuma
│   ├── loco
│   ├── locoex-customop
│   ├── locomotiv
│   ├── locop
│   ├── logo
│   ├── logo-core
│   ├── logo-ex
│   ├── luci
│   ├── luci-compute
│   ├── luci-eval-driver
│   ├── luci-interpreter
│   ├── luci-pass-value-py-test
│   ├── luci-pass-value-test
│   ├── luci-ref-value-py-test
│   ├── luci-value-py-test
│   ├── luci-value-test
│   ├── minmax-embedder
│   ├── minmax-embedder-value-test
│   ├── mio-circle
│   ├── mio-tf
│   ├── mio-tflite
│   ├── moco-log
│   ├── morph
│   ├── nest
│   ├── nike
│   ├── nnop
│   ├── one-cmds
│   ├── one-global-conf-template
│   ├── onecc-docker
│   ├── oneco
│   ├── oneco-value-pbtxt-test
│   ├── onnx-tools
│   ├── onnxkit
│   ├── oops
│   ├── pepper-assert
│   ├── pepper-csv2vec
│   ├── pepper-env
│   ├── pepper-str
│   ├── pepper-strcast
│   ├── pics
│   ├── plier-tf
│   ├── pota-quantization-value-test
│   ├── pp
│   ├── q-implant
│   ├── q-implant-qparam-test
│   ├── rawdata2hdf5
│   ├── record-minmax
│   ├── record-minmax-conversion-test
│   ├── record-minmax-thread-safety-test
│   ├── safemain
│   ├── souschef
│   ├── tf2nnpackage-value-remote-test
│   ├── tf2tfliteV2
│   ├── tf2tfliteV2-conversion-test
│   ├── tfinfo-v2
│   ├── tfkit
│   ├── tfl-inspect
│   ├── tfl-verify
│   ├── tflchef
│   ├── tfldump
│   ├── tflite2circle
│   ├── tflite2circle-conversion-test
│   ├── v4tf
│   ├── vconone
│   ├── visq
│   └── visq-unittest
├── docs
│   ├── Makefile
│   ├── common-ir
│   ├── compiler
│   ├── conf.py
│   ├── contents.rst
│   ├── device
│   ├── howto
│   ├── images
│   ├── index.rst
│   ├── make.bat
│   ├── nncc
│   ├── overview
│   ├── package
│   ├── platform
│   ├── release
│   ├── requirements.txt
│   ├── runtime
│   └── test
├── externals
│   ├── FLATBUFFERS-23.5.26
│   ├── FLATBUFFERS-23.5.26.stamp
│   ├── FP16
│   ├── FP16.stamp
│   ├── GTEST
│   ├── GTEST.stamp
│   ├── HDF5
│   ├── HDF5.stamp
│   ├── JSONCPP
│   ├── JSONCPP.stamp
│   ├── LIBNPY
│   ├── LIBNPY.stamp
│   ├── NEON2SSE
│   ├── NEON2SSE.stamp
│   ├── PROTOBUF
│   ├── PROTOBUF.stamp
│   ├── PYBIND11
│   ├── PYBIND11.stamp
│   ├── TENSORFLOW-2.19.0
│   ├── TENSORFLOW-2.19.0-EIGEN
│   ├── TENSORFLOW-2.19.0-EIGEN-eigen-33d0937c6bdf5ec999939fb17f2a553183d14a74.tar.gz
│   ├── TENSORFLOW-2.19.0-EIGEN.stamp
│   ├── TENSORFLOW-2.19.0-GEMMLOWP
│   ├── TENSORFLOW-2.19.0-GEMMLOWP.stamp
│   ├── TENSORFLOW-2.19.0-RUY
│   ├── TENSORFLOW-2.19.0-RUY.stamp
│   ├── TENSORFLOW-2.19.0-THREADPOOL
│   ├── TENSORFLOW-2.19.0-THREADPOOL.stamp
│   └── TENSORFLOW-2.19.0.stamp
├── infra
│   ├── cmake
│   ├── command
│   ├── config
│   ├── debian
│   ├── docker
│   ├── doxygen
│   ├── git-hooks
│   ├── nncc
│   ├── onert-micro
│   ├── packaging
│   └── scripts
├── logo
│   ├── ONE_logo_guideline_ENG.pdf
│   ├── ONE_logo_guideline_KOR.pdf
│   └── images
├── nnas
├── nncc
├── nnfw
├── nnpackage
│   ├── examples
│   ├── schema
│   └── spec
├── onert-micro
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── cmake
│   ├── eval-driver
│   ├── examples
│   ├── externals
│   ├── git_version.h.in
│   ├── helpers
│   ├── luci-interpreter
│   ├── onert-micro
│   ├── remove_stablehlo_from_cir0.8.patch
│   ├── requires.cmake
│   ├── standalone
│   ├── tests
│   └── training-configure-tool
├── packaging
│   ├── CPUINFO.tar.gz
│   ├── EIGEN.tar.gz
│   ├── EXTERNALS_FOR_ODC.tar.gz
│   ├── GEMMLOWP.tar.gz
│   ├── RUY.tar.gz
│   ├── SENTENCEPIECE.tar.gz
│   ├── nnfw.manifest
│   └── nnfw.spec
├── pyproject.toml
├── res
│   ├── BVLCCaffeTests
│   ├── CircleRecipes
│   ├── CircleSchema
│   ├── ONNXTests
│   ├── PyTorchExamples
│   ├── TensorFlowLiteRecipes
│   ├── TensorFlowLiteSchema
│   ├── TensorFlowPythonExamples
│   ├── TensorFlowPythonModels
│   └── TensorFlowTests
├── runtime
│   ├── 3rdparty
│   ├── CMakeLists.txt
│   ├── coding-rules.md
│   ├── compute
│   ├── contrib
│   ├── ggma
│   ├── infra
│   ├── libs
│   ├── onert
│   ├── pyproject.toml
│   ├── tests
│   └── tools
└── tools
    ├── circle_plus_gen
    ├── cross
    ├── extract_weights_from_tflite
    ├── generate_datafile
    ├── image_importer
    ├── kernel_report
    ├── model_explorer_circle
    ├── model_partition_tool
    ├── nnpackage_tool
    ├── onnx_subgraph
    ├── pareto_profiler
    ├── pbfile_tool
    ├── release_tool
    ├── stab
    ├── tensorflow_model_freezer
    ├── tflitefile_tool
    └── tflkit
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
rule15: The model parameters should less than 7M, and the GMacs should less than 3GMacs/s
rule16: Other limitations imposed by the ONE compiler
```

## Validation
Because we do not have the finally deploy device with the NPU on hand, we could just validate the conversion/compliation by following steps:
1. Write suitable test scripts to valide model structure, make use it respect to the rules
2. Write test scripts to validate the training/inference pipline make the inputs/outputs are consistant
3. Export the model to onnx format
4. Export the onnx model to circle with the ONE compilation toolchain, please following the guide at "/home/cmj/works/ASS/OPERATION_MANUAL_PYTORCH_TO_ONE_NPU.md", it should finish the whole procedure successfully: import -> optimization -> quantization
Loop these steps to get a finally working model.


---
## Other important points/notes:
1. Remember to write a markdown file(task_prefix_operation.md) to record the tries and changes, with detailed explaination for changes/designs and examples for ran commands
2. Use multip version for different tries/experiments, not just overwrite the origin versions, make sure the names are brief and representative 
3. You should also prepare the training/inference recipes, make sure they are consistant and working well

**Please consider the above context when you answer user questions or do any operations!**

