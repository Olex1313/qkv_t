### Сварить onnx

```bash
cd contrib/MNN-simiyutin/build$
./MNNConvert -f ONNX \
--modelFile ../../LightGlue-ONNX-simiyutin/weights/lightglue_aliked_3840.onnx \
--transformerFuse \
--MNNModel lightglue_matcher_large.mnn \
--saveStaticModel \
--optimizeLevel 1
```

### Сконфигурить флажки MNN "правильно"

```bash
#!/bin/sh

cmake .. \
-DMNN_BUILD_CONVERTER=ON \
-DMNN_VULKAN=OFF \
-DMNN_VULKAN_IMAGE=OFF \
-DMNN_OPENCL=ON \
-DMNN_CUDA=ON \
-DMNN_OPENGL=OFF \
-DMNN_BUILD_BENCHMARK=OFF \
-DMNN_USE_THREAD_POOL=ON \
-DMNN_USE_SSE=ON \
-DMNN_SEP_BUILD=OFF \
-DMNN_GPU_TIME_PROFILE=OFF \
-DMNN_BUILD_PROTOBUFFER=ON \
-DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
-DMNN_QNN=OFF \
-DMNN_QNN_CONVERT_MODE=OFF \
-DMNN_WITH_PLUGIN=OFF \
-DMNN_BUILD_TOOLS=ON \
-DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```