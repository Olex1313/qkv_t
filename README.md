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