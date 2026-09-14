from pathlib import Path

import joblib
from onnxruntime.quantization import QuantType, quantize_dynamic
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline as SklearnPipeline

ARTIFACTS_DIR = Path("model/artifacts")
SKLEARN_MODEL_PATH = ARTIFACTS_DIR / "baseline_model.joblib"
ONNX_FP32_PATH = ARTIFACTS_DIR / "model_fp32.onnx"
ONNX_INT8_PATH = ARTIFACTS_DIR / "model_int8.onnx"


def extract_inference_pipeline(imblearn_pipeline) -> SklearnPipeline:
    """O RandomOverSampler só age no fit, nunca no predict - então, para
    exportar, remontamos só os passos que participam da inferência
    (tfidf + clf, já treinados) num Pipeline sklearn padrão, que é o que
    o skl2onnx sabe interpretar."""
    tfidf = imblearn_pipeline.named_steps["tfidf"]
    clf = imblearn_pipeline.named_steps["clf"]
    return SklearnPipeline([("tfidf", tfidf), ("clf", clf)])


def main():
    print(f"Carregando pipeline sklearn de {SKLEARN_MODEL_PATH}...")
    full_pipeline = joblib.load(SKLEARN_MODEL_PATH)
    inference_pipeline = extract_inference_pipeline(full_pipeline)

    print("Convertendo para ONNX (precisão original, FP32)...")
    onnx_model = convert_sklearn(
        inference_pipeline,
        initial_types=[("input", StringTensorType([None, 1]))],
        target_opset=17,
    )
    with open(ONNX_FP32_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Salvo em {ONNX_FP32_PATH} ({ONNX_FP32_PATH.stat().st_size / 1024:.1f} KB)")

    print("\nAplicando quantização dinâmica (pesos em INT8)...")
    quantize_dynamic(
        model_input=str(ONNX_FP32_PATH),
        model_output=str(ONNX_INT8_PATH),
        weight_type=QuantType.QInt8,
    )
    print(f"Salvo em {ONNX_INT8_PATH} ({ONNX_INT8_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
