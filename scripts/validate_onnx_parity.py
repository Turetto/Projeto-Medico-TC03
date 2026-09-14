from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort

ARTIFACTS_DIR = Path("model/artifacts")

SAMPLE_TEXTS = [
    "Patient presented with chest pain and elevated troponin levels consistent with myocardial infarction",
    "Biopsy revealed abnormal cell proliferation suggestive of malignant neoplasm in the colon",
    "MRI showed lesions in the white matter consistent with demyelinating nervous system disease",
    "Patient reports chronic abdominal pain with symptoms of inflammatory bowel disease",
    "General weakness and fatigue with nonspecific laboratory findings",
]


def predict_onnx(session, texto: str) -> int:
    input_name = session.get_inputs()[0].name
    input_array = np.array([[texto]], dtype=object)
    outputs = session.run(None, {input_name: input_array})
    return int(outputs[0][0])


def main():
    sklearn_model = joblib.load(ARTIFACTS_DIR / "baseline_model.joblib")
    session_fp32 = ort.InferenceSession(str(ARTIFACTS_DIR / "model_fp32.onnx"))
    session_int8 = ort.InferenceSession(str(ARTIFACTS_DIR / "model_int8.onnx"))

    divergencias = 0
    for texto in SAMPLE_TEXTS:
        pred_sklearn = int(sklearn_model.predict([texto])[0])
        pred_fp32 = predict_onnx(session_fp32, texto)
        pred_int8 = predict_onnx(session_int8, texto)

        status = "OK" if (pred_sklearn == pred_fp32 == pred_int8) else "DIVERGIU"
        if status == "DIVERGIU":
            divergencias += 1
        print(f"[{status}] sklearn={pred_sklearn} onnx_fp32={pred_fp32} onnx_int8={pred_int8} | {texto[:50]}...")

    print(f"\n{len(SAMPLE_TEXTS) - divergencias}/{len(SAMPLE_TEXTS)} concordam entre os 3 modelos.")


if __name__ == "__main__":
    main()
