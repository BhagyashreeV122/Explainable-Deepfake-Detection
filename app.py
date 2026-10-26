"""Gradio local web interface for Deepfake Detection & Explainability.

Upload any image to detect whether it is Real or Deepfake, inspect
the confidence scores, and visualize the model's Grad-CAM attention heatmap.
"""

from pathlib import Path
import sys

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from src.inference.predict import InferencePipeline

# Global pipeline instance initialized on app launch
pipeline: InferencePipeline = None


def get_pipeline() -> InferencePipeline:
    global pipeline
    if pipeline is None:
        pipeline = InferencePipeline(
            config_path="configs/config.yaml",
            checkpoint_path="outputs/checkpoints/best_model.pt",
        )
    return pipeline


def analyze_image(input_img):
    if input_img is None:
        return (
            "<div class='warning-box'>⚠️ Please upload an image first.</div>",
            {"Real": 0.0, "Fake": 0.0},
            None,
            None,
            None,
        )

    pipe = get_pipeline()
    # Run prediction
    res = pipe.predict(image_input=input_img, save_outputs=True, output_dir="outputs/inference")

    verdict = res["verdict"]
    prob_real = res["prob_real"]
    prob_fake = res["prob_fake"]
    face_detected = res["face_detected"]

    # Format verdict card HTML
    is_fake = (verdict.upper() == "FAKE")
    verdict_badge_class = "verdict-fake" if is_fake else "verdict-real"
    icon = "🚨" if is_fake else "✅"

    warning_html = ""
    if not face_detected:
        warning_html = (
            "<div class='warning-banner'>"
            "<strong>⚠️ Warning:</strong> No human face detected. Inference was performed on the full image fallback."
            "</div>"
        )

    verdict_html = f"""
    <div class='verdict-card {verdict_badge_class}'>
        <div class='verdict-title'>{icon} Prediction: <strong>{verdict}</strong></div>
        <div class='verdict-sub'>Confidence: Real {prob_real*100:.1f}% &nbsp;|&nbsp; Fake {prob_fake*100:.1f}%</div>
        {warning_html}
    </div>
    """

    label_confidences = {
        "Real": float(prob_real),
        "Fake": float(prob_fake),
    }

    face_crop = res["face_crop"]
    overlay = res["heatmap_overlay"]
    combined = res["combined_panel"]

    return verdict_html, label_confidences, face_crop, overlay, combined


CUSTOM_CSS = """
:root {
    --bg-dark: #0f1117;
    --card-dark: #1b1e2b;
    --border-dark: #2d3348;
    --accent: #6366f1;
    --accent-hover: #4f46e5;
    --success: #10b981;
    --danger: #ef4444;
}

body {
    background-color: var(--bg-dark);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 24px !important;
}

.header-container {
    text-align: center;
    padding: 28px 20px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%);
    border: 1px solid var(--border-dark);
    border-radius: 16px;
    margin-bottom: 24px;
}

.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.header-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 750px;
    margin: 0 auto;
}

.verdict-card {
    padding: 18px 20px;
    border-radius: 12px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    transition: all 0.2s ease;
}

.verdict-real {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
    border: 1.5px solid #10b981;
    color: #ecfdf5;
}

.verdict-fake {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.08) 100%);
    border: 1.5px solid #ef4444;
    color: #fef2f2;
}

.verdict-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 4px;
}

.verdict-sub {
    font-size: 1.05rem;
    opacity: 0.9;
}

.warning-banner {
    margin-top: 10px;
    padding: 8px 12px;
    border-radius: 8px;
    background-color: rgba(245, 158, 11, 0.2);
    border: 1px solid #f59e0b;
    color: #fbbf24;
    font-size: 0.9rem;
}

.disclaimer-box {
    margin-top: 24px;
    padding: 14px 18px;
    border-radius: 10px;
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid #334155;
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.5;
}

button.primary-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: white !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

button.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
}
"""


def create_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    ).set(
        body_background_fill="#0b0f19",
        block_background_fill="#131b2e",
        block_border_color="#1f293d",
    )

    with gr.Blocks(title="Deepfake Detector & Grad-CAM Explainability", theme=theme, css=CUSTOM_CSS) as demo:
        with gr.Column(elem_classes=["header-container"]):
            gr.HTML(
                """
                <h1 class='header-title'>🛡️ Deepfake Detector & Explainability Studio</h1>
                <p class='header-subtitle'>
                    Production-grade real vs. fake face authentication powered by EfficientNet-B0 trained on
                    FaceForensics++ (C23) with real-time Grad-CAM saliency heatmaps.
                </p>
                """
            )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="Upload Image (Face / Portrait)",
                    elem_id="input_image_box",
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze Image & Explain",
                    variant="primary",
                    elem_classes=["primary-btn"],
                    elem_id="btn_analyze",
                )

                gr.Examples(
                    examples=[
                        "data/faces/test/real/test_real_031_f0048.png",
                        "data/faces/test/fake/test_fake_031_f0048.png",
                        "data/faces/test/real/test_real_038_f0048.png",
                        "data/faces/test/fake/test_fake_038_f0048.png",
                    ],
                    inputs=input_image,
                    label="Sample Benchmark Crops (Real & Fake)",
                )

            with gr.Column(scale=1):
                verdict_output = gr.HTML(
                    """
                    <div style='padding: 20px; text-align: center; color: #64748b; border: 1px dashed #334155; border-radius: 12px;'>
                        Upload an image and click <strong>Analyze Image & Explain</strong> to see predictions.
                    </div>
                    """
                )
                confidence_output = gr.Label(
                    num_top_classes=2,
                    label="Classification Probabilities",
                    elem_id="confidence_meter",
                )

                with gr.Tab("Visual Explainability"):
                    with gr.Row():
                        face_crop_output = gr.Image(
                            label="Extracted Face Crop (224x224)",
                            elem_id="img_face_crop",
                        )
                        heatmap_output = gr.Image(
                            label="Grad-CAM Saliency Overlay",
                            elem_id="img_heatmap",
                        )

                with gr.Tab("Combined 3-Panel Card"):
                    combined_output = gr.Image(
                        label="Full Diagnostic Summary Panel",
                        elem_id="img_combined_panel",
                    )

        with gr.Row():
            gr.HTML(
                """
                <div class='disclaimer-box'>
                    <strong>⚠️ Important Forensic Disclaimer:</strong>
                    The classification verdict and probabilities presented above are statistical estimates generated by a machine-learning model (EfficientNet-B0) trained on the FaceForensics++ C23 benchmark dataset. Deepfake detection models can exhibit false positives or false negatives, especially when tested on unseen manipulation methods, extreme lighting, or heavy social-media compression. Explainability heatmaps reflect regions that positively influenced the model's feature representations, not legal or forensic proof of manipulation.
                </div>
                """
            )

        # Connect button click
        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image],
            outputs=[
                verdict_output,
                confidence_output,
                face_crop_output,
                heatmap_output,
                combined_output,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
