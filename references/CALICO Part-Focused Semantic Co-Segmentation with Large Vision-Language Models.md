# Academic Context: CALICO - Part-Focused Semantic Co-Segmentation with LVLMs

## 1. Metadata
*   **Authors:** Tuan Nguyen, et al.
*   **Conference:** IEEE/CVF CVPR 2025.
*   **Source URL:** [Computer Vision Foundation Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_CALICO_Part-Focused_Semantic_Co-Segmentation_with_Large_Vision-Language_Models_CVPR_2025_paper.pdf)
*   **Code & Model:** [Hugging Face PLAN-Lab/CALICO](https://huggingface.co/PLAN-Lab/CALICO)

## 2. Core Problem & Hypotheses
*   Standard Large Vision-Language Models (LVLMs) can perform text-prompted segmentation on isolated single images but fail on **segmentation-grounded reasoning across multiple frames**.
*   This paper formalizes the task of **part-focused semantic co-segmentation**: identifying common objects, as well as tracking common or completely unique fine-grained parts (like individual modular panels or handles) across several multi-angle views.

## 3. Methodology & Architecture
*   **The CALICO Architecture:** Built upon a Vicuna-based LLM, it integrates a Q-Former over EVA-CLIP-G visual features and uses a Segment Anything (SAM ViT-H) mask decoder for pixel-perfect zero-shot outputs.
*   **Correspondence Extraction Module (CEM):** It utilizes frozen DINOv2 self-supervised features to capture cross-image semantic relationships. This enables the model to understand that a door seen from the left side in Image A is the same identical component seen from the front in Image B.
*   **Parameter Efficiency:** By training adapter modules on less than 0.3% of the network’s original parameters using the *MixedParts* dataset (~2.4M samples), it provides a fast, zero-shot alternative to custom dataset training for parts parsing.
