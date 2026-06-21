---
marp: true
theme: default
paginate: true
size: 16:9
footer: "Vision-to-Blueprint · Franz Antony Bendezu Isidro"
style: |
  :root {
    --ink: #17324d;
    --muted: #5f7182;
    --paper: #f7f3ea;
    --panel: #ffffff;
    --teal: #168b8c;
    --teal-dark: #0c6267;
    --wood: #d88935;
    --line: #d9e1df;
  }

  section {
    background:
      linear-gradient(135deg, rgba(22, 139, 140, 0.07), transparent 38%),
      linear-gradient(315deg, rgba(216, 137, 53, 0.08), transparent 35%),
      var(--paper);
    color: var(--ink);
    font-family: "Aptos", "Segoe UI", sans-serif;
    font-size: 23px;
    line-height: 1.28;
    padding: 44px 58px 48px;
  }

  section::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 8px;
    background: linear-gradient(90deg, var(--teal-dark), var(--teal) 68%, var(--wood));
  }

  h1, h2, h3 {
    font-family: "Aptos Display", "Segoe UI Semibold", sans-serif;
    color: var(--ink);
    letter-spacing: -0.025em;
  }

  h1 {
    color: var(--teal-dark);
    font-size: 34px;
    margin: 0 0 12px;
  }

  h2 {
    font-size: 27px;
    margin: 10px 0 8px;
  }

  h3 {
    font-size: 23px;
  }

  strong {
    color: var(--teal-dark);
  }

  code {
    background: #e8efec;
    color: #9b541c;
    border-radius: 5px;
    padding: 0.08em 0.28em;
  }

  ul, ol {
    padding-left: 1.25em;
  }

  li {
    margin: 0.16em 0;
  }

  table {
    background: rgba(255, 255, 255, 0.82);
    border-radius: 10px;
    box-shadow: 0 8px 24px rgba(23, 50, 77, 0.08);
    font-size: 20px;
  }

  th {
    background: var(--teal-dark);
    color: white;
  }

  img {
    border-radius: 10px;
    box-shadow: 0 8px 22px rgba(23, 50, 77, 0.16);
  }

  em {
    color: var(--muted);
    font-size: 0.82em;
  }

  footer {
    color: var(--muted);
    font-size: 12px;
  }

  section.lead {
    background:
      radial-gradient(circle at 82% 22%, rgba(216, 137, 53, 0.32), transparent 22%),
      linear-gradient(135deg, #102d46 0%, #0d6167 100%);
    color: #eef8f5;
  }

  section.lead::before {
    background: var(--wood);
  }

  section.lead h1,
  section.lead h2,
  section.lead strong {
    color: #ffffff;
  }

  section.lead h1 {
    font-size: 38px;
    opacity: 0.86;
  }

  section.lead h2 {
    font-size: 42px;
    line-height: 1.12;
    max-width: 980px;
  }

  section.lead footer {
    color: rgba(255, 255, 255, 0.7);
  }

  section.closing {
    background:
      radial-gradient(circle at 10% 85%, rgba(216, 137, 53, 0.18), transparent 28%),
      linear-gradient(145deg, #edf7f4, #f7f3ea 62%);
  }
---

<!-- _class: lead -->

# Vision-to-Blueprint

## An Intelligent AI Convergence System for Automated Furniture Decomposition and Personalized Construction

**Franz Antony Bendezu Isidro**  
Student ID: **2024115282**  
AI Convergence Project · Week 16

---

# 1. Problem and Scope

- **AS-IS:** manual measurements, trial and error, material waste.
- **TO-BE:** image → editable model → validated exports.
- **MVP:** product categorization plus taxonomy-based generation.
- **Limit:** no fabrication-accurate 3D reconstruction from one photo.

<p align="center">
  <img src="assets/wardrobe_sample.jpg" alt="Wardrobe input" width="31%">
  <img src="assets/shelf_sample.jpg" alt="Shelf input" width="31%">
  <img src="assets/desk_sample.jpg" alt="Desk input" width="31%">
</p>

---

# 2. Computational Thinking

- **Decomposition:** data, inference, model, validation, export.
- **Patterns:** panels, shelves, doors, drawers, joints.
- **Abstraction:** keep geometry and topology; remove scene noise.
- **Sequence:** upload → infer → generate → validate → export.
- **Selection:** invalid data is rejected.
- **Loop:** edits regenerate and revalidate the model.

---

# 3. End-to-End Flow

```mermaid
flowchart LR
    A[Project] --> B[Images]
    B --> C[Job]
    C --> D[Inference]
    D --> E[Parametric Model]
    E --> F{Valid?}
    F -- No --> G[Edit or Retry]
    G --> E
    F -- Yes --> H[PDF · CSV · DXF · ZIP]
```

- **Input:** images and project parameters.
- **Evidence:** category, confidence, per-image result.
- **Output:** validated model and files.

<p align="center">
  <img src="assets/frontend_workspace.png" alt="Workspace" width="47%">
  <img src="assets/frontend_jobs.png" alt="Jobs" width="47%">
</p>

---

# 4. Why Jobs Matter

- **Trace:** image → inference → model.
- **Retry:** create a new attempt without deleting history.
- **Compare:** evaluate future model or taxonomy changes.
- **Debug:** isolate asset, inference, contract, or validation errors.
- **Scale:** ready for asynchronous workers.

![Jobs view](assets/frontend_jobs.png)

---

# 5. Runtime Architecture

```mermaid
sequenceDiagram
    actor U as User
    participant F as Frontend
    participant B as Backend
    participant I as Inference
    participant D as Database

    U->>F: Upload images
    F->>B: Create job
    B->>I: Run inference
    I-->>B: Category + evidence
    B->>B: Generate and validate
    B->>D: Save job and model
    B-->>F: Status + result
    U->>B: Export files
```

**API groups:** projects · assets · jobs · validation · exports

---

# 6. Domain Model

```mermaid
classDiagram
    ProjectModel --> ProductSpec
    ProjectModel --> Component
    ProjectModel --> MaterialSpec
    ProjectModel --> HardwareItem
    ProjectModel --> JointSpec
    ProjectModel --> FeatureSpec
    Component --> FaceSpec
    InferenceOutput --> InferenceComponent
    InferenceOutput --> InferenceImageEvidence
```

- `ProductSpec`: dimensions and counts.
- `Component`: physical part.
- `FaceSpec` + `JointSpec`: assembly relation.
- `InferenceOutput`: strict backend input contract.
- `ProjectModel`: source for editing, validation, and export.

<p align="center">
  <img src="assets/runtime_database_erd.png" alt="Database ERD" width="47%">
  <img src="assets/frontend_model.png" alt="Model view" width="47%">
</p>

---

# 7. Data and Training

- Source: Promart catalog endpoints.
- Preparation: null checks, normalization, splits, pseudolabels.
- Data quality: 0 missing values in key image-metadata fields.
- Active split status: train 2,400 · validation 600 · test 0 (MVP stage).

| Evidence | Count |
| :-- | --: |
| Products | 162 |
| Catalog images | 1,124 |
| Usable images | 1,124 |
| YOLO train | 2,400 |
| YOLO validation | 600 |

<p align="center">
  <img src="assets/training_results_overview.png" alt="Training metrics" width="31%">
  <img src="assets/training_pr_curve.png" alt="PR curve" width="31%">
  <img src="assets/training_confusion_matrix_norm.png" alt="Confusion matrix" width="31%">
</p>

---

# 8. Implemented MVP

- Project-first web workflow.
- Multi-image inference contract.
- Parametric dimensions and topology controls.
- Geometry and reference validation.
- PDF, CSV, DXF, and ZIP exports.

| Test suite | Result |
| :-- | --: |
| Inference | 24 / 24 passed |
| Backend | 24 / 24 passed |
| Combined | 48 / 48 passed |

*Note: inference tests also report 38 non-blocking dependency warnings (torch deprecation and optional SAM2 extension).*

<p align="center">
  <img src="assets/frontend_home.png" alt="Home" width="31%">
  <img src="assets/frontend_project_status.png" alt="Project status" width="31%">
  <img src="assets/frontend_workspace.png" alt="Workspace" width="31%">
</p>

---

# 9. Capability and Limits

| Current capability | Current limit |
| :-- | :-- |
| Furniture categorization | Taxonomy-dependent classes |
| Multi-image evidence | Occlusion and perspective errors |
| Editable parametric model | No calibrated dimensions from one photo |
| Rule-based validation | Human fabrication review required |
| Row-based nesting export | No global waste optimization |
| Contract-ready regression pipeline | No active trained regression model in MVP |

<p align="center">
  <img src="assets/training_val_batch_pred.jpg" alt="Prediction sample" width="47%">
  <img src="assets/frontend_model.png" alt="Editable model" width="47%">
</p>

---

# 10. Next Steps

1. **Quality:** maintain test coverage and version every job.
2. **Data:** review YOLO-assisted labels with human approval.
3. **Vision:** improve component detection, segmentation, and calibration.
4. **Jobs:** add queues, retries, progress, and result comparison.
5. **Fabrication:** improve nesting and validate with furniture makers.

---

<!-- _class: closing -->

# 11. Individual Work and Closing

- One developer covered data, training, inference, backend, frontend, exports, tests, and documentation.
- Future work needs annotation, computer vision, furniture/CAD, QA, and UX support.

<p align="center">
  <img src="assets/frontend_fabrication_output.png" alt="Fabrication output" width="32%">
  <img src="assets/mock_wardrobe_blueprint_from_endpoint.png" alt="Blueprint" width="32%">
  <img src="assets/mock_wardrobe_bom_from_endpoint.png" alt="BOM" width="32%">
</p>

**Result:** a traceable path from image evidence to editable fabrication files.
