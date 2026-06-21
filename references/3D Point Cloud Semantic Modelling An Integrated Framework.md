# Academic Context: 3D Point Cloud Semantic Modelling

## 1. Metadata
*   **Authors:** Xuan Nguyen, et al.
*   **Source URL:** [ResearchGate Repository](https://researchgate.net)

## 2. Core Problem & Hypotheses
*   Investigates how to convert unorganized, dense 3D spatial points (from LiDAR or Structure-from-Motion photogrammetry) into highly structured, semantically labeled BIM/CAD geometry.
*   It assumes that interior assets—especially modular panel setups—can be accurately decomposed into individual bounding volumes by analyzing spatial adjacency graphs and geometric primitives rather than relying entirely on neural network classification.

## 3. Methodology & Architecture
*   **Planar Extraction via RANSAC:** The framework runs iterative Random Sample Consensus (RANSAC) math layers to extract major flat surfaces, automatically categorizing points into structural frames (carcass walls) and internal divisions.
*   **Spatial Adjacency Graphs (SAG):** Once individual planes are isolated, the system constructs a relational graph measuring proximity and perpendicular/parallel alignment.
*   **Geometric Rectification:** By feeding 2D neural network hints (like those from YOLO) into this spatial map, jagged point cloud clusters are snapped into exact 3D bounding cubes, correcting sensor noise along the edges of the panels.
