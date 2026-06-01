# **Intelligent Information System for Image-Based Structural Recognition and Personalized DIY Furniture Design**


# **I. Abstract**

Advances in artificial intelligence and data-driven design are expanding access to tools that were once limited to trained professionals. In DIY furniture making, however, a clear gap remains between finding a reference image and producing a design that can actually be built. Non-expert users may recognize the form they want, but they often cannot determine the panel dimensions, joint relationships, and hardware requirements needed for construction. This limitation frequently leads to design errors, wasted materials, and repeated trial and error.  
This project proposes a web-based intelligent information system for image-based recognition and personalized furniture design, in which a CNN-based vision module is used together with supporting methods for structural segmentation, parametric reconstruction, and fabrication planning of panel-based furniture, such as cabinets, shelves, and desks. These components are stored as an editable parametric model consisting of panels, joints, materials, and hardware rules. In the user interface, the model appears in an interactive 3D workspace where users can change width, height, depth, shelf count, and surface finish through parameter controls.  
Users can also select individual panels and reposition or resize them directly, while the parametric engine recalculates dependent dimensions, preserves valid spatial relationships, and flags unsupported structural changes. As the design is edited, the system updates joint definitions, infers the required hardware, and prepares a 3D preview, printable 2D blueprints, a bill of materials, and nesting layouts for efficient material use. By combining image recognition with parametric editing and fabrication output, the system reduces technical barriers and supports more accurate and resource-efficient DIY furniture production.

# **II. Problem Definition and Gap Analysis**

The starting point of this study is a clear definition of the practical gap faced by DIY furniture makers. In the current workflow, users often begin with an image of a product they would like to reproduce, but they lack the tools to convert that image into a technically valid design. The proposed system addresses this gap by turning visual references into editable parametric models and fabrication-ready output.

* **Current State (AS-IS):** Non-expert DIY furniture construction is still largely manual and error-prone. Survey data indicates that 70% of DIY practitioners have avoided home improvement projects because of the "Fear of Messing Up" (FOMU), while 58% of projects exceed budget because of dimensional mistakes and material waste (url reference : [http://www.prnewswire.com/news-releases/nailing-it-or-failing-it-new-study-reveals-70-of-diy-ers-have-avoided-projects-over-fomu-fear-of-messing-up-302151548.html](http://www.prnewswire.com/news-releases/nailing-it-or-failing-it-new-study-reveals-70-of-diy-ers-have-avoided-projects-over-fomu-fear-of-messing-up-302151548.html)). Users also face difficulty selecting appropriate hardware, such as hinges and fasteners, for panel-based furniture.  
* **Target State (TO-BE):** The target outcome is a web-based "Vision-to-Blueprint" workflow in which users upload a furniture image, receive an editable parametric model, and export construction documents with minimal manual drafting. The system should also update topology, hardware requirements, and nesting layouts automatically as the user modifies the design.  
* **Problem Classification:** From a computational perspective, the project combines **Computational** problems, such as translating 2D image data into structured 3D parametric representations, with **Optimization** problems, particularly the reduction of material waste through 2D nesting.

*The following Use Case Diagram establishes the functional boundaries of this target state, illustrating how the DIY User interacts with the hybrid vision pipeline and the Parametric Engine to bridge the identified gap:*

``` 
@startuml
left to right direction
actor "DIY User" as User
actor "Hybrid Vision Pipeline" as AI
database "Parametric Parts & Hardware DB" as DB

rectangle "Web-Based Intelligent Furniture System" {
  usecase "Upload Image of Panel Furniture" as UC1
  usecase "Hybrid Structural Recognition (CNN + rjMCMC)" as UC2
  usecase "Adjust Topology via Sliders" as UC3
  usecase "Infer Assembly Hardware (Cam/Bolts)" as UC4
  usecase "Real-Time 3D Visualization" as UC5
  usecase "Generate 2D Nesting & Blueprints" as UC6
}

User --> UC1
User --> UC3
UC1 --> UC2
AI --> UC2
UC2 --> DB
DB --> UC4
UC3 --> UC4
UC4 --> UC5
UC5 --> UC6
User --> UC6

note right of UC4: Automatically aligns with\nDesign for Disassembly (DfD)\nhardware rules.
@enduml
```


# **III. Problem Decomposition**

To make the problem tractable, the proposed system is divided into four main subproblems, each corresponding to a distinct stage in the pipeline:

* **Data Acquisition:** Collecting and preprocessing images and metadata for panel-based furniture, such as cabinets and wardrobes, from retail platforms. This stage follows a Knowledge Discovery in Databases (KDD) approach so that the input data can be selected, cleaned, and standardized before model training.  
* **Structural Recognition:** Segmenting a 2D image into hierarchical furniture components, including doors, drawers, and shelves. At this stage, one possible supporting method for structural interpretation is reversible jump Markov Chain Monte Carlo (rjMCMC), which can be used to estimate spatial structure and functional labels in modular furniture while operating over spaces of varying dimensionality (research article : 3D Semantic Segmentation of Modular Furniture using rjMCMC, Ishrat Badami et al., page 64\~72).  
* **Model Transformation & Interaction:** Converting recognized visual components into an interactive parametric model. Detected panels, labels, and spatial relationships are translated into editable components, formula-based dimensions, and joint definitions, so the segmentation output becomes a structured parametric assembly. Variables such as width, depth, height, and shelf count are mapped to web-based controls so that users can modify the structure in real time without rebuilding the design from scratch (research article : Parametric Modelling in Furniture Design A Case Study: Two Door Wardrope, Seval Ozgel Felek, page 62\~74).  
* **Validation & Output:** Checking structural consistency, inferring required hardware, and generating fabrication-oriented outputs such as 2D nesting layouts and printable blueprints.

**IV. Pattern Recognition**  
Pattern recognition is used to identify repeated structural features in panel-based furniture and to reduce the search space during interpretation and reconstruction.

* **Structural Regularities:** Panel furniture often follows modular geometric patterns. The system can use ideas such as "rectangle coverings" to estimate the number of likely structural elements, including repeated shelves or drawer fronts, and thereby constrain the number of possible interpretations during semantic segmentation (research article : 3D Semantic Segmentation of Modular Furniture using rjMCMC, Ishrat Badami et al., page 64\~72).  
* **Joinery Rules and Hardware Inference:** The system identifies recurring topological relationships, such as perpendicular panel joints, and uses them to infer the hardware required for assembly. To align with Design for Disassembly (DfD) principles, the inferred hardware can prioritize reversible connectors such as cam-bolt fasteners, expandable anchors, and snap-on hinges (research article : Analyzing Joinery for Furniture Designed for Disassembly, Maciej Sydor et al., page 162).  
* **Modular Logic:** Repeated user actions, such as increasing the number of shelves through a slider, can be represented as reusable parametric patterns so that component geometry and hardware counts update automatically.

**V. Abstraction and Modeling**  
Abstraction is necessary because the input image contains much more information than the system needs. The goal is to retain the structural information relevant to furniture design while ignoring visual details that do not affect fabrication.  
**Essential Data Extraction:** At this stage, the system focuses on geometric and relational information, including panel components, intersection points, dimensions, and hardware types. Background context such as room scenery, lighting conditions, and unrelated objects is treated as noise and excluded from the design model.  
**Modeling Framework:** To move from image interpretation to a computable design representation, the system follows three modeling levels:  
**1\. Conceptual Modeling (Core Entities)**  
This level defines the core entities in the furniture domain without yet specifying how they are stored or computed.

In the first implementation, entities such as Product, Component, Joint, Feature, Material, and Hardware are represented directly in the data model, while Assembly Step remains a higher-level planning concept.

* **The Product (The Boundary):** The abstract envelope or container that defines the total spatial volume of the final object.  
* **The Material (The Constraint):** The physical substance that dictates fundamental structural limits, most notably thickness, which drives all internal volume math.  
* **The Component (The Parametric Primitive):** A dynamically resizable geometric solid (typically a rectangular prism) representing a physical board.  
* **The Hardware (The Static Asset):** A fixed, non-scalable physical object (e.g., hinges, screws, cam-locks) that facilitates connection or articulation.  
* **The Joint (The Spatial Relationship):** The invisible topological anchor that binds two Components together in 3D space.  
* **The Feature (The Topological Modification):** A subtractive or additive operation on a specific face of a Component (e.g., a drill hole, a routing groove).  
* **The Assembly Step (The Temporal State):** A sequential time-slice that groups specific Components and Joints to represent a single phase of construction.

**2\. Logical Modeling (Cardinality and Rules)**  
This level specifies how the conceptual entities relate to one another and which parameter dependencies must be enforced.

* **Product-to-Component (1 : N):** A Product is an assembly of multiple Components. However, the dimensions of the Component are logically dependent on the bounding box of the Product.  
* **Component-to-Material (N : 1):** Many Components can share a single Material. If the Material thickness changes, all dependent Components must logically recalculate their spatial offsets.  
* **Component-to-Component (1 : N via Joints):** A Component acts as a parent to other Components. For example, a "Drawer Box" (Parent) dictates the logical position of the "Drawer Front" (Child).  
* **Component-to-Hardware (1 : N):** A Component acts as the mounting surface for Hardware. In the initial schema, hardware requirements are associated with component and joint types, and their mounting positions are derived during model generation and output preparation.  
* **Component-to-Feature (1 : N):** A Component contains multiple Features (machining operations) mapped to its specific indexed faces ($Face_1$ through $Face_6$).

**3\. Physical Modeling (Instantiation and Schema)**  
This level translates the logical relationships into computable variables, database tables, and transformation rules for rendering and fabrication output.

* **Parametric Variables (Formulas vs. Integers):** Rather than storing all dimensions as fixed values, the system stores some dimensions as formulas so they can respond dynamically to changes in product size or material thickness.  
  * *Example:* $$Component_{Length} = Product_{Height} - (Material_{Thickness} * 2)$$ 
* **Spatial Instantiation (The Transformation Matrix):**  
  Joints are instantiated through translation and rotation values that determine the global position of each child component relative to its parent:  
  $$P_{global} = M_{translation} \cdot M_{rotation} \cdot P_{local} + P_{parent\_origin}$$
* **Database Schema Design:**  
  * products table: id, sku, target\_width, target\_height, target\_depth.  
  * materials table: id, thickness\_mm, texture\_map\_url.  
  * components table: id, material\_id (FK), length\_formula, width\_formula.  
  * hardware\_library table: id, 3d\_mesh\_path (.glb), 2d\_svg\_path.  
  * joints\_bom table (Junction): parent\_id, child\_id, pos\_x, pos\_y, pos\_z, rot\_x, rot\_y, rot\_z.  
  * features table: component\_id (FK), face\_index \[1-6\], u\_coord, v\_coord, operation\_type (e.g., $5mm$ Drill).  
* **Output Serialization (JSON):** The relational data is serialized into a hierarchical JSON payload that serves as the shared source of truth for both the WebGL-based 3D view and the Canvas/SVG-based 2D output.

*The following Class Diagram expresses these physical relationships between data structures and model components, mapping out the architecture of the finalized database schemas:*  
```
@startuml
class Product {
  +String id
  +String sku
  +Float target_width
  +Float target_height
  +Float target_depth
  +generateBOM()
}

class Material {
  +String id
  +Float thickness_mm
  +String texture_map_url
}

class Component {
  +String id
  +String material_id
  +String length_formula
  +String width_formula
  +calculateDimensions()
}

class Hardware {
  +String id
  +String mesh_path
  +String svg_path
  +String joint_type
}

class Joint {
  +String parent_id
  +String child_id
  +Float pos_x
  +Float pos_y
  +Float pos_z
  +Float rot_x
  +Float rot_y
  +Float rot_z
}

class Feature {
  +String component_id
  +Int face_index
  +Float u_coord
  +Float v_coord
  +String operation_type
}

Product "1" *-- "many" Component : contains
Component "many" -- "1" Material : uses
Component "1" -- "many" Hardware : mounts
Component "1" -- "many" Joint : connects to
Component "1" -- "many" Feature : contains
@enduml
```
 

# **VI. Algorithm Design**

The algorithm is organized as a pipeline that converts visual input into an editable model and, ultimately, into fabrication-ready output.

* **Sequential Structure:** The main process follows a clear sequence: image upload, panel segmentation, parametric 3D generation, user-driven adjustment, and export of blueprints and nesting layouts.  
* **Selection Structure (If-Then Logic):** Real-time validation is implemented through conditional rules. For example, if a user adds a new shelf, the system updates the required cam-bolt fasteners in the bill of materials. If a panel span exceeds a load threshold, the interface can recommend a structural divider.  
* **Loop Structure (2D Nesting):** To reduce material waste, the nesting stage iterates through the generated panel set using a **bottom-left-fill** heuristic with a semi-discrete representation. Parts are placed at the lowest and leftmost feasible positions on a standard sheet, and multiple orientations are evaluated to reduce the total occupied area (research article : A fast and scalable bottom-left-fill algorithm to solve nesting problems using a semi-discrete representation, Sahar Chehrazad et al., page 809\~826).

*The following Sequence Diagram represents this request-response structure in chronological order, illustrating how interactions occur over time as a user request passes through the interface to trigger model inference and subsequent processing:*

```
@startuml
actor User
participant "Web Interface" as Web
participant "Hybrid Vision Pipeline (CNN + rjMCMC)" as AI
participant "Parametric Engine" as Engine
participant "CAM Processor" as CAM

User -> Web: Upload Furniture Image
activate Web
Web -> AI: Request Semantic Segmentation
activate AI
AI --> Web: Return Segmentation Masks
deactivate AI

Web -> Engine: Extract Dimensions & Infer Hardware
activate Engine
Engine --> Web: Return 3D Parametric Model
deactivate Engine

Web --> User: Display Interactive 3D Model
deactivate Web

User -> Web: Adjust Sliders (e.g., add shelf)
activate Web
Web -> Engine: Update Structural Topology
activate Engine
Engine --> Web: Return Updated Model & Hardware
deactivate Engine
Web --> User: Render Updated 3D View
deactivate Web

User -> Web: Export Fabrication Files
activate Web
Web -> CAM: Generate 2D Nesting & G-Code
activate CAM
CAM --> Web: Return Blueprints & BOM
deactivate CAM
Web --> User: Download Fabrication Package
deactivate Web
@enduml
```

*Additionally, this Activity Diagram displays the explicit chronological control flow, utilizing decision nodes to check conditions and direct the algorithmic behavior continuously:*  

```
@startuml

|DIY User|
start
:Upload 2D Furniture Image;
:Set Baseline Constraints (e.g., Max Width);

|Hybrid Vision Pipeline|
:Extract image features with CNNs;
:Refine panel hypotheses and labels with rjMCMC;

|Web Interface & Parametric Engine|
:Generate Initial 3D Parametric Model;
repeat

|DIY User|
  :Interact with Slider Controls (Add/Remove Shelves);
  

|Web Interface & Parametric Engine|
  :Update Structural Topology;
  :Infer & Assign Necessary Assembly Hardware;
  :Validate Span Load and Joint Connections;
repeat while (User modifying design?) is (Yes)

|CAM Processor|
:Execute Bottom-Left-Fill Nesting Algorithm;
:Minimize Sheet Material Waste;
:Export 2D Printable Blueprints;
:Generate Complete Parts & Hardware Inventory;

|DIY User|
:Download Fabrication Package;
stop
@enduml
```

# **VII. Data Management Plan**

The Data Management Plan supports the reliability and scalability of the proposed system by defining how data is collected, cleaned, and stored.

* **Data Collection:** Automated scraping pipelines gather high-resolution product images and related metadata from retail catalogs of panel-based furniture.  
* **Data Preprocessing:** Before training or inference, the raw dataset is cleaned using standard Exploratory Data Analysis (EDA) practices. Missing values are imputed or removed, and geometrically implausible samples are filtered out so they do not distort model behavior.  
* **Data Storage:** The finalized product, component, joint, feature, and hardware data are stored in a Relational Database Management System (RDBMS). Core tables such as products, materials, components, joints\_bom, features, and hardware\_library are linked through entity-relationship constraints, enabling the application to retrieve sheet thicknesses, hinge dimensions, and nesting parameters during interactive design updates.