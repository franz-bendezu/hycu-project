# **Intelligent Information System for Image-Based Structural Recognition and Personalized DIY Furniture Design**


# **I. Abstract**

Advances in Artificial Intelligence are automating complex design tasks, making them increasingly accessible to non-professionals. However, while visual inspiration for furniture is abundant online, a significant gap remains between these images and the precise technical documentation required for DIY construction. Non-expert users lack the structural knowledge to translate visual ideas into buildable designs, which often leads to wasted time and materials.  
To solve this, this project proposes a web-based system using Convolutional Neural Networks (CNNs) to decompose images of panel-based furniture (e.g., cabinets, shelving, desks) into basic structural components. The system extracts dimensions and spatial relationships, storing them parametrically to generate an interactive 3D model. Through the web interface, users can interact with this parametric model using slider-based controls or direct manipulation to adjust dimensions, modify the structural topology (e.g., adding/removing shelves), and apply textures in real time.  
Behind the scenes, the system performs continuous structural validation and automatically infers necessary assembly hardware (e.g., hinges, fasteners) based on detected panel joints. Upon finalization, the platform generates spatial 3D visualizations, automated 2D printable blueprints, and a complete parts inventory. Embedded algorithms also produce efficient 2D nesting layouts to minimize material waste during fabrication. Ultimately, this application reduces technical barriers for DIY makers, bridging the gap between visual inspiration and technical execution for resource-efficient furniture production.

# **II. Problem Definition and Gap Analysis**

The foundational step of any computational architecture is the rigorous definition of the problem, characterized within the computational thinking framework as the identification of the structural gap between a current, flawed state and a desired target state. The architecture of the proposed intelligent information system is predicated on a meticulous gap analysis of the modern do-it-yourself (DIY) manufacturing landscape.

* **Current State (AS-IS):** The existing paradigm for non-expert DIY furniture construction relies on a manual, error-prone process. Survey data indicates that 70% of DIY practitioners have avoided home improvement projects due to the "Fear of Messing Up" (FOMU), and 58% of projects exceed their budgets due to dimensional miscalculations and material waste (url reference : [http://www.prnewswire.com/news-releases/nailing-it-or-failing-it-new-study-reveals-70-of-diy-ers-have-avoided-projects-over-fomu-fear-of-messing-up-302151548.html](http://www.prnewswire.com/news-releases/nailing-it-or-failing-it-new-study-reveals-70-of-diy-ers-have-avoided-projects-over-fomu-fear-of-messing-up-302151548.html)). Furthermore, non-experts struggle with determining appropriate hardware (hinges, fasteners) for panel-based furniture.  
* **Target State (TO-BE):** A fully automated, web-based "Vision-to-Blueprint" pipeline. This system empowers users to interactively customize panel-based furniture via slider controls, automatically recalculating the topology, extracting structural hardware requirements, and exporting 2D nesting blueprints to minimize waste.  
* **Problem Classification:** Within the computational framework, the system fundamentally addresses a nexus of **Optimization** (minimizing material waste via 2D nesting) and **Computational** problems (translating 2D RGB pixels into interactive 3D parametric components).

*The following Use Case Diagram establishes the functional boundaries of this target state, illustrating how the DIY User interacts with the CNN Vision Model and the Parametric Engine to bridge the identified gap:*

``` 
@startuml
left to right direction
actor "DIY User" as User
actor "CNN Vision Model" as AI
database "Parametric Parts & Hardware DB" as DB

rectangle "Web-Based Intelligent Furniture System" {
  usecase "Upload Image of Panel Furniture" as UC1
  usecase "rjMCMC Semantic Segmentation" as UC2
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

Problem decomposition serves as the critical starting point of computational thinking, dividing a massive task into smaller, solvable subproblems. The system is systematically decomposed into four primary subproblems:

* **Data Acquisition:** Capturing and preprocessing visual images and metadata of panel-based furniture (cabinets, wardrobes) from retail platforms. This relies on the Knowledge Discovery in Databases (KDD) framework to systematically select and clean input data.  
* **Structural Recognition:** Segmenting the 2D image into hierarchical components (doors, drawers, shelves). To solve this, the system decomposes complex visual data into solvable sub-units. As an advanced, highly referenced option, the system can employ reversible jump Markov Chain Monte Carlo (rjMCMC) sampling to estimate the spatial structure and functional labels of the modular furniture, optimizing over spaces of varying dimensions (research article : 3D Semantic Segmentation of Modular Furniture using rjMCMC, Ishrat Badami et al., page 64\~72).  
* **Model Transformation & Interaction:** Converting visual segments into interactive parametric models. Variables such as width, depth, height, and the number of shelves are mapped to slider-based web controls, allowing users to modify the structural topology non-destructively in real-time (research article : Parametric Modelling in Furniture Design A Case Study: Two Door Wardrope, Seval Ozgel Felek, page 62\~74).  
* **Validation & Output:** Performing continuous structural validation on panel intersections, inferring necessary assembly hardware, and generating efficient 2D nesting layouts for fabrication.

**IV. Pattern Recognition**  
Pattern recognition identifies similarities and repeating variables across datasets to construct generalized rules, dramatically reducing the computational search space.

* **Structural Regularities:** Panel-based furniture exhibits distinct modular patterns. The system utilizes "rectangle coverings" to establish mathematical bounds on the number of structural elements (like identical shelving units or drawer faces), narrowing the hypothesis search space during semantic segmentation (research article : 3D Semantic Segmentation of Modular Furniture using rjMCMC, Ishrat Badami et al., page 64\~72).  
* **Joinery Rules and Hardware Inference:** The system recognizes specific topological intersections (e.g., perpendicular panel meeting points) and infers the necessary hardware. To comply with Design for Disassembly (DfD) principles, the system automatically assigns fully disassemblable hardware—such as cam/bolt fasteners, expandable anchors, and snap-on hinges—to panel joints, ensuring the furniture can be repeatedly repaired or recycled (research article : Analyzing Joinery for Furniture Designed for Disassembly, Maciej Sydor et al., page 162).  
* **Modular Logic:** Grouping repeated interactive actions (e.g., dynamically adding three shelves via a slider control) into reusable arrays, instantaneously updating the required hardware inventory without manual recalculation.

**V. Abstraction and Modeling**  
Abstraction distills complex phenomena down to their essential characteristics, filtering out unnecessary background noise to create a clean design representation.  
**Essential Data Extraction:** The abstraction layer isolates pure geometric properties. This is achieved by spotting the "main nouns" (e.g., hierarchical panel components, intersection nodes, volumetric dimensions, joint hardware types) to identify entities, and spotting the "actions" (e.g., connect, support) to identify relationships. Meanwhile, the system actively ignores room background, original lighting, and arbitrary environmental clutter.  
**Modeling Framework:** The abstraction process strictly follows a tripartite data modeling flow to transition from a real-world problem to database implementation:  
**1\. Conceptual Modeling (Core Entities)**  
This phase establishes the abstract ontology of the furniture system, defining "what" exists without worrying about how it is stored or measured.

* **The Product (The Boundary):** The abstract envelope or container that defines the total spatial volume of the final object.  
* **The Material (The Constraint):** The physical substance that dictates fundamental structural limits, most notably thickness, which drives all internal volume math.  
* **The Component (The Parametric Primitive):** A dynamically resizable geometric solid (typically a rectangular prism) representing a physical board.  
* **The Hardware (The Static Asset):** A fixed, non-scalable physical object (e.g., hinges, screws, cam-locks) that facilitates connection or articulation.  
* **The Joint (The Spatial Relationship):** The invisible topological anchor that binds two Components together in 3D space.  
* **The Feature (The Topological Modification):** A subtractive or additive operation on a specific face of a Component (e.g., a drill hole, a routing groove).  
* **The Assembly Step (The Temporal State):** A sequential time-slice that groups specific Components and Joints to represent a single phase of construction.

**2\. Logical Modeling (Cardinality and Rules)**  
This phase defines the strict relational rules and parametric dependencies between the conceptual entities. By interpreting these relationships, we establish the cardinality constraints.

* **Product-to-Component (1 : N):** A Product is an assembly of multiple Components. However, the dimensions of the Component are logically dependent on the bounding box of the Product.  
* **Component-to-Material (N : 1):** Many Components can share a single Material. If the Material thickness changes, all dependent Components must logically recalculate their spatial offsets.  
* **Component-to-Component (1 : N via Joints):** A Component acts as a parent to other Components. For example, a "Drawer Box" (Parent) dictates the logical position of the "Drawer Front" (Child).  
* **Component-to-Hardware\_Placement (1 : N):** A Component acts as the mounting surface for Hardware. The Hardware is anchored to a specific 2D coordinate on a specific face of the Component.  
* **Component-to-Feature (1 : N):** A Component contains multiple Features (machining operations) mapped to its specific indexed faces ($Face_1$ through $Face_6$).

**3\. Physical Modeling (Instantiation and Schema)**  
This phase translates the logical rules into computable variables, database schemas, and mathematical matrices required for 3D rendering and 2D instruction generation.

* **Parametric Variables (Formulas vs. Integers):** Instead of static floats, dimensions are stored as computable strings.  
  * *Example:* Component\_Length \= Product\_Height \- (Material\_Thickness \* 2\)  
* **Spatial Instantiation (The Transformation Matrix):**  
  Joints are physically modeled using offset and rotation vectors to calculate the final global position of any child component in the 3D space:  
  $$P_{global} = M_{translation} \cdot M_{rotation} \cdot P_{local} + P_{parent\_origin}$$
* **Database Schema Design:**  
  * products table: id, sku, target\_width, target\_height, target\_depth.  
  * materials table: id, thickness\_mm, texture\_map\_url.  
  * components table: id, material\_id (FK), length\_formula, width\_formula.  
  * hardware\_library table: id, 3d\_mesh\_path (.glb), 2d\_svg\_path.  
  * joints\_bom table (Junction): parent\_id, child\_id, pos\_x, pos\_y, pos\_z, rot\_x, rot\_y, rot\_z.  
  * features table: component\_id (FK), face\_index \[1-6\], u\_coord, v\_coord, operation\_type (e.g., $5mm$ Drill).  
* **Output Serialization (JSON):** The relational data is compiled into a hierarchical JSON payload, serving as the single source of truth passed to the WebGL (3D) and Canvas/SVG (2D) rendering engines.

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

Product "1" *-- "many" Component : contains
Component "many" -- "1" Material : uses
Component "1" -- "many" Hardware : mounts
Component "1" -- "many" Joint : connects to
@enduml
```
 

# **VI. Algorithm Design**

The algorithmic logic provides a structured sequence of operations to process the abstracted data into physical blueprints.

* **Sequential Structure:** 1\. Upload Image → 2\. Semantic Segmentation of Panels → 3\. Parametric 3D Generation → 4\. User Interaction (Slider Adjustments) → 5\. Blueprint & Nesting Export.  
* **Selection Structure (If-Then Logic):** Used for real-time validation. **If** the user uses direct manipulation to add a new shelf, **then** the algorithm automatically infers and adds the required cam-bolt fasteners to the parts inventory. **If** a panel span exceeds load-bearing thresholds, **then** the web interface prompts the user to add a vertical divider.  
* **Loop Structure (2D Nesting):** To minimize material waste, the system loops through the generated panel parts using a scalable **bottom-left-fill** heuristic combined with a semi-discrete representation. The algorithm sequentially places parts at the lowest and leftmost available coordinates on standard sheet materials, evaluating multiple rotational states to optimize the total consumed bounding area (research article : A fast and scalable bottom-left-fill algorithm to solve nesting problems using a semi-discrete representation, Sahar Chehrazad et al., page 809\~826).

*The following Sequence Diagram represents this request-response structure in chronological order, illustrating how interactions occur over time as a user request passes through the interface to trigger model inference and subsequent processing:*

```
@startuml
actor User
participant "Web Interface" as Web
participant "CNN Vision Model" as AI
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

|Vision Pipeline|
:Extract Image Features;
:Identify Panels (Doors, Drawers, Shelves) via rjMCMC;

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

The Data Management Plan ensures the reliability and scalability of the system through strict Extract, Transform, and Load (ETL) methodologies.

* **Data Collection:** Automated web scraping pipelines capture high-resolution imagery and metadata of panel-based furniture from retail catalogs.  
* **Data Preprocessing:** Utilizing Exploratory Data Analysis (EDA) principles, raw data undergoes rigorous cleansing. Missing values are imputed or removed, and physically impossible geometrical outliers are excluded to prevent distortion during neural network training.  
* **Data Storage:** A highly organized Relational Database Management System (RDBMS) stores the finalized physical models. Tables for Panel\_Geometries and DfD\_Hardware are linked via strict entity-relationship constraints (as shown in the Class Diagram), allowing the application to instantly query standard sheet thicknesses, hinge dimensions, and nesting parameters during real-time user interaction.