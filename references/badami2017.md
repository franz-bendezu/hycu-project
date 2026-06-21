2017 IEEE Winter Conference on Applications of Computer Vision
3D Semantic Segmentation of Modular Furniture using rjMCMC
IshratBadami1∗ ManuTom2∗ MarkusMathias3 BastianLeibe4
VisualComputingInstitute,ComputerVisionGroup PhotogrammetryandRemoteSensingGroup
RWTHAachenUniversity1,3,4 ETHZurich2
{badami, mathias, leibe}@vision.rwth-aachen.de manu.tom@geod.baug.ethz.ch
Abstract
In this paper we propose a novel approach to iden-
tify and label the structural elements of furniture e.g.
wardrobes, cabinets etc. Given a furniture item, the sub-
divisionintoitsstructuralcomponentslikedoors, drawers
and shelves is difficult as the number of components and
their spatial arrangements varies severely. Furthermore,
structural elements are primarily distinguished by their
function rather than by unique color or texture based
appearance features. It is therefore difficult to classify
them, even if their correct spatial extent were known. In
our approach we jointly estimate the number of functional
units,theirspatialstructure,andtheircorrespondinglabels
byusingreversiblejumpMCMC(rjMCMC),amethodwell
Figure 1: Semantic segmentation of modular furniture in
suited for optimization on spaces of varying dimensions
RGBD images: Left column is the input front face of the
(thenumberofstructuralelements). Optionally,oursystem
furnitureandrightcolumnisthesegmentationoutput(door,
permits to invoke depth information e.g. from RGB-D
drawerandshelf).
cameras, which are already frequently mounted on mobile
robotplatforms.Weshowaconsiderableimprovementover
a baseline method even without using depth data, and an
formmoreadvanced,human-likeinteractionsinindooren-
additionalperformancegainwhendepthinputisenabled.
vironments,likearranginggroceriesinthekitchen,sorting
books on the book shelves, opening/closing a locker etc.
These tasks require a detailed structural inference and the
classificationofthespatiallyvariablepartsofthefurniture.
1.Introduction
Oursegmentationproblemdiffersfromconventionalse-
mantic segmentation of the entire scene. A noisy, pixel-
Visual understanding of indoor scenes is a crucial task
wisesegmentationwouldnotbesufficientinordertoinfer
in robotics. Accurate semantic labeling and object clas-
thestructuralinformationthatisrequiredforaninteraction.
sification provides rich information about the complex in-
Furthermore,asalsonotedbyZhengetal.[40],traditional
door environment, which is crucial for navigation, manip-
visual cues such as color and texture are not particularly
ulation, and interaction with the scene. Most methods fo-
useful for labeling furniture items due to their often uni-
cus on coarse scene understanding. They identify walka-
formlycoloredandtexturedappearance.
blesurfaces[11]andobjects[8],estimatethe3Dgeometry
[41,25,33],andprovideroughlabelsforthedifferententi- In this paper, we present an approach to perform fine
ties[37]. grained segmentation of modular furniture like cabinets,
wardrobes,cupboards,orlockersintotheirfunctionalunits,
Only a limited amount of work aims at a more detailed
namely drawers, doors and shelves. These furniture items
objectlevelsegmentation[26]. Suchadetailedanalysisof
followamodulardesignastheirentirevolumeiscomposed
the object semantics will allow autonomous robots to per-
ofavariablenumberoffunctionalunits,thesocalledinter-
∗indicatesequalcontribution. actionelements(IEs)(seeFigure1).Additionally,mostfur-
978-1-5090-4822-9/17 $31.00 © 2017 IEEE 64
DOI 10.1109/WACV.2017.15

nitureitemsarenotonlyrectangularasawhole,butalsothe proachesbasedonHoughvoting[27]strugglewithclasses
internalstructurefollowsarectangularsubdivisionscheme. ofunconstrainedsizeandaspectratios. Finally,approaches
Weexploitthesemodularpropertiesinouroptimizationand based on deep learning require significantly more data to
propose a two stage segmentation approach. In the first train.
stagewegenerateanovercompletesetofrectanglepropos-
Indoorsceneparsingapproaches. Themajorityofman-
als such that each true IE is represented by a rectangle in
made objects can be modeled as a combination of differ-
the set. A rectangle proposal consists of the rectangle it-
ent geometric shapes [9, 39]. Han et al. [9] and Zhao et
self,andaclasslabeldistributionforthatrectangle. Inthe
al.[39]advancesthepixelgroupingtohigherlevelofpara-
secondstageweselectasubset(withunknownsize)ofthe
metricshapeclusteringinahierarchicalmanner,suchthatat
proposalsthatrepresentourfinalsemanticsegmentationof
eachlevelthecorrespondingclusterrepresentsapredefined
the furniture into interaction elements. We formulate the
geometric shape. These approaches are very successful at
proposal selection as rjMCMC based energy minimization
capturing geometric structure but they lack semantic label
problem.
information. Gupta et al. [8] utilizes general and object-
After the advent of 3D cameras, many previously pro-
classspecificappearancefeaturesaswellascontextualin-
posed RGB based methods are improved by using addi-
formatione.g.objectboundariesforsemanticsegmentation.
tional depth information [13, 33, 30]. Undoubtedly, depth
Theirapproachmainlyfocusesonlocalpatternsratherthan
provides powerful additional information when estimating
aglobalstructure.
the real object size or geometry of the scene. We show
thatusingRGB-Dimagesalsoimprovethesegmentationof Facade parsing approaches. Parsing building facades
furniture significantly. Our method is able to include such intothearchitecturalelementse.g.windows,walls,roofand
depthdatawhenavailable. parsingfurnitureintointeractionelementse.g.door,drawer,
shelf appear quite similar. Both contain rectangular grid-
Contributions.
like structures which have to be determined. The Facade
1. WeproposeanovelrjMCMCbasedfurnituresegmen- parsingproblemistackledfromdifferentdirections.Mu¨ller
tation method which achieves state-of-the-art results. et al. [22] detect repetitive structures in large, grid like fa-
Unlike[26],ourmethodallowstopredictstructureand cadesinordertoobtainmeaningfulhierarchicalfacadesub-
labelsjointlywithinasingleoptimization. divisions. Several methods exploit high-level information
in the form of shape grammars [32] combined with low-
2. Weintroduceanewdata-drivenaugmentationmethod
level appearance cues derived from an image [36, 34, 28].
togeneraterectangleproposalsleadingtoasignificant
The underlying grammars can either be designed manu-
higherrecall, i.e.thesetofIEproposalsbetterreflect
ally[20]orlearnedfromdata[18].Mathiasetal.[19]com-
thetrueIEs.
bine low-level, mid-level and high-level cues in form of a
3. We present a new 3D furniture dataset with corre- pixel-wise semantic segmentation, the output of an object
spondinggroundtruthannotations. detectors, and a shape grammar respectively. These shape
grammar based techniques assume a strong, style specific
structure and do not generalize well to different architec-
tural styles [21]. In case of furniture parsing, we would
2.RelatedWork
require a very generic grammar, which could only weakly
impose a structural layout. In case of facade parsing, ar-
Segmentation approaches. Segmentation can be per-
chitecturalelementsshowsignificantinter-classvariancein
formedwithorwithoutusingthesemanticinformation.Al-
color and texture and often exhibit regular and repetitive
gorithms that do not invoke semantic information, cluster
structure. Bothofthesepropertiesareabsentincaseoffur-
theimagepixelsbasedonfeaturesimilarities [2,14,24,4].
nitureparsingproblems.
Theresultingsegmentsnaturallydonotcarryanysemantic
label information. A wide range of segmentation methods FurnitureParsing. Limetal.[17]addressestheproblem
also aim to get semantic information from the segments, ofinstancelevelfurnituredetectionandposeestimationby
e.g. by classifying grouped pixels using supervised learn- using a predefined 3D CAD model. The approach targets
ingtechniques[5]. Alternatively,semanticknowledgemay to find the same model within a 3D scene. Our goal is to
influence the segmentation itself [29, 1, 15, 16]. Most in- parse any modular furniture. Pohlen et al. [26] addresses
stance segmentation methods are based on combining an theproblemoffurnituresegmentationfromasingleimage.
object detector output/region proposals with segmentation Ahugesetofpossiblefurnitureelementsisgeneratedfrom
[10,6]. Suchapproachesaredifficulttoapplyinourcase, theinputimage;thefinalsegmentationresultsfromselect-
asweexhibitahighinterclasssimilarityandobjectdetec- ingasuitableelementsubset.Ourapproachcloselyfollows
torsdonotcapturerelationshipsbetweentheinstances.Ap- the method described in [26]. We adopt the described ap-
65

3.1.ProposalGeneration
| pearance      | model     | by incorporating |             | depth | information  | which       |           |                |          |               |         |               |                |
| ------------- | --------- | ---------------- | ----------- | ----- | ------------ | ----------- | --------- | -------------- | -------- | ------------- | ------- | ------------- | -------------- |
| improves      | the label | inference        | performance |       | by           | almost 33%. |           |                |          |               |         |               |                |
|               |           |                  |             |       |              |             | We assume |                | that the | front         | face of | the furniture | item has       |
| As the number |           | of furniture     | elements    |       | is variable, | [26] per-   |           |                |          |               |         |               |                |
|               |           |                  |             |       |              |             | already   | been extracted |          | and rectified |         | during        | pre-processing |
formsoptimizationindependentlyforvariouspossiblenum-
|     |     |     |     |     |     |     | (e.g. using | [12]). | As such | our | search | space | is restricted to |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------ | ------- | --- | ------ | ----- | ---------------- |
bersofelementsusingMCMC.Inordertofindthebestso-
|     |     |     |     |     |     |     | axis-aligned, | rectangular |     | IEs. | Following |     | the approach in |
| --- | --- | --- | --- | --- | --- | --- | ------------- | ----------- | --- | ---- | --------- | --- | --------------- |
lutionamongthedifferentMarkovchains,themostmodular
|             |         |             |     |         |            |          | [26]firstwedetect            |     | rectanglesfromtheedgemapandthen |     |                      |     |     |
| ----------- | ------- | ----------- | --- | ------- | ---------- | -------- | ---------------------------- | --- | ------------------------------- | --- | -------------------- | --- | --- |
| solution is | chosen. | In contrast |     | to this | we perform | a single |                              |     |                                 |     |                      |     |     |
|             |         |             |     |         |            |          | assigneachIEcandidateaweight |     |                                 |     | andaclasslabelproba- |     |     |
multiobjectiveoptimizationusingatrans-dimensionalvari-
bility.
| antofMarkovchainnamelyrjMCMC[7]. |     |     |     |     | Thisseamlessly |     |     |     |     |     |     |     |     |
| -------------------------------- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
combinestheoptimizationofcorrectnumberofparts,their
3.1.1 RectangleDetection
| spatialarrangements,andtheclasslabelinference. |     |     |     |     |     | Ourap- |     |     |     |     |     |     |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
proachthereforereducesoverallcomputationalcostandre- Wefollowtwostrategiestogenerateamultitudeofrectan-
sultsinfasterconvergence.
glesthatserveasIEcandidates.
RectangleSetGenerationbyPohlenetal.[26].
| Varying | dimension | problems. |     | There | are | a number |     |     |     |     |     |     | Aseman- |
| ------- | --------- | --------- | --- | ----- | --- | -------- | --- | --- | --- | --- | --- | --- | ------- |
of challenging inference problems i.e. segmentation [35], tic edge map is generated from the image in a supervised
|     |     |     |     |     |     |     | mannerusingrandomforests[3]. |     |     |     | Intheedgemaphorizon- |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | --- | -------------------- | --- | --- |
multi-objecttracking[31],sceneparsing[38]etc.wherethe
dimension of the model of inference is not fixed. Usually, talandverticallinesaredetectedthroughHoughtransform.
Bayesian approaches are suitable for such problems. Re- Rectangle hypothesis are then generated as a convex hull
formedbyiterativelysamplingtwohorizontalandtwover-
| versible jump | MCMC | is  | capable | of computing |     | such infer- |     |     |     |     |     |     |     |
| ------------- | ---- | --- | ------- | ------------ | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
encebyjumpingbetweensubspacesofdifferingdimension- ticallines. Ahypothesisisacceptedasavalidrectangleif
themaximumdistancefromanyboundarypixeloftherect-
| ality. Tu | et al. | [35] propose | a   | data driven | MCMC | for im- |     |     |     |     |     |     |     |
| --------- | ------ | ------------ | --- | ----------- | ---- | ------- | --- | --- | --- | --- | --- | --- | --- |
agesegmentation. HeretheMarkovchaindynamicsisgov- angletotheclosestedgepixelintheimageisbelowtheset
ernedbyimportanceprobabilitiesdesignedusingtheimage threshold. Thisprocedureleadstoagoodinitialsetofpos-
sibleIEs,butneedsfurtherrefinementwhichweachieveby
data. Therearesevenimagemodelsforintensityandcolor
which describe the segments. The solution is obtained by thefollowingaugmentationmethod.
maximizingthejointposteriorofthesesegmentsusingthe
|     |     |     |     |     |     |     | Rectangle | Set | Augmentation. |     | Due | to complex | textures, |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- | ------------- | --- | --- | ---------- | --------- |
defined image models. Zhao et al. [38] propose a scene bad lighting conditions, or skewed perspective angles, the
| parsing approach |     | using | a stochastic | grammar |     | model. This |                  |     |         |           |     |            |             |
| ---------------- | --- | ----- | ------------ | ------- | --- | ----------- | ---------------- | --- | ------- | --------- | --- | ---------- | ----------- |
|                  |     |       |              |         |     |             | initial strategy |     | for the | rectangle | set | generation | is insuffi- |
modelisahierarchicalstructurewhichincludesscenecate-
cient. Weproposetoextendthesetofrectanglesincluding
gory,functionalgroups,functionalobjects,functionalparts, splitting and merging operations on the existing rectangle
| and 3D geometric |     | shapes | in a | top down | fashion. | Starting |                             |     |     |     |                           |     |     |
| ---------------- | --- | ------ | ---- | -------- | -------- | -------- | --------------------------- | --- | --- | --- | ------------------------- | --- | --- |
|                  |     |        |      |          |          |          | set. Asanadditionalbenefit, |     |     |     | thisstepofproposalsetaug- |     |     |
from extracted 3D shapes from the image, the objects at mentationmimicscostlyonlinedata-drivensplitandmerge
everylevelareclusteredaccordingtotheirfunctionandap- movesusuallydefinedinrjMCMCoptimizations.
| pearance | in a bottom | up  | fashion | using | rjMCMC. | Smith et |     |     |     |     |     |     |     |
| -------- | ----------- | --- | ------- | ----- | ------- | -------- | --- | --- | --- | --- | --- | --- | --- |
Tokeeptheproblemtractableweclustertheinitiallyde-
al.[31]developedtherjMCMCparticlefilterframeworkfor tectedrectanglessuchthatallrectanglesofaclusteroverlap
| robusttrackingofavariablenumberoftargets. |     |     |     |     |     | Ineachof |          |      |          |     |           |     |                |
| ----------------------------------------- | --- | --- | --- | --- | --- | -------- | -------- | ---- | -------- | --- | --------- | --- | -------------- |
|                                           |     |     |     |     |     |          | (IoU) by | more | than 95% | and | only keep | one | representative |
thediscussedmethods,asetoffourreversiblejumpmoves
|     |     |     |     |     |     |     | rectangle | of each | cluster. | We  | then | perform | two types of |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------- | -------- | --- | ---- | ------- | ------------ |
suchasbirth,death,update,andswaparedesignedtosearch rectanglesetaugmentations:
| though trans-dimensional |     |     | space. | The | different | move types |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | ------ | --- | --------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
are selected based on a time varying prior which depends • Splitaugmentationdividesarectangleintotworect-
|     |     |     |     |     |     |     | angles. | Weiterateovereachrectangleintheproposal |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | --------------------------------------- | --- | --- | --- | --- | --- |
onthepreviousstateoftheMarkovchain.
|     |     |     |     |     |     |     | pool.  | First,         | thehorizontalandverticaledgepixelhis- |          |           |     |                 |
| --- | --- | --- | --- | --- | --- | --- | ------ | -------------- | ------------------------------------- | -------- | --------- | --- | --------------- |
|     |     |     |     |     |     |     | togram | are            | computed.                             | A        | rectangle | is  | subdivided into |
|     |     |     |     |     |     |     | two    | new rectangles |                                       | at every | peak      | in  | edge histogram  |
3.ProposedApproach
greaterthanapredefinedthreshold.
| In the | first stage | of the | algorithm | we  | generate | an over- |         |              |     |          |     |       |                |
| ------ | ----------- | ------ | --------- | --- | -------- | -------- | ------- | ------------ | --- | -------- | --- | ----- | -------------- |
|        |             |        |           |     |          |          | • Merge | augmentation |     | combines |     | pairs | of rectangles. |
completesetofproposalswiththegoaltogenerateatleast
|              |           |       |          |      |               |         | All | neighboring | rectangles    |     | of         | nearly | the same height  |
| ------------ | --------- | ----- | -------- | ---- | ------------- | ------- | --- | ----------- | ------------- | --- | ---------- | ------ | ---------------- |
| one matching | proposal  |       | for each | true | interaction   | element |     |             |               |     |            |        |                  |
|              |           |       |          |      |               |         | are | merged      | horizontally, |     | rectangles |        | of similar width |
| (IE) of the  | furniture | item. | Having   | an   | over-complete | set of  |     |             |               |     |            |        |                  |
vertically.
proposalsallowsustocomputethesemanticsegmentation
by performing subset selection. This is formulated as an All newly generated rectangles are only added to the pool
energy minimization problem in a high dimensional state if they do not considerably overlap with already existing
spacedetailedinSection3.2. rectangles(usinga85%IoUthreshold).
66

InclusionofDepthInformation TofurtherimproveIEde- theshapepriortermbyusingtherelativedepthofeachrect-
tectionrecall,ourmethodisabletoincludedepthinforma- angle compared to the front face of the furniture, and the
tioninthisstageofthealgorithm. Tothatendweextended aspectratioofdepthinrelationtowidthandheight.
the semantic edge detection forest to learn semantic edges
3.2.ProposalSelection
based on depth maps. We fuse the resulting edge with the
initial edge map via the pixel-wise or operator. Although Fromthesetofdetectedrectangleswewishtochoosea
we will incorporate depth information throughout our al- subset of rectangles that best explains the image I m. Let
gorithm, we are able to disable depth and roll back to an P :={(r k ,l k)|k =1,...,K}beasubsetofK rectangles.
RGB-onlysetup. OurgoalistofindthebestsubsetofrectanglesSˆ⊂P such
that
3.1.2 RectangleWeighting Sˆ=argmaxp(Sˆ|I m). (3)
Sˆ
Given the image I m and rectangle r, the weight of the IE WejointlyestimatethetruenumberofIEsK,theirspa-
with label l ∈ {door,drawer,shelf} is quantified by the
tialarrangementsr kandtheirrespectiveclasslabelsl k.The
conditionalprobabilityasinEquation1.
optimizationisformalizedasamulti-objectiveoptimization
(cid:2) p(l| (cid:3) r (cid:4) ,I m(cid:5) )∝ (cid:2) p(I m(cid:3)(cid:4) |r,l (cid:5) ) p (cid:2) (r (cid:3)(cid:4) |l (cid:5) ) (cid:2) p (cid:3) ( (cid:4) l) (cid:5) (1) problem:
Label Appearance Shape Label p(S|I m)∝e−E total (S), (4)
posterior likelihood prior prior
where
Due to a high visual inter-class similarity between the IEs
ofasinglefurnitureinstance,colorandtextureappearance
E total(S)=E c(S)+E o(S)+E w(S)+ (5)
cues are not sufficiently discriminative. We therefore ex-
ploitasetofcommontraitsthatcanbeobservedforIEsof
E ls(S)+E lv(S)+E s(S)
thesameclass. Thesetraitsaretherectangle’saspectratio,
Maximizingp(S|I m)isequivalenttominimizingtheen-
thepositionofthehandle,andtheedgeprofile.
ergyofEquation5. EachenergyterminE total(S)captures
Following Pohlen et al. [26], we learn a codebook rep-
resentationoverJ codewordsp(1,l),...,p(J,l) ∈RM2 based adedicatedpropertyasdescribedinthefollowing.
ontheM ×M rescaledgradientmagnitudeimageforeach TheCoverenergyE c securesamaximumcoverageofthe
of the classes independently. The objective for the train- areaΩofthefurniture’sface(Figure3(a)).
⎛ ⎞
ingprocedureistoapproximateeachtrainingelementasa
(cid:14)K (cid:10)
l w in e e ll a a rc n o e m w b r i e n c a ta ti n o g n le of de c fi od n e e b d o o o v k er en an tri i e m s. ag D e e r p e e g n io d n in f g r o ,I n m h c o a w n E c =− Ω 1 ⎝ k=1 |r k |− k(cid:3)=j |r k ∩r j |⎠ (6)
beexpressedbyanyofthelearnedcodebooks,theappear-
ancelikelihoodisdefinedasfollows:
⎛ (cid:9) (cid:9) ⎞ The Overlap energy E o ensures minimum overlap be-
p(I m |r,l)∝ (cid:2)π∈ m j [ π 0 a , j 1 x = ]J 1 exp ⎜ ⎝− (cid:9) (cid:9) (cid:9) (cid:9) (cid:9) f r,Im − (cid:10) j= J 1 π j p(j,l) (cid:9) (cid:9) (cid:9) (cid:9) (cid:9) 2 2 ⎟ ⎠, tweenallpa E ir o so = fr λ e o ct · a 1 (cid:15) n K g 2 l (cid:16) es (cid:10) k(cid:3)= in j a m s i t n a | t r ( e | j r ( ∩ j F | i , r g | k r u | k re |) 3 ) (b)). (7)
(2)
whereπ 1 ,...,π J arethecodebookcoefficients. whereλ o = 0.15isanempiricallydeterminedoverlappa-
rameter.
Theshapepriorisestimatedwithaprobabilisticsupport
vectormachineusingrelativeheight,widthandaspectratio TheRectangleweightenergyE w chosesrectangleswith
asfeatures. ahighappearancelikelihood(Figure3(c)).
Finally,thelabelpriorrepresentstheobservedlabelfre-
(cid:10)K
que A n d ci d e i s ti i o n na th l e de tr t a a i i n ls in c g an da b t e a. foundin[26]. E w = K 1 1−p(l k |r k ,I m) (8)
k=1
InclusionofDepthInformation Fromtheresultsin[26]it
isapparentthatproposedweightingschemeworkswellfor The Label smoothing energy E ls encourages label con-
doorsanddrawersbutsuffersahighconfusionbetweenthe sistency given the structure of the furniture. For modular
drawerandtheshelfclass. Whiletheseclassesaredifficult furniture, amodularitytreeΓcanbebuilt. Theentireface
todifferentiatevisually,depthcuesshouldclearlyimprove of the furniture shown in Figure 2 defines the root of the
performance. Here,weincorporatedepthinformationover tree. Elementsthataresimilarinstructureareclusteredand
67

of the total energy. By sampling different states of the
chainonecantraversethroughthemulti-dimensionalstate
space. From the current state S, the new state S∗ is sam-
pledwithproposalprobabilityp(S∗|S).Ifp(S∗ )p(S∗|S)>
p(S)p(S|S∗ ), thenweacceptthe“better”stateS∗. Other-
wise, we accept the new state with a probability propor-
tionaltop(S∗ )T 1 i p(S|S∗ )/p(S)T 1 i p(S∗|S)wherei ∈ Nis
thecurrentiterationandT i isthetemperatureatstepi. Ef-
Figure 2: Rectangle clusters within a furniture. Two child fectively,theacceptanceprobabilityofthenewstateis:
rectangles(door)inpurplecluster,fourchild(drawer)rect- (cid:17) (cid:18)
anglesinredclusterandtwochild(door)rectanglesingreen a(S,S∗ )=min 1, p(S∗ )T 1 i p(S|S∗ ) (12)
cluster. p(S)T 1
i
p(S∗|S)
Pohlenetal.[26]performsasimulatedannealingbased
definethefirstthreechildnodes(denotedinpurpleredand
optimizationinmultiplerounds,eachtimewithadifferent,
green). Eachchildnodecanbefurtherdividedinto,smaller
fixed dimension. Bounds on the dimension are estimated
nodes sharing height and/or width. As can be observed in
inaseparateprocess,solelybasedonrectangleshapes,i.e.
theexample,allleafnodesthatshareaparentnodetendto
decoupledfromappearance.Therepeatedoptimizationpro-
be of the same class. The label smoothing energy favors
cess is prone to errors in the bounds estimation and ineffi-
suchlabelconfigurations.
cient.
(cid:10) (cid:10) In contrast, we perform optimization using a trans-
1 1
E ls = M n∈Γ (cid:15) C 2 n (cid:16) c1,c2∈child(n) I[l(c 1)(cid:6)=l(c 2))] d M im C e M n C sio [ n 7 a ] l . v T a h ri e an i t de o a f b M eh C in M d C r , jM n C am M e C ly is re t v o er a s l i l b o l w e j s u a m m p -
c1,c2areleafnodes plingthetrans-dimensionalspacewithastationarydistribu-
(9)
tion. To achievea stationary distribution a careful balance
where M is the total number of leaf clusters in an image
of the chain dynamics must be fulfilled. In rjMCMC, this
tree Γ, C n is the number of children of node n and l(·) is balanceisobtainedthroughdimensionmatchingreversible
theclasslabelofachildIE.
jumpmoves. Inourarchitecture,wedesignonejumpmove
The Layout variance energy E lv incites the structural pair (birth and death move) to search over the variable di-
modularityinthetreeandpenalizesshapeandpositionde- mensionalspaceandonediffusion(exchange)movetoex-
viationswithinatreebranch. ploreafixeddimensionalspace:
1 (cid:10) (cid:10) Birth move. The new state S∗ is generated from S by
E lv = M [h(c i)−h m] 2 +[w(c i)−w m] 2 adding a new rectangle r from the rectangle pool while
n∈Γci∈child(n)
keeping all the other rectangles fixed. The birth move in-
(10)
creases the dimension of S by the dimension of the added
where h(·) and w(·) determine the height and width of
rectangledim(r). Theacceptanceprobabilityforthismove
a child rectangle, h m and w m denote the average cluster
is:
widthandheight. (cid:19) (cid:20)
aB (S,S∗
)=min
1,θB (S,S∗
) (13)
TheStatesizeenergyE s favorsahighernumberofIEs.
where
K
whereN isthenumbero E f s re = cta − n N glesintheproposalpo ( o 1 l 1 . ) θB (S,S∗ )= (cid:2) p p ( ( S S (cid:3) ∗ (cid:4) ) ) T 1 T i 1 i (cid:5) · (cid:2) q q D B ( ( S S , ∗ r | | S (cid:3) S , (cid:4) r ∗ ) ) · · p p ( ( D B) ) (cid:5) · ja (cid:2) c J (cid:3) o B b (cid:4) i (cid:5) an
proposal
posterior
3.2.1 ReversibleJumpMCMCMoves ratio
ratio
(cid:21) (cid:21) (14)
T re h p e re s s o e l n u t t s io a n h s i p g a h ce di o m f e t n h s e io e n n a e l rg s y pa f c u e n . ct O io p n ti d m e i s z c a r t i i b o e n d i a n b t o h v i e s where, J B = (cid:21) (cid:21) ∂ ∂ ( ( S S , ∗ r ) ) (cid:21) (cid:21) is a Jacobian for the transformation
space can be achieved efficiently by sampling the Markov
from(S,r)toS∗. p(B)andp(D)aretheprobabilitiesfor
chainwithsimulatedannealing, anMCMCbasedstochas-
choosingbirthanddeathmovesrespectively,q B(S∗ |S,r),
is the probability to add rectangle r to the current state S,
tic optimization method. The Markov chain with a sta-
tionarydistributionisconstructedsuchthatthemajorityof
andsimilarlyq D(S,r |S∗ )definestheprobabilitytodelete
the probability mass concentrates at the global minimum
a certain rectangle r from the current state S. p(S) and
68

(a)cover(E c) (b)overlap(E o) (c)weight(E w) (d)E o+E (e)E o+E c+E (f)total
|     |     |     |     |     |     |     |     |     | c   |         |     | w            |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | ------------ | --- | --- |
|     |     |     |     |     |     |     |     |     |     | cover(E |     | c),overlap(E |     |     |
Figure3: Effectofdifferentenergiesontheproposalsubsetselection(fromlefttoright): o),rectangle
weight (E w), E + E c, E + E + E w, Total (E). We show the most important energy terms and their combination that
|     |     | o   | o   | c   |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
contributemostofthesegmentationperformanceduringtheproposalselectionstep.
p(S∗
| )aredeterminedfromtheenergyEaccordingtoEqua- |     |     |     |     |     |     |     | 4.Evaluation |     |     |     |     |     |     |
| -------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- | --- |
tion 4. Weassumeuniformproposalprobabilityforselect-
Tothebestofourknowledge,wearethefirsttopresent
ingtherectangle,hencetheproposalratioonlydependson
|     |     |     |     |     |     |     |     | afurnituredatasetincludingboth,RGBanddepth. |     |     |     |     |     | Forthe |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------- | --- | --- | --- | --- | --- | ------ |
thenumberofrectanglesinthecurrentstate(k)andinthe
proposalpool(N−k).Theprobabilitiesfordeathandbirth evaluationofourproposedmethodweintroduceanewsyn-
theticdatasetconsistingof160imageswitharesolutionof
movesaresetsothattheoverallacceptancerateishigh.The
|                    |            |                         |           |       |                  |     |     | 640×480.         | The | ground truth                     | structures |     | and labels | are an- |
| ------------------ | ---------- | ----------------------- | --------- | ----- | ---------------- | --- | --- | ---------------- | --- | -------------------------------- | ---------- | --- | ---------- | ------- |
| Jacobian           | is derived | to be                   | 1 (Please | refer | to supplementary |     |     |                  |     |                                  |            |     |            |         |
|                    |            |                         |           |       |                  |     |     | notatedmanually. |     | Inourexperimentsweperforma4-fold |            |     |            |         |
| forthederivation). |            | Equation14simplifiesto: |           |       |                  |     |     |                  |     |                                  |            |     |            |         |
crossvalidation.Ineachround,75%oftheimagesareused
|     |     |     | ( 1 | )   |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
p(S∗ ) T N −k p(D) t o t r a in t h e a p p ea r a n c e c o d e b oo k a nd s h ap e p r io rs a n d 25 %
|     | θB (S,S∗ |     | i   | ·   | ·   |     |     |     |     |     |     |     |     |     |
| --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
)= (15) i m a g e s a r e u se d f o r t e st i n g . W e ar e on l y aw a r e o ft h e w o rk
|     |     |     | p(S) ( 1 | ) k | p(B) |     |     |                                                     |     |     |     |     |     |     |
| --- | --- | --- | -------- | --- | ---- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|     |     |     | T i      |     |      |     |     | ofPohlenetal.[26]whichtacklesthisproblemoffurniture |     |     |     |     |     |     |
segmentationandthereforeservesasabaselineforcompar-
| Deathmove. |     | Thismoveisthereverseofabirthmove. |     |     |     |     | It  |                       |     |                             |     |     |     |     |
| ---------- | --- | --------------------------------- | --- | --- | --- | --- | --- | --------------------- | --- | --------------------------- | --- | --- | --- | --- |
|            |     |                                   |     |     |     |     |     | isonintheexperiments. |     | Wegenerateourdatasetbymodi- |     |     |     |     |
removes one rectangle while keeping all the other rectan- fyingreadilyavailable3Dfurnituremodelsinblender. All
glesfix. Deathandbirthmovesareareversiblemovepair,
thegivenmodelsareorientedinaestheticallybeautifulori-
| ensuring | balance | in the | chain. | The acceptance |     | probability |     |                               |     |     |                         |     |     |     |
| -------- | ------- | ------ | ------ | -------------- | --- | ----------- | --- | ----------------------------- | --- | --- | ----------------------- | --- | --- | --- |
|          |         |        |        |                |     |             |     | entationandlightingcondition. |     |     | Wechangetheorientation, |     |     |     |
ofadeathmovecanbegivenas: texture and lighting condition of these models. Addition-
(cid:19) (cid:20) ally,weaddartificialaxialandlateralKinectnoisedepend-
|     | aD  | (S,S∗ |     | 1,θD (S,S∗ |     |     |      |                                            |     |     |     |     |     |     |
| --- | --- | ----- | --- | ---------- | --- | --- | ---- | ------------------------------------------ | --- | --- | --- | --- | --- | --- |
|     |     | )=min |     |            | )   |     | (16) | ingondepthandorientationasdescribedin[23]. |     |     |     |     |     |     |
where
4.1.QuantitativeResults
|     |     | (cid:15) |     | (cid:16) |     |     |     |     |     |     |     |     |     |     |
| --- | --- | -------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
−1
|     | θD (S,S∗ | )=  | θB (S,S∗ | )   |     |     |     |     |     |     |     |     |     |     |
| --- | -------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Inthissectionweevaluatevariousaspectsofourpipeline
|     |     |     | p(S∗ ( 1 | )      |      |     |      |                                          |     |               |     |        |                |     |
| --- | --- | --- | -------- | ------ | ---- | --- | ---- | ---------------------------------------- | --- | ------------- | --- | ------ | -------------- | --- |
|     |     |     | ) T i    | k      | p(B) |     |      | andpresenttheoverallquantitativeresults. |     |               |     |        |                |     |
|     |     | =   |          | ·      | ·    |     | (17) |                                          |     |               |     |        |                |     |
|     |     |     | ( 1      | ) N −k | p(D) |     |      |                                          |     |               |     |        |                |     |
|     |     |     | p(S) T   |        |      |     |      | Rectangle                                | Set | Augmentation. |     | During | the generation | of  |
i
|         |         |            |     |           |      |        |       | our over-complete |     | set of   | IE proposals, |     | we use | the previ-  |
| ------- | ------- | ---------- | --- | --------- | ---- | ------ | ----- | ----------------- | --- | -------- | ------------- | --- | ------ | ----------- |
| Another | popular | reversible |     | jump move | pair | is the | split |                   |     |          |               |     |        |             |
|         |         |            |     |           |      |        |       | ously suggested   |     | proposal | generation    | by  | Pohlen | et al. [26] |
and the merge move. As these moves are computationally andthenextendtheresultingproposalsetbyourrectangle
expensive, we avoid using them during optimization. In- setaugmentation. Weevaluatethemaximumachievablere-
stead,thesplit/mergeaugmentationasdescribedinSection
|     |     |     |     |     |     |     |     | call at this | stage | of the pipeline |     | and the | resulting | overall |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ----- | --------------- | --- | ------- | --------- | ------- |
3.1.1serveasaproxy. improvement. Theinitialproposalgenerationstepachieves
arecallof79.8%,ouraugmentationimprovesthisrecallto
| Exchangemove. |     | Isadiffusionmovewhichpreservesdi- |     |     |     |     |     |     |     |     |     |     |     |     |
| ------------- | --- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
83.3%.
mensions. A rectangle is randomly selected from the cur- Forourfullpipelineweobservedanimprovement
of1%pointbyaddingIEaugmentation.
rentstateandisexchangedwithanotherrectanglewhichis
randomlysampledfromtheproposalpool:
|     |     |     |          |     |          |     |     | Structural | Inference. | Compared |     | to [26], | in  | our method |
| --- | --- | --- | -------- | --- | -------- | --- | --- | ---------- | ---------- | -------- | --- | -------- | --- | ---------- |
|     |     |     | (cid:19) |     | (cid:20) |     |     |            |            |          |     |          |     |            |
weremovethecostlyrectanglepruningstepandeffectively
|     | aE  | (S,S∗ )=min |     | 1,θE (S,S∗ | )   |     | (18) |     |     |     |     |     |     |     |
| --- | --- | ----------- | --- | ---------- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
addacorrespondingpruningcriteriatotheobjectivefunc-
|     |     |     |     |     |     |     |     | tion, Equations |     | 5, 6, 7. | This improves |     | the overall | speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- | -------- | ------------- | --- | ----------- | ----- |
where
p(S∗ ( 1 ) c o n s id e r a b l y ( 1 . 9 x f as t er ) a n d a v o id s h a rd d e c is io n s b e fo re
|     |     |     |     | ) T | i   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
θE (S,S∗ )= (19) o p ti m i z a ti o n , l e a d in g t o b e t te r re c a ll a n d he n c e o v er a ll s eg -
|     |     |     |     | ( 1    | )   |     |     |            |     |                   |     |            |     |            |
| --- | --- | --- | --- | ------ | --- | --- | --- | ---------- | --- | ----------------- | --- | ---------- | --- | ---------- |
|     |     |     |     | p(S) T |     |     |     |            |     |                   |     |            | F   |            |
|     |     |     |     | i      |     |     |     | mentation. | We  | report precision, |     | recall and | 1   | measure of |
69

]62[.latenelhoP
hturTdnuorG
hcaorppAruO
Figure4: Qualitativecomparisonofsegmentationon3Dsyntheticdata: ThefirstrowistheinputRGBimages. Secondrow
isgroundtruthannotation.Thirdrowshowsresultofsegmentationby[26].Thelastrowshowsoursegmentationresultusing
depth. (door,drawerandshelf).
Figure 5: Qualitative results: The input real Kinect images and corresponding segmentation are displayed respectively on
row1androw2. Columns1-5showsthesuccesscaseswhilecolumns6displaysthefailedcases. Thetwomainreasonsfor
| failurearemissededgesandhighamountoftexture. |     |     |     | (door,drawerandshelf). |               |                      |                   |
| -------------------------------------------- | --- | --- | --- | ---------------------- | ------------- | -------------------- | ----------------- |
|                                              |     |     |     | Table                  | 1: Comparison | of overall structure | inference perfor- |
ourstructureinference.Atthisstage,weareonlyinterested
in the subdivision of the furniture, not in the resulting se- mance. Here we compare our method with and without
depthtotheapproachpresentedin[26].
manticlabeling.Forcomparison,Table1servesthe2Dand
| 3D versions | of our method | and the work | of [26]. Table | 1   |     |     |     |
| ----------- | ------------- | ------------ | -------------- | --- | --- | --- | --- |
F
showsthatusing3Dimprovesourmethodbyalargemar- precision recall 1
|     |     |     |     |     | [26] | 68.8% 49.8% | 57.8% |
| --- | --- | --- | --- | --- | ---- | ----------- | ----- |
gin,yetevenour2Dversionsetsanewstate-of-the-art.
|                                                   |     |                            |     |     | our(2D) | 63.5% 68.7% | 66.0% |
| ------------------------------------------------- | --- | -------------------------- | --- | --- | ------- | ----------- | ----- |
| ClassLabelInference.                              |     | Herewemeasuretheaccuracyof |     |     |         |             |       |
|                                                   |     |                            |     |     | our(3D) | 73.5% 79.9% | 76.6% |
| thepredictedlabelsonlyforthecorrectlydetectedIEs. |     |                            |     | We  |         |             |       |
consideraIEdetectediftheIoUwithagroundtruthannota-
| tionexceeds65%. | Thisallowsustomeasuretheefficiency |     |     |     |     |     |     |
| --------------- | ---------------------------------- | --- | --- | --- | --- | --- | --- |
ofourappearancemodelindependentofthestructuralsub- division. Table2reportsaccuracyofclasslabelprediction
70

| for[26]andourapproachwithandwithoutusingdepth. |     |     |     |     |     |     |     | 80  |     |     |     |     |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
E
o
E ls+E
lv
| Table2: | ClasslabelaccuracyforcorrectlydetectedIEs. |     |     |     |     |     |     |     |     |     |     |     |     |
| ------- | ------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
E o+E
|     |     |     |     |     |     |     |     | 60  |     |     |     |     | c   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
E
|     |      |     | door  | drawer | shelf |     |     |     |     |     |     | s   |     |
| --- | ---- | --- | ----- | ------ | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | [26] |     | 91.9% | 71.7%  | 15.5% |     |     |     |     |     |     | E   |     |
c
E
|     | our(2D) |     | 76.4% | 77.6% | 40.9% |     |     | 40  |     |     |     | w       |     |
| --- | ------- | --- | ----- | ----- | ----- | --- | --- | --- | --- | --- | --- | ------- | --- |
|     | our(3D) |     | 99.3% | 96.2% | 98.8% |     |     |     |     |     | E   | o+E c+E | w   |
|     |         |     |       |       |       |     |     |     |     |     |     | E -E    |     |
s
|                                                  |     |     |     |     |     |     |     | 20  |     |     |     | E -E |     |
| ------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- |
| Itisapparentthatdepthisacrucialcuetoapproachper- |     |     |     |     |     |     |     |     |     |     |     |      | c   |
E
fectclasslabelinference.Weshowthefullconfusionmatri-
cesfortheoverallsegmentationforboth,withoutandwith
|           |          |     |           |               |     |         |     | StructureF | 1   | Labeling |     |     |     |
| --------- | -------- | --- | --------- | ------------- | --- | ------- | --- | ---------- | --- | -------- | --- | --- | --- |
| depth(see | Table3). |     | Usingonly | 2Dinformation |     | leadsto | a   |            |     |          |     |     |     |
Accuracy
highconfusionbetween“drawer”and“shelf”,whichcanbe
resolvedusingdepth.
|     |     |     |     |     |     |     |     | Figure6: Bargraphshowingperformanceonstructurepre- |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- |
diction(Left)andlabelaccuracy(right)withdifferentcom-
| Table | 3: Detailed | class | label | performance | of  | our approach |     |                            |     |                          |     |     |     |
| ----- | ----------- | ----- | ----- | ----------- | --- | ------------ | --- | -------------------------- | --- | ------------------------ | --- | --- | --- |
|       |             |       |       |             |     |              |     | binationsoftheenergyterms. |     | Bestresultisachievedwhen |     |     |     |
withandwithoutusingdepth.
alltheenergytermsareincorporated.
Prediction2D
|     |       |        | Door  |     | Drawer      | Shelf |     |                                                      |     |     |     |     |     |
| --- | ----- | ------ | ----- | --- | ----------- | ----- | --- | ---------------------------------------------------- | --- | --- | --- | --- | --- |
|     |       | Door   | 75.4% |     | 1.5% 23.1%  |       |     |                                                      |     |     |     |     |     |
|     | hturT |        |       |     |             |       |     | ourproposedsyntheticdataset,wealsoperformedaqualita- |     |     |     |     |     |
|     |       |        | 3.7%  |     | 77.0% 19.3% |       |     |                                                      |     |     |     |     |     |
|     |       | Drawer |       |     |             |       |     | tivestudyonrealworldimagesamplesoriginatingfromthe   |     |     |     |     |     |
Shelf 12.9% 44.4% 42.7% Kinectsensor. Forthisexperiment,wetraintheappearance
|     |     |     |     |     |     |     |     | codebook | and shape prior | on the | entire | synthetic | dataset |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | --------------- | ------ | ------ | --------- | ------- |
Prediction3D (160images). Figure5showsexamplesegmentationresult.
|     |       |      | Door  |     | Drawer | Shelf |     |                                                  |           |        |        |                 |     |
| --- | ----- | ---- | ----- | --- | ------ | ----- | --- | ------------------------------------------------ | --------- | ------ | ------ | --------------- | --- |
|     |       |      |       |     |        |       |     | Contribution                                     | of Energy | Terms. | Figure | 3 qualitatively |     |
|     |       | Door | 99.3% |     | 0.7%   | 0%    |     |                                                  |           |        |        |                 |     |
|     | hturT |      |       |     |        |       |     | showstheinfluenceofeachenergytermgivenasingleex- |           |        |        |                 |     |
|     |       |      | 3.0%  |     | 96.2%  | 0.8%  |     |                                                  |           |        |        |                 |     |
Drawer
|     |     |     | 1.1% |     | 1.1% 98.8% |     |     | ample. |     |     |     |     |     |
| --- | --- | --- | ---- | --- | ---------- | --- | --- | ------ | --- | --- | --- | --- | --- |
Shelf
| Segmentationperformance.      |              |     |     | Tocomparetheoverallseg- |                        |     |         |              |     |     |     |     |     |
| ----------------------------- | ------------ | --- | --- | ----------------------- | ---------------------- | --- | ------- | ------------ | --- | --- | --- | --- | --- |
| mentation                     | performance, |     | we  | combine                 | the structure          |     | and la- | 5.Conclusion |     |     |     |     |     |
| belingaccuracyofthealgorithm. |              |     |     |                         | Wemultiplythestructure |     |         |              |     |     |     |     |     |
accuracywiththelabelaccuracyforeachlabelandtakethe
Weproposeamethodforsemanticsegmentationoffur-
average. The combined performance of the baseline from niture into their interaction elements for RGB-D images.
Pohlenetal.[26]reaches33.8%,comparedtoourmethod Weshowthatdepthinformationiscrucialtothestructural
(2D)reaching45.5%. Whenenablingdepththefinalresult inferenceandclassificationoftheIEs. Weproposeamulti-
ofourfullpipelineis78.3%. objective optimization method using an effective energy
|              |     |           |        |     |            |     |      | maximization | formulation. | We  | successfully | demonstrate |     |
| ------------ | --- | --------- | ------ | --- | ---------- | --- | ---- | ------------ | ------------ | --- | ------------ | ----------- | --- |
| Contribution |     | of Energy | Terms. |     | We examine | how | each |              |              |     |              |             |     |
oftheenergytermsaffectthestructureandlabelprediction. the strength of our rjMCMC optimization design for our
|            |              |     |     |       |                        |     |     | trans-dimensional | model | space. | Finally, | we show | consid- |
| ---------- | ------------ | --- | --- | ----- | ---------------------- | --- | --- | ----------------- | ----- | ------ | -------- | ------- | ------- |
| We perform | segmentation |     |     | using | different combinations |     | of  |                   |       |        |          |         |         |
erableimprovementonthepreviousstate-of-the-artresults
energytermsinFigure6.Asexpected,therectangleweight
energy(E for furniture parsing given on novel 3D furniture dataset.
w)iscrucialforlabelaccuracy,asisthecoveren-
| ergy(E |                           |     |     |     |                       |     |     | ThisworkisalsotransferabletorealKinectimages,open- |     |     |     |     |     |
| ------ | ------------------------- | --- | --- | --- | --------------------- | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- |
|        | c)forpredictingstructure. |     |     |     | Whilethestructurepre- |     |     |                                                    |     |     |     |     |     |
dictionreachescompetitiveresultsusingonlysingleenergy ingdoorsfortheadvanceresearchinroboticsforinteraction
|        |           |          |        |          |     |        |        | with furniture. | Our code1 | and annotated |     | dataset2 | are pub- |
| ------ | --------- | -------- | ------ | -------- | --- | ------ | ------ | --------------- | --------- | ------------- | --- | -------- | -------- |
| terms, | the label | accuracy | highly | benefits | of  | energy | combi- |                 |           |               |     |          |          |
liclyavailable.
nations. Thefullenergytermresultsinthebestoverallper-
| formance. |     |     |     |     |     |     |     | Acknowlegment.Theworkinthispaperwasfundedbythe |     |     |     |     |     |
| --------- | --- | --- | --- | --- | --- | --- | --- | ---------------------------------------------- | --- | --- | --- | --- | --- |
EUprojectSTRANDS(ICT-2011-600623).
4.2.QualitativeResults
| Figure | 4 shows |     | an overall | qualitative | comparison |     | be- |     |     |     |     |     |     |
| ------ | ------- | --- | ---------- | ----------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
1www.vision.rwth-aachen.de/publications/
tween[26](2D)andourapproach(3D).Besidesusingonly
2www.vision.rwth-aachen.de/furniture
71

References [23] C.V.Nguyen,S.Izadi,andD.Lovell. Modelingkinectsen-
|                        |     |     |                               |     |     |     | sor noise     | for improved | 3d reconstruction | and tracking. | In  |
| ---------------------- | --- | --- | ----------------------------- | --- | --- | --- | ------------- | ------------ | ----------------- | ------------- | --- |
| [1] A.BarbuandS.C.Zhu. |     |     | Graphpartitionbyswendsen-wang |     |     |     | 3DIMPVT,2012. |              |                   |               |     |
cuts. InICCV,2003. [24] S.OsherandN.Paragios. GeometricLevelSetMethodsin
[2] D.ComaniciuandP.Meer. Meanshift: Arobustapproach Imaging,Vision,andGraphics. Springer,2003.
| towardfeaturespaceanalysis. |     |     |     | PAMI,24(5):603–619,2002. |     |     |                                               |     |     |          |     |
| --------------------------- | --- | --- | --- | ------------------------ | --- | --- | --------------------------------------------- | --- | --- | -------- | --- |
|                             |     |     |     |                          |     |     | [25] B.Pepik,P.V.Gehler,M.Stark,andB.Schiele. |     |     | 3d2pm-3d |     |
[3] P.Dolla´randC.Zitnick. Structuredforestsforfastedgede- deformablepartmodels. InECCV,2012.
tection. InICCV,2013. [26] T.Pohlen,I.Badami,M.Mathias,andB.Leibe. Semantic
[4] P.F.FelzenszwalbandD.P.Huttenlocher. Efficientgraph- segmentationofmodularfurniture. InWACV,2016.
basedimagesegmentation. IJCV,pages167–181,2004. [27] H.Riemenschneider,S.Sternig,M.Donoser,P.M.Roth,and
[5] B.Fulkerson,A.Vedaldi,andS.Soatto. Classsegmentation H.Bischof. Houghregionsforjoininginstancelocalization
and object localization with superpixel neighborhoods. In andsegmentation. InECCV,2012.
ICCV. [28] N.RipperdaandC.Brenner. Reconstructionoffaadestruc-
[6] R.Girshick. Fastr-cnn. InICCV,December2015. tures using a formal grammar and rjmcmc. In DAGM-
| [7] P. J. | Green. | Reversible | jump | markov | chain monte | carlo | Symposium,2006. |     |     |     |     |
| --------- | ------ | ---------- | ---- | ------ | ----------- | ----- | --------------- | --- | --- | --- | --- |
computationandbayesianmodeldetermination.Biometrika, [29] J.ShiandJ.Malik. Normalizedcutsandimagesegmenta-
tion. PAMI,22(8):888–905,2000.
82:711–732,1995.
[8] S.Gupta,P.Arbelaez,andJ.Malik. Perceptualorganization [30] N. Silberman, D. Hoiem, P. Kohli, and R. Fergus. Indoor
and recognition of indoor scenes from RGB-D images. In segmentation and support inference from rgbd images. In
| CVPR,2013.         |     |                                  |     |     |     |     | ECCV,2012.     |           |               |                      |     |
| ------------------ | --- | -------------------------------- | --- | --- | --- | --- | -------------- | --------- | ------------- | -------------------- | --- |
|                    |     |                                  |     |     |     |     | [31] K. Smith, | S. O. Ba, | J.-M. Odobez, | and D. Gatica-Perez. |     |
| [9] F.HanandS.Zhu. |     | Bottom-up/top-downimageparsingby |     |     |     |     |                |           |               |                      |     |
attributegraphgrammar. InICCV,2005. Trackingthevisualfocusofattentionforavaryingnumber
[10] B.Hariharan,P.Arbela´ez,R.Girshick,andJ.Malik. Hyper- ofwanderingpeople. PAMI,30(7):1212–1229,2008.
|         |            |              |     |                  |           |     | [32] G.Stiny. | Pictorialandformalaspectsofshapeandshape |     |     |     |
| ------- | ---------- | ------------ | --- | ---------------- | --------- | --- | ------------- | ---------------------------------------- | --- | --- | --- |
| columns | for object | segmentation |     | and fine-grained | localiza- |     |               |                                          |     |     |     |
|         |            |              |     |                  |           |     | grammars.     | 1975.                                    |     |     |     |
tion. InCVPR,2015.
[11] V. Hedau. Recovering free space of indoor scenes from a [33] M. Sun, S. S. Kumar, G. Bradski, and S. Savarese. Ob-
singleimage. InCVPR,2012. ject detection, shape recovery, and 3d modelling by depth-
encodedhoughvoting.ComputerVisionImageUnderstand-
| [12] V. Hedau, | D.  | Hoiem, | and D. | Forsyth. | Thinking inside | the |     |     |     |     |     |
| -------------- | --- | ------ | ------ | -------- | --------------- | --- | --- | --- | --- | --- | --- |
ing,117(9):1190–1202,2013.
box: Usingappearancemodelsandcontextbasedonroom
geometry. InECCV,2010. [34] O. Teboul, I. Kokkinos, L. Simon, P. Koutsourakis, and
|     |     |     |     |     |     |     | N.Paragios. | Parsingfacadeswithshapegrammarsandre- |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------------------------------------- | --- | --- | --- |
[13] A.KanezakiandT.Harada.3dselectivesearchforobtaining
|                                           |     |              |     |     |               |     | inforcementlearning. |                | PAMI,35(7):1744–1756,2013. |                |     |
| ----------------------------------------- | --- | ------------ | --- | --- | ------------- | --- | -------------------- | -------------- | -------------------------- | -------------- | --- |
| objectcandidates.                         |     | InIROS,2015. |     |     |               |     |                      |                |                            |                |     |
|                                           |     |              |     |     |               |     | [35] Z. Tu           | and S. C. Zhu. | Image segmentation         | by data-driven |     |
| [14] M.Kass.,A.P.Witkin,andD.Terzopoulos. |     |              |     |     | Snakes:Active |     |                      |                |                            |                |     |
markovchainmontecarlo.IEEETrans.PatternAnal.Mach.
| contourmodels. |     | IJCV,1(4):321–331,1988. |     |     |     |     |     |     |     |     |     |
| -------------- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Intell.,24(5):657–673,2002.
| [15] V.KolmogorovandR.Zabih. |           |              |                                  | Whatenergyfunctionscanbe |             |      |                                              |     |     |                |     |
| ---------------------------- | --------- | ------------ | -------------------------------- | ------------------------ | ----------- | ---- | -------------------------------------------- | --- | --- | -------------- | --- |
|                              |           |              |                                  |                          |             |      | [36] C.A.Vanegas,D.G.Aliaga,andB.Benes.      |     |     | Buildingrecon- |     |
| minimizedviagraphcuts?       |           |              | InECCV,2002.                     |                          |             |      |                                              |     |     |                |     |
|                              |           |              |                                  |                          |             |      | structionusingmanhattan-worldgrammars.       |     |     | InCVPR,2010.   |     |
| [16] J. D.                   | Lafferty, | A. McCallum, |                                  | and F. C.                | N. Pereira. | Con- |                                              |     |     |                |     |
|                              |           |              |                                  |                          |             |      | [37] J.Zhang,C.Kan,A.G.Schwing,andR.Urtasun. |     |     | Estimat-       |     |
| ditionalrandomfields:        |           |              | Probabilisticmodelsforsegmenting |                          |             |      |                                              |     |     |                |     |
ingthe3dlayoutofindoorscenesanditsclutterfromdepth
| andlabelingsequencedata.                 |     |     | InICML,2001. |                         |                |     |                              |              |                                    |     |     |
| ---------------------------------------- | --- | --- | ------------ | ----------------------- | -------------- | --- | ---------------------------- | ------------ | ---------------------------------- | --- | --- |
|                                          |     |     |              |                         |                |     | sensors.                     | InICCV,2013. |                                    |     |     |
| [17] J.J.Lim,H.Pirsiavash,andA.Torralba. |     |     |              |                         | ParsingIKEAob- |     |                              |              |                                    |     |     |
|                                          |     |     |              |                         |                |     | [38] Y.ZhaoandS.Zhu.         |              | Sceneparsingbyintegratingfunction, |     |     |
| jects:Fineposeestimation.                |     |     | InICCV,2013. |                         |                |     |                              |              |                                    |     |     |
|                                          |     |     |              |                         |                |     | geometryandappearancemodels. |              | InCVPR,2013.                       |     |     |
| [18] A.MartinovicandL.VanGool.           |     |     |              | Bayesiangrammarlearning |                |     |                              |              |                                    |     |     |
|                                          |     |     |              |                         |                |     | [39] Y.ZhaoandS.C.Zhu.       |              | Imageparsingwithstochasticscene    |     |     |
| forinverseproceduralmodeling.            |     |     |              | InCVPR,2013.            |                |     |                              |              |                                    |     |     |
|                                          |     |     |              |                         |                |     | grammar.                     | InNIPS,2011. |                                    |     |     |
[19] M.Mathias,A.Martinovic,andL.V.Gool.ATLAS:Athree- [40] B.Zheng,Y.Zhao,J.C.Yu,K.Ikeuchi,andS.Zhu.Beyond
layered approach to facade parsing. IJCV, 118(1):22–48, point clouds: Scene understanding by reasoning geometry
2016.
|                  |            |             |          |                 |        |       | andphysics.    | InCVPR,2013. |              |                 |      |
| ---------------- | ---------- | ----------- | -------- | --------------- | ------ | ----- | -------------- | ------------ | ------------ | --------------- | ---- |
| [20] M. Mathias, | A.         | Martinovic, |          | J. Weissenberg, | and L. | J. V. |                |              |              |                 |      |
|                  |            |             |          |                 |        |       | [41] B. Zheng, | Y. Zhao,     | J. C. Yu, K. | Ikeuchi, and S. | Zhu. |
| Gool.            | Procedural | 3d          | building | reconstruction  | using  | shape |                |              |              |                 |      |
Sceneunderstandingbyreasoningstabilityandsafety.IJCV,
| grammarsanddetectors. |     |     | In3DIMPVT,2011. |     |     |     | 112(2):221–238,2015. |     |     |     |     |
| --------------------- | --- | --- | --------------- | --- | --- | --- | -------------------- | --- | --- | --- | --- |
[21] M.Mathias,A.Martinovic´,J.Weissenberg,S.Haegler,and
| L.VanGool. |     | Automaticarchitecturalstylerecognition. |     |     |     | In  |     |     |     |     |     |
| ---------- | --- | --------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
F.RemondinoandS.El-Hakim,editors,Proceedingsofthe
4thISPRSInternationalWorkshop3D-ARCH2011.ISPRS,
2011.
[22] P.Mu¨ller,G.Zeng,P.Wonka,andL.VanGool.Image-based
| proceduralmodelingoffacades. |     |     |     | InSIGGRAPH,2007. |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- |
72