# 隧道场景 UGV 导航避障技术路线深度调研报告

## Executive summary

结合你们当前的项目状态、场景约束和工程目标，我的核心判断是：**不要把项目的主线理解成“车道线检测”问题，也不要把 YOLO 当成唯一主模型问题；你们真正要做的是一个“受限通道内的可通行区域识别 + 硬边界识别 + 动态/静态障碍物检测 + 局部风险地图 + 保守型局部规划”的模块化导航系统。**这也是工业无人车、AMR/AGV、以及多数可落地自动驾驶系统更常见的组织方式：感知输出可供规划消费的环境表示，规划与控制单独成层，系统外再叠一层独立安全监控，而不是直接做端到端驾驶策略。Autoware 公开架构仍然沿用 perception / planning / control 的分层设计；Nav2 也以 costmap、planner、controller、collision monitor 作为核心运行框架。citeturn12search3turn12search1turn8search2turn10search2

就你们当前阶段而言，**最务实、最稳妥的主线**不是“纯传统视觉”也不是“直接上 BEV occupancy 大模型”，而是：**RGB 单目多任务感知 + 经典局部规划 + 显式安全约束**。其中，`ego_passable`、`ditch / barrier / wall` 这类“区域/边界/不可跨越语义”应以**语义分割**为主；`worker / construction_vehicle / suspended_object / debris` 这类“可数目标”应以**目标检测**为主；随后再把图像空间结果通过标定和地面几何投影为局部 BEV risk grid / costmap，交给 DWA/DWB 或后续 MPPI 去做局部避障。这个路线与 YOLOP、HybridNets 这类多任务驾驶感知工作在思路上是一致的，但你们需要做更适合隧道硬边界约束的“工业版变体”，而不是照搬公路 lane detection 任务。citeturn11search0turn11search1turn8search2turn8search3turn3search2turn3search3

**YOLO 的边界非常明确。**它非常适合做人、工程车、悬挂物、可枚举障碍物的检测，必要时也可做实例分割；但它不是你们“不可跨越边界与可通行区域”这类核心任务的最佳单一建模方式。官方文档也明确区分了 object detection、instance segmentation、semantic segmentation：检测只给框，实例分割给单个目标 mask，语义分割才是给每个像素赋类标签，最适合场景理解和自动驾驶类 freespace / drivable area 任务。citeturn16search1turn16search3turn16search4

从更长周期看，**LiDAR 或其他距离传感器几乎肯定值得接入**。地下/隧道环境通常同时具备 GNSS 不可用、低照度、重复纹理、长直走廊几何退化、灰尘/反光/积水等特征；DARPA SubT、LAMP、SIGNAV 等地下机器人和视觉退化环境工作，基本都依赖 LiDAR、视觉、IMU、里程计等多传感器协同，而不是把单目 RGB 作为唯一安全基座。不过在 MVP 阶段，你们完全可以先以 RGB 主线跑通，再为后续 2D/3D LiDAR 预留接口。citeturn14academia19turn14academia18turn3search6turn14search3turn14search10

对你们团队此刻最重要的，不是再讨论一个“更先进”的模型名词，而是尽快建立起这四条工程主线并行推进：**标注规范、数据转换与可视化、轻量分割 baseline、规划/安全联调接口。**只要这四件事先立住，后续无论你们换 PIDNet、BiSeNet、SegFormer，还是增加 YOLO / RT-DETR / LiDAR 融合，都会比较顺。citeturn1search1turn1search0turn0search0turn0search13turn10search0

## 行业常见做法与任务建模

地下隧道、矿区、工业通道、施工通道这类场景，与标准城市道路自动驾驶有一个关键差异：**它们通常不是“识别车道线再沿车道行驶”，而是“在有限宽度、强约束、局部动态障碍存在的通道里识别哪里能走、哪里绝不能跨越、哪里需要减速或停车”。**因此，业界和机器人界更常见的建模语言不是 lane line本身，而是 **drivable area / freespace、hard boundary、occupancy、traversability、costmap**。Nav2 的 costmap 体系本身就是把环境表征转换成机器人可执行的栅格代价；keepout filter 则可以把禁入区、首选通道、速度限制区直接编码进规划层。citeturn8search2turn8search3turn8search9

从地下机器人与工业移动机器人实践看，**成熟系统通常至少做两件事**：第一，做稳定的几何/占据表达，而不是只输出目标框；第二，把安全功能做成**独立于主规划器的保底层**。ISO 3691-4 针对 driverless industrial trucks/AMR 明确强调了安全要求、人员检测、制动和验证机制；Nav2 的 Collision Monitor 也是“绕过 costmap 和 planner，在 emergency-stop 级别拦截 `cmd_vel`”的独立安全节点。对于你们项目，这意味着“模型输出一个 mask 就够了”还不够，**还需要有硬边界 keepout、膨胀安全距离、低置信度降速/停车、感知失效时的安全降级**。citeturn4search0turn4search4turn10search2turn10search5

地下与隧道环境的传感挑战也决定了为什么行业很少把**纯单目视觉**当成最终安全闭环。LAMP 和 DARPA SubT 的地下系统强调了黑暗、粉尘、泥泞、长廊自相似场景、轮速不准等问题；SIGNAV 在视觉退化环境里专门把前向相机、LED 灯、LiDAR、IMU 结合起来；而关于 tunnel-like environments 的研究则指出，LiDAR 虽然抗低照度，但在长直隧道里也会出现几何退化，可观测性变差。换句话说，**“视觉不稳”与“LiDAR 万能”都是错误认知**，真正稳的是多传感器 + 保守安全策略。citeturn14academia19turn14academia18turn14search3turn14search10

对你们的具体任务，我建议按下面这套建模来理解，而不是沿用“车道识别”这个公路语境：

| 任务对象 | 更合理的建模 | 原因 | 你们现在是否必须做 |
|---|---|---|---|
| `ego_passable` 本车侧可通行地面 | 语义分割 / drivable area / freespace segmentation | 这是连续区域，不是离散目标 | 必须 |
| `ditch / left_barrier / right_barrier / tunnel_wall` | 语义分割 + 运行时合成 hard-boundary keepout | 这是“不可跨越区域/边界”，核心是像素级空间约束 | 必须 |
| `worker / construction_vehicle / suspended_object` | 目标检测；必要时再加实例分割 | 是可数目标，要用于动态避障、速度策略 | 必须 |
| `debris` | 先检测；对关键类别再补实例/语义分割 | 小样本长尾多、形状不规则，先做可见即可 | 建议 |
| `hard_boundary_edge` | 辅助边界监督 | 边界薄、几何敏感时有帮助，但不一定要人工单独标 | 可选 |
| `surface_risk_map` | 先规则派生 costmap，后续再学习残差风险 | 风险本质上是规划代价，不必一开始就作为独立强监督任务 | 第二阶段 |

这个划分与语义分割、实例分割、检测三类任务的本质定义是一致的：语义分割适合像素级场景理解；实例分割适合需要单个目标轮廓时；目标检测适合离散目标定位。YOLOP 和 HybridNets 等多任务驾驶感知工作之所以同时做 detection + drivable area + lane，是因为它们面对的也是“离散目标 + 连续可行驶区域 + 关键边界”的混合感知问题。你们只是把公路 lane 换成了“沟槽/墙体/隔离带/隧道硬边界”而已。citeturn16search1turn16search3turn11search0turn11search1turn6search6

## 候选技术路线比较

下面这张表重点不是比较“谁论文更先进”，而是比较**谁更适合你们当前阶段把系统跑起来**。表中的“推荐度”是针对你们当前条件：仅 RGB 视频、部分标注、需要 MVP、场景相对固定、实时与安全要求高。相关模型和系统能力来自各自论文或官方文档；“是否适合当前阶段”是结合你们 ODD 的工程判断。citeturn0search0turn1search0turn1search1turn0search13turn2search8turn2search5turn2search19turn7search1turn9search5turn9search2turn9search7

| 路线 | 长处 | 短板 | 对你们当前阶段的判断 |
|---|---|---|---|
| OpenCV 规则法 | 无需训练、可快速出 demo、便于理解失效机理 | 强依赖光照/纹理/阈值，遇到反光、水渍、阴影很脆弱 | **必须保留**，但只应作为 baseline / fallback / QA 工具 |
| YOLO / RT-DETR 检测 | 对人、车、悬挂物、离散障碍物很合适；实时性强 | 不擅长把连续可通行区域和硬边界建成核心环境表示 | **必须有，但不是唯一主线** |
| UNet / DeepLab / SegFormer / BiSeNet / PIDNet 语义分割 | 适合 passable / ditch / wall / barrier 等像素级任务 | 仅做分割不足以替代目标检测；某些模型偏重 | **当前主线** |
| 多任务轻量网络 | 共享 backbone，统一输出 detection + passable + boundary，工程上更顺 | 训练和 loss balance 比单任务难；需要较清晰标注规范 | **很适合你们** |
| 纯 BEV occupancy prediction | 规划友好、表达统一 | 通常更依赖多相机或 LiDAR 与更重 supervision | **现在不该当主线** |
| 单目深度 + traversability | 可补几何线索，未来有潜力 | 绝对尺度、跨域泛化和低光稳健性仍是风险 | **可做增强，不宜做主安全层** |
| LiDAR-RGB 融合 | 对边界、距离、低光更稳 | 需要新硬件、标定、同步、栈复杂度上升 | **中期强烈建议** |
| SAM / SAM2 / Grounding DINO | 非常适合离线辅助标注、视频传播、开集找物体 | 不是固定语义、安全实时车载主模型的最佳形态 | **只建议用于数据工具链** |
| 端到端学习导航 | 研究前沿，理论上可省中间模块 | 数据量、闭环验证、可解释性、安全封装压力极大 | **不适合 MVP** |
| RL / diffusion planner | 在基准测试里有亮点 | 训练/部署门槛高，工程可控性与安全验证压力大 | **现阶段不建议投入主线** |

如果把这张表再压缩成一句话：**你们不是在 YOLO 和分割之间二选一，而是在“检测负责人/车/障碍物，分割负责可通行区域和硬边界，风险地图负责把它们统一成规划可消费表示”这条线上做系统集成。**

关于 **YOLO 的适用边界**，结论可以说得很直接。第一，YOLO 非常适合你们的 `worker`、`construction_vehicle`、`suspended_object`，以及大多数 `debris` 的第一版落地；如果后面发现碎石/箱体轮廓对路径边缘很关键，再把 `debris` 从 box 升级为 instance segmentation。第二，YOLO 不应该承担 `ego_passable`、`ditch`、`wall`、`hard boundary` 的核心建模责任，因为这些任务本质上属于 dense scene understanding。Ultralytics 官方文档对 semantic segmentation 与 instance segmentation 的区分已经非常明确；RT-DETR 也主要是实时检测器，不会替代高质量场景分割头。citeturn16search1turn16search3turn16search4turn0search13

关于 **具体分割骨干**，如果你们追求“尽快训出可用模型 + 后续逐步提精度”，我建议采用“双 baseline 策略”：**一个轻量实时线，一个精度线。**轻量实时线优先考虑 PIDNet-S / BiSeNetV2；二者都明确以 real-time semantic segmentation 为目标。精度线优先考虑 SegFormer-B0/B1；它在效率与稳健性之间通常比 DeepLab、传统 UNet 更像现代工程基线。DeepLab 仍然是老牌强 baseline，U-Net 仍然是非常好的小样本起步模板，但若以自动驾驶/机器人场景的实时部署为目标，PIDNet、BiSeNet、SegFormer 往往更贴合。citeturn1search1turn1search0turn0search0turn1search6turn1search3

关于 **SAM / SAM2 的合理用途**，你们现在的理解基本是对的。SAM 的设计目标是“promptable segmentation”，SAM 2 进一步扩展到视频并引入 streaming memory，Grounding DINO 则擅长 open-vocabulary detection。它们很适合做：离线自动 proposal、基于文本找长尾对象、视频级 mask 传播、标注 QA；但它们并不是为固定类别、车载实时、安全可解释的窄 ODD 感知主模型而生。你们现在把它们放在**数据生产工具链**里最划算，而不是放在在线主感知闭环里。citeturn5search0turn5search1turn5search2turn5search12

至于 **端到端、RL、diffusion planner**，我建议明确降级到情报关注而非主线投入。UniAD 代表的是 planning-oriented 统一框架，Diffusion-ES / Diffusion-Planner 代表的是学习式规划前沿，RL-based motion planning 也有持续综述更新；但这些路线当前更适合有大量闭环数据、成熟仿真基座、系统验证团队的研究型/平台型组织。对你们这种刚接手、还在补标注、要先做 MVP 的工程局面，它们的机会成本太高。citeturn9search5turn9search2turn9search10turn9search7

## 适合你们项目的推荐系统架构

我建议你们把主系统明确拆成 **感知层、几何/栅格层、规划控制层、安全层** 四层。这样做有三个好处：第一，便于先跑 MVP；第二，便于之后换模型不动 planner；第三，便于未来接 LiDAR、深度传感器或者更强定位模块而不推翻现有代码。Autoware 的模块化架构、Nav2 的 planner/controller/costmap 组合，都是这一思路。citeturn12search3turn12search1turn8search2

**感知层输出建议如下。**你们至少应同时输出两类结果：一类是**分割类输出**，包括 `ego_passable`、`ditch`、`left_barrier`、`right_barrier`、`tunnel_wall`；另一类是**检测类输出**，包括 `worker`、`construction_vehicle`、`suspended_object`、`debris`。如果资源允许，再加一个**边界辅助头**，输出 `hard_boundary_edge`。这类“区域 + 边界 + 目标”的多任务结构，与 YOLOP/HybridNets 的共享 encoder、多 decoder 设计高度相似，只是你们的语义集合是隧道工业化版本。citeturn11search0turn11search1

**几何/栅格层** 不要等到 BEV occupancy 大模型成熟再做；你们现在就可以做。工程上最简单且非常有效的方案是：对前视相机做内外参标定，假设近场地面近似可投影，用分割 mask 和检测结果通过地面几何映射到本车局部坐标系，生成一个 2D risk grid / costmap。这里不要求“学出 3D 世界”，只要求把近场 5–15 米范围内的可通行、禁入、障碍和不确定区域表达清楚。Nav2 的 costmap 本来就是为这种表示服务的，keepout filter 能表达绝对禁入区，inflation layer 能把障碍周围变成高代价带。citeturn8search2turn8search3turn8search0

我建议在 costmap 中做四级语义。**第一级是 free**：高置信 `ego_passable`。**第二级是 lethal keepout**：`ditch / barrier / tunnel_wall` 以及动态障碍占据单元。**第三级是 inflated risk**：人员、工程车、悬挂物和障碍物周围的安全膨胀区。**第四级是 unknown / low-confidence caution**：暗光、强反光、水渍、边界断裂、分割熵高、投影不稳定等区域。你们现在完全可以先用规则从 mask 派生 risk，而不是一上来就训练 `surface_risk_map`。从规划角度讲，这样的派生 risk map 比一个未经验证的学习式 risk score 更可控。citeturn8search0turn8search2turn10search2

**planner 选择上，我建议非常保守。**如果你们当前任务以“沿单侧可通行区域前进”为主，而不是复杂会车、掉头、倒车入位，那么**局部规划的优先级高于复杂全局规划**。现阶段可直接沿用 DWA/DWB 作为本地控制器，原因不是它最先进，而是它简单、可解释、参数透明、现有 baseline 已经在用，而且 Nav2 中 DWB 本就是 DWA 系思想的默认控制器。等你们把 risk grid 稳定住之后，如果发现 DWA 在狭窄通道、动态障碍让行、速度平滑性上明显不足，再升级到 MPPI。Hybrid A* / State Lattice 更适合需要更强非完整约束全局轨迹搜索的情形；在你们这个“隧道通道 + 局部避障”问题里，不应抢占第一阶段资源。citeturn3search2turn3search3turn3search4turn3search9

**安全层** 是这个项目最不该省的部分。我建议至少实现以下硬约束：一，任何 `hard_boundary` 命中本车膨胀 footprint 都必须触发 stop，不允许 planner“尝试挤过去”；二，人员、工程车、未知大障碍前设置更大的 inflation 半径与更低速度上限；三，感知低置信度、画面极暗、镜头污染、短时丢帧时必须自动降速或停车；四，感知结果 stale、时间同步异常、投影失败时必须进入安全降级；五，Collision Monitor 之类的外层拦截要独立于主 planner 存在。工业 AMR/AGV 的标准和公开实践都强调安全保护装置、人员检测和制动验证；这一层应视作“保命层”，而不是“优化项”。citeturn4search0turn4search4turn10search2turn10search5

如果用一句更具体的话来概括推荐架构，那就是：

**前视 RGB → 多任务感知网络（passable + barrier + obstacles）→ 近场 BEV risk grid / keepout costmap → DWB/DWA 控制器 → Collision Monitor / 安全状态机兜底。**

这条链路是你们现在最应该先打透的主线。

## 数据与标注建议

你们现有标签整体上是**合理的，而且方向是对的**。其中 `ego_passable`、`ditch`、`left_barrier`、`right_barrier`、`tunnel_wall` 用 polygon 非常合适；`worker`、`construction_vehicle`、`suspended_object` 用 box 也合适；`debris` 则建议采用“**默认 box，关键子类补 polygon**”的折中策略。这样做的原因并不复杂：可通行区、沟槽、墙体这类对象本质上是连续区域，最适合语义分割；人员、车辆这类可数实体最适合检测；而 `debris` 既有离散目标属性，又可能对边缘栅格造成细碎侵占，所以不必一刀切。citeturn16search1turn16search3

关于 `hard_boundary`，我的建议是：**训练标签里不要只保留一个合并后的 hard_boundary，而要继续保留 `ditch / left_barrier / right_barrier / tunnel_wall` 的细分类；运行时再把它们合成为统一 hard-boundary keepout。**这样做有三个好处。第一，不同类别后续可以赋不同代价和安全半径，比如 `ditch` 绝对 lethal，`wall` 近壁区可以加高代价但保留极窄可通道。第二，出问题时可诊断是“沟槽漏检”还是“墙脚误分”。第三，未来若接 LiDAR 或深度，几何融合时不同类别的先验不同。换言之，**训练时保持细粒度，规划时做统一抽象**。这比一开始就把所有不可跨越物合并成一个大类更有工程弹性。citeturn8search2turn8search3

关于 `hard_boundary_edge`，我建议**要，但不建议一开始人工逐帧单独标**。更务实的做法是：先人工标好 polygon，然后在数据转换阶段自动从 `ditch / barrier / wall` 的 mask 轮廓里生成 edge target，用作辅助 loss。原因是 lane/area 与 boundary 本来就具有互补关系，联合监督往往会提升薄边界质量；但让标注同事再多做一套精细 edge 标会显著提高成本且一致性差。citeturn6search6

关于 `surface_risk_map`，我不建议你们现在把它作为强监督人工标注任务。风险图本质上是规划代价图，不是天然客观、单一正确答案的“事实标签”。在你们这个阶段，更好的做法是：**先从语义标签和检测结果规则派生 risk**。例如：`ditch / barrier / wall` 映射为 lethal；人员、工程车和大障碍周围做膨胀；边界附近增加代价；低置信度区域打 caution；强反光/水渍可先用启发式视觉质量检测加额外代价。等系统跑起来之后，再考虑学习一个 residual risk head 去修正规则图，而不是从零手工标 dense risk。Nav2 的 layered costmap + inflation 本身就非常适合做这种“规则起步、后续再学习”的演进。citeturn8search0turn8search2

关于**最小可用训练集**，我建议把目标分成三个层次，而不要追求一个神奇数字。第一层是 **smoke test 集**：大约 300–500 张关键帧，覆盖白天/暗光/反光/积水/阴影/工人/工程车/沟槽破损等主要模式，用来验证数据管道、loss 是否收敛、可视化是否正常。第二层是 **首个 MVP 集**：大约 1,000–3,000 张多序列关键帧，重点保证场景多样性和长尾覆盖，而不是只堆相邻视频帧。第三层是 **长尾增强集**：专门补 hardest cases。迁移学习和预训练确实可以显著降低从零开始所需数据量，但前提是标注定义清晰、数据切分正确、长尾样本被主动补齐。citeturn13search8turn13search6turn13search1

`train / val / test` 的切分一定要**按序列、按地点、按时间段切**，不能随机按帧切。隧道视频相邻帧高度相似，如果你按帧随机切，val/test 指标会虚高，完全不能反映真实泛化。建议至少做到：测试集来自不同采集日期、不同通道段、不同光照条件；同时单独维护一个 **hard set**，专门放暗光、强反光、水渍、遮挡、人员贴边、工程车占道、沟槽边沿模糊、悬挂物靠近车体等样本。这个 hard set 不一定大，但必须稳定。citeturn14academia19turn14search3

标注质检我建议做成“工具化而不是口头化”。最低限度要有：**overlay 浏览器、类别颜色统一、mask 面积分布统计、空标/重叠/自交 polygon 检查、box 是否越界检查、每类样本数趋势、按序列的漏标热图**。如果能再往前走一步，建议抽取 5%–10% 数据做双人复核，对关键类（尤其 `ditch` 和 `tunnel_wall`）计算一致性；同时把难例和争议例收进一本“标注裁决手册”。SAM2 和 Grounding DINO 很适合放到这条链路里做 proposal 和视频传播，但最终要有人复核。citeturn5search1turn5search2

## 风险、失效模式与安全机制

你们这个项目真正危险的，不是平均场景下“精度差一点”，而是**在少数恶劣场景里系统自信地错**。地下和隧道环境的典型失效模式包括：暗光导致边界消失、反光和积水把沟槽误当可通行地面、墙脚阴影与沟槽混淆、长直隧道自相似场景导致定位/几何感弱、粉尘/污渍降低可见度、工程车遮挡真实边界、悬挂物因尺度和视角问题漏检、以及单目视觉对真实距离与高度缺乏可靠约束。地下机器人和视觉退化环境研究反复指出，这类环境同时对视觉与几何感知都不友好。citeturn14academia19turn14academia18turn14search3turn14search10turn2search19turn2search15

第一个关键原则是：**hard boundary 永远当成安全任务，不当成一般语义任务。**也就是说，`ditch / barrier / wall` 的漏检代价要远大于误检代价。工程实现上，这意味着你们在 loss、阈值、后处理和状态机里都要偏保守：宁愿把边界附近多判一些不可走，也不要让 planner 试图“擦边蹭过去”。配合 inflation layer，把边界和大障碍周围扩成高代价带，而不是只当一条细线。citeturn8search0turn8search2

第二个关键原则是：**unknown 不等于 free。**如果图像质量差、模型置信度低、时序一致性崩掉、投影失败、目标框抖动剧烈，你们的系统应默认转入 caution 或 stop，而不是继续按上一个正常状态高速行驶。Nav2 的 keepout、speed filter、collision monitor 提供了与这种保守策略相符合的工程框架；工业安全标准也要求系统对危险情形有验证过的制动和保护机制。citeturn8search3turn8search9turn10search2turn4search0

第三个关键原则是：**主模型之外必须有“平行保底链路”。**你们其实已经有这个基础：OpenCV baseline。虽然它不够鲁棒，但它非常适合作为两个角色存在：一是**快速 fallback**，在深度模型置信度低时提供极保守的边缘/沟槽报警；二是**数据 QA 工具**，帮助发现标注漏标和模型系统性错误。工业系统里，真正稳定的方案很少是“一个模型全都做对”，而更像是“主模型 + 规则校验 + 外层安全拦截”。citeturn10search2turn4search0

第四个关键原则是：**未来即使上 LiDAR，也不要把 LiDAR 神化。**LiDAR 对低照度更强，但 tunnel-like 环境中的长直走廊会导致几何退化；因此未来若接 2D/3D LiDAR，最好与 IMU、轮速、相机而不是孤立使用。你们未来的传感器升级路线，我建议优先级是：**补主动光源/相机曝光质量 → 接 2D safety LiDAR 或前向深度 → 接 3D LiDAR / RGB 融合 → 再考虑更复杂的 occupancy / traversability 学习**。citeturn14search10turn14academia19turn14search3turn7search21turn7search0

## MVP 与长期路线图

你们给出的六阶段划分，本身就是一个很好的骨架；我建议把它进一步压实为“**八到十二周内可交付的 MVP 主线**”和“**之后三到六个月的增强主线**”。

**MVP 主线**应聚焦四个交付件：一，数据工具链可运行；二，分割 baseline 能稳定输出 `ego_passable + hard boundary`；三，局部 risk grid 能驱动 planner；四，安全状态机能在低置信/感知异常时降速或停车。这个阶段不要同时追求 BEV 大模型、深度估计、重定位、端到端策略。只要做到“在典型隧道通道里，车辆能稳定沿本车侧可通行区前进，不碰墙、不跨沟、不撞显著障碍，感知异常时会保守停车”，MVP 就成立。citeturn8search2turn3search2turn10search2

下面是我建议的阶段化落地表：

| 阶段 | 目标 | 主要产物 |
|---|---|---|
| 阶段一 | 不训练先跑通底线 | OpenCV baseline 复现；视频回放；错误场景清单；标注规范初稿 |
| 阶段二 | 打通数据工具链 | Labelme JSON → mask/YOLO 格式；overlay 可视化；类别统计；自动质检 |
| 阶段三 | 先训分割主线 | `ego_passable + ditch + barriers + wall` 的单任务/多类分割 baseline；smoke test 指标 |
| 阶段四 | 接规划闭环 | 图像 mask → BEV risk grid / costmap；DWA/DWB 联调；keepout / inflation 参数集 |
| 阶段五 | 加动态目标检测 | YOLO 或 RT-DETR 接 `worker / vehicle / suspended / debris`；风险膨胀与速度策略 |
| 阶段六 | 做系统级保守化 | 低置信度降速、时序滤波、Collision Monitor、失效降级、hard set 回归测试 |
| 后续增强 | 提升鲁棒性 | 多任务网络、边界辅助头、主动光源优化、深度/2D LiDAR/3D LiDAR 融合 |

对 **长期路线**，我建议按下面顺序升级，而不是反过来。先从“单任务分割 + 独立检测”升级到“共享 encoder 的多任务网络”；再从“纯图像空间”升级到“更稳健的局部 BEV risk fusion”；再接 LiDAR 或深度；最后才看是否值得投入学习式 traversability / occupancy。原因很简单：前面每一步都能直接增强现有系统，而后面几步如果前置条件不够，只会放大系统复杂度。BEVFormer、SurroundOcc 这类工作代表了高阶 BEV / occupancy 方向，但它们多面向多相机自动驾驶和更重 supervision，不适合你们此刻当主线。citeturn2search8turn2search5turn2search12

对一个**两人团队**，最现实的并行分工如下。

一位同事负责**数据与场景资产**：持续标注；维护标注规范；整理场景标签（暗光、反光、水渍、工人、工程车、沟槽破损、遮挡、悬挂物）；每天输出可抽检的 overlay；维护 hard set；用 SAM2 / Grounding DINO 辅助提效，但保证人工最终裁决。另一位同事负责**工具链与模型/规划联调**：写转换脚本；搭建可视化、统计和自动质检；跑 PIDNet / BiSeNet / SegFormer 三条 baseline；实现 mask→risk grid；接 DWB/DWA、Collision Monitor 和安全状态机；把检测模块在第五阶段插入同一 risk grid。这样的分工能让“标注生产”和“系统消费”同步前进，避免出现一边疯狂标、一边没人能吃数据，或者一边写模型、一边数据格式天天变的典型混乱。citeturn1search1turn1search0turn0search0turn5search1turn5search2turn3search2turn10search2

如果让我把**下周就该做的具体动作**再压缩成一张清单，那会是下面这十件事：

1. 冻结 v1 类别定义与颜色表，特别是 `ditch / left_barrier / right_barrier / tunnel_wall` 的判定边界。  
2. 写 `Labelme -> semantic mask` 转换器，并自动导出可视化 overlay。  
3. 自动从 polygon 生成 `hard_boundary_edge` 辅助标签，不额外人工重标。  
4. 建立数据 dashboard：每类数量、面积分布、序列统计、空标/重叠检查。  
5. 做 300–500 张 smoke set，先训一个轻量 PIDNet/BiSeNet 基线。  
6. 同时训一个 SegFormer-B0/B1 精度基线，确定“速度线”和“精度线”。  
7. 实现图像 mask 到局部 risk grid 的投影，先不追求 fancy BEV。  
8. 沿用现有 DWA/DWB，先把 keepout、inflation、stop zone 调通。  
9. 在第五阶段插入 YOLO/RT-DETR，只接人、车、悬挂物、碎石等检测。  
10. 搭建 hard set 回归脚本，每次训练后固定回放最危险 20–50 段视频。  

最后给一个明确结论：**你们当前最优路线不是“YOLO 还是不 YOLO”，也不是“要不要直接上 BEV occupancy”，而是“先把分割主线、检测副线、risk grid、局部规划和安全降级做成一个可运行、可调试、可回归的系统”。**只要这条主线成立，后面接 LiDAR、做更强多任务网络、引入深度估计或更高级 planner，都会是顺势增强；反过来，如果现在跳去做大而全的新范式，极大概率会把项目带入“模型很多、系统不稳、没人敢开”的状态。这个项目最需要的不是论文式新颖性，而是**工程上可验证的可靠性递增**。citeturn12search3turn8search2turn4search0turn10search2turn14academia19