# Robot4Art

**[English](README.md) | [中文](README.zh.md)**

一个人机协作的互动绘画系统，实时创作极简主义画作。

访客回答几个简短的个人问题——最喜欢的颜色、最喜欢的城市、当下的一个梦想或愿望——并选择一个机器人平台。一个真实的 Claude 模型直接构图：每一笔的形状、颜色和粗细都是模型自己的选择，最多 20 笔，使用一小组常见的笔色（不是自由调配的颜料）。机器人手臂随后在几分钟内现场画出这幅作品，同时屏幕上展示模型自己给出的、未经编辑的创作理由。本项目为 [CoRL](https://www.corl.org/)（机器人学习会议）的公开演示而建。

**[立即体验 →](https://jack-sherman01.github.io/Robot4Art/)** —— 在浏览器中生成一幅画作（通过一个小后端发起真实的模型调用，见 [`worker/`](worker/)），然后看同一个构图在 NVIDIA Isaac Sim 中由模拟的 Franka 机械臂画出来，两边并排对比：模型构思的画作 vs. 机器人全身运动的真实录制视频。关于演示展示了什么、没有覆盖什么，见下方的[工作原理](#工作原理)。

Robot4Art 是卡内基梅隆大学 Safe AI Lab、破壳机器人（Poke Robotics）和洛可可创新设计集团（LKK Design Group）之间的合作项目。

## 工作原理

1. **互动输入。** 访客回答几个简短的个人问题，并选择一个机器人平台（Franka Emika Panda、Kinova Gen3 或 UFACTORY xArm）。
2. **语义构图。** 一个真实的 Claude 模型直接创作整幅作品——每一笔自己的 2-6 个控制点、笔的颜色和粗细，最多 20 笔，颜色来自一小组常见的笔色，外加它自己选择的一种画笔/工具——而不是从模板中挑选。[GitHub Pages 演示](https://jack-sherman01.github.io/Robot4Art/) 会调用一个小型 Cloudflare Worker 后端，该后端在服务器端保存模型 API 密钥（见 [`worker/`](worker/)）；如果无法连接，则会改用一个快速的确定性本地预览（[`sim/composition.py`](sim/composition.py) 移植到客户端 JavaScript），并会明确标注为预览。无论哪种方式，模型（或预览）自己给出的解释都会在画作被绘制的同时显示在屏幕上。
3. **笔画规划。** 模型给出的每一笔的少量控制点会被平滑成一条密集的曲线（Catmull-Rom 样条），并带有抬笔/落笔状态——这是生成步骤与机器人执行步骤之间共享的表示方式。
4. **机器人无关执行。** 笔画计划会被重新映射到所选机器人的任务坐标系并实际执行。

完整的技术细节——研究问题、相关工作、评估计划、时间表——记录在一份双语（中英文）研究计划书中，该文档未发布在本仓库中。

## 当前状态

| 部分 | 状态 |
| --- | --- |
| 确定性构图（`sim/composition.py`） | 已完成。在几个手写的笔画"语法"之间选择——作为网页演示的本地预览兜底方案，同时也可用作快速、免费、无需仿真器的几何/渲染流程检查。 |
| 真实 LLM 构图（`sim/llm_composer.py`、`worker/`） | 已完成，有两种调用方式。批处理：使用无头 `claude` CLI（无需单独的 API 密钥），用于生成缓存的 Isaac Sim 示例画作。实时：一个 Cloudflare Worker（`worker/`），在服务器端保存真实的 Anthropic API 密钥，驱动公开网页演示的"Paint it"按钮——由于每次请求都会花费真实费用，因此按 IP 限流并设有全局每日上限。两者都要求模型直接创作每一笔的形状、颜色和粗细，而不是从模板中挑选。 |
| 网页演示（`docs/`） | 已完成。调用实时 Worker 获取真实构图；如果无法连接，则回退到确定性本地预览（并明确标注）。同时展示一段真实录制的 Isaac Sim 视频，内容是 Franka 机械臂画出由真实 Claude 创作的作品，并与模型构图的静态渲染图并排对比。通过 GitHub Pages 发布；Worker 是唯一的非静态部分。 |
| Isaac Sim 验证 + 视频（`sim/franka_paint_sim.py`、`sim/franka_paint_video.py`） | 已完成，通过 Docker 运行（见 [`sim/README.md`](sim/README.md)）。已用模拟的 Franka 机械臂完整验证：所有笔画都能执行，记录的轨迹与预期构图吻合。视频现在也会在画布上渲染出画出的笔画痕迹（每笔一个网格），而不再是空白画布——不过实时视口捕获存在一个间歇性的冻结问题，目前仍在排查中，可能导致录制视频中较晚的笔画渲染不完整。 |
| Kinova / xArm 适配器 | 尚未开始。 |
| 实体硬件 | 尚未开始。 |

## 仓库结构

```
sim/            构图模块 + Isaac Sim 仿真流程（Python）
docs/           静态网页演示，通过 GitHub Pages 发布
worker/         为网页演示的实时模型调用提供服务的 Cloudflare Worker 后端
private/        研究计划书及其他内部草稿（已加入 .gitignore，不在本仓库历史记录中）
```

关于如何运行构图预览和 Isaac Sim 验证，见 [`sim/README.md`](sim/README.md)。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。
