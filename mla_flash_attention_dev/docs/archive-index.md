# 文档归档索引

更新时间：`2026-04-13`

这份索引用来标记当前 `mla_flash_attention_dev/docs/` 里已经进入**历史参考**状态的文档。

这些文档不是删除，而是**就地归档**：

- 保留原始内容，方便回看历史思路
- 但不再作为当前论文主线或项目执行主线
- 后续阅读时，应优先参考 `current-docs.md`

## 已归档文档

### 1. 旧的论文方向/路线规划

- `tenstorrent-mla-paper-ideas.md`
- `tenstorrent-mla-expanded-roadmap.md`

归档原因：

- 这两份文档包含较多早期 brainstorming 与方向并列方案
- 当前已经有更清晰的统一口径，不希望继续把论文主线理解成多个并列候选题目

### 2. 旧的阶段执行计划

- `next-step-execution-plan.md`
- `phase-0-scope-note.md`
- `phase-1-code-path-map.md`

归档原因：

- 这几份文档带有明显的 `prefill-first` / `non-causal KV forwarding` 阶段性主线
- 作为历史工程记录仍有价值，但容易干扰当前以 `SF-MLA` 为核心的方法论文主线

### 3. 旧的 ICCAD / KV forwarding 投稿叙事

- `tt-metal_kv_forwarding_iccad_experiment_plan.md`
- `tt-metal_kv_forwarding_design_space_autotuner.md`

归档原因：

- 这两份文档聚焦早期 `prefill non-causal KV forwarding` 论文包
- 其中的 profiling、design-space、autotuner 想法仍可复用，但不再作为当前论文的直接标题级叙事

## 当前处理原则

对已归档文档统一采用以下规则：

1. 在文档顶部增加“已归档”提示。
2. 不再继续大规模改写其正文。
3. 若其中某段内容仍有技术价值，应复制或吸收到当前活跃文档，而不是继续在归档文档里演进。

## 当前主参考入口

请优先参考：

- `current-docs.md`
- `sf-mla-paper-positioning.md`
