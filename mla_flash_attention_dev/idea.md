1. 弄明白deepseek在GPU上如何实现FlashMLA，如何利用硬件特性，和TT有什么区别
2. 弄清楚TT的MLA的2种实现方法（现有和实验性），对于实验性MLA对所有数据参数化，做DSE和寻优模型
3. 将实验性MLA从BH迁移到WH，单芯，多芯，注意两种架构的区别
4. 备注：现有tt的两种MLA实现：
实现 A（生产路径，mla1d.py + ttnn SDPA）：

将 MLA 拆解为 ~12 个独立 ttnn op（wq_kv_a → wq_b → wkv_b1 → RoPE → SDPA → wkv_b2 → wo 等），Python 串联调度
SDPA 内核是通用的 Flash Attention（sdpa_decode_program_factory.cpp），支持 MHA/GQA/MLA
Decode 时每个 head 由 1 个核独立处理全部 K chunks，无序列并行
K 从 interleaved DRAM 读取，因果模式下无 multicast，每核独立重复读
实现 B（实验性 FlashMLA，deepseek_v3_b1）：

Blackhole 专用的 MLA decode 深度优化算子
创新的 S Block 架构：将 Blackhole 的 110 核划分为 8 个 S Block（每个 8 核），每个绑定一个 DRAM bank
每个 Q head 跨 8 个核并行处理不同的 K chunk 范围（序列并行）
K 通过 ND Sharded DRAM 按 chunk 分布到 8 个 bank，每个 S Block 的 sender 只读自己 bank 的 K，然后通过 NoC multicast 分发给 block 内 7 个 receiver
NCRISC 用 NOC Transaction ID (trid) 流水线实现 page 级 DRAM 读取/multicast 重叠
8 个 S Block 的部分结果通过 3 步 tree reduction（log₂8=3）高效归约
使用 Tiny Tile（8×32）减少 Q 的 head padding 浪费
编程模型用 UnifiedKernelDescriptor + ttnn.generic_op，单一 dispatch