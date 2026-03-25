# tt-metal 单播与多播实现说明

本文整理 `tt-metal` 里一个常见的片上通信模式：

- 先由一个 `sender core` 从 DRAM 读取数据到本地 L1
- 再把这份数据通过 NoC 分发给其他核心
- 分发方式可以是 **单播**，也可以是 **多播**

本文写成一个**通用版本**，适用于一对一、一对多、链式转发、矩形区域广播等场景；同时用 `2x2` 四核场景作为最小例子，帮助把抽象概念落到具体实现上。

如果你想看更底层的 NoC 地址格式、multicast 编码和 SDPA 里的真实用法，可以配合下面两篇文档一起看：

- `docs/NoC_片上网络与多播详解.md`
- `docs/NoC_与多播代码实现详解.md`

## 1. 适用问题

这类实现一般解决的是下面这种问题：

1. 某个核心负责从 DRAM 读取一份数据。
2. 这份数据接下来要被别的核心消费。
3. 不希望每个核心都重复从 DRAM 独立读取。

在 `tt-metal` 里，常见的两种分发方式是：

- **单播**
  - sender 分别给每个 receiver 发一份
- **多播**
  - sender 一次把同一份数据发给一个矩形目标区域

从抽象上讲，可以把问题写成：

```text
DRAM -> sender local L1 -> receivers local L1
```

这和你具体在做什么算子无关。  
无论后面是做 attention、reduce、matmul 还是别的东西，只要模式是“一个核先取数据，再分发给别的核”，都可以套这个框架。

## 2. 基本角色

建议先把角色区分清楚：

- `sender`
  - 负责从 DRAM 读取数据
  - 负责向外转发
- `receiver`
  - 不直接从 DRAM 读取这份共享数据
  - 只等待 sender 把数据写到自己 L1

常见拓扑有三种：

### 2.1 星型单播

```text
sender -> receiver0
sender -> receiver1
sender -> receiver2
...
```

优点是最直观，适合 bring-up。  
缺点是目标越多，sender 发的次数越多。

### 2.2 链式单播

```text
sender -> receiver0 -> receiver1 -> receiver2
```

这更接近 SDPA 里的 chain forwarding。  
优点是 sender 压力小一些。  
缺点是同步复杂，而且总延迟和链长相关。

### 2.3 多播

```text
sender => 一个矩形区域内的多个 receiver
```

优点是 sender 一次发出去，适合“同一份数据发给一组核”。  
缺点是目标必须能表示成**矩形区域**，不能是任意散点集合。

## 3. 核心 API

实现这个模式常用到的 API 很少，主要就是下面这些：

| 类别 | API | 作用 |
| --- | --- | --- |
| DRAM/L1 读 | `noc_async_read` | 从 DRAM 或远端 L1 读到本地 L1 |
| L1/L1 单播写 | `noc_async_write` | 把本地 L1 数据写到某个目标核的 L1 |
| L1/L1 多播写 | `noc_async_write_multicast` | 把本地 L1 数据写到一个矩形目标区域 |
| 含自身的多播写 | `noc_async_write_multicast_loopback_src` | 当 sender 自己也需要通过 multicast 收到数据时使用 |
| 单播地址 | `get_noc_addr(x, y, addr)` | 构造某个目标核的 NoC 地址 |
| 多播地址 | `get_noc_multicast_addr(x0, y0, x1, y1, addr)` | 构造矩形目标区域的 NoC 地址 |
| 本地等待 | `noc_semaphore_wait` | 等待本地 semaphore 变成指定值 |
| 远端加一 | `noc_semaphore_inc` | 对远端 semaphore 做原子加 |
| 远端单播写信号量 | `noc_semaphore_set_remote` | 通知某个 receiver 数据已可用 |
| 远端多播写信号量 | `noc_semaphore_set_multicast` | 一次通知多个 receiver 数据已可用 |

## 4. 通用实现框架

不管是单播还是多播，最小程序结构都很类似。

### 4.1 Host 侧需要做什么

通常需要准备下面几样东西：

1. 一个 `Program`
2. 一组参与通信的核心
3. 一个所有相关核心都创建的 CB 或一致的 L1 buffer
4. 若干 semaphore
5. 一个 dataflow kernel
6. 每个核心各自的 runtime args

最小流程可以写成：

```text
Host
  -> CreateProgram
  -> CreateCircularBuffer / L1 buffer
  -> CreateSemaphore
  -> CreateKernel
  -> SetRuntimeArgs(per core)
  -> EnqueueProgram
```

### 4.2 Device 侧需要做什么

在 device kernel 里，一般分两条路径：

```text
if sender:
  从 DRAM 读到本地 L1
  用单播或多播发出去

if receiver:
  先等待 sender
  等 sender 写完后继续消费本地 L1 数据
```

## 5. 先统一几个关键前提

### 5.1 远端接收地址必须一致

单播和多播都依赖一个非常重要的前提：

- sender 发出去时，通常直接拿自己的 `cb` 起始地址当作远端 L1 目标地址
- 这要求所有目标核上这个 CB 的布局一致

最稳妥的做法是：

- 所有相关核心都创建同一个 `cb_id`
- 使用同样的 `CircularBufferConfig`
- 所有参与通信的核心跑同一个 kernel，或者至少保证 L1 布局一致

### 5.2 坐标必须是 NoC 可用坐标

Host 侧不要直接把逻辑坐标当作 NoC 坐标传进去。  
应该先做一次转换，例如：

```cpp
CoreCoord logical_core(x, y);
CoreCoord physical_core = device->worker_core_from_logical_core(logical_core);
```

然后再把 `physical_core.x`、`physical_core.y` 填进 runtime args。

### 5.3 第一个 demo 建议只搬很小的数据

建议一开始只搬：

- `1` 个 tile
- 或者一段明确对齐、总大小不超过 `8KB` 的小数据

这样能最大限度减少下面几类问题的干扰：

- 地址错
- 对齐错
- barrier 放错
- semaphore 协议写错
- `num_dests` 写错

## 6. 通用同步协议

最常见、也最容易调试的一种同步方式，是用两个 semaphore：

- `ready_sem`
  - receiver 用它通知 sender：“我已经准备好接收”
- `valid_sem`
  - sender 用它通知 receiver：“数据已经写完了”

### 6.1 通用版协议

假设有 `num_receivers` 个 receiver。

receiver 侧：

1. 把本地 `valid_sem` 置成 `0`
2. `noc_semaphore_inc(sender.ready_sem, 1)`
3. 等待本地 `valid_sem == 1`
4. 收到后再继续

sender 侧：

1. 等待本地 `ready_sem == num_receivers`
2. 把本地 `ready_sem` 清零
3. 从 DRAM 读数据到本地 L1
4. 做单播或多播
5. 把所有 receiver 的 `valid_sem` 置成 `1`

这个协议的优点是：

- 通俗
- 单播和多播都能复用
- 不需要先做复杂的分层同步

## 7. 从 DRAM 读到 sender 本地 L1

这一步通常最简单，核心就是：

1. 在本地 CB 预留空间
2. 拿到本地写地址
3. 用 `noc_async_read(...)` 把 DRAM 数据搬进来

一个典型模式如下：

```c
cb_reserve_back(cb_id, 1);
uint32_t dst_l1 = get_write_ptr(cb_id);

uint64_t src_noc_addr = get_noc_addr(tile_id, src_accessor);
noc_async_read(src_noc_addr, dst_l1, page_bytes);
noc_async_read_barrier();
```

如果源数据已经是 interleaved tensor，那么可以通过 `TensorAccessor` 来生成 `src_noc_addr`。

## 8. Host 端通用骨架

下面是一份偏伪代码的 host 结构。  
它不是某个具体 op 的完整实现，而是一个“最小 bring-up”模板。

```cpp
Program program = CreateProgram();

CoreRange core_range(core_start, core_end);

constexpr uint32_t cb_id = tt::CBIndex::c_0;
constexpr uint32_t page_bytes = 2048;  // 例如 1 个 BF16 tile

CircularBufferConfig cb_cfg(page_bytes, {{cb_id, tt::DataFormat::Float16_b}})
    .set_page_size(cb_id, page_bytes);
CreateCircularBuffer(program, core_range, cb_cfg);

uint32_t ready_sem_id = CreateSemaphore(program, core_range, 0);
uint32_t valid_sem_id = CreateSemaphore(program, core_range, 0);

std::vector<uint32_t> compile_args = {
    ready_sem_id,
    valid_sem_id,
    // 这里继续追加 TensorAccessor 的 compile-time args
};

KernelHandle kernel_id = CreateKernel(
    program,
    "path/to/unicast_multicast_demo.cpp",
    core_range,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_0,
        .compile_args = compile_args,
    });

for (auto logical_core : all_cores) {
    CoreCoord physical_core = device->worker_core_from_logical_core(logical_core);

    std::vector<uint32_t> rt_args = {
        src_dram_addr,
        page_bytes,
        is_sender(logical_core),
        use_mcast,
        num_receivers,
        sender_physical.x,
        sender_physical.y,
        rect_start.x,
        rect_start.y,
        rect_end.x,
        rect_end.y,
        // 然后再追加 receiver 坐标列表
    };

    SetRuntimeArgs(program, kernel_id, logical_core, rt_args);
}
```

这里要点有两个：

1. semaphore id 一般作为 compile-time arg 传给 kernel，再由 `get_semaphore(...)` 获取地址。
2. sender/receiver 的角色、坐标、是否走多播，通常更适合作为 runtime arg。

## 9. Device 侧通用骨架

下面是一份通用伪代码，表达的是“读 DRAM -> 发给别的核”的结构，而不是某个具体可直接编译的最终内核：

```c
void kernel_main() {
    // 读 runtime args
    // 读 compile-time semaphore ids
    // 创建 TensorAccessor
    // 申请本地 CB 空间

    if (!is_sender) {
        set(valid_sem, 0);
        inc(sender.ready_sem, 1);
        wait(valid_sem == 1);
        // 这里继续消费本地 CB 数据
        return;
    }

    wait(ready_sem == num_receivers);
    set(ready_sem, 0);

    // DRAM -> local L1
    noc_async_read(...);
    noc_async_read_barrier();

    if (!use_mcast) {
        for each receiver:
            noc_async_write(local_l1, receiver_l1, size);
        noc_async_write_barrier();

        for each receiver:
            noc_semaphore_set_remote(valid_sem, receiver_valid_sem_addr);
    } else {
        noc_async_write_multicast(local_l1, mcast_addr, size, num_dests, linked=true);
        noc_semaphore_set_multicast(valid_sem, mcast_sem_addr, num_dests);
    }
}
```

这份骨架最重要的不是语法细节，而是结构：

1. sender 和 receiver 的职责分清楚
2. receiver 先发 ready，再等 valid
3. sender 在 DRAM 读完之后，只切换“单播”还是“多播”的发送方式

## 10. 单播方法

### 10.1 通用思路

最简单的单播实现是：

```text
sender -> receiver0
sender -> receiver1
sender -> receiver2
...
```

也就是 sender 做多次独立的 `noc_async_write(...)`。

### 10.2 核心代码模式

```c
for each receiver {
    uint64_t dst_noc_addr = get_noc_addr(receiver_x, receiver_y, dst_l1);
    noc_async_write(dst_l1, dst_noc_addr, page_bytes);
}
noc_async_write_barrier();

for each receiver {
    uint64_t valid_noc_addr = get_noc_addr(receiver_x, receiver_y, valid_sem_addr);
    noc_semaphore_set_remote(valid_sem_addr, valid_noc_addr);
}
```

### 10.3 什么时候更适合用单播

单播更适合：

- 第一个 bring-up demo
- 目标数很少
- 目标不是矩形，没法一次多播覆盖
- 你想先验证地址和同步逻辑是否正确

### 10.4 链式单播是另一种选择

如果你想做的是类似 SDPA chain forwarding 的结构，也可以写成：

```text
sender -> mid0 -> mid1 -> mid2
```

但这不再是“最简单实现”，因为：

- 每一跳都要做自己的 ready/valid 协议
- 中间核心既是 receiver，又是 sender
- 更容易出错

所以第一次做 demo，建议先用星型单播。

## 11. 多播方法

### 11.1 通用思路

多播适合“同一份数据要发给一批核，而且这批核能表示成一个矩形区域”的场景。

核心代码模式是：

```c
uint64_t mcast_data_addr =
    get_noc_multicast_addr(x_start, y_start, x_end, y_end, dst_l1);

uint64_t mcast_sem_addr =
    get_noc_multicast_addr(x_start, y_start, x_end, y_end, valid_sem_addr);

noc_async_write_multicast(
    dst_l1,
    mcast_data_addr,
    page_bytes,
    num_dests,
    true);

noc_semaphore_set_multicast(
    valid_sem_addr,
    mcast_sem_addr,
    num_dests);
```

### 11.2 multicast 的限制

最重要的限制只有一个：

- **目标必须是矩形**

也就是说，如果你的 receiver 是一个散点集合，而不是矩形，就不能用一次 multicast 精确覆盖它们。

这种时候有两个选择：

1. 拆成多个矩形，多次 multicast
2. 直接退回单播

### 11.3 non-loopback 和 loopback 的区别

`tt-metal` 里需要明确区分两种 multicast：

- `noc_async_write_multicast(...)`
  - non-loopback
  - sender 即使落在目标矩形内，也**不会写回自己**
  - `num_dests` **不包含 self**
- `noc_async_write_multicast_loopback_src(...)`
  - loopback
  - sender 也会收到 multicast 数据
  - `num_dests` **包含 self**

同样的语义也适用于：

- `noc_semaphore_set_multicast(...)`
- `noc_semaphore_set_multicast_loopback_src(...)`

### 11.4 为什么通常要配 `linked=true`

multicast 常见的安全写法是：

1. 先发数据：`noc_async_write_multicast(..., linked=true)`
2. 紧接着发信号量：`noc_semaphore_set_multicast(...)`

这样可以保证：

- 数据先发
- 紧跟着 signal 发
- 中间不插入别的 NoC 事务

对 receiver 来说，这样最不容易出现“先看见信号量，但数据还没完全到”的竞态。

## 12. 以 2x2 四核为例

下面用一个最小 `2x2` 例子，把上面的通用框架具体化。

### 12.1 布局

```text
(0,0)  sender     (1,0)  receiver
(0,1)  receiver   (1,1)  receiver
```

这个例子的好处是：

- sender 放在角上，逻辑最清楚
- 单播目标就是另外 3 个核
- 整个 `2x2` 又天然构成一个 multicast 矩形

### 12.2 单播版

这个例子下，单播最直接的写法就是 sender 对另外 3 个核各做一次写：

```c
noc_async_write(dst_l1, get_noc_addr(r10_x, r10_y, dst_l1), page_bytes);
noc_async_write(dst_l1, get_noc_addr(r01_x, r01_y, dst_l1), page_bytes);
noc_async_write(dst_l1, get_noc_addr(r11_x, r11_y, dst_l1), page_bytes);
noc_async_write_barrier();

noc_semaphore_set_remote(valid_sem_addr, get_noc_addr(r10_x, r10_y, valid_sem_addr));
noc_semaphore_set_remote(valid_sem_addr, get_noc_addr(r01_x, r01_y, valid_sem_addr));
noc_semaphore_set_remote(valid_sem_addr, get_noc_addr(r11_x, r11_y, valid_sem_addr));
```

这就是最简单的星型单播。

### 12.3 多播版

这个例子下可以直接把整个 `2x2` 当作 multicast 矩形：

- `rect_start = (0,0)`
- `rect_end = (1,1)`

如果 sender 用的是 **non-loopback multicast**，那么：

- sender 虽然位于矩形中
- 但不会写回自己
- 所以 `num_dests = 3`

对应代码模式是：

```c
uint64_t mcast_data_addr =
    get_noc_multicast_addr(rect_x0, rect_y0, rect_x1, rect_y1, dst_l1);

uint64_t mcast_sem_addr =
    get_noc_multicast_addr(rect_x0, rect_y0, rect_x1, rect_y1, valid_sem_addr);

noc_async_write_multicast(
    dst_l1,
    mcast_data_addr,
    page_bytes,
    3,
    true);

noc_semaphore_set_multicast(
    valid_sem_addr,
    mcast_sem_addr,
    3);
```

如果你硬要让 sender 自己也通过 multicast 收到这份数据，那么就改成 loopback 版本，同时把 `num_dests` 改成 `4`。

### 12.4 为什么 2x2 是一个很好的最小 demo

因为它同时满足：

- 可以看懂单播的全部路径
- 又可以自然地测试 multicast 的矩形约束
- 目标数正好是 `3`
- 错误定位也简单

所以它非常适合作为 first bring-up case。

## 13. 如何验证是否成功

最简单的验证方式，不是直接在 kernel 里猜 L1 内容，而是让每个核心把接收到的数据再写回各自的 DRAM 输出 buffer，然后由 host 回读比较。

建议的验证流程：

1. Host 准备一份可识别的输入数据
   - 例如全 1、递增序列、固定随机种子
2. Sender 从输入 DRAM 读到本地
3. Receiver 收到后，把本地数据写回自己的输出 DRAM
4. Host 回读所有输出
5. 比较 sender 与所有 receiver 的输出是否一致

这样做的优点是：

- 不需要在 bring-up 初期直接调试远端 L1 地址
- 很容易用 host 侧打印和比较结果

## 14. 常见坑

### 14.1 把 logical 坐标当成 NoC 坐标

这是最常见错误之一。  
应该先在 host 侧做 `worker_core_from_logical_core(...)`，再把结果传给 kernel。

### 14.2 receiver 集合不是矩形

如果 receiver 集合不能表示成矩形，就不能靠一次 multicast 覆盖。  
这时要么拆成多个矩形，要么改用单播。

### 14.3 `num_dests` 写错

一定要记住：

- non-loopback multicast：`num_dests` 不包含 self
- loopback multicast：`num_dests` 包含 self

### 14.4 `linked=true` 后中间插了别的操作

如果你用了：

```c
noc_async_write_multicast(..., linked=true);
```

那么后面应该马上接 companion 的 multicast semaphore 设置。  
中间不要插 barrier 或别的 NoC 操作。

### 14.5 所有目标核的接收地址不一致

这通常意味着：

- CB id 不一致
- CB 配置不一致
- 或者不同 kernel 的 L1 布局不同

对于最小 demo，最简单安全的做法就是让所有相关核心都创建同一个 CB。

### 14.6 第一个 demo 就做太复杂

建议按下面顺序 bring-up：

1. sender 只做 DRAM -> local L1
2. sender -> 1 个 receiver 的单播
3. sender -> 多个 receiver 的单播
4. 再切到 multicast

这样一旦挂住，定位会非常快。

## 15. 一句话总结

通用地看，`tt-metal` 里的单播和多播都可以理解成：

```text
一个 sender 先把数据从 DRAM 拉到本地 L1，再用 NoC 分发给其他核心
```

其中：

- **单播** 适合先 bring-up，也适合目标不是矩形的场景
- **多播** 适合“同一份数据发给一个矩形区域”的场景
- `2x2` 四核是最适合起手的例子，因为它既能清楚展示星型单播，也能天然满足 multicast 的矩形约束

## 16. 参考文件

- `docs/NoC_片上网络与多播详解.md`
- `docs/NoC_与多播代码实现详解.md`
- `tt-metal/tt_metal/hw/inc/api/dataflow/dataflow_api.h`
- `tt-metal/ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/reader_unary_interleaved_start_id_gen.cpp`
- `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`
