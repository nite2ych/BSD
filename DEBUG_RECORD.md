# V853 GC2053 摄像头调试记录

## 目标

在 V853 开发板上通过 MPP/VI 管线，使用 GC2053 MIPI 摄像头实现实时抓流（NV21/NV21M 格式）。

## 硬件/软件环境

- **板子**: V853 (Cortex-A7 ARMv7, sun8iw21), TinaLinux (Linux 4.9.191)
- **Sensor**: GC2053 MIPI (1920x1088 原生分辨率)
- **内核模块**: vin_v4l2, gc2053_mipi, vin_io, videobuf2_dma_contig
- **设备节点**: /dev/video0 (vin_video0), /dev/v4l-subdev0 (gc2053_mipi), /dev/media0
- **工具链**: arm-openwrt-linux-muslgnueabi-gcc 6.4.1 (hard-float, Cortex-A7)
- **SDK 路径**: `/home/ubuntu/tina-v853-100ask/external/eyesee-mpp`

---

## 尝试过的路线

### 路线 1: MPP API 方式 (AW_MPI_VI_*/AW_MPI_ISP_*)

**流程**:
```
AW_MPI_SYS_SetConf → AW_MPI_SYS_Init → AW_MPI_VI_Init
→ AW_MPI_VI_CreateVipp → AW_MPI_VI_SetVippAttr
→ AW_MPI_VI_GetIspDev → AW_MPI_ISP_Init → AW_MPI_ISP_Run
→ AW_MPI_VI_EnableVipp → AW_MPI_VI_CreateVirChn → AW_MPI_VI_EnableVirChn
→ AW_MPI_VI_GetFrame
```

**结果**: 阶段性推进但最终失败

#### 问题 1: SYS_Init 返回 0xA0008010 (ERR_SYS_NOTREADY)
- **根因**: 必须先调用 `AW_MPI_SYS_SetConf()` 配 nAlignWidth=32
- **解决**: 在 SYS_Init 前加 SetConf

#### 问题 2: 编译错误 — enum mipi_pix_num 未定义
- **根因**: SDK 头文件 `sunxi_camera_v2.h` 缺 mipi_pix_num 枚举
- **解决**: 在两个位置的头文件中补上枚举定义

#### 问题 3: AW_MPI_ISP_Run 崩溃 (SIGSEGV)
- **根因**: ISP_Run 内部调用 `isp_init()` → `media_open("/dev/media0")`，ISP 初始化路径崩溃
- **发现**: ISP 通过 `/mnt/extsd/ConfigIspRun` 文件控制；内容为 "0" 时跳过硬件 ISP 初始化
- **解决**: 在 camera_init 中加 `AW_MPI_ISP_Init()` 调用（该函数读取 ConfigIspRun 决定是否启用 ISP）
- **新问题**: /mnt/extsd 分区满（8151/8151 blocks），删掉 core dump 文件后解决

#### 问题 4: AW_MPI_VI_EnableVipp 崩溃
- **根因**: ISP 禁用时，CapThread 中 `gpVIDevManager->media->video_dev[0]` 为 NULL，`video_to_isp_id(NULL)` 导致空指针
- **尝试**: 启用 ISP（删掉 ConfigIspRun）→ ISP_Run 再次崩溃
- **根本原因**: MPP 的 VI 库通过 `isp_md_open` 打开 /dev/media0，而 ISP 库通过 `media_open` 再次打开同一设备，两边冲突
- **结论**: MPP VI/ISP 抽象层在 V853 + GC2053 上不可用

---

### 路线 2: 直接 V4L2 方式 (绕过 MPP VI/ISP)

**灵感来源**: SDK 中的 `sample_driverVipp.c`

**流程**:
```
open("/dev/media0") → MEDIA_IOC_DEVICE_INFO → MEDIA_IOC_ENUM_ENTITIES (循环)
→ 找到 "vin_video0" → readlink sysfs 获取设备路径 → open("/dev/video0")
→ VIDIOC_S_INPUT → VIDIOC_S_PARM → VIDIOC_S_FMT (NV21M)
→ VIDIOC_G_FMT (确认格式) → usleep(100ms) → VIDIOC_REQBUFS → usleep(100ms)
→ VIDIOC_QUERYBUF (循环) → mmap → VIDIOC_QBUF (循环) → VIDIOC_STREAMON
→ select + VIDIOC_DQBUF (抓帧) → VIDIOC_QBUF (放回)
```

**结果**: **成功！** 管线全程无崩溃，持续抓帧。

#### 关键细节

1. **必须先打开 /dev/media0** — 直接开 /dev/video0 会导致内核 VIN 驱动 `vin_g_volatile_ctrl` 空指针崩溃
2. **REQBUFS 前后需 usleep(100ms)** — sample 代码里有，不加可能导致后续 ioctl 超时
3. **单平面 vs 多平面 NV21**: 请求 NV21M 返回 2 planes (Y 和 UV 分开)，请求 NV21 返回 1 plane (Y+UV 在同一个 buffer)
4. **硬浮点**: musl 工具链默认 hard-float (VFPv4)，不能用 `-mfloat-abi=soft`
5. **G_PARM 返回 capturemode=0x2** — 不是请求的 0 (V4L2_MODE_VIDEO)，可能是 VIN 驱动内部模式

#### 编译参数
```bash
arm-openwrt-linux-muslgnueabi-gcc \
  -O2 -mfloat-abi=hard -mcpu=cortex-a7 -Wall -static \
  capture_v4l2.c -o capture_v4l2
```

---

## 当前状态

### 已解决
- [x] 抓流管线端到端跑通（3 帧/次，无崩溃）
- [x] 确认 sensor 输出真实数据（非噪声，有空间结构）
- [x] NV21M 格式 640x360 @ 20fps 正常工作

### 待解决
- [ ] **图像很暗**: Y 值 ~0x1e（30/255），所有帧几乎完全黑
- [ ] **编译服务器 (192.168.1.100) 宕机**: SSH 连接到 key exchange 阶段被远端关闭，无法交叉编译
- [ ] **板子需重启**: 当前有 2 个 D 状态进程（check_ctls, capture_v4l2）

### 新发现 (2026-05-21)

#### 发现 1: 真正的控制项在 /dev/video0 上，不是 subdev0
`check_ctls` 二进制扫描发现 `/dev/video0` 上有 **59 个 V4L2 控制项**，而 `/dev/v4l-subdev0`（gc2053_mipi）只有 **5 个**。

关键控制项对比：
| 控制项 | /dev/video0 | /dev/v4l-subdev0 |
|--------|-------------|------------------|
| Exposure (0x00980911) | min=1 max=1048576 | min=1 max=1048576 |
| Gain (0x00980913) | min=16 max=96000 | min=1600 max=409600 |
| Auto Exposure (0x009a0901) | **存在** (MENU型) | **不存在** |
| Gain, Automatic (0x00980912) | **存在** (BOOL型) | **不存在** |

#### 发现 2: 自动曝光/自动增益默认开启
在 `/dev/video0` 上：
- `Auto Exposure (0x009a0901)` = **0 (Auto Mode)** — 自动曝光开启
- `Gain, Automatic (0x00980912)` = **1 (ON)** — 自动增益开启

**这就是图像暗的根本原因**：自动模式忽略了手动设置的曝光/增益值，使用默认（最低）增益。

#### 发现 3: 直接访问 /dev/video0 导致 I2C 错误
直接打开 `/dev/video0` 并设置控制项（不走 media controller 初始化）会导致：
```
[VIN_DEV_I2C_ERR] sensor_write error! sensor is not used!
```
内核 VIN 驱动在未通过 media controller 初始化时无法访问 sensor I2C，导致进程陷入 D 状态。

#### 发现 4: Sensor I2C 信息
- **I2C 总线**: twi2（Linux i2c-2）
- **I2C 地址**: 0x37（7-bit）= 0x6e（8-bit，设备树中存储的值）
- **I2C 访问**: 被内核驱动独占（`i2cdump -f` 全返回 XX），VIN 挂起时 I2C 总线被门控
- 设备树属性: `sensor0_twi_addr=0x6e`, `sensor0_twi_cci_id=1`, `sensor0_isp_used=1`

#### 发现 5: 正确的控制流程（已写好代码）
```
open(/dev/media0) → MEDIA_IOC_* → 找到 vin_video0
→ open(/dev/video0)
→ S_CTRL Auto Exposure = 1 (Manual Mode)      ← 必须先设
→ S_CTRL Gain, Automatic = 0 (OFF)             ← 必须先设
→ S_CTRL Exposure = 50000+ (高值)
→ S_CTRL Gain = 50000+ (高值)
→ S_FMT / REQBUFS / STREAMON / 抓帧
```

此流程已写入 `capture_bright.c`，**等待编译服务器恢复后编译**。

### 下一步方向
1. 断电重启板子
2. 等待编译服务器恢复，编译 `capture_bright.c`
3. 在 /dev/video0 上（走完 media 初始化后）设置手动曝光/增益模式
4.  如果自动曝光/增益能成功关闭，图像亮度应有显著改善

---

## 关键代码文件

| 文件 | 用途 | 状态 |
|------|------|------|
| `camera_mpp.c` | MPP API 方式（原始方案） | 崩溃（EnableVipp） |
| `capture_v4l2.c` | 直接 V4L2 方式 | **工作** |
| `read_driver_vipp.py` | 读取 sample_driverVipp.c | 完成 |
| `compile_and_test_v4l2.py` | 编译+测试 V4L2 | **工作**（640x360） |
| `test_v4l2_detail.py` | V4L2 详细数据统计分析 | 完成 |
| `test_sensor_controls.py` | Sensor 控制测试 | 部分工作 |
| `test_capture_bright.py` | 高曝光/增益抓流测试 | 完成（但亮度没变） |
| `check_vin_cap_controls.py` | 枚举所有子设备控制项 | 编译失败（内核API冲突） |
| `probe_controls.c` | 探测两子设备控制项+抓帧对比 | 未编译（服务器宕机） |
| `capture_bright.c` | **正确流程**：media初始化→关自动曝光/增益→设高值→抓帧 | **待编译测试** |

## 重要教训

1. **永远不要直接打开 /dev/video0**，必须先通过 /dev/media0 初始化管线
2. **sample_driverVipp 是正确参考** — 它直接用 kernel V4L2/media controller 接口，不用 MPP VI/ISP
3. **ISP 不是可选的** — 它负责配置 sensor 的模拟增益和曝光。绕过 ISP 需要自己通过子设备控制 sensor
4. **MPP 的 VI 和 ISP 互相冲突** — 两边都试图打开 /dev/media0，导致各种崩溃
