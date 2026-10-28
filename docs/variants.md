# HMS Pro / Elite 版本选择

本仓库保留两个独立实现，统一入口只选择运行哪个版本：

| 版本 | 源码位置 | 原生服务入口 | 配置 | 原有评测 |
| --- | --- | --- | --- | --- |
| `pro`（默认） | 原 HMS 目录，保持原路径 | `hms_api.main` | 根目录 `.env` / `HMS_API_*` | LongMemEval |
| `elite` | `variants/elite/` | `hms_elite.server.app` | Elite 原生 JSON / 环境变量 | EgoLife |

Pro 面向准确率，Elite 面向速度；这是版本定位，不代表本次整合重新测得的性能结论。

## 启动与切换

### Docker Compose

先配置原生环境文件，修改模板中的数据库、模型和密钥：

```bash
cp .env.example .env
bash scripts/generate_keys.sh
cp variants/elite/.env.example variants/elite/.env
```

Pro 的配置说明见[根 README](../README.zh-CN.md)，Elite 的模型和存储配置见
[Elite README](../variants/elite/README.zh.md) 及其 `deploy/config.dev.json`。
入口不复制配置、替换模型或给两种实现添加统一默认值。

```bash
python hmsctl.py --variant pro compose up -d --build
python hmsctl.py --variant elite compose up -d --build

# 使用各版本原生 Compose 文件、项目名、卷、端口和健康检查
python hmsctl.py --variant elite compose ps
python hmsctl.py --variant elite compose logs --tail 100
python hmsctl.py --variant elite compose down
```

切换参数只决定本次命令的目标，不会停止已经启动的另一版本。需要仅运行一个版本时，
先对旧版本执行 `compose down`，再启动另一个；不加 `-v`，保留原有数据卷。
两个版本可以使用原有不同端口并行运行，也可以由操作者显式配置服务端口。

### Python 服务

保留各版本原有依赖和版本约束，建议使用独立环境。Pro 按原文档安装；Elite 可安装：

```bash
python -m venv .venv/elite
.venv/elite/bin/python -m pip install -e './variants/elite[server]'
python hmsctl.py --variant elite --python .venv/elite/bin/python serve --port 8000
```

在已安装对应依赖的环境中：

```bash
python hmsctl.py --variant pro serve
python hmsctl.py --variant elite serve

# 也可以把选择放在环境变量中；显式 --variant 优先
HMS_VARIANT=elite python hmsctl.py serve

python hmsctl.py --list
python hmsctl.py --variant elite --dry-run serve --config deploy/config.dev.json
```

所有入口选项必须放在动作（`serve` / `compose` / `benchmark` / `test` / `run`）之前。
动作后面的参数逐项转交给原命令，包括原命令自己的 `--help`；入口不解析或改写它们。
`--python` 只指定 Python 命令的解释器，Shell 脚本继续沿用各自原有的环境选择方式。
路径参数（例如 Elite 的 `--config`）相对于该版本根目录解析：Pro 为仓库根目录，
Elite 为 `variants/elite/`。`--python` 的相对路径则相对于调用入口时的目录解析。
从其他目录调用时，使用 `hmsctl.py` 的绝对路径。

## 保持原始逻辑与评测口径

- Pro 的核心源码、启动脚本、依赖声明、锁文件、配置模板和评测代码保持原样。
  根 README 添加 News / 使用入口，根 `.gitignore` 仅放行 Elite 已提交的证据文档。
- Elite 上游跟踪的文件全部按字节导入，包括源码、提示词、配置、部署、测试、数据和评测脚本；
  不导入嵌套 `.git`、未跟踪的本地数据或凭据。Elite 的包元数据保留其 Apache-2.0 声明。
- `hmsctl.py` 不导入后端。它根据注册表选择工作目录和源码搜索路径，然后用进程替换
  执行原生入口，保留标准输入输出、退出码和信号。其他环境变量原样传递；不自动安装依赖。
- API、认证、数据库和结果格式保留原生约定。这是启动选择器；不提供跨版本 API 转换、
  数据迁移或同一进程中的热切换，也不会将 Pro 的评测适配器用于 Elite。

用原生评测入口运行，原参数、提示词、随机性、评测集和评分方式不变：

```bash
# Pro：仍执行原来的 .aaaSCRIPT/run_benchmark.sh
# 先按 LongMemEval 文档配置数据库、模型及 .env.longmemeval
HMS_ENV_FILE=.env.longmemeval python hmsctl.py --variant pro benchmark

# Elite：仍执行原来的 benchmarks/egolife/run_egolife.py
# 先按原文档安装该基准所需的 mem0ai 等额外依赖；此处不替换原默认参数
python hmsctl.py --variant elite benchmark --help

# 其他原生 Python 评测脚本也可以直接运行，参数保持原样
python hmsctl.py --variant elite run benchmarks/egolife/run_real.py --help

# 原来的运行方式继续可用
bash .aaaSCRIPT/run_benchmark.sh
(cd variants/elite && python benchmarks/egolife/run_egolife.py --help)
```

`variants/sources.json` 记录来源仓库、原始提交以及原文件 SHA-256。
Pro 基线仅排除允许编辑的两个根 README 和根 `.gitignore`；Elite 不排除任何上游跟踪文件。
可以在不安装任何后端依赖的情况下核对：

```bash
python scripts/verify_variants.py
python -m unittest discover -s tests -p 'test_variant_*.py' -v

# 原生测试在各自工作目录、环境及 pytest 配置下执行
python hmsctl.py --variant elite test
python hmsctl.py --variant pro test
```

文件一致性和离线回归验证代码未被改写，不等于重新取得线上评测分数。
比较最终分数时仍需保持原数据库状态、依赖版本、模型服务、数据集、配置和随机性条件；
新提交会使评测产物中的 Git 提交等来源元数据变化。原有恢复运行的一致性检查保持生效。
Elite 原有个别辅助 Shell 脚本包含作者机器的绝对路径，按原样保留；可直接运行对应的
Python 基准入口并显式传入路径。

## 后续接入 PMM 等版本

将实现及其原生依赖、配置、测试、评测完整放入 `variants/pmm/`，然后只在
[`variants/registry.json`](../variants/registry.json) 新增一个条目，例如：

```json
"pmm": {
  "name": "PMM",
  "root": "variants/pmm",
  "python_paths": ["."],
  "commands": {
    "serve": ["{python}", "-m", "pmm.server"],
    "compose": ["docker", "compose"],
    "test": ["{python}", "-m", "pytest"],
    "run": ["{python}"]
  }
}
```

模块名和命令按 PMM 的真实入口填写；也可注册该版本自己的评测命令。
该示例没有注册当前尚未接入的 PMM，也不需要改写选择器中的条件分支。
`{python}` 是唯一的命令占位符，仅在注册的命令中替换；用户参数原样传递。
若要冻结新实现的来源，可按 `sources.json` 的同一格式记录其提交及文件哈希。

## English

Use `python hmsctl.py --variant pro|elite serve` to select the native Python
server, or `python hmsctl.py --variant pro|elite compose up -d --build` to run
its existing Compose stack. Pro is the default; `HMS_VARIANT` is an alternative.
Install/configure each backend using its original documentation and use
`--python /path/to/venv/bin/python` before the action for Python commands.

Backend arguments are forwarded verbatim. Relative backend paths resolve from
that backend's root, not from the caller. Only the working directory and source
import paths are selected; the launcher does not adapt APIs, storage, prompts,
defaults, or scoring. Existing processes are not stopped automatically.

Run `python scripts/verify_variants.py` to verify imported source bytes. New
backends need only their own directory and an entry in `variants/registry.json`.
Source parity and offline regression checks do not establish a new live score;
retain the original dependencies, data, providers, settings, and random state
when reproducing evaluations.
