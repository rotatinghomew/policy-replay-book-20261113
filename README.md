# policy_replay_book

保存智能体在请求确认、直接办理和转人工之间的决策记录，核对曝光概率并汇总实际收益，
并为候选澄清策略提供离线决策支持。

当前公开命令：

- `inspect`：核对数据并查看已记录的观测，只做事实汇总，不对观测结果作超出数据支持的推断。
- `evaluate`：对候选策略做离线策略评估，回答“相对当前日志策略是否值得进入小流量”，
  不代替用户执行任何操作。方法与取舍见 `docs/methodology.md`。

输入和输出使用 UTF-8 JSON。示例数据均为本地虚构记录，不连接终端、供应方或个人账户。

## 安装与运行

使用 Python 3.11.2，依赖精确版本在 `requirements.txt`。依赖可预先装好；运行和测试不执行下载。

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall -q policy_replay_book tools
python3 -m unittest discover -s tests -v
python3 -m policy_replay_book inspect --input fixtures/study.json --output /tmp/policy-factual.json
python3 -m policy_replay_book evaluate --input fixtures/study.json --output /tmp/policy-eval.json
```

隔离模式（`python3 -I`，cwd 不入 `sys.path`）下使用根目录封装入口，输入仍以路径参数传入，
解释器不会从数据目录加载模块：

```bash
python3 -I run_evaluate.py --input fixtures/study.json --output /tmp/policy-eval.json
python3 -I tools/run_simulations.py --data-dir /tmp/prb-sim-data --output-dir /tmp/prb-sim-out
```

`evaluate` 的结果包含 `estimand`、`candidates`、`support`、`uncertainty`、
`recommendation` 和 `provenance`。`mode` 为 `single_step` 时按独立决策识别单步价值；
为 `trajectory` 时按整段轨迹的联合似然比识别总收益，不把多步输入降格为单步结果。
观察截止后才结算的结果按未结算处理、不置零；候选落在日志零概率动作上或有效支持不足时
拒绝给上线结论。

正常运行会写出完整结果文件，标准输出给出保存位置。格式冲突、数据边界错误或缺失关键字段会以非零状态退出；不要将失败输出当作空业务结果。复用临时目录前应清理同名输出与数据库，以免混淆不同运行。

## 数据与目录

`policy_replay_book/decisions.py` 提供当前观测计算；`policy_replay_book/offpolicy.py`
实现重权重、交叉拟合收益模型与支持度检查；`policy_replay_book/evaluate.py` 编排评估与
建议口径；`policy_replay_book/io.py` 负责严格 JSON 读取和原子输出；`tools/run_simulations.py`
用已知生成概率的本地模拟核验方法；`fixtures/` 是可重放的演练记录；`tests/` 验证领域边界、
输入输出行为和评估不变量。

decisions 按 request_id、step 表示顺序；user_id 是重复观测簇。available、propensity、action 是决策时事实，reward_known_at 是结算到达时钟。candidate_policies 保存每条 decision_id 对应的概率映射。mode 为 single_step 或 trajectory，cutoff 限定可知结果。

演练规模可在普通 CPU 上完成，运行说明不要求外部服务或真实设备。输出中的未定义值使用 JSON null，不用零替代。涉及版本的输入应保留原文件，方便复查当时的事实。每次评估把奖励口径
（reward_version）与候选版本指纹写入 provenance，更换候选文件不会改写已有结论。
