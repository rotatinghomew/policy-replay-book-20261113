# policy_replay_book

保存智能体在请求确认、直接办理和转人工之间的决策记录，核对曝光概率并汇总实际收益。

当前公开命令是 `inspect`，用于核对数据和查看已记录的观测，不对观测结果作超出数据支持的推断。输入和输出使用 UTF-8 JSON。示例数据均为本地虚构记录，不连接终端、供应方或个人账户。

## 安装与运行

使用 Python 3.11.2，依赖精确版本在 `requirements.txt`。依赖可预先装好；运行和测试不执行下载。

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall -q policy_replay_book
python3 -m unittest discover -s tests -v
python3 -m policy_replay_book inspect --input fixtures/study.json --output /tmp/policy-factual.json
```

正常运行会写出完整结果文件，标准输出给出保存位置。格式冲突、数据边界错误或缺失关键字段会以非零状态退出；不要将失败输出当作空业务结果。复用临时目录前应清理同名输出与数据库，以免混淆不同运行。

## 数据与目录

`policy_replay_book/decisions.py` 提供当前观测计算；`policy_replay_book/io.py` 负责严格 JSON 读取和原子输出；`fixtures/` 是可重放的演练记录；`tests/` 验证领域边界和输入输出行为。

decisions 按 request_id、step 表示顺序；user_id 是重复观测簇。available、propensity、action 是决策时事实，reward_known_at 是结算到达时钟。candidate_policies 保存每条 decision_id 对应的概率映射。mode 为 single_step 或 trajectory，cutoff 限定可知结果。

演练规模可在普通 CPU 上完成，运行说明不要求外部服务或真实设备。输出中的未定义值使用 JSON null，不用零替代。涉及版本的输入应保留原文件，方便复查当时的事实。
