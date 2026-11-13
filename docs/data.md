# 记录口径

保存智能体在请求确认、直接办理和转人工之间的决策记录，核对曝光概率并汇总实际收益。

decisions 按 request_id、step 表示顺序；user_id 是重复观测簇。available、propensity、action 是决策时事实，reward_known_at 是结算到达时钟。candidate_policies 保存每条 decision_id 对应的概率映射。mode 为 single_step 或 trajectory，cutoff 限定可知结果。

数据字段采用稳定的英文标识符。编号是业务身份，不由输入数组中的位置决定。未到达的结果与已经观察到的否定结果是不同事实。货币、时间和统计单位应按字段约定解释，不在读取时默默变换。

当前命令的输出只解释当前公开计算，不承诺示例覆盖全部业务情形。批量重放应使用新的输出位置，避免人工查看上一次运行留下的文件。
