这两个目录可以理解为：**训练过程的完整记录**，以及**加载已有模型后单独导出的评估报告**。它们不是两次独立训练。

### `results_grain4v3/`：训练产物

这里记录模型怎么训练出来的，核心文件包括：

- `best_model.pth`：训练得到的模型权重。本次记录显示，保存的是验证集 Macro-F1 最好时的模型，最佳轮次为第 12 轮。
- `train_log.csv`：每轮训练、验证过程的数值记录。
- `accuracy_curve.png`、`loss_curve.png`、`val_macro_f1_curve.png`：训练曲线，可用来看模型是否收敛、验证效果如何变化。
- `data_info.json`、`training_final_metrics.json`：数据规模、类别、模型参数量和训练结果等摘要。
- `final_eval/val/`、`final_eval/test/`：训练结束时一并生成的验证集和测试集评估结果，包括混淆矩阵、预测记录和错分图片。
- `summary_*.csv`：汇总训练运行结果，便于排序或比较模型。

### `eval_results_grain4v3/`：单独评估导出

这个目录没有训练权重，主要保存评估结果。评估程序会读取 `results_grain4v3/.../best_model.pth`，在验证集或测试集上预测，然后导出：

- `val/`、`test/`：分类报告、混淆矩阵、预测表、错分样本等详细评估材料。
- `summary_*.csv`：各模型或运行的指标汇总表。
- `paper_outputs/`：汇总好的报告和图表，适合整理实验材料时使用。

我检查到它与训练目录里的评估指标相同：测试集准确率约 **99.80%**、Macro-F1 约 **99.63%**，错分 **59 张**；验证集准确率约 **99.81%**、Macro-F1 约 **99.66%**，错分 **9 张**。所以 eval 目录更像是对已有 checkpoint 的独立评估和整理导出，**不是新的训练结果**。

### 你可以怎么用

- 想了解模型训练过程、查看曲线或拿模型做预测：看 `results_grain4v3/`，其中 `best_model.pth` 是关键文件。
- 想分析各类别表现、查看错分图片或整理报告：看 `eval_results_grain4v3/`。
- 写实验报告时，训练曲线和训练设置取自 `results`；评估表格、混淆矩阵和错分分析可取自 `eval_results`。记得注明这是历史运行产物，不能当成本次新运行的结果。

**有一个评估方法上的注意点：**`eval_results_grain4v3/grain4v3/paper_outputs/best_model_selection_test.json` 写明模型选择依据包含测试集 Macro-F1。测试集如果参与挑选模型或 checkpoint，就不再是完全独立的最终评估集。因此这份测试分数适合用于了解这次历史运行，但若要形成严谨的最终结论，应按验证集选模型，再在测试集上评估一次。