# arXiv 论文助手 - 文件保存功能使用示例

## 场景演示

### 示例 1：保存论文总结

**用户输入**：

```
请总结论文 "Attention Is All You Need" 的核心贡献，并保存为笔记
```

**助手响应流程**：

1. 检索论文内容，生成总结
2. 调用 `save_note` 工具：
   ```python
   save_note("attention_is_all_you_need_summary",
             "# Transformer 模型总结\n\n论文 'Attention Is All You Need' 提出了...")
   ```
3. 返回结果：

   ```
   论文总结完成！

   笔记已保存至：notes/attention_is_all_you_need_summary.md
   ```

### 示例 2：保存研究计划

**用户输入**：

```
我计划研究强化学习在对话系统中的应用，请帮我保存这个研究方向
```

**助手响应**：

```python
save_note("rl_dialogue_research_plan",
          "# 强化学习在对话系统中的研究计划\n\n## 研究方向\n- 基于RL的对话策略优化\n- 奖励函数设计...")
```

**输出**：

```
研究计划已保存至：notes/rl_dialogue_research_plan.md
```

### 示例 3：列出所有笔记

**用户输入**：

```
列出我保存的所有笔记
```

**助手响应**：

```python
list_notes()
```

**输出**：

```
共有 3 个笔记文件：
  [1] attention_is_all_you_need_summary.md (1024 字节)
  [2] rl_dialogue_research_plan.md (768 字节)
  [3] math_theorem_notes.md (512 字节)
```

## 实际 CLI 对话示例

```
arXiv 论文助手 v1.0
正在扫描论文文件夹...

知识库共有 5 篇论文：
  [1] 2301.12345 本地缓存存在
  [2] 2403.11046 本地缓存存在
  ...

You: 请总结第 2 篇论文的主要方法

助手：[检索论文内容...]
这篇论文提出了基于注意力机制的新型架构，主要方法包括：
1. 多头注意力机制 [第3页]
2. 位置编码方案 [第4页]
3. 前馈网络设计 [第5页]

You: 保存这个总结

助手：正在保存笔记...
笔记已保存至：notes/2403.11046_method_summary.md

You: 列出我的笔记

助手：共有 1 个笔记文件：
  [1] 2403.11046_method_summary.md (856 字节)

You: 再保存一个研究想法：将这个方法应用到图像生成

助手：正在保存笔记...
笔记已保存至：notes/image_generation_idea.md
```

## 高级用法

### 1. 指定自定义文件名

```
用户：保存这个总结，文件名为 "transformer_innovations"
助手：笔记已保存至：notes/transformer_innovations.md
```

### 2. 自动命名（按论文ID）

```
用户：总结第一篇论文并保存
助手：自动生成文件名 "2301.12345_summary.md"
```

### 3. 批量操作

```
用户：为所有论文生成简短摘要并保存
助手：依次保存为：
- notes/2301.12345_abstract.md
- notes/2403.11046_abstract.md
- ...
```

## 文件管理

### 查看笔记内容

```bash
# 在终端中查看
cat notes/attention_is_all_you_need_summary.md

# 使用 Markdown 查看器
glow notes/context_summary.md
```

### 备份笔记

```bash
# 压缩笔记目录
tar -czf notes_backup.tar.gz notes/

# 复制到其他位置
cp -r notes/ ~/research_notes/
```

## 故障排除

### 1. 文件保存失败

**问题**：权限不足或磁盘已满
**解决**：检查目录权限和磁盘空间

### 2. 中文乱码

**问题**：文件内容显示乱码
**解决**：确保使用 UTF-8 编码读取文件

### 3. 找不到 notes 目录

**问题**：首次运行时目录未创建
**解决**：程序会自动创建 `notes/` 目录

## 集成到工作流程

### 研究阶段

1. 查找论文 → `search_arxiv_papers`
2. 下载论文 → `load_paper_for_deep_analysis`
3. 阅读总结 → Agent 检索问答
4. **保存笔记** → `save_note`

### 写作阶段

1. 查看历史笔记 → `list_notes`
2. 整理素材 → 编辑笔记文件
3. 引用参考 → 基于保存的笔记写作

## 扩展建议

未来可以添加：

1. 笔记搜索功能（在笔记内容中检索）
2. 笔记分类标签
3. 笔记版本控制
4. 云端同步支持
