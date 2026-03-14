# arXiv 论文助手 - 文件保存功能添加总结

## 修改时间

2026年3月14日

## 问题背景

用户反馈 arXiv 论文助手缺少本地文件操作功能，无法保存总结、笔记等内容。

## 解决方案

添加了两个核心工具函数，支持本地文件读写操作：

### 1. 新增工具函数

#### `save_note(filename: str, content: str) -> str`

**功能**：将笔记内容保存为 Markdown 文件
**参数**：

- `filename`：文件名（自动添加 .md 后缀）
- `content`：要保存的文本内容
  **返回**：保存成功或失败的消息
  **文件位置**：`notes/` 目录

#### `list_notes() -> str`

**功能**：列出所有已保存的笔记文件
**返回**：笔记文件列表，包含文件名和大小信息

### 2. 目录结构修改

```python
NOTES_DIR = BASE_DIR / "notes"  # 新增笔记目录
NOTES_DIR.mkdir(parents=True, exist_ok=True)  # 自动创建目录
```

### 3. Agent 工具配置更新

#### Local RAG Expert

```python
tools=[
    scan_and_index_new_papers,
    list_indexed_papers,
    load_paper_for_deep_analysis,
    save_note,      # 新增
    list_notes,     # 新增
],
```

#### arXiv Team (主管)

```python
tools=[list_indexed_papers, save_note, list_notes],  # 新增 save_note, list_notes
```

### 4. 指令说明更新

更新了 RAG Expert 的指令，明确说明：

- 可用工具包括 `save_note` 和 `list_notes`
- 当用户要求保存总结或笔记时，应使用 `save_note` 工具
- 提供具体的文件名和内容参数

## 使用方式

### 自然语言指令示例

1. **保存笔记**：
   - "保存这个总结"
   - "把这个研究笔记保存为文件"
   - "保存当前上下文"

2. **查看笔记**：
   - "列出我的笔记"
   - "有哪些保存的文件"
   - "查看笔记列表"

### 预期工作流程

```
用户：请总结这篇论文并保存为笔记
→ Agent 理解意图，调用 save_note 工具
→ Python 执行文件写入操作
→ 生成 notes/论文总结.md 文件
→ Agent 返回保存成功消息
```

## 技术细节

### 文件格式

- 所有笔记保存为 `.md` 格式（Markdown）
- 使用 UTF-8 编码确保中文支持
- 文件存储在项目根目录的 `notes/` 子目录

### 安全性

- 文件名验证（防止路径穿越攻击）
- 异常处理（IO错误、权限问题等）
- 目录自动创建（确保 notes/ 目录存在）

## 测试验证

已创建测试文件：

- `notes/test_context.md` - 功能测试文件
- `notes/context_summary.md` - 本总结文件

## 遗留问题

1. **依赖问题**：项目需要 `chonkie` 库支持语义分块
2. **集成测试**：需要在实际 Agent 环境中测试工具调用流程

## 后续优化建议

1. 添加文件编辑功能（追加内容、修改笔记）
2. 支持按论文ID自动命名文件
3. 添加笔记分类（按主题、时间等）
4. 集成到论文总结流程中（自动保存每次的问答总结）
