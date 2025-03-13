# DRESS: 动态表示空间干预技术分析

本文档分析了DRESS (Direction-based REpresentation Space Surgery) 技术的实现，该技术用于编辑大型语言模型的行为。分析基于两个核心文件：`edit_weight.py`和`generate.py`。

## 1. 技术概述

DRESS技术通过以下步骤修改语言模型的行为：

1. **识别关键注意力头**：找出与特定行为（如特定风格）最相关的注意力头
2. **计算干预向量**：基于正确样本和不正确样本的激活差异计算干预向量
3. **应用干预**：将干预向量作为偏置添加到模型的输出投影层
4. **动态调整**：在生成过程中动态调整干预强度

## 2. 核心文件分析

### 2.1 `edit_weight.py` - 静态干预向量计算

这个文件负责识别关键注意力头并计算初始干预向量。

#### 主要功能

1. **加载模型和数据**：
   ```python
   tokenizer = qwen2.Qwen2Tokenizer.from_pretrained(MODEL)
   model = qwen2.Qwen2ForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
   ```

2. **识别关键注意力头**：
   ```python
   top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)
   ```

3. **计算干预向量**：
   ```python
   for head_no, head_vec, std in list_int_vec:
       activations = tuning_activations[:,layer_no,head_no,:]
       correct_activations = activations[::2, :]
       incorrect_activations = activations[1::2, :]
       correct_activations = np.mean(correct_activations, axis=0)
       incorrect_activations = np.mean(incorrect_activations, axis=0)
       displacement[head_no] = args.alpha * (correct_activations - incorrect_activations)
   ```

4. **应用干预向量**：
   ```python
   displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
   bias_tobe = F.linear(displacement.to(torch.float16), model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
   model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)
   ```

5. **保存编辑后的模型**：
   ```python
   model.save_pretrained(save_folder, safe_serialization=False, max_shard_size="10GB")
   tokenizer.save_pretrained(save_folder)
   ```

#### 关键参数

- `num_heads`：要干预的注意力头数量
- `alpha`：干预强度系数
- `val_ratio`：验证集比例
- `use_center_of_mass`/`use_random_dir`：干预方向选择策略

### 2.2 `generate.py` - 动态干预生成

这个文件实现了在生成过程中动态调整干预向量的技术。

#### 主要功能

1. **SVD分解预处理**：
   ```python
   def svd_decomposition(layer_no, head_no, X):
       U, s, Vh = svd(X, full_matrices=False)
       key = 'L' + str(layer_no) + 'H' + str(head_no)
       svd_s_dict[key] = s
       svd_Vh_dict[key] = Vh
   ```

2. **动态干预向量计算**：
   ```python
   def get_steering_vector(layer_no, head_no, vector, cur_activations):
       # ...
       w = np.dot(Vh, x.T)
       w2 = np.dot(Vh, cur_activations.T)
       # ...
       w4 = np.dot(Vh, correct_activations.T)
       w *= (1.0 + 0.5 * np.sign(w) * (w4 - w2))
       xx = np.dot(V, w)
       return xx
   ```

3. **自回归生成过程中的实时干预**：
   ```python
   def my_generate(w0, q_tokens, inputs):
       # ...
       for i in range(max_length):
           # ...
           token = torch.tensor([probs.argmax().item()]).unsqueeze(0).to('cuda:0')
           generated = torch.cat((generated, token), dim=1)
           q_tokens = torch.cat((q_tokens, token), dim=1)
           sequence.append(token.cpu().numpy()[0][0])
           get_activations(q_tokens)
           # ...
   ```

#### 关键变量

- `q_tokens`：原始问题的tokenized版本，用于激活计算
- `inputs`：完整提示（包含指令和格式化问题）的tokenized版本，用于实际生成
- `svd_s_dict`/`svd_Vh_dict`：缓存的SVD分解结果
- `activations_dict`：从`edit_weight.py`保存的干预向量

## 3. 自适应干预机制分析

### 3.1 核心公式

```python
w *= (1.0 + 0.5 * np.sign(w) * (w4 - w2))
```

这个公式动态调整干预强度，其中：
- `w`：原始干预向量在SVD空间的投影
- `w2`：当前激活状态在SVD空间的投影
- `w4`：目标激活状态在SVD空间的投影
- `np.sign(w)`：确保调整与原始干预方向一致
- `0.5`：调整幅度系数

### 3.2 边界情况分析

- 当`w4 == w2`（当前激活已达目标状态）：
  - 公式简化为`w *= 1.0`
  - 干预向量保持不变，不做额外调整
  
- 当`w4 > w2`（当前激活低于目标状态）：
  - 如果`w > 0`，干预强度增加
  - 如果`w < 0`，干预强度减弱
  
- 当`w4 < w2`（当前激活高于目标状态）：
  - 如果`w > 0`，干预强度减弱
  - 如果`w < 0`，干预强度增加

## 4. 实现特点与优化空间

### 4.1 实现特点

1. **分离的干预计算和应用**：
   - 在`edit_weight.py`中计算初始干预向量
   - 在`generate.py`中动态调整并应用干预

2. **SVD降维**：
   - 使用SVD将高维激活空间降维到主成分空间
   - 只保留前K个主成分（K=64）

3. **重置式干预**：
   - 每次生成新token前，先重置所有偏置为零
   - 基于当前激活状态重新计算干预向量

4. **双轨输入表示**：
   - `q_tokens`：用于激活计算的原始问题表示
   - `inputs`：用于实际生成的完整提示表示

### 4.2 优化空间

1. **计算效率优化**：
   - 避免重复重置偏置
   - 使用批处理计算多个头的干预向量
   - 减少设备间数据传输
   - 使用向量化操作替代循环

2. **架构改进**：
   - 分离关注点，使代码更模块化
   - 实现增量式干预，而非完全重置
   - 设计更灵活的干预策略
   - 自动调整超参数

3. **高级优化方向**：
   - 分布式计算
   - 量化技术
   - 自动微分
   - 元学习机制

## 5. 总结

DRESS技术通过识别关键注意力头并应用动态调整的干预向量，实现了对大型语言模型行为的精细控制。其核心创新在于：

1. 使用SVD降维来捕获最重要的激活方向
2. 在生成过程中动态调整干预强度
3. 将干预应用为模型输出投影层的偏置

这种方法相比传统微调有几个优势：
- 更精确地控制特定行为
- 保持模型其他能力不变
- 计算成本较低

然而，当前实现存在优化空间，特别是在计算效率和架构设计方面。通过改进，这种技术可以更高效地应用于各种语言模型行为编辑任务。 