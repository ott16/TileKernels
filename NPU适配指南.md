# NPU适配指南

## 概述

本文档总结了TileKernels项目从GPU到NPU的适配经验，提供了完整的适配规则和操作流程，方便后续其他测试文件的NPU适配工作。

## 适配背景

- **目标平台**: NPU (Neural Processing Unit) - 华为硬件
- **原始平台**: GPU (CUDA)
- **适配框架**: TileLang
- **适配对象**: TileKernels项目中的kernel文件

## NPU适配规则

### 规则1: 注释不支持的配置项

**问题描述**: NPU不支持某些GPU特定的配置项，会导致 `AttributeError`。

**修改方法**:
```python
# 修改前
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
        tilelang.PassConfigKey.TL_DISABLE_WGMMA: True,
    },
)

# 修改后
@tilelang.jit(
    pass_configs={},  # NPU适配: 注释掉不支持的配置项
)
```

**不支持的配置项**:
- `TL_DISABLE_WARP_SPECIALIZED`
- `TL_DISABLE_THREAD_STORAGE_SYNC`
- `TL_DISABLE_OUT_OF_BOUND_WARNING`
- `TL_DISABLE_WGMMA`
- `TL_PTXAS_REGISTER_USAGE_LEVEL`
- `TL_DISABLE_VECTORIZE_256`

### 规则2: 替换T.dtype为torch.dtype (v1.1已修正)

**问题描述**: NPU环境不支持 `T.dtype`，需要使用 `torch.dtype`。

**v1.1修正**: 实际上应该移除类型注解，让Python根据默认值自动推断类型。

**修改方法**:
```python
# 修改前
def kernel_func(m: int, n: int, dtype: T.dtype = T.float32) -> tilelang.JITKernel:

# v1.0版本（错误）
def kernel_func(m: int, n: int, dtype: torch.dtype = torch.float32) -> tilelang.JITKernel:

# v1.1版本（正确）
def kernel_func(m: int, n: int, dtype=T.float32) -> tilelang.JITKernel:  # NPU适配: 让Python推断类型
```

**说明**: 移除类型注解，让Python根据默认值 `T.float32` 自动推断类型。

### 规则3: 移除T.Ref声明

**问题描述**: NPU环境不支持 `T.Ref` 声明。

**修改方法**: 检查并删除所有的 `T.Ref` 声明，通常不需要替换。

### 规则4: 替换T.alloc_fragment为T.Buffer

**问题描述**: NPU环境使用 `T.Buffer` 替代 `T.alloc_fragment`。

**修改方法**:
```python
# 修改前
out_frag = T.alloc_fragment((token_block, 32), T.float32)

# 修改后
out_frag = T.Buffer((token_block, 32), T.float32)  # NPU适配: T.alloc_fragment -> T.Buffer
```

### 规则5: 替换T.dynamic为T.symbolic

**问题描述**: NPU环境使用 `T.symbolic` 替代 `T.dynamic`。

**修改方法**:
```python
# 修改前
num_tokens = T.dynamic('num_tokens')

# 修改后
num_tokens = T.symbolic('num_tokens')  # NPU适配: T.dynamic -> T.symbolic
```

### 规则6: 替换T.StridedTensor为T.Tensor

**问题描述**: NPU环境不支持 `T.StridedTensor`，需要使用 `T.Tensor`。

**修改方法**:
```python
# 修改前
k: T.StridedTensor[(num_tokens, hc_mult, hidden_size), (k_stride_s, k_stride_h, 1), T.bfloat16]
v: T.StridedTensor[(num_tokens, hidden_size), (v_stride_s, 1), T.bfloat16]

# 修改后
k: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16]  # NPU适配: T.StridedTensor -> T.Tensor
v: T.Tensor[(num_tokens, hidden_size), T.bfloat16]  # NPU适配: T.StridedTensor -> T.Tensor
```

### 规则7: 替换T.clear为T.tile.fill

**问题描述**: NPU环境使用 `T.tile.fill` 替代 `T.clear`。

**修改方法**:
```python
# 修改前
T.clear(out_frag)
T.clear(rms_grad_frag)

# 修改后
T.tile.fill(out_frag, 0.0)  # NPU适配: T.clear -> T.tile.fill
T.tile.fill(rms_grad_frag, 0.0)  # NPU适配: T.clear -> T.tile.fill
```

### 规则8: 修复T.reduce_sum接口

**问题描述**: NPU环境的 `T.reduce_sum` 需要显式指定 `dim` 参数。

**修改方法**:
```python
# 修改前
T.reduce_sum(sqrsum_part, sqrsum_l)

# 修改后
T.reduce_sum(sqrsum_part, sqrsum_l, dim=0)  # NPU适配: 添加dim参数
```

 ### 规则9: 替换T.fill为T.tile.fill
 **修改方法**:
 ```python
  T.fill调用替换为T.tile.fill(buffer, 0.0)
 ```

### 规则10: 修复类型注解不匹配问题

**问题描述**: NPU环境不支持函数参数中的 `T.dtype` 类型注解，需要移除类型注解让Python自动推断类型。

**修改方法**:
```python
# 修改前
def kernel_func(m: int, n: int, dtype: T.dtype = T.float32) -> tilelang.JITKernel:

# 修改后
def kernel_func(m: int, n: int, dtype=T.float32) -> tilelang.JITKernel:  # NPU适配: 移除类型注解
```

**说明**: 移除 `dtype: T.dtype` 类型注解，只保留默认值 `T.float32`，让Python自动推断类型。

### 规则11: 修复T.Tensor形状声明

**问题描述**: NPU环境要求所有 `T.Tensor` 形状声明必须使用元组形式，包括一维数组。

**修改方法**:
```python
# 修改前
normw: T.Tensor[n, dtype]
fn_grad: T.Tensor[(m, n), dtype]

# 修改后
normw: T.Tensor[(n,), dtype]  # NPU适配: 一维数组使用元组形式
fn_grad: T.Tensor[(m, n), dtype]
```

**说明**: 无论是一维还是多维数组，形状都必须使用元组形式。

### 规则12: dtype使用字符串

**问题描述**: NPU环境不支持 `T.bfloat16`、`T.float32` 等类型属性，需要使用字符串形式。

**修改方法**:
```python
# 修改前
x: T.Tensor[(num_tokens, n_rms_group * rms_group_size), T.bfloat16]
fn: T.Tensor[(mhc_mult3, n_rms_group * rms_group_size), T.float32]
out_frag = T.alloc_fragment((token_block, 32), T.float32)

# 修改后
x: T.Tensor[(num_tokens, n_rms_group * rms_group_size), "bfloat16"]  # NPU适配: dtype使用字符串
fn: T.Tensor[(mhc_mult3, n_rms_group * rms_group_size), "float32"]  # NPU适配: dtype使用字符串
out_frag = T.alloc_fragment((token_block, 32), "float32")  # NPU适配: dtype使用字符串
```

 **说明**: 将所有 `T.bfloat16`、`T.float32`、`T.int32` 等类型改为对应的字符串形式 `"bfloat16"`、`"float32"`、`"int32"`。

### 规则13: 添加target="npuir"

**问题描述**: TileLang 的 `@tilelang.jit` 装饰器在编译时会自动检测可用的目标设备（CUDA/HIP/MPS），但需要显式指定 NPU 目标。

**修改方法**:
```python
# 修改前
@tilelang.jit(
    pass_configs={},  # NPU适配: 注释掉不支持的配置项
)

# 修改后
@tilelang.jit(
    pass_configs={},  # NPU适配: 注释掉不支持的配置项
    target="npuir",   # NPU适配: 添加target="npuir"参数
)
```

**说明**: 在所有 `@tilelang.jit` 装饰器中添加 `target="npuir"` 参数，明确指定编译目标为 NPU IR（Intermediate Representation）。

### 通用提示词

```
请为TileKernels项目中的kernel文件进行NPU适配，首先分析该测试文件的依赖关系：
1. 查看测试文件导入了哪些函数
2. 追踪这些函数的实现位置
 3. 确定需要适配的kernel文件列表

 需要应用以下13个NPU适配规则：1. 注释不支持的配置项：将所有@tilelang.jit装饰器中的pass_configs中不支持的配置项注释掉，只保留空字典 {}
  不支持的配置项包括：TL_DISABLE_WARP_SPECIALIZED, TL_DISABLE_THREAD_STORAGE_SYNC, TL_DISABLE_OUT_OF_BOUND_WARNING, TL_DISABLE_WGMMA, TL_PTXAS_REGISTER_USAGE_LEVEL, TL_DISABLE_VECTORIZE_256

2. 替换T.dtype为torch.dtype（v1.1修正）：将函数参数中的T.dtype替换为移除类型注解，让Python推断类型
  例如：dtype: T.dtype = T.float32 -> dtype=T.float32

3. 移除T.Ref声明：删除所有的T.Ref声明

4. 替换T.alloc_fragment为T.Buffer：将所有T.alloc_fragment调用替换为T.Buffer

5. 替换T.dynamic为T.symbolic：将所有T.dynamic调用替换为T.symbolic

6. 替换T.StridedTensor为T.Tensor：将所有T.StridedTensor类型声明替换为T.Tensor

7. 替换T.clear为T.tile.fill：将所有T.clear调用替换为T.tile.fill(buffer, 0.0)

8. 修复T.reduce_sum接口：为T.reduce_sum调用添加dim参数

9. 替换T.fill为T.tile.fill：将所有T.fill调用替换为T.tile.fill(buffer, 0.0)

10. 修复类型注解不匹配问题（v1.1新增）：移除函数参数中的T.dtype类型注解，让Python推断类型
  例如：dtype: T.dtype = T.float32 -> dtype=T.float32

11. 修复T.Tensor形状声明（v1.1新增）：所有T.Tensor形状声明，包括一维数组，都必须使用元组形式
  例如：n 改为 (n,)

 12. dtype使用字符串（v1.1新增）：将所有T.bfloat16、T.float32等类型改为字符串形式"bfloat16"、"float32"
  例如：T.bfloat16 -> "bfloat16", T.float32 -> "float32"

 13. 添加target="npuir"（v1.2新增）：在所有@tilelang.jit装饰器中添加target="npuir"参数
  例如：@tilelang.jit -> @tilelang.jit(target="npuir")

 注意：1、请逐个检查文件，应用这些规则，并在修改处添加注释说明是NPU适配。2、请只适配实际需要的文件，不要过度适配。```
```

## 常见问题
### Q1: 如何确定需要适配哪些文件？

**A**: 分析测试文件的依赖关系：
1. 查看测试文件导入的函数
2. 追踪函数的实现位置
3. 只适配实际需要的kernel文件

### Q2: 为什么类型注解 `dtype: torch.dtype = T.float32` 会报错？

**A**: 类型注解 `torch.dtype` 声明参数应该是 `torch.float32`、`torch.float64` 等类型，但默认值 `T.float32` 是 TileLang 的类型对象。当传入参数时，TileLang 内部期望的是 `T.float32` 类型，但类型注解误导了类型检查。

**解决方案**: 移除类型注解，改为 `dtype=T.float32`，让 Python 自动推断类型。

### Q3: 为什么 `T.tile.fill` 不存在？

**A**: TileLang API 中没有 `T.tile.fill` 方法，只有 `T.fill` 方法。之前的适配规则中错误地将 `T.fill` 替换为 `T.tile.fill`。

**解决方案**: 将所有 `T.tile.fill` 替换回 `T.fill`。

### Q4: 为什么 `@tilelang.jit` 在 NPU 上报错 "No CUDA or HIP or MPS available"？

**A**: TileLang 的 `@tilelang.jit` 装饰器在编译时会自动检测可用的目标设备（CUDA/HIP/MPS），但当前版本不支持 NPU。


## 总结

NPU适配是一个系统性的工作，需要：

1. **精准分析依赖关系** - 只适配实际需要的文件
2. **严格按照规则适配** - 应用NPU适配规则（当前共13个规则，v1.1版本新增3个，v1.2版本新增1个）
3. **正确配置运行环境** - 使用PYTHONPATH确保使用修改后的源代码
4. **验证适配结果** - 运行测试确保适配正确
5. **持续更新规则** - 根据实际遇到的问题，不断完善和修正适配规则
 ## 版本历史

 ### v1.2 (2026-05-06)

 **新增规则**:
 - 规则13: 添加 target="npuir"

 ### v1.1 (2026-05-06)

 **新增规则**:
 - 规则10: 修复类型注解不匹配问题
 - 规则11: 修复T.Tensor形状声明
 - 规则12: dtype使用字符串

 ### v1.0 (初始版本)
**初始规则**:
- 规则1: 注释不支持的配置项
- 规则2: 替换T.dtype为torch.dtype
- 规则3: 移除T.Ref声明
- 规则4: 替换T.alloc_fragment为T.Buffer
- 规则5: 替换T.dynamic为T.symbolic
- 规则6: 替换T.StridedTensor为T.Tensor
- 规则7: 替换T.clear为T.tile.fill
- 规则8: 修复T.reduce_sum接口
- 规则9: 替换T.fill为T.tile.fill

遵循本文档的指南和提示词，可以高效地完成NPU适配工作。
