# KDLOR
the implement of KDLOR and orthogonal KDLOR

## 使用
```maltab
n = 5000; d = 20; K = 10;
data = rand(d, n);              % 样本随机生成
label = randi([1, K], 1, n);    % 随机生成对应标签

% C 超参数， p方向数
C = 10; p = 5;

% 求解 => dxp
w = KDLOR_orth(data, label, C, K, p);
```
