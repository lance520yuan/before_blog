# 论文阅读要求文档
> 截止时间：

> 论文阅读数目：14

> 论文文章概览：
> System
> 
> ML
> 
> ML - complier

## 论文

[Latte: A Language, Compiler, and Runtime for Elegant and Efficient Deep Neural Networks-]()

[Willump: A Statistically-Aware End-to-end Optimizer for Machine Learning Inference](#)

[TensorFlow Eager: A multi-stage, Python-embedded DSL for machine learning]()

[RLgraph: Modular Computation Graphs for Deep Reinforcement Learning]()

[Optimizing DNN Computation with Relaxed Graph Substitutions]()

[AutoGraph: Imperative-style Coding with Graph-based Performance]()

[TVM: An Automated End-to-End Optimizing Compiler for Deep Learning]()

[Retiarii: A Deep Learning Exploratory-Training Framework]()

[Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks]()

[KungFu: Making Training in Distributed Machine Learning Adaptive]()

[Ansor: Generating High-Performance Tensor Programs for Deep Learning]()

[A Tensor Compiler Approach for One-size-fits-all ML Prediction Serving]()

[TASO: Optimizing Deep Learning Computation with Automated Generation of Graph Substitutions]()

[Bridging the Gap Between Neural Networks and Neuromorphic Hardware with A Neural Network Compiler]()

## 论文核心阅读目的

1. 这个云计算的本质是只使用远程的 `CPU` 使用

![Alt text](https://g.gravizo.com/svg?
digraph finite_state_machine {
    rankdir=LR;
    size="8,5"

    node [shape = circle]; CPU;
    node [shape = circle]; GPU;
    CPU -> GPU[label = "Transport_demand"];
}
)
