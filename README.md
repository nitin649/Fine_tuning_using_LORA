# Fine tuning using LoRA

# What is LoRA

Low-Rank Adaptation of Large Language Models (LoRA) is a training method that accelerates the training of large models while consuming less memory. 
It adds pairs of rank-decomposition weight matrices (called update matrices) to existing weights, and only trains those newly added weights.


# Why we need LoRA
Large models are trained to capture the general representation of their domain (language for LLMs, audio + language for models like Whisper, 
and vision for image generation models). These models capture a variety of features which allow them to be used for diverse tasks with reasonable zero-shot accuracy.
However, when adapting such a model to a specific task or dataset, only a few features need to be emphasized or re-learnt. This means that the update matrix (ΔW) can be a low-rank matrix.

# Method
The technique constrains the rank of the update matrix ΔW using its rank decomposition. It represents ΔWₙₖ as the product of 2 low-rank matrices Bₙᵣ and Aᵣₖ where r << min(n, k). 
This implies that the forward pass of the layer, originally Wx, is modified to Wx + BAx (as shown in the figure below). A random Gaussian initialization is used for A and B is initially to 0, 
so BA=0 at the start of training. The update BA is additionally scaled with a factor α/r.
![64649977d084d2b4b66c6492_1_e5pYWjrZR3eA_YbCKu8deQ](https://github.com/nitin649/Fine_tuning_using_LORA/assets/55678844/a9f8db6f-39c4-4cc0-ba69-66da6d3bcbd0)


# Benefits of using Lora
1. Reduction of training time and space: Using the technique shown above, r(n + k) parameters have to be tuned during model adaption. Since r << min(n, k), this is much lesser than the number of parameters
   that would have to be tuned otherwise (nk). This reduces the time and space required to finetune the model by a large margin. Some numbers from the paper and our experiments are discussed in the sections below.
2. No additional inference time: If used in production, we can explicitly compute W’ = W + BA and store the results, performing inference as usual. This guarantees that we do not introduce any additional latency during inference.
3. Easier task switching: Swapping only the LoRA weights as opposed to all the parameters allows cheaper and faster switching between tasks. Multiple customized models can be created and swapped in and out easily.

