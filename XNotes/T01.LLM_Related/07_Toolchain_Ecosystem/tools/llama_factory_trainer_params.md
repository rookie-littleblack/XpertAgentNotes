# LLaMA Factory Trainer Params

> 这份训练参数列表，通过命令“llamafactory-cli train -h”于 2025-04-27 11:10 获取。

```bash
INFO 04-27 11:08:41 [__init__.py:239] Automatically detected platform cuda.
[INFO|2025-04-27 11:08:43] llamafactory.cli:143 >> Initializing 8 distributed tasks at: 127.0.0.1:46249
usage: launcher.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH]
                   [--adapter_name_or_path ADAPTER_NAME_OR_PATH]
                   [--adapter_folder ADAPTER_FOLDER] [--cache_dir CACHE_DIR]
                   [--use_fast_tokenizer [USE_FAST_TOKENIZER]]
                   [--no_use_fast_tokenizer] [--resize_vocab [RESIZE_VOCAB]]
                   [--split_special_tokens [SPLIT_SPECIAL_TOKENS]]
                   [--add_tokens ADD_TOKENS]
                   [--add_special_tokens ADD_SPECIAL_TOKENS]
                   [--model_revision MODEL_REVISION]
                   [--low_cpu_mem_usage [LOW_CPU_MEM_USAGE]]
                   [--no_low_cpu_mem_usage]
                   [--rope_scaling {linear,dynamic,yarn,llama3}]
                   [--flash_attn {auto,disabled,sdpa,fa2}]
                   [--shift_attn [SHIFT_ATTN]]
                   [--mixture_of_depths {convert,load}]
                   [--use_unsloth [USE_UNSLOTH]]
                   [--use_unsloth_gc [USE_UNSLOTH_GC]]
                   [--enable_liger_kernel [ENABLE_LIGER_KERNEL]]
                   [--moe_aux_loss_coef MOE_AUX_LOSS_COEF]
                   [--disable_gradient_checkpointing [DISABLE_GRADIENT_CHECKPOINTING]]
                   [--use_reentrant_gc [USE_REENTRANT_GC]]
                   [--no_use_reentrant_gc]
                   [--upcast_layernorm [UPCAST_LAYERNORM]]
                   [--upcast_lmhead_output [UPCAST_LMHEAD_OUTPUT]]
                   [--train_from_scratch [TRAIN_FROM_SCRATCH]]
                   [--infer_backend {huggingface,vllm,sglang}]
                   [--offload_folder OFFLOAD_FOLDER] [--use_cache [USE_CACHE]]
                   [--no_use_cache]
                   [--infer_dtype {auto,float16,bfloat16,float32}]
                   [--hf_hub_token HF_HUB_TOKEN] [--ms_hub_token MS_HUB_TOKEN]
                   [--om_hub_token OM_HUB_TOKEN]
                   [--print_param_status [PRINT_PARAM_STATUS]]
                   [--trust_remote_code [TRUST_REMOTE_CODE]]
                   [--quantization_method {bnb,gptq,awq,aqlm,quanto,eetq,hqq}]
                   [--quantization_bit QUANTIZATION_BIT]
                   [--quantization_type {fp4,nf4}]
                   [--double_quantization [DOUBLE_QUANTIZATION]]
                   [--no_double_quantization]
                   [--quantization_device_map {auto}]
                   [--image_max_pixels IMAGE_MAX_PIXELS]
                   [--image_min_pixels IMAGE_MIN_PIXELS]
                   [--image_do_pan_and_scan [IMAGE_DO_PAN_AND_SCAN]]
                   [--crop_to_patches [CROP_TO_PATCHES]]
                   [--use_audio_in_video [USE_AUDIO_IN_VIDEO]]
                   [--video_max_pixels VIDEO_MAX_PIXELS]
                   [--video_min_pixels VIDEO_MIN_PIXELS]
                   [--video_fps VIDEO_FPS] [--video_maxlen VIDEO_MAXLEN]
                   [--audio_sampling_rate AUDIO_SAMPLING_RATE]
                   [--export_dir EXPORT_DIR] [--export_size EXPORT_SIZE]
                   [--export_device {cpu,auto}]
                   [--export_quantization_bit EXPORT_QUANTIZATION_BIT]
                   [--export_quantization_dataset EXPORT_QUANTIZATION_DATASET]
                   [--export_quantization_nsamples EXPORT_QUANTIZATION_NSAMPLES]
                   [--export_quantization_maxlen EXPORT_QUANTIZATION_MAXLEN]
                   [--export_legacy_format [EXPORT_LEGACY_FORMAT]]
                   [--export_hub_model_id EXPORT_HUB_MODEL_ID]
                   [--vllm_maxlen VLLM_MAXLEN] [--vllm_gpu_util VLLM_GPU_UTIL]
                   [--vllm_enforce_eager [VLLM_ENFORCE_EAGER]]
                   [--vllm_max_lora_rank VLLM_MAX_LORA_RANK]
                   [--vllm_config VLLM_CONFIG] [--sglang_maxlen SGLANG_MAXLEN]
                   [--sglang_mem_fraction SGLANG_MEM_FRACTION]
                   [--sglang_tp_size SGLANG_TP_SIZE]
                   [--sglang_config SGLANG_CONFIG] [--template TEMPLATE]
                   [--dataset DATASET] [--eval_dataset EVAL_DATASET]
                   [--dataset_dir DATASET_DIR] [--media_dir MEDIA_DIR]
                   [--cutoff_len CUTOFF_LEN]
                   [--train_on_prompt [TRAIN_ON_PROMPT]]
                   [--mask_history [MASK_HISTORY]] [--streaming [STREAMING]]
                   [--buffer_size BUFFER_SIZE]
                   [--mix_strategy {concat,interleave_under,interleave_over}]
                   [--interleave_probs INTERLEAVE_PROBS]
                   [--overwrite_cache [OVERWRITE_CACHE]]
                   [--preprocessing_batch_size PREPROCESSING_BATCH_SIZE]
                   [--preprocessing_num_workers PREPROCESSING_NUM_WORKERS]
                   [--max_samples MAX_SAMPLES]
                   [--eval_num_beams EVAL_NUM_BEAMS]
                   [--ignore_pad_token_for_loss [IGNORE_PAD_TOKEN_FOR_LOSS]]
                   [--no_ignore_pad_token_for_loss] [--val_size VAL_SIZE]
                   [--packing PACKING] [--neat_packing [NEAT_PACKING]]
                   [--tool_format TOOL_FORMAT]
                   [--tokenized_path TOKENIZED_PATH] [--output_dir OUTPUT_DIR]
                   [--overwrite_output_dir [OVERWRITE_OUTPUT_DIR]]
                   [--do_train [DO_TRAIN]] [--do_eval [DO_EVAL]]
                   [--do_predict [DO_PREDICT]]
                   [--eval_strategy {no,steps,epoch}]
                   [--prediction_loss_only [PREDICTION_LOSS_ONLY]]
                   [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]
                   [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE]
                   [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
                   [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
                   [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                   [--eval_accumulation_steps EVAL_ACCUMULATION_STEPS]
                   [--eval_delay EVAL_DELAY]
                   [--torch_empty_cache_steps TORCH_EMPTY_CACHE_STEPS]
                   [--learning_rate LEARNING_RATE]
                   [--weight_decay WEIGHT_DECAY] [--adam_beta1 ADAM_BETA1]
                   [--adam_beta2 ADAM_BETA2] [--adam_epsilon ADAM_EPSILON]
                   [--max_grad_norm MAX_GRAD_NORM]
                   [--num_train_epochs NUM_TRAIN_EPOCHS]
                   [--max_steps MAX_STEPS]
                   [--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau,cosine_with_min_lr,warmup_stable_decay}]
                   [--lr_scheduler_kwargs LR_SCHEDULER_KWARGS]
                   [--warmup_ratio WARMUP_RATIO] [--warmup_steps WARMUP_STEPS]
                   [--log_level {detail,debug,info,warning,error,critical,passive}]
                   [--log_level_replica {detail,debug,info,warning,error,critical,passive}]
                   [--log_on_each_node [LOG_ON_EACH_NODE]]
                   [--no_log_on_each_node] [--logging_dir LOGGING_DIR]
                   [--logging_strategy {no,steps,epoch}]
                   [--logging_first_step [LOGGING_FIRST_STEP]]
                   [--logging_steps LOGGING_STEPS]
                   [--logging_nan_inf_filter [LOGGING_NAN_INF_FILTER]]
                   [--no_logging_nan_inf_filter]
                   [--save_strategy {no,steps,epoch,best}]
                   [--save_steps SAVE_STEPS]
                   [--save_total_limit SAVE_TOTAL_LIMIT]
                   [--save_safetensors [SAVE_SAFETENSORS]]
                   [--no_save_safetensors]
                   [--save_on_each_node [SAVE_ON_EACH_NODE]]
                   [--save_only_model [SAVE_ONLY_MODEL]]
                   [--restore_callback_states_from_checkpoint [RESTORE_CALLBACK_STATES_FROM_CHECKPOINT]]
                   [--no_cuda [NO_CUDA]] [--use_cpu [USE_CPU]]
                   [--use_mps_device [USE_MPS_DEVICE]] [--seed SEED]
                   [--data_seed DATA_SEED] [--jit_mode_eval [JIT_MODE_EVAL]]
                   [--use_ipex [USE_IPEX]] [--bf16 [BF16]] [--fp16 [FP16]]
                   [--fp16_opt_level FP16_OPT_LEVEL]
                   [--half_precision_backend {auto,apex,cpu_amp}]
                   [--bf16_full_eval [BF16_FULL_EVAL]]
                   [--fp16_full_eval [FP16_FULL_EVAL]] [--tf32 TF32]
                   [--local_rank LOCAL_RANK]
                   [--ddp_backend {nccl,gloo,mpi,ccl,hccl,cncl,mccl}]
                   [--tpu_num_cores TPU_NUM_CORES]
                   [--tpu_metrics_debug [TPU_METRICS_DEBUG]]
                   [--debug DEBUG [DEBUG ...]]
                   [--dataloader_drop_last [DATALOADER_DROP_LAST]]
                   [--eval_steps EVAL_STEPS]
                   [--dataloader_num_workers DATALOADER_NUM_WORKERS]
                   [--dataloader_prefetch_factor DATALOADER_PREFETCH_FACTOR]
                   [--past_index PAST_INDEX] [--run_name RUN_NAME]
                   [--disable_tqdm DISABLE_TQDM]
                   [--remove_unused_columns [REMOVE_UNUSED_COLUMNS]]
                   [--no_remove_unused_columns]
                   [--label_names LABEL_NAMES [LABEL_NAMES ...]]
                   [--load_best_model_at_end [LOAD_BEST_MODEL_AT_END]]
                   [--metric_for_best_model METRIC_FOR_BEST_MODEL]
                   [--greater_is_better GREATER_IS_BETTER]
                   [--ignore_data_skip [IGNORE_DATA_SKIP]] [--fsdp FSDP]
                   [--fsdp_min_num_params FSDP_MIN_NUM_PARAMS]
                   [--fsdp_config FSDP_CONFIG] [--tp_size TP_SIZE]
                   [--fsdp_transformer_layer_cls_to_wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP]
                   [--accelerator_config ACCELERATOR_CONFIG]
                   [--deepspeed DEEPSPEED]
                   [--label_smoothing_factor LABEL_SMOOTHING_FACTOR]
                   [--optim {adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_torch_npu_fused,adamw_apex_fused,adafactor,adamw_anyprecision,adamw_torch_4bit,adamw_torch_8bit,ademamix,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,ademamix_8bit,lion_8bit,lion_32bit,paged_adamw_32bit,paged_adamw_8bit,paged_ademamix_32bit,paged_ademamix_8bit,paged_lion_32bit,paged_lion_8bit,rmsprop,rmsprop_bnb,rmsprop_bnb_8bit,rmsprop_bnb_32bit,galore_adamw,galore_adamw_8bit,galore_adafactor,galore_adamw_layerwise,galore_adamw_8bit_layerwise,galore_adafactor_layerwise,lomo,adalomo,grokadamw,schedule_free_radam,schedule_free_adamw,schedule_free_sgd,apollo_adamw,apollo_adamw_layerwise}]
                   [--optim_args OPTIM_ARGS] [--adafactor [ADAFACTOR]]
                   [--group_by_length [GROUP_BY_LENGTH]]
                   [--length_column_name LENGTH_COLUMN_NAME]
                   [--report_to REPORT_TO]
                   [--ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS]
                   [--ddp_bucket_cap_mb DDP_BUCKET_CAP_MB]
                   [--ddp_broadcast_buffers DDP_BROADCAST_BUFFERS]
                   [--dataloader_pin_memory [DATALOADER_PIN_MEMORY]]
                   [--no_dataloader_pin_memory]
                   [--dataloader_persistent_workers [DATALOADER_PERSISTENT_WORKERS]]
                   [--skip_memory_metrics [SKIP_MEMORY_METRICS]]
                   [--no_skip_memory_metrics]
                   [--use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]]
                   [--push_to_hub [PUSH_TO_HUB]]
                   [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                   [--hub_model_id HUB_MODEL_ID]
                   [--hub_strategy {end,every_save,checkpoint,all_checkpoints}]
                   [--hub_token HUB_TOKEN]
                   [--hub_private_repo HUB_PRIVATE_REPO]
                   [--hub_always_push [HUB_ALWAYS_PUSH]]
                   [--gradient_checkpointing [GRADIENT_CHECKPOINTING]]
                   [--gradient_checkpointing_kwargs GRADIENT_CHECKPOINTING_KWARGS]
                   [--include_inputs_for_metrics [INCLUDE_INPUTS_FOR_METRICS]]
                   [--include_for_metrics INCLUDE_FOR_METRICS [INCLUDE_FOR_METRICS ...]]
                   [--eval_do_concat_batches [EVAL_DO_CONCAT_BATCHES]]
                   [--no_eval_do_concat_batches]
                   [--fp16_backend {auto,apex,cpu_amp}]
                   [--push_to_hub_model_id PUSH_TO_HUB_MODEL_ID]
                   [--push_to_hub_organization PUSH_TO_HUB_ORGANIZATION]
                   [--push_to_hub_token PUSH_TO_HUB_TOKEN]
                   [--mp_parameters MP_PARAMETERS]
                   [--auto_find_batch_size [AUTO_FIND_BATCH_SIZE]]
                   [--full_determinism [FULL_DETERMINISM]]
                   [--torchdynamo TORCHDYNAMO] [--ray_scope RAY_SCOPE]
                   [--ddp_timeout DDP_TIMEOUT]
                   [--torch_compile [TORCH_COMPILE]]
                   [--torch_compile_backend TORCH_COMPILE_BACKEND]
                   [--torch_compile_mode TORCH_COMPILE_MODE]
                   [--include_tokens_per_second [INCLUDE_TOKENS_PER_SECOND]]
                   [--include_num_input_tokens_seen [INCLUDE_NUM_INPUT_TOKENS_SEEN]]
                   [--neftune_noise_alpha NEFTUNE_NOISE_ALPHA]
                   [--optim_target_modules OPTIM_TARGET_MODULES]
                   [--batch_eval_metrics [BATCH_EVAL_METRICS]]
                   [--eval_on_start [EVAL_ON_START]]
                   [--use_liger_kernel [USE_LIGER_KERNEL]]
                   [--eval_use_gather_object [EVAL_USE_GATHER_OBJECT]]
                   [--average_tokens_across_devices [AVERAGE_TOKENS_ACROSS_DEVICES]]
                   [--sortish_sampler [SORTISH_SAMPLER]]
                   [--predict_with_generate [PREDICT_WITH_GENERATE]]
                   [--generation_max_length GENERATION_MAX_LENGTH]
                   [--generation_num_beams GENERATION_NUM_BEAMS]
                   [--generation_config GENERATION_CONFIG]
                   [--ray_run_name RAY_RUN_NAME]
                   [--ray_storage_path RAY_STORAGE_PATH]
                   [--ray_num_workers RAY_NUM_WORKERS]
                   [--resources_per_worker RESOURCES_PER_WORKER]
                   [--placement_strategy {SPREAD,PACK,STRICT_SPREAD,STRICT_PACK}]
                   [--ray_init_kwargs RAY_INIT_KWARGS]
                   [--freeze_trainable_layers FREEZE_TRAINABLE_LAYERS]
                   [--freeze_trainable_modules FREEZE_TRAINABLE_MODULES]
                   [--freeze_extra_modules FREEZE_EXTRA_MODULES]
                   [--additional_target ADDITIONAL_TARGET]
                   [--lora_alpha LORA_ALPHA] [--lora_dropout LORA_DROPOUT]
                   [--lora_rank LORA_RANK] [--lora_target LORA_TARGET]
                   [--loraplus_lr_ratio LORAPLUS_LR_RATIO]
                   [--loraplus_lr_embedding LORAPLUS_LR_EMBEDDING]
                   [--use_rslora [USE_RSLORA]] [--use_dora [USE_DORA]]
                   [--pissa_init [PISSA_INIT]] [--pissa_iter PISSA_ITER]
                   [--pissa_convert [PISSA_CONVERT]]
                   [--create_new_adapter [CREATE_NEW_ADAPTER]]
                   [--pref_beta PREF_BETA] [--pref_ftx PREF_FTX]
                   [--pref_loss {sigmoid,hinge,ipo,kto_pair,orpo,simpo}]
                   [--dpo_label_smoothing DPO_LABEL_SMOOTHING]
                   [--kto_chosen_weight KTO_CHOSEN_WEIGHT]
                   [--kto_rejected_weight KTO_REJECTED_WEIGHT]
                   [--simpo_gamma SIMPO_GAMMA]
                   [--ppo_buffer_size PPO_BUFFER_SIZE]
                   [--ppo_epochs PPO_EPOCHS]
                   [--ppo_score_norm [PPO_SCORE_NORM]]
                   [--ppo_target PPO_TARGET]
                   [--ppo_whiten_rewards [PPO_WHITEN_REWARDS]]
                   [--ref_model REF_MODEL]
                   [--ref_model_adapters REF_MODEL_ADAPTERS]
                   [--ref_model_quantization_bit REF_MODEL_QUANTIZATION_BIT]
                   [--reward_model REWARD_MODEL]
                   [--reward_model_adapters REWARD_MODEL_ADAPTERS]
                   [--reward_model_quantization_bit REWARD_MODEL_QUANTIZATION_BIT]
                   [--reward_model_type {lora,full,api}]
                   [--use_galore [USE_GALORE]] [--galore_target GALORE_TARGET]
                   [--galore_rank GALORE_RANK]
                   [--galore_update_interval GALORE_UPDATE_INTERVAL]
                   [--galore_scale GALORE_SCALE]
                   [--galore_proj_type {std,reverse_std,right,left,full}]
                   [--galore_layerwise [GALORE_LAYERWISE]]
                   [--use_apollo [USE_APOLLO]] [--apollo_target APOLLO_TARGET]
                   [--apollo_rank APOLLO_RANK]
                   [--apollo_update_interval APOLLO_UPDATE_INTERVAL]
                   [--apollo_scale APOLLO_SCALE] [--apollo_proj {svd,random}]
                   [--apollo_proj_type {std,right,left}]
                   [--apollo_scale_type {channel,tensor}]
                   [--apollo_layerwise [APOLLO_LAYERWISE]]
                   [--apollo_scale_front [APOLLO_SCALE_FRONT]]
                   [--use_badam [USE_BADAM]] [--badam_mode {layer,ratio}]
                   [--badam_start_block BADAM_START_BLOCK]
                   [--badam_switch_mode {ascending,descending,random,fixed}]
                   [--badam_switch_interval BADAM_SWITCH_INTERVAL]
                   [--badam_update_ratio BADAM_UPDATE_RATIO]
                   [--badam_mask_mode {adjacent,scatter}]
                   [--badam_verbose BADAM_VERBOSE]
                   [--use_swanlab [USE_SWANLAB]]
                   [--swanlab_project SWANLAB_PROJECT]
                   [--swanlab_workspace SWANLAB_WORKSPACE]
                   [--swanlab_run_name SWANLAB_RUN_NAME]
                   [--swanlab_mode {cloud,local}]
                   [--swanlab_api_key SWANLAB_API_KEY]
                   [--swanlab_logdir SWANLAB_LOGDIR]
                   [--swanlab_lark_webhook_url SWANLAB_LARK_WEBHOOK_URL]
                   [--swanlab_lark_secret SWANLAB_LARK_SECRET]
                   [--pure_bf16 [PURE_BF16]] [--stage {pt,sft,rm,ppo,dpo,kto}]
                   [--finetuning_type {lora,freeze,full}]
                   [--use_llama_pro [USE_LLAMA_PRO]]
                   [--use_adam_mini [USE_ADAM_MINI]] [--use_muon [USE_MUON]]
                   [--freeze_vision_tower [FREEZE_VISION_TOWER]]
                   [--no_freeze_vision_tower]
                   [--freeze_multi_modal_projector [FREEZE_MULTI_MODAL_PROJECTOR]]
                   [--no_freeze_multi_modal_projector]
                   [--freeze_language_model [FREEZE_LANGUAGE_MODEL]]
                   [--compute_accuracy [COMPUTE_ACCURACY]]
                   [--disable_shuffling [DISABLE_SHUFFLING]]
                   [--early_stopping_steps EARLY_STOPPING_STEPS]
                   [--plot_loss [PLOT_LOSS]]
                   [--include_effective_tokens_per_second [INCLUDE_EFFECTIVE_TOKENS_PER_SECOND]]
                   [--do_sample [DO_SAMPLE]] [--no_do_sample]
                   [--temperature TEMPERATURE] [--top_p TOP_P] [--top_k TOP_K]
                   [--num_beams NUM_BEAMS] [--max_length MAX_LENGTH]
                   [--max_new_tokens MAX_NEW_TOKENS]
                   [--repetition_penalty REPETITION_PENALTY]
                   [--length_penalty LENGTH_PENALTY]
                   [--default_system DEFAULT_SYSTEM]
                   [--skip_special_tokens [SKIP_SPECIAL_TOKENS]]
                   [--no_skip_special_tokens]

options:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH, --model-name-or-path MODEL_NAME_OR_PATH
                        Path to the model weight or identifier from
                        huggingface.co/models or modelscope.cn/models.
                        (default: None)
  --adapter_name_or_path ADAPTER_NAME_OR_PATH, --adapter-name-or-path ADAPTER_NAME_OR_PATH
                        Path to the adapter weight or identifier from
                        huggingface.co/models. Use commas to separate multiple
                        adapters. (default: None)
  --adapter_folder ADAPTER_FOLDER, --adapter-folder ADAPTER_FOLDER
                        The folder containing the adapter weights to load.
                        (default: None)
  --cache_dir CACHE_DIR, --cache-dir CACHE_DIR
                        Where to store the pre-trained models downloaded from
                        huggingface.co or modelscope.cn. (default: None)
  --use_fast_tokenizer [USE_FAST_TOKENIZER], --use-fast-tokenizer [USE_FAST_TOKENIZER]
                        Whether or not to use one of the fast tokenizer
                        (backed by the tokenizers library). (default: True)
  --no_use_fast_tokenizer, --no-use-fast-tokenizer
                        Whether or not to use one of the fast tokenizer
                        (backed by the tokenizers library). (default: False)
  --resize_vocab [RESIZE_VOCAB], --resize-vocab [RESIZE_VOCAB]
                        Whether or not to resize the tokenizer vocab and the
                        embedding layers. (default: False)
  --split_special_tokens [SPLIT_SPECIAL_TOKENS], --split-special-tokens [SPLIT_SPECIAL_TOKENS]
                        Whether or not the special tokens should be split
                        during the tokenization process. (default: False)
  --add_tokens ADD_TOKENS, --add-tokens ADD_TOKENS
                        Non-special tokens to be added into the tokenizer. Use
                        commas to separate multiple tokens. (default: None)
  --add_special_tokens ADD_SPECIAL_TOKENS, --add-special-tokens ADD_SPECIAL_TOKENS
                        Special tokens to be added into the tokenizer. Use
                        commas to separate multiple tokens. (default: None)
  --model_revision MODEL_REVISION, --model-revision MODEL_REVISION
                        The specific model version to use (can be a branch
                        name, tag name or commit id). (default: main)
  --low_cpu_mem_usage [LOW_CPU_MEM_USAGE], --low-cpu-mem-usage [LOW_CPU_MEM_USAGE]
                        Whether or not to use memory-efficient model loading.
                        (default: True)
  --no_low_cpu_mem_usage, --no-low-cpu-mem-usage
                        Whether or not to use memory-efficient model loading.
                        (default: False)
  --rope_scaling {linear,dynamic,yarn,llama3}, --rope-scaling {linear,dynamic,yarn,llama3}
                        Which scaling strategy should be adopted for the RoPE
                        embeddings. (default: None)
  --flash_attn {auto,disabled,sdpa,fa2}, --flash-attn {auto,disabled,sdpa,fa2}
                        Enable FlashAttention for faster training and
                        inference. (default: AttentionFunction.AUTO)
  --shift_attn [SHIFT_ATTN], --shift-attn [SHIFT_ATTN]
                        Enable shift short attention (S^2-Attn) proposed by
                        LongLoRA. (default: False)
  --mixture_of_depths {convert,load}, --mixture-of-depths {convert,load}
                        Convert the model to mixture-of-depths (MoD) or load
                        the MoD model. (default: None)
  --use_unsloth [USE_UNSLOTH], --use-unsloth [USE_UNSLOTH]
                        Whether or not to use unsloth's optimization for the
                        LoRA training. (default: False)
  --use_unsloth_gc [USE_UNSLOTH_GC], --use-unsloth-gc [USE_UNSLOTH_GC]
                        Whether or not to use unsloth's gradient checkpointing
                        (no need to install unsloth). (default: False)
  --enable_liger_kernel [ENABLE_LIGER_KERNEL], --enable-liger-kernel [ENABLE_LIGER_KERNEL]
                        Whether or not to enable liger kernel for faster
                        training. (default: False)
  --moe_aux_loss_coef MOE_AUX_LOSS_COEF, --moe-aux-loss-coef MOE_AUX_LOSS_COEF
                        Coefficient of the auxiliary router loss in mixture-
                        of-experts model. (default: None)
  --disable_gradient_checkpointing [DISABLE_GRADIENT_CHECKPOINTING], --disable-gradient-checkpointing [DISABLE_GRADIENT_CHECKPOINTING]
                        Whether or not to disable gradient checkpointing.
                        (default: False)
  --use_reentrant_gc [USE_REENTRANT_GC], --use-reentrant-gc [USE_REENTRANT_GC]
                        Whether or not to use reentrant gradient
                        checkpointing. (default: True)
  --no_use_reentrant_gc, --no-use-reentrant-gc
                        Whether or not to use reentrant gradient
                        checkpointing. (default: False)
  --upcast_layernorm [UPCAST_LAYERNORM], --upcast-layernorm [UPCAST_LAYERNORM]
                        Whether or not to upcast the layernorm weights in
                        fp32. (default: False)
  --upcast_lmhead_output [UPCAST_LMHEAD_OUTPUT], --upcast-lmhead-output [UPCAST_LMHEAD_OUTPUT]
                        Whether or not to upcast the output of lm_head in
                        fp32. (default: False)
  --train_from_scratch [TRAIN_FROM_SCRATCH], --train-from-scratch [TRAIN_FROM_SCRATCH]
                        Whether or not to randomly initialize the model
                        weights. (default: False)
  --infer_backend {huggingface,vllm,sglang}, --infer-backend {huggingface,vllm,sglang}
                        Backend engine used at inference. (default:
                        EngineName.HF)
  --offload_folder OFFLOAD_FOLDER, --offload-folder OFFLOAD_FOLDER
                        Path to offload model weights. (default: offload)
  --use_cache [USE_CACHE], --use-cache [USE_CACHE]
                        Whether or not to use KV cache in generation.
                        (default: True)
  --no_use_cache, --no-use-cache
                        Whether or not to use KV cache in generation.
                        (default: False)
  --infer_dtype {auto,float16,bfloat16,float32}, --infer-dtype {auto,float16,bfloat16,float32}
                        Data type for model weights and activations at
                        inference. (default: auto)
  --hf_hub_token HF_HUB_TOKEN, --hf-hub-token HF_HUB_TOKEN
                        Auth token to log in with Hugging Face Hub. (default:
                        None)
  --ms_hub_token MS_HUB_TOKEN, --ms-hub-token MS_HUB_TOKEN
                        Auth token to log in with ModelScope Hub. (default:
                        None)
  --om_hub_token OM_HUB_TOKEN, --om-hub-token OM_HUB_TOKEN
                        Auth token to log in with Modelers Hub. (default:
                        None)
  --print_param_status [PRINT_PARAM_STATUS], --print-param-status [PRINT_PARAM_STATUS]
                        For debugging purposes, print the status of the
                        parameters in the model. (default: False)
  --trust_remote_code [TRUST_REMOTE_CODE], --trust-remote-code [TRUST_REMOTE_CODE]
                        Whether to trust the execution of code from
                        datasets/models defined on the Hub or not. (default:
                        False)
  --quantization_method {bnb,gptq,awq,aqlm,quanto,eetq,hqq}, --quantization-method {bnb,gptq,awq,aqlm,quanto,eetq,hqq}
                        Quantization method to use for on-the-fly
                        quantization. (default: QuantizationMethod.BNB)
  --quantization_bit QUANTIZATION_BIT, --quantization-bit QUANTIZATION_BIT
                        The number of bits to quantize the model using on-the-
                        fly quantization. (default: None)
  --quantization_type {fp4,nf4}, --quantization-type {fp4,nf4}
                        Quantization data type to use in bitsandbytes int4
                        training. (default: nf4)
  --double_quantization [DOUBLE_QUANTIZATION], --double-quantization [DOUBLE_QUANTIZATION]
                        Whether or not to use double quantization in
                        bitsandbytes int4 training. (default: True)
  --no_double_quantization, --no-double-quantization
                        Whether or not to use double quantization in
                        bitsandbytes int4 training. (default: False)
  --quantization_device_map {auto}, --quantization-device-map {auto}
                        Device map used to infer the 4-bit quantized model,
                        needs bitsandbytes>=0.43.0. (default: None)
  --image_max_pixels IMAGE_MAX_PIXELS, --image-max-pixels IMAGE_MAX_PIXELS
                        The maximum number of pixels of image inputs.
                        (default: 589824)
  --image_min_pixels IMAGE_MIN_PIXELS, --image-min-pixels IMAGE_MIN_PIXELS
                        The minimum number of pixels of image inputs.
                        (default: 1024)
  --image_do_pan_and_scan [IMAGE_DO_PAN_AND_SCAN], --image-do-pan-and-scan [IMAGE_DO_PAN_AND_SCAN]
                        Use pan and scan to process image for gemma3.
                        (default: False)
  --crop_to_patches [CROP_TO_PATCHES], --crop-to-patches [CROP_TO_PATCHES]
                        Whether to crop the image to patches for internvl.
                        (default: False)
  --use_audio_in_video [USE_AUDIO_IN_VIDEO], --use-audio-in-video [USE_AUDIO_IN_VIDEO]
                        Whether or not to use audio in video inputs. (default:
                        False)
  --video_max_pixels VIDEO_MAX_PIXELS, --video-max-pixels VIDEO_MAX_PIXELS
                        The maximum number of pixels of video inputs.
                        (default: 65536)
  --video_min_pixels VIDEO_MIN_PIXELS, --video-min-pixels VIDEO_MIN_PIXELS
                        The minimum number of pixels of video inputs.
                        (default: 256)
  --video_fps VIDEO_FPS, --video-fps VIDEO_FPS
                        The frames to sample per second for video inputs.
                        (default: 2.0)
  --video_maxlen VIDEO_MAXLEN, --video-maxlen VIDEO_MAXLEN
                        The maximum number of sampled frames for video inputs.
                        (default: 128)
  --audio_sampling_rate AUDIO_SAMPLING_RATE, --audio-sampling-rate AUDIO_SAMPLING_RATE
                        The sampling rate of audio inputs. (default: 16000)
  --export_dir EXPORT_DIR, --export-dir EXPORT_DIR
                        Path to the directory to save the exported model.
                        (default: None)
  --export_size EXPORT_SIZE, --export-size EXPORT_SIZE
                        The file shard size (in GB) of the exported model.
                        (default: 5)
  --export_device {cpu,auto}, --export-device {cpu,auto}
                        The device used in model export, use `auto` to
                        accelerate exporting. (default: cpu)
  --export_quantization_bit EXPORT_QUANTIZATION_BIT, --export-quantization-bit EXPORT_QUANTIZATION_BIT
                        The number of bits to quantize the exported model.
                        (default: None)
  --export_quantization_dataset EXPORT_QUANTIZATION_DATASET, --export-quantization-dataset EXPORT_QUANTIZATION_DATASET
                        Path to the dataset or dataset name to use in
                        quantizing the exported model. (default: None)
  --export_quantization_nsamples EXPORT_QUANTIZATION_NSAMPLES, --export-quantization-nsamples EXPORT_QUANTIZATION_NSAMPLES
                        The number of samples used for quantization. (default:
                        128)
  --export_quantization_maxlen EXPORT_QUANTIZATION_MAXLEN, --export-quantization-maxlen EXPORT_QUANTIZATION_MAXLEN
                        The maximum length of the model inputs used for
                        quantization. (default: 1024)
  --export_legacy_format [EXPORT_LEGACY_FORMAT], --export-legacy-format [EXPORT_LEGACY_FORMAT]
                        Whether or not to save the `.bin` files instead of
                        `.safetensors`. (default: False)
  --export_hub_model_id EXPORT_HUB_MODEL_ID, --export-hub-model-id EXPORT_HUB_MODEL_ID
                        The name of the repository if push the model to the
                        Hugging Face hub. (default: None)
  --vllm_maxlen VLLM_MAXLEN, --vllm-maxlen VLLM_MAXLEN
                        Maximum sequence (prompt + response) length of the
                        vLLM engine. (default: 4096)
  --vllm_gpu_util VLLM_GPU_UTIL, --vllm-gpu-util VLLM_GPU_UTIL
                        The fraction of GPU memory in (0,1) to be used for the
                        vLLM engine. (default: 0.7)
  --vllm_enforce_eager [VLLM_ENFORCE_EAGER], --vllm-enforce-eager [VLLM_ENFORCE_EAGER]
                        Whether or not to disable CUDA graph in the vLLM
                        engine. (default: False)
  --vllm_max_lora_rank VLLM_MAX_LORA_RANK, --vllm-max-lora-rank VLLM_MAX_LORA_RANK
                        Maximum rank of all LoRAs in the vLLM engine.
                        (default: 32)
  --vllm_config VLLM_CONFIG, --vllm-config VLLM_CONFIG
                        Config to initialize the vllm engine. Please use JSON
                        strings. (default: None)
  --sglang_maxlen SGLANG_MAXLEN, --sglang-maxlen SGLANG_MAXLEN
                        Maximum sequence (prompt + response) length of the
                        SGLang engine. (default: 4096)
  --sglang_mem_fraction SGLANG_MEM_FRACTION, --sglang-mem-fraction SGLANG_MEM_FRACTION
                        The memory fraction (0-1) to be used for the SGLang
                        engine. (default: 0.7)
  --sglang_tp_size SGLANG_TP_SIZE, --sglang-tp-size SGLANG_TP_SIZE
                        Tensor parallel size for the SGLang engine. (default:
                        -1)
  --sglang_config SGLANG_CONFIG, --sglang-config SGLANG_CONFIG
                        Config to initialize the SGLang engine. Please use
                        JSON strings. (default: None)
  --template TEMPLATE   Which template to use for constructing prompts in
                        training and inference. (default: None)
  --dataset DATASET     The name of dataset(s) to use for training. Use commas
                        to separate multiple datasets. (default: None)
  --eval_dataset EVAL_DATASET, --eval-dataset EVAL_DATASET
                        The name of dataset(s) to use for evaluation. Use
                        commas to separate multiple datasets. (default: None)
  --dataset_dir DATASET_DIR, --dataset-dir DATASET_DIR
                        Path to the folder containing the datasets. (default:
                        data)
  --media_dir MEDIA_DIR, --media-dir MEDIA_DIR
                        Path to the folder containing the images, videos or
                        audios. Defaults to `dataset_dir`. (default: None)
  --cutoff_len CUTOFF_LEN, --cutoff-len CUTOFF_LEN
                        The cutoff length of the tokenized inputs in the
                        dataset. (default: 2048)
  --train_on_prompt [TRAIN_ON_PROMPT], --train-on-prompt [TRAIN_ON_PROMPT]
                        Whether or not to disable the mask on the prompt.
                        (default: False)
  --mask_history [MASK_HISTORY], --mask-history [MASK_HISTORY]
                        Whether or not to mask the history and train on the
                        last turn only. (default: False)
  --streaming [STREAMING]
                        Enable dataset streaming. (default: False)
  --buffer_size BUFFER_SIZE, --buffer-size BUFFER_SIZE
                        Size of the buffer to randomly sample examples from in
                        dataset streaming. (default: 16384)
  --mix_strategy {concat,interleave_under,interleave_over}, --mix-strategy {concat,interleave_under,interleave_over}
                        Strategy to use in dataset mixing (concat/interleave)
                        (undersampling/oversampling). (default: concat)
  --interleave_probs INTERLEAVE_PROBS, --interleave-probs INTERLEAVE_PROBS
                        Probabilities to sample data from datasets. Use commas
                        to separate multiple datasets. (default: None)
  --overwrite_cache [OVERWRITE_CACHE], --overwrite-cache [OVERWRITE_CACHE]
                        Overwrite the cached training and evaluation sets.
                        (default: False)
  --preprocessing_batch_size PREPROCESSING_BATCH_SIZE, --preprocessing-batch-size PREPROCESSING_BATCH_SIZE
                        The number of examples in one group in pre-processing.
                        (default: 1000)
  --preprocessing_num_workers PREPROCESSING_NUM_WORKERS, --preprocessing-num-workers PREPROCESSING_NUM_WORKERS
                        The number of processes to use for the pre-processing.
                        (default: None)
  --max_samples MAX_SAMPLES, --max-samples MAX_SAMPLES
                        For debugging purposes, truncate the number of
                        examples for each dataset. (default: None)
  --eval_num_beams EVAL_NUM_BEAMS, --eval-num-beams EVAL_NUM_BEAMS
                        Number of beams to use for evaluation. This argument
                        will be passed to `model.generate` (default: None)
  --ignore_pad_token_for_loss [IGNORE_PAD_TOKEN_FOR_LOSS], --ignore-pad-token-for-loss [IGNORE_PAD_TOKEN_FOR_LOSS]
                        Whether or not to ignore the tokens corresponding to
                        the pad label in loss computation. (default: True)
  --no_ignore_pad_token_for_loss, --no-ignore-pad-token-for-loss
                        Whether or not to ignore the tokens corresponding to
                        the pad label in loss computation. (default: False)
  --val_size VAL_SIZE, --val-size VAL_SIZE
                        Size of the validation set, should be an integer or a
                        float in range `[0,1)`. (default: 0.0)
  --packing PACKING     Enable sequences packing in training. Will
                        automatically enable in pre-training. (default: None)
  --neat_packing [NEAT_PACKING], --neat-packing [NEAT_PACKING]
                        Enable sequence packing without cross-attention.
                        (default: False)
  --tool_format TOOL_FORMAT, --tool-format TOOL_FORMAT
                        Tool format to use for constructing function calling
                        examples. (default: None)
  --tokenized_path TOKENIZED_PATH, --tokenized-path TOKENIZED_PATH
                        Path to save or load the tokenized datasets. If
                        tokenized_path not exists, it will save the tokenized
                        datasets. If tokenized_path exists, it will load the
                        tokenized datasets. (default: None)
  --output_dir OUTPUT_DIR, --output-dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written. Defaults to
                        'trainer_output' if not provided. (default: None)
  --overwrite_output_dir [OVERWRITE_OUTPUT_DIR], --overwrite-output-dir [OVERWRITE_OUTPUT_DIR]
                        Overwrite the content of the output directory. Use
                        this to continue training if output_dir points to a
                        checkpoint directory. (default: False)
  --do_train [DO_TRAIN], --do-train [DO_TRAIN]
                        Whether to run training. (default: False)
  --do_eval [DO_EVAL], --do-eval [DO_EVAL]
                        Whether to run eval on the dev set. (default: False)
  --do_predict [DO_PREDICT], --do-predict [DO_PREDICT]
                        Whether to run predictions on the test set. (default:
                        False)
  --eval_strategy {no,steps,epoch}, --eval-strategy {no,steps,epoch}
                        The evaluation strategy to use. (default: no)
  --prediction_loss_only [PREDICTION_LOSS_ONLY], --prediction-loss-only [PREDICTION_LOSS_ONLY]
                        When performing evaluation and predictions, only
                        returns the loss. (default: False)
  --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE, --per-device-train-batch-size PER_DEVICE_TRAIN_BATCH_SIZE
                        Batch size per device accelerator core/CPU for
                        training. (default: 8)
  --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE, --per-device-eval-batch-size PER_DEVICE_EVAL_BATCH_SIZE
                        Batch size per device accelerator core/CPU for
                        evaluation. (default: 8)
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE, --per-gpu-train-batch-size PER_GPU_TRAIN_BATCH_SIZE
                        Deprecated, the use of `--per_device_train_batch_size`
                        is preferred. Batch size per GPU/TPU core/CPU for
                        training. (default: None)
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE, --per-gpu-eval-batch-size PER_GPU_EVAL_BATCH_SIZE
                        Deprecated, the use of `--per_device_eval_batch_size`
                        is preferred. Batch size per GPU/TPU core/CPU for
                        evaluation. (default: None)
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS, --gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass. (default: 1)
  --eval_accumulation_steps EVAL_ACCUMULATION_STEPS, --eval-accumulation-steps EVAL_ACCUMULATION_STEPS
                        Number of predictions steps to accumulate before
                        moving the tensors to the CPU. (default: None)
  --eval_delay EVAL_DELAY, --eval-delay EVAL_DELAY
                        Number of epochs or steps to wait for before the first
                        evaluation can be performed, depending on the
                        eval_strategy. (default: 0)
  --torch_empty_cache_steps TORCH_EMPTY_CACHE_STEPS, --torch-empty-cache-steps TORCH_EMPTY_CACHE_STEPS
                        Number of steps to wait before calling
                        `torch.<device>.empty_cache()`.This can help avoid
                        CUDA out-of-memory errors by lowering peak VRAM usage
                        at a cost of about [10{'option_strings': ['--
                        torch_empty_cache_steps', '--torch-empty-cache-
                        steps'], 'dest': 'torch_empty_cache_steps', 'nargs':
                        None, 'const': None, 'default': None, 'type': 'int',
                        'choices': None, 'required': False, 'help': 'Number of
                        steps to wait before calling
                        `torch.<device>.empty_cache()`.This can help avoid
                        CUDA out-of-memory errors by lowering peak VRAM usage
                        at a cost of about [10% slower performance](https://gi
                        thub.com/huggingface/transformers/issues/31372).If
                        left unset or set to None, cache will not be
                        emptied.', 'metavar': None, 'container':
                        <argparse._ArgumentGroup object at 0x7f8381a2fa40>,
                        'prog': 'launcher.py'}lower performance](https://githu
                        b.com/huggingface/transformers/issues/31372).If left
                        unset or set to None, cache will not be emptied.
                        (default: None)
  --learning_rate LEARNING_RATE, --learning-rate LEARNING_RATE
                        The initial learning rate for AdamW. (default: 5e-05)
  --weight_decay WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        Weight decay for AdamW if we apply some. (default:
                        0.0)
  --adam_beta1 ADAM_BETA1, --adam-beta1 ADAM_BETA1
                        Beta1 for AdamW optimizer (default: 0.9)
  --adam_beta2 ADAM_BETA2, --adam-beta2 ADAM_BETA2
                        Beta2 for AdamW optimizer (default: 0.999)
  --adam_epsilon ADAM_EPSILON, --adam-epsilon ADAM_EPSILON
                        Epsilon for AdamW optimizer. (default: 1e-08)
  --max_grad_norm MAX_GRAD_NORM, --max-grad-norm MAX_GRAD_NORM
                        Max gradient norm. (default: 1.0)
  --num_train_epochs NUM_TRAIN_EPOCHS, --num-train-epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform. (default:
                        3.0)
  --max_steps MAX_STEPS, --max-steps MAX_STEPS
                        If > 0: set total number of training steps to perform.
                        Override num_train_epochs. (default: -1)
  --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau,cosine_with_min_lr,warmup_stable_decay}, --lr-scheduler-type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau,cosine_with_min_lr,warmup_stable_decay}
                        The scheduler type to use. (default: linear)
  --lr_scheduler_kwargs LR_SCHEDULER_KWARGS, --lr-scheduler-kwargs LR_SCHEDULER_KWARGS
                        Extra parameters for the lr_scheduler such as
                        {'num_cycles': 1} for the cosine with hard restarts.
                        (default: {})
  --warmup_ratio WARMUP_RATIO, --warmup-ratio WARMUP_RATIO
                        Linear warmup over warmup_ratio fraction of total
                        steps. (default: 0.0)
  --warmup_steps WARMUP_STEPS, --warmup-steps WARMUP_STEPS
                        Linear warmup over warmup_steps. (default: 0)
  --log_level {detail,debug,info,warning,error,critical,passive}, --log-level {detail,debug,info,warning,error,critical,passive}
                        Logger log level to use on the main node. Possible
                        choices are the log levels as strings: 'debug',
                        'info', 'warning', 'error' and 'critical', plus a
                        'passive' level which doesn't set anything and lets
                        the application set the level. Defaults to 'passive'.
                        (default: passive)
  --log_level_replica {detail,debug,info,warning,error,critical,passive}, --log-level-replica {detail,debug,info,warning,error,critical,passive}
                        Logger log level to use on replica nodes. Same choices
                        and defaults as ``log_level`` (default: warning)
  --log_on_each_node [LOG_ON_EACH_NODE], --log-on-each-node [LOG_ON_EACH_NODE]
                        When doing a multinode distributed training, whether
                        to log once per node or just once on the main node.
                        (default: True)
  --no_log_on_each_node, --no-log-on-each-node
                        When doing a multinode distributed training, whether
                        to log once per node or just once on the main node.
                        (default: False)
  --logging_dir LOGGING_DIR, --logging-dir LOGGING_DIR
                        Tensorboard log dir. (default: None)
  --logging_strategy {no,steps,epoch}, --logging-strategy {no,steps,epoch}
                        The logging strategy to use. (default: steps)
  --logging_first_step [LOGGING_FIRST_STEP], --logging-first-step [LOGGING_FIRST_STEP]
                        Log the first global_step (default: False)
  --logging_steps LOGGING_STEPS, --logging-steps LOGGING_STEPS
                        Log every X updates steps. Should be an integer or a
                        float in range `[0,1)`. If smaller than 1, will be
                        interpreted as ratio of total training steps.
                        (default: 500)
  --logging_nan_inf_filter [LOGGING_NAN_INF_FILTER], --logging-nan-inf-filter [LOGGING_NAN_INF_FILTER]
                        Filter nan and inf losses for logging. (default: True)
  --no_logging_nan_inf_filter, --no-logging-nan-inf-filter
                        Filter nan and inf losses for logging. (default:
                        False)
  --save_strategy {no,steps,epoch,best}, --save-strategy {no,steps,epoch,best}
                        The checkpoint save strategy to use. (default: steps)
  --save_steps SAVE_STEPS, --save-steps SAVE_STEPS
                        Save checkpoint every X updates steps. Should be an
                        integer or a float in range `[0,1)`. If smaller than
                        1, will be interpreted as ratio of total training
                        steps. (default: 500)
  --save_total_limit SAVE_TOTAL_LIMIT, --save-total-limit SAVE_TOTAL_LIMIT
                        If a value is passed, will limit the total amount of
                        checkpoints. Deletes the older checkpoints in
                        `output_dir`. When `load_best_model_at_end` is
                        enabled, the 'best' checkpoint according to
                        `metric_for_best_model` will always be retained in
                        addition to the most recent ones. For example, for
                        `save_total_limit=5` and
                        `load_best_model_at_end=True`, the four last
                        checkpoints will always be retained alongside the best
                        model. When `save_total_limit=1` and
                        `load_best_model_at_end=True`, it is possible that two
                        checkpoints are saved: the last one and the best one
                        (if they are different). Default is unlimited
                        checkpoints (default: None)
  --save_safetensors [SAVE_SAFETENSORS], --save-safetensors [SAVE_SAFETENSORS]
                        Use safetensors saving and loading for state dicts
                        instead of default torch.load and torch.save.
                        (default: True)
  --no_save_safetensors, --no-save-safetensors
                        Use safetensors saving and loading for state dicts
                        instead of default torch.load and torch.save.
                        (default: False)
  --save_on_each_node [SAVE_ON_EACH_NODE], --save-on-each-node [SAVE_ON_EACH_NODE]
                        When doing multi-node distributed training, whether to
                        save models and checkpoints on each node, or only on
                        the main one (default: False)
  --save_only_model [SAVE_ONLY_MODEL], --save-only-model [SAVE_ONLY_MODEL]
                        When checkpointing, whether to only save the model, or
                        also the optimizer, scheduler & rng state.Note that
                        when this is true, you won't be able to resume
                        training from checkpoint.This enables you to save
                        storage by not storing the optimizer, scheduler & rng
                        state.You can only load the model using
                        from_pretrained with this option set to True.
                        (default: False)
  --restore_callback_states_from_checkpoint [RESTORE_CALLBACK_STATES_FROM_CHECKPOINT], --restore-callback-states-from-checkpoint [RESTORE_CALLBACK_STATES_FROM_CHECKPOINT]
                        Whether to restore the callback states from the
                        checkpoint. If `True`, will override callbacks passed
                        to the `Trainer` if they exist in the checkpoint.
                        (default: False)
  --no_cuda [NO_CUDA], --no-cuda [NO_CUDA]
                        This argument is deprecated. It will be removed in
                        version 5.0 of 🤗 Transformers. (default: False)
  --use_cpu [USE_CPU], --use-cpu [USE_CPU]
                        Whether or not to use cpu. If left to False, we will
                        use the available torch device/backend
                        (cuda/mps/xpu/hpu etc.) (default: False)
  --use_mps_device [USE_MPS_DEVICE], --use-mps-device [USE_MPS_DEVICE]
                        This argument is deprecated. `mps` device will be used
                        if available similar to `cuda` device. It will be
                        removed in version 5.0 of 🤗 Transformers (default:
                        False)
  --seed SEED           Random seed that will be set at the beginning of
                        training. (default: 42)
  --data_seed DATA_SEED, --data-seed DATA_SEED
                        Random seed to be used with data samplers. (default:
                        None)
  --jit_mode_eval [JIT_MODE_EVAL], --jit-mode-eval [JIT_MODE_EVAL]
                        Whether or not to use PyTorch jit trace for inference
                        (default: False)
  --use_ipex [USE_IPEX], --use-ipex [USE_IPEX]
                        Use Intel extension for PyTorch when it is available,
                        installation: 'https://github.com/intel/intel-
                        extension-for-pytorch' (default: False)
  --bf16 [BF16]         Whether to use bf16 (mixed) precision instead of
                        32-bit. Requires Ampere or higher NVIDIA architecture
                        or using CPU (use_cpu) or Ascend NPU. This is an
                        experimental API and it may change. (default: False)
  --fp16 [FP16]         Whether to use fp16 (mixed) precision instead of
                        32-bit (default: False)
  --fp16_opt_level FP16_OPT_LEVEL, --fp16-opt-level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in
                        ['O0', 'O1', 'O2', and 'O3']. See details at
                        https://nvidia.github.io/apex/amp.html (default: O1)
  --half_precision_backend {auto,apex,cpu_amp}, --half-precision-backend {auto,apex,cpu_amp}
                        The backend to be used for half precision. (default:
                        auto)
  --bf16_full_eval [BF16_FULL_EVAL], --bf16-full-eval [BF16_FULL_EVAL]
                        Whether to use full bfloat16 evaluation instead of
                        32-bit. This is an experimental API and it may change.
                        (default: False)
  --fp16_full_eval [FP16_FULL_EVAL], --fp16-full-eval [FP16_FULL_EVAL]
                        Whether to use full float16 evaluation instead of
                        32-bit (default: False)
  --tf32 TF32           Whether to enable tf32 mode, available in Ampere and
                        newer GPU architectures. This is an experimental API
                        and it may change. (default: None)
  --local_rank LOCAL_RANK, --local-rank LOCAL_RANK
                        For distributed training: local_rank (default: -1)
  --ddp_backend {nccl,gloo,mpi,ccl,hccl,cncl,mccl}, --ddp-backend {nccl,gloo,mpi,ccl,hccl,cncl,mccl}
                        The backend to be used for distributed training
                        (default: None)
  --tpu_num_cores TPU_NUM_CORES, --tpu-num-cores TPU_NUM_CORES
                        TPU: Number of TPU cores (automatically passed by
                        launcher script) (default: None)
  --tpu_metrics_debug [TPU_METRICS_DEBUG], --tpu-metrics-debug [TPU_METRICS_DEBUG]
                        Deprecated, the use of `--debug tpu_metrics_debug` is
                        preferred. TPU: Whether to print debug metrics
                        (default: False)
  --debug DEBUG [DEBUG ...]
                        Whether or not to enable debug mode. Current options:
                        `underflow_overflow` (Detect underflow and overflow in
                        activations and weights), `tpu_metrics_debug` (print
                        debug metrics on TPU). (default: None)
  --dataloader_drop_last [DATALOADER_DROP_LAST], --dataloader-drop-last [DATALOADER_DROP_LAST]
                        Drop the last incomplete batch if it is not divisible
                        by the batch size. (default: False)
  --eval_steps EVAL_STEPS, --eval-steps EVAL_STEPS
                        Run an evaluation every X steps. Should be an integer
                        or a float in range `[0,1)`. If smaller than 1, will
                        be interpreted as ratio of total training steps.
                        (default: None)
  --dataloader_num_workers DATALOADER_NUM_WORKERS, --dataloader-num-workers DATALOADER_NUM_WORKERS
                        Number of subprocesses to use for data loading
                        (PyTorch only). 0 means that the data will be loaded
                        in the main process. (default: 0)
  --dataloader_prefetch_factor DATALOADER_PREFETCH_FACTOR, --dataloader-prefetch-factor DATALOADER_PREFETCH_FACTOR
                        Number of batches loaded in advance by each worker. 2
                        means there will be a total of 2 * num_workers batches
                        prefetched across all workers. Default is 2 for
                        PyTorch < 2.0.0 and otherwise None. (default: None)
  --past_index PAST_INDEX, --past-index PAST_INDEX
                        If >=0, uses the corresponding part of the output as
                        the past state for next step. (default: -1)
  --run_name RUN_NAME, --run-name RUN_NAME
                        An optional descriptor for the run. Notably used for
                        wandb, mlflow comet and swanlab logging. (default:
                        None)
  --disable_tqdm DISABLE_TQDM, --disable-tqdm DISABLE_TQDM
                        Whether or not to disable the tqdm progress bars.
                        (default: None)
  --remove_unused_columns [REMOVE_UNUSED_COLUMNS], --remove-unused-columns [REMOVE_UNUSED_COLUMNS]
                        Remove columns not required by the model when using an
                        nlp.Dataset. (default: True)
  --no_remove_unused_columns, --no-remove-unused-columns
                        Remove columns not required by the model when using an
                        nlp.Dataset. (default: False)
  --label_names LABEL_NAMES [LABEL_NAMES ...], --label-names LABEL_NAMES [LABEL_NAMES ...]
                        The list of keys in your dictionary of inputs that
                        correspond to the labels. (default: None)
  --load_best_model_at_end [LOAD_BEST_MODEL_AT_END], --load-best-model-at-end [LOAD_BEST_MODEL_AT_END]
                        Whether or not to load the best model found during
                        training at the end of training. When this option is
                        enabled, the best checkpoint will always be saved. See
                        `save_total_limit` for more. (default: False)
  --metric_for_best_model METRIC_FOR_BEST_MODEL, --metric-for-best-model METRIC_FOR_BEST_MODEL
                        The metric to use to compare two different models.
                        (default: None)
  --greater_is_better GREATER_IS_BETTER, --greater-is-better GREATER_IS_BETTER
                        Whether the `metric_for_best_model` should be
                        maximized or not. (default: None)
  --ignore_data_skip [IGNORE_DATA_SKIP], --ignore-data-skip [IGNORE_DATA_SKIP]
                        When resuming training, whether or not to skip the
                        first epochs and batches to get to the same training
                        data. (default: False)
  --fsdp FSDP           Whether or not to use PyTorch Fully Sharded Data
                        Parallel (FSDP) training (in distributed training
                        only). The base option should be `full_shard`,
                        `shard_grad_op` or `no_shard` and you can add CPU-
                        offload to `full_shard` or `shard_grad_op` like this:
                        full_shard offload` or `shard_grad_op offload`. You
                        can add auto-wrap to `full_shard` or `shard_grad_op`
                        with the same syntax: full_shard auto_wrap` or
                        `shard_grad_op auto_wrap`. (default: )
  --fsdp_min_num_params FSDP_MIN_NUM_PARAMS, --fsdp-min-num-params FSDP_MIN_NUM_PARAMS
                        This parameter is deprecated. FSDP's minimum number of
                        parameters for Default Auto Wrapping. (useful only
                        when `fsdp` field is passed). (default: 0)
  --fsdp_config FSDP_CONFIG, --fsdp-config FSDP_CONFIG
                        Config to be used with FSDP (Pytorch Fully Sharded
                        Data Parallel). The value is either a fsdp json config
                        file (e.g., `fsdp_config.json`) or an already loaded
                        json file as `dict`. (default: None)
  --tp_size TP_SIZE, --tp-size TP_SIZE
                        Use tp_size to enable pytorch tensor
                        parallelism.Tensor parallelism support is only
                        available to models having `base_tp_plan` in their
                        respective config classes.Set a value greater than 1
                        to activate TP.The same is used to prepare device mesh
                        internally.Requires accelerate>1.3.0. (default: 0)
  --fsdp_transformer_layer_cls_to_wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP, --fsdp-transformer-layer-cls-to-wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP
                        This parameter is deprecated. Transformer layer class
                        name (case-sensitive) to wrap, e.g, `BertLayer`,
                        `GPTJBlock`, `T5Block` .... (useful only when `fsdp`
                        flag is passed). (default: None)
  --accelerator_config ACCELERATOR_CONFIG, --accelerator-config ACCELERATOR_CONFIG
                        Config to be used with the internal Accelerator object
                        initialization. The value is either a accelerator json
                        config file (e.g., `accelerator_config.json`) or an
                        already loaded json file as `dict`. (default: None)
  --deepspeed DEEPSPEED
                        Enable deepspeed and pass the path to deepspeed json
                        config file (e.g. `ds_config.json`) or an already
                        loaded json file as a dict (default: None)
  --label_smoothing_factor LABEL_SMOOTHING_FACTOR, --label-smoothing-factor LABEL_SMOOTHING_FACTOR
                        The label smoothing epsilon to apply (zero means no
                        label smoothing). (default: 0.0)
  --optim {adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_torch_npu_fused,adamw_apex_fused,adafactor,adamw_anyprecision,adamw_torch_4bit,adamw_torch_8bit,ademamix,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,ademamix_8bit,lion_8bit,lion_32bit,paged_adamw_32bit,paged_adamw_8bit,paged_ademamix_32bit,paged_ademamix_8bit,paged_lion_32bit,paged_lion_8bit,rmsprop,rmsprop_bnb,rmsprop_bnb_8bit,rmsprop_bnb_32bit,galore_adamw,galore_adamw_8bit,galore_adafactor,galore_adamw_layerwise,galore_adamw_8bit_layerwise,galore_adafactor_layerwise,lomo,adalomo,grokadamw,schedule_free_radam,schedule_free_adamw,schedule_free_sgd,apollo_adamw,apollo_adamw_layerwise}
                        The optimizer to use. (default: adamw_torch)
  --optim_args OPTIM_ARGS, --optim-args OPTIM_ARGS
                        Optional arguments to supply to optimizer. (default:
                        None)
  --adafactor [ADAFACTOR]
                        Whether or not to replace AdamW by Adafactor.
                        (default: False)
  --group_by_length [GROUP_BY_LENGTH], --group-by-length [GROUP_BY_LENGTH]
                        Whether or not to group samples of roughly the same
                        length together when batching. (default: False)
  --length_column_name LENGTH_COLUMN_NAME, --length-column-name LENGTH_COLUMN_NAME
                        Column name with precomputed lengths to use when
                        grouping by length. (default: length)
  --report_to REPORT_TO, --report-to REPORT_TO
                        The list of integrations to report the results and
                        logs to. (default: None)
  --ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS, --ddp-find-unused-parameters DDP_FIND_UNUSED_PARAMETERS
                        When using distributed training, the value of the flag
                        `find_unused_parameters` passed to
                        `DistributedDataParallel`. (default: None)
  --ddp_bucket_cap_mb DDP_BUCKET_CAP_MB, --ddp-bucket-cap-mb DDP_BUCKET_CAP_MB
                        When using distributed training, the value of the flag
                        `bucket_cap_mb` passed to `DistributedDataParallel`.
                        (default: None)
  --ddp_broadcast_buffers DDP_BROADCAST_BUFFERS, --ddp-broadcast-buffers DDP_BROADCAST_BUFFERS
                        When using distributed training, the value of the flag
                        `broadcast_buffers` passed to
                        `DistributedDataParallel`. (default: None)
  --dataloader_pin_memory [DATALOADER_PIN_MEMORY], --dataloader-pin-memory [DATALOADER_PIN_MEMORY]
                        Whether or not to pin memory for DataLoader. (default:
                        True)
  --no_dataloader_pin_memory, --no-dataloader-pin-memory
                        Whether or not to pin memory for DataLoader. (default:
                        False)
  --dataloader_persistent_workers [DATALOADER_PERSISTENT_WORKERS], --dataloader-persistent-workers [DATALOADER_PERSISTENT_WORKERS]
                        If True, the data loader will not shut down the worker
                        processes after a dataset has been consumed once. This
                        allows to maintain the workers Dataset instances
                        alive. Can potentially speed up training, but will
                        increase RAM usage. (default: False)
  --skip_memory_metrics [SKIP_MEMORY_METRICS], --skip-memory-metrics [SKIP_MEMORY_METRICS]
                        Whether or not to skip adding of memory profiler
                        reports to metrics. (default: True)
  --no_skip_memory_metrics, --no-skip-memory-metrics
                        Whether or not to skip adding of memory profiler
                        reports to metrics. (default: False)
  --use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP], --use-legacy-prediction-loop [USE_LEGACY_PREDICTION_LOOP]
                        Whether or not to use the legacy prediction_loop in
                        the Trainer. (default: False)
  --push_to_hub [PUSH_TO_HUB], --push-to-hub [PUSH_TO_HUB]
                        Whether or not to upload the trained model to the
                        model hub after training. (default: False)
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT, --resume-from-checkpoint RESUME_FROM_CHECKPOINT
                        The path to a folder with a valid checkpoint for your
                        model. (default: None)
  --hub_model_id HUB_MODEL_ID, --hub-model-id HUB_MODEL_ID
                        The name of the repository to keep in sync with the
                        local `output_dir`. (default: None)
  --hub_strategy {end,every_save,checkpoint,all_checkpoints}, --hub-strategy {end,every_save,checkpoint,all_checkpoints}
                        The hub strategy to use when `--push_to_hub` is
                        activated. (default: every_save)
  --hub_token HUB_TOKEN, --hub-token HUB_TOKEN
                        The token to use to push to the Model Hub. (default:
                        None)
  --hub_private_repo HUB_PRIVATE_REPO, --hub-private-repo HUB_PRIVATE_REPO
                        Whether to make the repo private. If `None` (default),
                        the repo will be public unless the organization's
                        default is private. This value is ignored if the repo
                        already exists. (default: None)
  --hub_always_push [HUB_ALWAYS_PUSH], --hub-always-push [HUB_ALWAYS_PUSH]
                        Unless `True`, the Trainer will skip pushes if the
                        previous one wasn't finished yet. (default: False)
  --gradient_checkpointing [GRADIENT_CHECKPOINTING], --gradient-checkpointing [GRADIENT_CHECKPOINTING]
                        If True, use gradient checkpointing to save memory at
                        the expense of slower backward pass. (default: False)
  --gradient_checkpointing_kwargs GRADIENT_CHECKPOINTING_KWARGS, --gradient-checkpointing-kwargs GRADIENT_CHECKPOINTING_KWARGS
                        Gradient checkpointing key word arguments such as
                        `use_reentrant`. Will be passed to
                        `torch.utils.checkpoint.checkpoint` through
                        `model.gradient_checkpointing_enable`. (default: None)
  --include_inputs_for_metrics [INCLUDE_INPUTS_FOR_METRICS], --include-inputs-for-metrics [INCLUDE_INPUTS_FOR_METRICS]
                        This argument is deprecated and will be removed in
                        version 5 of 🤗 Transformers. Use `include_for_metrics`
                        instead. (default: False)
  --include_for_metrics INCLUDE_FOR_METRICS [INCLUDE_FOR_METRICS ...], --include-for-metrics INCLUDE_FOR_METRICS [INCLUDE_FOR_METRICS ...]
                        List of strings to specify additional data to include
                        in the `compute_metrics` function.Options: 'inputs',
                        'loss'. (default: [])
  --eval_do_concat_batches [EVAL_DO_CONCAT_BATCHES], --eval-do-concat-batches [EVAL_DO_CONCAT_BATCHES]
                        Whether to recursively concat
                        inputs/losses/labels/predictions across batches. If
                        `False`, will instead store them as lists, with each
                        batch kept separate. (default: True)
  --no_eval_do_concat_batches, --no-eval-do-concat-batches
                        Whether to recursively concat
                        inputs/losses/labels/predictions across batches. If
                        `False`, will instead store them as lists, with each
                        batch kept separate. (default: False)
  --fp16_backend {auto,apex,cpu_amp}, --fp16-backend {auto,apex,cpu_amp}
                        Deprecated. Use half_precision_backend instead
                        (default: auto)
  --push_to_hub_model_id PUSH_TO_HUB_MODEL_ID, --push-to-hub-model-id PUSH_TO_HUB_MODEL_ID
                        The name of the repository to which push the
                        `Trainer`. (default: None)
  --push_to_hub_organization PUSH_TO_HUB_ORGANIZATION, --push-to-hub-organization PUSH_TO_HUB_ORGANIZATION
                        The name of the organization in with to which push the
                        `Trainer`. (default: None)
  --push_to_hub_token PUSH_TO_HUB_TOKEN, --push-to-hub-token PUSH_TO_HUB_TOKEN
                        The token to use to push to the Model Hub. (default:
                        None)
  --mp_parameters MP_PARAMETERS, --mp-parameters MP_PARAMETERS
                        Used by the SageMaker launcher to send mp-specific
                        args. Ignored in Trainer (default: )
  --auto_find_batch_size [AUTO_FIND_BATCH_SIZE], --auto-find-batch-size [AUTO_FIND_BATCH_SIZE]
                        Whether to automatically decrease the batch size in
                        half and rerun the training loop again each time a
                        CUDA Out-of-Memory was reached (default: False)
  --full_determinism [FULL_DETERMINISM], --full-determinism [FULL_DETERMINISM]
                        Whether to call enable_full_determinism instead of
                        set_seed for reproducibility in distributed training.
                        Important: this will negatively impact the
                        performance, so only use it for debugging. (default:
                        False)
  --torchdynamo TORCHDYNAMO
                        This argument is deprecated, use
                        `--torch_compile_backend` instead. (default: None)
  --ray_scope RAY_SCOPE, --ray-scope RAY_SCOPE
                        The scope to use when doing hyperparameter search with
                        Ray. By default, `"last"` will be used. Ray will then
                        use the last checkpoint of all trials, compare those,
                        and select the best one. However, other options are
                        also available. See the Ray documentation (https://doc
                        s.ray.io/en/latest/tune/api_docs/analysis.html#ray.tun
                        e.ExperimentAnalysis.get_best_trial) for more options.
                        (default: last)
  --ddp_timeout DDP_TIMEOUT, --ddp-timeout DDP_TIMEOUT
                        Overrides the default timeout for distributed training
                        (value should be given in seconds). (default: 1800)
  --torch_compile [TORCH_COMPILE], --torch-compile [TORCH_COMPILE]
                        If set to `True`, the model will be wrapped in
                        `torch.compile`. (default: False)
  --torch_compile_backend TORCH_COMPILE_BACKEND, --torch-compile-backend TORCH_COMPILE_BACKEND
                        Which backend to use with `torch.compile`, passing one
                        will trigger a model compilation. (default: None)
  --torch_compile_mode TORCH_COMPILE_MODE, --torch-compile-mode TORCH_COMPILE_MODE
                        Which mode to use with `torch.compile`, passing one
                        will trigger a model compilation. (default: None)
  --include_tokens_per_second [INCLUDE_TOKENS_PER_SECOND], --include-tokens-per-second [INCLUDE_TOKENS_PER_SECOND]
                        If set to `True`, the speed metrics will include `tgs`
                        (tokens per second per device). (default: False)
  --include_num_input_tokens_seen [INCLUDE_NUM_INPUT_TOKENS_SEEN], --include-num-input-tokens-seen [INCLUDE_NUM_INPUT_TOKENS_SEEN]
                        If set to `True`, will track the number of input
                        tokens seen throughout training. (May be slower in
                        distributed training) (default: False)
  --neftune_noise_alpha NEFTUNE_NOISE_ALPHA, --neftune-noise-alpha NEFTUNE_NOISE_ALPHA
                        Activates neftune noise embeddings into the model.
                        NEFTune has been proven to drastically improve model
                        performances for instruction fine-tuning. Check out
                        the original paper here:
                        https://arxiv.org/abs/2310.05914 and the original code
                        here: https://github.com/neelsjain/NEFTune. Only
                        supported for `PreTrainedModel` and `PeftModel`
                        classes. (default: None)
  --optim_target_modules OPTIM_TARGET_MODULES, --optim-target-modules OPTIM_TARGET_MODULES
                        Target modules for the optimizer defined in the
                        `optim` argument. Only used for the GaLore optimizer
                        at the moment. (default: None)
  --batch_eval_metrics [BATCH_EVAL_METRICS], --batch-eval-metrics [BATCH_EVAL_METRICS]
                        Break eval metrics calculation into batches to save
                        memory. (default: False)
  --eval_on_start [EVAL_ON_START], --eval-on-start [EVAL_ON_START]
                        Whether to run through the entire `evaluation` step at
                        the very beginning of training as a sanity check.
                        (default: False)
  --use_liger_kernel [USE_LIGER_KERNEL], --use-liger-kernel [USE_LIGER_KERNEL]
                        Whether or not to enable the Liger Kernel for model
                        training. (default: False)
  --eval_use_gather_object [EVAL_USE_GATHER_OBJECT], --eval-use-gather-object [EVAL_USE_GATHER_OBJECT]
                        Whether to run recursively gather object in a nested
                        list/tuple/dictionary of objects from all devices.
                        (default: False)
  --average_tokens_across_devices [AVERAGE_TOKENS_ACROSS_DEVICES], --average-tokens-across-devices [AVERAGE_TOKENS_ACROSS_DEVICES]
                        Whether or not to average tokens across devices. If
                        enabled, will use all_reduce to synchronize
                        num_tokens_in_batch for precise loss calculation.
                        Reference: https://github.com/huggingface/transformers
                        /issues/34242 (default: False)
  --sortish_sampler [SORTISH_SAMPLER], --sortish-sampler [SORTISH_SAMPLER]
                        Whether to use SortishSampler or not. (default: False)
  --predict_with_generate [PREDICT_WITH_GENERATE], --predict-with-generate [PREDICT_WITH_GENERATE]
                        Whether to use generate to calculate generative
                        metrics (ROUGE, BLEU). (default: False)
  --generation_max_length GENERATION_MAX_LENGTH, --generation-max-length GENERATION_MAX_LENGTH
                        The `max_length` to use on each evaluation loop when
                        `predict_with_generate=True`. Will default to the
                        `max_length` value of the model configuration.
                        (default: None)
  --generation_num_beams GENERATION_NUM_BEAMS, --generation-num-beams GENERATION_NUM_BEAMS
                        The `num_beams` to use on each evaluation loop when
                        `predict_with_generate=True`. Will default to the
                        `num_beams` value of the model configuration.
                        (default: None)
  --generation_config GENERATION_CONFIG, --generation-config GENERATION_CONFIG
                        Model id, file path or url pointing to a
                        GenerationConfig json file, to use during prediction.
                        (default: None)
  --ray_run_name RAY_RUN_NAME, --ray-run-name RAY_RUN_NAME
                        The training results will be saved at
                        `<ray_storage_path>/ray_run_name`. (default: None)
  --ray_storage_path RAY_STORAGE_PATH, --ray-storage-path RAY_STORAGE_PATH
                        The storage path to save training results to (default:
                        ./saves)
  --ray_num_workers RAY_NUM_WORKERS, --ray-num-workers RAY_NUM_WORKERS
                        The number of workers for Ray training. Default is 1
                        worker. (default: 1)
  --resources_per_worker RESOURCES_PER_WORKER, --resources-per-worker RESOURCES_PER_WORKER
                        The resources per worker for Ray training. Default is
                        to use 1 GPU per worker. (default: {'GPU': 1})
  --placement_strategy {SPREAD,PACK,STRICT_SPREAD,STRICT_PACK}, --placement-strategy {SPREAD,PACK,STRICT_SPREAD,STRICT_PACK}
                        The placement strategy for Ray training. Default is
                        PACK. (default: PACK)
  --ray_init_kwargs RAY_INIT_KWARGS, --ray-init-kwargs RAY_INIT_KWARGS
                        The arguments to pass to ray.init for Ray training.
                        Default is None. (default: None)
  --freeze_trainable_layers FREEZE_TRAINABLE_LAYERS, --freeze-trainable-layers FREEZE_TRAINABLE_LAYERS
                        The number of trainable layers for freeze (partial-
                        parameter) fine-tuning. Positive numbers mean the last
                        n layers are set as trainable, negative numbers mean
                        the first n layers are set as trainable. (default: 2)
  --freeze_trainable_modules FREEZE_TRAINABLE_MODULES, --freeze-trainable-modules FREEZE_TRAINABLE_MODULES
                        Name(s) of trainable modules for freeze (partial-
                        parameter) fine-tuning. Use commas to separate
                        multiple modules. Use `all` to specify all the
                        available modules. (default: all)
  --freeze_extra_modules FREEZE_EXTRA_MODULES, --freeze-extra-modules FREEZE_EXTRA_MODULES
                        Name(s) of modules apart from hidden layers to be set
                        as trainable for freeze (partial-parameter) fine-
                        tuning. Use commas to separate multiple modules.
                        (default: None)
  --additional_target ADDITIONAL_TARGET, --additional-target ADDITIONAL_TARGET
                        Name(s) of modules apart from LoRA layers to be set as
                        trainable and saved in the final checkpoint. Use
                        commas to separate multiple modules. (default: None)
  --lora_alpha LORA_ALPHA, --lora-alpha LORA_ALPHA
                        The scale factor for LoRA fine-tuning (default:
                        lora_rank * 2). (default: None)
  --lora_dropout LORA_DROPOUT, --lora-dropout LORA_DROPOUT
                        Dropout rate for the LoRA fine-tuning. (default: 0.0)
  --lora_rank LORA_RANK, --lora-rank LORA_RANK
                        The intrinsic dimension for LoRA fine-tuning.
                        (default: 8)
  --lora_target LORA_TARGET, --lora-target LORA_TARGET
                        Name(s) of target modules to apply LoRA. Use commas to
                        separate multiple modules. Use `all` to specify all
                        the linear modules. (default: all)
  --loraplus_lr_ratio LORAPLUS_LR_RATIO, --loraplus-lr-ratio LORAPLUS_LR_RATIO
                        LoRA plus learning rate ratio (lr_B / lr_A). (default:
                        None)
  --loraplus_lr_embedding LORAPLUS_LR_EMBEDDING, --loraplus-lr-embedding LORAPLUS_LR_EMBEDDING
                        LoRA plus learning rate for lora embedding layers.
                        (default: 1e-06)
  --use_rslora [USE_RSLORA], --use-rslora [USE_RSLORA]
                        Whether or not to use the rank stabilization scaling
                        factor for LoRA layer. (default: False)
  --use_dora [USE_DORA], --use-dora [USE_DORA]
                        Whether or not to use the weight-decomposed lora
                        method (DoRA). (default: False)
  --pissa_init [PISSA_INIT], --pissa-init [PISSA_INIT]
                        Whether or not to initialize a PiSSA adapter.
                        (default: False)
  --pissa_iter PISSA_ITER, --pissa-iter PISSA_ITER
                        The number of iteration steps performed by FSVD in
                        PiSSA. Use -1 to disable it. (default: 16)
  --pissa_convert [PISSA_CONVERT], --pissa-convert [PISSA_CONVERT]
                        Whether or not to convert the PiSSA adapter to a
                        normal LoRA adapter. (default: False)
  --create_new_adapter [CREATE_NEW_ADAPTER], --create-new-adapter [CREATE_NEW_ADAPTER]
                        Whether or not to create a new adapter with randomly
                        initialized weight. (default: False)
  --pref_beta PREF_BETA, --pref-beta PREF_BETA
                        The beta parameter in the preference loss. (default:
                        0.1)
  --pref_ftx PREF_FTX, --pref-ftx PREF_FTX
                        The supervised fine-tuning loss coefficient in DPO
                        training. (default: 0.0)
  --pref_loss {sigmoid,hinge,ipo,kto_pair,orpo,simpo}, --pref-loss {sigmoid,hinge,ipo,kto_pair,orpo,simpo}
                        The type of DPO loss to use. (default: sigmoid)
  --dpo_label_smoothing DPO_LABEL_SMOOTHING, --dpo-label-smoothing DPO_LABEL_SMOOTHING
                        The robust DPO label smoothing parameter in cDPO that
                        should be between 0 and 0.5. (default: 0.0)
  --kto_chosen_weight KTO_CHOSEN_WEIGHT, --kto-chosen-weight KTO_CHOSEN_WEIGHT
                        The weight factor of the desirable losses in KTO
                        training. (default: 1.0)
  --kto_rejected_weight KTO_REJECTED_WEIGHT, --kto-rejected-weight KTO_REJECTED_WEIGHT
                        The weight factor of the undesirable losses in KTO
                        training. (default: 1.0)
  --simpo_gamma SIMPO_GAMMA, --simpo-gamma SIMPO_GAMMA
                        The target reward margin term in SimPO loss. (default:
                        0.5)
  --ppo_buffer_size PPO_BUFFER_SIZE, --ppo-buffer-size PPO_BUFFER_SIZE
                        The number of mini-batches to make experience buffer
                        in a PPO optimization step. (default: 1)
  --ppo_epochs PPO_EPOCHS, --ppo-epochs PPO_EPOCHS
                        The number of epochs to perform in a PPO optimization
                        step. (default: 4)
  --ppo_score_norm [PPO_SCORE_NORM], --ppo-score-norm [PPO_SCORE_NORM]
                        Use score normalization in PPO training. (default:
                        False)
  --ppo_target PPO_TARGET, --ppo-target PPO_TARGET
                        Target KL value for adaptive KL control in PPO
                        training. (default: 6.0)
  --ppo_whiten_rewards [PPO_WHITEN_REWARDS], --ppo-whiten-rewards [PPO_WHITEN_REWARDS]
                        Whiten the rewards before compute advantages in PPO
                        training. (default: False)
  --ref_model REF_MODEL, --ref-model REF_MODEL
                        Path to the reference model used for the PPO or DPO
                        training. (default: None)
  --ref_model_adapters REF_MODEL_ADAPTERS, --ref-model-adapters REF_MODEL_ADAPTERS
                        Path to the adapters of the reference model. (default:
                        None)
  --ref_model_quantization_bit REF_MODEL_QUANTIZATION_BIT, --ref-model-quantization-bit REF_MODEL_QUANTIZATION_BIT
                        The number of bits to quantize the reference model.
                        (default: None)
  --reward_model REWARD_MODEL, --reward-model REWARD_MODEL
                        Path to the reward model used for the PPO training.
                        (default: None)
  --reward_model_adapters REWARD_MODEL_ADAPTERS, --reward-model-adapters REWARD_MODEL_ADAPTERS
                        Path to the adapters of the reward model. (default:
                        None)
  --reward_model_quantization_bit REWARD_MODEL_QUANTIZATION_BIT, --reward-model-quantization-bit REWARD_MODEL_QUANTIZATION_BIT
                        The number of bits to quantize the reward model.
                        (default: None)
  --reward_model_type {lora,full,api}, --reward-model-type {lora,full,api}
                        The type of the reward model in PPO training. Lora
                        model only supports lora training. (default: lora)
  --use_galore [USE_GALORE], --use-galore [USE_GALORE]
                        Whether or not to use the gradient low-Rank projection
                        (GaLore). (default: False)
  --galore_target GALORE_TARGET, --galore-target GALORE_TARGET
                        Name(s) of modules to apply GaLore. Use commas to
                        separate multiple modules. Use `all` to specify all
                        the linear modules. (default: all)
  --galore_rank GALORE_RANK, --galore-rank GALORE_RANK
                        The rank of GaLore gradients. (default: 16)
  --galore_update_interval GALORE_UPDATE_INTERVAL, --galore-update-interval GALORE_UPDATE_INTERVAL
                        Number of steps to update the GaLore projection.
                        (default: 200)
  --galore_scale GALORE_SCALE, --galore-scale GALORE_SCALE
                        GaLore scaling coefficient. (default: 2.0)
  --galore_proj_type {std,reverse_std,right,left,full}, --galore-proj-type {std,reverse_std,right,left,full}
                        Type of GaLore projection. (default: std)
  --galore_layerwise [GALORE_LAYERWISE], --galore-layerwise [GALORE_LAYERWISE]
                        Whether or not to enable layer-wise update to further
                        save memory. (default: False)
  --use_apollo [USE_APOLLO], --use-apollo [USE_APOLLO]
                        Whether or not to use the APOLLO optimizer. (default:
                        False)
  --apollo_target APOLLO_TARGET, --apollo-target APOLLO_TARGET
                        Name(s) of modules to apply APOLLO. Use commas to
                        separate multiple modules. Use `all` to specify all
                        the linear modules. (default: all)
  --apollo_rank APOLLO_RANK, --apollo-rank APOLLO_RANK
                        The rank of APOLLO gradients. (default: 16)
  --apollo_update_interval APOLLO_UPDATE_INTERVAL, --apollo-update-interval APOLLO_UPDATE_INTERVAL
                        Number of steps to update the APOLLO projection.
                        (default: 200)
  --apollo_scale APOLLO_SCALE, --apollo-scale APOLLO_SCALE
                        APOLLO scaling coefficient. (default: 32.0)
  --apollo_proj {svd,random}, --apollo-proj {svd,random}
                        Type of APOLLO low-rank projection algorithm (svd or
                        random). (default: random)
  --apollo_proj_type {std,right,left}, --apollo-proj-type {std,right,left}
                        Type of APOLLO projection. (default: std)
  --apollo_scale_type {channel,tensor}, --apollo-scale-type {channel,tensor}
                        Type of APOLLO scaling (channel or tensor). (default:
                        channel)
  --apollo_layerwise [APOLLO_LAYERWISE], --apollo-layerwise [APOLLO_LAYERWISE]
                        Whether or not to enable layer-wise update to further
                        save memory. (default: False)
  --apollo_scale_front [APOLLO_SCALE_FRONT], --apollo-scale-front [APOLLO_SCALE_FRONT]
                        Whether or not to use the norm-growth limiter in front
                        of gradient scaling. (default: False)
  --use_badam [USE_BADAM], --use-badam [USE_BADAM]
                        Whether or not to use the BAdam optimizer. (default:
                        False)
  --badam_mode {layer,ratio}, --badam-mode {layer,ratio}
                        Whether to use layer-wise or ratio-wise BAdam
                        optimizer. (default: layer)
  --badam_start_block BADAM_START_BLOCK, --badam-start-block BADAM_START_BLOCK
                        The starting block index for layer-wise BAdam.
                        (default: None)
  --badam_switch_mode {ascending,descending,random,fixed}, --badam-switch-mode {ascending,descending,random,fixed}
                        the strategy of picking block to update for layer-wise
                        BAdam. (default: ascending)
  --badam_switch_interval BADAM_SWITCH_INTERVAL, --badam-switch-interval BADAM_SWITCH_INTERVAL
                        Number of steps to update the block for layer-wise
                        BAdam. Use -1 to disable the block update. (default:
                        50)
  --badam_update_ratio BADAM_UPDATE_RATIO, --badam-update-ratio BADAM_UPDATE_RATIO
                        The ratio of the update for ratio-wise BAdam.
                        (default: 0.05)
  --badam_mask_mode {adjacent,scatter}, --badam-mask-mode {adjacent,scatter}
                        The mode of the mask for BAdam optimizer. `adjacent`
                        means that the trainable parameters are adjacent to
                        each other, `scatter` means that trainable parameters
                        are randomly choosed from the weight. (default:
                        adjacent)
  --badam_verbose BADAM_VERBOSE, --badam-verbose BADAM_VERBOSE
                        The verbosity level of BAdam optimizer. 0 for no
                        print, 1 for print the block prefix, 2 for print
                        trainable parameters. (default: 0)
  --use_swanlab [USE_SWANLAB], --use-swanlab [USE_SWANLAB]
                        Whether or not to use the SwanLab (an experiment
                        tracking and visualization tool). (default: False)
  --swanlab_project SWANLAB_PROJECT, --swanlab-project SWANLAB_PROJECT
                        The project name in SwanLab. (default: llamafactory)
  --swanlab_workspace SWANLAB_WORKSPACE, --swanlab-workspace SWANLAB_WORKSPACE
                        The workspace name in SwanLab. (default: None)
  --swanlab_run_name SWANLAB_RUN_NAME, --swanlab-run-name SWANLAB_RUN_NAME
                        The experiment name in SwanLab. (default: None)
  --swanlab_mode {cloud,local}, --swanlab-mode {cloud,local}
                        The mode of SwanLab. (default: cloud)
  --swanlab_api_key SWANLAB_API_KEY, --swanlab-api-key SWANLAB_API_KEY
                        The API key for SwanLab. (default: None)
  --swanlab_logdir SWANLAB_LOGDIR, --swanlab-logdir SWANLAB_LOGDIR
                        The log directory for SwanLab. (default: None)
  --swanlab_lark_webhook_url SWANLAB_LARK_WEBHOOK_URL, --swanlab-lark-webhook-url SWANLAB_LARK_WEBHOOK_URL
                        The Lark(飞书) webhook URL for SwanLab. (default: None)
  --swanlab_lark_secret SWANLAB_LARK_SECRET, --swanlab-lark-secret SWANLAB_LARK_SECRET
                        The Lark(飞书) secret for SwanLab. (default: None)
  --pure_bf16 [PURE_BF16], --pure-bf16 [PURE_BF16]
                        Whether or not to train model in purely bf16 precision
                        (without AMP). (default: False)
  --stage {pt,sft,rm,ppo,dpo,kto}
                        Which stage will be performed in training. (default:
                        sft)
  --finetuning_type {lora,freeze,full}, --finetuning-type {lora,freeze,full}
                        Which fine-tuning method to use. (default: lora)
  --use_llama_pro [USE_LLAMA_PRO], --use-llama-pro [USE_LLAMA_PRO]
                        Whether or not to make only the parameters in the
                        expanded blocks trainable. (default: False)
  --use_adam_mini [USE_ADAM_MINI], --use-adam-mini [USE_ADAM_MINI]
                        Whether or not to use the Adam-mini optimizer.
                        (default: False)
  --use_muon [USE_MUON], --use-muon [USE_MUON]
                        Whether or not to use the Muon optimizer. (default:
                        False)
  --freeze_vision_tower [FREEZE_VISION_TOWER], --freeze-vision-tower [FREEZE_VISION_TOWER]
                        Whether ot not to freeze the vision tower in MLLM
                        training. (default: True)
  --no_freeze_vision_tower, --no-freeze-vision-tower
                        Whether ot not to freeze the vision tower in MLLM
                        training. (default: False)
  --freeze_multi_modal_projector [FREEZE_MULTI_MODAL_PROJECTOR], --freeze-multi-modal-projector [FREEZE_MULTI_MODAL_PROJECTOR]
                        Whether or not to freeze the multi modal projector in
                        MLLM training. (default: True)
  --no_freeze_multi_modal_projector, --no-freeze-multi-modal-projector
                        Whether or not to freeze the multi modal projector in
                        MLLM training. (default: False)
  --freeze_language_model [FREEZE_LANGUAGE_MODEL], --freeze-language-model [FREEZE_LANGUAGE_MODEL]
                        Whether or not to freeze the language model in MLLM
                        training. (default: False)
  --compute_accuracy [COMPUTE_ACCURACY], --compute-accuracy [COMPUTE_ACCURACY]
                        Whether or not to compute the token-level accuracy at
                        evaluation. (default: False)
  --disable_shuffling [DISABLE_SHUFFLING], --disable-shuffling [DISABLE_SHUFFLING]
                        Whether or not to disable the shuffling of the
                        training set. (default: False)
  --early_stopping_steps EARLY_STOPPING_STEPS, --early-stopping-steps EARLY_STOPPING_STEPS
                        Number of steps to stop training if the
                        `metric_for_best_model` does not improve. (default:
                        None)
  --plot_loss [PLOT_LOSS], --plot-loss [PLOT_LOSS]
                        Whether or not to save the training loss curves.
                        (default: False)
  --include_effective_tokens_per_second [INCLUDE_EFFECTIVE_TOKENS_PER_SECOND], --include-effective-tokens-per-second [INCLUDE_EFFECTIVE_TOKENS_PER_SECOND]
                        Whether or not to compute effective tokens per second.
                        (default: False)
  --do_sample [DO_SAMPLE], --do-sample [DO_SAMPLE]
                        Whether or not to use sampling, use greedy decoding
                        otherwise. (default: True)
  --no_do_sample, --no-do-sample
                        Whether or not to use sampling, use greedy decoding
                        otherwise. (default: False)
  --temperature TEMPERATURE
                        The value used to modulate the next token
                        probabilities. (default: 0.95)
  --top_p TOP_P, --top-p TOP_P
                        The smallest set of most probable tokens with
                        probabilities that add up to top_p or higher are kept.
                        (default: 0.7)
  --top_k TOP_K, --top-k TOP_K
                        The number of highest probability vocabulary tokens to
                        keep for top-k filtering. (default: 50)
  --num_beams NUM_BEAMS, --num-beams NUM_BEAMS
                        Number of beams for beam search. 1 means no beam
                        search. (default: 1)
  --max_length MAX_LENGTH, --max-length MAX_LENGTH
                        The maximum length the generated tokens can have. It
                        can be overridden by max_new_tokens. (default: 1024)
  --max_new_tokens MAX_NEW_TOKENS, --max-new-tokens MAX_NEW_TOKENS
                        The maximum numbers of tokens to generate, ignoring
                        the number of tokens in the prompt. (default: 1024)
  --repetition_penalty REPETITION_PENALTY, --repetition-penalty REPETITION_PENALTY
                        The parameter for repetition penalty. 1.0 means no
                        penalty. (default: 1.0)
  --length_penalty LENGTH_PENALTY, --length-penalty LENGTH_PENALTY
                        Exponential penalty to the length that is used with
                        beam-based generation. (default: 1.0)
  --default_system DEFAULT_SYSTEM, --default-system DEFAULT_SYSTEM
                        Default system message to use in chat completion.
                        (default: None)
  --skip_special_tokens [SKIP_SPECIAL_TOKENS], --skip-special-tokens [SKIP_SPECIAL_TOKENS]
                        Whether or not to remove special tokens in the
                        decoding. (default: True)
  --no_skip_special_tokens, --no-skip-special-tokens
                        Whether or not to remove special tokens in the
                        decoding. (default: False)

```