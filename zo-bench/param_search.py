import os

#IMPORTANT!!!
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import random

import wandb
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForTokenClassification
)

from metrics import calculate_metric
from modeling_mistral import (
    MistralForCausalLM,
    MistralConfig
)
from tasks import get_task
from trainer import OurTrainer
from utils import *

os.environ["TRANSFORMERS_CACHE"] = "./cache"
#!tried to get rid of the warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AutoConfig.register("mistral", MistralConfig)
AutoModelForCausalLM.register(MistralConfig, MistralForCausalLM)

@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2"  # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP

    # Number of examples
    num_train: int = 0  # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None  # (only enabled with training) number of development samples
    num_eval: int = None  # number of evaluation samples
    num_train_sets: int = None  # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = 0  # designated seed to sample training samples/demos
    result_file: str = None  # file name for saving performance; if None, then use the task name, model name, and config

    # Model loading
    model_name: str = "facebook/opt-125m"  # HuggingFace model name
    load_float16: bool = False  # load model parameters as float16
    load_bfloat16: bool = False  # load model parameters as bfloat16
    load_int8: bool = False  # load model parameters as int8
    max_length: int = 2048  # max length the model can take
    no_auto_device: bool = False  # do not load model by auto device; should turn this on when using FSDP

    # Calibration
    sfc: bool = False  # whether to use SFC calibration
    icl_sfc: bool = False  # whether to use SFC calibration for ICL samples

    template_ver: int = 0  # template. For some tasks (SST2, RTE, Copa), we add template ver=1 as the empty template.

    # Training
    trainer: str = "none"
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo_sgd: zeroth-order SGD (MeZO) training
    ## - zo_conserv: zeroth-order SGD conservative training
    ## - zo_adam: zeroth-order Adam training
    ## - zo_sign_opt: zeroth-order sign sgd training
    ## - forward_grad: forward gradient
    num_cycles: int = 0 #option for cosine_with_restarts scheduler

    optimizer: str = "adamw_torch"
    ## options
    ## - sgd
    ## - adam
    ## - adamw # this is huggingface default
    only_train_option: bool = True  # whether to only train the option part of the input
    train_as_classification: bool = False  # take the log likelihood of all options and train as classification
    momentum: float = 0.0  # only work for SGD optimizer

    #^ added some parameters
    distribution: str = "normal" #distribution law for stochastic algorithms
    restart_rate: int = 10**9 #reset iteration count every reset_rate steps
    #options for DS optimizer
    do_reset: bool = False
    aggressive: bool = False
    adaptable: bool = False

    # MeZO
    zo_eps: float = 1e-3  # eps in MeZO
    perturbation_mode: str = "two_side"
    q: int = 1  # number of Gaussian samples for zeroth-order trainers

    # Prefix tuning
    prefix_tuning: bool = False  # whether to use prefix tuning
    num_prefix: int = 5  # number of prefixes to use
    no_reparam: bool = True  # do not use reparameterization trick
    prefix_init_by_real_act: bool = True  # initialize prefix by real activations of random words

    # prompt tuning hyperparameters
    prompt_tuning: bool = False  # whether to use prompt tuning
    num_virtual_tokens: int = 10  # number of prompt tokens to use
    prompt_init_by_real_tokens: bool = False  # whether to sample random tokens from Embedding layer

    # LoRA
    lora: bool = False  # whether to use LoRA
    lora_alpha: int = 16  # alpha in LoRA
    lora_r: int = 8  # r in LoRA

    # Generation
    sampling: bool = False  # whether to use sampling
    temperature: float = 1.0  # temperature for generation
    num_beams: int = 1  # number of beams for generation
    top_k: int = None  # top-k for generation
    top_p: float = 0.95  # top-p for generation
    max_new_tokens: int = 50  # max number of new tokens to generate
    eos_token: str = "\n"  # end of sentence token

    # Saving
    save_model: bool = False  # whether to save the model
    no_eval: bool = False  # whether to skip evaluation
    tag: str = ""  # saving tag

    # Linear probing
    linear_probing: bool = False  # whether to do linear probing
    lp_early_stopping: bool = False  # whether to do early stopping in linear probing
    head_tuning: bool = False  # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False  # untie the embeddings and LM head

    # Display
    verbose: bool = False  # verbose output

    # Non-diff objective
    non_diff: bool = False  # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False  # save model when interrupted (useful for long training)

    clean_model_at_end: bool = True  # remove everthing at the end.

    # sparse gradient pruning
    gradient_sparsity: float = None
    sparse_gradient_resample_steps: int = 1
    sparse_gradient_group: str = "layer"
    """
    Options
    ## - global: global sparsity will assign different sparsity to each layer, based on the pretrained weight magnitude
    ## - layer: each layer has the same sparsity
    """

    # module-wise perturbation
    module_wise_perturbation: bool = False
    perturbed_module_level: str = "transformer-block"
    coordinate_perturbation: bool = True  # If True, will update weight right after the gradient is computed
    """
    Options
    ## - transformer-block: perturb one transformer block at a time
    ## - mlp-attn: perturb one mlp/attention layer at a time
    ## - linear: perturb one linear layer at a time
    """


class EvaluationCallback:
    def __init__(self, args, task, eval_samples, dev_samples):
        self.args = args
        self.task = task
        self.samples = {
            "eval": eval_samples,
            "dev": dev_samples
        }

    def forward(self, model, tokenizer, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(model.device)

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature,
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)),
                num_return_sequences=1,
                eos_token_id=[
                    tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                    tokenizer.eos_token_id
                ]
            )
            # For generation, directly return the text output
            output_text = tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                model.eval()
                logits = model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1]
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()

            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]

    def one_step_pred(self, model, tokenizer, eval_sample):
        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task,
            self.task.get_template(template_version=self.args.template_ver),
            [], eval_sample,
            tokenizer, max_length=self.args.max_length,
            generation=self.task.generation,
            max_new_tokens=self.args.max_new_tokens
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(
                self.task, self.task.get_template(template_version=self.args.template_ver),
                [], eval_sample,
                tokenizer, max_length=self.args.max_length,
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc,
                generation=self.task.generation,
                max_new_tokens=self.args.max_new_tokens
            )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(model, tokenizer, encoded_candidates[0], generation=True)
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(model, tokenizer, encoded_candidate, option_len=option_lens[candidate_id])

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(
                        model, tokenizer,
                        sfc_encoded_candidates[candidate_id],
                        option_len=sfc_option_lens[candidate_id]
                    )

                outputs.append({
                    "log_probs": selected_log_probs,
                    "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None
                })

            if self.args.sfc or self.args.icl_sfc:
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))

    def evaluate(self, model, tokenizer, split="eval"):
        # Prediction loop
        predictions = []
        for id, sample in enumerate(tqdm(self.samples[split], desc=f"Evaluating on {split} set")):
            predictions.append(
                self.one_step_pred(model, tokenizer, sample)
            )

        # Calculate metrics 
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import optuna

#with seed 52 starts at accuracy 0.5160550458715596
if __name__ == "__main__":
    args = parse_args()

    if args.prefix_tuning:
        from prefix_tuning import PrefixTuning
        args.mode = "prefix"
    elif args.lora:
        from lora import LoRA
        args.mode = "lora"
    elif args.prompt_tuning:
        from prompt_tuning import PromptTuning
        args.mode = "prompt"
    else:
        args.mode = "ft"

    set_seed(args.seed)
    task = get_task(args.task_name)

    ## Load datasets and prepare them ##
    train_sets = task.sample_train_sets(
        num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval,
        num_train_sets=args.num_train_sets, seed=args.train_set_seed
    )

    assert len(train_sets) == 1, "Multiple training sets are not supported"
    assert args.train_set_seed is not None, "Train set seed must be specified"

    # Select one and only train set
    train_samples = train_sets[0]

    # Sample eval samples
    if args.num_eval is not None:
        eval_samples = task.sample_subset(data_split="valid", seed=args.train_set_seed, num=args.num_eval)
    else:
        eval_samples = task.valid_samples

    # Here the training samples are seperated
    if args.num_dev is not None:
        # Dev samples
        dev_samples = train_samples[-args.num_dev:]
        train_samples = train_samples[:-args.num_dev]
    else:
        dev_samples = None

    args.dev_samples = dev_samples
    args.eval_samples = eval_samples

    ## Load tokenizer and set it up for use ##
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    if "opt" in args.model_name:
        tokenizer.bos_token_id = 0
    if ("llama" in args.model_name) or ("mistral" in args.model_name.lower()):
        tokenizer.pad_token_id = 0

    tokenizer.padding_side = "left"

    class HFDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    def _convert(samples):
        """
        Convert samples to HF-compatible dataset
        """
        data = []
        for sample in samples:
            encoded_candidates, option_lens = encode_prompt(
                task, task.get_template(template_version=args.template_ver),
                [], sample,
                tokenizer, max_length=args.max_length,
                generation=task.generation, generation_with_gold=True,
                max_new_tokens=args.max_new_tokens
            )

            if task.generation:
                correct_candidate_id = 0
            elif isinstance(sample.correct_candidate, list):
                correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
            else:
                correct_candidate_id = sample.candidates.index(sample.correct_candidate)

            if args.non_diff:
                # For non-differentiable objective, there is no teacher forcing thus the 
                # current answer part is removed
                encoded_candidates[correct_candidate_id] = (encoded_candidates[correct_candidate_id])[
                    :-option_lens[correct_candidate_id]
                ]

            if args.train_as_classification:
                # For classification, we provide the label as the correct candidate id
                data.append([
                    {
                        "input_ids": encoded_candidates[_i],
                        "labels": correct_candidate_id,
                        "option_len": option_lens[_i],
                        "num_options": len(sample.candidates)
                    } for _i in range(len(encoded_candidates))
                ])
            elif args.only_train_option:
                # Otherwise, it is just LM-style teacher forcing
                if args.non_diff:
                    # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                    data.append({
                        "input_ids": encoded_candidates[correct_candidate_id],
                        "labels": encoded_candidates[correct_candidate_id],
                        "option_len": option_lens[correct_candidate_id],
                        "gold": sample.correct_candidate
                    })
                else:
                    data.append({
                        "input_ids": encoded_candidates[correct_candidate_id],
                        "labels": encoded_candidates[correct_candidate_id],
                        "option_len": option_lens[correct_candidate_id]
                    })
            else:
                data.append({
                    "input_ids": encoded_candidates[correct_candidate_id],
                    "labels": encoded_candidates[correct_candidate_id]
                })

        return data

    with count_time("Tokenizing training samples"):
        train_dataset = HFDataset(_convert(train_samples))
        eval_dataset = HFDataset(_convert(eval_samples))
        dev_dataset = HFDataset(_convert(dev_samples))

    collator = DataCollatorForTokenClassification

    #!removed support for head tunning
    def model_init(trial):
        config = AutoConfig.from_pretrained(args.model_name)

        #free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)

        torch_dtype = torch.float32
        if args.load_float16:
            torch_dtype = torch.float16
        elif args.load_bfloat16:
            torch_dtype = torch.bfloat16
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, config=config, device_map='auto', torch_dtype=torch_dtype,
            #max_memory={i: f'{free_in_GB - 5}GB' for i in range(torch.cuda.device_count())}
        )

        model.eval()

        if args.prefix_tuning:
            PrefixTuning(
                model, num_prefix=args.num_prefix, reparam=(not args.no_reparam),
                float16=args.load_float16, init_by_real_act=args.prefix_init_by_real_act
            )

        if args.lora:
            LoRA(model, r=args.lora_r, alpha=args.lora_alpha, float16=args.load_float16)

        if args.prompt_tuning:
            PromptTuning(
                model, num_virtual_tokens=args.num_virtual_tokens,
                init_by_real_tokens=args.prompt_init_by_real_tokens,
                hide_virtual_token_logits=True #a workaround for the other loss/prediction functions
            )
        
        if args.only_train_option and not args.non_diff:
            model.original_forward = model.forward
            model.forward = forward_wrap_with_option_len.__get__(model, type(model))

        return model

    callback = EvaluationCallback(args, task, eval_samples, dev_samples)

    trainer = OurTrainer(
        model=None, #!required to be None for hyperparam_search
        model_init=model_init, #!also required for hyperparam_search
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=(
            DataCollatorWithPaddingAndNesting(tokenizer, pad_to_multiple_of=8)
            if args.train_as_classification
            else collator(tokenizer, pad_to_multiple_of=8)
        ),
        evaluate_func=callback.evaluate,
        perturb_module_regex=None,
    )

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "zo_eps": trial.suggest_categorical("zo_eps", [1e-3, 1e-4, 1e-5]),
            "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["cosine", "cosine_with_restarts"]),
            # "restart_rate": trial.suggest_int("restart_rate", 100, args.max_steps, log=True),
            # "do_reset": trial.suggest_categorical("do_reset", [True, False]),
            # "aggressive": trial.suggest_categorical("aggressive", [True, False]),
            # "adaptable": trial.suggest_categorical("adaptable", [True, False]),
        }

    # def optuna_hp_space(trial):
    #     return {
    #         "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
    #         "momentum": trial.suggest_float("momentum", 0.001, 0.999, log=True),
    #     }

    def generate_name(trial):
        return f"zo-SGD cosine {trial.number}"

    best_trials = trainer.hyperparameter_search(
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=20, #TODO: set this to something appropriate
        hp_name=generate_name,
        study_name="zo-SGD RTE",
        sampler=optuna.samplers.TPESampler(), #!set by default, added to make SURE
        pruner=optuna.pruners.HyperbandPruner(), #!said to be the best for TPE sampler
    )

#--model_name=roberta-large --task_name=RTE --output_dir=result/RTE-ft-69 --overwrite_output_dir --per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps --save_strategy=steps --save_total_limit=1 --eval_steps=500 --save_steps=500 --max_steps=15000 --logging_steps=500 --num_eval=3000 --num_train=1743 --num_dev=747 --train_as_classification --train_set_seed=52 --trainer=zo_sgd --warmup_steps=150