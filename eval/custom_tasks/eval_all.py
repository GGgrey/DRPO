import ast
import random
import re
from string import ascii_uppercase, ascii_lowercase
from aenum import extend_enum
from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics, math_scorer
from lighteval.metrics.utils.metric_utils import (
    SampleLevelMetric,
    SamplingMethod,
    SampleLevelMetricGrouping,
)
from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    extract_target_from_pred,
    get_extraction_regexes_inspect,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from lighteval.metrics.metrics_sample import (
    PassAtK,
    AvgAtN,
    GPassAtK,
    MajAtN,
)
import numpy as np


MATH_QUERY_TEMPLATE = """
You are a helpful AI Assistant, designed to provide well-reasoned and detailed responses. You FIRST think about the reasoning process step by step and then provide the user with the answer. Please enclose your final answer in the box: \\boxed{{Your Answer}}.

{question}
""".strip()

GPQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

MMLU_PRO_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDEFGHIJ. Think step by step before answering.

{question}

{choices}

Answer:""".strip()

MATHQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of abcde. Think step by step before answering.

{question}

{choices}

Answer:""".strip()

COMMONSENSEQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDE. Think step by step before answering.

{question}

{choices}

Answer:""".strip()


def math_500_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(question=line["problem"]),
        choices=[line["solution"]],
        gold_index=0,
    )


def gsm8k_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(question=line["question"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def aime_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(question=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def gpqa_prompt_fn(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query = GPQA_QUERY_TEMPLATE.format(
        A=choices[0].strip(),
        B=choices[1].strip(),
        C=choices[2].strip(),
        D=choices[3].strip(),
        question=line["Question"].strip()
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=list(ascii_uppercase)[: len(choices)],
        gold_index=gold_index,
        instruction=query,
    )


def minerva_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(question=line["problem"]),
        choices=[line["solution"]],
        gold_index=0,
    )


def amc_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(question=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def olympiadbench_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(question=line["question"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def gsm_plus_prompt_fn(line, task_name: str = None):
    if line["perturbation_type"] == "critical thinking":
        return None

    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(question=line["question"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def mmlu_pro_prompt_fn(line, task_name: str = None):
    choices = "\n".join([f"{letter}: {choice}" for letter, choice in zip(ascii_uppercase, line["options"])])
    query = MMLU_PRO_QUERY_TEMPLATE.format(
        question=line["question"],
        choices=choices,
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=ascii_uppercase[: len(choices)],
        gold_index=line["answer_index"],
        instruction=query,
    )


def record_to_sample(record):
    query = record["problem"]
    target = record["answer"]
    return Sample(
        input=query,
        target=target
    )


def gpqa_record_to_sample(record):
    gold_index = random.randint(0, 3)
    choices = [record["Incorrect Answer 1"], record["Incorrect Answer 2"], record["Incorrect Answer 3"]]
    choices.insert(gold_index, record["Correct Answer"])
    return Sample(
        input=record["Question"].strip(),
        choices=choices,
        target=ascii_uppercase[gold_index],
    )


def mmlu_record_to_sample(record):
    query = record["question"]
    choices = record["options"]
    target = record["answer"]
    return Sample(
        input=query,
        choices=choices,
        target=target,
    )


latex_gold_match = SampleLevelMetric(
    metric_name="latex_gold_match",
    sample_level_fn=MultilingualExtractiveMatchMetric(
        language=Language.ENGLISH,
        fallback_mode="first_match",
        precision=5,
        gold_extraction_target=(LatexExtractionConfig(),),
        # Match boxed first before trying other regexes
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
        aggregation_function=max,
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

expr_gold_match = SampleLevelMetric(
    metric_name="expr_gold_match",
    sample_level_fn=MultilingualExtractiveMatchMetric(
        language=Language.ENGLISH,
        fallback_mode="first_match",
        precision=5,
        gold_extraction_target=(ExprExtractionConfig(),),
        # Match boxed first before trying other regexes
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
        aggregation_function=max,
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

math_pass_at_k = SampleLevelMetric(
    metric_name="math_pass@k",
    sample_level_fn=PassAtK(
        strip_strings=True,
        # Extracting mathematical expressions and latex expressions
        sample_scoring_function=MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            gold_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig()],
            pred_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig()],
            precision=6,
        ),
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

math_avg_at_n = SampleLevelMetric(
    metric_name="math_avg@n",
    sample_level_fn=AvgAtN(
        sample_scoring_function=MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            gold_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig()],
            pred_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig()],
            precision=6,
        ),
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

math_g_pass_at_k = SampleLevelMetricGrouping(
    metric_name="math_g_pass@k",
    sample_level_fn=GPassAtK(
        name_prefix="math",
        strip_strings=True,
        sample_scoring_function=MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            fallback_mode="first_match",
            precision=5,
            gold_extraction_target=(ExprExtractionConfig(),),
            # Match boxed first before trying other regexes
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
            aggregation_function=max,
        ),
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

math_maj_at_n = SampleLevelMetric(
    metric_name="math_maj@n",
    sample_level_fn=MajAtN(
        sample_scoring_function=MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            gold_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig()],
            pred_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig()],
            precision=6,
        ),
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

gpqa_extractive_match = SampleLevelMetric(
    metric_name="gpqa_extractive_match",
    sample_level_fn=MultilingualExtractiveMatchMetric(
        language=Language.ENGLISH,
        gold_extraction_target=[
            IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
        ],
        pred_extraction_target=[
            IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
        ],
        precision=6,
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

gpqa_pass_at_k = SampleLevelMetric(
    metric_name="gpqa_pass@k",
    sample_level_fn=PassAtK(
        sample_scoring_function=MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            gold_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
            ],
            pred_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
            ],
            precision=6,
        ),
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

gpqa_avg_at_n = SampleLevelMetric(
    metric_name="gpqa_avg@n",
    sample_level_fn=AvgAtN(
        sample_scoring_function=MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            gold_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
            ],
            pred_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
            ],
            precision=6,
        ),
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

gpqa_g_pass_at_k = SampleLevelMetric(
    metric_name="gpqa_g_pass@k",
    sample_level_fn=GPassAtK(
        sample_scoring_function=MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            gold_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
            ],
            pred_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
            ],
            precision=6,
        ),
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

gpqa_maj_at_n = SampleLevelMetric(
    metric_name="gpqa_maj@n",
    sample_level_fn=MajAtN(
        sample_scoring_function=MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            gold_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
            ],
            pred_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
            ],
            precision=6,
        ),
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

extend_enum(Metrics, "latex_gold_match", latex_gold_match)
extend_enum(Metrics, "expr_gold_match", expr_gold_match)
extend_enum(Metrics, "math_pass_at_k", math_pass_at_k)
extend_enum(Metrics, "math_avg_at_n", math_avg_at_n)
extend_enum(Metrics, "math_g_pass_at_k", math_g_pass_at_k)
extend_enum(Metrics, "math_maj_at_n", math_maj_at_n)
extend_enum(Metrics, "gpqa_extractive_match", gpqa_extractive_match)
extend_enum(Metrics, "gpqa_pass_at_k", gpqa_pass_at_k)
extend_enum(Metrics, "gpqa_avg_at_n", gpqa_avg_at_n)
extend_enum(Metrics, "gpqa_g_pass_at_k", gpqa_g_pass_at_k)
extend_enum(Metrics, "gpqa_maj_at_n", gpqa_maj_at_n)

math500_custom = LightevalTaskConfig(
    name="math500_custom",
    prompt_function=math_500_prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.latex_gold_match],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=model_graded_fact(),
)

math500_passk_custom = LightevalTaskConfig(
    name="math500_passk_custom",
    prompt_function=math_500_prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_pass_at_k(sample_params={"k": 16, "n": 32})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

math500_avgn_custom = LightevalTaskConfig(
    name="math500_avgn_custom",
    prompt_function=math_500_prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_avg_at_n(sample_params={"n": 16})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

math500_gpassk_custom = LightevalTaskConfig(
    name="math500_gpassk_custom",
    prompt_function=math_500_prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_g_pass_at_k(sample_params={"k": 16, "n": 32})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

math500_majn_custom = LightevalTaskConfig(
    name="math500_majn_custom",
    prompt_function=math_500_prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_maj_at_n(sample_params={"n": 32})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

gsm8k_custom = LightevalTaskConfig(
    name="gsm8k_custom",
    prompt_function=gsm8k_prompt_fn,
    hf_repo="openai/gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1024,
    metrics=[Metrics.expr_gold_match],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

gsm8k_passk_custom = LightevalTaskConfig(
    name="gsm8k_passk_custom",
    prompt_function=gsm8k_prompt_fn,
    hf_repo="openai/gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1024,
    metrics=[Metrics.math_pass_at_k(sample_params={"k": 16, "n": 64})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

gsm8k_avgn_custom = LightevalTaskConfig(
    name="gsm8k_avgn_custom",
    prompt_function=gsm8k_prompt_fn,
    hf_repo="openai/gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1024,
    metrics=[Metrics.math_avg_at_n(sample_params={"n": 16})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

gsm8k_gpassk_custom = LightevalTaskConfig(
    name="gsm8k_gpassk_custom",
    prompt_function=gsm8k_prompt_fn,
    hf_repo="openai/gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1024,
    metrics=[Metrics.math_g_pass_at_k(sample_params={"k": 16, "n": 32})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

gsm8k_majn_custom = LightevalTaskConfig(
    name="gsm8k_majn_custom",
    prompt_function=gsm8k_prompt_fn,
    hf_repo="openai/gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1024,
    metrics=[Metrics.math_maj_at_n(sample_params={"n": 32})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

aime24_custom = LightevalTaskConfig(
    name="aime24_custom",
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.expr_gold_match],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

aime24_passk_custom = LightevalTaskConfig(
    name="aime24_passk_custom",
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_pass_at_k(sample_params={"k": 16, "n": 32})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

aime24_avgn_custom = LightevalTaskConfig(
    name="aime24_avgn_custom",
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_avg_at_n(sample_params={"n": 16})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

aime24_gpassk_custom = LightevalTaskConfig(
    name="aime24_gpassk_custom",
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_g_pass_at_k(sample_params={"k": 16, "n": 32})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

aime24_majn_custom = LightevalTaskConfig(
    name="aime24_majn_custom",
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_maj_at_n(sample_params={"n": 32})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

aime25_custom = LightevalTaskConfig(
    name="aime25_custom",
    prompt_function=aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.expr_gold_match],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

aime25_passk_custom = LightevalTaskConfig(
    name="aime25_passk_custom",
    prompt_function=aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_pass_at_k(sample_params={"k": 16, "n": 32})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

aime25_avgn_custom = LightevalTaskConfig(
    name="aime25_avgn_custom",
    prompt_function=aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_avg_at_n(sample_params={"n": 16})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

aime25_gpassk_custom = LightevalTaskConfig(
    name="aime25_gpassk_custom",
    prompt_function=aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_g_pass_at_k(sample_params={"k": 16, "n": 32})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

aime25_majn_custom = LightevalTaskConfig(
    name="aime25_majn_custom",
    prompt_function=aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_maj_at_n(sample_params={"n": 32})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

aime90_custom = LightevalTaskConfig(
    name="aime90_custom",
    prompt_function=aime_prompt_fn,
    hf_repo="xiaoyuanliu/AIME90",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.expr_gold_match],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

gpqa_diamond_custom = LightevalTaskConfig(
    name="gpqa_diamond_custom",
    prompt_function=gpqa_prompt_fn,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.gpqa_extractive_match],
    stop_sequence=[],  # No stop sequence, will use eos token
    version=1,
    sample_fields=gpqa_record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

gpqa_diamond_avgn_custom = LightevalTaskConfig(
    name="gpqa_diamond_avgn_custom",
    prompt_function=gpqa_prompt_fn,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.gpqa_avg_at_n(sample_params={"n": 16})],
    stop_sequence=[],
    version=1,
    sample_fields=gpqa_record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

minerva_custom = LightevalTaskConfig(
    name="minerva_custom",
    prompt_function=minerva_prompt_fn,
    hf_repo="knoveleng/Minerva-Math",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.latex_gold_match],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

minerva_avgn_custom = LightevalTaskConfig(
    name="minerva_avgn_custom",
    prompt_function=minerva_prompt_fn,
    hf_repo="knoveleng/Minerva-Math",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_avg_at_n(sample_params={"n": 16})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

amc23_custom = LightevalTaskConfig(
    name="amc23_custom",
    prompt_function=amc_prompt_fn,
    hf_repo="knoveleng/AMC-23",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.expr_gold_match],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

amc23_passk_custom = LightevalTaskConfig(
    name="amc23_passk_custom",
    prompt_function=amc_prompt_fn,
    hf_repo="knoveleng/AMC-23",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_pass_at_k(sample_params={"k": 16, "n": 32})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

amc23_avgn_custom = LightevalTaskConfig(
    name="amc23_avgn_custom",
    prompt_function=amc_prompt_fn,
    hf_repo="knoveleng/AMC-23",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_avg_at_n(sample_params={"n": 16})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

amc23_gpassk_custom = LightevalTaskConfig(
    name="amc23_gpassk_custom",
    prompt_function=amc_prompt_fn,
    hf_repo="knoveleng/AMC-23",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_g_pass_at_k(sample_params={"k": 16, "n": 32})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

amc23_majn_custom = LightevalTaskConfig(
    name="amc23_majn_custom",
    prompt_function=amc_prompt_fn,
    hf_repo="knoveleng/AMC-23",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_maj_at_n(sample_params={"n": 32})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

olympiadbench_custom = LightevalTaskConfig(
    name="olympiadbench_custom",
    prompt_function=olympiadbench_prompt_fn,
    hf_repo="knoveleng/OlympiadBench",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.latex_gold_match],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

olympiadbench_avgn_custom = LightevalTaskConfig(
    name="olympiadbench_avgn_custom",
    prompt_function=olympiadbench_prompt_fn,
    hf_repo="knoveleng/OlympiadBench",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.math_avg_at_n(sample_params={"n": 16})],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

gsm_plus_custom = LightevalTaskConfig(
    name="gsm_plus_custom",
    prompt_function=gsm_plus_prompt_fn,
    hf_repo="qintongli/GSM-Plus",
    hf_subset="default",
    hf_avail_splits=["test", "testmini"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1024,
    metrics=[Metrics.expr_gold_match],
    stop_sequence=None,
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

gsm_plus_avgn_custom = LightevalTaskConfig(
    name="gsm_plus_avgn_custom",
    prompt_function=gsm_plus_prompt_fn,
    hf_repo="qintongli/GSM-Plus",
    hf_subset="default",
    hf_avail_splits=["test", "testmini"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1024,
    metrics=[Metrics.math_avg_at_n(sample_params={"n": 4})],
    stop_sequence=None,
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
)

mmlu_pro_custom = LightevalTaskConfig(
    name="mmlu_pro_custom",
    prompt_function=mmlu_pro_prompt_fn,
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
    evaluation_splits=("test",),
    few_shots_split=None,
    generation_size=4096,
    metrics=[Metrics.gpqa_extractive_match],
    sample_fields=mmlu_record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

mmlu_pro_avgn_custom = LightevalTaskConfig(
    name="mmlu_pro_avgn_custom",
    prompt_function=mmlu_pro_prompt_fn,
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
    evaluation_splits=("test",),
    few_shots_split=None,
    generation_size=4096,
    metrics=[Metrics.gpqa_avg_at_n(sample_params={"n": 4})],
    sample_fields=mmlu_record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

TASKS_TABLE = [
    math500_custom,
    math500_passk_custom,
    math500_avgn_custom,
    math500_gpassk_custom,
    math500_majn_custom,

    gsm8k_custom,
    gsm8k_passk_custom,
    gsm8k_avgn_custom,
    gsm8k_gpassk_custom,
    gsm8k_majn_custom,

    aime90_custom,
    
    gpqa_diamond_custom,
    gpqa_diamond_avgn_custom,
    
    minerva_custom,
    minerva_avgn_custom,

    olympiadbench_custom,
    olympiadbench_avgn_custom,

    gsm_plus_custom,
    gsm_plus_avgn_custom,

    mmlu_pro_custom,
    mmlu_pro_avgn_custom,

    aime24_custom,
    aime24_passk_custom,
    aime24_avgn_custom,
    aime24_gpassk_custom,
    aime24_majn_custom,

    aime25_custom,
    aime25_passk_custom,
    aime25_avgn_custom,
    aime25_gpassk_custom,
    aime25_majn_custom,

    amc23_custom,
    amc23_passk_custom,
    amc23_avgn_custom,
    amc23_gpassk_custom,
    amc23_majn_custom,
]
