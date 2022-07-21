from tasks.glue.dataset import task_to_keys as glue_tasks

GLUE_DATASETS = list(glue_tasks.keys())


TASKS = ["glue"]

DATASETS = GLUE_DATASETS

ADD_PREFIX_SPACE = {
    'bert': False,
    'roberta': True,
}

USE_FAST = {
    'bert': True,
    'roberta': True,
}