from enum import Enum

from model.sequence_classification import (
    BertPrefixForSequenceClassification,
    BertForSequenceClassification,
    RobertaPrefixForSequenceClassification,
    RobertaForSequenceClassification
)

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)

class TaskType(Enum):
    TOKEN_CLASSIFICATION = 1,
    SEQUENCE_CLASSIFICATION = 2,
    QUESTION_ANSWERING = 3,
    MULTIPLE_CHOICE = 4

FINETUNE_MODELS = {
    "bert": {
        TaskType.SEQUENCE_CLASSIFICATION: BertForSequenceClassification
    },
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaForSequenceClassification
    }
}

PREFIX_MODELS = {
    "bert": {
        TaskType.SEQUENCE_CLASSIFICATION: BertPrefixForSequenceClassification
    },
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPrefixForSequenceClassification
    },
}

AUTO_MODELS = {
    TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification
}

def get_model(model_args, task_type: TaskType, config: AutoConfig, fix_bert: bool = False):
    if model_args.prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size
        
        model_class = PREFIX_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    elif model_args.prompt:
        config.pre_seq_len = model_args.pre_seq_len
        model_class = PROMPT_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    else:
        model_class = AUTO_MODELS[task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )

        bert_param = 0
        if fix_bert:
            if config.model_type == "bert":
                for param in model.bert.parameters():
                    param.requires_grad = False
                for _, param in model.bert.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "roberta":
                for param in model.roberta.parameters():
                    param.requires_grad = False
                for _, param in model.roberta.named_parameters():
                    bert_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('***** total param is {} *****'.format(total_param))
    return model


def get_model_deprecated(model_args, task_type: TaskType, config: AutoConfig, fix_bert: bool = False):
    if model_args.prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size

        if task_type == TaskType.SEQUENCE_CLASSIFICATION:
            from model.sequence_classification import BertPrefixModel, RobertaPrefixModel

        if config.model_type == "bert":
            model = BertPrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "roberta":
            model = RobertaPrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        else:
            raise NotImplementedError


    elif model_args.prompt:
        config.pre_seq_len = model_args.pre_seq_len

        from model.sequence_classification import BertPromptModel, RobertaPromptModel
        if config.model_type == "bert":
            model = BertPromptModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "roberta":
            model = RobertaPromptModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        else:
            raise NotImplementedError
            

    else:
            
        if task_type == TaskType.SEQUENCE_CLASSIFICATION:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
    
        bert_param = 0
        if fix_bert:
            if config.model_type == "bert":
                for param in model.bert.parameters():
                    param.requires_grad = False
                for _, param in model.bert.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "roberta":
                for param in model.roberta.parameters():
                    param.requires_grad = False
                for _, param in model.roberta.named_parameters():
                    bert_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('***** total param is {} *****'.format(total_param))
    return model
