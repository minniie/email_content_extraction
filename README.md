# email_content_extraction

## About

COS485 Final Project (Spring 2023)

## Authors

Min Lee '23

Simon Park '23

## Abstract
In this paper, we propose a double fine-tuning method for an email content extraction task with limited availability for annotated data. We compare the results of 1) a task-adaptive approach where a pre-trained model is first fine-tuned on a large dataset with a similar task as the target dataset before being fine-tuned again on the small target dataset and 2) a domain-adaptive approach where the dataset for the first fine-tuning stage is replaced with one with a similar data domain as the target dataset. We observe that both approaches are effective at aligning the pre-trained model for the downstream task.
