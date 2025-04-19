import pandas as pd
import sklearn
from multistage_pipeline.llms import llm_prompt_classifier, llm_concept_classifier
import json
import os
import numpy as np

def save_train_concepts_word_level_blacklist(save_path):

    from utils import prepare_dataset
    dataset_json, train_concepts, test_concepts, train_raw_data_list, valid_raw_data_list, test_raw_data_list = prepare_dataset()

    train_concepts_word_level_blacklist = list()

    for concept in train_concepts:
        is_blacklist = llm_concept_classifier(concept)
        is_blacklist = int(is_blacklist)
        if is_blacklist == 1:
            train_concepts_word_level_blacklist.append(concept)

        
        with open(save_path, 'w') as json_file:
            json.dump(train_concepts_word_level_blacklist, json_file)
            print(f'save at {save_path}')

def word_level_blacklist_flag(prompt, word_level_blacklist_concepts):
  for concept in word_level_blacklist_concepts:
    if concept in prompt:
      return 1
  return 0

def get_prompt(target, unsafe_prompt, safe_prompt):
  if target == 1:
    return unsafe_prompt 
  elif target == 0: 
    return safe_prompt

def compute_auroc(df):
  y_true = df['target']
  y_score = df['prediction']

  auroc = sklearn.metrics.roc_auc_score(y_true, y_score)
  auroc = float(auroc)
  return auroc

def compute_accuracy(df, threshold):
  temp_df = df
  temp_df['prediction_binary'] = temp_df['prediction'].apply(lambda x: 1 if x>= threshold else 0)

  y_true = df['target']
  y_pred = df['prediction_binary']
  acc = sklearn.metrics.accuracy_score(y_true, y_pred)
  acc = float(acc)
  return acc

def get_prediction_llm(prediction_binary, llm_prediction, prediction, delta, threshold=4.47):
  print('llm_prediction:', llm_prediction)
  if np.isnan(llm_prediction):
    return prediction_binary
  elif prediction < threshold+delta and prediction > threshold - delta:
    return llm_prediction
  else: 
    return prediction_binary

def get_all_predictions(df, llm_classification_cache_path, threshold = 4.47, deltas = [0.1, 0.5, 1, 2]):
    '''
    Compute prediction of all strategies (baseline, prelatent, postlatentguard)

    Input: df: pd.DataFrame with ['prompt', 'prediction', 'target'] or ['unsafe_prompt', 'safe_prompt', 'prediction', 'target']
    '''

    max_delta = max(deltas)
    min_threshold = threshold - max_delta
    max_threshold = threshold + max_delta

#   load data
    if 'prompt' not in df.columns:
        df['prompt'] = df.apply(lambda row: get_prompt(row['target'], row['unsafe_prompt'], row['safe_prompt']), axis=1)
    df = df[['prompt', 'prediction', 'target']]

    # baseline
    df['prediction_binary'] = df['prediction'].apply(lambda x: 1 if x>= threshold else 0)

    # prelatent guard
    train_concepts_word_level_blacklist_path = 'multistage_pipeline/train_concepts_word_level_blacklist.json'
    if not os.path.exists(train_concepts_word_level_blacklist_path):
        save_train_concepts_word_level_blacklist(train_concepts_word_level_blacklist_path)
    with open(train_concepts_word_level_blacklist_path, 'r') as json_file:
        train_concepts_word_level_blacklist = json.load(json_file)
    
    df['word_level_blacklist_flag'] = df['prompt'].apply(lambda x: word_level_blacklist_flag(x, train_concepts_word_level_blacklist))

    # llm prediction
    if os.path.exists(llm_classification_cache_path):
       temp_df = pd.read_csv(llm_classification_cache_path)
       df['llm_prediction'] = temp_df['llm_prediction']
    else:
        df['llm_prediction'] = df.apply(lambda row: llm_prompt_classifier(row['prompt']) 
                                        if ((row['prediction'] > min_threshold) and (row['prediction'] < max_threshold)) 
                                        else None, axis=1)
        df.to_csv(llm_classification_cache_path, index=False)
        print(f'saved at {llm_classification_cache_path}')
    
    df['prediction_pre_latentguard'] = df.apply(lambda row: row['word_level_blacklist_flag'] if row['word_level_blacklist_flag']==1 
                                                else row['prediction_binary'], axis=1)

    # override latentguard with llm prediction if the score is within (threshold-delta, threshold+delta)
    for delta in deltas:
        df[f'prediction_post_latentguard_delta_{delta}'] = df.apply(lambda row: get_prediction_llm(row['prediction_binary'], 
                                                                                                    row['llm_prediction'],
                                                                                                    row['prediction'],
                                                                                                    delta = delta,
                                                                                                    threshold = threshold
                                                                                                    )
                                                                                                , axis=1)
    return df
    
def get_metrics(df, threshold = 4.47, deltas = [0.1, 0.5, 1, 2]):
    metrics = dict()
    auroc = compute_auroc(df)
    acc = compute_accuracy(df, threshold=threshold)
    metrics['auroc'] = auroc
    metrics['acc'] = acc

    y_true = df['target']

    y_pred = df['prediction_pre_latentguard']
    metrics['acc_pre_latentguard'] = sklearn.metrics.accuracy_score(y_true, y_pred)

    for delta in deltas:
       y_pred = df[f'prediction_post_latentguard_delta_{delta}']
       metrics[f'acc_postlatentguard_delta_{delta}'] = sklearn.metrics.accuracy_score(y_true, y_pred)
    
    return metrics


def main():

    # print('Processing CoPro dataset...')
    # valid_id_explicit_df = pd.read_csv('predictions/valid/is_train_concepts_True/explicit.csv')
    # valid_id_synonym_df = pd.read_csv('predictions/valid/is_train_concepts_True/synonym.csv')
    # valid_id_adversarial_df = pd.read_csv('predictions/valid/is_train_concepts_True/adversarial.csv')

    # valid_ood_explicit_df = pd.read_csv('predictions/valid/is_train_concepts_False/explicit.csv')
    # valid_ood_synonym_df = pd.read_csv('predictions/valid/is_train_concepts_False/synonym.csv')
    # valid_ood_adversarial_df = pd.read_csv('predictions/valid/is_train_concepts_False/adversarial.csv')

    # test_id_explicit_df = pd.read_csv('predictions/test/is_train_concepts_True/explicit.csv')
    # test_id_synonym_df = pd.read_csv('predictions/test/is_train_concepts_True/synonym.csv')
    # test_id_adversarial_df = pd.read_csv('predictions/test/is_train_concepts_True/adversarial.csv')

    # test_ood_explicit_df = pd.read_csv('predictions/test/is_train_concepts_False/explicit.csv')
    # test_ood_synonym_df = pd.read_csv('predictions/test/is_train_concepts_False/synonym.csv')
    # test_ood_adversarial_df = pd.read_csv('predictions/test/is_train_concepts_False/adversarial.csv')

    # valid_dfs = {
    #     'id_explicit': valid_id_explicit_df,
    #     'id_synonym': valid_id_synonym_df,
    #     'id_adversarial': valid_id_adversarial_df,
    #     'ood_explicit': valid_ood_explicit_df,
    #     'ood_synonym': valid_ood_synonym_df,
    #     'ood_adversarial': valid_ood_adversarial_df
    # }

    # test_dfs = {
    #     'id_explicit': test_id_explicit_df,
    #     'id_synonym': test_id_synonym_df,
    #     'id_adversarial': test_id_adversarial_df,
    #     'ood_explicit': test_ood_explicit_df,
    #     'ood_synonym': test_ood_synonym_df,
    #     'ood_adversarial': test_ood_adversarial_df
    # }
    # # Validation set
    # valid_metrics = list()

    # for setting in valid_dfs.keys():
    #     df = valid_dfs[setting]
    #     llm_classification_cache_path = f'multistage_pipeline/valid/{setting}.csv'

    #     result_df = get_all_predictions(df, llm_classification_cache_path=llm_classification_cache_path)
    #     #    print(result_df)

    #     result_path = f'multistage_pipeline/predictions/valid/{setting}.csv'
    #     result_dir = os.path.dirname(result_path)
    #     if not os.path.exists(result_dir):
    #         os.makedirs(result_dir)
    #     result_df.to_csv(result_path, index=False)
        
    #     metrics = get_metrics(result_df)
    #     metrics['setting'] = setting
    #     valid_metrics.append(metrics)

    #     valid_metrics_df = pd.DataFrame(valid_metrics)
    #     valid_metrics_df = valid_metrics_df[['setting']+[col for col in valid_metrics_df.columns if col != 'setting']]

    #     metrics_path = 'multistage_pipeline/evaluation/valid.csv'
    #     metrics_dir = os.path.dirname(metrics_path)
    #     if not os.path.exists(metrics_dir):
    #         os.makedirs(metrics_dir)
    #     valid_metrics_df.to_csv(metrics_path, index=False)


    # # Test set
    # test_metrics = list()

    # for setting in test_dfs.keys():
    #     df = test_dfs[setting]
    #     llm_classification_cache_path = f'multistage_pipeline/test/{setting}.csv'

    #     result_df = get_all_predictions(df, llm_classification_cache_path=llm_classification_cache_path)
    #     #    print(result_df)

    #     result_path = f'multistage_pipeline/predictions/test/{setting}.csv'
    #     result_dir = os.path.dirname(result_path)
    #     if not os.path.exists(result_dir):
    #         os.makedirs(result_dir)
    #     result_df.to_csv(result_path, index=False)
        
    #     metrics = get_metrics(result_df)
    #     metrics['setting'] = setting
    #     test_metrics.append(metrics)

    #     test_metrics_df = pd.DataFrame(test_metrics)
    #     test_metrics_df = test_metrics_df[['setting']+[col for col in test_metrics_df.columns if col != 'setting']]

    #     metrics_path = 'multistage_pipeline/evaluation/test.csv'
    #     metrics_dir = os.path.dirname(metrics_path)
    #     if not os.path.exists(metrics_dir):
    #         os.makedirs(metrics_dir)
    #     test_metrics_df.to_csv(metrics_path, index=False)


    # UD dataset

    # load precomputed latent guard score
    print('Processing Unsafe Diffusion dataset...')
    UD_path = 'UD_dynamic_threshold_results.json'

    with open(UD_path, 'r') as file:
        ud_data = json.load(file)

    df = pd.DataFrame(ud_data)
    df = df[['prompt', 'label', 'score']]
    df = df.rename(columns={'label': 'target', 'score': 'prediction'})

    llm_classification_cache_path = f'multistage_pipeline/test/unsafe_diffusion.csv'
    result_df = get_all_predictions(df, llm_classification_cache_path=llm_classification_cache_path, threshold = 4.47, deltas=[0.1, 0.5, 1, 2, 3, 4, 5])


    result_path = f'multistage_pipeline/predictions/test/unsafe_diffusion.csv'
    result_dir = os.path.dirname(result_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_df.to_csv(result_path, index=False)

    metrics = get_metrics(result_df, threshold = 4.47, deltas=[0.1, 0.5, 1, 2, 3, 4, 5])
    metrics_df = pd.DataFrame([metrics])

    metrics_path = 'multistage_pipeline/evaluation/test_unsafe_diffusion.csv'
    metrics_dir = os.path.dirname(metrics_path)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    metrics_df.to_csv(metrics_path, index=False)

    print('Evaluation completed...')


if __name__ == "__main__":
   main()
    