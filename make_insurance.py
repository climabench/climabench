from pdb import Pdb
from utils.common_imports import *
from collections import Counter
from sklearn.model_selection import train_test_split


def remove_first_sentence(x):
    return "".join(x.split('.')[1:])


def process(df):
    df['label'] = df['label'].apply(lambda x: {'Y': 1, 'N': 0}[x])
    df = df.dropna()
    df['text'] = df['text'].apply(remove_first_sentence)
    df = df[df['text'].apply(len) > 0]
    df = df[combined['text'] != 'test data']
    return df


def main(all_cfg: DictConfig):
    data_dir = Path('ClimateInsurance')
    num_files = 72
    total_num_rows = 0
    all_columns = set()
    all_answers = []
    all_labels = []
    all_metadata = []
    counter = Counter()
    
    for i in range(num_files):
        if i == 0:
            num = ''
        else:
            num = f' ({i})'
        df = pd.read_csv(data_dir / f'raw/ClimateRiskData{num}.csv', encoding='unicode_escape')
        all_columns.update(df.columns)
        num_rows, num_cols = df.shape
        total_num_rows += num_rows
        num_questions = 8
        for j in range(num_questions):
            question_id = f'Question {j+1}'
            
            if j+1 == 5:
                question_types = [' A', ' B']
            else:
                question_types = ['']
            
            for question_type in question_types:
                question_label = f'{question_id} Yes No{question_type}'
                if question_id in df.columns:
                    all_labels.extend(df[question_label].values)
                    all_answers.extend(df[question_id].values)
                    all_metadata.extend(df['Company Name'].values)

                # counter.update(df['Company Name'].values)
                # all_answers.update(df[question_id].unique())

        print('total_num_rows', total_num_rows, i)
        print('all_columns', all_columns, len(all_columns))
        print('all_answers', len(all_answers))
        # print top 10 answers
    print('all_labels', len(all_labels))
    print('all_answers', len(all_answers))
    # only get unique answers
    all_labels = np.array(all_labels)
    all_answers = np.array(all_answers)
    all_metadata = np.array(all_metadata)

    unique_answers, unique_id = np.unique(all_answers, return_index=True)
    unique_labels = all_labels[unique_id]

    print('unique_labels', unique_labels.shape)
    print('unique_answers', unique_answers.shape)

    companies = all_metadata[unique_id]
    unique_companies = np.unique(companies)
    train_co, val_test_co = train_test_split(unique_companies, test_size=0.2, random_state=42)
    val_co, test_co = train_test_split(val_test_co, test_size=0.5, random_state=42)
    print('train_co', len(train_co))
    print('val_co', len(val_co))
    print('test_co', len(test_co))
    print()
    train_ids = np.isin(companies, train_co)
    val_ids = np.isin(companies, val_co)
    test_ids = np.isin(companies, test_co)

    train_companies = companies[train_ids]
    val_companies = companies[val_ids]
    test_companies = companies[test_ids]

    train_labels = unique_labels[train_ids]
    val_labels = unique_labels[val_ids]
    test_labels = unique_labels[test_ids]

    train_answers = unique_answers[train_ids]
    val_answers = unique_answers[val_ids]
    test_answers = unique_answers[test_ids]

    assert len(train_companies) == len(train_labels) == len(train_answers)
    assert len(val_companies) == len(val_labels) == len(val_answers)
    assert len(test_companies) == len(test_labels) == len(test_answers)
    print(f'Train: {len(train_companies)} Val: {len(val_companies)} Test: {len(test_companies)}')
    
    #save to file
    train_df = pd.DataFrame({'company': train_companies, 'label': train_labels, 'answer': train_answers})
    val_df = pd.DataFrame({'company': val_companies, 'label': val_labels, 'answer': val_answers})
    test_df = pd.DataFrame({'company': test_companies, 'label': test_labels, 'answer': test_answers})
    train_df = process(train_df)
    val_df = process(val_df)
    test_df = process(test_df)


train_df.to_csv(data_dir / 'train.csv', index=False)
val_df.to_csv(data_dir / 'val.csv', index=False)
test_df.to_csv(data_dir / 'test.csv', index=False)


def load_dataset():
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    return train_df, val_df, test_df


if __name__ == "__main__":
    main()
