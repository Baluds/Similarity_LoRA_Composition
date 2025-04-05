import pandas as pd
import logging

logger = logging.getLogger()

def format_input(row, input_columns):
    return "\n".join([str(row[col]) if pd.notna(row[col]) else "" for col in input_columns])


class Transform_Data:
    def __init__(self, type, input_file_path=None, output_dir=None):
        self.type = type
        if not input_file_path:
            self.input_file_path = '/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/'+type
        else:
            self.input_file_path = input_file_path
        if not output_dir:
            self.output_file_path = ''
            # self.output_file_path = '/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/'
        else:
            self.output_file_path = output_dir
        # add provision for different file types?

    def type_to_prompt_mapper(self, type):
        '''incorporate specific tranformations according to the dataset.

        prompt is the specific directive to the model, incited_response is the trigger word for response, columns include the column names for the dataset originally to extract and perform the right tranformations.
        '''
        if type == 'commonsense_qa':
            prompt = ""
            incited_response = 'Right Option'
            input_columns = ['question', 'CombinedOptions']
            output_columns = ['answerKey']
        
        if type == 'imdb':
            prompt = "What is the sentiment of the below text?"
            incited_response = 'Sentiment'
            input_columns = ['text']
            output_columns = ['name_label']

        if type == 'squad':
            prompt = ""
            incited_response = 'Answer'
            input_columns = ['context', 'question']
            output_columns = ['answer']

        if type == 'story_cloze':
            prompt = ""
            incited_response = 'Answer'
            input_columns = ['prompt', 'options']
            output_columns = ['chosen']

        if type == 'piqa':
            prompt = ""
            incited_response = 'Right Answer'
            input_columns = ['goal', 'options']
            output_columns = ['label']

        if type == 'sst2':
            prompt = "What is the Sentiment of the below statment?"
            incited_response = 'Answer'
            input_columns = ['sentence']
            output_columns = ['label']

        if type == 'yelp':
            prompt = "Please rate the sentiment of the following Yelp review on a scale from 1 (very negative) to 5 (very positive).\nReview:"
            incited_response = 'Answer'
            input_columns = ['statement']
            output_columns = ['label']

        if type == 'cosmos_qa':
            prompt = ""
            incited_response = 'Answer'
            input_columns = ['context', 'question', 'options']
            output_columns = ['label']

        if type == 'paws':
            prompt = "Are these two Sentence Paraphrases of each other?"
            incited_response = 'Answer'
            input_columns = ['sentence1', 'sentence2']
            output_columns = ['label']
            
        return prompt, incited_response, input_columns, output_columns
 
    def transform(self, test):
        # this function assumes csv inputs
        df = pd.read_csv(self.input_file_path)
        prompt, incited_response, input_columns, output_columns = self.type_to_prompt_mapper(self.type)
        output_file_path = self.output_file_path+'Transformed_' + \
            self.input_file_path.split('/')[-1].split('.')[0]+'.csv'

        if not test:

            try:
                prompt_part = f"{prompt}\n" if prompt else ""
                df["Text"] = df.apply(lambda row: (
                    f"{prompt_part}"
                    f"{format_input(row, input_columns)}\n"
                    f"{incited_response}:\n"
                    f"{row[output_columns[0]]}"
                ), axis=1)

            except Exception as e:
                logger.error(
                    f"Could not tranform the train split of dataset. Error: {e}")

        else:
            try:
                prompt_part = f"{prompt}\n" if prompt else ""
                df["Text"] = df.apply(lambda row: (
                    f"{prompt_part}"
                    f"{format_input(row, input_columns)}\n"
                    f"{incited_response}:\n"
                ), axis=1)

            except Exception as e:
                logger.error(
                    f"Could not transform the test split of dataset. Error: {e}")

        try:

            df.to_csv(output_file_path, index=False)
            print(
                f'Formatted CSV for type {self.type} saved in {output_file_path}. Returning tranformed dataframe object.')
            return df

        except Exception as e:
            print(
                f"Could not save the file in {output_file_path} post tranformation. Error: {e}")
