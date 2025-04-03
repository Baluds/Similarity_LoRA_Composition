import pandas as pd
import logging

logger = logging.getLogger()

def format_input(row, input_columns):
    return "\n".join([row[ic] for ic in input_columns])


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
            
        return prompt, incited_response, input_columns, output_columns
 
    def transform(self, test):
        # this function assumes csv inputs
        df = pd.read_csv(self.input_file_path)
        prompt, incited_response, input_columns, output_columns = self.type_to_prompt_mapper(self.type)
        output_file_path = self.output_file_path+'Transformed_' + \
            self.input_file_path.split('/')[-1].split('.')[0]+'.csv'

        if not test:

            try:

                df["Text"] = df.apply(lambda row: (
                    f"<|begin_of_text|>{prompt}\n"
                    f"{format_input(row, input_columns)}\n"
                    f"{incited_response}:\n"
                    f"{row[output_columns[0]]}<|end_of_text|>"
                ), axis=1)

            except Exception as e:
                logger.error(
                    f"Could not tranform the train split of dataset. Error: {e}")

        else:
            try:

                df["Text"] = df.apply(lambda row: (
                    f"<|begin_of_text|>{prompt}\n"
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
