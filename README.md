# TA-tool
All resources are available in Release.

A small tool based on BERT-wwm for identifying the referents of the gender-neutral pronoun "TA" in Chinese

This tool performs coreference resolution for the Chinese gender-neutral pronoun “ta” by leveraging a fine-tuned Chinese BERT-wwm model to predict its referent.
Note: This tool cannot resolve zero anaphora for “ta.” In other words, if the referent of “ta” does not explicitly appear in the text and must be inferred by the reader, the tool will not work.

A.	If you do not wish to use Python or modify the code, please follow the steps below:
  1.	Download the fine-tuned bert.wwm model from the Release section: “bert-ta-export.rar“. Save it to any directory and     extract it. You will see five files: config.json, model.safetensors, pytorch_model.bin, special_tokens_map.json,          tokenizer_config.json, and vocab.txt.
  2.	Download "main.rar" directly from the Release section. After unzipping it, click "main.exe" to run the program.
  3.	In the main.exe panel, there are two sections. The upper section is for manually inputting a single sentence and making a prediction: click the “Select Model” button and choose the path where the bert-ta-export model is saved (i.e., the location of the five files mentioned above). Then, enter a sentence containing “ta” that you want to analyze, click “Start Prediction,” and after a short wait, the result will appear in the text box below.
  4.	If you want to perform batch prediction of the referents of “ta” in multiple sentences, you need to use the buttons in  the lower section. First, create a CSV file and make sure it contains a column named “input_text” (if you are unsure how to use it, you can refer to the test.csv file). Then, in the lower part of the panel, click “Select Input CSV” to choose the CSV file you created, then click “Select Model Directory” to choose the model's directory. Next, click “Select Output CSV” to specify the save path for the output file, and finally click “Run.” The program will start running, and you can monitor the progress through the “Processing” indicator. Once processing is complete, a dialog will inform you that the task is done. You can then open the output file to see the results.

B.	If you prefer to run this tool directly through code:
  1	Download the fine-tuned bert.wwm model “bert-ta-export.rar“ from the release, save it to any directory, and extract it. You will see the following five files: config.json, model.safetensors, pytorch_model.bin, special_tokens_map.json, tokenizer_config.json, and vocab.txt.
  2	Run main.py; the other steps are the same as in part A. The code includes the part that generates the interactive interface, which is the same as the interface described in part A.

