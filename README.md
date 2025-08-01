For people running the scripts for me (I'm assuming Windows and no previous  - if you're on Linux, you can probably solve any minor discrepancies)

Initial steps:
1. Download data and extract into C:\data\tf3
2. Install Python 3.12 
3. Install CUDA Toolkit
4. Install cuDNN
5. Copy readme, script, requirements.txt into C:\code\tf3

Command Line:
6. run "cd C:\code\tf3"
7. run "pip install requirements.txt"
8. Follow the instructions here (https://pytorch.org/get-started/locally/) to install the correct version of Pytorch for your machine (in case it doesn't match mine). 
9. run "python -m tf3_train_X.py" (Where X is replaced with the model name).

Once the script completes, there should be a small number of .pkl files in C:\data\tf3_X_output (where X is replaced with the model name). Send these to me via whatever method.