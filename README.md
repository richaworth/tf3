For people running the scripts for me (I'm assuming Windows and no previous python installs - if you're on Linux, you can probably solve any minor discrepancies)

Initial steps:
1. Install Python 3.12 
2. Install CUDA Toolkit
3. Install cuDNN
4. Copy contents of Google Drive to C:\ (i.e. C:\code\tf3, C:\data\tf3)

Command Line:
5. run "cd C:\code\tf3"
6. run "pip install requirements.txt" - there may be an error about MONAI and Torch - you can ignore this.
7. Follow the instructions here (https://pytorch.org/get-started/locally/) to install the correct version of Pytorch for your machine (in case it doesn't match mine). 
8. run "python tf3_train_X.py" (Where X is replaced with the model name).

Once the script completes, there should be a small number of .pkl files in C:\data\tf3_X_output (where X is replaced with the model name). Send these to me via whatever method (Megaupload, Google Drive etc.).
