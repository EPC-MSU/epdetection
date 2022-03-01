set eyepoint_root=%~dp0
set eyepoint_root=%eyepoint_root:~0,-1%
pushd %eyepoint_root%\..

venv\Scripts\python utilities\nn_train.py
pause