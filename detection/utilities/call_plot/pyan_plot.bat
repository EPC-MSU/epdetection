@ECHO OFF
set FILE_TO_PLOT=nn_data_checker.py


echo Plot for file: %FILE_TO_PLOT%
tar -xzf pyan.tar.gz

python .//pyan//pyan.py %FILE_TO_PLOT% --uses --no-defines --colored --grouped --annotated --dot >delete_me.dot
dot -Tsvg delete_me.dot > %FILE_TO_PLOT%.svg

echo Done. Plot saved to: %FILE_TO_PLOT%

rmdir /s /q pyan
del /f delete_me.dot

pause

