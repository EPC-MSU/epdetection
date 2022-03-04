@ECHO OFF
echo ---------------------------------------------------------------
echo Current directory: %cd%
echo ---------------------------------------------------------------
python --version

:choice
set /P c=Is python version above 3.6.8? (y/n)?  
if /I "%c%" EQU "Y" goto :install
if /I "%c%" EQU "N" goto :cancel_install
if /I "%c%" EQU "y" goto :install
if /I "%c%" EQU "n" goto :cancel_install

goto :choice
:install
echo ---------------------------------------------------------------
echo Starting...
echo Install virtual environment...
echo ---------------------------------------------------------------

python -m pip install --user --no-warn-script-location --disable-pip-version-check --no-index -f ./deps virtualenv
python -m venv venv
echo ---------------------------------------------------------------
echo Changing pip version to specified...
echo ---------------------------------------------------------------
venv\Scripts\python -m pip install --no-warn-script-location --disable-pip-version-check --no-index -f ./deps/pip pip==20.1.1
echo ---------------------------------------------------------------
echo Installing packages into venv...
echo ---------------------------------------------------------------
venv\Scripts\python -m pip install -r requirements.txt
echo ---------------------------------------------------------------
echo Virtual environment info:
echo ---------------------------------------------------------------
venv\Scripts\python --version
venv\Scripts\python -m pip --version
venv\Scripts\python -m pip list
:cancel_install
echo ---------------------------------------------------------------
pause