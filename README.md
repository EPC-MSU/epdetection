# epdetection

Модуль для распознавания компонентов на платах. Содержание модуля смотрите в `doc/readme.md`

### Установить этот модуль в venv:

* Версия питона **строго** 3.6.8 (для совместимости с основным проектом)
* Нажать на `requirements_to_venv.bat` (создастся виртуальное окружение)
* Активировать виртуальное окружение **cmd**:`venv\Scripts\activate`
* Установить модуль **cmd**:```python setup.py install```

### Проверить работоспособность (из под venv в корне):

```bash
python -m detection --image tests//elm_test1/image.png --draw-elements --save-json-result
```
Модуль возьмёт изображение image.png, распознает на нём элементы PCB, выведет их в консоль, а так же создаст папку log в которую положит распознанную картинку и файл с элементами.

### Добавить модуль в requirements.txt вашего проекта
```bash
# В этом случае скачается репозиторий.
-e git+https://github.com/EPC-MSU/epdetection@main#egg=epdetection
# В этом случае установится в site-packages
git+https://github.com/EPC-MSU/epdetection@main#egg=epdetection

# Если нужна конкретная версия epdetection, замените main на V.V.V
```

### Запустить тесты (из под venv в корне):

```bash
python -m unittest discover tests
```
Оценка точности классификатора производится по формуле: найдено элементов/(всего на размеченной плате + не верно найденные)
