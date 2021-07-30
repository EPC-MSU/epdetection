# epdetection

Модуль для распознавания компонентов на платах.

Краткая инструкция по установке и запуску модуля detection. Описание содержания модуля смотрите в `doc/readme.md`

### Установить этот проект:

* Перейти в виртуальное окружение в которое нужно установить модуль

* Перейти в папку с модулем. Запустить:

* ```python setup.py install```

### Добавить модуль в requirements.txt вашего проекта
```bash
# Формально говоря, в этом случае скачается репозиторий.
-e git+https://github.com/EPC-MSU/epdetection@1.0.2#egg=epdetection
```

### Запустить этот проект (из корня):

```bash
python -m detection --image tests//elm_test1/image.png --draw-elements --save-json-result
```
Модуль возмет изображение image.png, распознает на нём элементы PCB, выведет их в консоль, а так же создат папку log в которую положит распознанную картинку и файл с элеметами.

### Запустить тесты (из корня):

```bash
python -m unittest discover tests
```
Оценка точности классификатора производится по формуле: найдено элементов/(всего на размеченой плате + не верно найденные)
