# epdetection

Модуль для распознавания компонентов на платах.

Краткая инструкция по установке и запуску модуля `detection`. Описание содержания модуля смотрите в [doc/readme.md](./doc/readme.md).

Для работы модуля **требуется Python 3.6.8**.

### Запуск примера

* Установите зависимости:
  
  ```commandline
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  ```

* Запустите пример:

  ```commandline
  python -m detection --image tests/elm_test1/image.png --draw-elements --save-json-result
  ```
  
  Модуль возьмет изображение `image.png`, распознает на нем элементы PCB, выведет их в консоль, а так же создаст папку `log`, в которую положит распознанную картинку и файл с элементами.

### Запуск тестов

```commandline
python -m unittest discover tests
```

Оценка точности классификатора производится по формуле: (найдено элементов) / (всего на размеченной плате + не верно найденные).

### Генерация документации

Чтобы сгенерировать документацию, выполните действия:

```commandline
python -m pip install pdoc3
python -m pdoc --html detection
```
