# python-package-template

Простой пример-шаблон проекта python-пакета с проверкой стилей, тестами и разными стандартными файлами\папками  
Все новые python-пакеты на github.com/EPC-MSU нужно создавать из этого шаблона

Запустить этот проект (из корня):
```bash
python -m hello_world
```
Запустить тесты (из корня):
```bash
python -m unittest discover tests
```
Установить этот проект (из корня):
```bash
python setup.py install
```
После установки им можно пользоваться:
```python
import hello_world
hello_world.say_hello()
```

Пишите unit-тесты к своим пакетам в tests/

Не забудьте актуализировать информацию о пакете в setup.py: имя проекта, версия, зависимости и пр.

Создан в рамках https://ximc.ru/issues/44427
